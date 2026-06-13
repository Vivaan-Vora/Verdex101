"""Deep Q-Network agent."""
from __future__ import annotations
import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment import generate_environment
from sensors import get_sensor_vector
from logger import append_episode_log
from diagnostics import HealthState, update_health
from scorer import compute_vns
from pathfinder import astar, manhattan
from failure_logger import classify_failure
ACTIONS=[(-1,0),(1,0),(0,-1),(0,1)]

class DQN(nn.Module):
    """Feed-forward network for deep Q-value approximation."""
    def __init__(self):
        """Initialize MLP."""
        super().__init__(); self.net=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,x):
        """Forward pass."""
        return self.net(x)

class DQNAgent:
    """DQN with replay and target model."""
    def __init__(self,cfg:dict):
        """Initialize model, optimizer, replay buffer."""
        self.cfg=cfg; self.model=DQN(); self.target=DQN(); self.target.load_state_dict(self.model.state_dict()); self.opt=optim.Adam(self.model.parameters(),lr=cfg['learning_rate']); self.loss=nn.MSELoss(); self.buf=deque(maxlen=cfg['buffer_size']); self.steps=0
    def choose_action(self,obs,eps,rng):
        """Epsilon-greedy action."""
        if rng.random()<eps: return rng.randrange(4)
        with torch.no_grad(): return int(torch.argmax(self.model(torch.tensor(obs,dtype=torch.float32).unsqueeze(0)),dim=1).item())
    def remember(self,t):
        """Store transition."""
        self.buf.append(t)
    def train_step(self):
        """Sample minibatch and update weights."""
        if len(self.buf)<self.cfg['batch_size']: return
        b=random.sample(list(self.buf),self.cfg['batch_size'])
        s=torch.tensor(np.array([x[0] for x in b]),dtype=torch.float32); a=torch.tensor([x[1] for x in b],dtype=torch.int64); r=torch.tensor([x[2] for x in b],dtype=torch.float32); ns=torch.tensor(np.array([x[3] for x in b]),dtype=torch.float32); d=torch.tensor([x[4] for x in b],dtype=torch.float32)
        q=self.model(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad(): t=r+(1-d)*self.cfg['gamma']*self.target(ns).max(1).values
        loss=self.loss(q,t); self.opt.zero_grad(); loss.backward(); self.opt.step(); self.steps+=1
        if self.steps%self.cfg['target_update_interval']==0: self.target.load_state_dict(self.model.state_dict())
    def save(self,path:Path):
        """Save model weights."""
        path.parent.mkdir(parents=True,exist_ok=True); torch.save(self.model.state_dict(),path)

def _step(env,pos,a):
    """Take one environment step."""
    dr,dc=ACTIONS[a]; n=(pos[0]+dr,pos[1]+dc)
    if not env.in_bounds(n) or env.is_obstacle(n): return pos,-10.0,False,True
    if n==env.goal: return n,100.0,True,False
    return n,-1.0,False,False

def train_dqn_agent(config: dict, seed:int=42, episodes_override:int|None=None) -> DQNAgent:
    """Train DQN and append logs per episode."""
    rng=random.Random(seed); np_rng=np.random.default_rng(seed)
    ep=episodes_override or int(config['training']['episodes']); ms=int(config['training']['max_steps'])
    q=config['q_learning']; d=config['diagnostics']; f=config['failure_logger']; w=config['vns']; dqn=config['dqn']
    agent=DQNAgent(dqn); eps=q['epsilon_start']; log=Path(config['paths']['log_csv'])
    for e in range(1,ep+1):
        env=generate_environment(config['environment']['grid_size'],config['environment']['difficulty'],config['environment']['moving_obstacles'],seed+e)
        _,opt,_=astar(env); opt=max(1,opt)
        health=HealthState(); pos=env.start; path=[pos]; reward=0.0; done=False
        revisits={}; max_rev=0; same=0; max_same=0; cwin=[]; cburst=0
        for _ in range(ms):
            env.update_moving_obstacles(rng)
            s=get_sensor_vector(env,pos,rng=np_rng); a=agent.choose_action(s,eps,rng)
            n,r,done,col=_step(env,pos,a); sn=get_sensor_vector(env,n,rng=np_rng)
            agent.remember((s,a,r,sn,float(done))); agent.train_step()
            same = same+1 if n==pos else 0; max_same=max(max_same,same)
            revisits[n]=revisits.get(n,0)+1; max_rev=max(max_rev,revisits[n])
            cwin.append(1 if col else 0); cwin=cwin[-f['collision_storm_window']:]
            if sum(cwin)>f['collision_storm_hits']: cburst+=1
            health=update_health(health,1 if col else 0,1 if same>5 else 0,1 if len(path)>3*opt else 0,d['degradation_per_collision'],d['degradation_per_stuck_event'],d['degraded_threshold'],d['critical_threshold'],d['maintenance_cycle_steps'])
            pos=n; path.append(pos); reward+=r
            if done: break
        v=compute_vns(len(path),opt,path,env.grid,w)
        far=manhattan(pos,env.goal)>config['environment']['grid_size']//2
        ft=classify_failure(done,len(path),ms,max_rev,max_same,cburst,far,f['timeout_threshold_steps'],f['loop_threshold_revisits'])
        append_episode_log(log,{'episode':e,'reward':reward,'steps':len(path),'success':bool(done),'epsilon':eps,'final_health':health.health,'degraded_events':health.degraded_events,'critical_events':health.critical_events,'vns':v['vns'],'path_efficiency_ratio':v['path_efficiency_ratio'],'smoothness_ratio':v['smoothness_ratio'],'obstacle_avoidance_ratio':v['obstacle_avoidance_ratio'],'failure_type':ft,'final_row':pos[0],'final_col':pos[1]})
        eps=max(q['epsilon_end'],eps*q['epsilon_decay'])
        if e%int(config['training']['progress_interval'])==0: print(f"[DQN] Episode {e}/{ep} reward={reward:.2f} success={done} epsilon={eps:.3f}")
    agent.save(Path(config['paths']['dqn_model']))
    return agent
