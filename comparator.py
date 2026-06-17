"""Policy comparator."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from environment import generate_environment
from pathfinder import astar
from q_agent import QLearningAgent, ACTIONS
from dqn_agent import DQNAgent
from sensors import get_sensor_vector
from scorer import compute_vns

def _rollout(agent: QLearningAgent, env, max_steps: int) -> dict:
    """Roll out one greedy episode."""
    pos=env.start; path=[pos]; reward=0.0; collisions=0
    for _ in range(max_steps):
        s=get_sensor_vector(env,pos)
        a=int(max(range(4), key=lambda i: agent._get_q(agent._state_key(s))[i]))
        dr,dc=ACTIONS[a]; n=(pos[0]+dr,pos[1]+dc)
        if not env.in_bounds(n) or env.is_obstacle(n):
            collisions+=1; reward-=10; n=pos
        elif n==env.goal:
            reward+=100; path.append(n); return {'path':path,'reward':reward,'collisions':collisions,'success':True}
        else:
            reward-=1
        pos=n; path.append(pos)
    return {'path':path,'reward':reward,'collisions':collisions,'success':False}

def _rollout_dqn(agent: DQNAgent, env, max_steps: int) -> dict:
    """Roll out one greedy episode for a DQN agent."""
    pos=env.start; path=[pos]; reward=0.0; collisions=0
    for _ in range(max_steps):
        s=get_sensor_vector(env,pos)
        with torch.no_grad():
            a=int(torch.argmax(agent.model(torch.tensor(s,dtype=torch.float32).unsqueeze(0)),dim=1).item())
        dr,dc=ACTIONS[a]; n=(pos[0]+dr,pos[1]+dc)
        if not env.in_bounds(n) or env.is_obstacle(n):
            collisions+=1; reward-=10; n=pos
        elif n==env.goal:
            reward+=100; path.append(n); return {'path':path,'reward':reward,'collisions':collisions,'success':True}
        else:
            reward-=1
        pos=n; path.append(pos)
    return {'path':path,'reward':reward,'collisions':collisions,'success':False}

def compare_agents(config: dict, q_agent: QLearningAgent, dqn_agent: DQNAgent, environments: int=10, seed: int=42) -> None:
    """Create side-by-side frames and summary plot."""
    out=Path('logs/plots')/f"comparator_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(environments):
        env=generate_environment(config['environment']['grid_size'],config['environment']['difficulty'],config['environment']['moving_obstacles'],seed+i)
        _,opt,_=astar(env); opt=max(1,opt)
        left=_rollout(q_agent,env,config['training']['max_steps'])
        right=_rollout_dqn(dqn_agent,env,config['training']['max_steps'])
        left_vns=compute_vns(len(left['path']),opt,left['path'],env.grid,config['vns'])['vns']
        right_vns=compute_vns(len(right['path']),opt,right['path'],env.grid,config['vns'])['vns']
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].imshow(env.grid,cmap='gray_r'); ax[0].plot([p[1] for p in left['path']],[p[0] for p in left['path']],c='b'); ax[0].set_title('Q-Learning')
        ax[1].imshow(env.grid,cmap='gray_r'); ax[1].plot([p[1] for p in right['path']],[p[0] for p in right['path']],c='purple'); ax[1].set_title('DQN')
        fig.suptitle(
            f"Episode {i+1} | Q reward {left['reward']:.1f} steps {len(left['path'])} collisions {left['collisions']} VNS {left_vns:.1f} "
            f"| DQN reward {right['reward']:.1f} steps {len(right['path'])} collisions {right['collisions']} VNS {right_vns:.1f}"
        )
        plt.tight_layout(); plt.savefig(out/f'frame_{i:03d}.png',dpi=140); plt.close()
        rows.append({
            'q_reward':left['reward'],'q_success':1 if left['success'] else 0,'q_vns':left_vns,'q_steps':len(left['path']),
            'dqn_reward':right['reward'],'dqn_success':1 if right['success'] else 0,'dqn_vns':right_vns,'dqn_steps':len(right['path'])
        })
    if rows:
        q_vals=[sum(r[k] for r in rows)/len(rows) for k in ['q_reward','q_success','q_vns','q_steps']]
        d_vals=[sum(r[k] for r in rows)/len(rows) for k in ['dqn_reward','dqn_success','dqn_vns','dqn_steps']]
        labels=['Avg Reward','Success Rate','Avg VNS','Avg Steps']
        x=range(len(labels))
        plt.figure(figsize=(10,4))
        plt.bar([i-0.2 for i in x],[q_vals[0],q_vals[1]*100,q_vals[2],q_vals[3]],width=0.4,label='Q-Learning')
        plt.bar([i+0.2 for i in x],[d_vals[0],d_vals[1]*100,d_vals[2],d_vals[3]],width=0.4,label='DQN')
        plt.xticks(list(x),labels); plt.title('Comparator Summary'); plt.legend(); plt.tight_layout(); plt.savefig(out/'summary.png',dpi=140); plt.close()
