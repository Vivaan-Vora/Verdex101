"""CLI entry point for Astra-Path."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from environment import generate_environment, plot_environment
from sensors import visualize_sensor_rays
from pathfinder import astar, compare_algorithms, plot_path
from q_agent import train_q_agent, QLearningAgent
from dqn_agent import train_dqn_agent, DQNAgent
from logger import read_log, plot_metrics
from diagnostics import diagnostics_report, plot_health_curve
from scorer import print_vns_leaderboard
from analyzer import compute_edi, print_edi
from failure_logger import plot_failure_breakdown, plot_failure_heatmap
from benchmark import run_benchmark
from comparator import compare_agents
from visualizer import replay_from_log
from scorer import compute_vns
from pathfinder import manhattan
from sensors import get_sensor_vector
import torch

def load_config(path: Path=Path('config.json')) -> dict:
    """Load JSON configuration."""
    return json.loads(path.read_text(encoding='utf-8'))

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p=argparse.ArgumentParser(description='Astra-Path warehouse navigation simulator')
    p.add_argument('--mode',required=True,choices=['train-q','train-dqn','test','astar','benchmark','replay','diagnostics','score','analyze-env','failures','compare'])
    p.add_argument('--difficulty',choices=['easy','medium','hard'],default=None)
    p.add_argument('--episodes',type=int,default=None)
    p.add_argument('--seed',type=int,default=42)
    return p.parse_args()

def main() -> None:
    """Dispatch mode handlers."""
    args=parse_args(); config=load_config()
    if args.difficulty:
        config['environment']['difficulty']=args.difficulty
    if args.mode=='train-q':
        train_q_agent(config,seed=args.seed,episodes_override=args.episodes)
        rows=read_log(Path(config['paths']['log_csv']))
        plot_metrics(Path(config['paths']['log_csv']),Path(config['paths']['plot_dir']))
        if rows:
            plot_health_curve([float(r.get('final_health',100.0)) for r in rows],Path(config['paths']['plot_dir'])/'health_curve.png')
    elif args.mode=='train-dqn':
        train_dqn_agent(config,seed=args.seed,episodes_override=args.episodes)
        plot_metrics(Path(config['paths']['log_csv']),Path(config['paths']['plot_dir']))
    elif args.mode=='test':
        env=generate_environment(config['environment']['grid_size'],config['environment']['difficulty'],config['environment']['moving_obstacles'],args.seed)
        compare_algorithms(env); plot_environment(env)
    elif args.mode=='astar':
        env=generate_environment(difficulty=config['environment']['difficulty'],seed=args.seed)
        path,_,_=astar(env); plot_path(env,path,title='A* Path')
    elif args.mode=='benchmark':
        run_benchmark(config,20,args.seed)
    elif args.mode=='replay':
        points=replay_from_log(read_log(Path(config['paths']['log_csv']))); print(f'Loaded replay trace with {len(points)} points')
    elif args.mode=='diagnostics':
        report=diagnostics_report(read_log(Path(config['paths']['log_csv'])),Path(config['paths']['reports_dir']))
        print(f'Diagnostics report saved to {report}')
    elif args.mode=='score':
        qa=QLearningAgent(4,0.1,0.95,0.0); qa.load(Path(config['paths']['q_table']))
        scores=[]
        for i in range(20):
            env=generate_environment(config['environment']['grid_size'],config['environment']['difficulty'],config['environment']['moving_obstacles'],args.seed+i)
            from pathfinder import astar as _astar
            _,opt,_=_astar(env); opt=max(1,opt)
            pos=env.start; path=[pos]
            for _ in range(config['training']['max_steps']):
                s=get_sensor_vector(env,pos)
                a=int(max(range(4), key=lambda idx: qa._get_q(qa._state_key(s))[idx]))
                dr,dc=[(-1,0),(1,0),(0,-1),(0,1)][a]
                nxt=(pos[0]+dr,pos[1]+dc)
                if not env.in_bounds(nxt) or env.is_obstacle(nxt):
                    nxt=pos
                pos=nxt; path.append(pos)
                if pos==env.goal:
                    break
            metrics=compute_vns(len(path),opt,path,env.grid,config['vns'])
            scores.append(metrics['vns'])
        print_vns_leaderboard({'Q-Learning':scores})
    elif args.mode=='analyze-env':
        env=generate_environment(difficulty=config['environment']['difficulty'],seed=args.seed)
        print_edi(compute_edi(env)); visualize_sensor_rays(env,env.start)
    elif args.mode=='failures':
        rows=read_log(Path(config['paths']['log_csv']))
        out=Path(config['paths']['plot_dir'])
        plot_failure_breakdown(rows,out/'failure_breakdown.png')
        gs=config['environment']['grid_size']
        plot_failure_heatmap(rows,(gs,gs),out/'failure_heatmap.png')
    elif args.mode=='compare':
        qa=QLearningAgent(4,0.1,0.95,0.1); qa.load(Path(config['paths']['q_table']))
        dqn=DQNAgent(config['dqn'])
        dqn_model_path=Path(config['paths']['dqn_model'])
        if dqn_model_path.exists():
            dqn.model.load_state_dict(torch.load(dqn_model_path,map_location='cpu'))
            dqn.target.load_state_dict(dqn.model.state_dict())
        compare_agents(config,qa,dqn,environments=config['comparator']['num_test_environments'],seed=args.seed)

if __name__=='__main__':
    """Program entry."""
    main()
