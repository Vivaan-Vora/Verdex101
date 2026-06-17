"""Benchmark runner."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from environment import generate_environment
from pathfinder import astar, bfs, dijkstra
from analyzer import compute_edi
from scorer import compute_vns, print_vns_leaderboard

def run_benchmark(config: dict, runs: int = 20, seed: int = 42) -> dict:
    """Run benchmark across pathfinding methods."""
    methods={'A*':astar,'BFS':bfs,'Dijkstra':dijkstra}
    stats={m:{'success':0,'steps':0.0,'reward':0.0} for m in methods}
    vns_by_method=defaultdict(list)
    vns_vs_edi=defaultdict(list)
    samples=[]
    for i in range(runs):
        env=generate_environment(config['environment']['grid_size'],config['environment']['difficulty'],config['environment']['moving_obstacles'],seed+i)
        edi=compute_edi(env)['edi']
        for name,fn in methods.items():
            path,length,_=fn(env)
            ok=1 if path else 0
            reward=100-max(0,length-1) if ok else -100
            stats[name]['success']+=ok
            stats[name]['steps']+=length if ok else config['training']['max_steps']
            stats[name]['reward']+=reward
            if path:
                v=compute_vns(length,max(1,length),path,env.grid,config['vns'])['vns']
                vns_by_method[name].append(v)
                vns_vs_edi[name].append((edi,v))
                samples.append({'method':name,'edi':edi,'success':ok,'steps':length,'reward':reward,'vns':v})
    print('Method     | Success Rate | Avg Steps | Avg Reward')
    print('-----------+--------------+-----------+-----------')
    for name in methods:
        print(f"{name:<10} | {stats[name]['success']/runs:<12.2%} | {stats[name]['steps']/runs:<9.2f} | {stats[name]['reward']/runs:<9.2f}")
    print_vns_leaderboard(vns_by_method)
    _print_edi_quartiles(samples)
    _plot_bar(stats,runs,Path(config['paths']['plot_dir']))
    _plot_scatter(vns_vs_edi,Path(config['paths']['plot_dir']))
    return stats

def _plot_bar(stats: dict, runs: int, plot_dir: Path) -> None:
    """Plot benchmark reward bar chart."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels=list(stats.keys()); vals=[stats[x]['reward']/runs for x in labels]
    plt.figure(figsize=(7,4)); plt.bar(labels, vals); plt.title('Benchmark Average Reward'); plt.tight_layout(); plt.savefig(plot_dir/f"benchmark_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",dpi=140); plt.close()

def _plot_scatter(points: dict, plot_dir: Path) -> None:
    """Plot VNS vs EDI scatter."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5))
    for name, arr in points.items():
        if arr:
            plt.scatter([x for x,_ in arr],[y for _,y in arr],label=name,alpha=0.7)
    plt.xlabel('EDI'); plt.ylabel('VNS'); plt.title('VNS vs EDI'); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir/f"vns_vs_edi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",dpi=140); plt.close()

def _print_edi_quartiles(samples: list[dict]) -> None:
    """Print benchmark summary grouped by EDI quartiles."""
    if not samples:
        return
    sorted_edi=sorted(x['edi'] for x in samples)
    q1=sorted_edi[max(0,len(sorted_edi)//4 - 1)]
    q2=sorted_edi[max(0,len(sorted_edi)//2 - 1)]
    q3=sorted_edi[max(0,(3*len(sorted_edi))//4 - 1)]
    def bucket(edi: float) -> str:
        if edi <= q1:
            return 'easy'
        if edi <= q2:
            return 'medium'
        if edi <= q3:
            return 'hard'
        return 'very_hard'
    grouped=defaultdict(lambda: {'count':0,'success':0,'steps':0.0,'reward':0.0})
    for row in samples:
        key=(row['method'], bucket(row['edi']))
        grouped[key]['count'] += 1
        grouped[key]['success'] += row['success']
        grouped[key]['steps'] += row['steps']
        grouped[key]['reward'] += row['reward']
    print('\nPerformance By EDI Quartile')
    print('Method     | Quartile  | Success Rate | Avg Steps | Avg Reward')
    print('-----------+-----------+--------------+-----------+-----------')
    for (method, quartile), values in sorted(grouped.items()):
        count=max(1,values['count'])
        print(f"{method:<10} | {quartile:<9} | {values['success']/count:<12.2%} | {values['steps']/count:<9.2f} | {values['reward']/count:<9.2f}")
