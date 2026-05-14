"""Logging and metrics plotting."""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

def append_episode_log(path: Path, row: dict) -> None:
    """Append episode metrics row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=['episode','reward','steps','success','epsilon','final_health','degraded_events','critical_events','vns','path_efficiency_ratio','smoothness_ratio','obstacle_avoidance_ratio','failure_type','final_row','final_col']
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)

def read_log(path: Path) -> list[dict]:
    """Read metrics CSV rows."""
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def _roll(values: list[int], window: int = 100) -> list[float]:
    """Rolling average helper."""
    out=[]
    for i in range(len(values)):
        seg = values[max(0,i-window+1):i+1]
        out.append(sum(seg)/len(seg))
    return out

def plot_metrics(log_path: Path, plot_dir: Path) -> list[Path]:
    """Plot reward, success, steps, epsilon, and VNS curves."""
    rows = read_log(log_path)
    if not rows:
        return []
    plot_dir.mkdir(parents=True, exist_ok=True)
    ep=[int(r['episode']) for r in rows]; rw=[float(r['reward']) for r in rows]; s=[1 if str(r['success']).lower()=='true' else 0 for r in rows]; st=[int(r['steps']) for r in rows]; eps=[float(r['epsilon']) for r in rows]; v=[float(r.get('vns',0)) for r in rows]
    ts=datetime.now().strftime('%Y%m%d_%H%M%S'); out=[]
    for y,title,name in [(rw,'Reward Over Time','reward'),(_roll(s),'Success Rate Rolling 100','success'),(st,'Steps Per Episode','steps'),(eps,'Epsilon Decay','epsilon'),(v,'VNS Over Time','vns')]:
        plt.figure(figsize=(8,4)); plt.plot(ep,y); plt.title(title); plt.xlabel('Episode'); plt.tight_layout(); p=plot_dir/f'{name}_{ts}.png'; plt.savefig(p,dpi=140); plt.close(); out.append(p)
    return out
