"""Navigation scoring utilities for Routix."""
from __future__ import annotations
import numpy as np

# VNS combines path efficiency, smoothness, and obstacle avoidance.

def direction_changes(path: list[tuple[int,int]]) -> int:
    """Count direction changes."""
    if len(path) < 3:
        return 0
    changes = 0
    prev = (path[1][0]-path[0][0], path[1][1]-path[0][1])
    for i in range(2, len(path)):
        cur = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        if cur != prev:
            changes += 1
        prev = cur
    return changes

def near_obstacle_count(path: list[tuple[int,int]], grid: np.ndarray, radius: int = 2) -> int:
    """Count steps near any obstacle."""
    out = 0
    for r, c in path:
        patch = grid[max(0,r-radius):min(grid.shape[0],r+radius+1), max(0,c-radius):min(grid.shape[1],c+radius+1)]
        if np.any(patch == 1):
            out += 1
    return out

def compute_vns(actual_steps: int, optimal_steps: int, path: list[tuple[int,int]], grid: np.ndarray, weights: dict) -> dict:
    """Compute VNS components and total score."""
    actual_steps = max(1, actual_steps)
    optimal_steps = max(1, optimal_steps)
    eff_ratio = actual_steps / optimal_steps
    smooth_ratio = direction_changes(path) / actual_steps
    avoid_ratio = near_obstacle_count(path, grid) / actual_steps
    eff_score = max(0.0, min(100.0, 100.0 / eff_ratio))
    smooth_score = max(0.0, min(100.0, 100.0 * (1.0 - smooth_ratio)))
    avoid_score = max(0.0, min(100.0, 100.0 * (1.0 - avoid_ratio)))
    vns = (weights['path_efficiency_weight']*eff_score + weights['smoothness_weight']*smooth_score + weights['obstacle_avoidance_weight']*avoid_score)
    return {'path_efficiency_ratio':eff_ratio,'smoothness_ratio':smooth_ratio,'obstacle_avoidance_ratio':avoid_ratio,'vns':float(vns)}

def print_vns_leaderboard(method_stats: dict[str,list[float]]) -> None:
    """Print aligned leaderboard."""
    ranked = sorted(((k, sum(v)/max(1,len(v))) for k,v in method_stats.items()), key=lambda x: x[1], reverse=True)
    print('Rank | Method           | Average VNS')
    print('-----+------------------+------------')
    for i, (name, avg) in enumerate(ranked, start=1):
        print(f'{i:<4} | {name:<16} | {avg:>10.2f}')
