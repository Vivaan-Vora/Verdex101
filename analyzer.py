"""Environment difficulty analysis for Navra."""
from __future__ import annotations
from environment import WarehouseEnvironment
from pathfinder import astar, manhattan

def _walkable_neighbors(grid, r: int, c: int) -> int:
    """Count free 4-neighbors."""
    n = 0
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        rr,cc = r+dr,c+dc
        if 0<=rr<grid.shape[0] and 0<=cc<grid.shape[1] and grid[rr,cc]==0:
            n += 1
    return n

def compute_edi(env: WarehouseEnvironment) -> dict:
    """Compute EDI metrics."""
    g = env.grid
    free = 0
    bottlenecks = 0
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r,c] == 0:
                free += 1
                if _walkable_neighbors(g,r,c) <= 2:
                    bottlenecks += 1
    open_space_ratio = free / (g.shape[0]*g.shape[1])
    goal_distance = manhattan(env.start, env.goal)
    _, path_len, _ = astar(env)
    path_len = path_len or goal_distance*4
    path_complexity = path_len / max(1,goal_distance)
    edi = 100*(0.35*min(1,bottlenecks/max(1,free)) + 0.25*min(1,(path_complexity-1)/4) + 0.2*(1-open_space_ratio) + 0.2*min(1,goal_distance/max(g.shape)))
    return {'bottleneck_score':float(bottlenecks),'open_space_ratio':float(open_space_ratio),'goal_distance':float(goal_distance),'path_complexity':float(path_complexity),'edi':float(max(0,min(100,edi)))}

def print_edi(metrics: dict) -> None:
    """Print EDI metrics."""
    print('Environment Difficulty Analysis')
    for key, value in metrics.items():
        print(f'- {key}: {value:.4f}')
