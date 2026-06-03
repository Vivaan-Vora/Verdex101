"""Navra 2D environment generation module."""
from __future__ import annotations
import json, random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

@dataclass
class WarehouseEnvironment:
    """Grid environment state."""
    grid: np.ndarray
    start: tuple[int,int]
    goal: tuple[int,int]
    moving_obstacles: set[tuple[int,int]]
    def in_bounds(self, p: tuple[int,int]) -> bool:
        """Check bounds."""
        return 0 <= p[0] < self.grid.shape[0] and 0 <= p[1] < self.grid.shape[1]
    def is_obstacle(self, p: tuple[int,int]) -> bool:
        """Check blocked cell."""
        return self.grid[p]==1 or p in self.moving_obstacles
    def update_moving_obstacles(self, rng: random.Random) -> None:
        """Random walk dynamic obstacles."""
        out=set()
        for r,c in self.moving_obstacles:
            candidates=[]
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                n=(r+dr,c+dc)
                if self.in_bounds(n) and self.grid[n]==0 and n not in (self.start,self.goal): candidates.append(n)
            out.add(rng.choice(candidates) if candidates else (r,c))
        self.moving_obstacles=out

def bfs_path_exists(grid: np.ndarray, start: tuple[int,int], goal: tuple[int,int]) -> bool:
    """Connectivity check."""
    q=deque([start]); seen={start}
    while q:
        r,c=q.popleft()
        if (r,c)==goal: return True
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            n=(r+dr,c+dc)
            if 0<=n[0]<grid.shape[0] and 0<=n[1]<grid.shape[1] and n not in seen and grid[n]==0:
                seen.add(n); q.append(n)
    return False

def generate_environment(grid_size: int=30, difficulty: str='medium', moving_obstacle_count: int=3, seed: int|None=None) -> WarehouseEnvironment:
    """Generate valid warehouse map."""
    rng=random.Random(seed); dens={'easy':0.2,'medium':0.35,'hard':0.5}; d=dens[difficulty]
    start=(1,1); goal=(grid_size-2,grid_size-2)
    while True:
        g=np.zeros((grid_size,grid_size),dtype=np.int32); g[0,:]=1; g[-1,:]=1; g[:,0]=1; g[:,-1]=1
        target=int(grid_size*grid_size*d); placed=0
        while placed<target:
            c=rng.randint(1,grid_size-2); sr=rng.randint(1,grid_size-4); span=rng.randint(grid_size//3,grid_size-2)
            for r in range(sr,min(grid_size-1,sr+span)):
                if g[r,c]==0:
                    g[r,c]=1; placed+=1
                    if placed>=target: break
        g[start]=0; g[goal]=0
        if bfs_path_exists(g,start,goal): break
    free=[(r,c) for r in range(1,grid_size-1) for c in range(1,grid_size-1) if g[r,c]==0 and (r,c) not in (start,goal)]
    rng.shuffle(free)
    return WarehouseEnvironment(g,start,goal,set(free[:moving_obstacle_count]))

def save_environment(env: WarehouseEnvironment, path: Path) -> None:
    """Save environment json."""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'grid':env.grid.tolist(),'start':list(env.start),'goal':list(env.goal),'moving_obstacles':[list(x) for x in env.moving_obstacles]},indent=2),encoding='utf-8')

def load_environment(path: Path) -> WarehouseEnvironment:
    """Load environment json."""
    d=json.loads(path.read_text(encoding='utf-8'))
    return WarehouseEnvironment(np.array(d['grid'],dtype=np.int32),tuple(d['start']),tuple(d['goal']),{tuple(x) for x in d['moving_obstacles']})

def plot_environment(env: WarehouseEnvironment, title: str='Environment', path: Path|None=None) -> None:
    """Plot environment map."""
    # Cell encoding: 0 free, 1 static obstacle, 2 moving obstacle, 3 start, 4 goal
    view=np.zeros_like(env.grid, dtype=np.int32)
    view[env.grid == 1] = 1
    for r,c in env.moving_obstacles:
        view[r,c] = 2
    view[env.start] = 3
    view[env.goal] = 4

    cmap=ListedColormap([
        '#f1f5f9',  # free
        '#0f172a',  # wall
        '#ff4d6d',  # moving obstacle
        '#7c3aed',  # start
        '#22c55e',  # goal
    ])
    legend=[
        Patch(facecolor='#f1f5f9', edgecolor='black', label='Open Cell'),
        Patch(facecolor='#0f172a', edgecolor='black', label='Shelf / Wall'),
        Patch(facecolor='#ff4d6d', edgecolor='black', label='Moving Obstacle'),
        Patch(facecolor='#7c3aed', edgecolor='black', label='Start'),
        Patch(facecolor='#22c55e', edgecolor='black', label='Goal'),
    ]

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(view, cmap=cmap, vmin=0, vmax=4, origin='upper')
    ax.set_facecolor('#e2e8f0')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#0f172a')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    if path:
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=180,bbox_inches='tight')
    else:
        plt.show()
    plt.close()
