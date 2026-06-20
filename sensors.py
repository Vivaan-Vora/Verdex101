"""Routix 2D sensor simulation with eight ray directions."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from environment import WarehouseEnvironment
DIRS=[(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
def _cast(env: WarehouseEnvironment,pos: tuple[int,int],d: tuple[int,int],max_r: int)->float:
    """Cast one ray."""
    for s in range(1,max_r+1):
        p=(pos[0]+d[0]*s,pos[1]+d[1]*s)
        if not env.in_bounds(p) or env.is_obstacle(p): return s/max_r
    return 1.0
def get_sensor_vector(env: WarehouseEnvironment,pos: tuple[int,int],noise_std: float=0.01,rng: np.random.Generator|None=None)->np.ndarray:
    """Return 10-value sensor vector."""
    rng=rng or np.random.default_rng(); m=max(env.grid.shape); rays=np.array([_cast(env,pos,d,m) for d in DIRS],dtype=np.float32); rays=np.clip(rays+rng.normal(0,noise_std,size=8),0,1); dy,dx=env.goal[0]-pos[0],env.goal[1]-pos[1]; ang=(math.atan2(dy,dx)+math.pi)/(2*math.pi); dist=min(math.sqrt(dx*dx+dy*dy)/(math.sqrt(2)*m),1.0); return np.concatenate([rays,np.array([ang,dist],dtype=np.float32)])
def visualize_sensor_rays(env: WarehouseEnvironment,pos: tuple[int,int],path: Path|None=None)->None:
    """Overlay rays."""
    plt.figure(figsize=(6,6)); plt.imshow(env.grid,cmap='gray_r'); plt.scatter([pos[1]],[pos[0]],c='b'); m=max(env.grid.shape)
    for d in DIRS:
        l=_cast(env,pos,d,m)*m; er,ec=pos[0]+d[0]*l,pos[1]+d[1]*l; plt.plot([pos[1],ec],[pos[0],er],c='orange')
    if path: path.parent.mkdir(parents=True,exist_ok=True); plt.savefig(path,dpi=140,bbox_inches='tight')
    else: plt.show()
    plt.close()
