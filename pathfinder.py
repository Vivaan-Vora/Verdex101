"""Routix 2D pathfinding algorithms (BFS, Dijkstra, A*)."""
from __future__ import annotations
import heapq
from collections import deque
import matplotlib.pyplot as plt
from environment import WarehouseEnvironment
def manhattan(a: tuple[int,int],b: tuple[int,int])->int:
    """Manhattan distance."""
    return abs(a[0]-b[0])+abs(a[1]-b[1])
def _n(env: WarehouseEnvironment,p: tuple[int,int]):
    """Neighbors."""
    r,c=p
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        n=(r+dr,c+dc)
        if env.in_bounds(n) and not env.is_obstacle(n): yield n
def _rec(came,cur):
    """Reconstruct path."""
    out=[cur]
    while cur in came: cur=came[cur]; out.append(cur)
    return list(reversed(out))
def bfs(env: WarehouseEnvironment):
    """Run BFS."""
    q=deque([env.start]); seen={env.start}; came={}; exp=0
    while q:
        u=q.popleft(); exp+=1
        if u==env.goal: p=_rec(came,u); return p,len(p),exp
        for v in _n(env,u):
            if v in seen: continue
            seen.add(v); came[v]=u; q.append(v)
    return [],0,exp
def dijkstra(env: WarehouseEnvironment):
    """Run Dijkstra."""
    h=[(0,env.start)]; dist={env.start:0}; came={}; exp=0
    while h:
        g,u=heapq.heappop(h); exp+=1
        if u==env.goal: p=_rec(came,u); return p,len(p),exp
        for v in _n(env,u):
            ng=g+1
            if v not in dist or ng<dist[v]: dist[v]=ng; came[v]=u; heapq.heappush(h,(ng,v))
    return [],0,exp
def astar(env: WarehouseEnvironment):
    """Run A*."""
    h=[(manhattan(env.start,env.goal),0,env.start)]; gs={env.start:0}; came={}; exp=0
    while h:
        _,g,u=heapq.heappop(h); exp+=1
        if u==env.goal: p=_rec(came,u); return p,len(p),exp
        for v in _n(env,u):
            ng=g+1
            if v not in gs or ng<gs[v]: gs[v]=ng; came[v]=u; heapq.heappush(h,(ng+manhattan(v,env.goal),ng,v))
    return [],0,exp
def compare_algorithms(env: WarehouseEnvironment):
    """Print comparison."""
    m={'A*':astar,'Dijkstra':dijkstra,'BFS':bfs}; out={}; print('Method     | Path Length | Explored Nodes'); print('-----------+-------------+---------------')
    for n,f in m.items():
        _,l,e=f(env); out[n]=(l,e); print(f'{n:<10} | {l:<11} | {e:<13}')
    return out
def plot_path(env: WarehouseEnvironment,path: list[tuple[int,int]],title: str='Path',save_path: str|None=None)->None:
    """Plot path."""
    plt.figure(figsize=(6,6)); plt.imshow(env.grid,cmap='gray_r');
    if path: plt.plot([p[1] for p in path],[p[0] for p in path],c='r')
    plt.scatter([env.start[1]],[env.start[0]],c='b'); plt.scatter([env.goal[1]],[env.goal[0]],c='g'); plt.title(title)
    if save_path: plt.savefig(save_path,dpi=140,bbox_inches='tight')
    else: plt.show()
    plt.close()
