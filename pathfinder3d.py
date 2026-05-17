"""3D pathfinding algorithms for voxel environments."""
from __future__ import annotations

import heapq
from collections import deque

from environment3d import WarehouseEnvironment3D

Coord3D = tuple[int, int, int]


def manhattan3d(a: Coord3D, b: Coord3D) -> int:
    """Compute 3D Manhattan distance."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])


def _neighbors(env: WarehouseEnvironment3D, p: Coord3D) -> list[Coord3D]:
    """Get valid 6-connected neighbors."""
    x, y, z = p
    out: list[Coord3D] = []
    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        n = (x + dx, y + dy, z + dz)
        if env.in_bounds(n) and not env.is_obstacle(n):
            out.append(n)
    return out


def _reconstruct(came: dict[Coord3D, Coord3D], cur: Coord3D) -> list[Coord3D]:
    """Reconstruct path from predecessor map."""
    path = [cur]
    while cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path


def bfs3d(env: WarehouseEnvironment3D) -> tuple[list[Coord3D], int, int]:
    """Run BFS in 3D and return path, length, explored count."""
    q = deque([env.start])
    seen = {env.start}
    came: dict[Coord3D, Coord3D] = {}
    explored = 0
    while q:
        u = q.popleft()
        explored += 1
        if u == env.goal:
            p = _reconstruct(came, u)
            return p, len(p), explored
        for v in _neighbors(env, u):
            if v in seen:
                continue
            seen.add(v)
            came[v] = u
            q.append(v)
    return [], 0, explored


def dijkstra3d(env: WarehouseEnvironment3D) -> tuple[list[Coord3D], int, int]:
    """Run Dijkstra in 3D and return path, length, explored count."""
    heap: list[tuple[int, Coord3D]] = [(0, env.start)]
    dist: dict[Coord3D, int] = {env.start: 0}
    came: dict[Coord3D, Coord3D] = {}
    explored = 0
    while heap:
        cost, u = heapq.heappop(heap)
        explored += 1
        if u == env.goal:
            p = _reconstruct(came, u)
            return p, len(p), explored
        for v in _neighbors(env, u):
            ncost = cost + 1
            if v not in dist or ncost < dist[v]:
                dist[v] = ncost
                came[v] = u
                heapq.heappush(heap, (ncost, v))
    return [], 0, explored


def astar3d(env: WarehouseEnvironment3D) -> tuple[list[Coord3D], int, int]:
    """Run A* in 3D and return path, length, explored count."""
    heap: list[tuple[int, int, Coord3D]] = [(manhattan3d(env.start, env.goal), 0, env.start)]
    gscore: dict[Coord3D, int] = {env.start: 0}
    came: dict[Coord3D, Coord3D] = {}
    explored = 0
    while heap:
        _, g, u = heapq.heappop(heap)
        explored += 1
        if u == env.goal:
            p = _reconstruct(came, u)
            return p, len(p), explored
        for v in _neighbors(env, u):
            ng = g + 1
            if v not in gscore or ng < gscore[v]:
                gscore[v] = ng
                came[v] = u
                heapq.heappush(heap, (ng + manhattan3d(v, env.goal), ng, v))
    return [], 0, explored


def compare_algorithms_3d(env: WarehouseEnvironment3D) -> dict[str, tuple[int, int]]:
    """Print side-by-side results for BFS, Dijkstra, and A* in 3D."""
    methods = {'A*_3D': astar3d, 'Dijkstra_3D': dijkstra3d, 'BFS_3D': bfs3d}
    out: dict[str, tuple[int, int]] = {}
    print('Method        | Path Length | Explored Nodes')
    print('--------------+-------------+---------------')
    for name, fn in methods.items():
        _, length, explored = fn(env)
        out[name] = (length, explored)
        print(f'{name:<13} | {length:<11} | {explored:<13}')
    return out
