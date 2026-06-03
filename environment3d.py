"""Navra 3D warehouse environment generation and visualization."""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Coord3D = tuple[int, int, int]


@dataclass
class WarehouseEnvironment3D:
    """Represents a voxel warehouse for 3D navigation."""

    grid: np.ndarray  # shape: (x, y, z)
    start: Coord3D
    goal: Coord3D
    moving_obstacles: set[Coord3D]

    def in_bounds(self, p: Coord3D) -> bool:
        """Return True if coordinate is in range."""
        x, y, z = p
        sx, sy, sz = self.grid.shape
        return 0 <= x < sx and 0 <= y < sy and 0 <= z < sz

    def is_obstacle(self, p: Coord3D) -> bool:
        """Return True when voxel is blocked."""
        return self.grid[p] == 1 or p in self.moving_obstacles

    def update_moving_obstacles(self, rng: random.Random) -> None:
        """Move dynamic obstacles in 6-connected directions."""
        dirs = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        updated: set[Coord3D] = set()
        for p in self.moving_obstacles:
            options: list[Coord3D] = []
            for dx, dy, dz in dirs:
                n = (p[0] + dx, p[1] + dy, p[2] + dz)
                if not self.in_bounds(n):
                    continue
                if n in (self.start, self.goal):
                    continue
                if self.grid[n] == 0 and n not in self.moving_obstacles:
                    options.append(n)
            updated.add(rng.choice(options) if options else p)
        self.moving_obstacles = updated


def _neighbors(p: Coord3D, shape: tuple[int, int, int]) -> list[Coord3D]:
    """Return 6-connected neighbors in bounds."""
    x, y, z = p
    sx, sy, sz = shape
    out: list[Coord3D] = []
    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        n = (x + dx, y + dy, z + dz)
        if 0 <= n[0] < sx and 0 <= n[1] < sy and 0 <= n[2] < sz:
            out.append(n)
    return out


def bfs_path_exists_3d(grid: np.ndarray, start: Coord3D, goal: Coord3D) -> bool:
    """Check connectivity using 3D BFS."""
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        for n in _neighbors(cur, grid.shape):
            if n in seen or grid[n] == 1:
                continue
            seen.add(n)
            q.append(n)
    return False


def generate_environment_3d(
    size: tuple[int, int, int] = (20, 20, 8),
    difficulty: str = 'medium',
    moving_obstacle_count: int = 5,
    seed: int | None = None,
) -> WarehouseEnvironment3D:
    """Generate a valid 3D warehouse environment."""
    rng = random.Random(seed)
    densities = {'easy': 0.12, 'medium': 0.2, 'hard': 0.3}
    density = densities[difficulty]
    sx, sy, sz = size
    start = (1, 1, 1)
    goal = (sx - 2, sy - 2, sz - 2)

    while True:
        grid = np.zeros(size, dtype=np.int32)
        grid[0, :, :] = 1
        grid[-1, :, :] = 1
        grid[:, 0, :] = 1
        grid[:, -1, :] = 1
        grid[:, :, 0] = 1
        grid[:, :, -1] = 1

        target = int(sx * sy * sz * density)
        placed = 0
        while placed < target:
            base_x = rng.randint(1, sx - 2)
            base_y = rng.randint(1, sy - 2)
            col_height = rng.randint(max(2, sz // 3), sz - 2)
            for z in range(1, min(sz - 1, 1 + col_height)):
                if grid[base_x, base_y, z] == 0:
                    grid[base_x, base_y, z] = 1
                    placed += 1
                    if placed >= target:
                        break

        grid[start] = 0
        grid[goal] = 0
        if bfs_path_exists_3d(grid, start, goal):
            break

    free = [(x, y, z) for x in range(1, sx - 1) for y in range(1, sy - 1) for z in range(1, sz - 1) if grid[x, y, z] == 0 and (x, y, z) not in (start, goal)]
    rng.shuffle(free)
    moving = set(free[:moving_obstacle_count])
    return WarehouseEnvironment3D(grid=grid, start=start, goal=goal, moving_obstacles=moving)


def save_environment_3d(env: WarehouseEnvironment3D, path: Path) -> None:
    """Save a 3D environment as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'grid': env.grid.tolist(),
        'start': list(env.start),
        'goal': list(env.goal),
        'moving_obstacles': [list(v) for v in sorted(env.moving_obstacles)],
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def load_environment_3d(path: Path) -> WarehouseEnvironment3D:
    """Load a 3D environment from JSON."""
    data = json.loads(path.read_text(encoding='utf-8'))
    return WarehouseEnvironment3D(
        grid=np.array(data['grid'], dtype=np.int32),
        start=tuple(data['start']),
        goal=tuple(data['goal']),
        moving_obstacles={tuple(v) for v in data['moving_obstacles']},
    )


def plot_environment_3d(env: WarehouseEnvironment3D, title: str = '3D Warehouse', path: Path | None = None) -> None:
    """Plot 3D voxel environment using a clean color palette."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#eaf2ff')

    obs = np.argwhere(env.grid == 1)
    if len(obs):
        ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], c='#0f172a', s=10, alpha=0.45, label='Shelf / Wall')

    if env.moving_obstacles:
        mo = np.array(list(env.moving_obstacles))
        ax.scatter(mo[:, 0], mo[:, 1], mo[:, 2], c='#ff4d6d', s=38, alpha=0.9, label='Moving Obstacle')

    ax.scatter(*env.start, c='#7c3aed', s=110, marker='o', label='Start')
    ax.scatter(*env.goal, c='#22c55e', s=110, marker='*', label='Goal')

    ax.set_title(title, fontsize=13, fontweight='bold', color='#0f172a')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left')
    fig.tight_layout()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)
