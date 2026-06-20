"""Routix 3D voxel sensor ray simulation."""
from __future__ import annotations

import math
import numpy as np

from environment3d import WarehouseEnvironment3D

Coord3D = tuple[int, int, int]

RAY_DIRS_3D = [
    (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
    (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
    (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
]


def _cast_ray_3d(env: WarehouseEnvironment3D, pos: Coord3D, direction: Coord3D, max_range: int) -> float:
    """Cast one 3D ray and return normalized hit distance."""
    for step in range(1, max_range + 1):
        p = (pos[0] + direction[0] * step, pos[1] + direction[1] * step, pos[2] + direction[2] * step)
        if not env.in_bounds(p) or env.is_obstacle(p):
            return step / max_range
    return 1.0


def get_sensor_vector_3d(
    env: WarehouseEnvironment3D,
    position: Coord3D,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return flattened 3D sensor vector with goal orientation features."""
    rng = rng or np.random.default_rng()
    max_range = max(env.grid.shape)
    rays = np.array([_cast_ray_3d(env, position, d, max_range) for d in RAY_DIRS_3D], dtype=np.float32)
    rays = np.clip(rays + rng.normal(0.0, noise_std, size=rays.shape), 0.0, 1.0)

    dx = env.goal[0] - position[0]
    dy = env.goal[1] - position[1]
    dz = env.goal[2] - position[2]
    azimuth = (math.atan2(dy, dx) + math.pi) / (2 * math.pi)
    horizontal = math.sqrt(dx * dx + dy * dy)
    elevation = (math.atan2(dz, horizontal) + math.pi / 2) / math.pi
    distance = min(math.sqrt(dx * dx + dy * dy + dz * dz) / math.sqrt(sum((s - 1) ** 2 for s in env.grid.shape)), 1.0)
    return np.concatenate([rays, np.array([azimuth, elevation, distance], dtype=np.float32)])
