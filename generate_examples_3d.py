"""Generate Navra 3D example diagrams for documentation."""
from __future__ import annotations

from pathlib import Path

from environment3d import generate_environment_3d, plot_environment_3d


def generate_examples_3d(count: int = 4) -> None:
    """Generate and save 3D environment diagrams."""
    out = Path('examples')
    out.mkdir(parents=True, exist_ok=True)
    diffs = ['easy', 'medium', 'hard', 'medium']
    for i in range(count):
        env = generate_environment_3d(size=(20, 20, 8), difficulty=diffs[i % len(diffs)], seed=400 + i)
        plot_environment_3d(env, title=f'{diffs[i % len(diffs)].capitalize()} Voxel Layout', path=out / f'environment3d_{i + 1}.png')


if __name__ == '__main__':
    """CLI entry point for 3D example generation."""
    generate_examples_3d(4)
