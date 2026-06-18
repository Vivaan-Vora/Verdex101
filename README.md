# Navra

## Foundation

Navra is a warehouse navigation intelligence simulator built for both 2D grid navigation and 3D voxel navigation. It combines classical planning and reinforcement learning so teams can test route quality, reliability, and failure behavior under realistic warehouse constraints.

Navigation is often the biggest failure point in warehouse automation. When paths break due to floor changes, operations lose throughput and teams are forced into manual intervention. Navra addresses this with a simulation-first workflow so teams can validate behavior before live deployment.

The project is implemented in **Python** and uses **NumPy** for numerical processing, **PyTorch** for deep Q-network training, and **Matplotlib** for 2D and 3D visualizations.

<p align="center">
  <img src="examples/layout-comparison.png" alt="Grid Layout and Voxel Layout" width="75%" />
</p>

## Core Capabilities

- 2D environment generation with moving obstacles and difficulty control
- 3D voxel environment generation with connectivity validation
- 2D and 3D sensor simulation modules
- Baseline pathfinding in both dimensions (BFS, Dijkstra, A*)
- Q-learning and DQN training support
- Logging, diagnostics, scoring, failure analysis, and benchmark tools

<p align="center">
  <img src="examples/layout-grid.png" alt="Grid Layout" width="75%" />
</p>

<p align="center">
  <img src="examples/layout-voxel.png" alt="Voxel Layout" width="75%" />
</p>

## Repository Structure

```text
navra/
├── main.py
├── environment.py
├── environment3d.py
├── sensors.py
├── sensors3d.py
├── pathfinder.py
├── pathfinder3d.py
├── q_agent.py
├── dqn_agent.py
├── benchmark.py
├── diagnostics.py
├── scorer.py
├── analyzer.py
├── failure_logger.py
├── comparator.py
├── logger.py
├── visualizer.py
├── generate_examples.py
├── generate_examples_3d.py
├── config.json
├── models/
├── examples/
├── logs/
└── README.md
```

## CLI Modes

| Mode | Purpose |
|------|---------|
| `train-q` | Train tabular Q-learning agent |
| `train-dqn` | Train deep Q-network agent |
| `astar` / `astar-3d` | Run baseline planning |
| `benchmark` | Compare pathfinding methods |
| `diagnostics` | Generate health reports |

## Quickstart

```bash
pip install -r requirements.txt
python main.py --mode astar --difficulty medium
python main.py --mode train-q --episodes 100
```

## Evaluation Modes

Navra supports side-by-side evaluation of classical planners and learned policies. Use `benchmark` for pathfinding comparisons and `compare` for agent rollouts on shared environments.

## Pipeline

Navra supports a unified workflow for both 2D and 3D navigation experiments.

1. Generate a valid 2D or 3D warehouse layout.
2. Simulate observations from the relevant sensor module.
3. Run pathfinding and/or learning agents.
4. Log reward, stability, health, and failure metrics.
5. Analyze outputs with plots, leaderboards, and diagnostics reports.

## Output Paths

Training artifacts are written to the paths configured in `config.json` (`models/`, `logs/`, and `reports/`).

## Dependencies

Pinned versions live in `requirements.txt` (NumPy, PyTorch, Matplotlib).

## Troubleshooting

- If plots do not appear, confirm Matplotlib is installed and writable output folders exist.
- For unstable training, reduce obstacle density or increase `max_steps` in `config.json`.
- Use `--seed` on the CLI to reproduce a specific environment layout.

## Configuration

The Navra `config.json` file includes settings for both environment types, training hyperparameters, diagnostics thresholds, navigation scoring weights, and output paths.

For 3D, `environment_3d` controls voxel size, difficulty, and moving obstacle count.

## Lessons From This Project

Building Navra showed that supporting both 2D and 3D navigation in one system is useful but requires careful design. Good simulation setup matters because changing obstacles and sensor noise can quickly change agent results. Reward alone is not enough to judge performance, so adding diagnostics, failure tracking, and scoring made it easier to understand what was working and what needed improvement. Keeping modules separate for environment setup, sensing, planning, training, and analysis also made the project easier to build and extend over time.

## License

Navra is released under the MIT License. See `LICENSE` for details.
