# Grid-Nexus

## Why Verdex Exists

Verdex started from a stubborn operational fact: the warehouse industry often loses more money to navigation failure recovery than to robot hardware itself. Teams budget for robots, but they quietly bleed margin through reroutes, stalled lanes, emergency overrides, and manual intervention when paths break under real floor conditions. That cost compounds across every shift.

Most deployed systems still rely on brittle path maps and lightweight obstacle logic. In theory they look stable, but in practice warehouses are living systems with moved pallets, temporary staging, blocked aisles, and unpredictable human traffic. Those events are routine, not rare, and they can strip 10-15% of annual operational efficiency through downtime and missed throughput windows.

Verdex exists to close that gap with a simulation-first navigation intelligence layer. Instead of waiting for failures in production, operators can train and stress test policies on realistic digital environments before rollout. The objective is not a better robot body. The objective is better route decisions under dynamic uncertainty.

Grid-Nexus is the technical rebrand of that vision inside this repository. It is designed as a serious college-level project that still feels like a startup-grade foundation: understandable modules, reproducible experiments, measurable outcomes, and a practical path toward productization for real warehouse operators.

## Verdex

Grid-Nexus is a 2D warehouse robot navigation simulator written in Python. It combines classical pathfinding and reinforcement learning to study how agents perform when environments change during runtime. The platform intentionally stays transparent and educational while still covering the core lifecycle needed for applied navigation intelligence.

The project includes environment generation, sensor simulation, baseline planners, Q-learning, DQN, diagnostics, failure analysis, efficiency scoring, difficulty analysis, benchmarking, and side-by-side policy comparison. That combination makes it possible to evaluate not only whether an agent reaches a goal, but how reliably and efficiently it does so.

## The Problem

Static route plans fail whenever warehouse layouts drift from assumptions. A moved box, temporary pallet stack, human crossing, or disabled robot can break a previously valid path and force expensive intervention. Traditional systems often recover slowly because they were designed around predictable maps and low-change dynamics.

Operationally, this creates two risks: throughput volatility and maintenance overhead. Throughput volatility reduces service quality and planning confidence, while maintenance overhead pulls staff into constant exception handling. A robust system needs adaptive behavior plus clear diagnostics that explain policy breakdowns before they become business incidents.

## The Solution

Grid-Nexus addresses the problem with a simulation and learning pipeline that starts with realistic random warehouse generation, then trains policies against dynamic obstacles and noisy sensor observations. The simulator enforces path validity and configurable difficulty so experiments remain both challenging and reproducible.

The platform evaluates navigation through multiple lenses: success rate, reward, step cost, health degradation, failure type distribution, environment difficulty, and the Verdex Navigation Score. This gives a fuller signal than success alone and supports policy decisions that map to real warehouse priorities like reliability, wear reduction, and safe flow.

## What Was Built

The codebase includes a complete environment engine with save and load support, dynamic obstacle updates, and visualization. A sensor module emits 10-dimensional observations from ray casting and goal-relative features. A pathfinding module implements BFS, Dijkstra, and A* from scratch for baseline comparisons.

On the learning side, the repository includes a tabular Q-learning agent and a PyTorch DQN implementation with replay buffer and target network updates. Supporting modules provide diagnostics, scoring, failure classification, difficulty analysis, benchmark execution, and side-by-side policy comparison artifacts.

## Tools And Libraries

The project uses Python with NumPy for numerical operations, PyTorch for DQN training, and Matplotlib for visual output and analysis plots. No extra external framework is required for core functionality, which keeps setup straightforward while retaining enough power for meaningful experiments.

All file paths are handled with pathlib and configuration is centralized in `config.json`. This makes the project portable and easier to adapt to different local environments, CI jobs, or class demonstration workflows.

## Repository Structure

```text
grid-nexus/
├── main.py
├── environment.py
├── sensors.py
├── pathfinder.py
├── q_agent.py
├── dqn_agent.py
├── visualizer.py
├── logger.py
├── benchmark.py
├── diagnostics.py
├── scorer.py
├── analyzer.py
├── failure_logger.py
├── comparator.py
├── generate_examples.py
├── config.json
├── requirements.txt
├── models/
├── logs/
│   └── plots/
├── reports/
├── environments/
├── examples/
└── README.md
```

## How The System Works

The pipeline begins with environment generation. A warehouse grid is built with aisle-like obstacle clusters, start and goal are fixed in opposite corners, and BFS validates that at least one path exists. Dynamic obstacles are added to represent moving humans or forklifts that alter route safety during an episode.

The agent receives a compact sensor vector each step. Eight rays estimate obstacle distance by direction, and two goal features encode normalized angle and distance. That vector drives either a Q-table policy or a neural policy. Both policies are evaluated under the same reward structure and logging scheme.

After each episode, metrics are written to CSV and consumed by analytics modules. These analytics generate plots, diagnostics reports, benchmark comparisons, failure visualizations, and policy comparison artifacts. This creates a full loop from environment generation to model behavior interpretation.

## Environment Generation

The environment generator supports easy, medium, and hard presets tied to obstacle density. It places obstacles in vertical clusters to mimic warehouse aisle structure and avoids trivial maps by preserving structural complexity and connectivity checks.

Dynamic obstacles are updated each episode so the navigation policy must remain reactive instead of memorizing static routes. Environments can be saved to and loaded from JSON for reproducible experiments and regression checks.

<!-- Add screenshot here: easy difficulty environment layout -->

![Description](docs/images/environment_easy.png)

<!-- Add screenshot here: medium difficulty environment layout -->

![Description](docs/images/environment_medium.png)

<!-- Add screenshot here: hard difficulty environment layout -->

![Description](docs/images/environment_hard.png)

## Sensor Simulation

The sensor system emulates a lightweight perception stack. It casts eight rays in cardinal and diagonal directions and normalizes distances to nearest walls or obstacles. Small Gaussian noise is added to mimic real-world sensor instability and prevent overfitting to perfect measurements.

Two additional goal-relative features are appended to the ray vector, producing a 10-value input that both tabular and neural agents can consume. The module also includes a visualization helper for inspecting ray geometry and perception quality in difficult zones.

<!-- Add screenshot here: sensor rays in open corridor -->

![Description](docs/images/sensor_open_corridor.png)

<!-- Add screenshot here: sensor rays near dense obstacles -->

![Description](docs/images/sensor_dense_obstacles.png)

## Pathfinding

BFS, Dijkstra, and A* are implemented from scratch and return path coordinates, path length, and explored node count. These methods provide deterministic baselines for evaluating whether learned policies are improving actual route quality or only optimizing reward shortcuts.

The comparison function runs all three algorithms on the same environment and prints a compact terminal table. A plotting helper overlays selected paths on the map for visual inspection and debugging.

<!-- Add screenshot here: A-star path overlay -->

![Description](docs/images/path_astar_overlay.png)

<!-- Add screenshot here: pathfinding comparison table output -->

![Description](docs/images/pathfinder_comparison_table.png)

## Q-Learning Agent

The Q-learning agent rounds sensor values to one decimal place and stores action values in a dictionary-backed Q-table. Training uses epsilon-greedy exploration with decay and manual Bellman updates, making the learning logic explicit and easy to study.

Episode rewards penalize wasteful movement and collisions while strongly rewarding goal completion. Training logs include reward, steps, success, epsilon, health, failure type, and VNS metrics so model behavior can be analyzed from multiple angles.

<!-- Add screenshot here: Q-learning reward curve -->

![Description](docs/images/q_reward_curve.png)

<!-- Add screenshot here: Q-learning success rate curve -->

![Description](docs/images/q_success_rate_curve.png)

<!-- Add screenshot here: Q-learning steps per episode -->

![Description](docs/images/q_steps_curve.png)

<!-- Add screenshot here: Q-learning epsilon decay curve -->

![Description](docs/images/q_epsilon_curve.png)

## Deep Q-Network Agent

The DQN agent uses a compact 10-64-64-4 network with ReLU activations. It trains from a replay buffer with random minibatch sampling and stabilizes updates through a target network refreshed on a fixed interval.

This setup is intentionally simple enough for coursework and rapid iteration, while still representing core production RL patterns such as experience replay, target bootstrapping, and batched optimization with Adam and MSE loss.

<!-- Add screenshot here: DQN reward curve -->

![Description](docs/images/dqn_reward_curve.png)

<!-- Add screenshot here: DQN success rate curve -->

![Description](docs/images/dqn_success_rate_curve.png)

<!-- Add screenshot here: DQN steps per episode -->

![Description](docs/images/dqn_steps_curve.png)

<!-- Add screenshot here: DQN epsilon decay curve -->

![Description](docs/images/dqn_epsilon_curve.png)

## Benchmark Results

Benchmark mode runs shared test environments and reports method-level success rate, average steps, and average reward. It also computes VNS rankings and generates comparative plots so results are visible in both terminal and visual form.

The benchmark now includes EDI-based quartile grouping to show method behavior across objectively measured environment difficulty rather than only user-selected labels.

<!-- Add screenshot here: benchmark bar chart comparing methods -->

![Description](docs/images/benchmark_bar_chart.png)

<!-- Add screenshot here: benchmark summary table in terminal -->

![Description](docs/images/benchmark_terminal_summary.png)

## Robot Health And Diagnostics

The diagnostics system tracks a robot health score from 100 down to 0 based on collisions, stuck behavior, and severe inefficiency. Health below the degraded threshold raises warnings, while critical health triggers a maintenance cycle pause to simulate operational safety intervention.

After training, diagnostics mode reads the full log and writes a timestamped plain text report under `reports/`. The report summarizes average final health, degraded events, critical events, and best and worst episodes.

<!-- Add screenshot here: health curve with green yellow red zones -->

![Description](docs/images/health_curve_zones.png)

<!-- Add screenshot here: diagnostics report text output -->

![Description](docs/images/diagnostics_text_report.png)

## Navigation Efficiency And The Verdex Navigation Score

The Verdex Navigation Score combines three components: path efficiency ratio, smoothness ratio, and obstacle avoidance ratio. The weighted blend produces a 0 to 100 score that better reflects warehouse-ready behavior than binary success alone.

Path efficiency rewards direct travel, smoothness rewards lower turn churn, and obstacle avoidance rewards safer clearance behavior. In real operations, smoother and shorter paths can reduce mechanical wear, energy waste, and congestion cascades.

<!-- Add screenshot here: VNS curve across training -->

![Description](docs/images/vns_training_curve.png)

<!-- Add screenshot here: VNS leaderboard from benchmark -->

![Description](docs/images/vns_leaderboard_benchmark.png)

## Environment Difficulty Analysis

The analyzer computes objective metrics including bottleneck score, open space ratio, start-goal Manhattan distance, and path complexity ratio. These metrics are combined into EDI so map hardness is measurable and consistent across runs.

During benchmarking, EDI quartiles provide a difficulty-aware lens on performance. This helps teams distinguish between policy weakness and environment bias when interpreting results.

<!-- Add screenshot here: VNS versus EDI scatter plot -->

![Description](docs/images/vns_vs_edi_scatter.png)

<!-- Add screenshot here: EDI quartile performance breakdown -->

![Description](docs/images/edi_quartile_breakdown.png)

## Failure Analysis

Failure logging classifies each episode as success, timeout, loop, stuck, collision storm, or battery death. This makes non-success outcomes interpretable and gives actionable labels for retraining priorities.

The failure module produces a pie chart for distribution analysis and a positional heatmap to reveal warehouse regions where policies fail disproportionately.

<!-- Add screenshot here: failure type pie chart -->

![Description](docs/images/failure_type_pie.png)

<!-- Add screenshot here: failure position heatmap -->

![Description](docs/images/failure_position_heatmap.png)

## Side-By-Side Policy Comparison

Comparator mode runs Q-learning and DQN on identical environment sequences and produces paired visual frames. Each frame includes both trajectories and a live performance summary to make policy differences obvious without manual synchronization.

After all episodes, a summary chart compares average reward, success rate, average VNS, and average steps side by side, making method-level tradeoffs easy to present.

<!-- Add screenshot here: side-by-side comparator frame -->

![Description](docs/images/comparator_side_by_side_frame.png)

<!-- Add screenshot here: final comparator summary panel -->

![Description](docs/images/comparator_summary_panel.png)

## The Verdex Startup Vision

Verdex would be sold as a navigation intelligence software license for warehouse operators. The product bundle would include simulation tooling, deployable trained policies, and a diagnostics dashboard that continuously tracks policy health and operational risk.

Go-to-market starts with small and medium fulfillment centers that cannot absorb large robotics platform costs from enterprise incumbents. Instead of replacing hardware, operators upload floor plans, auto-generate a matching simulation environment, train policies in 24 hours, and deploy navigation updates as software.

Verdex Lite would provide simulation-only workflows for research teams and early operators. Verdex Core would add trained model delivery, health diagnostics, and recurring scoring and benchmark tools under subscription pricing.

Verdex Enterprise would add custom environment modeling, support SLAs, and integration layers for fleet systems and warehouse management software. This tier focuses on operators running multi-site rollouts where predictability and auditability matter as much as raw path success.

Roadmap milestones include automated floor plan ingestion, transfer policy libraries across similar facilities, controlled online policy adaptation with safety guardrails, and fleet-level performance analytics tied directly to throughput and downtime KPIs.

<!-- Add screenshot here: product tier architecture diagram -->

![Description](docs/images/product_tier_architecture.png)

## How To Run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train Q-learning:

```bash
python main.py --mode train-q --difficulty medium --episodes 500 --seed 42
```

Train DQN:

```bash
python main.py --mode train-dqn --difficulty medium --episodes 500 --seed 42
```

Test mode:

```bash
python main.py --mode test --difficulty medium --seed 42
```

A-star visualization:

```bash
python main.py --mode astar --difficulty medium --seed 42
```

Benchmark:

```bash
python main.py --mode benchmark --difficulty medium --seed 42
```

Replay:

```bash
python main.py --mode replay
```

Diagnostics report:

```bash
python main.py --mode diagnostics
```

Score trained agent on 20 environments:

```bash
python main.py --mode score --difficulty medium --seed 42
```

Analyze a new environment and print EDI:

```bash
python main.py --mode analyze-env --difficulty hard --seed 7
```

Failure analysis plots:

```bash
python main.py --mode failures
```

Compare Q-learning and DQN:

```bash
python main.py --mode compare --seed 42
```

Generate example environment images:

```bash
python generate_examples.py
```

## Configuration

`config.json` centralizes all runtime parameters. The `environment` section controls grid size, difficulty, obstacle density presets, moving obstacle count, and base seed behavior.

The `training` section controls episode count, maximum steps, and progress print interval. `q_learning` and `dqn` define learning rates, discounting, exploration behavior, replay size, batch size, and target update interval.

The `diagnostics` section controls health degradation per collision and stuck event, maintenance cycle length, and warning thresholds. The `vns` section sets weights for path efficiency, smoothness, and obstacle avoidance.

The `failure_logger` section sets timeout and loop thresholds plus collision storm sensitivity. The `comparator` section sets the number of paired test environments. The `paths` section defines model, log, plot, and report destinations.

## Engineering Lessons From This Project

One lesson is that realism in simulation is non-negotiable for learning quality. Policies trained on clean static worlds can appear effective but collapse under ordinary warehouse volatility. Introducing dynamic obstacles and noise early produces more trustworthy behavior profiles.

A second lesson is that reward alone is a weak instrument for operations-grade interpretation. Health scoring, failure taxonomy, and VNS components reveal why an agent behaves poorly and where corrective training should focus.

A third lesson is that deterministic baselines remain essential in ML-heavy systems. BFS, Dijkstra, and A-star provide grounding constraints for expected path quality and expose whether learned behavior is truly improving or only gaming reward shaping.

A fourth lesson is that modular architecture accelerates iteration speed. Cleanly separated environment, sensing, learning, scoring, and visualization modules made it possible to add advanced analytics without destabilizing training logic.

A fifth lesson is that visual diagnostics improve decision quality for mixed technical teams. Side-by-side policy frames, heatmaps, and trend plots communicate failure modes faster than raw logs and help align engineering and operations stakeholders.

## What Verdex Could Become

Grid-Nexus can evolve into a full simulation-to-deployment navigation intelligence platform for existing robot fleets. With stronger data ingestion, deployment automation, and continuous monitoring, it could become a practical bridge between research-grade policy training and real warehouse uptime improvement.
