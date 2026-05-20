"""Robot health and diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class HealthState:
    """Track health state for an episode."""
    health: float = 100.0
    degraded_events: int = 0
    critical_events: int = 0
    maintenance_remaining: int = 0

def update_health(state: HealthState, collisions: int, stuck_events: int, inefficient_events: int, degradation_per_collision: float, degradation_per_stuck_event: float, degraded_threshold: float, critical_threshold: float, maintenance_cycle_steps: int) -> HealthState:
    """Update health using degradation rules."""
    if state.maintenance_remaining > 0:
        state.maintenance_remaining -= 1
        return state
    state.health -= collisions*degradation_per_collision
    state.health -= stuck_events*degradation_per_stuck_event
    state.health -= inefficient_events*(degradation_per_stuck_event*0.5)
    state.health = max(0.0, state.health)
    if state.health < degraded_threshold:
        state.degraded_events += 1
        print(f'[Diagnostics] degraded health={state.health:.1f}')
    if state.health < critical_threshold:
        state.critical_events += 1
        state.maintenance_remaining = maintenance_cycle_steps
        print('[Diagnostics] critical maintenance triggered')
    return state

def plot_health_curve(values: list[float], path: Path) -> None:
    """Plot health curve with zones."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = range(1, len(values)+1)
    plt.figure(figsize=(8,4))
    plt.plot(x, values, color='black')
    plt.axhspan(70,100,color='green',alpha=0.2)
    plt.axhspan(30,70,color='yellow',alpha=0.2)
    plt.axhspan(0,30,color='red',alpha=0.2)
    plt.ylim(0,100)
    plt.title('Robot Health Over Episodes')
    plt.savefig(path, dpi=140)
    plt.close()

def diagnostics_report(rows: list[dict], reports_dir: Path) -> Path:
    """Write text diagnostics report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    health = [float(r.get('final_health',100.0)) for r in rows] or [100.0]
    text = '\n'.join([
        'Verxify Diagnostics Report',
        '='*30,
        f'Episodes analyzed: {len(rows)}',
        f'Average final health: {sum(health)/len(health):.2f}',
        f'Degraded events: {sum(int(r.get("degraded_events",0)) for r in rows)}',
        f'Critical events: {sum(int(r.get("critical_events",0)) for r in rows)}',
        f'Worst health episode: {health.index(min(health))+1}',
        f'Best health episode: {health.index(max(health))+1}'
    ]) + '\n'
    out = reports_dir / f'diagnostics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    out.write_text(text, encoding='utf-8')
    return out
