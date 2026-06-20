"""Routix failure analysis and episode classification tools."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def classify_failure(success: bool, steps: int, max_steps: int, revisit_max: int, max_same_cell_run: int, collision_bursts: int, far_from_goal: bool, timeout_threshold: int, loop_threshold: int) -> str:
    """Classify episode as success or failure type."""
    if success:
        return 'success'
    if steps >= min(max_steps, timeout_threshold):
        return 'timeout'
    if revisit_max > loop_threshold:
        return 'loop'
    if max_same_cell_run > 8:
        return 'stuck'
    if collision_bursts > 5:
        return 'collision_storm'
    if far_from_goal:
        return 'battery_death'
    return 'timeout'

def plot_failure_breakdown(rows: list[dict], path: Path) -> None:
    """Plot failure pie chart."""
    counts = Counter(r.get('failure_type','timeout') for r in rows)
    labels = list(counts.keys())
    vals = [counts[x] for x in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.pie(vals, labels=labels, autopct='%1.1f%%')
    plt.title('Failure Breakdown')
    plt.savefig(path, dpi=140)
    plt.close()

def plot_failure_heatmap(rows: list[dict], grid_shape: tuple[int,int], path: Path) -> None:
    """Plot failure final-position heatmap."""
    heat = np.zeros(grid_shape, dtype=float)
    for r in rows:
        if r.get('failure_type') == 'success':
            continue
        rr = int(r.get('final_row',0)); cc = int(r.get('final_col',0))
        if 0<=rr<grid_shape[0] and 0<=cc<grid_shape[1]:
            heat[rr,cc] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.imshow(heat, cmap='viridis')
    plt.colorbar(label='Failure Count')
    plt.title('Failure Heatmap')
    plt.savefig(path, dpi=140)
    plt.close()
