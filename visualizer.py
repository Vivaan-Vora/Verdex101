"""Navra visualization and episode replay helpers."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_episode(grid, path, goal, rays=None):
    """Animate one episode path."""
    fig,ax=plt.subplots(figsize=(6,6)); ax.imshow(grid,cmap='gray_r'); dot,=ax.plot([],[],'bo'); ax.plot(goal[1],goal[0],'go')
    def init():
        """Initialize animation artist."""
        dot.set_data([],[]); return (dot,)
    def update(i):
        """Update animation frame."""
        r,c=path[min(i,len(path)-1)]; dot.set_data([c],[r]); return (dot,)
    return FuncAnimation(fig,update,frames=len(path),init_func=init,interval=100,blit=True,repeat=False)

def plot_live_rewards(rewards, path: Path|None=None):
    """Plot reward curve."""
    plt.figure(figsize=(8,4)); plt.plot(range(1,len(rewards)+1),rewards); plt.title('Episode Reward'); plt.tight_layout()
    if path: path.parent.mkdir(parents=True,exist_ok=True); plt.savefig(path,dpi=140)
    else: plt.show()
    plt.close()

def replay_from_log(rows):
    """Extract replay trace from rows."""
    return [(int(r.get('final_row',0)),int(r.get('final_col',0))) for r in rows]
