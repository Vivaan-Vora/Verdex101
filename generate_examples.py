"""Navra example image generator."""
from __future__ import annotations
from pathlib import Path
from environment import generate_environment, plot_environment

def generate_examples(count: int=5) -> None:
    """Generate and save sample environment images."""
    out=Path('examples'); out.mkdir(parents=True,exist_ok=True); dif=['easy','medium','hard','medium','easy']
    for i in range(count):
        env=generate_environment(difficulty=dif[i%len(dif)],seed=100+i)
        title=f'{dif[i%len(dif)].capitalize()} Grid Layout'
        plot_environment(env,title=title,path=out/f'environment_{i+1}.png')

if __name__=='__main__':
    """Script entry."""
    generate_examples(5)
