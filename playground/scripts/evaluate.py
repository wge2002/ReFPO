"""Load a saved policy; with --flow-steps 10, also evaluate the paired 1-step policy."""
from pathlib import Path

import tyro

from flow_policy.runner import evaluate


def main(checkpoint: Path, output: Path = Path("evaluation.json"), flow_steps: int = 10,
         episodes: int = 128, seed: int = 0, episode_length: int | None = None):
    evaluate(checkpoint, output, flow_steps, episodes, seed, episode_length)


if __name__ == "__main__":
    tyro.cli(main)
