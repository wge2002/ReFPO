"""Train FPO on a MuJoCo Playground task."""

from pathlib import Path

import tyro

from flow_policy import fpo
from flow_policy.runner import train


def main(env_name: str, output_dir: Path, config: fpo.FpoConfig, seed: int = 0):
    train("fpo", env_name, output_dir, config, seed)


if __name__ == "__main__":
    tyro.cli(main)
