"""Train REFPO on a MuJoCo Playground task."""

from pathlib import Path

import tyro

from flow_policy import refpo
from flow_policy.runner import train


def main(env_name: str, output_dir: Path, config: refpo.FpoConfig, seed: int = 0):
    train("refpo", env_name, output_dir, config, seed)


if __name__ == "__main__":
    tyro.cli(main)
