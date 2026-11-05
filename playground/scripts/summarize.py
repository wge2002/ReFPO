"""Reduce paired checkpoint evaluations to one Playground row."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

TASKS = ("BallInCup", "CartpoleBalance", "CheetahRun", "FingerSpin", "FingerTurnEasy",
         "FingerTurnHard", "FishSwim", "PointMass", "ReacherEasy", "ReacherHard")
METRICS = ("return_n10", "return_n1", "cfm_mse_n10", "excursion")


def summarize(paths, partial=False):
    groups = defaultdict(dict)
    protocol = None
    for path in paths:
        record = json.loads(path.read_text())
        task, seed = record["env_name"], record["train_seed"]
        settings = (record["method"], record["episodes"], record["eval_seed"], record["horizon"],
                    record["env_steps"], record["train_config"])
        if protocol is not None and protocol != settings:
            raise ValueError(f"{path}: inconsistent training or evaluation settings")
        protocol = settings
        if task not in TASKS or seed in groups[task]:
            raise ValueError(f"{path}: unknown task or duplicate training seed for {task}")
        if not all(math.isfinite(record[key]) for key in METRICS):
            raise ValueError(f"{path}: missing or nonfinite score")
        groups[task][seed] = record
    if not partial:
        if set(groups) != set(TASKS) or any(len(runs) != 5 for runs in groups.values()):
            raise ValueError("The full table requires five distinct training seeds for each of the ten tasks")
        method, episodes, _, horizon, steps, cfg = protocol
        batch = cfg["num_minibatches"] * cfg["batch_size"] * cfg["unroll_length"]
        if (episodes != 128 or horizon != 1000 or cfg["num_timesteps"] != 100_000_000
                or steps != 100_000_000 // batch * batch or cfg["flow_steps"] != 10):
            raise ValueError("Expected final 100M-budget checkpoints, ten training steps per action and 128 full episodes")
    scores = {}
    for metric in METRICS:
        per_task = [[row[metric] for row in seeds.values()] for seeds in groups.values()]
        scores[metric] = {"mean": mean(mean(values) for values in per_task)}
        if all(len(values) > 1 for values in per_task):
            scores[metric]["mean_task_seed_sd"] = mean(stdev(values) for values in per_task)
    return {"method": protocol[0], "seeds_per_task": {k: len(v) for k, v in groups.items()}, "scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", type=Path, nargs="+")
    parser.add_argument("--partial", action="store_true", help="Summarize a smaller set of runs")
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.evaluations, args.partial), indent=2))
    except (KeyError, ValueError) as error:
        parser.error(str(error))
