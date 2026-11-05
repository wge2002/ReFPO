"""Summarize five checkpoint evaluations for one humanoid table row."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def summarize(records):
    if len(records) != 5 or len({r["train_seed"] for r in records}) != 5:
        raise ValueError("Expected five distinct training seeds")
    settings = ("conditioning", "parameterization", "reflow_coeff", "step_size", "eval_seed", "motion_file")
    if any(tuple(r[key] for key in settings) != tuple(records[0][key] for key in settings) for r in records):
        raise ValueError("Mixed conditioning, loss, solver, evaluation seed or motion file")
    if any(r["train_seed"] is None or r["reflow_coeff"] is None or r["motions"] != 11313
           or r["env_steps"] is None or not 100_000_000 <= r["env_steps"] < 100_000_000 + 131072 for r in records):
        raise ValueError("Expected final 100M-step checkpoints evaluated on 11,313 motions")
    scores = {}
    for field, name, scale in [("eval/success_rate", "success_percent", 100),
                               ("alive_steps", "alive_steps", 1), ("eval/mpjpe_all", "mpjpe_all_mm", 1),
                               ("eval/mpjpe_succ", "mpjpe_success_mm", 1)]:
        if field == "eval/mpjpe_succ" and any(r[field] is None for r in records):
            scores[name] = None
            continue
        values = [r[field] * scale for r in records]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite " + field)
        scores[name] = {"mean": mean(values), "seed_sd": stdev(values)}
    return dict(settings={k: records[0][k] for k in settings}, scores=scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize([json.loads(p.read_text()) for p in args.evaluations]), indent=2))
    except (KeyError, ValueError) as error:
        parser.error(str(error))
