# GridWorld

Train a PyTorch flow policy in the two-wall environment. A successful episode reaches a goal cell. Death cells and timeouts count as failures.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python main.py --method refpo --seed 0 --output-dir runs/refpo
for n in 10 1; do
  python main.py --mode test --method refpo --actor-model runs/refpo/actor.pth \
    --num-steps "$n" --seed 0 --eval-episodes 100 --output-dir "runs/eval-n$n"
done
```

Training writes `actor.pth` and `critic.pth`. Each evaluation writes `eval_metrics.json` with the goal-reaching success rate and episode returns. Use `--method fpo` for the baseline and `python main.py --help` for other options.
