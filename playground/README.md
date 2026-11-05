# MuJoCo Playground

Train ReFPO or FPO on a DeepMind Control task and compare 10-step and 1-step inference using the same policy weights, observation statistics, resets, and action-noise keys.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
# For NVIDIA GPUs on Linux:
python -m pip install 'jax[cuda12]==0.6.2'
python -c 'import jax; print(jax.devices())'
```

MuJoCo Playground downloads its official Menagerie assets on first use. CPU is sufficient for the short check below.

## Train and evaluate one seed

```bash
python scripts/train_refpo.py --env-name CartpoleBalance --seed 0 --output-dir runs/cartpole-s0
python scripts/evaluate.py --checkpoint runs/cartpole-s0/checkpoint.npz \
  --flow-steps 10 --episodes 128 --seed 0 --output scores/cartpole-s0.json
```

One evaluation writes both returns (`return_n10`, `return_n1`) to the output JSON. Use `scripts/train_fpo.py` for the baseline. Run either trainer or the evaluator with `--help` to see its options.

## Aggregate the main table

Run each of the following environments with five training seeds and evaluate 128 episodes per seed. Use the same evaluation seed and settings across checkpoints.

```text
BallInCup        CartpoleBalance   CheetahRun   FingerSpin   FingerTurnEasy
FingerTurnHard   FishSwim          PointMass    ReacherEasy  ReacherHard
```

Collect the 50 evaluation JSONs for one method and configuration in one directory:

```bash
python scripts/summarize.py scores/*.json
```

For a partial set of tasks or seeds, add `--partial`.

## Short CPU check

```bash
python scripts/train_refpo.py --env-name CartpoleBalance --output-dir runs/small \
  --config.num-timesteps 32 --config.num-envs 2 --config.batch-size 2 \
  --config.num-minibatches 2 --config.unroll-length 4 --config.num-updates-per-batch 1 \
  --config.num-evals 1 --config.episode-length 8
python scripts/evaluate.py --checkpoint runs/small/checkpoint.npz --episodes 3 --output scores/small.json
```
