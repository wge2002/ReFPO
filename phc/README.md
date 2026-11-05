# Humanoid motion tracking

This directory adds ReFPO training and evaluation to the FPO implementation of Puffer PHC. The [patch](refpo.patch) applies to FPO revision `418c2554f7cd22d52e14c07d951280929d73bf2f`. It adds the raw CFM loss to the policy update and exposes the tracking metrics for a saved checkpoint. [FPO](https://github.com/akanazawa/fpo) · [Puffer PHC](https://github.com/kywch/puffer-phc)

## Setup

Use Linux with an NVIDIA GPU, a CUDA-compatible toolchain, Pixi and an Isaac Gym installation from NVIDIA. Starting in this directory:

```bash
git clone https://github.com/akanazawa/fpo.git fpo
git -C fpo checkout --detach 418c2554f7cd22d52e14c07d951280929d73bf2f
git -C fpo apply --check ../refpo.patch
git -C fpo apply ../refpo.patch
cd fpo/phc
pixi install
pixi shell
python -m pip install -e /path/to/isaacgym/python
python -c 'import isaacgym, torch; print(torch.cuda.is_available())'
```

Obtain AMASS motions and SMPL model parameters from their original providers under their respective research licenses. Follow PHC's motion preparation instructions and place the processed training motion file at `amass/amass_train_11313_upright.pkl`. Place the neutral, male and female models at `smpl/SMPL_NEUTRAL.pkl`, `smpl/SMPL_MALE.pkl`, and `smpl/SMPL_FEMALE.pkl`. The simulation backend and motion/model data are external dependencies. [Isaac Gym](https://developer.nvidia.com/isaac-gym) · [AMASS](https://amass.is.tue.mpg.de/) · [SMPL](https://smpl.is.tue.mpg.de/) · [PHC preparation](https://github.com/ZhengyiLuo/PHC)

## Train

From `fpo/phc/`, choose one conditioning regime and one seed:

```bash
python scripts/train.py --config configs/fpo.ini --train.seed 0
```

Use `configs/fpo.ini` for all joints, `configs/fpo_hand.ini` for root and hands, or `configs/fpo_root.ini` for root only. Run `python scripts/train.py --config configs/fpo.ini --help` for options.

## Evaluate the same checkpoint

Set `ckpt` to the saved `model_*.pt` in the training run's directory. Use the same conditioning and parameterization as training. Keep the evaluation seed and motion file fixed.

```bash
ckpt=experiments/YOUR_RUN/YOUR_CHECKPOINT.pt
python scripts/train.py --config configs/fpo.ini --mode eval -c "$ckpt" \
  --train.seed 0 --policy.solver-step-size 0.1 --eval-output scores/full-n10-s0.json
python scripts/train.py --config configs/fpo.ini --mode eval -c "$ckpt" \
  --train.seed 0 --policy.solver-step-size 1.0 --eval-output scores/full-n1-s0.json
```

The JSON reports motion completion as a fraction (`eval/success_rate`), mean steps alive (`alive_steps`), and global MPJPE in millimetres for all motions (`eval/mpjpe_all`) and successful motions (`eval/mpjpe_succ`).

For a table row, evaluate five independently trained 100M-step policies on all 11,313 motions with one conditioning regime and solver setting. From `fpo/phc/`:

```bash
python ../../summarize.py scores/full-n1-s*.json
```
