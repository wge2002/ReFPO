# ReFPO: Reflow Regularization for Flow Matching Policy Gradients

[Paper (NeurIPS 2026)](https://openreview.net/forum?id=QGCmtNLIKF)

Ge Wang, Yibo Peng, Fan Feng, Shenhao Yan, Chengsi Yao, Jiahao Yang, Honghao Cai, Yiming Zhao, Xi Li, Jinke Ren, Shuguang Cui, Yatong Han, and Zhen Li

ReFPO adds an advantage-independent flow-matching loss to Flow Policy Optimization (FPO). It reuses the policy update's sampled actions, noise, and times. A trained policy can then be evaluated with a single Euler step. Our implementation builds directly on the [FPO codebase](https://github.com/akanazawa/fpo).

## Training and evaluation

Each benchmark has its own environment. Follow the setup and commands in the corresponding directory.

| Benchmark | Implementation | Evaluation |
| --- | --- | --- |
| [GridWorld](gridworld/README.md) | PyTorch, CPU or CUDA | Goal-reaching success rate and episode return |
| [MuJoCo Playground](playground/README.md) | JAX | Paired 10-step and 1-step returns from one checkpoint, with a script to aggregate the ten-task table |
| [Humanoid tracking](phc/README.md) | PyTorch and Isaac Gym on Linux | Motion completion rate, alive duration in steps, and global joint position error in mm |

## Acknowledgments

We thank David McAllister, Songwei Ge, Brent Yi, Chung Min Kim, Ethan Weber, Hongsuk Choi, Haiwen Feng, and Angjoo Kanazawa for releasing **Flow Matching Policy Gradients**. We adapted their training code and benchmark implementations for ReFPO. [FPO paper](https://arxiv.org/abs/2507.21053) · [FPO code](https://github.com/akanazawa/fpo)

The GridWorld trainer builds on Eric Yang Yu's PPO implementation. The continuous control experiments use MuJoCo Playground and Brax. Our humanoid implementation extends FPO's adaptation of Kyoung Whan Choe's Puffer PHC, which builds on Perpetual Humanoid Control and PufferLib. We thank the authors of these projects for making their implementations available. [PPO for Beginners](https://github.com/ericyangyu/PPO-for-Beginners) · [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) · [Brax](https://github.com/google/brax) · [Puffer PHC](https://github.com/kywch/puffer-phc) · [PHC](https://github.com/ZhengyiLuo/PHC) · [PufferLib](https://github.com/PufferAI/PufferLib)

## Citation

```bibtex
@inproceedings{wang2026refpo,
  title     = {{ReFPO}: Reflow Regularization for Flow Matching Policy Gradients},
  author    = {Wang, Ge and Peng, Yibo and Feng, Fan and Yan, Shenhao and
               Yao, Chengsi and Yang, Jiahao and Cai, Honghao and Zhao, Yiming and
               Li, Xi and Ren, Jinke and Cui, Shuguang and Han, Yatong and Li, Zhen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=QGCmtNLIKF}
}
```

## Licenses

Original ReFPO contributions are available under the [MIT License](LICENSE). The distributed GridWorld and Playground code retains its [GridWorld MIT notices](gridworld/LICENSE) and [Playground Apache 2.0 license](playground/LICENSE), respectively. The humanoid directory provides our changes against a pinned upstream revision. Isaac Gym, AMASS, and SMPL are obtained separately under their providers' terms. [Humanoid setup](phc/README.md)
