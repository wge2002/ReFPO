"""Train or evaluate ReFPO, FPO, and PPO in GridWorld."""
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

from models.ppo import PPO
from models.fpo import FPO
from models.refpo import ReFPO
from models.network import FeedForwardNN
from models.diffusion_policy import DiffusionPolicy
from utils.arguments import get_args
from utils.eval_policy import evaluate_episodes
from utils.gridworld import GridWorldEnv


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv(mode=args.grid_mode, max_steps=args.max_timesteps_per_episode)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    output = Path(args.output_dir or f"runs/{args.method}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    policy_classes = {"fpo": DiffusionPolicy, "refpo": DiffusionPolicy}
    try:
        if args.mode == "test":
            obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
            if args.method == "ppo":
                policy = FeedForwardNN(obs_dim, act_dim).to(device)
            else:
                policy = policy_classes[args.method](obs_dim + act_dim + 1, act_dim, num_steps=args.num_steps).to(device)
            policy.load_state_dict(torch.load(args.actor_model, map_location=device, weights_only=True))
            policy.eval()
            action_fn = policy if args.method == "ppo" else policy.sample_action
            results = evaluate_episodes(action_fn, env, args.eval_episodes, args.max_timesteps_per_episode, args.seed, args.render)
            results.update(method=args.method, flow_steps=None if args.method == "ppo" else args.num_steps, seed=args.seed)
            (output / "eval_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results, indent=2))
            return

        cfm_coef = (0.1 if args.method == "refpo" else 0.0) if args.cfm_coef is None else args.cfm_coef
        hyperparameters = {
            "timesteps_per_batch": args.timesteps_per_batch,
            "max_timesteps_per_episode": args.max_timesteps_per_episode,
            "gamma": 0.99, "n_updates_per_iteration": args.n_updates_per_iteration,
            "lr": 3e-4, "clip": 0.2, "render": args.render, "render_every_i": 1000,
            "grid_mode": args.grid_mode, "num_fpo_samples": args.num_fpo_samples,
            "positive_advantage": False, "total_timesteps": args.total_timesteps,
            "cfm_coef": cfm_coef, "num_minibatches": args.num_minibatches,
            "actor_kwargs": {"num_steps": args.num_steps}, "seed": args.seed,
            "save_freq": args.save_freq, "method": args.method,
            "output_dir": str(output.resolve()), "run_name": output.name,
        }
        (output / "run_config.json").write_text(json.dumps(hyperparameters, indent=2) + "\n")
        wandb.init(project=args.wandb_project, name=output.name, mode=args.wandb_mode, config=hyperparameters, dir=str(output))
        if args.method == "ppo":
            model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
        else:
            model = {"fpo": FPO, "refpo": ReFPO}[args.method](
                actor_class=policy_classes[args.method], critic_class=FeedForwardNN, env=env, **hyperparameters)
        if args.actor_model:
            # Weight-only warm start: optimizer and step counters start afresh.
            model.actor.load_state_dict(torch.load(args.actor_model, map_location=device, weights_only=True))
            model.critic.load_state_dict(torch.load(args.critic_model, map_location=device, weights_only=True))
        model.learn(args.total_timesteps)
        model.save_models()
        print(f"Saved actor.pth and critic.pth in {output.resolve()}")
    finally:
        env.close()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main(get_args())
