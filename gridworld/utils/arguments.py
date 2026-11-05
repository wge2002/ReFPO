"""Command-line settings for GridWorld training and finite evaluation."""
import argparse


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--method", choices=("ppo", "fpo", "refpo"), default="refpo")
    parser.add_argument("--actor-model", "--actor_model", default="")
    parser.add_argument("--critic-model", "--critic_model", default="")
    parser.add_argument("--cfm-coef", "--cfm_coef", type=float, default=None)
    parser.add_argument("--total-timesteps", type=positive_int, default=260000)
    parser.add_argument("--timesteps-per-batch", type=positive_int, default=2048)
    parser.add_argument("--max-timesteps-per-episode", type=positive_int, default=200)
    parser.add_argument("--num-minibatches", type=positive_int, default=6)
    parser.add_argument("--n-updates-per-iteration", type=positive_int, default=10)
    parser.add_argument("--num-fpo-samples", type=positive_int, default=50)
    parser.add_argument("--num-steps", type=positive_int, default=10)
    parser.add_argument("--eval-episodes", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-mode", choices=("two_walls", "three_goals", "tree_in_the_middle", "two_slits", "cshape"), default="two_walls")
    parser.add_argument("--output-dir", default=None, help="New run directory; defaults to runs/<method>_<timestamp>.")
    parser.add_argument("--save-freq", type=positive_int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb-project", default="refpo-gridworld")
    args = parser.parse_args(argv)
    if args.cfm_coef is not None and not 0 <= args.cfm_coef < float("inf"):
        parser.error("--cfm-coef must be finite and nonnegative")
    if args.method in ("fpo", "ppo") and args.cfm_coef not in (None, 0):
        parser.error("--cfm-coef requires --method refpo")
    if args.num_minibatches > args.timesteps_per_batch:
        parser.error("--num-minibatches must not exceed --timesteps-per-batch")
    if args.mode == "test" and not args.actor_model:
        parser.error("--mode test requires --actor-model")
    if args.mode == "train" and bool(args.actor_model) != bool(args.critic_model):
        parser.error("weight warm starts require both --actor-model and --critic-model")
    return args
