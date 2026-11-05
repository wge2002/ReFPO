"""Train a flow policy and evaluate returns at two Euler step counts."""
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from mujoco_playground import registry

from . import fpo, refpo, rollouts
from .checkpoints import restore_policy, save_policy

METHODS = {"fpo": (fpo.FpoConfig, fpo.FpoState),
           "refpo": (refpo.FpoConfig, refpo.RefpoState)}


def initialize(method, task, config, seed):
    env = registry.load(task, config=registry.get_default_config(task))
    _, state_type = METHODS[method]
    return state_type.init(jax.random.key(seed), env, config)


def train(method, env_name, output_dir, config, seed):
    batch = config.unroll_length * config.num_minibatches * config.batch_size
    if min(config.num_envs, config.num_minibatches, config.num_updates_per_batch,
           config.unroll_length, config.batch_size, config.flow_steps, config.num_evals,
           config.n_samples_per_action, config.episode_length) < 1 or config.num_timesteps < batch:
        raise ValueError("Use positive rollout dimensions and at least one batch of timesteps")
    if (config.num_minibatches * config.batch_size) % config.num_envs:
        raise ValueError("The batch must split into whole unrolls for every environment")
    if config.loss_mode != "fpo" or not config.average_losses_before_exp:
        raise ValueError("This entry point uses the averaged CFM proxy")
    agent = initialize(method, env_name, config, seed)
    rollout = rollouts.BatchedRolloutState.init(agent.env, jax.random.key(seed), config.num_envs)
    metadata = dict(method=method, env_name=env_name, train_seed=seed, config=dataclasses.asdict(config))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    updates = config.num_timesteps // batch
    save_at = set(np.linspace(1, updates, min(config.num_evals, updates), dtype=int)) | {updates}
    excursion_sum = 0.0
    with (output_dir / "train_metrics.jsonl").open("x") as log:
        for update in range(1, updates + 1):
            rollout, transitions = rollout.rollout(agent, config.episode_length, config.iterations_per_env)
            agent, metrics = agent.training_step(transitions)
            row = {k: float(v.mean()) for k, v in metrics.items()}
            if not all(np.isfinite(v) for v in row.values()):
                raise FloatingPointError(f"Non-finite training metric at update {update}")
            excursion_sum += row["log_ratio_excursion"]
            row["env_steps"] = update * batch
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if update in save_at:
                metadata.update(env_steps=row["env_steps"], excursion=excursion_sum / update)
                save_policy(output_dir / "checkpoint.npz", agent, metadata)
                print(f"Saved {row['env_steps']} steps to {output_dir / 'checkpoint.npz'}", flush=True)


@jdc.jit
def rollout_scores(agent, key, episodes: jdc.Static[int], horizon: jdc.Static[int]):
    """Include the terminal reward and ignore all frames following episode end."""
    env = agent.env
    reset_key, action_key = jax.random.split(key)
    state = jax.vmap(env.reset)(jax.random.split(reset_key, episodes))
    # Both methods emit a velocity field. Score both against the same velocity target.
    velocity_agent = jdc.replace(agent, config=jdc.replace(agent.config, output_mode="u"))
    def step(carry, _):
        state, key, active = carry
        key, sample_key, time_key = jax.random.split(key, 3)
        action, info = agent.sample_action(state.obs, sample_key, deterministic=True)
        obs = state.obs
        if agent.config.normalize_observations:
            obs = (obs - agent.obs_stats.mean) / agent.obs_stats.std
        times = jax.random.uniform(time_key, info.loss_t.shape)
        cfm = velocity_agent._compute_cfm_loss(obs, action, eps=info.loss_eps, t=times).mean(axis=-1)
        state = jax.vmap(env.step)(state, jnp.tanh(action))
        scores = (jnp.where(active, state.reward, 0), active, jnp.where(active, cfm, 0))
        return (state, key, active & ~state.done.astype(bool)), scores
    _, (rewards, active, cfm) = jax.lax.scan(
        step, (state, action_key, jnp.ones(episodes, dtype=bool)), None, length=horizon)
    return rewards.sum(axis=0).mean(), cfm.sum() / active.sum()


def evaluate(checkpoint, output, flow_steps=10, episodes=128, seed=0, episode_length=None):
    if flow_steps < 1 or episodes < 1 or (episode_length is not None and episode_length < 1):
        raise ValueError("Use positive step counts, episodes, and horizon")
    with np.load(checkpoint, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
        cfg_type, _ = METHODS[meta["method"]]
        agent = initialize(meta["method"], meta["env_name"], cfg_type(**meta["config"]), meta["train_seed"])
        agent = restore_policy(data, agent)
    horizon = episode_length or meta["config"]["episode_length"]
    scores = dict(method=meta["method"], env_name=meta["env_name"], train_seed=meta["train_seed"],
                  env_steps=meta["env_steps"], budget=meta["config"]["num_timesteps"],
                  train_config=meta["config"], episodes=episodes, eval_seed=seed, horizon=horizon,
                  excursion=meta["excursion"])
    for steps in dict.fromkeys((flow_steps, 1)):
        policy = jdc.replace(agent, config=jdc.replace(agent.config, flow_steps=steps))
        ret, mse = rollout_scores(policy, jax.random.key(seed), episodes, horizon)
        scores[f"return_n{steps}"] = float(ret)
        scores[f"cfm_mse_n{steps}"] = float(mse)
    payload = json.dumps(scores, indent=2, allow_nan=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(payload + "\n")
    print(payload)
