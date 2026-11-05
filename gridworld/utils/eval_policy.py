import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_episodes(policy, env, episodes, max_steps, seed=0, render=False):
    """Evaluate bounded episodes with repeatable resets and action noise."""
    returns, lengths, successes = [], [], []
    with torch.no_grad():
        for episode in range(episodes):
            torch.manual_seed(seed + episode)
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            for step in range(max_steps):
                if render:
                    env.render()
                action = policy(to_device(obs)).detach().cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            successes.append(info["success"])
            returns.append(total)
            lengths.append(step + 1)
    return {"episodes": episodes, "reward_mean": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "reward_std": float(np.std(returns)), "returns": returns, "lengths": lengths}

def to_device(t):
    return torch.tensor(t, dtype=torch.float).to(device)

