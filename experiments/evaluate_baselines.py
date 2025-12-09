"""
Baseline evaluation for FAS Port Selection.

Compares Sum Rate of three strategies on the same channel realizations:
1) Random selection
2) Greedy (path-gain maximization, PGMax)
3) DRL (trained PPO agent)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.envs import FAS_PortSelection_Env
from stable_baselines3 import PPO


def build_action_from_indices(n_ports, selected_indices, high=1.0, low=-1.0):
    """
    Build an action vector where selected indices have higher values so that
    env.step selects them via argsort.
    """
    action = np.full(n_ports, low, dtype=np.float32)
    action[selected_indices] = high
    return action


def greedy_pgmax_action(channel, n_select):
    """
    Greedy port selection by maximizing path gain.

    Args:
        channel (np.ndarray): Channel matrix H of shape (K, N)
        n_select (int): Number of ports to select

    Returns:
        np.ndarray: Action vector for env.step
    """
    # Power per port: sum over users of |h_{k,n}|^2
    port_power = np.sum(np.abs(channel) ** 2, axis=0)
    top_indices = np.argsort(port_power)[-n_select:]

    # Rank the selected ports so the strongest gets the highest action value
    ranked = top_indices[np.argsort(port_power[top_indices])]
    action = np.full(channel.shape[1], -1.0, dtype=np.float32)
    ranked_values = np.linspace(0.5, 1.0, n_select, dtype=np.float32)
    for idx, val in zip(ranked, ranked_values):
        action[idx] = val
    return action


def restore_env_state(env, channel, user_distances):
    """
    Restore the environment to a given channel (and path-loss distances)
    so that each baseline sees identical physics.
    """
    env.current_channel = channel.copy()
    env.user_distances = user_distances.copy() if user_distances is not None else None
    env.current_step = 0
    env.selected_ports = None
    env.H_sub = None
    # Rebuild observation with power feature and scaling
    env.current_obs = env._build_observation(env.current_channel)


def evaluate_baselines(num_episodes=200, model_path="results/ppo_fas_isac.zip", results_dir="results"):
    env = FAS_PortSelection_Env(max_steps=1)

    # Try to load PPO model
    ppo_model = None
    if os.path.exists(model_path):
        try:
            ppo_model = PPO.load(model_path)
            print(f"Loaded PPO model from {model_path}")
        except Exception as err:
            print(f"Could not load PPO model ({err}); skipping DRL baseline.")
    else:
        print(f"PPO model not found at {model_path}; skipping DRL baseline.")

    random_rewards = []
    greedy_rewards = []
    drl_rewards = []

    for episode in range(num_episodes):
        # Single channel realization for all baselines
        obs, _ = env.reset()
        saved_channel = env.current_channel.copy()
        saved_distances = env.user_distances.copy() if env.user_distances is not None else None

        # Random baseline
        restore_env_state(env, saved_channel, saved_distances)
        rand_indices = np.random.choice(env.N, env.N_S, replace=False)
        random_action = build_action_from_indices(env.N, rand_indices, high=1.0)
        _, reward, _, _, _ = env.step(random_action)
        random_rewards.append(reward)

        # Greedy (PGMax) baseline
        restore_env_state(env, saved_channel, saved_distances)
        greedy_action = greedy_pgmax_action(saved_channel, env.N_S)
        _, reward, _, _, _ = env.step(greedy_action)
        greedy_rewards.append(reward)

        # DRL (PPO) baseline
        if ppo_model is not None:
            restore_env_state(env, saved_channel, saved_distances)
            action, _ = ppo_model.predict(env.current_obs, deterministic=True)
            _, reward, _, _, _ = env.step(action)
            drl_rewards.append(reward)

    # Aggregate stats
    methods = ["Random", "Greedy (PGMax)"]
    means = [np.mean(random_rewards), np.mean(greedy_rewards)]
    stds = [np.std(random_rewards), np.std(greedy_rewards)]

    if ppo_model is not None and drl_rewards:
        methods.append("DRL (PPO)")
        means.append(np.mean(drl_rewards))
        stds.append(np.std(drl_rewards))

    # Plot
    os.makedirs(results_dir, exist_ok=True)
    x = np.arange(len(methods))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=6, color=["#6CA6CD", "#F4A460", "#90EE90"][: len(methods)])
    plt.xticks(x, methods)
    plt.ylabel("Average Sum Rate (bps/Hz)")
    plt.title("Baseline Comparison for FAS Port Selection")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "baseline_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    # Print summary
    print("\n=== Baseline Results ===")
    for m, mean, std in zip(methods, means, stds):
        print(f"{m:15s}: {mean:.3f} ± {std:.3f} bps/Hz (n={num_episodes})")


if __name__ == "__main__":
    evaluate_baselines()
