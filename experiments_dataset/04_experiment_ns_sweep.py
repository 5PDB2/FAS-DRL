"""
Experiment 4: Sweep number of active ports (N_s) and measure average sum rate.

Uses rho=0.4 channel subset (100 channels from channels_dataset.pkl), P_max=40 dBm.
Trains PPO per N_s and compares against Random, Greedy, and Upper Bound (proxy).

Outputs:
    - experiments_dataset/data/ns_sweep_results.pkl
    - experiments_dataset/results/04_ns_impact.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.envs import FAS_PortSelection_Env  # noqa: E402
from core.agent import ISAC_Agent  # noqa: E402
from docs.basic_funcs import N, K, P_MAX_DBM, NOISE_POWER  # noqa: E402


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def dbm_to_watts(dbm):
    return 10 ** (dbm / 10) / 1000


def calculate_sum_rate(H, selected_indices, P_max, noise_power):
    H_sub = H[:, selected_indices]  # (K, n_s)
    H_sub_H = H_sub.conj().T
    G = H_sub @ H_sub_H + noise_power * np.eye(H_sub.shape[0])
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)
    P_sub = H_sub_H @ G_inv  # (n_s, K)
    norm_sq = np.sum(np.abs(P_sub) ** 2)
    if norm_sq > 0:
        P = P_sub * np.sqrt(P_max / norm_sq)
    else:
        P = P_sub

    H_eff = H_sub @ P
    sinr_list = []
    for k in range(H_eff.shape[0]):
        signal = np.abs(H_eff[k, k]) ** 2
        interference = np.sum(np.abs(H_eff[k, :]) ** 2) - signal
        sinr = signal / (interference + noise_power)
        sinr_list.append(sinr)
    return float(np.sum(np.log2(1.0 + np.array(sinr_list))))


def greedy_indices(H, n_s):
    power = np.sum(np.abs(H) ** 2, axis=0)
    return np.argsort(power)[-n_s:]


def random_indices(n_ports, n_s):
    return np.random.choice(n_ports, n_s, replace=False)


def upper_bound_proxy(H, P_max, noise_power, n_s, num_samples=1000):
    best = -np.inf
    candidates = [greedy_indices(H, n_s=n_s)]
    for _ in range(num_samples):
        candidates.append(random_indices(H.shape[1], n_s))
    seen = set()
    for idx in candidates:
        key = tuple(sorted(idx.tolist()))
        if key in seen:
            continue
        seen.add(key)
        rate = calculate_sum_rate(H, np.array(idx), P_max, noise_power)
        if rate > best:
            best = rate
    return best


# -----------------------------------------------------------------------------#
# Environment for Ns sweep
# -----------------------------------------------------------------------------#
class FAS_NsEnv(FAS_PortSelection_Env):
    """
    Samples a channel from pool; allows dynamic N_S (number of active ports).
    Observation includes [Re, Im, Power]; reward is raw sum rate.
    """

    def __init__(self, channel_list, n_s, max_steps=50):
        super().__init__(max_steps=max_steps)
        self.channel_list = channel_list
        self.N_S = n_s  # override number of selected ports
        self.current_channel = None

    def _select_ports(self, action):
        # Override to use self.N_S instead of global constant
        selected_indices = np.argsort(action)[-self.N_S:]
        return np.sort(selected_indices)

    def reset(self, seed=None):
        super().reset(seed=seed)
        idx = np.random.randint(0, len(self.channel_list))
        self.current_channel = self.channel_list[idx]
        self.current_obs = self._build_observation(self.current_channel)
        self.current_step = 0
        return self.current_obs, {}

    def step(self, action):
        self.current_step += 1

        self.selected_ports = self._select_ports(action)
        self.H_sub = self.current_channel[:, self.selected_ports]
        P_sub = self._compute_mmse_precoder(self.H_sub)
        P = self._normalize_power(P_sub)
        H_eff = self.H_sub @ P

        sinr_list = []
        for k in range(self.K):
            signal_power = np.abs(H_eff[k, k]) ** 2
            interference_power = np.sum(np.abs(H_eff[k, :]) ** 2) - signal_power
            sinr_k = signal_power / (interference_power + self.NOISE_POWER)
            sinr_list.append(sinr_k)

        sum_rate = np.sum(np.log2(1.0 + np.array(sinr_list)))
        reward = sum_rate

        terminated = (self.current_step >= self.max_steps)
        truncated = False

        info = {
            "sinr": sinr_list,
            "sum_rate": sum_rate,
            "selected_ports": self.selected_ports.tolist(),
        }

        self.current_obs = self._build_observation(self.current_channel)
        return self.current_obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------------#
# Main experiment
# -----------------------------------------------------------------------------#
def main():
    data_path = ROOT / "experiments_dataset" / "data" / "channels_dataset.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Channel dataset not found at {data_path}")
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    if 0.4 not in dataset:
        raise ValueError("Dataset does not contain rho=0.4 channels.")
    channels = dataset[0.4]

    P_max = dbm_to_watts(P_MAX_DBM)  # 40 dBm
    ns_list = [3, 4, 5, 6, 7, 8, 9]

    random_curve = []
    greedy_curve = []
    upper_curve = []
    drl_curve = []
    results = {}

    for n_s in ns_list:
        if n_s < K:
            raise ValueError(f"n_s must be >= K. Got n_s={n_s}, K={K}")

        # Baselines
        rand_vals = []
        greedy_vals = []
        upper_vals = []
        for H in channels:
            rand_samples = []
            for _ in range(10):
                idx = random_indices(H.shape[1], n_s)
                rand_samples.append(calculate_sum_rate(H, idx, P_max, NOISE_POWER))
            rand_vals.append(float(np.mean(rand_samples)))

            g_idx = greedy_indices(H, n_s=n_s)
            g_val = calculate_sum_rate(H, g_idx, P_max, NOISE_POWER)
            greedy_vals.append(g_val)

            u_val = upper_bound_proxy(H, P_max, NOISE_POWER, n_s=n_s, num_samples=1000)
            upper_vals.append(u_val)

        mean_random = float(np.mean(rand_vals))
        mean_greedy = float(np.mean(greedy_vals))
        mean_upper = float(np.mean(upper_vals))

        random_curve.append(mean_random)
        greedy_curve.append(mean_greedy)
        upper_curve.append(mean_upper)

        # DRL training
        env = FAS_NsEnv(channel_list=channels, n_s=n_s, max_steps=1)
        agent = ISAC_Agent(
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
        )

        total_timesteps = 102400
        print(f"Training PPO for n_s={n_s} over channel pool (rho=0.4)...")
        agent.train(total_timesteps=total_timesteps)

        # Evaluate on pool
        mean_drl = evaluate_agent_on_pool(agent, channels, n_s=n_s, max_steps=1)
        drl_curve.append(mean_drl)

        results[n_s] = {
            "mean_random": mean_random,
            "mean_greedy": mean_greedy,
            "mean_upper": mean_upper,
            "mean_drl": mean_drl,
        }
        print(f"n_s={n_s} | Random={mean_random:.3f}, Greedy={mean_greedy:.3f}, "
              f"Upper={mean_upper:.3f}, DRL={mean_drl:.3f}")

    # Save results
    data_dir = ROOT / "experiments_dataset" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "ns_sweep_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved Ns sweep results to {out_path}")

    # Plot
    res_dir = ROOT / "experiments_dataset" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(ns_list, random_curve, 'g--o', label='Random')
    plt.plot(ns_list, greedy_curve, 'r-o', label='Greedy')
    plt.plot(ns_list, upper_curve, 'k--o', label='Upper Bound (proxy)')
    plt.plot(ns_list, drl_curve, 'b-*', label='DRL')
    plt.xlabel('Number of Active Ports (N_s)')
    plt.ylabel('Average Sum Rate (bps/Hz)')
    plt.title('Impact of Active Ports on Sum Rate (rho=0.4, P_max=40 dBm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = res_dir / "04_ns_impact.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")


def evaluate_agent_on_pool(agent, channels, n_s, max_steps=50):
    """
    Evaluate agent deterministically on each channel in the pool (one episode per channel).
    Returns average episode reward.
    """
    env = FAS_NsEnv(channel_list=channels, n_s=n_s, max_steps=max_steps)
    rewards = []
    for H in channels:
        env.current_channel = H
        env.current_step = 0
        env.current_obs = env._build_observation(env.current_channel)
        done = False
        ep_reward = 0.0
        obs = env.current_obs
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return float(np.mean(rewards))


if __name__ == "__main__":
    main()
