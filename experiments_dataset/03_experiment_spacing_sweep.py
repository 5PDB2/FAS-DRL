"""
Experiment 3: Spacing sweep (Δd/λ) impact on average sum rate.

Generates fresh channel pools per spacing ratio (geometry changes), computes baselines,
trains PPO per ratio, and plots results.

Outputs:
    - experiments_dataset/data/spacing_sweep_results.pkl
    - experiments_dataset/results/03_spacing_impact.png
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
from utils.channel import generate_channel  # noqa: E402
from docs.basic_funcs import N, K, NX, NY, LAMBDA, P_MAX_DBM, NOISE_POWER  # noqa: E402


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def dbm_to_watts(dbm):
    return 10 ** (dbm / 10) / 1000


def calculate_sum_rate(H, selected_indices, P_max, noise_power):
    H_sub = H[:, selected_indices]  # (K, N_S)
    H_sub_H = H_sub.conj().T
    G = H_sub @ H_sub_H + noise_power * np.eye(H_sub.shape[0])
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)
    P_sub = H_sub_H @ G_inv  # (N_S, K)
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


def greedy_indices(H, N_S=5):
    power = np.sum(np.abs(H) ** 2, axis=0)
    return np.argsort(power)[-N_S:]


def random_indices(N, N_S):
    return np.random.choice(N, N_S, replace=False)


def upper_bound_proxy(H, P_max, noise_power, N_S=5, num_samples=1000):
    best = -np.inf
    candidates = [greedy_indices(H, N_S=N_S)]
    for _ in range(num_samples):
        candidates.append(random_indices(H.shape[1], N_S))
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
# Environment sampling from channel pool (uses dynamic geometry channels)
# -----------------------------------------------------------------------------#
class FAS_ChannelPool_Env(FAS_PortSelection_Env):
    """
    Samples a channel from a provided pool at each reset.
    Observation includes [Re, Im, Power] via base _build_observation.
    Reward: raw sum rate (no normalization).
    """

    def __init__(self, channel_list, max_steps=50):
        super().__init__(max_steps=max_steps)
        self.channel_list = channel_list
        self.current_channel = None

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
        reward = sum_rate  # raw reward

        terminated = (self.current_step >= self.max_steps)
        truncated = False

        info = {
            "sinr": sinr_list,
            "sum_rate": sum_rate,
            "selected_ports": self.selected_ports.tolist(),
        }

        self.current_obs = self._build_observation(self.current_channel)
        return self.current_obs, reward, terminated, truncated, info


def eval_agent_on_pool(agent, channels, max_steps=50):
    """
    Evaluate agent deterministically on each channel in the pool (one episode per channel).
    Returns average episode reward.
    """
    env = FAS_ChannelPool_Env(channel_list=channels, max_steps=max_steps)
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


# -----------------------------------------------------------------------------#
# Main experiment
# -----------------------------------------------------------------------------#
def main():
    spacing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    P_max = dbm_to_watts(P_MAX_DBM)  # 40 dBm high power
    rho = 0.0  # synthetic correlation off; only geometry drives correlation

    random_curve = []
    greedy_curve = []
    upper_curve = []
    drl_curve = []
    results = {}

    for ratio in spacing_ratios:
        spacing = ratio * LAMBDA
        wx = (NX - 1) * spacing
        wy = (NY - 1) * spacing

        # Generate fresh pool of channels for this geometry
        channels = []
        for _ in range(50):
            H = generate_channel(
                K=K,
                N=N,
                Nx=NX,
                Ny=NY,
                Wx=wx,
                Wy=wy,
                wavelength=LAMBDA,
                user_correlation_factor=rho,
            )
            channels.append(H)

        # Baselines
        rand_vals = []
        greedy_vals = []
        upper_vals = []
        for H in channels:
            rand_samples = []
            for _ in range(10):
                idx = random_indices(N, 5)
                rand_samples.append(calculate_sum_rate(H, idx, P_max, NOISE_POWER))
            rand_vals.append(float(np.mean(rand_samples)))

            g_idx = greedy_indices(H, N_S=5)
            g_val = calculate_sum_rate(H, g_idx, P_max, NOISE_POWER)
            greedy_vals.append(g_val)

            u_val = upper_bound_proxy(H, P_max, NOISE_POWER, N_S=5, num_samples=1000)
            upper_vals.append(u_val)

        mean_random = float(np.mean(rand_vals))
        mean_greedy = float(np.mean(greedy_vals))
        mean_upper = float(np.mean(upper_vals))

        random_curve.append(mean_random)
        greedy_curve.append(mean_greedy)
        upper_curve.append(mean_upper)

        # DRL training
        env = FAS_ChannelPool_Env(channel_list=channels, max_steps=1)
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

        total_timesteps = 204800
        print(f"Training PPO for spacing ratio={ratio:.2f} (spacing={spacing:.3e} m)...")
        agent.train(total_timesteps=total_timesteps)

        mean_drl = eval_agent_on_pool(agent, channels, max_steps=1)
        drl_curve.append(mean_drl)

        results[ratio] = {
            "mean_random": mean_random,
            "mean_greedy": mean_greedy,
            "mean_upper": mean_upper,
            "mean_drl": mean_drl,
        }
        print(f"ratio={ratio:.2f} | Random={mean_random:.3f}, Greedy={mean_greedy:.3f}, "
              f"Upper={mean_upper:.3f}, DRL={mean_drl:.3f}")

    # Save results
    data_dir = ROOT / "experiments_dataset" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "spacing_sweep_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved spacing sweep results to {out_path}")

    # Plot
    res_dir = ROOT / "experiments_dataset" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(spacing_ratios, random_curve, 'g--o', label='Random')
    plt.plot(spacing_ratios, greedy_curve, 'r-o', label='Greedy')
    plt.plot(spacing_ratios, upper_curve, 'k--o', label='Upper Bound (proxy)')
    plt.plot(spacing_ratios, drl_curve, 'b-*', label='DRL')
    plt.xlabel('Antenna Spacing (Δd / λ)')
    plt.ylabel('Average Sum Rate (bps/Hz)')
    plt.title('Impact of Antenna Spacing on Sum Rate (rho=0.0, P_max=40 dBm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = res_dir / "03_spacing_impact.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    main()
