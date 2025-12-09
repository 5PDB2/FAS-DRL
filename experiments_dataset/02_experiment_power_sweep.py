"""
Experiment 2: Power sweep (P_max) impact on average sum rate across a 100-channel pool at rho=0.4.

Loads channels_dataset.pkl (regenerates rho=0.4 channels on-the-fly if missing),
computes baselines (Random, Greedy, Upper proxy) and trains PPO per power level.
Saves results to experiments_dataset/data/power_sweep_results.pkl and
plots to experiments_dataset/results/02_power_impact.png.
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
from docs.basic_funcs import N, K, NX, NY, LAMBDA, NOISE_POWER  # noqa: E402


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


def load_or_generate_channels():
    """
    Load channels_dataset.pkl; if missing, regenerate 100 channels for rho=0.4.
    Returns dict with at least rho=0.4 key.
    """
    data_path = ROOT / "experiments_dataset" / "data" / "channels_dataset.pkl"
    if data_path.exists():
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        if 0.4 in dataset:
            return dataset
        # Else fallthrough to regenerate rho=0.4

    # Regenerate rho=0.4 only
    print("channels_dataset.pkl missing or rho=0.4 not present; regenerating rho=0.4 channels...")
    side = int(np.sqrt(N))
    spacing = 0.5 * LAMBDA
    wx = (side - 1) * spacing
    wy = (side - 1) * spacing
    channels = []
    for _ in range(100):
        H = generate_channel(
            K=K,
            N=N,
            Nx=NX,
            Ny=NY,
            Wx=wx,
            Wy=wy,
            wavelength=LAMBDA,
            user_correlation_factor=0.4,
        )
        channels.append(H)
    dataset = {0.4: channels}

    out_dir = data_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved regenerated dataset to {data_path}")
    return dataset


# -----------------------------------------------------------------------------#
# Environment for power sweep
# -----------------------------------------------------------------------------#
class FAS_PowerEnv(FAS_PortSelection_Env):
    """
    Samples a channel from a pool each reset; uses dynamic P_MAX from p_max_dbm.
    Observation includes [Re, Im, Power]; reward is raw sum rate.
    """

    def __init__(self, channel_list, p_max_dbm, max_steps=50):
        super().__init__(max_steps=max_steps)
        self.channel_list = channel_list
        self.P_MAX = dbm_to_watts(p_max_dbm)  # override transmit power
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
        P = self._normalize_power(P_sub)  # uses dynamic self.P_MAX
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


def eval_agent_on_pool(agent, channels, p_max_dbm, max_steps=50):
    """
    Evaluate agent deterministically on each channel in the pool (one episode per channel).
    Returns average episode reward.
    """
    env = FAS_PowerEnv(channel_list=channels, p_max_dbm=p_max_dbm, max_steps=max_steps)
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
    dataset = load_or_generate_channels()
    if 0.4 not in dataset:
        raise ValueError("Dataset does not contain rho=0.4 channels.")
    channels = dataset[0.4]

    power_dbm_list = [10, 15, 20, 25, 30, 35, 40]

    results = {}
    random_curve = []
    greedy_curve = []
    upper_curve = []
    drl_curve = []

    for p_dbm in power_dbm_list:
        P_max = dbm_to_watts(p_dbm)

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
        env = FAS_PowerEnv(channel_list=channels, p_max_dbm=p_dbm, max_steps=1)
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
        print(f"Training PPO for p_max={p_dbm} dBm over channel pool (rho=0.4)...")
        agent.train(total_timesteps=total_timesteps)

        mean_drl = eval_agent_on_pool(agent, channels, p_max_dbm=p_dbm, max_steps=1)
        drl_curve.append(mean_drl)

        results[p_dbm] = {
            "mean_random": mean_random,
            "mean_greedy": mean_greedy,
            "mean_upper": mean_upper,
            "mean_drl": mean_drl,
        }
        print(f"p_max={p_dbm} dBm | Random={mean_random:.3f}, Greedy={mean_greedy:.3f}, "
              f"Upper={mean_upper:.3f}, DRL={mean_drl:.3f}")

    # Save results
    data_dir = ROOT / "experiments_dataset" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "power_sweep_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved power sweep results to {out_path}")

    # Plot
    res_dir = ROOT / "experiments_dataset" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(power_dbm_list, random_curve, 'g--o', label='Random')
    plt.plot(power_dbm_list, greedy_curve, 'r-o', label='Greedy')
    plt.plot(power_dbm_list, upper_curve, 'k--o', label='Upper Bound (proxy)')
    plt.plot(power_dbm_list, drl_curve, 'b-*', label='DRL')
    plt.xlabel('Transmit Power (dBm)')
    plt.ylabel('Average Sum Rate (bps/Hz)')
    plt.title('Impact of Transmit Power on Sum Rate (rho=0.4, pool of 100 channels)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = res_dir / "02_power_impact.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    main()
