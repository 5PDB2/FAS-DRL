"""
Step 3: Overfit PPO on typical channels (one per rho) to test interference mitigation.

Outputs experiments/data/drl_results.pkl with PPO rewards per rho.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.envs import FAS_PortSelection_Env  # noqa: E402
from core.agent import ISAC_Agent  # noqa: E402
from docs.basic_funcs import K, N, N_S  # noqa: E402


class FAS_FixedChannel_Env(FAS_PortSelection_Env):
    """
    Environment that uses a fixed channel for all episodes/steps.
    Observation includes [Re, Im, Power] via _build_observation.
    Reward: raw sum rate (no per-step normalization).
    """

    def __init__(self, fixed_channel, max_steps=50):
        super().__init__(max_steps=max_steps)
        self.fixed_channel = fixed_channel
        self.current_channel = fixed_channel

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_channel = self.fixed_channel
        self.current_obs = self._build_observation(self.current_channel)
        self.current_step = 0
        return self.current_obs, {}

    def step(self, action):
        # Force fixed channel
        self.current_channel = self.fixed_channel
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
        reward = sum_rate  # no normalization

        terminated = (self.current_step >= self.max_steps)
        truncated = False

        info = {
            "sinr": sinr_list,
            "sum_rate": sum_rate,
            "selected_ports": self.selected_ports.tolist(),
        }

        self.current_obs = self._build_observation(self.current_channel)
        return self.current_obs, reward, terminated, truncated, info


def evaluate_agent(env, agent, episodes=10):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def main():
    baseline_path = ROOT / "experiments" / "data" / "baseline_results.pkl"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results not found at {baseline_path}")

    with open(baseline_path, "rb") as f:
        baselines = pickle.load(f)

    drl_scores = {}

    for rho, data in baselines.items():
        H_typical = data["typical_channel"]
        env = FAS_FixedChannel_Env(fixed_channel=H_typical, max_steps=1)

        agent = ISAC_Agent(
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,  # disable exploration for static env
            verbose=0,
        )

        total_timesteps = 30000
        print(f"Training PPO on rho={rho:.2f} for {total_timesteps} steps...")
        agent.train(total_timesteps=total_timesteps)

        mean_reward = evaluate_agent(env, agent, episodes=10)
        drl_scores[rho] = mean_reward
        print(f"rho={rho:.2f} | PPO reward={mean_reward:.3f} | "
              f"Greedy={data['typical_greedy']:.3f} | Upper={data['typical_upper']:.3f}")

    out_dir = ROOT / "experiments" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drl_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(drl_scores, f)
    print(f"Saved DRL scores to {out_path}")


if __name__ == "__main__":
    main()
