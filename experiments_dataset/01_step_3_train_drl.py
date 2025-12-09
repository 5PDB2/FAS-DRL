"""
Step 3 (dataset variant): Train PPO on a pool of 100 channels per rho to test generalization.

Loads channels_dataset.pkl, trains a PPO agent with mild exploration (ent_coef=0.01)
over a channel pool, and evaluates average reward across all channels.
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


class FAS_ChannelPool_Env(FAS_PortSelection_Env):
    """
    Environment that samples a channel from a provided pool at each reset.
    Observation includes [Re, Im, Power] via base _build_observation.
    Reward: raw sum rate (no normalization).
    """

    def __init__(self, channel_list, max_steps=50):
        super().__init__(max_steps=max_steps)
        self.channel_list = channel_list
        self.current_channel = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Sample a channel uniformly from the pool
        idx = np.random.randint(0, len(self.channel_list))
        self.current_channel = self.channel_list[idx]
        self.current_obs = self._build_observation(self.current_channel)
        self.current_step = 0
        return self.current_obs, {}

    def step(self, action):
        # Use the sampled channel for all steps within the episode
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
        reward = sum_rate  # raw (no per-step normalization)

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


def load_dataset():
    path = ROOT / "experiments_dataset" / "data" / "channels_dataset.pkl"
    if not path.exists():
        raise FileNotFoundError(f"channels_dataset.pkl not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    dataset = load_dataset()
    drl_scores = {}
    out_dir = ROOT / "experiments_dataset" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rho, channels in dataset.items():
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
            ent_coef=0.01,  # mild exploration across channel pool
            verbose=0,
        )

        total_timesteps = 204800
        print(f"Training PPO on rho={rho:.2f} for {total_timesteps} steps over channel pool...")
        agent.train(total_timesteps=total_timesteps)

        mean_reward = eval_agent_on_pool(agent, channels, max_steps=1)
        drl_scores[rho] = mean_reward
        print(f"rho={rho:.2f} | PPO mean reward over pool={mean_reward:.3f}")

    out_path = out_dir / "drl_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(drl_scores, f)
    print(f"Saved DRL pool scores to {out_path}")


if __name__ == "__main__":
    main()
