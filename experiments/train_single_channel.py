"""
Train PPO on a single fixed channel to sanity-check learnability.
If PPO cannot beat greedy on a static channel, the architecture is likely flawed.
"""

import os
import sys
import numpy as np
from pathlib import Path

import torch
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.envs import FAS_PortSelection_Env  # noqa: E402
from core.agent import ISAC_Agent  # noqa: E402
from utils.channel import generate_channel  # noqa: E402
from docs.basic_funcs import N, K, NX, NY, WX, WY, LAMBDA  # noqa: E402


class FAS_SingleChannel_Env(FAS_PortSelection_Env):
    """
    Environment with a single fixed channel realization.
    Uses the same observation engineering [Re, Im, Power] and scaling as the base env.
    """

    def __init__(self, max_steps=50):
        super().__init__(max_steps=max_steps)

        # Generate one fixed channel and store it
        self.fixed_channel = generate_channel(
            K=K,
            N=N,
            Nx=NX,
            Ny=NY,
            Wx=WX,
            Wy=WY,
            wavelength=LAMBDA,
            path_loss_distances=None,
        )
        self.current_channel = self.fixed_channel

        # Compute greedy (PGMax) target reward on this channel
        self.greedy_target_reward = self._compute_greedy_reward(self.fixed_channel)
        print(f"Greedy Target Reward: {self.greedy_target_reward:.4f}")

    def _compute_greedy_reward(self, channel):
        """
        Compute greedy reward for a given channel by selecting top N_S ports.
        """
        port_power = np.sum(np.abs(channel) ** 2, axis=0)
        top_indices = np.argsort(port_power)[-self.N_S:]
        # Build an action that selects these ports
        action = np.full(self.N, -1.0, dtype=np.float32)
        action[top_indices] = 1.0

        # Use base step logic but with fixed channel
        self.selected_ports = self._select_ports(action)
        self.H_sub = channel[:, self.selected_ports]
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
        return sum_rate   # normalized per step to match env reward

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_channel = self.fixed_channel
        self.current_obs = self._build_observation(self.current_channel)
        self.current_step = 0
        return self.current_obs, {}

    def step(self, action):
        """
        Override to ensure fixed channel is always used.
        """
        self.current_channel = self.fixed_channel
        return super().step(action)


def main():
    print("\n" + "=" * 70)
    print("PPO Training on Single Fixed Channel")
    print("=" * 70)

    # Initialize environment
    env = FAS_SingleChannel_Env(max_steps=50)
    print(f"Greedy Target Reward: {env.greedy_target_reward:.4f}")

    # PPO agent with low entropy (no exploration needed on static env)
    agent = ISAC_Agent(
        env=env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,  # disable entropy to focus on exploitation
        verbose=1,
    )

    total_timesteps = 50000
    steps = env.max_steps
    print(f"Training for {total_timesteps} timesteps on a single channel...")
    _, metrics = agent.train(total_timesteps=total_timesteps)

    # Evaluate deterministic policy over 10 episodes
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        # 将总奖励除以步数，得到平均单步奖励
        rewards.append(ep_reward/50)

    mean_reward = float(np.mean(rewards))
    print(f"Mean PPO reward over 10 eval episodes: {mean_reward:.4f}")
    print(f"Greedy Target Reward: {env.greedy_target_reward:.4f}")
    if mean_reward > env.greedy_target_reward:
        print("PPO exceeded greedy on the fixed channel.")
    else:
        print("PPO did NOT exceed greedy on the fixed channel. Architecture may be insufficient.")


if __name__ == "__main__":
    main()
