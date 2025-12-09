"""
Main Training Loop for FAS-DRL Agent.

train_agent
===========
Implements the complete training pipeline for the PPO agent on the FAS-ISAC environment.

Training Pipeline:
    1. Initialize environment and agent
    2. Collect trajectories from environment
    3. Compute advantages (GAE)
    4. Update actor and critic networks
    5. Log metrics and save checkpoints
    6. Repeat until convergence

Key Functions:
    - main(): Main entry point for training
    - run_episode(env, agent): Run a single evaluation episode
    - print_episode_summary(episode_num, episode_reward, sinr_values, comm_gain, sensing_gain)

Logging and Checkpoints:
    - Training curves: rewards, policy loss, value loss
    - Model checkpoints: saved at regular intervals
    - Tensorboard logs: for real-time monitoring

Configuration:
    - num_episodes: Total training episodes
    - steps_per_epoch: Trajectories collected before update
    - learning_rate: Policy and value function learning rates
    - discount factor (gamma), GAE lambda
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.envs import FAS_PortSelection_Env
from core.agent import ISAC_Agent
from docs.basic_funcs import print_system_config


def plot_training_curves(log_data, save_path=None):
    """
    Plot training curves for Loss, PG_Loss, V_Loss, and Episode Reward Mean.
    
    Args:
        log_data (dict): Dictionary containing training metrics with keys:
                        'timesteps', 'ep_rew_mean', 'loss', 'pg_loss', 'v_loss'
        save_path (str, optional): Path to save the figure. If None, saves as 'training_curves.png'
    """
    timesteps = log_data['timesteps']
    ep_rew_mean = log_data['ep_rew_mean']
    loss = log_data['loss']
    pg_loss = log_data['pg_loss']
    v_loss = log_data['v_loss']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves - FAS Port Selection with MMSE Beamforming', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Reward Mean
    ax = axes[0, 0]
    ax.plot(timesteps, ep_rew_mean, linewidth=2, color='#2E86AB', label='Episode Reward Mean')
    ax.fill_between(timesteps, ep_rew_mean, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Timesteps', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Episode Reward Mean vs Timesteps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 2: Total Loss
    ax = axes[0, 1]
    # Filter out NaN values
    valid_idx = ~np.isnan(loss)
    if np.any(valid_idx):
        ax.plot(timesteps[valid_idx], loss[valid_idx], linewidth=2, color='#A23B72', label='Total Loss')
        ax.fill_between(timesteps[valid_idx], loss[valid_idx], alpha=0.3, color='#A23B72')
    ax.set_xlabel('Timesteps', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Total Loss vs Timesteps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 3: Policy Gradient Loss
    ax = axes[1, 0]
    # Filter out NaN values
    valid_idx = ~np.isnan(pg_loss)
    if np.any(valid_idx):
        ax.plot(timesteps[valid_idx], pg_loss[valid_idx], linewidth=2, color='#F18F01', label='PG Loss')
        ax.fill_between(timesteps[valid_idx], pg_loss[valid_idx], alpha=0.3, color='#F18F01')
    ax.set_xlabel('Timesteps', fontsize=11)
    ax.set_ylabel('Policy Gradient Loss', fontsize=11)
    ax.set_title('Policy Gradient Loss vs Timesteps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Plot 4: Value Function Loss
    ax = axes[1, 1]
    # Filter out NaN values
    valid_idx = ~np.isnan(v_loss)
    if np.any(valid_idx):
        ax.plot(timesteps[valid_idx], v_loss[valid_idx], linewidth=2, color='#C73E1D', label='Value Loss')
        ax.fill_between(timesteps[valid_idx], v_loss[valid_idx], alpha=0.3, color='#C73E1D')
    ax.set_xlabel('Timesteps', fontsize=11)
    ax.set_ylabel('Value Function Loss', fontsize=11)
    ax.set_title('Value Function Loss vs Timesteps', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()

def run_episode(env, agent, deterministic=True, max_steps=50):
    """
    Run a single episode with the trained agent.
    
    Args:
        env: The FAS_PortSelection_Env environment
        agent: The ISAC_Agent to evaluate
        deterministic (bool): If True, use mean of policy. If False, sample actions.
        max_steps (int): Maximum steps per episode
    
    Returns:
        dict: Episode statistics including total reward and metrics
    """
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_sinr = []
    episode_sum_rate = []
    episode_sensing_gain = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=deterministic)
        
        # Execute action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_sinr.extend(info['sinr'])
        episode_sum_rate.append(info['sum_rate'])
        episode_sensing_gain.append(info['sensing_gain'])
        
        done = terminated or truncated
        step += 1
    
    # Compute statistics
    avg_sinr = np.mean(episode_sinr) if episode_sinr else 0.0
    avg_sum_rate = np.mean(episode_sum_rate) if episode_sum_rate else 0.0
    avg_sensing = np.mean(episode_sensing_gain) if episode_sensing_gain else 0.0
    
    return {
        'total_reward': episode_reward,
        'avg_sinr': avg_sinr,
        'avg_sum_rate': avg_sum_rate,
        'avg_sensing_gain': avg_sensing,
        'steps': step
    }


def print_episode_summary(episode_num, episode_reward, sinr_list, sum_rate, sensing_gain):
    """
    Print a summary of episode statistics.
    
    Args:
        episode_num (int): Episode number
        episode_reward (float): Total reward for the episode
        sinr_list (list): SINR values for each user
        sum_rate (float): Sum rate (communication reward)
        sensing_gain (float): Sensing beampattern gain
    """
    avg_sinr = np.mean(sinr_list) if sinr_list else 0.0
    print(f"\nEpisode {episode_num:3d}")
    print(f"  Total Reward:       {episode_reward:8.4f}")
    print(f"  Sum Rate:           {sum_rate:8.4f} bits/s")
    print(f"  Sensing Gain:       {sensing_gain:8.4f}")
    print(f"  Avg SINR (linear):  {avg_sinr:8.4f} ({10*np.log10(avg_sinr):6.2f} dB)")


def load_pretrained_weights(agent, path):
    """
    Load pre-trained BC actor weights into the PPO policy network.

    Mapping:
        BCActor.net[0] -> policy.mlp_extractor.policy_net[0]
        BCActor.net[2] -> policy.mlp_extractor.policy_net[2]
        BCActor.net[4] -> policy.action_net
    """
    if not os.path.exists(path):
        print(f"[BC Init] Pretrained weights not found at {path}, skipping warm start.")
        return

    try:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
    except Exception as err:
        print(f"[BC Init] Failed to load {path}: {err}")
        return

    policy_net = agent.model.policy.mlp_extractor.policy_net
    action_net = agent.model.policy.action_net

    def copy_linear(src_w, src_b, dst_linear, name):
        if dst_linear.weight.shape != src_w.shape or dst_linear.bias.shape != src_b.shape:
            print(f"[BC Init] Shape mismatch for {name}: src {src_w.shape}, dst {dst_linear.weight.shape}")
            return False
        with torch.no_grad():
            dst_linear.weight.copy_(src_w)
            dst_linear.bias.copy_(src_b)
        return True

    ok = True
    ok &= copy_linear(state_dict["net.0.weight"], state_dict["net.0.bias"], policy_net[0], "layer0")
    ok &= copy_linear(state_dict["net.2.weight"], state_dict["net.2.bias"], policy_net[2], "layer1")
    ok &= copy_linear(state_dict["net.4.weight"], state_dict["net.4.bias"], action_net, "action_net")

    if ok:
        print(f"[BC Init] Loaded pretrained weights from {path} into PPO policy.")
    else:
        print(f"[BC Init] Completed with mismatches; verify architecture.")


def main():
    """
    Main training function.
    
    Workflow:
        1. Print system configuration
        2. Initialize environment
        3. Initialize PPO agent
        4. Train for specified timesteps
        5. Save trained model
        6. Run evaluation episodes
        7. Print final statistics
    """
    print("\n" + "="*70)
    print("FAS-ISAC DRL Training Script")
    print("="*70)
    
    # Print system configuration
    print_system_config()
    
    # ========================================================================
    # Initialize Environment
    # ========================================================================
    # Initialize Environment
    # ========================================================================
    print("\nInitializing Environment...")
    env = FAS_PortSelection_Env(max_steps=1)
    print(f"Environment initialized successfully")
    print(f"  Action Space:      {env.action_space}")
    print(f"  Observation Space: {env.observation_space}")
    
    # ========================================================================
    # Initialize Agent
    # ========================================================================
    print("\nInitializing Agent...")
    agent = ISAC_Agent(
        env=env,
        learning_rate=1e-4,  # Reduced from 3e-4 for numerical stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration to escape local optima
        verbose=1
    )
    agent.get_model_info()
    # Warm start policy with BC-pretrained weights if available
    bc_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'bc_actor.pth')
    load_pretrained_weights(agent, bc_path)
    
    # ========================================================================
    # Training
    # ========================================================================
    total_timesteps = 40960  # Increased for better convergence
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    
    # Train and get metrics
    _, training_metrics = agent.train(total_timesteps=total_timesteps)
    
    # ========================================================================
    # Plot Training Curves
    # ========================================================================
    print("\nGenerating training curves...")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    curves_path = os.path.join(results_dir, 'training_curves.png')
    plot_training_curves(training_metrics, save_path=curves_path)
    
    # ========================================================================
    # Save Model
    # ========================================================================
    print("\nSaving Model...")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    model_path = os.path.join(results_dir, 'ppo_fas_isac')
    agent.save(model_path)
    
    # ========================================================================
    # Evaluation: Run Test Episodes
    # ========================================================================
    print("\n" + "="*70)
    print("Evaluation Phase: Running Test Episodes")
    print("="*70)
    
    num_eval_episodes = 5
    eval_rewards = []
    eval_sinrs = []
    eval_sum_rates = []
    eval_sensing_gains = []
    
    for episode in range(num_eval_episodes):
        stats = run_episode(env, agent, deterministic=True, max_steps=50)
        
        eval_rewards.append(stats['total_reward'])
        eval_sinrs.append(stats['avg_sinr'])
        eval_sum_rates.append(stats['avg_sum_rate'])
        eval_sensing_gains.append(stats['avg_sensing_gain'])
        
        print(f"\nTest Episode {episode + 1}/{num_eval_episodes}")
        print(f"  Total Reward:       {stats['total_reward']:8.4f}")
        print(f"  Avg SINR (linear):  {stats['avg_sinr']:8.4f} ({10*np.log10(stats['avg_sinr']):6.2f} dB)")
        print(f"  Sum Rate:           {stats['avg_sum_rate']:8.4f} bits/s")
        print(f"  Sensing Gain:       {stats['avg_sensing_gain']:8.4f}")
        print(f"  Steps:              {stats['steps']:3d}")
    
    # ========================================================================
    # Print Final Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("Final Statistics")
    print("="*70)
    print(f"Test Episodes Run:           {num_eval_episodes}")
    print(f"Average Total Reward:        {np.mean(eval_rewards):.4f} ± {np.std(eval_rewards):.4f}")
    print(f"Average SINR (dB):           {10*np.log10(np.mean(eval_sinrs)):.2f} dB")
    print(f"Average Sum Rate:            {np.mean(eval_sum_rates):.4f} bits/s")
    print(f"Average Sensing Gain:        {np.mean(eval_sensing_gains):.4f}")
    print(f"Model saved to:              {model_path}.zip")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
