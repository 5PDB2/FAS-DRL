"""
Deep Reinforcement Learning Agent: Proximal Policy Optimization (PPO).

PPO_Agent
=========
Implements the PPO algorithm for training Fluid Antenna Systems controllers.

Responsibilities:
- Policy network: Maps state to antenna position actions
- Value network: Estimates state value for advantage computation
- Training loop: Collects experience, computes GAE, updates networks
- Action sampling: Continuous control with Gaussian policy

Key Methods:
    - act(state): Sample action from current policy
    - train(trajectories): Update networks using PPO loss
    - save_checkpoint(path): Save model weights
    - load_checkpoint(path): Load model weights

Hyperparameters to configure:
    - Learning rate (actor & critic)
    - Entropy coefficient
    - Clip ratio
    - GAE lambda and discount factor
"""

import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class TqdmCallback(BaseCallback):
    """
    Custom callback to integrate tqdm progress bar with PPO training.
    Displays real-time training metrics on the progress bar.
    """
    def __init__(self, total_timesteps, pbar=None):
        super(TqdmCallback, self).__init__()
        self.total_timesteps = total_timesteps
        self.pbar = pbar
        self.last_num_timesteps = 0

    def _on_step(self) -> bool:
        # Update progress bar
        if self.pbar is not None:
            # Calculate steps since last update
            current_timesteps = self.model.num_timesteps
            steps_since_last = current_timesteps - self.last_num_timesteps
            
            if steps_since_last > 0:
                self.pbar.update(steps_since_last)
                self.last_num_timesteps = current_timesteps
            
            # Extract and display metrics from logger
            postfix_dict = {}
            
            # Try to get average episode reward
            if self.logger.name_to_value.get('rollout/ep_rew_mean') is not None:
                avg_rew = self.logger.name_to_value['rollout/ep_rew_mean']
                postfix_dict['Avg_Rew'] = f"{avg_rew:.2f}"
            
            # Try to get policy loss
            if self.logger.name_to_value.get('train/loss') is not None:
                loss = self.logger.name_to_value['train/loss']
                postfix_dict['Loss'] = f"{loss:.2f}"
            
            # Try to get policy gradient loss
            if self.logger.name_to_value.get('train/policy_gradient_loss') is not None:
                pg_loss = self.logger.name_to_value['train/policy_gradient_loss']
                postfix_dict['PG_Loss'] = f"{pg_loss:.4f}"
            
            # Try to get value loss
            if self.logger.name_to_value.get('train/value_loss') is not None:
                v_loss = self.logger.name_to_value['train/value_loss']
                postfix_dict['V_Loss'] = f"{v_loss:.2f}"
            
            # Try to get FPS
            if self.logger.name_to_value.get('time/fps') is not None:
                fps = self.logger.name_to_value['time/fps']
                postfix_dict['fps'] = f"{fps:.0f}"
            
            # Update progress bar description with metrics
            if postfix_dict:
                self.pbar.set_postfix(postfix_dict)
        
        return True


class MetricLoggerCallback(BaseCallback):
    """
    Custom callback to log training metrics for visualization.
    
    This callback extracts and stores training metrics from the logger
    so they can be plotted and analyzed after training.
    """
    def __init__(self):
        super(MetricLoggerCallback, self).__init__()
        # Lists to store metrics
        self.timesteps = []
        self.ep_rew_mean = []
        self.loss = []
        self.pg_loss = []
        self.v_loss = []
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Extract metrics from logger and append to lists.
        """
        # No-op: metrics are collected at rollout boundaries to align with SB3 logging
        return True

    def _log_metrics(self):
        """
        Collect metrics from the logger/episode buffer and store them once per timestep count.
        Called at rollout boundaries (start/end) to align with SB3 logging behaviour.
        """
        current_timestep = self.model.num_timesteps

        # Avoid duplicating the same timestep
        if self.timesteps and self.timesteps[-1] == current_timestep:
            return

        log_values = getattr(self.logger, "name_to_value", {})

        # Compute episode reward mean from SB3 ep_info_buffer (equivalent to rollout/ep_rew_mean)
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_rew = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        else:
            ep_rew = np.nan

        metrics = {
            "loss": log_values.get("train/loss", np.nan),
            "pg_loss": log_values.get("train/policy_gradient_loss", np.nan),
            "v_loss": log_values.get("train/value_loss", np.nan),
        }

        # Skip if everything is NaN (no useful metrics yet)
        values = [ep_rew] + list(metrics.values())
        if all(np.isnan(v) for v in values):
            return

        self.timesteps.append(current_timestep)
        self.ep_rew_mean.append(ep_rew)
        self.loss.append(metrics["loss"])
        self.pg_loss.append(metrics["pg_loss"])
        self.v_loss.append(metrics["v_loss"])
    
    def _on_rollout_start(self) -> bool:
        """
        SB3 clears logger values after each dump and records train metrics after the update.
        At the start of the next rollout those train metrics are still available, so capture them here.
        """
        self._log_metrics()
        return True

    def _on_training_end(self) -> None:
        """
        Capture final metrics after the last update.
        """
        self._log_metrics()
    
    def get_metrics_dict(self):
        """
        Return a dictionary containing all logged metrics.
        
        Returns:
            dict: Dictionary with keys 'timesteps', 'ep_rew_mean', 'loss', 'pg_loss', 'v_loss'
        """
        return {
            'timesteps': np.array(self.timesteps),
            'ep_rew_mean': np.array(self.ep_rew_mean),
            'loss': np.array(self.loss),
            'pg_loss': np.array(self.pg_loss),
            'v_loss': np.array(self.v_loss)
        }


class ISAC_Agent:
    """
    Proximal Policy Optimization (PPO) Agent for FAS-ISAC systems.
    
    This agent wraps the Stable-Baselines3 PPO implementation with additional
    utilities for saving, loading, and managing training.
    
    The agent uses:
    - Actor (Policy) network: Maps state to continuous actions (precoding matrix)
    - Critic (Value) network: Estimates value function for advantage computation
    - Clipped objective function for stable policy updates
    - Entropy regularization for exploration
    
    Attributes:
        env: The FAS_ISAC_Env environment
        model: The underlying PPO model from stable_baselines3
    """
    
    def __init__(self, env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 ent_coef=0.0, verbose=1):
        """
        Initialize the PPO Agent.
        
        Args:
            env: The gym environment (FAS_ISAC_Env)
            learning_rate (float): Learning rate for the optimizer. Default: 3e-4
            n_steps (int): Number of steps to collect per update. Default: 2048
            batch_size (int): Minibatch size for updates. Default: 64
            n_epochs (int): Number of epochs for SGD per update. Default: 10
            gamma (float): Discount factor. Default: 0.99
            gae_lambda (float): GAE lambda parameter. Default: 0.95
            clip_range (float): Clipping range for policy update. Default: 0.2
            ent_coef (float): Entropy coefficient for exploration. Default: 0.0
            verbose (int): Verbosity level. Default: 1
        """
        self.env = env
        
        # Initialize PPO model with MlpPolicy
        # Use verbose=0 to silence default ASCII table output
        self.model = PPO(
            policy=MlpPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            max_grad_norm=0.5,  # Gradient clipping for numerical stability
            verbose=0,  # Silence default logging, use custom callback instead
            tensorboard_log=None
        )
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
    
    def train(self, total_timesteps):
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps (int): Total number of timesteps to train
        
        Returns:
            self.model: The trained model
        """
        print(f"\n{'='*70}")
        print(f"Starting Training: {total_timesteps:,} timesteps")
        print(f"{'='*70}")
        print(f"Learning Rate:     {self.learning_rate}")
        print(f"Discount Factor:   {self.gamma}")
        print(f"GAE Lambda:        {self.gae_lambda}")
        print(f"Batch Size:        {self.batch_size}")
        print(f"{'='*70}\n")
        
        # Create callbacks for training
        metric_logger = MetricLoggerCallback()
        
        # Create tqdm progress bar
        with tqdm(total=total_timesteps, desc="Training Progress", unit="timesteps") as pbar:
            tqdm_callback = TqdmCallback(total_timesteps, pbar=pbar)
            # Use both callbacks: metrics logging + progress bar display
            self.model.learn(
                total_timesteps=total_timesteps, 
                callback=[metric_logger, tqdm_callback]
            )
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}\n")
        
        return self.model, metric_logger.get_metrics_dict()
    
    def predict(self, observation, deterministic=True):
        """
        Get action from the learned policy.
        
        Args:
            observation (np.ndarray): Current state observation
            deterministic (bool): If True, use mean of policy. If False, sample.
        
        Returns:
            action (np.ndarray): Action to take
            _state: Internal state (for recurrent policies, None for MLP)
        """
        action, _state = self.model.predict(observation, deterministic=deterministic)
        return action, _state
    
    def save(self, path):
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model (without .zip extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        self.model.save(path)
        print(f"Model saved to {path}.zip")
    
    @staticmethod
    def load(path, env=None):
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model (without .zip extension)
            env: The environment to attach the model to (optional)
        
        Returns:
            loaded_model: The loaded PPO model
        """
        loaded_model = PPO.load(path, env=env)
        print(f"Model loaded from {path}.zip")
        return loaded_model
    
    def get_model_info(self):
        """
        Print information about the model.
        """
        print(f"\n{'='*70}")
        print(f"PPO Agent Configuration")
        print(f"{'='*70}")
        print(f"Policy Type:       {type(self.model.policy).__name__}")
        print(f"Learning Rate:     {self.learning_rate}")
        print(f"Discount Factor:   {self.gamma}")
        print(f"GAE Lambda:        {self.gae_lambda}")
        print(f"Clip Range:        {self.clip_range}")
        print(f"Entropy Coeff:     {self.ent_coef}")
        print(f"Steps per Update:  {self.n_steps}")
        print(f"Batch Size:        {self.batch_size}")
        print(f"Epochs per Update: {self.n_epochs}")
        print(f"{'='*70}\n")
