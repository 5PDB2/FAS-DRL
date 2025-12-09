"""
Gym Environment for 2D Planar Fluid Antenna System (FAS) with Port Selection.

FAS_PortSelection_Env
=====================
Implements a reinforcement learning environment for port selection in 2D Planar FAS.
The RL agent selects which antenna ports to activate, and MMSE beamforming is 
computed deterministically inside the environment.

Paper: "Fluid Antenna System Liberating Multiuser MIMO for ISAC via Deep RL"
Problem: Port Selection with MMSE Precoding

Key Methods:
    - reset(): Initialize environment and generate channel
    - step(action): Agent selects ports, MMSE precoding applied, reward computed
    - _select_ports(action): Extract top n_s ports from action weights
    - _compute_mmse_precoder(H_sub): Compute MMSE precoding for selected ports

State Space:
    - Full Channel State Information (CSI): H (K × N) flattened
    - Agent uses global CSI to decide which ports to select

Action Space:
    - Vector of size (N,) representing weights/logits for each port
    - Environment selects top n_s ports with highest weights (via argsort)

Reward:
    - Sum rate (Shannon capacity): Σ log2(1 + SINRₖ)
    - Info includes sensing metrics (beampattern gain) for tracking

MMSE Precoding Formula:
    P_sub = H_sub^H (H_sub H_sub^H + σ²I_K)^(-1)
    P = P_sub / ||P_sub||_F * sqrt(P_max)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from docs.basic_funcs import (
    N, K, N_S, NX, NY, WX, WY, P_MAX_DBM, NOISE_POWER, 
    TARGET_ANGLE, TARGET_DISTANCE, dbm_to_watts, LAMBDA
)
from utils.channel import generate_channel
from utils.math_ops import calculate_sinr, calculate_sensing_gain


class FAS_PortSelection_Env(gym.Env):
    """
    Port Selection Environment for 2D Planar Fluid Antenna System.
    
    The RL agent performs port selection (choosing n_s best ports from N total),
    while MMSE beamforming is computed deterministically using the selected channel.
    
    This implements the "MMSE baseline" setup where:
    - Agent: Selects antenna ports
    - Environment: Computes MMSE precoding and calculates communication metrics
    
    Attributes:
        N (int): Total number of antenna ports (64)
        K (int): Number of communication users (3)
        N_S (int): Number of selected/active ports (5)
        current_channel (ndarray): Full channel H (K, N) complex
        selected_ports (ndarray): Indices of selected ports
        H_sub (ndarray): Sub-channel for selected ports (K, N_S) complex
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, max_steps=50):
        """
        Initialize the Port Selection environment.
        
        Parameters:
            max_steps (int): Maximum steps per episode (default: 50)
        """
        super(FAS_PortSelection_Env, self).__init__()
        
        # System parameters
        self.N = N                              # Total antenna ports
        self.K = K                              # Number of users
        self.N_S = N_S                          # Number of selected ports
        self.P_MAX = dbm_to_watts(P_MAX_DBM)   # Max transmit power (Watts)
        self.NOISE_POWER = NOISE_POWER          # Noise power (Watts)
        self.TARGET_ANGLE = TARGET_ANGLE
        self.TARGET_DISTANCE = TARGET_DISTANCE
        self.max_steps = max_steps
        self.current_step = 0
        
        # Channel and state
        self.current_channel = None  # H (K, N)
        self.selected_ports = None   # Indices of selected ports
        self.H_sub = None            # Sub-channel (K, N_S)
        self.current_obs = None
        self.user_distances = None   # Per-user distances (meters) for path loss
        self.obs_scale = 1e5         # Scale channel for stable NN training
        self._printed_scale = False  # Print scaled range once for verification
        
        # Action space: Weights/logits for each of N ports
        # Environment will select top N_S ports via argsort
        # Bounded to [-1, 1] for stable RL training
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.N,),
            dtype=np.float32
        )
        
        # Observation space: Flattened CSI with power features [real; imag; power]
        # Shape: (3*K*N,)
        low_vec = np.concatenate([
            np.full(2 * self.K * self.N, -10.0, dtype=np.float32),  # real and imag
            np.zeros(self.K * self.N, dtype=np.float32)             # power is non-negative
        ])
        high_vec = np.full(3 * self.K * self.N, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_vec,
            high=high_vec,
            shape=(3 * self.K * self.N,),
            dtype=np.float32
        )
    
    def _flatten_complex(self, matrix):
        """
        Flatten a complex matrix to [real_parts; imag_parts].
        
        Args:
            matrix (ndarray): Complex matrix (M, N)
        
        Returns:
            ndarray: Flattened real array (2*M*N,)
        """
        real_parts = np.real(matrix).flatten()
        imag_parts = np.imag(matrix).flatten()
        return np.concatenate([real_parts, imag_parts]).astype(np.float32)

    def _build_observation(self, channel):
        """
        Build observation vector [Re(H), Im(H), Power(H)] using scaled channel.
        """
        scaled_channel = channel * self.obs_scale
        real_parts = np.real(scaled_channel).flatten()
        imag_parts = np.imag(scaled_channel).flatten()
        power_parts = (real_parts ** 2 + imag_parts ** 2)
        obs = np.concatenate([real_parts, imag_parts, power_parts]).astype(np.float32)
        return obs
    
    def _unflatten_to_complex(self, flat_array, shape):
        """
        Convert flattened [real_parts; imag_parts] back to complex matrix.
        
        Args:
            flat_array (ndarray): Flattened array (2*M*N,)
            shape (tuple): Target shape (M, N)
        
        Returns:
            ndarray: Complex matrix (M, N)
        """
        size = np.prod(shape)
        real_parts = flat_array[:size].reshape(shape)
        imag_parts = flat_array[size:].reshape(shape)
        return real_parts + 1j * imag_parts
    
    def _select_ports(self, action):
        """
        Select top N_S ports based on action weights.
        
        The action is a vector of size (N,) representing weights/logits for each port.
        This function selects the top N_S ports with highest weights.
        
        Args:
            action (ndarray): Weight vector of size (N,)
        
        Returns:
            selected_indices (ndarray): Indices of selected ports, shape (N_S,)
        """
        # Get indices of top N_S ports (highest weights)
        selected_indices = np.argsort(action)[-self.N_S:]  # Top N_S indices
        selected_indices = np.sort(selected_indices)  # Sort for consistency
        return selected_indices
    
    def _compute_mmse_precoder(self, H_sub):
        """
        Compute MMSE precoding matrix for the sub-channel.
        
        MMSE formula (unnormalized):
            P_sub = H_sub^H (H_sub H_sub^H + σ²I_K)^(-1)
        
        Shape: H_sub (K, N_S) -> P_sub (N_S, K)
        
        Args:
            H_sub (ndarray): Sub-channel matrix (K, N_S) complex
        
        Returns:
            P_sub (ndarray): MMSE precoding matrix (N_S, K) complex
        """
        K = H_sub.shape[0]
        
        # H_sub^H: (N_S, K)
        H_sub_H = H_sub.conj().T
        
        # H_sub H_sub^H: (K, K)
        H_H = H_sub @ H_sub_H  # (K, N_S) @ (N_S, K) -> (K, K)
        
        # Add noise: H_sub H_sub^H + σ²I_K
        G = H_H + self.NOISE_POWER * np.eye(K)
        
        # Inverse
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudo-inverse
            G_inv = np.linalg.pinv(G)
        
        # P_sub = H_sub^H @ G_inv: (N_S, K) @ (K, K) -> (N_S, K)
        P_sub = H_sub_H @ G_inv
        
        return P_sub
    
    def _normalize_power(self, P_sub):
        """
        Normalize precoding matrix to satisfy total power constraint.
        
        Formula: P = P_sub * sqrt(P_max) / ||P_sub||_F
        
        Ensures: Tr(P P^H) ≤ P_max
        
        Args:
            P_sub (ndarray): Precoding matrix (N_S, K) complex
        
        Returns:
            P (ndarray): Power-normalized precoding matrix (N_S, K) complex
        """
        # Frobenius norm squared
        norm_sq = np.sum(np.abs(P_sub) ** 2)
        
        if norm_sq > 0:
            # Scale to meet power constraint
            P = P_sub * np.sqrt(self.P_MAX / norm_sq)
        else:
            # Avoid division by zero
            P = P_sub
        
        return P
    
    def reset(self, seed=None):
        """
        Reset environment and generate new channel.
        
        Returns:
            observation (ndarray): Flattened full CSI (2*K*N,)
            info (dict): Empty info dict
        """
        super().reset(seed=seed)
        
        # Generate new 2D FAS channel: H (K, N)
        # Users are uniformly distributed in an annulus r in [40, 60] m (paper Sec. VI)
        self.user_distances = self.np_random.uniform(40.0, 60.0, size=self.K)
        self.current_channel = generate_channel(
            K=self.K,
            N=self.N,
            Nx=NX,
            Ny=NY,
            Wx=WX,
            Wy=WY,
            wavelength=LAMBDA,
            path_loss_distances=self.user_distances
        )
        # Build scaled observation with power feature
        self.current_obs = self._build_observation(self.current_channel)

        if not self._printed_scale:
            obs_min, obs_max = self.current_obs.min(), self.current_obs.max()
            print(f"[FAS Env] Scaled obs range: min={obs_min:.3e}, max={obs_max:.3e}")
            self._printed_scale = True
        self.current_step = 0
        self.selected_ports = None
        self.H_sub = None
        
        return self.current_obs, {}
    
    def step(self, action):
        """
        Execute one step: Port selection + MMSE precoding + reward computation.
        
        Workflow:
            1. Select top N_S ports from action weights
            2. Extract sub-channel H_sub (K, N_S)
            3. Compute MMSE precoding P_sub
            4. Normalize power
            5. Compute SINR and sum rate
            6. Calculate sensing metrics
            7. Return reward and info
        
        Args:
            action (ndarray): Port weight vector (N,)
        
        Returns:
            observation (ndarray): Current CSI observation (2*K*N,)
            reward (float): Sum rate (communication reward)
            terminated (bool): Episode finished
            truncated (bool): Episode truncated
            info (dict): Additional metrics (SINR, sensing, etc.)
        """
        self.current_step += 1
        
        # Step 1: Select ports
        self.selected_ports = self._select_ports(action)
        
        # Step 2: Extract sub-channel
        # H (K, N) -> H_sub (K, N_S) using selected port indices
        self.H_sub = self.current_channel[:, self.selected_ports]
        
        # Step 3: Compute MMSE precoding
        P_sub = self._compute_mmse_precoder(self.H_sub)
        
        # Step 4: Normalize power
        P = self._normalize_power(P_sub)
        
        # Step 5: Compute SINR and sum rate
        # Effective channel: H_sub @ P (K, N_S) @ (N_S, K) -> (K, K)
        # For SINR: diagonal elements represent each user's received signal
        H_eff = self.H_sub @ P  # (K, K)
        
        # SINR for user k: |H_eff[k,k]|^2 / (noise + interference)
        sinr_list = []
        for k in range(self.K):
            # Desired signal power: |H_eff[k,k]|^2
            signal_power = np.abs(H_eff[k, k]) ** 2
            
            # Interference power: sum of other users' signals
            interference_power = np.sum(np.abs(H_eff[k, :]) ** 2) - signal_power
            
            # SINR
            sinr_k = signal_power / (interference_power + self.NOISE_POWER)
            sinr_list.append(sinr_k)
        
        # Sum rate: communication reward
        sum_rate = np.sum(np.log2(1.0 + np.array(sinr_list)))
        
        # Step 6: Calculate sensing metrics
        # Beampattern gain at target angle (using P for sensing)
        sensing_gain = calculate_sensing_gain(P, self.TARGET_ANGLE)
        
        # Step 7: Prepare output
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        # Reward is sum rate (communication only, as per MMSE baseline)
        #reward = sum_rate
        reward = sum_rate  # Normalize reward per step
        
        # Additional information
        info = {
            'sinr': sinr_list,
            'sum_rate': sum_rate,
            'sensing_gain': sensing_gain,
            'selected_ports': self.selected_ports.tolist(),
            'num_selected': len(self.selected_ports),
            'power_used': np.sum(np.abs(P) ** 2),
            'step': self.current_step
        }
        # Return scaled observation to maintain consistent magnitude
        self.current_obs = self._build_observation(self.current_channel)
        
        return self.current_obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render environment state (placeholder).
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            print(f"\n--- Step {self.current_step} ---")
            if self.selected_ports is not None:
                print(f"Selected Ports: {self.selected_ports}")
            print(f"Max Steps: {self.max_steps}")
