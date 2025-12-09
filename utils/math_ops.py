"""
Matrix Operations and Beamforming Calculations.

This module provides signal processing functions for FAS systems.

Key Functions:
    - normalize_power(W, p_max): Enforce power constraint on precoding matrix
    - calculate_sinr(H, W, noise_power, selection_indices=None): Compute SINR for each user
    - calculate_sensing_gain(W, target_angle, antenna_positions, wavelength): Compute beampattern gain
    - steering_vector(angle, positions, wavelength): Compute array steering vector
    - precoder_design(H, method='mrt'): Design precoding matrix using various methods

Beamforming Methods:
    - Maximum Ratio Transmission (MRT)
    - Zero-Forcing (ZF)
    - Regularized Zero-Forcing (RZF)
    - Eigenbeamforming

Parameters:
    - SNR (Signal-to-Noise Ratio)
    - Number of antennas: N
    - Number of users: K
    - Antenna spacing: d (typically lambda/2)
"""

import numpy as np


def normalize_power(W, p_max):
    """
    Normalize precoding matrix to satisfy power constraint.
    
    If the total transmit power P_tx = tr(W @ W^H) exceeds p_max,
    scale W down by factor sqrt(p_max / P_tx).
    
    Args:
        W (np.ndarray): Precoding matrix of shape (N_RF, K) complex
        p_max (float): Maximum transmit power in Watts
    
    Returns:
        np.ndarray: Normalized precoding matrix (N_RF, K) complex
        float: Actual transmit power after normalization
    """
    # Calculate total power: P_tx = trace(W @ W^H)
    power_tx = np.trace(W @ W.conj().T).real
    
    if power_tx > p_max:
        # Scale down to meet power constraint
        scale_factor = np.sqrt(p_max / power_tx)
        W_normalized = W * scale_factor
        power_tx_actual = p_max
    else:
        W_normalized = W
        power_tx_actual = power_tx
    
    return W_normalized, power_tx_actual


def steering_vector(angle, antenna_positions, wavelength):
    """
    Compute array steering vector for a given direction.
    
    For a ULA (Uniform Linear Array) or FAS with positions p_i,
    the steering vector is:
        a(θ) = [e^(j*2π*p_1*sin(θ)/λ), ..., e^(j*2π*p_N*sin(θ)/λ)]^T
    
    Args:
        angle (float): Direction of arrival/departure in radians
        antenna_positions (np.ndarray): Antenna positions (N,) in meters
        wavelength (float): Wavelength in meters
    
    Returns:
        np.ndarray: Steering vector of shape (N,) complex
    """
    exponent = 1j * 2 * np.pi * antenna_positions * np.sin(angle) / wavelength
    return np.exp(exponent)


def calculate_sinr(H, W, noise_power, selection_indices=None):
    """
    Calculate Signal-to-Interference-plus-Noise Ratio (SINR) for each user.
    
    Given:
        - Channel matrix H: (K, N) complex
        - Precoding matrix W: (N_RF, K) complex
        - Noise power: scalar
    
    The received signal for user k is: y_k = h_k^H @ w_k + interference + noise
    
    SINR_k = |h_k^H @ w_k|^2 / (Σ_{j≠k} |h_k^H @ w_j|^2 + noise_power)
    
    Args:
        H (np.ndarray): Channel matrix (K, N) complex
        W (np.ndarray): Precoding matrix (N_RF, K) complex
        noise_power (float): Thermal noise power in Watts
        selection_indices (np.ndarray, optional): Indices of active antenna ports (N_RF,)
            If None, assumes first N_RF ports are used
    
    Returns:
        list: SINR values for each user, shape (K,)
    """
    K = H.shape[0]  # Number of users
    N_RF = W.shape[0]  # Number of RF chains
    
    # Determine which antenna ports are active
    if selection_indices is None:
        # Default: use first N_RF ports
        selection_indices = np.arange(min(N_RF, H.shape[1]))
    
    # Extract effective channel for selected ports
    # H_eff: (K, N_RF) complex
    H_eff = H[:, selection_indices]
    
    # Compute received signal components for each user
    sinr_list = []
    
    for k in range(K):
        # Signal component: |h_k^H @ w_k|^2
        signal_power = np.abs(H_eff[k, :] @ W[:, k]) ** 2
        
        # Interference component: Σ_{j≠k} |h_k^H @ w_j|^2
        interference_power = 0.0
        for j in range(K):
            if j != k:
                interference_power += np.abs(H_eff[k, :] @ W[:, j]) ** 2
        
        # SINR = Signal / (Interference + Noise)
        sinr = signal_power / (interference_power + noise_power)
        sinr_list.append(sinr)
    
    return sinr_list


def calculate_sensing_gain(W, target_angle, antenna_positions=None, wavelength=None):
    """
    Calculate the beampattern gain at the target angle.
    
    The beampattern gain is the directional power sent towards the target:
        P(θ) = |a(θ)^H @ W|^2
    where a(θ) is the steering vector at angle θ.
    
    For FAS, the effective antenna positions are determined by the RF chains.
    
    Args:
        W (np.ndarray): Precoding/beamforming matrix (N_RF, K) complex
        target_angle (float): Target angle in radians
        antenna_positions (np.ndarray, optional): Antenna positions for active ports (N_RF,)
            If None, assumes uniform linear spacing from 0 to (N_RF-1)*lambda/2
        wavelength (float, optional): Wavelength in meters
            If None, uses default lambda = 10.7 mm (28 GHz)
    
    Returns:
        float: Beampattern gain at target angle (normalized power)
    """
    N_RF = W.shape[0]
    
    # Default wavelength for 28 GHz
    if wavelength is None:
        wavelength = 3e8 / 28e9  # ~10.7 mm
    
    # Default antenna positions: uniform linear array
    if antenna_positions is None:
        antenna_positions = np.arange(N_RF) * (wavelength / 2)
    
    # Compute steering vector for target angle
    a_target = steering_vector(target_angle, antenna_positions, wavelength)
    
    # Compute beampattern response: |a^H @ W|^2
    # Sum across all users to get total power towards target
    beampattern_response = a_target.conj() @ W  # Shape: (K,)
    gain = np.sum(np.abs(beampattern_response) ** 2)
    
    return gain


def precoder_design(H, method='mrt', p_max=None):
    """
    Design precoding matrix using various methods.
    
    Args:
        H (np.ndarray): Channel matrix (K, N) complex
        method (str): Design method ('mrt', 'zf', 'rzf', 'eigen')
        p_max (float, optional): Maximum power constraint for normalization
    
    Returns:
        np.ndarray: Precoding matrix (N, K) complex
    """
    K, N = H.shape
    
    if method == 'mrt':
        # Maximum Ratio Transmission: w_k = h_k^H / ||h_k||
        W = np.zeros((N, K), dtype=complex)
        for k in range(K):
            W[:, k] = H[k, :].conj() / np.linalg.norm(H[k, :])
    
    elif method == 'zf':
        # Zero-Forcing: W = H^H (H @ H^H)^{-1}
        H_gram = H @ H.conj().T  # (K, K)
        W_zf = H.conj().T @ np.linalg.inv(H_gram)  # (N, K)
        W = W_zf
    
    elif method == 'rzf':
        # Regularized Zero-Forcing: W = H^H (H @ H^H + λI)^{-1}
        # Regularization parameter: λ = σ^2 / SNR
        lambda_reg = 0.1  # Tunable regularization parameter
        H_gram = H @ H.conj().T + lambda_reg * np.eye(K)
        W_rzf = H.conj().T @ np.linalg.inv(H_gram)
        W = W_rzf
    
    elif method == 'eigen':
        # Eigenbeamforming: uses dominant eigenvector of H^H @ H
        H_gram = H.conj().T @ H  # (N, N)
        eigvals, eigvecs = np.linalg.eigh(H_gram)
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(-eigvals)
        # Take K dominant eigenvectors as precoding matrix
        W = eigvecs[:, idx[:K]]
    
    else:
        raise ValueError(f"Unknown precoding method: {method}")
    
    # Normalize power if constraint is provided
    if p_max is not None:
        W, _ = normalize_power(W, p_max)
    
    return W


def capacity_shannon(snr):
    """
    Compute Shannon capacity.
    
    C = log2(1 + SNR) bits/s/Hz
    
    Args:
        snr (float or np.ndarray): Signal-to-noise ratio (linear)
    
    Returns:
        float or np.ndarray: Capacity in bits/s/Hz
    """
    return np.log2(1 + snr)


def mutual_information(tx_symbol, rx_symbol, noise_power):
    """
    Estimate mutual information between transmitted and received symbols.
    
    Args:
        tx_symbol (np.ndarray): Transmitted symbols
        rx_symbol (np.ndarray): Received symbols
        noise_power (float): Noise power
    
    Returns:
        float: Mutual information estimate
    """
    # Simple estimate: MI ≈ log2(1 + signal_power / noise_power)
    signal_power = np.mean(np.abs(rx_symbol - np.mean(rx_symbol)) ** 2)
    mi = np.log2(1 + signal_power / noise_power)
    return mi
