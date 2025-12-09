"""
Channel Modeling for 2D Planar Fluid Antenna System (FAS) using Jakes' Model.

This module implements realistic wireless channel simulation for FAS-ISAC systems
with 2D planar antenna geometry, matching the paper:
"Fluid Antenna System Liberating Multiuser MIMO for ISAC via Deep Reinforcement Learning"

Key Functions:
    - get_2d_antenna_positions(Nx, Ny, Wx, Wy): Get 2D coordinates of antenna ports
    - get_correlation_matrix_2d(N, Nx, Ny, Wx, Wy, wavelength): Compute 2D spatial correlation
    - calculate_path_loss(distance): Compute path loss using paper's model (Eq. 37)
    - generate_channel(K, N, Nx, Ny, Wx, Wy, wavelength, path_loss=None): Generate correlated channel
    - update_channel_time_varying(channel, doppler_freq, dt): Evolve channel in time

System Parameters (from paper):
    - Carrier frequency: f_c = 3.4 GHz (default)
    - FAS dimensions: W_s = 0.1 × 0.1 m² (default)
    - Antenna count: N_s = 8 × 8 = 64 ports (default)
    - Antenna spacing: Varies based on FAS dimensions

2D FAS Geometry:
    - Antenna ports distributed evenly on 2D surface
    - Distance between ports i and j:
        Δd_{i,j} = sqrt((x_i - x_j)² + (y_i - y_j)²)
    - Spatial correlation: J_{i,j} = J_0(2π * Δd_{i,j} / λ)

Channel Generation (Eigenvalue Decomposition):
    - J_s = U Λ U^† (Eigenvalue decomposition of correlation matrix)
    - h_k = sqrt(l_k) * g_k * sqrt(Λ) * U^†
    - g_k ~ CN(0, 1) - small-scale fading
    - l_k - large-scale path loss

Path Loss Model (Eq. 37):
    PL(d) [dB] = 43.07 + 31 log10(d)
    where d is distance in meters

References:
    - Jakes, W. C. (1974). Microwave Mobile Communications
    - Paper: Fluid Antenna System Liberating Multiuser MIMO for ISAC via DRL
"""

import numpy as np
from scipy.special import j0  # Bessel function of the first kind


def get_2d_antenna_positions(Nx, Ny, Wx, Wy):
    """
    Compute 2D antenna port positions on the FAS surface.
    
    The antenna ports are distributed evenly on a rectangular surface of
    dimensions Wx × Wy. Using column-major indexing for linear-to-2D mapping.
    
    Position of port at linear index i:
        n_x = i % Nx
        n_y = i // Nx
        x_i = (n_x / (Nx - 1)) * Wx
        y_i = (n_y / (Ny - 1)) * Wy
    
    Args:
        Nx (int): Number of antenna ports in x-direction
        Ny (int): Number of antenna ports in y-direction
        Wx (float): Surface width in x-direction (meters)
        Wy (float): Surface width in y-direction (meters)
    
    Returns:
        np.ndarray: Antenna positions of shape (N, 2) where N = Nx * Ny
                   Each row is [x, y] coordinate in meters
    """
    N = Nx * Ny
    positions = np.zeros((N, 2))
    
    for i in range(N):
        # Column-major indexing (Fortran-style): n_x varies fastest
        n_x = i % Nx
        n_y = i // Nx
        
        # Normalized coordinates in [0, 1]
        x_norm = n_x / (Nx - 1) if Nx > 1 else 0
        y_norm = n_y / (Ny - 1) if Ny > 1 else 0
        
        # Physical coordinates in meters
        positions[i, 0] = x_norm * Wx
        positions[i, 1] = y_norm * Wy
    
    return positions


def get_correlation_matrix_2d(N, Nx, Ny, Wx, Wy, wavelength):
    """
    Compute the 2D spatial correlation matrix using Jakes' model.
    
    The correlation between antenna ports i and j is given by:
        J_{i,j} = J_0(2π * Δd_{i,j} / λ)
    
    where the 2D Euclidean distance is:
        Δd_{i,j} = sqrt((x_i - x_j)² + (y_i - y_j)²)
    
    Args:
        N (int): Total number of antenna ports (N = Nx * Ny)
        Nx (int): Number of antenna ports in x-direction
        Ny (int): Number of antenna ports in y-direction
        Wx (float): FAS surface width in x-direction (meters)
        Wy (float): FAS surface width in y-direction (meters)
        wavelength (float): Wavelength in meters (λ = c / f_c)
    
    Returns:
        np.ndarray: Spatial correlation matrix J_s of shape (N, N), complex-valued
                   J_s[i, j] = J_0(2π * Δd_{i,j} / λ)
    """
    # Get 2D antenna positions
    positions = get_2d_antenna_positions(Nx, Ny, Wx, Wy)
    
    # Compute pairwise Euclidean distances
    # Distance matrix shape: (N, N)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 2)
    distance_matrix = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Apply Jakes' model: J_{i,j} = J_0(2π * Δd_{i,j} / λ)
    correlation_matrix = j0(2 * np.pi * distance_matrix / wavelength)
    
    return correlation_matrix.astype(np.complex128)


def calculate_path_loss(distance, model='paper'):
    """
    Calculate path loss based on the paper's model.
    
    According to Eq. (37) in the paper:
        PL(d) [dB] = 43.07 + 31 log10(d)
    
    where d is the distance in meters.
    
    Linear (non-dB) form:
        PL_linear = 10^(PL_dB / 10) = 10^((43.07 + 31 log10(d)) / 10)
    
    Args:
        distance (float or np.ndarray): Distance in meters
        model (str): Path loss model type ('paper' or 'free_space')
    
    Returns:
        float or np.ndarray: Path loss in linear scale (not dB)
    """
    if model == 'paper':
        # Paper's model: PL(d) [dB] = 43.07 + 31 log10(d)
        pl_db = 43.07 + 31 * np.log10(np.maximum(distance, 1e-6))
        pl_linear = 10 ** (pl_db / 10)
    elif model == 'free_space':
        # Free-space path loss: PL = (4πd/λ)²
        # Placeholder - using paper model as default
        pl_linear = (4 * np.pi * distance) ** 2
    else:
        raise ValueError(f"Unknown path loss model: {model}")
    
    return pl_linear


def generate_channel(K, N, Nx, Ny, Wx, Wy, wavelength,
                     path_loss_distances=None, large_scale_fading=None,
                     user_correlation_factor=0.0):
    """
    Generate spatially correlated Rayleigh fading channel matrix using eigenvalue decomposition.
    
    This implementation follows the paper's channel generation method:
        1. Compute 2D spatial correlation matrix J_s
        2. Perform eigenvalue decomposition: J_s = U Λ U^†
        3. For each user k, generate i.i.d. small-scale fading g_k ~ CN(0, 1)
        4. Channel: h_k = sqrt(l_k) * g_k * sqrt(Λ) * U^†
    
    where:
        - l_k is the large-scale path loss for user k
        - g_k is the small-scale fading (K×N complex Gaussian)
        - sqrt(Λ) is the square root of eigenvalues
        - U is the unitary eigenvector matrix
    
    Args:
        K (int): Number of users/channels
        N (int): Total number of antenna ports (N = Nx * Ny)
        Nx (int): Number of antenna ports in x-direction
        Ny (int): Number of antenna ports in y-direction
        Wx (float): FAS surface width in x-direction (meters)
        Wy (float): FAS surface width in y-direction (meters)
        wavelength (float): Wavelength in meters
        path_loss_distances (np.ndarray, optional): Path loss distances for K users (shape: K,)
                            If None, set to 100 meters for all users
        large_scale_fading (np.ndarray, optional): Pre-computed large-scale fading (shape: K,)
                           If None, will be computed from path_loss_distances
    
    Returns:
        np.ndarray: Channel matrix H of shape (K, N), complex-valued
                   Following CN(0, spatial_correlation) distribution
    """
    # Compute 2D spatial correlation matrix
    J_s = get_correlation_matrix_2d(N, Nx, Ny, Wx, Wy, wavelength)
    
    # Eigenvalue decomposition: J_s = U Λ U^†
    eigenvalues, eigenvectors = np.linalg.eigh(J_s)
    
    # Ensure non-negative eigenvalues (handle numerical issues)
    eigenvalues = np.maximum(eigenvalues, 0)
    Lambda_sqrt = np.sqrt(eigenvalues)  # Shape: (N,)
    U = eigenvectors  # Shape: (N, N)
    
    # Generate small-scale fading with optional inter-user correlation
    if user_correlation_factor <= 0.0:
        # Independent users (default behaviour)
        g_k = (np.random.normal(0, 1/np.sqrt(2), (K, N)) +
               1j * np.random.normal(0, 1/np.sqrt(2), (K, N)))
    else:
        rho = np.clip(user_correlation_factor, 0.0, 1.0)
        # Shared component across users
        g_common = (np.random.normal(0, 1/np.sqrt(2), (1, N)) +
                    1j * np.random.normal(0, 1/np.sqrt(2), (1, N)))
        # Independent components per user
        g_indep = (np.random.normal(0, 1/np.sqrt(2), (K, N)) +
                   1j * np.random.normal(0, 1/np.sqrt(2), (K, N)))
        g_k = np.sqrt(rho) * g_common + np.sqrt(1.0 - rho) * g_indep
    
    # Compute large-scale path loss if not provided
    if large_scale_fading is None:
        if path_loss_distances is None:
            # Default: 100 meters distance for all users
            path_loss_distances = np.ones(K) * 100.0
        
        # Path loss: l_k = 1 / PL(d_k)
        path_loss_linear = calculate_path_loss(path_loss_distances, model='paper')
        large_scale_fading = 1.0 / np.sqrt(path_loss_linear)  # Shape: (K,)
    
    # Channel generation: h_k = sqrt(l_k) * g_k * (sqrt(Λ) * U^†)
    # First compute the spatial part: (sqrt(Λ) * U^†)
    spatial_part = Lambda_sqrt[:, np.newaxis] * U.conj().T  # (N, N)
    
    # Then apply small-scale fading and large-scale fading
    # g_k * spatial_part gives (K, N) * (N, N) -> (K, N)
    H = (large_scale_fading[:, np.newaxis] *  # (K, 1)
         (g_k @ spatial_part))  # (K, N) @ (N, N) -> (K, N)
    
    return H.astype(np.complex128)


def doppler_shift(t, v, wavelength, f_c):
    """
    Compute Doppler frequency shift.
    
    Args:
        t (float): Time in seconds
        v (float): Velocity in m/s
        wavelength (float): Wavelength in meters
        f_c (float): Carrier frequency in Hz
    
    Returns:
        float: Doppler frequency f_d = v * f_c / c (in Hz)
    """
    c = 3e8  # Speed of light
    return v * f_c / c


def update_channel_time_varying(channel, doppler_freq, dt, method='jakes'):
    """
    Evolve the channel matrix in time using a time-varying Jakes' model.
    
    This function simulates channel fading over time by updating the phase of
    channel coefficients based on Doppler shift.
    
    Args:
        channel (np.ndarray): Current channel matrix (K, N) complex
        doppler_freq (float): Doppler frequency in Hz
        dt (float): Time step in seconds
        method (str): Method for time evolution ('jakes' or 'random_walk')
    
    Returns:
        np.ndarray: Updated channel matrix (K, N) complex
    """
    if method == 'jakes':
        # Phase rotation based on Doppler shift
        # Each coefficient rotates by angle = 2π * f_d * dt
        phase_rotation = np.exp(1j * 2 * np.pi * doppler_freq * dt)
        channel_updated = channel * phase_rotation
    
    elif method == 'random_walk':
        # Random walk: add small complex Gaussian increment
        # This models a more gradual channel change
        step_size = 0.01  # Tunable parameter
        noise = step_size * (np.random.normal(0, 1, channel.shape) + 
                            1j * np.random.normal(0, 1, channel.shape))
        channel_updated = channel + noise
        # Normalize to maintain average power
        channel_updated = channel_updated / np.mean(np.abs(channel_updated)) * np.mean(np.abs(channel))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return channel_updated

