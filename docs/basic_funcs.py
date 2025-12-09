"""
Global Configuration and Basic Functions.

This module defines system parameters and utility functions used across
the entire FAS-DRL project.

System Model:
    - Port Selection Problem: Agent selects n_s best antenna ports from N total ports
    - Beamforming: MMSE precoding calculated deterministically inside environment
    - Paper: "Fluid Antenna System Liberating Multiuser MIMO for ISAC via Deep RL"
    - Section VI: Simulation Results parameters

System Parameters:
    - N: Total number of fluid antenna elements (64)
    - K: Number of communication users / targets (3)
    - n_s: Number of active selected ports (5)
    - f_c: Carrier frequency (Hz)
    - lambda: Wavelength (m)
    - P_MAX_DBM: Maximum transmit power (dBm)
    - P_MAX: Maximum transmit power (Watts, linear)
    - NOISE_POWER_DBM: Thermal noise power (dBm)
    - NOISE_POWER: Thermal noise power (Watts, linear)
    - TARGET_DISTANCE: Sensing target distance (meters)

Unit Conversions:
    - dbm_to_watts(dbm): Convert dBm to Watts
    - watts_to_dbm(watts): Convert Watts to dBm
    - db_to_linear(db): Convert dB to linear ratio
    - linear_to_db(linear): Convert linear ratio to dB

Constants:
    - Speed of light: c = 3e8 m/s
    - Boltzmann constant: k_B = 1.38e-23 J/K
"""

import numpy as np

# ============================================================================
# System Parameters (from Paper Section VI)
# ============================================================================

# Antenna and RF Configuration
# 2D Planar FAS Parameters (from paper)
NX = 8              # Number of antenna ports in x-direction
NY = 8              # Number of antenna ports in y-direction
N = NX * NY         # Total number of Fluid Antenna ports (64)
K = 3               # Number of Users / Communication channels
N_S = 5             # Number of Selected/Active ports (from paper: n_s=5)

# Wireless Parameters
C = 3e8             # Speed of light (m/s)
FC = 3.4e9          # Carrier Frequency (3.4 GHz, from paper)
LAMBDA = C / FC     # Wavelength (m)
SPACING = LAMBDA / 2  # Half-wavelength spacing
D_ANT = SPACING       # Backward compatibility alias

# FAS Surface Dimensions computed from half-wavelength spacing
# WX/WY span (NX-1)/(NY-1) gaps of 0.5 lambda each
WX = (NX - 1) * SPACING
WY = (NY - 1) * SPACING

# Power Parameters (from paper Section VI)
P_MAX_DBM = 40              # Maximum transmit power (20 dBm, from paper)
P_MAX = 10 ** (P_MAX_DBM / 10) / 1000  # Convert to Watts (~0.1 W)

# Noise Parameters
NOISE_POWER_DBM = -80       # Thermal noise power (dBm)
NOISE_POWER = 10 ** (NOISE_POWER_DBM / 10) / 1000  # Convert to Watts

# Sensing Parameters
TARGET_DISTANCE = 200.0     # Sensing target distance (meters, from paper)
TARGET_ANGLE = np.pi / 4    # Target angle for sensing (45 degrees)

# Constants
BOLTZMANN = 1.38e-23        # Boltzmann constant (J/K)


# ============================================================================
# Unit Conversion Functions
# ============================================================================

def dbm_to_watts(dbm):
    """
    Convert power from dBm to Watts.
    
    Formula: P_watts = 10^(P_dBm / 10) / 1000
    
    Args:
        dbm (float): Power in dBm
    
    Returns:
        float: Power in Watts
    """
    return 10 ** (dbm / 10) / 1000


def watts_to_dbm(watts):
    """
    Convert power from Watts to dBm.
    
    Formula: P_dBm = 10 * log10(P_watts * 1000)
    
    Args:
        watts (float): Power in Watts
    
    Returns:
        float: Power in dBm
    """
    return 10 * np.log10(watts * 1000)


def db_to_linear(db):
    """
    Convert from dB scale to linear ratio.
    
    Formula: linear = 10^(dB / 10)
    
    Args:
        db (float): Value in dB
    
    Returns:
        float: Linear ratio
    """
    return 10 ** (db / 10)


def linear_to_db(linear):
    """
    Convert from linear ratio to dB scale.
    
    Formula: dB = 10 * log10(linear)
    
    Args:
        linear (float): Linear ratio
    
    Returns:
        float: Value in dB
    """
    return 10 * np.log10(linear)


# ============================================================================
# System Info
# ============================================================================

def print_system_config():
    """Print the current system configuration."""
    print("=" * 70)
    print("FAS PORT SELECTION SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"Total Antenna Ports (N):         {N}")
    print(f"Antenna Array Dimension:         {NX} × {NY} = {N} ports")
    print(f"FAS Surface Dimensions:          {WX} × {WY} m²")
    print(f"Number of Users (K):             {K}")
    print(f"Number of Selected Ports (n_s):  {N_S}")
    print(f"Carrier Frequency (f_c):         {FC / 1e9:.1f} GHz")
    print(f"Wavelength (λ):                  {LAMBDA * 1e3:.3f} mm")
    print(f"Antenna Spacing:                 {SPACING * 1e3:.2f} mm ({SPACING / LAMBDA:.2f} λ)")
    print(f"Max Transmit Power:              {P_MAX_DBM} dBm = {P_MAX:.4f} W")
    print(f"Noise Power:                     {NOISE_POWER_DBM} dBm = {NOISE_POWER:.2e} W")
    print(f"Sensing Target Angle:            {np.degrees(TARGET_ANGLE):.1f}°")
    print(f"Sensing Target Distance:         {TARGET_DISTANCE} m")
    print("=" * 70)
