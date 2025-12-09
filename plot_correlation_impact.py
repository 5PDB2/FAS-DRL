"""
Monte Carlo study: Impact of inter-user correlation on port selection strategies.

Plots average sum rate vs. correlation factor for:
    - Random selection
    - Greedy (PGMax)
    - Upper bound proxy (random search + greedy)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from utils.channel import generate_channel
from docs.basic_funcs import N, K, N_S, P_MAX_DBM, NOISE_POWER

# For reproducibility
random.seed(42)
np.random.seed(42)


def calculate_sum_rate(H, selected_indices, P_max, noise_power):
    """
    Compute sum rate with MMSE precoding for selected ports.
    """
    H_sub = H[:, selected_indices]  # (K, N_S)
    H_sub_H = H_sub.conj().T
    G = H_sub @ H_sub_H + noise_power * np.eye(H_sub.shape[0])
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)
    P_sub = H_sub_H @ G_inv  # (N_S, K)

    # Normalize power
    norm_sq = np.sum(np.abs(P_sub) ** 2)
    if norm_sq > 0:
        P = P_sub * np.sqrt(P_max / norm_sq)
    else:
        P = P_sub

    # Effective channel
    H_eff = H_sub @ P  # (K, K)
    sinr_list = []
    for k in range(H_eff.shape[0]):
        signal = np.abs(H_eff[k, k]) ** 2
        interference = np.sum(np.abs(H_eff[k, :]) ** 2) - signal
        sinr = signal / (interference + noise_power)
        sinr_list.append(sinr)
    return float(np.sum(np.log2(1.0 + np.array(sinr_list))))


def greedy_indices(H):
    power = np.sum(np.abs(H) ** 2, axis=0)
    return np.argsort(power)[-N_S:]


def random_indices():
    return np.random.choice(N, N_S, replace=False)


def upper_bound_proxy(H, num_samples=1000):
    """
    Random search proxy: sample combinations + greedy.
    """
    best_rate = -np.inf
    # Include greedy candidate
    candidates = [greedy_indices(H)]
    for _ in range(num_samples):
        candidates.append(random_indices())
    seen = set()
    for idx in candidates:
        key = tuple(sorted(idx.tolist()))
        if key in seen:
            continue
        seen.add(key)
        rate = calculate_sum_rate(H, np.array(idx), P_max, NOISE_POWER)
        if rate > best_rate:
            best_rate = rate
    return best_rate


def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10.0)


def main():
    correlation_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    runs = 100
    global P_max
    P_max = dbm_to_watts(P_MAX_DBM)

    results_random = []
    results_greedy = []
    results_upper = []

    for rho in correlation_values:
        rates_r = []
        rates_g = []
        rates_u = []
        spacing = 0.5 * 1.0  # wavelength is 1.0 in this script
        side = int(np.sqrt(N))
        calculated_wx = (side - 1) * spacing
        calculated_wy = (side - 1) * spacing
        for _ in range(runs):
            H = generate_channel(
                K=K, N=N, Nx=side, Ny=side,
                Wx=calculated_wx, Wy=calculated_wy, wavelength=1.0,
                user_correlation_factor=rho
            )
            # Random
            r_idx = random_indices()
            rates_r.append(calculate_sum_rate(H, r_idx, P_max, NOISE_POWER))
            # Greedy
            g_idx = greedy_indices(H)
            rates_g.append(calculate_sum_rate(H, g_idx, P_max, NOISE_POWER))
            # Upper bound proxy
            rates_u.append(upper_bound_proxy(H, num_samples=1000))
        results_random.append(np.mean(rates_r))
        results_greedy.append(np.mean(rates_g))
        results_upper.append(np.mean(rates_u))
        print(f"rho={rho:.2f} | Random={results_random[-1]:.3f}, Greedy={results_greedy[-1]:.3f}, Upper={results_upper[-1]:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(correlation_values, results_random, 'g-o', label='Random')
    plt.plot(correlation_values, results_greedy, 'r-o', label='Greedy (PGMax)')
    plt.plot(correlation_values, results_upper, 'k--o', label='Upper Bound (proxy)')
    plt.xlabel('Inter-User Correlation Factor (rho)')
    plt.ylabel('Average Sum Rate (bps/Hz)')
    plt.title('Impact of Inter-User Correlation on Port Selection Strategies')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = "correlation_impact.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
