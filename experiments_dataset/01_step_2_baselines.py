"""
Step 2: Compute average baselines (Random, Greedy, Upper proxy) per rho.

Outputs experiments_dataset/data/baseline_stats.pkl with mean scores per strategy.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from docs.basic_funcs import N, N_S, P_MAX_DBM, NOISE_POWER  # noqa: E402


def dbm_to_watts(dbm):
    return 10 ** (dbm / 10) / 1000


def calculate_sum_rate(H, selected_indices, P_max, noise_power):
    H_sub = H[:, selected_indices]  # (K, N_S)
    H_sub_H = H_sub.conj().T
    G = H_sub @ H_sub_H + noise_power * np.eye(H_sub.shape[0])
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)
    P_sub = H_sub_H @ G_inv  # (N_S, K)
    norm_sq = np.sum(np.abs(P_sub) ** 2)
    if norm_sq > 0:
        P = P_sub * np.sqrt(P_max / norm_sq)
    else:
        P = P_sub

    H_eff = H_sub @ P
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


def upper_bound_proxy(H, P_max, noise_power, num_samples=1000):
    best = -np.inf
    candidates = [greedy_indices(H)]
    for _ in range(num_samples):
        candidates.append(random_indices())
    seen = set()
    for idx in candidates:
        key = tuple(sorted(idx.tolist()))
        if key in seen:
            continue
        seen.add(key)
        rate = calculate_sum_rate(H, np.array(idx), P_max, noise_power)
        if rate > best:
            best = rate
    return best


def main():
    data_path = ROOT / "experiments_dataset" / "data" / "channels_dataset.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Channel dataset not found at {data_path}")

    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    P_max = dbm_to_watts(P_MAX_DBM)

    stats = {}
    for rho, channels in dataset.items():
        random_vals = []
        greedy_vals = []
        upper_vals = []

        for H in channels:
            # Random reward: mean over 10 draws
            rand_samples = []
            for _ in range(10):
                r_idx = random_indices()
                rand_samples.append(calculate_sum_rate(H, r_idx, P_max, NOISE_POWER))
            random_vals.append(float(np.mean(rand_samples)))

            g_idx = greedy_indices(H)
            g_val = calculate_sum_rate(H, g_idx, P_max, NOISE_POWER)
            greedy_vals.append(g_val)

            u_val = upper_bound_proxy(H, P_max, NOISE_POWER, num_samples=1000)
            upper_vals.append(u_val)

        stats[rho] = {
            "mean_random": float(np.mean(random_vals)),
            "mean_greedy": float(np.mean(greedy_vals)),
            "mean_upper": float(np.mean(upper_vals)),
        }
        print(f"rho={rho:.2f} | random={stats[rho]['mean_random']:.3f}, "
              f"greedy={stats[rho]['mean_greedy']:.3f}, upper={stats[rho]['mean_upper']:.3f}")

    out_dir = ROOT / "experiments_dataset" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_stats.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved baseline stats to {out_path}")


if __name__ == "__main__":
    main()
