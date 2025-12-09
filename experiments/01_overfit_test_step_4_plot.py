"""
Step 4: Plot DRL overfit results vs greedy and upper bound on typical channels.

Outputs experiments/results/01_overfit_benchmark.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))


def main():
    baseline_path = ROOT / "experiments" / "data" / "baseline_results.pkl"
    drl_path = ROOT / "experiments" / "data" / "drl_results.pkl"

    if not baseline_path.exists() or not drl_path.exists():
        raise FileNotFoundError("Missing baseline or DRL results. Run steps 1-3 first.")

    with open(baseline_path, "rb") as f:
        baselines = pickle.load(f)
    with open(drl_path, "rb") as f:
        drl_scores = pickle.load(f)

    rhos = sorted(baselines.keys())
    greedy_vals = [baselines[r]["typical_greedy"] for r in rhos]
    upper_vals = [baselines[r]["typical_upper"] for r in rhos]
    drl_vals = [drl_scores.get(r, np.nan) for r in rhos]

    plt.figure(figsize=(8, 5))
    plt.plot(rhos, greedy_vals, 'r-o', label='Greedy (typical)')
    plt.plot(rhos, upper_vals, 'k--o', label='Upper Bound (typical)')
    plt.plot(rhos, drl_vals, 'b-*', label='DRL Overfit')
    plt.xlabel('Correlation Factor (rho)')
    plt.ylabel('Sum Rate (bps/Hz)')
    plt.title('Overfit Benchmark: DRL vs Greedy vs Upper Bound')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "01_overfit_benchmark.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
