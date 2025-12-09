"""
Step 1: Generate channel datasets for multiple correlation factors.

Saves a dict mapping rho -> list of complex channel matrices to experiments/data/channels_dataset.pkl
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.channel import generate_channel  # noqa: E402
from docs.basic_funcs import N, K, NX, NY, LAMBDA  # noqa: E402


def main():
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    num_channels = 100
    side = int(np.sqrt(N))
    spacing = 0.5 * LAMBDA
    wx = (side - 1) * spacing
    wy = (side - 1) * spacing

    dataset = {}
    for rho in rhos:
        channels = []
        for _ in range(num_channels):
            H = generate_channel(
                K=K,
                N=N,
                Nx=NX,
                Ny=NY,
                Wx=wx,
                Wy=wy,
                wavelength=LAMBDA,
                user_correlation_factor=rho,
            )
            channels.append(H)
        dataset[rho] = channels
        print(f"Generated {num_channels} channels for rho={rho}")

    out_dir = ROOT / "experiments" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "channels_dataset.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
