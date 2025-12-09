"""
Behavior Cloning pretraining for FAS port selection.

Goal: Verify whether the current MLP (SB3-style) can map channel observations
      to greedy (PGMax) port selections. This mitigates the cold-start issue
      before PPO fine-tuning.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from core.envs import FAS_PortSelection_Env  # noqa: E402


class BCActor(nn.Module):
    """
    Simple MLP mirroring SB3's default MlpPolicy architecture (two hidden layers of 64 units).
    Outputs logits for each port (shape: N).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def generate_dataset(env, num_samples: int):
    """
    Generate (obs, label) pairs using the greedy (PGMax) expert.

    Label: binary vector of shape (N,), top-N_S ports set to 1.0.
    """
    obs_list = []
    label_list = []

    for _ in range(num_samples):
        obs, _ = env.reset()
        H = env.current_channel  # Complex channel (K, N)

        # Expert: power per port
        port_power = np.sum(np.abs(H) ** 2, axis=0)
        top_idx = np.argsort(port_power)[-env.N_S:]

        label = np.zeros(env.N, dtype=np.float32)
        label[top_idx] = 1.0

        obs_list.append(obs.astype(np.float32))
        label_list.append(label)

    obs_arr = np.stack(obs_list, axis=0)
    label_arr = np.stack(label_list, axis=0)
    return obs_arr, label_arr


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5):
    """
    Compute Top-k hit rate: fraction of true top-k ports predicted in model's top-k.
    """
    with torch.no_grad():
        pred_topk = torch.topk(logits, k, dim=1).indices
        true_topk = torch.topk(labels, k, dim=1).indices
        hits = 0
        total = 0
        for p, t in zip(pred_topk, true_topk):
            hits += len(set(p.tolist()) & set(t.tolist()))
            total += k
    return hits / total if total > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)

    # Environment for data generation (max_steps=1 for speed)
    env = FAS_PortSelection_Env(max_steps=1)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.N

    num_samples = 50_000
    print(f"Generating dataset with {num_samples} samples...")
    obs_arr, label_arr = generate_dataset(env, num_samples=num_samples)
    print("Dataset shapes:", obs_arr.shape, label_arr.shape)

    dataset = TensorDataset(
        torch.from_numpy(obs_arr), torch.from_numpy(label_arr)
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

    model = BCActor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 150
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_obs, batch_lbl in loader:
            batch_obs = batch_obs.to(device)
            batch_lbl = batch_lbl.to(device)

            optimizer.zero_grad()
            logits = model(batch_obs)
            loss = criterion(logits, batch_lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_obs.size(0)

        # Compute Top-5 accuracy on a random subset
        model.eval()
        with torch.no_grad():
            idx = np.random.choice(len(dataset), size=min(2000, len(dataset)), replace=False)
            sample_obs = torch.from_numpy(obs_arr[idx]).to(device)
            sample_lbl = torch.from_numpy(label_arr[idx]).to(device)
            sample_logits = model(sample_obs)
            acc_top5 = topk_accuracy(sample_logits, sample_lbl, k=env.N_S)

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch:02d}: loss={avg_loss:.4f}, top-{env.N_S} hit rate={acc_top5*100:.2f}%")

    # Save pretrained weights
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "bc_actor.pth"
    torch.save({"state_dict": model.state_dict(), "obs_dim": obs_dim, "action_dim": action_dim}, save_path)
    print(f"Saved BC actor weights to {save_path}")


if __name__ == "__main__":
    main()
