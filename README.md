# Fluid Antenna Systems for ISAC via Deep Reinforcement Learning

## Project Overview

This research project develops a **Deep Reinforcement Learning (DRL)** framework for optimizing **Fluid Antenna Systems (FAS)** in an **Integrated Sensing and Communication (ISAC)** scenario. The project leverages Proximal Policy Optimization (PPO) to dynamically control antenna element positions, maximizing both communication throughput and sensing accuracy in dynamic wireless environments.

## Mathematical Problem: FAS-ISAC

### System Model

Consider a base station equipped with a **Fluid Antenna System** with $N$ antenna elements that can be continuously repositioned along a linear aperture. The system must simultaneously serve:

- **Communication**: $K$ downlink users with channel vectors $\mathbf{h}_k \in \mathbb{C}^{N \times 1}$
- **Sensing**: $M$ radar observations for target detection and localization

The transmitted signal is:
$$\mathbf{x}(t) = \sum_{k=1}^{K} \mathbf{w}_k \, s_k(t)$$

where $\mathbf{w}_k$ is the precoding vector for user $k$ and $s_k(t)$ is the data symbol.

### Performance Objectives

**Communication Rate** (Shannon Capacity):
$$C_k = B \log_2\left(1 + \frac{\left|\mathbf{w}_k^H \mathbf{h}_k\right|^2}{\sum_{j \neq k} \left|\mathbf{w}_k^H \mathbf{h}_j\right|^2 + \sigma^2}\right)$$

**Sensing Performance** (e.g., Detection Probability):
$$P_d = Q\left(\sqrt{\frac{2 E_s}{N_0}} \left|\mathbf{w}_{\text{radar}}^H \mathbf{H}_{\text{target}}\right|^2\right)$$

### Fluid Antenna Positioning

The antenna positions $\mathbf{p} = [p_1, p_2, \ldots, p_N]$ directly affect channel responses through the steering vector:
$$\mathbf{a}(\theta, \mathbf{p}) = \left[\, e^{j 2\pi p_1 \sin\theta / \lambda}, \ldots, e^{j 2\pi p_N \sin\theta / \lambda} \,\right]^T$$

The DRL agent learns an optimal policy $\pi_\theta(\mathbf{p} | \mathbf{s})$ to adjust antenna positions based on channel state information (CSI), maximizing a combined reward:
$$R(t) = \alpha \cdot C_{\text{comm}}(t) + (1-\alpha) \cdot P_{\text{sense}}(t)$$

## Project Structure

```
FAS-DRL/
├── core/                   # Core RL components
│   ├── __init__.py
│   ├── envs.py            # FAS_ISAC_Env: Gym environment
│   └── agent.py           # PPO_Agent: Proximal Policy Optimization
│
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── channel.py         # Jakes' Model channel simulation
│   └── math_ops.py        # Beamforming & matrix operations
│
├── docs/                   # Documentation & global configs
│   ├── __init__.py
│   └── basic_funcs.py     # System parameters (N, K, SNR, etc.)
│
├── experiments/            # Training scripts
│   ├── __init__.py
│   └── train.py           # Main training loop
│
├── test/                   # Evaluation & testing
│   ├── __init__.py
│   └── evaluate.py        # Model evaluation & metrics
│
├── results/                # Output directory (plots, data)
│   └── (empty, to be populated during experiments)
│
└── README.md              # This file
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| **core/** | Contains the RL environment (`FAS_ISAC_Env`) implementing the Gym interface and the PPO agent (`PPO_Agent`) for policy optimization. |
| **utils/** | Implements wireless channel models (`Jakes' Model`), beamforming calculations, and matrix operations for signal processing. |
| **docs/** | Stores global system parameters (number of antennas $N$, users $K$, SNR, carrier frequency, etc.) and unit conversion utilities. |
| **experiments/** | Contains the main training loop (`train.py`) for executing DRL experiments with configurable hyperparameters. |
| **test/** | Implements model evaluation functions (`evaluate.py`) to assess trained agent performance, generate metrics, and create visualization plots. |
| **results/** | Directory for storing output files: training curves, model checkpoints, performance metrics, and plots. |

## Key Features

### 1. **Gym-Compatible Environment**
   - State space: Channel state information, antenna positions, performance metrics
   - Action space: Continuous antenna position adjustments
   - Reward function: Weighted combination of communication and sensing metrics
   - Realistic channel dynamics via Jakes' model

### 2. **Proximal Policy Optimization (PPO)**
   - Actor-Critic architecture with Generalized Advantage Estimation (GAE)
   - Clipped objective to stabilize training
   - Entropy regularization for exploration

### 3. **Wireless Channel Simulation**
   - Jakes' model for time-varying fading channels
   - Support for multiple users and sensing targets
   - CSI acquisition and feedback mechanisms

### 4. **Performance Metrics**
   - Communication capacity (bits/s/Hz)
   - Sensing detection probability and localization error
   - Training convergence analysis
   - Robustness evaluation

## Installation & Setup

### Requirements
- Python 3.8+
- NumPy, SciPy (numerical computing)
- PyTorch (deep learning)
- Gym (RL environment interface)
- Matplotlib, Seaborn (visualization)

### Installation Steps

```bash
# Clone or navigate to project directory
cd FAS-DRL

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy torch gym matplotlib seaborn
```

## Getting Started

### Training the Agent

```python
from experiments.train import train_agent
from utils.basic_funcs import load_config

# Load configuration
config = load_config('config.yaml')

# Train the agent
agent, training_history = train_agent(config)

# Evaluate
from test.evaluate import evaluate_agent
metrics = evaluate_agent(agent, config)
```

### Project Timeline

1. **Phase 1**: Implement core environment (`envs.py`) and channel model (`channel.py`)
2. **Phase 2**: Develop PPO agent (`agent.py`) with policy and value networks
3. **Phase 3**: Implement training loop (`train.py`) with logging and checkpointing
4. **Phase 4**: Evaluation and visualization (`evaluate.py`)
5. **Phase 5**: Experiments and results analysis

## File Descriptions

| File | Purpose |
|------|---------|
| `core/envs.py` | Gym environment for FAS-ISAC system with state, action, and reward definitions |
| `core/agent.py` | PPO agent with actor-critic networks and training methods |
| `utils/channel.py` | Jakes' model implementation for realistic channel simulation |
| `utils/math_ops.py` | Beamforming, precoding, and signal processing utilities |
| `docs/basic_funcs.py` | Global system parameters and unit conversion functions |
| `experiments/train.py` | Main training loop with trajectory collection and network updates |
| `test/evaluate.py` | Model evaluation, metrics computation, and visualization |

## System Parameters

Default parameters (defined in `docs/basic_funcs.py`):

```
N = 16              # Number of fluid antenna elements
K = 4               # Number of communication users
M = 8               # Number of sensing observations per slot
SNR = 20            # Signal-to-noise ratio (dB)
f_c = 28e9          # Carrier frequency (28 GHz mmWave)
lambda = c / f_c    # Wavelength
v = 5               # Velocity for Doppler modeling (m/s)
```

## References

- Jakes, W. C. (1974). "Microwave Mobile Communications." John Wiley & Sons.
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*
- Liu, F., et al. (2022). "Integrated Sensing and Communication: Toward Dual-Functional Wireless Networks." *IEEE Commun. Mag.*
- Schultze, T., et al. (2023). "Fluid Antenna Systems: Concept and Implementation." *IEEE Commun. Surv. Tutor.*

## Author

Research Project for IOTA 5201: Final Project
Hong Kong University of Science and Technology (HKUST-GZ)

## License

This project is for academic research purposes. Please cite appropriately if used in publications.

---

**Last Updated**: December 2025
