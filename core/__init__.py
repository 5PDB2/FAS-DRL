"""
Core module for FAS-DRL project.

This module contains the main components for training and deployment:
- FAS_PortSelection_Env: Gym environment for Fluid Antenna Systems with Port Selection
- PPO Agent: Proximal Policy Optimization agent for DRL
"""

from .envs import FAS_PortSelection_Env
from .agent import ISAC_Agent

__all__ = ['FAS_PortSelection_Env', 'ISAC_Agent']
