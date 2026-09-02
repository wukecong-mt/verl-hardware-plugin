# Copyright 2026
# Licensed under the Apache License, Version 2.0.

"""Register DRPO policy losses with verl when the plugin is imported."""

from .config import DrpoPolicyLossConfig
from .losses import compute_policy_loss_drpo, compute_policy_loss_drpo_eff

__all__ = [
    "DrpoPolicyLossConfig",
    "compute_policy_loss_drpo",
    "compute_policy_loss_drpo_eff",
]

__version__ = "0.1.0"
