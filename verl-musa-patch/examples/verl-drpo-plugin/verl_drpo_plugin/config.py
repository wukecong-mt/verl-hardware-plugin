# Copyright 2026
# Licensed under the Apache License, Version 2.0.

"""Hydra-compatible policy-loss configuration for DRPO."""

from dataclasses import dataclass

from verl.workers.config.actor import PolicyLossConfig


@dataclass
class DrpoPolicyLossConfig(PolicyLossConfig):
    """Extend verl 0.9's policy-loss config with DRPO parameters."""

    drpo_epsilon: float = 12.5
    mu_weighted: bool = True
    effective_ratio_metric: str = "log_ratio"
    effective_ratio_map: str = "log_tanh"
    effective_ratio_metric_clip: float | None = None
    effective_ratio_min: float = 0.0
    effective_ratio_max: float = 2.0
    effective_ratio_beta: float = 1.0
    effective_ratio_alpha: float = 0.5
    effective_ratio_lambda: float = 1e-4
    effective_ratio_eps: float = 1e-8
