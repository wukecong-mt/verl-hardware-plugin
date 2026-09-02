# Copyright 2026
# Licensed under the Apache License, Version 2.0.

"""DRPO policy losses registered against verl's dynamic loss registry."""

from typing import Any

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, agg_loss, register_policy_loss

from .effective_ratio import compute_effective_ratio, effective_ratio_metrics


def _safe_register_policy_loss(name: str):
    """Register a loss without silently replacing another implementation."""

    def decorator(function):
        existing = POLICY_LOSS_REGISTRY.get(name)
        if existing is not None and existing is not function:
            owner = f"{existing.__module__}.{existing.__name__}"
            raise RuntimeError(f"Policy loss '{name}' is already registered by {owner}")
        return register_policy_loss(name)(function)

    return decorator


@_safe_register_policy_loss("drpo")
def compute_policy_loss_drpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute raw-ratio Divergence Regularized Policy Optimization."""

    if config is None or config.policy_loss is None:
        raise ValueError("DRPO requires config.policy_loss")
    policy_loss_config = config.policy_loss
    epsilon = float(policy_loss_config.get("drpo_epsilon", 12.5))
    if epsilon <= 0:
        raise ValueError(f"drpo_epsilon must be positive, but got {epsilon}.")
    mu_weighted = bool(policy_loss_config.get("mu_weighted", True))

    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    ratio_delta = ratio - 1.0

    advantages = advantages.detach()
    old_prob = torch.exp(old_log_prob).detach()
    penalty_weight = old_prob if mu_weighted else torch.ones_like(old_prob)
    quadratic_penalty = advantages.abs() * penalty_weight * ratio_delta.square() / (2.0 * epsilon)
    pg_losses = -advantages * ratio + quadratic_penalty

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)
    approx_kl = verl_F.masked_mean((ratio - 1.0) - log_ratio, response_mask)
    drpo_penalty = verl_F.masked_mean(quadratic_penalty, response_mask)
    if mu_weighted:
        adaptive_eps = torch.where(old_prob > 0.0, epsilon / old_prob, torch.full_like(old_prob, float("inf")))
    else:
        adaptive_eps = torch.full_like(old_prob, epsilon)

    return pg_loss, {
        "actor/pg_clipfrac": verl_F.masked_mean((ratio > (1.0 + adaptive_eps)).float(), response_mask).item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": verl_F.masked_mean((ratio < (1.0 - adaptive_eps)).float(), response_mask).item(),
        "actor/drpo_approx_kl": approx_kl.detach().item(),
        "actor/drpo_penalty": drpo_penalty.detach().item(),
        "actor/drpo_epsilon": epsilon,
        "actor/drpo_mu_weighted": float(mu_weighted),
    }


@_safe_register_policy_loss("drpo_eff")
def compute_policy_loss_drpo_eff(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute DRPO using a designed effective ratio as policy coefficient."""

    if config is None or config.policy_loss is None:
        raise ValueError("DRPO-Eff requires config.policy_loss")
    policy_loss_config = config.policy_loss
    epsilon = float(policy_loss_config.get("drpo_epsilon", 12.5))
    if epsilon <= 0:
        raise ValueError(f"drpo_epsilon must be positive, but got {epsilon}.")
    mu_weighted = bool(policy_loss_config.get("mu_weighted", True))
    min_ratio = float(policy_loss_config.get("effective_ratio_min", 0.0))
    max_ratio = float(policy_loss_config.get("effective_ratio_max", 2.0))

    effective_ratio, raw_ratio, log_ratio, z_value = compute_effective_ratio(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        metric=str(policy_loss_config.get("effective_ratio_metric", "log_ratio")),
        map_fn=str(policy_loss_config.get("effective_ratio_map", "log_tanh")),
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        beta=float(policy_loss_config.get("effective_ratio_beta", 1.0)),
        alpha=float(policy_loss_config.get("effective_ratio_alpha", 0.5)),
        lambda_=float(policy_loss_config.get("effective_ratio_lambda", 1e-4)),
        metric_clip=policy_loss_config.get("effective_ratio_metric_clip", None),
        eps=float(policy_loss_config.get("effective_ratio_eps", 1e-8)),
    )

    advantages = advantages.detach()
    old_prob = torch.exp(old_log_prob).detach()
    penalty_weight = old_prob if mu_weighted else torch.ones_like(old_prob)
    raw_ratio_delta = raw_ratio - 1.0
    effective_ratio_delta = effective_ratio - 1.0
    policy_coeff = (advantages * effective_ratio).detach()
    penalty_coeff = (advantages.abs() * penalty_weight * effective_ratio * effective_ratio_delta / epsilon).detach()
    pg_losses = -policy_coeff * log_prob + penalty_coeff * log_prob

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    quadratic_penalty = advantages.abs() * penalty_weight * effective_ratio_delta.square() / (2.0 * epsilon)
    raw_quadratic_penalty = advantages.abs() * penalty_weight * raw_ratio_delta.square() / (2.0 * epsilon)
    metrics = effective_ratio_metrics(
        effective_ratio=effective_ratio,
        raw_ratio=raw_ratio,
        z_value=z_value,
        response_mask=response_mask,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )
    result = {
        "actor/pg_clipfrac": metrics["actor/effective_ratio_clipfrac"],
        "actor/ppo_kl": verl_F.masked_mean(-log_ratio, response_mask).detach().item(),
        "actor/pg_clipfrac_lower": metrics["actor/effective_ratio_clipfrac_lower"],
        "actor/drpo_approx_kl": verl_F.masked_mean((raw_ratio - 1.0) - log_ratio, response_mask).detach().item(),
        "actor/drpo_penalty": verl_F.masked_mean(quadratic_penalty, response_mask).detach().item(),
        "actor/drpo_raw_ratio_penalty": verl_F.masked_mean(raw_quadratic_penalty, response_mask).detach().item(),
        "actor/drpo_epsilon": epsilon,
        "actor/drpo_mu_weighted": float(mu_weighted),
    }
    result.update(metrics)
    return pg_loss, result
