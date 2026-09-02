# Copyright 2026
# Licensed under the Apache License, Version 2.0.

"""Effective-ratio construction and diagnostics used by DRPO-Eff."""

import math

import torch


def compute_effective_ratio(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    metric: str = "log_ratio",
    map_fn: str = "log_tanh",
    min_ratio: float = 0.0,
    max_ratio: float = 2.0,
    beta: float = 1.0,
    alpha: float = 0.5,
    lambda_: float = 1e-4,
    metric_clip: float | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map policy movement to a bounded, positive ratio-like coefficient."""

    if eps <= 0:
        raise ValueError(f"effective ratio eps must be positive, but got {eps}.")
    if min_ratio < 0:
        raise ValueError(f"effective ratio min_ratio must be non-negative, but got {min_ratio}.")
    if max_ratio <= 1.0:
        raise ValueError(f"effective ratio max_ratio must be greater than 1, but got {max_ratio}.")
    if min_ratio >= 1.0:
        raise ValueError(f"effective ratio min_ratio must be smaller than 1, but got {min_ratio}.")
    if beta <= 0:
        raise ValueError(f"effective ratio beta must be positive, but got {beta}.")

    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    raw_ratio = torch.exp(log_ratio)
    old_prob = torch.exp(old_log_prob).detach()
    prob = torch.exp(log_prob)

    if metric == "log_ratio":
        z_value = log_ratio
    elif metric == "ratio_delta":
        z_value = raw_ratio - 1.0
    elif metric == "prob_diff":
        z_value = prob - old_prob
    elif metric == "bkl_grad":
        z_value = (prob - old_prob) / torch.clamp(1.0 - prob, min=eps)
    elif metric == "fisher":
        z_value = (prob - old_prob) / torch.sqrt(old_prob * (1.0 - old_prob) + eps)
    else:
        raise ValueError(
            f"Unsupported effective_ratio_metric: {metric}. "
            "Supported values are: log_ratio, ratio_delta, prob_diff, bkl_grad, fisher."
        )

    if metric_clip is not None:
        if metric_clip <= 0:
            raise ValueError(f"effective_ratio_metric_clip must be positive when set, but got {metric_clip}.")
        z_value = torch.clamp(z_value, min=-metric_clip, max=metric_clip)

    if map_fn in ("none", "raw"):
        effective_ratio = raw_ratio
    elif map_fn == "clip":
        effective_ratio = torch.clamp(raw_ratio, min=min_ratio, max=max_ratio)
    elif map_fn == "sigmoid_bound":
        target = (1.0 - min_ratio) / (max_ratio - min_ratio)
        bias = math.log(target / (1.0 - target))
        effective_ratio = min_ratio + (max_ratio - min_ratio) * torch.sigmoid(beta * z_value + bias)
    elif map_fn == "tanh_bound":
        pos_room = max_ratio - 1.0
        neg_room = 1.0 - min_ratio
        z_pos = torch.clamp(z_value, min=0.0)
        z_neg = torch.clamp(-z_value, min=0.0)
        effective_ratio = (
            1.0 + pos_room * torch.tanh(beta * z_pos / pos_room) - neg_room * torch.tanh(beta * z_neg / neg_room)
        )
    elif map_fn == "log_tanh":
        if metric != "log_ratio":
            raise ValueError("effective_ratio_map='log_tanh' requires effective_ratio_metric='log_ratio'.")
        upper = math.log(max_ratio)
        log_ratio_pos = torch.clamp(log_ratio, min=0.0)
        mapped_log_ratio = upper * torch.tanh(beta * log_ratio_pos / upper)
        if min_ratio > 0:
            lower = -math.log(min_ratio)
            log_ratio_neg = torch.clamp(-log_ratio, min=0.0)
            mapped_log_ratio = mapped_log_ratio - lower * torch.tanh(beta * log_ratio_neg / lower)
        else:
            mapped_log_ratio = mapped_log_ratio + torch.clamp(log_ratio, max=0.0)
        effective_ratio = torch.exp(mapped_log_ratio)
    elif map_fn == "power":
        if metric != "log_ratio":
            raise ValueError("effective_ratio_map='power' requires effective_ratio_metric='log_ratio'.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"effective_ratio_alpha must be in (0, 1], but got {alpha}.")
        effective_ratio = torch.clamp(torch.exp(alpha * log_ratio), min=min_ratio, max=max_ratio)
    elif map_fn == "mass_power":
        if metric != "log_ratio":
            raise ValueError("effective_ratio_map='mass_power' requires effective_ratio_metric='log_ratio'.")
        if lambda_ <= 0:
            raise ValueError(f"effective_ratio_lambda must be positive, but got {lambda_}.")
        mass_alpha = old_prob / (old_prob + lambda_)
        effective_ratio = torch.clamp(torch.exp(mass_alpha * log_ratio), min=min_ratio, max=max_ratio)
    elif map_fn == "residual_sech":
        if metric != "log_ratio":
            raise ValueError("effective_ratio_map='residual_sech' requires effective_ratio_metric='log_ratio'.")
        sech_sq = 1.0 / torch.cosh(log_ratio / beta).square()
        effective_ratio = torch.clamp(1.0 + (raw_ratio - 1.0) * sech_sq, min=min_ratio, max=max_ratio)
    else:
        raise ValueError(
            f"Unsupported effective_ratio_map: {map_fn}. Supported values are: "
            "none, clip, sigmoid_bound, tanh_bound, log_tanh, power, mass_power, residual_sech."
        )

    return effective_ratio, raw_ratio, log_ratio, z_value


def effective_ratio_metrics(
    effective_ratio: torch.Tensor,
    raw_ratio: torch.Tensor,
    z_value: torch.Tensor,
    response_mask: torch.Tensor,
    min_ratio: float,
    max_ratio: float,
) -> dict[str, float]:
    """Return detached diagnostics over valid response tokens."""

    valid = response_mask > 0
    if not valid.any():
        return {
            "actor/effective_ratio_mean": 0.0,
            "actor/effective_ratio_min": 0.0,
            "actor/effective_ratio_max": 0.0,
            "actor/effective_ratio_p95": 0.0,
            "actor/effective_ratio_clipfrac": 0.0,
            "actor/effective_ratio_clipfrac_lower": 0.0,
            "actor/effective_ratio_suppression_frac": 0.0,
            "actor/raw_ratio_mean": 0.0,
            "actor/raw_ratio_max": 0.0,
            "actor/effective_ratio_metric_abs_mean": 0.0,
            "actor/effective_ratio_metric_abs_max": 0.0,
        }

    eff_valid = effective_ratio.detach()[valid]
    raw_valid = raw_ratio.detach()[valid]
    z_abs_valid = z_value.detach().abs()[valid]
    tolerance = 1e-6
    metrics = {
        "actor/effective_ratio_mean": eff_valid.mean().item(),
        "actor/effective_ratio_min": eff_valid.min().item(),
        "actor/effective_ratio_max": eff_valid.max().item(),
        "actor/effective_ratio_p95": torch.quantile(eff_valid.float(), 0.95).item(),
        "actor/effective_ratio_clipfrac": (raw_valid > max_ratio).float().mean().item(),
        "actor/effective_ratio_suppression_frac": (
            ((raw_valid > 1.0) & (eff_valid < raw_valid - tolerance)).float().mean().item()
        ),
        "actor/raw_ratio_mean": raw_valid.mean().item(),
        "actor/raw_ratio_max": raw_valid.max().item(),
        "actor/effective_ratio_metric_abs_mean": z_abs_valid.mean().item(),
        "actor/effective_ratio_metric_abs_max": z_abs_valid.max().item(),
    }
    metrics["actor/effective_ratio_clipfrac_lower"] = (
        (raw_valid < min_ratio).float().mean().item() if min_ratio > 0 else 0.0
    )
    return metrics
