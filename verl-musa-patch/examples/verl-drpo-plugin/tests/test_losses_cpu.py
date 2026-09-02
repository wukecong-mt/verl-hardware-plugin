import pytest
import torch
from verl.workers.config.actor import ActorConfig

from verl_drpo_plugin import DrpoPolicyLossConfig, compute_policy_loss_drpo, compute_policy_loss_drpo_eff
from verl_drpo_plugin.effective_ratio import compute_effective_ratio


def _actor_config(loss_mode: str, **kwargs) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        loss_agg_mode="token-mean",
        policy_loss=DrpoPolicyLossConfig(loss_mode=loss_mode, **kwargs),
    )


@pytest.mark.parametrize("mu_weighted", [True, False])
def test_drpo_matches_direct_value_and_gradient_formula(mu_weighted):
    old_log_prob = torch.tensor([[-1.2, -0.7, -0.3], [-0.9, -1.5, -0.4]], dtype=torch.float64)
    log_prob = torch.tensor([[-1.0, -0.9, -0.1], [-1.1, -1.0, -0.8]], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[1.5, -0.5, 9.0], [-2.0, 0.7, 4.0]], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    weights = torch.tensor([[1.0, 0.5, 7.0], [1.5, 2.0, 3.0]], dtype=torch.float64)
    epsilon = 3.25
    config = _actor_config("drpo", drpo_epsilon=epsilon, mu_weighted=mu_weighted)

    actual_loss, metrics = compute_policy_loss_drpo(
        old_log_prob, log_prob, advantages, mask, "token-mean", config, weights
    )
    actual_grad = torch.autograd.grad(actual_loss, log_prob, retain_graph=True)[0]

    log_ratio = torch.clamp(log_prob - old_log_prob, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    penalty_weight = torch.exp(old_log_prob) if mu_weighted else torch.ones_like(old_log_prob)
    penalty = advantages.abs() * penalty_weight * (ratio - 1.0).square() / (2.0 * epsilon)
    expected_loss = (((-advantages * ratio + penalty) * weights) * mask).sum() / mask.sum()
    expected_grad = torch.autograd.grad(expected_loss, log_prob)[0]

    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert metrics["actor/drpo_epsilon"] == epsilon
    assert metrics["actor/drpo_mu_weighted"] == float(mu_weighted)


def test_drpo_eff_clip_matches_designed_coefficient_and_gradient():
    old_log_prob = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float64)
    log_prob = torch.tensor([[-3.0, -1.2, -0.6, 0.2]], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([[1.0, -2.0, 0.5, -1.5]], dtype=torch.float64)
    mask = torch.ones_like(old_log_prob, dtype=torch.bool)
    epsilon = 4.0
    config = _actor_config(
        "drpo_eff",
        drpo_epsilon=epsilon,
        mu_weighted=True,
        effective_ratio_metric="ratio_delta",
        effective_ratio_map="clip",
        effective_ratio_min=0.25,
        effective_ratio_max=2.0,
    )

    actual_loss, metrics = compute_policy_loss_drpo_eff(old_log_prob, log_prob, advantages, mask, "token-mean", config)
    actual_grad = torch.autograd.grad(actual_loss, log_prob)[0]

    raw_ratio = torch.exp(torch.clamp(log_prob.detach() - old_log_prob, -20.0, 20.0))
    effective_ratio = torch.clamp(raw_ratio, 0.25, 2.0)
    old_prob = torch.exp(old_log_prob)
    policy_coeff = advantages * effective_ratio
    penalty_coeff = advantages.abs() * old_prob * effective_ratio * (effective_ratio - 1.0) / epsilon
    expected_loss = ((-policy_coeff + penalty_coeff) * log_prob).mean()
    expected_grad = (-policy_coeff + penalty_coeff) / mask.sum()

    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert metrics["actor/effective_ratio_min"] == pytest.approx(0.25)
    assert metrics["actor/effective_ratio_max"] == pytest.approx(2.0)
    assert metrics["actor/effective_ratio_clipfrac"] == pytest.approx(0.25)
    assert metrics["actor/effective_ratio_clipfrac_lower"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("map_fn", "metric"),
    [
        ("none", "log_ratio"),
        ("clip", "ratio_delta"),
        ("sigmoid_bound", "prob_diff"),
        ("tanh_bound", "fisher"),
        ("log_tanh", "log_ratio"),
        ("power", "log_ratio"),
        ("mass_power", "log_ratio"),
        ("residual_sech", "log_ratio"),
    ],
)
def test_all_effective_ratio_maps_are_finite_on_cpu(map_fn, metric):
    old_log_prob = torch.tensor([[-8.0, -2.0, -0.7, -0.1]], dtype=torch.float64)
    log_prob = torch.tensor([[-1.0, -3.0, -0.2, -1.5]], dtype=torch.float64)

    effective, raw, log_ratio, z_value = compute_effective_ratio(
        old_log_prob,
        log_prob,
        map_fn=map_fn,
        metric=metric,
        min_ratio=0.1,
        max_ratio=2.0,
        metric_clip=10.0,
    )

    assert torch.isfinite(effective).all()
    assert torch.isfinite(raw).all()
    assert torch.isfinite(log_ratio).all()
    assert torch.isfinite(z_value).all()
    if map_fn not in ("none", "raw"):
        assert effective.min() >= 0.1
        assert effective.max() <= 2.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_ratio": 1.0}, "max_ratio"),
        ({"min_ratio": 1.0}, "min_ratio"),
        ({"eps": 0.0}, "eps"),
        ({"map_fn": "power", "alpha": 0.0}, "alpha"),
        ({"map_fn": "missing"}, "Unsupported effective_ratio_map"),
    ],
)
def test_effective_ratio_rejects_invalid_configuration(kwargs, message):
    old_log_prob = torch.tensor([[-1.0]])
    log_prob = torch.tensor([[-0.9]])
    with pytest.raises(ValueError, match=message):
        compute_effective_ratio(old_log_prob, log_prob, **kwargs)


def test_drpo_requires_positive_epsilon():
    config = _actor_config("drpo", drpo_epsilon=0.0)
    tensor = torch.tensor([[-1.0]])
    with pytest.raises(ValueError, match="drpo_epsilon"):
        compute_policy_loss_drpo(tensor, tensor, tensor, torch.ones_like(tensor), config=config)
