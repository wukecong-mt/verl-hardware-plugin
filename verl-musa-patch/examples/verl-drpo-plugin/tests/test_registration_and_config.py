from hydra.utils import instantiate
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import get_policy_loss_fn

import verl_drpo_plugin


def test_plugin_registers_both_losses():
    assert get_policy_loss_fn("drpo") is verl_drpo_plugin.compute_policy_loss_drpo
    assert get_policy_loss_fn("drpo_eff") is verl_drpo_plugin.compute_policy_loss_drpo_eff


def test_nested_hydra_config_instantiates_plugin_config():
    config = OmegaConf.create(
        {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "rollout_n": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "policy_loss": {
                "_target_": "verl_drpo_plugin.config.DrpoPolicyLossConfig",
                "loss_mode": "drpo_eff",
                "drpo_epsilon": 7.5,
                "effective_ratio_map": "clip",
                "effective_ratio_min": 0.1,
                "effective_ratio_max": 2.0,
            },
        }
    )

    actor_config = instantiate(config, _convert_="partial")

    assert isinstance(actor_config.policy_loss, verl_drpo_plugin.DrpoPolicyLossConfig)
    assert actor_config.policy_loss.loss_mode == "drpo_eff"
    assert actor_config.policy_loss.drpo_epsilon == 7.5
