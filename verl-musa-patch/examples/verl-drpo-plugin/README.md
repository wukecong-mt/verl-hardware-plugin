# verl-drpo-plugin

An external policy-loss plugin for verl 0.9. It registers `drpo` and
`drpo_eff` without modifying the verl source tree or coupling the algorithms to
a hardware platform plugin.

## Install

Install the target verl 0.9 environment first, then install this package:

```bash
python -m pip install --no-deps -e /path/to/verl-drpo-plugin
```

verl discovers the `drpo` entry point automatically. For training, explicit
loading is recommended so an import failure is fatal instead of being reduced
to an entry-point debug message:

```bash
export VERL_USE_EXTERNAL_MODULES=verl_drpo_plugin
export VERL_USE_EXTERNAL_PLUGINS=hardware,drpo
```

The plugin must be installed, or otherwise importable, in the Python environment
used by the driver and every Ray worker.

## Configure

Replace the nested policy-loss target so Hydra accepts the DRPO-specific fields:

```text
actor_rollout_ref.actor.policy_loss._target_=verl_drpo_plugin.config.DrpoPolicyLossConfig
actor_rollout_ref.actor.policy_loss.loss_mode=drpo_eff
+actor_rollout_ref.actor.policy_loss.drpo_epsilon=12.5
+actor_rollout_ref.actor.policy_loss.mu_weighted=true
+actor_rollout_ref.actor.policy_loss.effective_ratio_metric=ratio_delta
+actor_rollout_ref.actor.policy_loss.effective_ratio_map=clip
+actor_rollout_ref.actor.policy_loss.effective_ratio_min=0.1
+actor_rollout_ref.actor.policy_loss.effective_ratio_max=2.0
```

Use `loss_mode=drpo` for raw-ratio DRPO. Effective-ratio-only settings are
ignored by `drpo`.

## Compatibility

This release targets verl `>=0.9.0,<0.10`. It uses verl's public dynamic
policy-loss registry and the policy-loss callable contract present in 0.9.0.
The package intentionally refuses to overwrite an existing `drpo` or
`drpo_eff` registration from another implementation.

## FSDP2 + SGLang launcher

`scripts/start_fsdp2_sglang_drpo.sh` is a single-node MUSA launcher derived
from the verl 0.8 S5000 FSDP2+SGLang recipe and adapted to verl 0.9's platform,
plugin, and Hydra config contracts.

Resolve and inspect the configuration without starting a GPU workload:

```bash
DRY_RUN=cfg \
VERL_PLATFORM=nvidia \
SKIP_PLATFORM_CHECK=1 \
bash scripts/start_fsdp2_sglang_drpo.sh
```

The `VERL_PLATFORM=nvidia` override above is only for CPU config composition.
Do not use it for a MUSA training run. A real run defaults to `VERL_PLATFORM=musa`
and requires the installed hardware plugin to have registered that platform.

Start the default raw-ratio DRPO run in the background:

```bash
bash scripts/start_fsdp2_sglang_drpo.sh
```

Use `LAUNCH_MODE=foreground` for an attached process. The launcher accepts
additional Hydra overrides as positional arguments. All paths and main training
parameters can also be overridden through environment variables documented in
the script's `user-adjustable` section.
