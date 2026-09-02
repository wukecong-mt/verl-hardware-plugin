# VERL MUSA User Guide

## Introduction

This document describes how to use verl for reinforcement learning training on
Moore Threads MUSA accelerators.

## Directory Structure

```text
verl_hardware_plugin/
├── engines
│   ├── fsdp_musa.py                  # FSDP engine support
│   └── megatron_musa.py              # Megatron engine support
└── platforms
    └── platform_musa.py              # MUSA platform settings
```

```text
user_guide_musa/
├── README.md                         # This file
├── install_guidance.md               # Installation and environment setup
└── quick_start.md                    # GSM8K GRPO quick start
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — run a GSM8K GRPO training job

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `musa` |
| Vendor identifier | `moore_threads` |
| Communication backend | `mccl` |
| Device visibility env var | `MUSA_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |

## MUSA Migration Patches

```bash
export VERL_PLATFORM=musa
export RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1
```

MUSA deployments may use two separate compatibility layers:

- The official Megatron code is adapted through the external
  `megatron-lm-musa-patch`, selected with `MUSA_PATCH_PATH` (usually
  `/home/megatron-lm-musa-patch` in the release image).
- VERL and SGLang runtime components may use an internal
  `verl-musa-patch` overlay supplied by the deployment environment. Set
  `VERL_MUSA_PATCH` to its path and add it to the workers' `PYTHONPATH`
  (usually `/home/verl-musa-patch/` in the internal release image). This
  overlay is not part of the public `verl-hardware-plugin` package.

For Megatron/MCore training, add the required external patch directories to
`PYTHONPATH`. FSDP training does not require `MUSA_PATCH_PATH`; if the internal
runtime overlay is used, pass `VERL_MUSA_PATCH` to every Ray worker. See the
[Installation Guide](./install_guidance.md) for the complete setup.
