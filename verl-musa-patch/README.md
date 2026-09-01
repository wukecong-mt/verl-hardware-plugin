# Internal VERL MUSA Migration

This directory is an internal runtime overlay. It is not part of the public
hardware-plugin API and is loaded through `sitecustomize.py` before importing
VERL.

```bash
export VERL_PLATFORM=musa
export VERL_MUSA_PATCH=/path/to/verl-hardware-plugin/verl-musa-patch
export MUSA_PATCH_PATH=/home/megatron-lm-musa-patch
export PYTHONPATH="${VERL_MUSA_PATCH}:${MUSA_PATCH_PATH}:${PYTHONPATH}"
```

The Megatron MUSA patch is provided externally by the runtime environment; it
is intentionally not copied into this directory.
