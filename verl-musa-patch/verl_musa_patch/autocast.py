"""Restore Torch-MUSA autocast state APIs after TransformerEngine patching."""


def install() -> None:
    import torch
    from torch_musa.core.amp.common import _musa_is_autocast_enabled

    # transformer_engine.musa replaces this process-global API with a function
    # that always returns False. PyTorch activation checkpointing relies on the
    # real value to replay the forward autocast context during recomputation.
    torch.is_autocast_enabled = _musa_is_autocast_enabled
