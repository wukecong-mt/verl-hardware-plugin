# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Megatron engine registration for Moore Threads MUSA."""

import logging

from verl.workers.engine.base import EngineRegistry

logger = logging.getLogger(__name__)

from verl.workers.engine.megatron.transformer_impl import (  # noqa: E402
    MegatronEngineWithLMHead,
)


@EngineRegistry.register(
    model_type="language_model",
    backend="megatron",
    device="musa",
    vendor="moore_threads",
)
class MegatronMUSAEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron engine registration for Moore Threads MUSA.

    MUSA-specific lifecycle hooks are installed by ``verl_musa_patch`` before
    this registry module is imported.
    """
