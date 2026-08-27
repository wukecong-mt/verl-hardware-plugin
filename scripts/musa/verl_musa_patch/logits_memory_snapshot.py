"""Optional MUSA allocator snapshot at the Megatron lm-head boundary.

This is disabled unless VERL_MUSA_LOGITS_SNAPSHOT=1.  It wraps the external
VERL Megatron engine only; the VERL checkout itself is not modified.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path

_PHASE = ContextVar("verl_musa_logits_snapshot_phase", default=None)


def _install_phase_markers() -> None:
    from verl.workers.engine.base import BaseEngine

    if getattr(BaseEngine, "_musa_snapshot_phase_wrapped", False):
        return
    original_train = BaseEngine.train_batch
    original_infer = BaseEngine.infer_batch

    def train(self, *args, **kwargs):
        token = _PHASE.set("update_actor")
        try:
            return original_train(self, *args, **kwargs)
        finally:
            _PHASE.reset(token)

    def infer(self, *args, **kwargs):
        token = _PHASE.set("compute_log_prob")
        try:
            return original_infer(self, *args, **kwargs)
        finally:
            _PHASE.reset(token)

    BaseEngine.train_batch = train
    BaseEngine.infer_batch = infer
    BaseEngine._musa_snapshot_phase_wrapped = True


def install() -> None:
    if os.getenv("VERL_MUSA_LOGITS_SNAPSHOT", "0") != "1":
        return

    import torch

    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    _install_phase_markers()

    # MUSA: allocator snapshots only contain call stacks when allocation
    # history is enabled before model execution starts.  Keep this opt-in so
    # normal VERL runs do not pay the tracing overhead.
    record_history = os.getenv("VERL_MUSA_LOGITS_RECORD_HISTORY", "0") == "1"
    if record_history:
        trace_entries = int(os.getenv("VERL_MUSA_LOGITS_TRACE_ALLOC_MAX_ENTRIES", "200000"))
        try:
            try:
                torch.musa.memory._record_memory_history(
                    enabled="all", trace_alloc_max_entries=trace_entries
                )
            except TypeError:
                # Some torch_musa builds expose the history API but do not
                # expose PyTorch's trace_alloc_max_entries keyword.
                torch.musa.memory._record_memory_history(enabled="all")
            print(
                "[musa_logits_snapshot] allocator history enabled, "
                f"trace_alloc_max_entries={trace_entries}"
            )
        except Exception as exc:
            # Keep the diagnostic hook usable across torch_musa versions whose
            # private allocator API has a different signature.
            print(f"[musa_logits_snapshot] allocator history unavailable: {exc}")

    if getattr(MegatronEngineWithLMHead, "_musa_logits_snapshot_wrapped", False):
        return

    wanted = {
        int(x) for x in os.getenv("VERL_MUSA_LOGITS_SNAPSHOT_RANKS", "").split(",") if x.strip()
    }
    output_dir = Path(os.getenv("VERL_MUSA_LOGITS_SNAPSHOT_DIR", "/tmp/verl_musa_logits_snapshot"))
    capture_mode = os.getenv("VERL_MUSA_LOGITS_SNAPSHOT_MODE", "all")
    original = MegatronEngineWithLMHead._lm_head_logits_processor
    captured = False

    def wrapped(self, logits, *args, **kwargs):
        nonlocal captured
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if wanted and rank not in wanted:
            return original(self, logits, *args, **kwargs)
        phase = _PHASE.get()
        if capture_mode == "update_actor" and phase != "update_actor":
            return original(self, logits, *args, **kwargs)
        if capture_mode == "old_log_prob" and phase != "compute_log_prob":
            return original(self, logits, *args, **kwargs)
        if not captured and logits.device.type == "musa":
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.musa.synchronize()
            torch.musa.memory._dump_snapshot(
                str(output_dir / f"rank{rank}_{capture_mode}_before_logits_processor.pickle")
            )
            print(
                "[musa_logits_snapshot] pre-processor snapshot saved: "
                f"rank={rank}, logits_shape={tuple(logits.shape)}, device={logits.device}"
            )
            captured = True
            result = original(self, logits, *args, **kwargs)
            torch.musa.synchronize()
            torch.musa.memory._dump_snapshot(
                str(output_dir / f"rank{rank}_{capture_mode}_after_logits_processor.pickle")
            )
            print(f"[musa_logits_snapshot] post-processor snapshot saved: rank={rank}")
            return result
        return original(self, logits, *args, **kwargs)

    MegatronEngineWithLMHead._lm_head_logits_processor = wrapped
    MegatronEngineWithLMHead._musa_logits_snapshot_wrapped = True
    print(f"[musa_logits_snapshot] hook installed, ranks={sorted(wanted) or 'all'}, dir={output_dir}")
