"""MUSA compatibility patches for VERL's SGLang rollout adapter."""

import os
from functools import wraps

_DEFAULT_HTTP_TIMEOUT = 600.0


def _restore_musa_device_capability() -> None:
    """Expose the real MUSA capability to SGLang's backend registry.

    The external Megatron compatibility patch intentionally makes CUDA-facing
    training libraries see a synthetic CUDA capability.  SGLang's MUSA FA3
    registry, however, interprets this value as an MP capability and requires
    MP >= 31.  Scheduler processes must therefore use the native MUSA query.
    """
    import torch

    if not hasattr(torch, "musa"):
        return

    native_get_device_capability = getattr(torch.musa, "get_device_capability", None)
    if native_get_device_capability is None:
        return

    torch.cuda.get_device_capability = native_get_device_capability


def _http_timeout() -> float:
    value = float(os.getenv("VERL_SGLANG_HTTP_TIMEOUT", str(_DEFAULT_HTTP_TIMEOUT)))
    if value <= 0:
        raise ValueError("VERL_SGLANG_HTTP_TIMEOUT must be greater than zero")
    return value


def _install_http_timeout() -> None:
    """Use a MUSA-safe timeout without editing VERL's HTTP adapter source."""
    from verl.workers.rollout.sglang_rollout import http_server_engine

    timeout = _http_timeout()
    http_server_engine.DEFAULT_TIMEOUT = timeout

    for adapter_cls in (
        http_server_engine.HttpServerAdapter,
        http_server_engine.AsyncHttpServerAdapter,
    ):
        original_init = adapter_cls.__init__
        if getattr(original_init, "_verl_musa_http_timeout", False):
            continue

        @wraps(original_init)
        def wrapped_init(self, *args, __original_init=original_init, **kwargs):
            # timeout is the third constructor argument after router_ip and
            # router_port.  Preserve an explicit caller value.
            if len(args) < 3 and "timeout" not in kwargs:
                kwargs["timeout"] = _http_timeout()
            return __original_init(self, *args, **kwargs)

        wrapped_init._verl_musa_http_timeout = True
        adapter_cls.__init__ = wrapped_init

    original_async_request = http_server_engine.AsyncHttpServerAdapter._make_async_request
    if not getattr(original_async_request, "_verl_musa_http_timeout", False):

        @wraps(original_async_request)
        async def wrapped_async_request(self, *args, **kwargs):
            # timeout is the fourth positional argument after endpoint,
            # payload, and method.  The upstream default was bound to 60s at
            # function definition time, so changing DEFAULT_TIMEOUT alone is
            # insufficient.
            if len(args) < 4 and "timeout" not in kwargs:
                kwargs["timeout"] = self.timeout
            return await original_async_request(self, *args, **kwargs)

        wrapped_async_request._verl_musa_http_timeout = True
        http_server_engine.AsyncHttpServerAdapter._make_async_request = wrapped_async_request


def _install_weight_sync_barrier() -> None:
    """Keep MUSA IPC buffers alive until all inference TP ranks finish sync."""
    import torch

    from verl.workers.rollout.sglang_rollout import sglang_rollout

    if getattr(sglang_rollout, "_MUSA_WEIGHT_SYNC_PATCHED", False):
        return

    original_update_weights = sglang_rollout.sgl_update_weights

    @wraps(original_update_weights)
    async def wrapped_update_weights(*args, **kwargs):
        result = await original_update_weights(*args, **kwargs)
        device_mesh = kwargs["device_mesh"]
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=device_mesh["infer_tp"].get_group())
        return result

    sglang_rollout.sgl_update_weights = wrapped_update_weights
    sglang_rollout._MUSA_WEIGHT_SYNC_PATCHED = True


def _install_mtp_ipc_tensor_cache() -> None:
    """Materialize each MUSA IPC tensor once when MTP loads draft and target."""
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor

    original_get = LocalSerializedTensor.get
    if getattr(original_get, "_verl_musa_mtp_ipc_cache", False):
        return

    @wraps(original_get)
    def cached_get(self, rank):
        # EAGLE passes the same LocalSerializedTensor first to the draft model
        # and then to the target model.  Opening the same MUSA IPC payload twice
        # leaves the producer storage pinned after the request completes.  Keep
        # one received tensor and share it between both loaders.
        cache = getattr(self, "_verl_musa_materialized", None)
        if cache is None:
            cache = self._verl_musa_materialized = {}
        if rank in cache:
            # The second call is the target loader.  Transfer the cached
            # tensor out of the request-owned cache so its IPC mapping is
            # released as soon as the target loader returns.  Keeping it in
            # the cache pins the producer storage for the lifetime of the
            # scheduler request and makes every update retain another model.
            return cache.pop(rank)
        tensor = original_get(self, rank)
        cache[rank] = tensor
        return tensor

    cached_get._verl_musa_mtp_ipc_cache = True
    LocalSerializedTensor.get = cached_get


def install() -> None:
    _install_http_timeout()
    _install_weight_sync_barrier()
    _install_mtp_ipc_tensor_cache()
