"""MUSA-specific hooks for VERL's Megatron engine.

This module is intentionally kept separate from the engine registration class.
The registration class should only select the MUSA engine; this module owns the
MUSA workarounds that alter Megatron engine construction or loss handling.
"""

import os


_INSTALLED = False


def _install_qwen35_vl_pre_wrap(original_build):
    """Wrap module construction for the Qwen3.5-VL TE LayerNorm workaround."""

    def build_megatron_module(self):
        provider_names = {"Qwen35VLModelProvider", "Qwen35VLMoEModelProvider"}
        false_values = {"0", "false", "no", "off", "disable", "disabled"}
        replace_flag = os.getenv("QWEN3_VL_REPLACE_TE_LAYER_NORMS", "1").strip().lower()
        needs_patch = (
            self.bridge is not None
            and self.provider is not None
            and not self.vanilla_bridge
            and not self.engine_config.use_dist_checkpointing
            and replace_flag not in false_values
            and type(self.provider).__name__ in provider_names
        )
        if not needs_patch:
            return original_build(self)

        import torch
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
            replace_qwen3_vl_vision_te_layernorms,
        )

        original_load_hf_weights = self.bridge.load_hf_weights
        load_completed = False

        def load_hf_weights_once(model, *args, **kwargs):
            nonlocal load_completed
            if load_completed:
                return None
            result = original_load_hf_weights(model, *args, **kwargs)
            load_completed = True
            return result

        def musa_qwen35_vl_pre_wrap(model):
            allowed_mismatched_params = ["output_layer.weight"] if self.is_value_model else []
            load_hf_weights_once(
                model,
                self.model_config.local_path,
                allowed_mismatched_params=allowed_mismatched_params,
            )
            replaced_count = replace_qwen3_vl_vision_te_layernorms(model)

            freeze_language = bool(getattr(self.provider, "freeze_language_model", False))
            freeze_vision = bool(getattr(self.provider, "freeze_vision_model", False))
            freeze_projection = bool(getattr(self.provider, "freeze_vision_projection", False))
            if freeze_language or freeze_vision or freeze_projection:
                model_chunks = model if isinstance(model, (list, tuple)) else [model]
                for model_chunk in model_chunks:
                    if hasattr(model_chunk, "freeze"):
                        model_chunk.freeze(
                            freeze_language_model=freeze_language,
                            freeze_vision_model=freeze_vision,
                            freeze_vision_projection=freeze_projection,
                        )

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            import logging

            logging.getLogger(__name__).info(
                "MUSA Qwen3.5-VL pre-wrap replaced %d vision TE modules", replaced_count
            )
            return model

        self.provider.register_pre_wrap_hook(musa_qwen35_vl_pre_wrap, prepend=True)
        bridge_dict = getattr(self.bridge, "__dict__", {})
        had_instance_loader = "load_hf_weights" in bridge_dict
        previous_instance_loader = bridge_dict.get("load_hf_weights")
        self.bridge.load_hf_weights = load_hf_weights_once
        try:
            module = original_build(self)
        finally:
            if had_instance_loader:
                self.bridge.load_hf_weights = previous_instance_loader
            else:
                del self.bridge.load_hf_weights
            pre_wrap_hooks = getattr(self.provider, "_pre_wrap_hooks", None)
            if pre_wrap_hooks is not None and musa_qwen35_vl_pre_wrap in pre_wrap_hooks:
                pre_wrap_hooks.remove(musa_qwen35_vl_pre_wrap)

        if not load_completed:
            raise RuntimeError("MUSA Qwen3.5-VL pre-wrap hook did not load HF weights")
        return module

    return build_megatron_module


def _install_false_loss_prescale(original_forward_backward, original_postprocess):
    """Preserve the MUSA legacy token-mean backward scaling workaround."""

    def get_false_loss_prescale(self, data, forward_only: bool) -> float:
        if forward_only:
            return 1.0
        if os.getenv("VERL_MEGATRON_FALSE_LOSS_PRESCALE", "").strip().lower() != "token_mean":
            return 1.0
        if self.tf_config is None or self.tf_config.calculate_per_token_loss:
            return 1.0
        if self.engine_config.dynamic_context_parallel:
            return 1.0

        import torch
        from verl.utils.device import get_device_id

        loss_mask = data["loss_mask"]
        batch_num_tokens = loss_mask.values().sum() if loss_mask.is_nested else loss_mask.sum()
        batch_num_tokens = batch_num_tokens.to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_data_parallel_group(),
        )
        return max(float(batch_num_tokens.item()) / float(self.get_data_parallel_size()), 1.0)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        from verl.utils import tensordict_utils as tu

        false_loss_prescale = get_false_loss_prescale(self, data, forward_only)
        tu.assign_non_tensor(data, false_loss_prescale=false_loss_prescale)
        output = original_forward_backward(self, data, loss_function, forward_only=forward_only)
        if false_loss_prescale != 1.0:
            for model_chunk in self.module:
                model_chunk.scale_gradients(1.0 / false_loss_prescale)
        return output

    def postprocess_micro_batch_func(
        self, output, data, forward_only: bool, loss_function, local_cp_size=None
    ):
        from verl.utils import tensordict_utils as tu

        result = original_postprocess(
            self, output, data, forward_only, loss_function, local_cp_size=local_cp_size
        )
        false_loss_prescale = tu.get_non_tensor_data(data, key="false_loss_prescale", default=1.0)
        if false_loss_prescale == 1.0:
            return result
        scaled_loss, loss_output = result
        return scaled_loss * false_loss_prescale, loss_output

    return get_false_loss_prescale, forward_backward_batch, postprocess_micro_batch_func


def install() -> None:
    """Install MUSA hooks on VERL's common Megatron engine exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return

    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    original_build = MegatronEngineWithLMHead._build_megatron_module
    original_forward_backward = MegatronEngineWithLMHead.forward_backward_batch
    original_postprocess = MegatronEngineWithLMHead.postprocess_micro_batch_func
    get_prescale, forward_backward, postprocess = _install_false_loss_prescale(
        original_forward_backward, original_postprocess
    )
    MegatronEngineWithLMHead._build_megatron_module = _install_qwen35_vl_pre_wrap(original_build)
    MegatronEngineWithLMHead._get_false_loss_prescale = get_prescale
    MegatronEngineWithLMHead.forward_backward_batch = forward_backward
    MegatronEngineWithLMHead.postprocess_micro_batch_func = postprocess
    _INSTALLED = True
