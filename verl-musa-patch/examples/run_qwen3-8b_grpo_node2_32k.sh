#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
RUNTIME_ENV="${RUNTIME_ENV:-${SCRIPT_DIR}/runtime_env.yaml}"

export VERL_PATH="${VERL_PATH:-/mnt/his_test/kechun.wu/0824/verl}"
export MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-Bridge/src}"

export MEGATRON_PATH="${MEGATRON_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-LM-core_v0.18.0}"
export MUSA_PATCH_PATH="${MUSA_PATCH_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/megatron-lm-musa-patch}"

#export MEGATRON_PATH="${MEGATRON_PATH:-/home/Megatron-LM}"
#export MUSA_PATCH_PATH="${MUSA_PATCH_PATH:-/home/megatron-lm-musa-patch}"


export VERL_PLUGIN_PATH="${VERL_PLUGIN_PATH:-/mnt/his_test/kechun.wu/0824/verl-hardware-plugin}"
export VERL_MUSA_PATCH="${VERL_MUSA_PATCH:-${VERL_PLUGIN_PATH}/verl-musa-patch}"

# Keep TensorBoard events outside Ray's temporary working directory.  Override
# this per run with TENSORBOARD_DIR=/path/to/run when needed.
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/mnt/his_test/kechun.wu/tensorboard_logs/}"

HF_MODEL_PATH="${HF_MODEL_PATH:-/ipfs/models/Qwen3-8B}"
DIST_CKPT_PATH="${DIST_CKPT_PATH:-}"
TRAIN_FILES="${TRAIN_FILES:-/home/verl/musatests/data/dapo_train_16k.parquet}"
VAL_FILES="${VAL_FILES:-/home/verl/musatests/data/dapo_val_1k.parquet}"

USE_DYNAMIC_BSZ=True
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
ACTOR_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1))
INFER_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1))
# [MUSA plugin run] Full-length training validation after the head_dim=128
# MATE backward smoke test completed its actor update successfully.
MAX_RESPONSE_LENGTH_=32768
# Optional actor-update trace. Keep it disabled for normal training and enable it
# only for a dedicated profiling run.
ENABLE_ACTOR_PROFILE="${ENABLE_ACTOR_PROFILE:-False}"
PROFILE_STEPS="${PROFILE_STEPS:-[2]}"
PROFILE_RANKS="${PROFILE_RANKS:-[0,1,2,3,4,5,6,7]}"
PROFILE_SAVE_PATH="${PROFILE_SAVE_PATH:-${SCRIPT_DIR}/profiles/qwen3_8b_actor_update}"

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=64
    # Match the implicit torch.Generator seed used by the legacy run when
    # data.seed was null, but make the ordering explicit for accuracy A/B.
    data.shuffle=True
    data.seed=67280421310721
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH_}
    data.filter_overlong_prompts=True
    data.prompt_key=source_prompt
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=False
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.data_loader_seed=42
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.seed=42
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.megatron.entropy_from_logits_chunk_size=2048
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.batch_p2p_comm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.no_gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.accumulate_allreduce_grads_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.no_masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.no_bias_swiglu_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=36
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=14
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
)

ROLLOUT=(
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.calculate_log_probs=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.load_format=auto
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=8192
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN}
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=0.8
    actor_rollout_ref.rollout.top_k=100
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8
    actor_rollout_ref.rollout.val_kwargs.top_k=50
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9
    actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    actor_rollout_ref.ref.megatron.sequence_parallel=False
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ==================== MUSA accuracy-alignment logic begin ====================
# Match the legacy DAPO reference run's DeepScaler verifier and binary 0/1
# reward scale through the shared reward directory.
REWARD=(
    reward.custom_reward_function.path="${SCRIPT_DIR}/reward/deepscaler_reward.py"
    reward.custom_reward_function.name=compute_score
)
# ===================== MUSA accuracy-alignment logic end =====================

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=musa-ci-qwen3-8b
    trainer.experiment_name=Qwen3-8B_megatron_sglang-dapo
    trainer.n_gpus_per_node=8
    trainer.val_before_train=False
    trainer.nnodes=2
    trainer.save_freq=1000
    trainer.test_freq=1000
    trainer.total_epochs=1
    # Never inherit a previous sampler/RNG state during an accuracy A/B run.
    trainer.resume_mode=disable
)

# ==================== MUSA external-plugin adaptation begin ====================
# The old ppo_megatron_trainer_demo config selected Megatron implicitly. The
# current verl branch selects it through model_engine, while the platform and
# SGLang device are supplied explicitly for MUSA workers.
MUSA_PLUGIN=(
    model_engine=megatron
    trainer.device=musa
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa
)
# ===================== MUSA external-plugin adaptation end =====================

# ==================== MUSA actor-update profiling begin ====================
# Profile only the selected training step and actor ranks. Rollout profiling is
# explicitly disabled; the resulting trace contains an "actor_update" range for
# isolating Megatron forward/backward/optimizer work.
PROFILER=()
if [[ "${ENABLE_ACTOR_PROFILE}" == "True" || "${ENABLE_ACTOR_PROFILE}" == "true" ]]; then
    PROFILER=(
        global_profiler.tool=torch
        global_profiler.steps="${PROFILE_STEPS}"
        global_profiler.save_path="${PROFILE_SAVE_PATH}"
        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.all_ranks=False
        actor_rollout_ref.actor.profiler.ranks="${PROFILE_RANKS}"
        actor_rollout_ref.actor.profiler.tool_config.torch.discrete=False
        actor_rollout_ref.actor.profiler.tool_config.torch.contents="['cpu','cuda']"
        actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=0
        actor_rollout_ref.rollout.profiler.enable=False
    )
fi
# ===================== MUSA actor-update profiling end =====================

if [[ "${DRY_RUN:-0}" == "cfg" ]]; then
    cd "${VERL_PATH}"
    exec python3 -m verl.trainer.main_ppo --cfg job --resolve \
        "${DATA[@]}" "${ALGORITHM[@]}" "${REWARD[@]}" "${MODEL[@]}" "${ROLLOUT[@]}" \
        "${ACTOR[@]}" "${REF[@]}" "${TRAINER[@]}" "${MUSA_PLUGIN[@]}" "${PROFILER[@]}" "$@"
fi

if [[ ! -f "${RUNTIME_ENV}" ]]; then
    echo "runtime env file not found: ${RUNTIME_ENV}" >&2
    exit 1
fi
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "hostfile not found: ${HOSTFILE}; run setup_ray.sh or set HOSTFILE" >&2
    exit 1
fi
if ! command -v envsubst >/dev/null 2>&1; then
    echo "envsubst is required to render ${RUNTIME_ENV}" >&2
    exit 1
fi

RENDERED_RUNTIME_ENV="${TMPDIR:-/tmp}/verl_runtime_env_${$}.yaml"
trap 'rm -f "${RENDERED_RUNTIME_ENV}"' EXIT
envsubst '${VERL_PATH} ${VERL_PLUGIN_PATH} ${VERL_MUSA_PATCH} ${MEGATRON_PATH} ${MEGATRON_BRIDGE_PATH} ${MUSA_PATCH_PATH} ${TENSORBOARD_DIR}' \
    < "${RUNTIME_ENV}" > "${RENDERED_RUNTIME_ENV}"

HEAD_IP="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")"
if [[ -z "${HEAD_IP}" ]]; then
    echo "hostfile has no node address: ${HOSTFILE}" >&2
    exit 1
fi

JOB_ID="${JOB_ID:-qwen3_8b_grpo_node2_musa_$(date +%Y%m%d_%H%M%S)}"
cd "${VERL_PATH}"

RAY_ADDRESS="http://${HEAD_IP}:8265" ray job submit \
    --submission-id "${JOB_ID}" \
    --runtime-env="${RENDERED_RUNTIME_ENV}" \
    -- python3 -u -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${MUSA_PLUGIN[@]}" \
    "${PROFILER[@]}" \
    "$@"
