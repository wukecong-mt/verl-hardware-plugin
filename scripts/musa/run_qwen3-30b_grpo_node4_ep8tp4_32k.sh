#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
RUNTIME_ENV="${RUNTIME_ENV:-${SCRIPT_DIR}/runtime_env.yaml}"

export VERL_PATH="${VERL_PATH:-/mnt/his_test/kechun.wu/0824/verl}"
export MEGATRON_PATH="${MEGATRON_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-LM-core_v0.18.0}"
export MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-Bridge/src}"
export MUSA_PATCH_PATH="${MUSA_PATCH_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/megatron-lm-musa-patch}"
export VERL_PLUGIN_PATH="${VERL_PLUGIN_PATH:-/mnt/his_test/kechun.wu/0824/verl-hardware-plugin}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SCRIPT_DIR}/tensorboard_logs/musa-ci-qwen3-30b/Qwen3-30B-A3B_megatron_sglang-dapo}"
mkdir -p "${TENSORBOARD_DIR}"

HF_MODEL_PATH="${HF_MODEL_PATH:-/ipfs/models/Qwen3-30B-A3B}"
DIST_CKPT_PATH="${DIST_CKPT_PATH:-/mnt/seed17/001688/zhaoping/LLMs/MCORE/Qwen3-30B-A3B}"
TRAIN_FILES="${TRAIN_FILES:-/home/verl/musatests/data/dapo_train_16k.parquet}"
VAL_FILES="${VAL_FILES:-/home/verl/musatests/data/dapo_val_1k.parquet}"

USE_DYNAMIC_BSZ=True
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=32768
ACTOR_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1))
INFER_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1))
#MAX_RESPONSE_LENGTH=4096
DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=64
    data.shuffle=True
    data.seed=67280421310721
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.prompt_key=source_prompt
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    # Make the effective defaults from /home/verl's legacy 30B launcher
    # explicit so a config-version change cannot alter the accuracy A/B.
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.data_loader_seed=42
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.profiler.enable=False
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.seed=42
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.batch_p2p_comm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=48
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=48
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
)

ROLLOUT=(
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.calculate_log_probs=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.load_format=auto
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN}
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65
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
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
)

# ==================== MUSA accuracy-alignment logic begin ====================
# Match /home/verl's legacy 30B default DeepScaler verifier and binary 0/1
# reward scale. scripts/musa/deepscaler_math is the same production logic with
# package-relative imports; do not replace it with the Slime reward adapter.
REWARD=(
    reward.custom_reward_function.path="${SCRIPT_DIR}/dapo_deepscaler_reward.py"
    reward.custom_reward_function.name=compute_score
)
# ===================== MUSA accuracy-alignment logic end =====================

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=musa-ci-qwen3-30b
    trainer.experiment_name=Qwen3-30B-A3B_megatron_sglang-dapo
    trainer.n_gpus_per_node=8
    trainer.val_before_train=False
    trainer.nnodes=4
    trainer.save_freq=1000
    trainer.test_freq=1000
    trainer.total_epochs=1
    # A comparison run must start from the configured checkpoint and sampler
    # seed rather than silently inheriting a previous run's RNG/data state.
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

if [[ "${DRY_RUN:-0}" == "cfg" ]]; then
    cd "${VERL_PATH}"
    exec python3 -m verl.trainer.main_ppo --cfg job --resolve \
        "${DATA[@]}" "${ALGORITHM[@]}" "${REWARD[@]}" "${MODEL[@]}" "${ROLLOUT[@]}" \
        "${ACTOR[@]}" "${REF[@]}" "${TRAINER[@]}" "${MUSA_PLUGIN[@]}" "$@"
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
envsubst '${VERL_PATH} ${VERL_PLUGIN_PATH} ${MEGATRON_PATH} ${MEGATRON_BRIDGE_PATH} ${MUSA_PATCH_PATH} ${TENSORBOARD_DIR}' \
    < "${RUNTIME_ENV}" > "${RENDERED_RUNTIME_ENV}"

HEAD_IP="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")"
if [[ -z "${HEAD_IP}" ]]; then
    echo "hostfile has no node address: ${HOSTFILE}" >&2
    exit 1
fi

JOB_ID="${JOB_ID:-qwen3_30b_grpo_node4_musa_$(date +%Y%m%d_%H%M%S)}"
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
    "$@"
