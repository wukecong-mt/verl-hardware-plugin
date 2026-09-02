#!/usr/bin/env bash
# Acceptance baseline: Qwen3-0.6B + GSM8K + Megatron/MCore on MUSA.
#
# This keeps the MUSA GSM8K baseline's data, reward and rollout settings, and
# changes only the actor/reference training backend to Megatron.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERL_PATH="${VERL_PATH:-/mnt/his_test/kechun.wu/0824/verl}"
VERL_PLUGIN_PATH="${VERL_PLUGIN_PATH:-${PLUGIN_ROOT}}"
VERL_MUSA_PATCH="${VERL_MUSA_PATCH:-${VERL_PLUGIN_PATH}/verl-musa-patch}"

MEGATRON_PATH="${MEGATRON_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-LM-core_v0.18.0}"
MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-Bridge/src}"
MUSA_PATCH_PATH="${MUSA_PATCH_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/megatron-lm-musa-patch}"

export VERL_PLATFORM=musa
export VERL_MUSA_PATCH
export MUSA_PATCH_PATH
export PYTHONPATH="${VERL_MUSA_PATCH}:${VERL_PLUGIN_PATH}:${VERL_PATH}:${MEGATRON_PATH}:${MEGATRON_BRIDGE_PATH}:${MUSA_PATCH_PATH}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/ipfs/models/gsm8k}"
MODEL_DIR="${MODEL_DIR:-/ipfs/models/Qwen/Qwen3-0.6B}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SCRIPT_DIR}/tensorboard_logs/qwen3_0.6b_megatron}"
mkdir -p "${TENSORBOARD_DIR}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0.6b_grpo_sglang_megatron_musa_$(date +%Y%m%d_%H%M)}"

RAY_INIT=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM='musa'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='verl_hardware_plugin'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MUSA_PATCH='${VERL_MUSA_PATCH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_PATCH_PATH='${MUSA_PATCH_PATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_PATH='${MEGATRON_PATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MEGATRON_BRIDGE_PATH='${MEGATRON_BRIDGE_PATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.ACCELERATOR_BACKEND='musa'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO='0'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_LIB='/usr/local/musa/lib/libmccl.so'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='/usr/local/musa/lib:${LD_LIBRARY_PATH:-}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PATCH_MUSA_CUSTOM_OPS='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_MUSA_FSDP2_ENABLE_CE_COMM='0'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_MUSA_FSDP2_OVERLAP_LEVEL='0'"
)

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files=['${DATA_DIR}/train.parquet']"
    "data.val_files=['${DATA_DIR}/test.parquet']"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    data.shuffle=True
    data.seed=42
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    "actor_rollout_ref.model.path=${MODEL_DIR}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.optimizer=adam
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
    # Match FSDP's diagnostic setting; this also removes Megatron's default
    # bias/1D-param weight-decay grouping from the comparison.
    actor_rollout_ref.actor.optim.weight_decay=0
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.grad_offload=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_piecewise_cuda_graph=True
)

REF=(
    actor_rollout_ref.ref.strategy=megatron
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.dtype=bfloat16
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=verl_grpo_gsm8k_math
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
)

MUSA=(
    model_engine=megatron
    trainer.device=musa
)

export TENSORBOARD_DIR
python3 -u -m verl.trainer.main_ppo \
    "${RAY_INIT[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${MUSA[@]}" \
    "$@" \
    2>&1 | tee "${LOG_FILE:-verl_gsm8k_musa_megatron.log}"
