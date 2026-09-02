#!/usr/bin/env bash
# Acceptance Baseline: Qwen3-0.6B + GSM8K + FSDP2 actor on Moore Threads MUSA.
#
# This keeps the standard VERL GSM8K baseline configuration and only adds the
# MUSA platform/runtime settings. Paths and resource values can be overridden
# through environment variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERL_PLUGIN_PATH="${VERL_PLUGIN_PATH:-${PLUGIN_ROOT}}"
VERL_MUSA_PATCH="${VERL_MUSA_PATCH:-${VERL_PLUGIN_PATH}/verl-musa-patch}"


export PYTHONPATH="${VERL_MUSA_PATCH}:${PYTHONPATH:-}"


########################### user-adjustable ###########################
INFER_BACKEND="${INFER_BACKEND:-sglang}"
DATA_DIR="${DATA_DIR:-/ipfs/models/gsm8k}"
MODEL_DIR="${MODEL_DIR:-/ipfs/models/Qwen/Qwen3-0.6B}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"

ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}"
ROLLOUT_N="${ROLLOUT_N:-5}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-verl_grpo_gsm8k_math}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0.6b_grpo_${INFER_BACKEND}_fsdp2_musa_$(date +%Y%m%d_%H%M)}"
########################### end user-adjustable ###########################

    # "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_EXECUTION_TIMEOUT='3000000'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_TIMEOUT='600000'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_PROTOS='2'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_CHECK_POINTERS='0'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_ALGOS='1'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_IB_GID_INDEX='3'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_BUFFSIZE='20971520'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_MAX_NCHANNELS='14'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_IB_HCA='mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8,mlx5_9'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_BLOCK_SCHEDULE_MODE='1'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_BLOCK_DISTRIBUTION_GRANULARITY='0'"
    # "+ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM='false'"

RAY_INIT=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM='musa'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='verl_hardware_plugin'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MUSA_PATCH='${VERL_MUSA_PATCH}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.ACCELERATOR_BACKEND='musa'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO='0'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_LIB='/usr/local/musa/lib/libmccl.so'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='/usr/local/musa/lib:${LD_LIBRARY_PATH:-}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PATCH_MUSA_CUSTOM_OPS='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.SGLANG_MUSA_GRAPH_COMPAT='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_BF16_DENSE_TFLOPS='458'"
    #"+ray_kwargs.ray_init.runtime_env.env_vars.ACCELERATE_USE_FSDP='1'"
    #"+ray_kwargs.ray_init.runtime_env.env_vars.FSDP_CPU_RAM_EFFICIENT_LOADING='1'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_MUSA_FSDP2_OVERLAP_LEVEL='0'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${VERL_MUSA_PATCH}:${PYTHONPATH:-}'"
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

# ACTOR=(
#     actor_rollout_ref.actor.strategy=fsdp2
#     "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
#     "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
#     actor_rollout_ref.actor.use_dynamic_bsz=True
#     "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
#     actor_rollout_ref.actor.use_kl_loss=True
#     "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl
#     "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
#     actor_rollout_ref.actor.use_torch_compile=False
#     actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
#     actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
#     actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
# )

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    #actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="${INFER_BACKEND}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    actor_rollout_ref.rollout.free_cache_engine=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_piecewise_cuda_graph=True
    #actor_rollout_ref.rollout.temperature=0.8
)

# REF=(
#     actor_rollout_ref.ref.strategy=fsdp2
#     actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
#     "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
#     actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
#     actor_rollout_ref.ref.fsdp_config.fsdp_size=-1
# )

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)


TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
)

MUSA=(
    model_engine=dp
    trainer.device=musa
)

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
    2>&1 | tee "${LOG_FILE:-verl_gsm8k_musa.log}"
