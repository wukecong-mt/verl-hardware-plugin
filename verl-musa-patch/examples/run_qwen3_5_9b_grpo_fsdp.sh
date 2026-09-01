#!/usr/bin/env bash
# GRPO | Qwen3.5-9B | SGLang rollout | FSDP training | Moore Threads MUSA

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
RUNTIME_ENV="${RUNTIME_ENV:-${SCRIPT_DIR}/runtime_env.yaml}"

export VERL_PATH="${VERL_PATH:-/mnt/his_test/kechun.wu/0824/verl}"
export VERL_PLUGIN_PATH="${VERL_PLUGIN_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export VERL_MUSA_PATCH="${VERL_MUSA_PATCH:-${VERL_PLUGIN_PATH}/verl-musa-patch}"
# runtime_env.yaml is shared with the Megatron examples. These paths keep
# plugin discovery compatible even though this example uses the FSDP engine.
export MEGATRON_PATH="${MEGATRON_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-LM-core_v0.18.0}"
export MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-Bridge/src}"
export MUSA_PATCH_PATH="${MUSA_PATCH_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/megatron-lm-musa-patch}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/mnt/his_test/kechun.wu/tensorboard_logs/qwen3_5_9b_fsdp}"

MODEL_PATH="${MODEL_PATH:-/ipfs/kechun.wu/models/qwen35/Qwen3___5-9B}"
TRAIN_FILES="${TRAIN_FILES:-/home/verl/musatests/data/dapo_train_16k.parquet}"
VAL_FILES="${VAL_FILES:-/home/verl/musatests/data/dapo_val_1k.parquet}"

NNODES="${NNODES:-2}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-33180}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"

ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"

PROJECT_NAME="${PROJECT_NAME:-musa-ci-qwen3-5-9b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3.5-9B_fsdp_sglang-grpo}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
TEST_FREQ="${TEST_FREQ:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.prompt_key=source_prompt
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=512
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_chunk_size=512
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2
)

ROLLOUT=(
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.expert_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.load_format=auto
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=8192
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device="musa"

    # Enable NEXTN speculative rollout; FSDP training itself has no MTP loss.
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=NEXTN
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_steps=3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_eagle_topk=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_draft_tokens=4
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"
    actor_rollout_ref.ref.fsdp_config.fsdp_size=-1
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

REWARD=(
    reward.custom_reward_function.path="${SCRIPT_DIR}/reward/deepscaler_reward.py"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.resume_mode=disable
)

MUSA_PLUGIN=(
    model_engine=dp
    trainer.device=musa
)

COMMAND=(
    python3 -u -m verl.trainer.main_ppo
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${REF[@]}"
    "${ALGORITHM[@]}"
    "${REWARD[@]}"
    "${TRAINER[@]}"
    "${MUSA_PLUGIN[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "cfg" ]]; then
    cd "${VERL_PATH}"
    exec "${COMMAND[@]}" --cfg job --resolve
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
envsubst '${VERL_PATH} ${VERL_PLUGIN_PATH} ${VERL_MUSA_PATCH} ${MEGATRON_PATH} ${MEGATRON_BRIDGE_PATH} ${MUSA_PATCH_PATH} ${TENSORBOARD_DIR} ${VERL_MUSA_WEIGHT_SYNC_IPC_COLLECT} ${VERL_MUSA_WEIGHT_SYNC_MEMORY_SNAPSHOT} ${VERL_MUSA_WEIGHT_SYNC_SNAPSHOT_BUCKETS} ${VERL_MUSA_WEIGHT_SYNC_SNAPSHOT_DIR} ${VERL_MUSA_WEIGHT_SYNC_REF_DEBUG} ${VERL_MUSA_WEIGHT_SYNC_SKIP_IPC_SERIALIZE} ${VERL_MUSA_WEIGHT_SYNC_CPU_SERIALIZE}' \
    < "${RUNTIME_ENV}" > "${RENDERED_RUNTIME_ENV}"

HEAD_IP="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")"
if [[ -z "${HEAD_IP}" ]]; then
    echo "hostfile has no node address: ${HOSTFILE}" >&2
    exit 1
fi

JOB_ID="${JOB_ID:-qwen3_5_9b_grpo_fsdp_musa_$(date +%Y%m%d_%H%M%S)}"
cd "${VERL_PATH}"

RAY_ADDRESS="http://${HEAD_IP}:8265" ray job submit \
    --submission-id "${JOB_ID}" \
    --runtime-env="${RENDERED_RUNTIME_ENV}" \
    -- "${COMMAND[@]}"
