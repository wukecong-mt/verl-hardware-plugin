#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRPO_PLUGIN_PATH="${DRPO_PLUGIN_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VERL_090_ROOT="$(cd "${DRPO_PLUGIN_PATH}/.." && pwd)"
VERL_PATH="${VERL_PATH:-${VERL_090_ROOT}/verl}"
VERL_HARDWARE_PLUGIN_PATH="${VERL_HARDWARE_PLUGIN_PATH:-${VERL_090_ROOT}/verl-hardware-plugin}"

########################### user-adjustable ###########################
MODEL_PATH="${MODEL_PATH:-/ipfs/rupert/verl_090_drpo/verl_090/models/Qwen3-0.6B}"
DATA_ROOT="${DATA_ROOT:-/ipfs/rupert/verl_090_drpo/verl_090/data}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_ROOT}/dapo_math/all/train-00000-of-00001.parquet}"
VAL_FILES="${VAL_FILES:-[\"${DATA_ROOT}/dapo_math/all/train-00000-of-00001.parquet\"]}"

PROJECT_NAME="${PROJECT_NAME:-s5000_musa_fsdp2_sglang_drpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_drpo_fsdp2_sglang}"
RUN_ROOT="${RUN_ROOT:-${DRPO_PLUGIN_PATH}/outputs}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs/${PROJECT_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
CKPTS_DIR="${CKPTS_DIR:-${RUN_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

POLICY_LOSS_MODE="${POLICY_LOSS_MODE:-drpo_eff}"
DRPO_EPSILON="${DRPO_EPSILON:-12.5}"
DRPO_MU_WEIGHTED="${DRPO_MU_WEIGHTED:-true}"
EFFECTIVE_RATIO_METRIC="${EFFECTIVE_RATIO_METRIC:-ratio_delta}"
EFFECTIVE_RATIO_MAP="${EFFECTIVE_RATIO_MAP:-clip}"
EFFECTIVE_RATIO_METRIC_CLIP="${EFFECTIVE_RATIO_METRIC_CLIP:-null}"
EFFECTIVE_RATIO_MIN="${EFFECTIVE_RATIO_MIN:-0.1}"
EFFECTIVE_RATIO_MAX="${EFFECTIVE_RATIO_MAX:-2.0}"
EFFECTIVE_RATIO_BETA="${EFFECTIVE_RATIO_BETA:-1.0}"
EFFECTIVE_RATIO_ALPHA="${EFFECTIVE_RATIO_ALPHA:-0.5}"
EFFECTIVE_RATIO_LAMBDA="${EFFECTIVE_RATIO_LAMBDA:-1.0e-4}"
EFFECTIVE_RATIO_EPS="${EFFECTIVE_RATIO_EPS:-1.0e-8}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-10240}"
ACTOR_LR="${ACTOR_LR:-3e-6}"
ACTOR_WEIGHT_DECAY="${ACTOR_WEIGHT_DECAY:-0.1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.99}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-100}"

OLD_LOGPROB_MODE="${OLD_LOGPROB_MODE:-bypass}"
NORM_ADV_BY_STD_IN_GRPO="${NORM_ADV_BY_STD_IN_GRPO:-true}"
LAUNCH_MODE="${LAUNCH_MODE:-background}"
######################### end user-adjustable #########################

case "${POLICY_LOSS_MODE}" in
    drpo|drpo_eff) ;;
    *)
        echo "POLICY_LOSS_MODE must be drpo or drpo_eff, got: ${POLICY_LOSS_MODE}" >&2
        exit 2
        ;;
esac

case "${OLD_LOGPROB_MODE}" in
    bypass)
        ROLLOUT_CORR_BYPASS_MODE=true
        ROLLOUT_CALCULATE_LOG_PROBS=true
        ;;
    recompute)
        ROLLOUT_CORR_BYPASS_MODE=false
        ROLLOUT_CALCULATE_LOG_PROBS=false
        ;;
    *)
        echo "OLD_LOGPROB_MODE must be bypass or recompute, got: ${OLD_LOGPROB_MODE}" >&2
        exit 2
        ;;
esac

for required_dir in "${VERL_PATH}" "${VERL_HARDWARE_PLUGIN_PATH}" "${DRPO_PLUGIN_PATH}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "required directory not found: ${required_dir}" >&2
        exit 1
    fi
done

time_str="$(TZ=Asia/Shanghai date +%Y-%m-%d-%H-%M-%S-%N)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}-${time_str}.log}"
PID_FILE="${PID_FILE:-${LOG_FILE}.pid}"
mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}" "${CKPTS_DIR}"

export VERL_PLATFORM="${VERL_PLATFORM:-musa}"
export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-verl_hardware_plugin,verl_drpo_plugin}"
export VERL_USE_EXTERNAL_PLUGINS="${VERL_USE_EXTERNAL_PLUGINS:-hardware,drpo}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
export RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_TIMEOUT="${MCCL_TIMEOUT:-1800000}"
export TORCH_MUSA_FSDP2_OVERLAP_LEVEL="${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}"
export LD_LIBRARY_PATH="/usr/local/musa/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export TENSORBOARD_DIR
export PYTHONPATH="${DRPO_PLUGIN_PATH}:${VERL_HARDWARE_PLUGIN_PATH}:${VERL_HARDWARE_PLUGIN_PATH}/scripts/musa:${VERL_PATH}:${PYTHONPATH:-}"

if [[ "${DRY_RUN:-}" != "cfg" ]]; then
    for required_file in "${MODEL_PATH}/config.json" "${TRAIN_FILES}"; do
        if [[ ! -e "${required_file}" ]]; then
            echo "required model/data input not found: ${required_file}" >&2
            exit 1
        fi
    done
fi

if [[ "${SKIP_PLATFORM_CHECK:-0}" != "1" ]]; then
    python3 - <<'PY'
from verl.plugin.platform.platform_manager import PlatformRegistry
from verl.trainer.ppo.core_algos import get_policy_loss_fn

if PlatformRegistry.get("musa") is None:
    raise RuntimeError(
        "The installed hardware plugin has not registered VERL platform 'musa'. "
        "Install a MUSA-capable verl-hardware-plugin before launching training."
    )
for name in ("drpo", "drpo_eff"):
    function = get_policy_loss_fn(name)
    print(f"REGISTERED_POLICY_LOSS={name}:{function.__module__}.{function.__name__}")
PY
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO}"
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode="${ROLLOUT_CORR_BYPASS_MODE}"
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.prompt_key=prompt
    data.truncation=left
    data.return_raw_chat=True
    data.shuffle=True
    data.dataloader_num_workers=0
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay="${ACTOR_WEIGHT_DECAY}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=True
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.policy_loss._target_=verl_drpo_plugin.config.DrpoPolicyLossConfig
    actor_rollout_ref.actor.policy_loss.loss_mode="${POLICY_LOSS_MODE}"
    +actor_rollout_ref.actor.policy_loss.drpo_epsilon="${DRPO_EPSILON}"
    +actor_rollout_ref.actor.policy_loss.mu_weighted="${DRPO_MU_WEIGHTED}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_metric="${EFFECTIVE_RATIO_METRIC}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_map="${EFFECTIVE_RATIO_MAP}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_metric_clip="${EFFECTIVE_RATIO_METRIC_CLIP}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_min="${EFFECTIVE_RATIO_MIN}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_max="${EFFECTIVE_RATIO_MAX}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_beta="${EFFECTIVE_RATIO_BETA}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_alpha="${EFFECTIVE_RATIO_ALPHA}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_lambda="${EFFECTIVE_RATIO_LAMBDA}"
    +actor_rollout_ref.actor.policy_loss.effective_ratio_eps="${EFFECTIVE_RATIO_EPS}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.calculate_log_probs="${ROLLOUT_CALCULATE_LOG_PROBS}"
    actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload=False
)

TRAINER=(
    trainer.device=musa
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard","file"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=disable
)

RAY_ENV=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM=\"${VERL_PLATFORM}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.ACCELERATOR_BACKEND=\"${ACCELERATOR_BACKEND}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=\"${VERL_USE_EXTERNAL_MODULES}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_PLUGINS=\"${VERL_USE_EXTERNAL_PLUGINS}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\"${PYTHONPATH}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_VISIBLE_DEVICES=\"${MUSA_VISIBLE_DEVICES}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=\"1\""
)

COMMAND=(
    python3 -u -m verl.trainer.main_ppo
    "${DATA[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${REF[@]}"
    "${TRAINER[@]}"
    "${RAY_ENV[@]}"
    "$@"
)

echo "VERL_PATH=${VERL_PATH}"
echo "VERL_HARDWARE_PLUGIN_PATH=${VERL_HARDWARE_PLUGIN_PATH}"
echo "DRPO_PLUGIN_PATH=${DRPO_PLUGIN_PATH}"
echo "POLICY_LOSS_MODE=${POLICY_LOSS_MODE}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILES=${TRAIN_FILES}"
echo "VAL_FILES=${VAL_FILES}"
echo "LOG_FILE=${LOG_FILE}"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "CKPTS_DIR=${CKPTS_DIR}"

cd "${VERL_PATH}"
if [[ "${DRY_RUN:-}" == "cfg" ]]; then
    exec "${COMMAND[@]:0:4}" --cfg job --resolve "${COMMAND[@]:4}"
fi

case "${LAUNCH_MODE}" in
    foreground)
        exec "${COMMAND[@]}"
        ;;
    background)
        nohup "${COMMAND[@]}" >"${LOG_FILE}" 2>&1 &
        pid=$!
        printf '%s\n' "${pid}" >"${PID_FILE}"
        echo "PID=${pid}"
        echo "PID_FILE=${PID_FILE}"
        ;;
    *)
        echo "LAUNCH_MODE must be foreground or background, got: ${LAUNCH_MODE}" >&2
        exit 2
        ;;
esac
