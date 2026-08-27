#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${HOSTFILE:-${SCRIPT_DIR}/hostfile}"
RUNTIME_ENV="${RUNTIME_ENV:-${SCRIPT_DIR}/runtime_env.yaml}"

export VERL_PATH="${VERL_PATH:-/mnt/his_test/kechun.wu/0824/verl}"
export VERL_PLUGIN_PATH="${VERL_PLUGIN_PATH:-/mnt/his_test/kechun.wu/0824/verl-hardware-plugin}"
export MEGATRON_BRIDGE_PATH="${MEGATRON_BRIDGE_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-Bridge/src}"
export MEGATRON_PATH="${MEGATRON_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/Megatron-LM-core_v0.18.0}"
export MUSA_PATCH_PATH="${MUSA_PATCH_PATH:-/mnt/his_test/kechun.wu/rl_workspace/0719/megatron-lm-musa-patch}"

# This is the HF checkpoint used by the reference Slime run. Override it when
# the checkpoint is mounted at a different shared-filesystem path.
HF_MODEL_PATH="${HF_MODEL_PATH:-/ipfs/kechun.wu/models/021-32B-SFT-epoch4_fixed}"
DIST_CKPT_PATH="${DIST_CKPT_PATH:-}"

DATASET_PATH="${DATASET_PATH:-/ipfs/kechun.wu/models/data}"
TRAIN_FILES="${TRAIN_FILES:-${DATASET_PATH}/amt_math17k_d.parquet}"
AIME_2024_FILE="${AIME_2024_FILE:-${DATASET_PATH}/aime_2024.parquet}"
AIME_2025_FILE="${AIME_2025_FILE:-${DATASET_PATH}/aime_2025.parquet}"
VAL_FILES="${VAL_FILES:-['${AIME_2025_FILE}','${AIME_2024_FILE}']}"
ZERO2ONE_REWARD_PATH="${ZERO2ONE_REWARD_PATH:-${SCRIPT_DIR}/reward/zero2one_reward.py}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
# VERL has one response-length limit for train and validation. Use the Slime
# evaluation limit (30718), which also safely covers its 30111 training limit.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32168}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"

ENABLE_ACTOR_PROFILE="${ENABLE_ACTOR_PROFILE:-False}"
PROFILE_STEPS="${PROFILE_STEPS:-[2]}"
PROFILE_RANKS="${PROFILE_RANKS:-[0,1,2,3,4,5,6,7]}"
PROFILE_SAVE_PATH="${PROFILE_SAVE_PATH:-${SCRIPT_DIR}/profiles/021_32b_actor_update}"

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=128
    data.shuffle=True
    data.seed=42
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.prompt_key=prompt
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.enable_activation_offload=False
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.optimizer=adam
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.data_loader_seed=42
    actor_rollout_ref.actor.ppo_mini_batch_size=128
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=3.001
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.0

    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.seed=42
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.megatron.entropy_from_logits_chunk_size=2048

    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=40
    # DS-V2 021-32B uses 32 MLA attention heads.  Do not let the generic HF
    # num_key_value_heads -> num_query_groups mapping produce an invalid GQA
    # pair for this checkpoint.
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_attention_heads=32
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_query_groups=32
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.accumulate_allreduce_grads_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.qk_layernorm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=seq_aux_loss
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.disable_bf16_reduced_precision_matmul=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=8
    actor_rollout_ref.rollout.expert_parallel_size=8
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.rollout.max_num_seqs=1024
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.top_p=0.999
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.rollout.val_kwargs.n=8
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa
    +actor_rollout_ref.rollout.engine_kwargs.sglang.load_format=auto
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=512
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_dense_tp_size=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_lm_head=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.allow_auto_truncate=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=1024
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=16384
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}"
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.megatron.sequence_parallel=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    actor_rollout_ref.ref.megatron.param_offload=True
    ++actor_rollout_ref.ref.megatron.override_transformer_config.num_attention_heads=32
    ++actor_rollout_ref.ref.megatron.override_transformer_config.num_query_groups=32
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=fused
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
)

REWARD=(
    reward.reward_manager.name=naive
    reward.custom_reward_function.path="${SCRIPT_DIR}/reward/021_reward.py"
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.source_path="${ZERO2ONE_REWARD_PATH}"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.balance_batch=True
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=musa-021-32b
    trainer.experiment_name=021-32B_megatron-bridge_sglang-grpo
    trainer.n_gpus_per_node=8
    trainer.nnodes=2
    trainer.val_before_train=False
    trainer.save_freq=500
    # VERL currently has no eval_start_step gate, so interval-5 validation
    # starts at step 5 rather than Slime's step 50.
    trainer.test_freq=5
    trainer.total_epochs=1
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.default_local_dir="${CHECKPOINT_DIR:-/tmp/verl_021_32b}"
    trainer.resume_mode=disable
)

MUSA_PLUGIN=(
    model_engine=megatron
    trainer.device=musa
)

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
        actor_rollout_ref.actor.profiler.tool_config.torch.contents='["cpu","cuda"]'
        actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=0
        actor_rollout_ref.rollout.profiler.enable=False
    )
fi

VERL_ARGS=(
    "${DATA[@]}"
    "${ALGORITHM[@]}"
    "${REWARD[@]}"
    "${MODEL[@]}"
    "${ROLLOUT[@]}"
    "${ACTOR[@]}"
    "${REF[@]}"
    "${TRAINER[@]}"
    "${MUSA_PLUGIN[@]}"
    "${PROFILER[@]}"
)

if [[ "${DRY_RUN:-0}" == "cfg" ]]; then
    cd "${VERL_PATH}"
    exec python3 -m verl.trainer.main_ppo --cfg job --resolve "${VERL_ARGS[@]}" "$@"
fi

for required_file in "${RUNTIME_ENV}" "${HOSTFILE}" "${TRAIN_FILES}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
    echo "HF checkpoint config not found: ${HF_MODEL_PATH}/config.json" >&2
    exit 1
fi
if ! command -v envsubst >/dev/null 2>&1; then
    echo "envsubst is required to render ${RUNTIME_ENV}" >&2
    exit 1
fi

RENDERED_RUNTIME_ENV="${TMPDIR:-/tmp}/verl_runtime_env_${$}.yaml"
trap 'rm -f "${RENDERED_RUNTIME_ENV}"' EXIT
envsubst '${VERL_PATH} ${VERL_PLUGIN_PATH} ${MEGATRON_PATH} ${MEGATRON_BRIDGE_PATH} ${MUSA_PATCH_PATH}' \
    < "${RUNTIME_ENV}" > "${RENDERED_RUNTIME_ENV}"

HEAD_IP="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "${HOSTFILE}")"
if [[ -z "${HEAD_IP}" ]]; then
    echo "hostfile has no node address: ${HOSTFILE}" >&2
    exit 1
fi

JOB_ID="${JOB_ID:-021_32b_grpo_node2_musa_$(date +%Y%m%d_%H%M%S)}"
cd "${VERL_PATH}"

RAY_ADDRESS="http://${HEAD_IP}:8265" ray job submit \
    --submission-id "${JOB_ID}" \
    --runtime-env="${RENDERED_RUNTIME_ENV}" \
    -- python3 -u -m verl.trainer.main_ppo "${VERL_ARGS[@]}" "$@"
