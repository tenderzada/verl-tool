#!/bin/bash
# Pixel Reasoner Inference Script - Single RTX 4090 (24GB)
# Minimal configuration for quick testing on a single GPU

set -x

echo "=========================================="
echo "Pixel Reasoner Inference - Single RTX 4090"
echo "=========================================="

# Model configuration
model_name=VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100

# Dataset configuration - use a small subset for testing
dataset_name=pixel_reasoner/info_vqa
val_data=[$(pwd)/data/${dataset_name}/test.parquet]

# Hardware configuration - Single GPU
n_gpus_per_node=1
n_nodes=1

# Sampling parameters
temperature=1.0
top_p=1.0
n=2  # Minimal sampling for quick testing

# Batch size - very conservative for single 4090
batch_size=8
val_batch_size=8
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=1

# Token length limits
max_prompt_length=8192  # Further reduced for single GPU
max_response_length=4096
max_obs_length=2048
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# Agent configuration
enable_agent=True
max_turns=5
max_action_length=1024
action_stop_tokens='</tool_call>'
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True

# VLLM configuration - very conservative for single 4090
tensor_model_parallel_size=1
gpu_memory_utilization=0.6  # Very conservative
log_prob_micro_batch_size_per_gpu=2
max_num_batched_tokens=2000
max_num_seqs=16

# Strategy and algorithm
strategy="fsdp2"
rl_alg=grpo
reward_manager=pixel_reasoner

# Optimization settings
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1

# Naming
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="inference_single4090_${reward_manager}-${model_pretty_name}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0

rollout_mode='async'

# Create temp file for action stop tokens
action_stop_tokens_file="$(mktemp)"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file

# Start tool server
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

echo "Starting tool server at $tool_server_url"
python -m verl_tool.servers.serve --host $host --port $port --tool_type "pixel_reasoner" --workers_per_tool 1 &
server_pid=$!

echo "Tool Server (pid=$server_pid) started"
echo "Model: $model_name"
echo "Hardware: Single RTX 4090 (24GB)"
echo "Batch size: $batch_size (small for testing)"
echo ""

sleep 5

# Run inference
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.val_files=$val_data \
    data.dataloader_num_workers=1 \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.agent.max_concurrent_trajectories=$batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    trainer.logger=['console'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.total_epochs=0 \
    trainer.total_training_steps=0

# Clean up
echo ""
echo "Cleaning up..."
pkill -P $server_pid 2>/dev/null
kill $server_pid 2>/dev/null
rm -f $action_stop_tokens_file

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
