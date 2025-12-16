#!/bin/bash
# VStar Quick Verification - 2x RTX 4090
# Minimal script for quick VStar evaluation

set -x

echo "=========================================="
echo "VStar Quick Verification - 2x RTX 4090"
echo "=========================================="
echo ""

# Load local paths if available
if [ -f "./local_paths.sh" ]; then
    echo "Loading local paths..."
    source ./local_paths.sh
fi

# Model configuration
if [ -z "$PIXEL_REASONER_MODEL" ]; then
    # Try user's custom path first
    if [ -d "/mnt/data/pixel_reasoner_3b" ]; then
        model_name="/mnt/data/pixel_reasoner_3b"
        echo "Using custom model path: $model_name"
    # Then try default local path
    elif [ -d "./models/pixel_reasoner_3b" ]; then
        model_name="./models/pixel_reasoner_3b"
        echo "Using local model: $model_name"
    else
        echo "Local model not found, using HuggingFace..."
        model_name=VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100
    fi
else
    model_name=$PIXEL_REASONER_MODEL
    echo "Using model from env: $model_name"
fi

# Dataset configuration - VStar only
# Check custom path first, then default path
if [ -f "/mnt/data/pixel_reasoner/vstar/test.parquet" ]; then
    val_data=[/mnt/data/pixel_reasoner/vstar/test.parquet]
    echo "Using custom dataset path: /mnt/data/pixel_reasoner/vstar/test.parquet"
elif [ -f "$(pwd)/data/pixel_reasoner/vstar/test.parquet" ]; then
    val_data=[$(pwd)/data/pixel_reasoner/vstar/test.parquet]
    echo "Using local dataset path: $(pwd)/data/pixel_reasoner/vstar/test.parquet"
else
    val_data=[$(pwd)/data/pixel_reasoner/vstar/test.parquet]
fi

# Check if dataset exists (check all possible locations)
dataset_found=false
if [ -f "/mnt/data/pixel_reasoner/vstar/test.parquet" ]; then
    dataset_found=true
    dataset_path="/mnt/data/pixel_reasoner/vstar/test.parquet"
elif [ -f "$(pwd)/data/pixel_reasoner/vstar/test.parquet" ]; then
    dataset_found=true
    dataset_path="$(pwd)/data/pixel_reasoner/vstar/test.parquet"
fi

if [ "$dataset_found" = false ]; then
    echo ""
    echo "Error: VStar dataset not found!"
    echo "Searched in:"
    echo "  - /mnt/data/pixel_reasoner/vstar/test.parquet"
    echo "  - $(pwd)/data/pixel_reasoner/vstar/test.parquet"
    echo ""
    echo "Please prepare the dataset first:"
    echo "  python examples/data_preprocess/pixel_reasoner/vstar.py \\"
    echo "    --dataset_path /mnt/data/pixel_reasoner/vstar_raw \\"
    echo "    --split test \\"
    echo "    --local_dir /mnt/data/pixel_reasoner/vstar"
    echo ""
    echo "This will convert the raw data to parquet format needed for inference."
    echo ""
    exit 1
else
    echo "Dataset found: $dataset_path"
fi

# Hardware - 2x RTX 4090
n_gpus_per_node=2
n_nodes=1

# Sampling
temperature=1.0
top_p=1.0
n=4

# Batch size - minimal settings for stability
batch_size=4
val_batch_size=4
ppo_mini_batch_size=2
ppo_micro_batch_size_per_gpu=1

# Token limits - balanced for vision-language model
max_prompt_length=8192
max_response_length=2048
max_obs_length=2048
ppo_max_token_len_per_gpu=$(expr $max_prompt_length + $max_response_length)

# Agent config
enable_agent=True
max_turns=5
max_action_length=2048
action_stop_tokens='</tool_call>'
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=True

# VLLM config - minimal for stability with tensor parallelism
tensor_model_parallel_size=2
gpu_memory_utilization=0.75
log_prob_micro_batch_size_per_gpu=1
max_num_batched_tokens=1024
max_num_seqs=8

# Algorithm
strategy="fsdp2"
rl_alg=grpo
reward_manager=pixel_reasoner

# Optimization
do_offload=False
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1

# Naming
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/_/g')
run_name="vstar_verify_2x4090_${model_pretty_name}"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0,1

rollout_mode='async'

# Action stop tokens file
action_stop_tokens_file="$(mktemp)"
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file

# Start tool server
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

echo "Starting tool server..."
python -m verl_tool.servers.serve --host $host --port $port --tool_type "pixel_reasoner" --workers_per_tool 2 &
server_pid=$!

echo ""
echo "Configuration Summary:"
echo "  Model: $model_name"
echo "  Dataset: VStar"
echo "  GPUs: 2x RTX 4090"
echo "  Batch size: $batch_size"
echo "  GPU memory: ${gpu_memory_utilization}"
echo "  Tool server: $tool_server_url (pid=$server_pid)"
echo ""

sleep 5

# Run inference
echo "Starting VStar evaluation..."
echo ""

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$val_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.dataloader_num_workers=2 \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.strategy=$strategy \
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
    actor_rollout_ref.agent.max_concurrent_trajectories=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    trainer.logger=['console'] \
    trainer.project_name=vstar_verification \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.total_epochs=0 \
    trainer.total_training_steps=0

# Cleanup
echo ""
echo "Cleaning up..."
pkill -P $server_pid 2>/dev/null
kill $server_pid 2>/dev/null
rm -f $action_stop_tokens_file

echo ""
echo "=========================================="
echo "VStar Verification Complete!"
echo "=========================================="
