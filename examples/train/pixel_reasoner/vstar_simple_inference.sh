#!/bin/bash
# Simplified Inference Script - No Actor Training Components
# Only initializes VLLM rollout engine for pure inference

set -x

echo "========================================"
echo "Step 1/6: Checking configuration..."
echo "========================================"

# Model
if [ -d "/mnt/data/pixel_reasoner_3b" ]; then
    model_name="/mnt/data/pixel_reasoner_3b"
else
    model_name=VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100
fi
echo "  Model: $model_name"

# Dataset
dataset_path="/mnt/data/pixel_reasoner/vstar/test.parquet"
if [ ! -f "$dataset_path" ]; then
    echo "Error: Dataset not found!"
    exit 1
fi
echo "  Dataset: $dataset_path"

echo ""
echo "========================================"
echo "Step 2/6: Starting tool server..."
echo "========================================"

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "pixel_reasoner" \
    --workers_per_tool 2 &
server_pid=$!
sleep 5
echo "  Tool server ready at $tool_server_url"

echo ""
echo "========================================"
echo "Step 3/6: Configuring environment..."
echo "========================================"

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export NCCL_DEBUG=WARN
export VERL_RUN_ID=vstar_simple_inference

echo "  GPUs: 0,1"
echo "  VLLM V1: Enabled"

echo ""
echo "========================================"
echo "Step 4/6: Loading model..."
echo "========================================"
echo "  This may take 2-3 minutes..."

echo ""
echo "========================================"
echo "Step 5/6: Running inference..."
echo "========================================"
echo "  Expected duration: 2-3 hours"
echo "  Progress will be shown below:"
echo ""

# Run with MINIMAL configuration - NO ACTOR COMPONENTS
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.val_files=[$dataset_path] \
    data.val_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=pixel_reasoner \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=4096 \
    actor_rollout_ref.agent.max_response_length=1024 \
    actor_rollout_ref.agent.max_start_length=4096 \
    actor_rollout_ref.agent.max_obs_length=1024 \
    actor_rollout_ref.agent.max_turns=5 \
    actor_rollout_ref.agent.additional_eos_token_ids=[151645] \
    actor_rollout_ref.agent.mask_observations=True \
    actor_rollout_ref.agent.max_action_length=2048 \
    actor_rollout_ref.agent.max_concurrent_trajectories=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=256 \
    trainer.logger=['console'] \
    trainer.project_name=vstar_simple \
    trainer.experiment_name=simple_inference \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=0 \
    trainer.total_training_steps=0

inference_exit_code=$?

echo ""
echo "========================================"
echo "Step 6/6: Cleaning up..."
echo "========================================"

pkill -P $server_pid 2>/dev/null
kill $server_pid 2>/dev/null

echo ""
if [ $inference_exit_code -eq 0 ]; then
    echo "========================================"
    echo "SUCCESS: VStar Inference Complete!"
    echo "========================================"
else
    echo "========================================"
    echo "FAILED: Inference exited with error"
    echo "========================================"
    exit $inference_exit_code
fi
