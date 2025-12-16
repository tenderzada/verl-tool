#!/bin/bash
# Single GPU Inference - Most Stable Configuration

set -x

echo "====================================================="
echo "VStar Inference - Single RTX 4090 (Ultra-Stable)"
echo "====================================================="

# Model
model_name="/mnt/data/pixel_reasoner_3b"
dataset_path="/mnt/data/pixel_reasoner/vstar/test.parquet"

echo "Step 1/6: Starting tool server..."
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

python -m verl_tool.servers.serve \
    --host $host --port $port \
    --tool_type "pixel_reasoner" \
    --workers_per_tool 1 &
server_pid=$!
sleep 5

echo "Step 2/6: Configuring single GPU environment..."
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=1

echo "Step 3/6: Loading model (may take 2-3 minutes)..."
echo "Step 4/6: Initializing VLLM inference engine..."
echo "Step 5/6: Running inference (estimated 4-5 hours)..."
echo ""

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=[$dataset_path] \
    data.val_files=[$dataset_path] \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=3072 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    reward_model.reward_manager=pixel_reasoner \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3840 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.agent.enable_agent=True \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=3072 \
    actor_rollout_ref.agent.max_response_length=768 \
    actor_rollout_ref.agent.max_start_length=3072 \
    actor_rollout_ref.agent.max_obs_length=768 \
    actor_rollout_ref.agent.max_turns=5 \
    actor_rollout_ref.agent.additional_eos_token_ids=[151645] \
    actor_rollout_ref.agent.mask_observations=True \
    actor_rollout_ref.agent.max_action_length=1024 \
    actor_rollout_ref.agent.max_concurrent_trajectories=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.max_model_len=3840 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=128 \
    trainer.logger=['console'] \
    trainer.project_name=vstar_single_gpu \
    trainer.experiment_name=single_gpu_inference \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=0 \
    trainer.total_training_steps=0

echo ""
echo "Step 6/6: Cleaning up..."
pkill -P $server_pid 2>/dev/null
kill $server_pid 2>/dev/null

echo ""
echo "====================================================="
echo "Inference Complete!"
echo "====================================================="
