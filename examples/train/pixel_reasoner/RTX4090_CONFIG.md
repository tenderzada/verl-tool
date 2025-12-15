# Pixel Reasoner - RTX 4090 配置指南

本指南专门针对使用 RTX 4090 显卡进行 Pixel-Reasoner (3B) 推理的用户。

## 硬件配置对比

| 配置 | 脚本 | 显存需求 | 批处理大小 | 推理速度 | 推荐用途 |
|------|------|----------|------------|----------|----------|
| **单卡 4090** | `inference_single_4090.sh` | 24GB | 8 | 慢 | 快速测试 |
| **双卡 4090** | `inference_2x4090.sh` | 48GB | 32 | 中等 | 完整评估 |
| **8卡 H100/A100** | `inference.sh` | 640GB | 128 | 快 | 生产环境 |

## 🎯 推荐配置：2x RTX 4090

对于你的硬件（2张RTX 4090），我们提供了专门优化的配置：

### 快速开始

```bash
bash examples/train/pixel_reasoner/inference_2x4090.sh
```

### 关键参数说明

```bash
# GPU配置
n_gpus_per_node=2              # 使用2张GPU
gpu_memory_utilization=0.65    # 使用65%显存（约15.6GB/卡）

# 批处理配置
batch_size=32                  # 总批处理大小
val_batch_size=32
n=4                           # 每个prompt采样4次（降低到4以节省内存）

# Token长度限制（针对24GB显存优化）
max_prompt_length=16384       # 降低到16K（原32K）
max_response_length=8192      # 降低到8K（原32K）
max_obs_length=4096           # 降低到4K（原8K）

# VLLM优化
tensor_model_parallel_size=1  # 3B模型不需要模型并行
max_num_seqs=32               # 同时处理的序列数
max_num_batched_tokens=3000   # 批处理token上限

# 性能优化
do_offload=False              # RTX 4090显存足够，无需offload
use_dynamic_bsz=True          # 启用动态批处理以提高利用率
```

## 📊 预期性能

### 2x RTX 4090

- **显存占用**: ~15-18GB/卡
- **推理速度**: ~3-5 samples/sec
- **完整InfoVQA评估**: 约30-60分钟
- **建议数据集顺序**: InfoVQA → TallyQA → VStar → MVBench

### 单卡 RTX 4090（快速测试）

- **显存占用**: ~12-15GB
- **推理速度**: ~1-2 samples/sec
- **小批量测试**: 建议用于快速验证

## 🔧 调优建议

### 如果遇到 OOM (显存不足)

#### 方案1: 降低批处理大小
```bash
# 编辑 inference_2x4090.sh
batch_size=16          # 从32降到16
val_batch_size=16
max_num_seqs=16        # 从32降到16
```

#### 方案2: 降低token长度
```bash
max_prompt_length=8192     # 从16384降到8192
max_response_length=4096   # 从8192降到4096
data.filter_overlong_prompts=True  # 过滤过长样本
```

#### 方案3: 降低GPU内存利用率
```bash
gpu_memory_utilization=0.55  # 从0.65降到0.55
```

#### 方案4: 启用 eager 模式（牺牲速度换显存）
```bash
actor_rollout_ref.rollout.enforce_eager=True
```

### 如果想提升速度

#### 方案1: 增加批处理大小（如果显存允许）
```bash
batch_size=48          # 从32提升到48
val_batch_size=48
max_num_seqs=48
```

#### 方案2: 提高GPU内存利用率
```bash
gpu_memory_utilization=0.75  # 从0.65提升到0.75
```

#### 方案3: 减少采样数
```bash
n=2  # 从4降到2，速度提升一倍
```

## 💡 实用技巧

### 1. 监控GPU使用情况

运行推理时，在另一个终端监控：
```bash
watch -n 1 nvidia-smi
```

关注：
- **GPU利用率**: 应该接近100%
- **显存使用**: 应该在15-18GB范围内
- **温度**: 保持在80°C以下

### 2. 小批量测试

首次运行时，建议先用小批量测试：
```bash
# 修改脚本中的数据集路径，只加载少量样本
# 或者创建一个小的测试集
head -n 100 data/pixel_reasoner/info_vqa/test.parquet > data/pixel_reasoner/info_vqa/test_small.parquet
```

### 3. 日志保存

保存完整日志以便调试：
```bash
bash examples/train/pixel_reasoner/inference_2x4090.sh 2>&1 | tee inference_log_$(date +%Y%m%d_%H%M%S).txt
```

### 4. 后台运行

长时间推理建议使用screen或tmux：
```bash
screen -S pixel_inference
bash examples/train/pixel_reasoner/inference_2x4090.sh
# Ctrl+A, D 来分离会话
```

## 📋 配置对比表

### 内存相关参数

| 参数 | 8x H100 | 2x 4090 | 1x 4090 | 说明 |
|------|---------|---------|---------|------|
| `batch_size` | 128 | 32 | 8 | 总批处理大小 |
| `n` | 8 | 4 | 2 | 每个prompt采样数 |
| `max_prompt_length` | 32768 | 16384 | 8192 | 最大输入长度 |
| `max_response_length` | 32768 | 8192 | 4096 | 最大输出长度 |
| `max_obs_length` | 8192 | 4096 | 2048 | 最大观察长度 |
| `max_num_seqs` | 128 | 32 | 16 | VLLM并发序列数 |
| `gpu_memory_utilization` | 0.8 | 0.65 | 0.6 | GPU显存利用率 |

### 计算资源参数

| 参数 | 8x H100 | 2x 4090 | 1x 4090 | 说明 |
|------|---------|---------|---------|------|
| `n_gpus_per_node` | 8 | 2 | 1 | GPU数量 |
| `tensor_model_parallel_size` | 2 | 1 | 1 | 模型并行度 |
| `workers_per_tool` | 4 | 2 | 1 | 工具服务器worker数 |
| `dataloader_num_workers` | 2 | 2 | 1 | 数据加载worker数 |

## 🎬 完整示例

### 场景1: 快速验证模型（单卡）

```bash
# 准备小数据集
python examples/data_preprocess/pixel_reasoner/infovqa.py \
    --dataset_path=JasperHaozhe/InfoVQA-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/info_vqa

# 运行单卡推理（5-10分钟）
bash examples/train/pixel_reasoner/inference_single_4090.sh
```

### 场景2: 完整评估（双卡）

```bash
# 准备所有数据集
for dataset in infovqa tallyqa vstar mvbench; do
    python examples/data_preprocess/pixel_reasoner/${dataset}.py \
        --dataset_path=JasperHaozhe/$(echo $dataset | sed 's/^./\u&/')*-EvalData-PixelReasoner \
        --split=test \
        --local_dir=data/pixel_reasoner/${dataset}
done

# 运行双卡推理（30-60分钟/数据集）
bash examples/train/pixel_reasoner/inference_2x4090.sh
```

### 场景3: 自定义图像测试

```bash
# 使用单样本测试脚本
python examples/train/pixel_reasoner/quick_inference.py \
    --model_path VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --image_path my_test_image.jpg \
    --question "请详细描述图片内容"
```

## ❓ 常见问题

### Q1: 为什么2卡比单卡快不到2倍？

A: 因为：
1. 通信开销（GPU间数据传输）
2. 模型加载开销（每个GPU都需要加载模型副本）
3. VLLM的并行效率损失

但batch_size可以增大4倍，总体吞吐量显著提升。

### Q2: 可以只使用1张4090吗？

A: 可以！使用 `inference_single_4090.sh`，但：
- batch_size会很小（8）
- 推理会慢很多
- 适合快速测试，不适合完整评估

### Q3: RTX 4090和A100/H100差距大吗？

A: 主要差异：
- **显存**: 4090是24GB，A100是40/80GB，H100是80GB
- **计算**: H100约是4090的2-3倍快
- **精度**: H100支持FP8，更快
- **对于3B模型**: 4090完全够用，只是慢一些

### Q4: 能否使用PyTorch原生推理？

A: 可以使用 `quick_inference.py`，但：
- 不支持工具调用（需要工具服务器）
- 适合理解模型基础能力
- 完整功能还是要用RL框架

## 📚 相关文档

- [快速开始指南](../../../PIXEL_REASONER_QUICKSTART.md)
- [详细推理指南](./INFERENCE_GUIDE.md)
- [训练文档](./README.md)

## 🚀 最终建议

**对于你的2x RTX 4090配置：**

1. ✅ 使用 `inference_2x4090.sh`
2. ✅ 从 InfoVQA 数据集开始测试
3. ✅ 监控nvidia-smi确保显存使用正常
4. ✅ 如果OOM，首先降低batch_size
5. ✅ 预期每个数据集需要30-60分钟

祝推理顺利！🎉
