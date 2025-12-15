# Pixel Reasoner Inference Guide

本指南将帮助你快速使用预训练的 Pixel-Reasoner 模型进行推理验证。

## 模型信息

预训练模型：`VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100`

这是一个3B参数的视觉语言模型，经过GRPO算法训练，能够使用工具（图像裁剪和帧选择）来解决复杂的视觉推理问题。

## 环境准备

### 1. 安装依赖

确保已经安装了VerlTool的所有依赖。如果还未安装，请参考主仓库的[安装指南](../../../assets/docs/install.md)。

**重要**: Pixel-Reasoner需要 `transformers<4.53.0`（因为Qwen2.5-VL模型代码的兼容性）

```bash
pip install "transformers<4.53.0"
```

### 2. 准备评估数据集

选择你想要评估的数据集并下载预处理：

#### InfoVQA (推荐用于快速验证)
```bash
python examples/data_preprocess/pixel_reasoner/infovqa.py \
    --dataset_path=JasperHaozhe/InfoVQA-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/info_vqa
```

#### TallyQA
```bash
python examples/data_preprocess/pixel_reasoner/tallyqa.py \
    --dataset_path=JasperHaozhe/TallyQA-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/tallyqa
```

#### VStar
```bash
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path=JasperHaozhe/VStar-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/vstar
```

#### MVBench
```bash
python examples/data_preprocess/pixel_reasoner/mvbench.py \
    --dataset_path=JasperHaozhe/MVBench-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/mvbench
```

## 运行推理

### 方法1: 使用推理脚本 (推荐)

我们提供了一个专门的推理脚本 `inference.sh`，它已经配置好了所有必要的参数：

```bash
# 默认使用InfoVQA数据集
bash examples/train/pixel_reasoner/inference.sh
```

**修改数据集**: 编辑 `inference.sh` 中的 `val_data` 变量来选择不同的数据集。

**调整GPU数量**: 如果你的GPU数量少于8张，修改 `n_gpus_per_node` 变量：
```bash
n_gpus_per_node=4  # 例如，如果你有4张GPU
```

### 方法2: 使用原始评估脚本

修改 `eval.sh` 中的模型路径：

```bash
# 取消注释第12行并注释第14行
model_name=VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100
# model_name=TIGER-Lab/PixelReasoner-WarmStart
```

然后运行：
```bash
bash examples/train/pixel_reasoner/eval.sh
```

## 硬件要求

- **推荐配置**: 8张 H100/A100 80GB GPU
- **最小配置**: 4张 A100 40GB GPU (需要调整batch_size和其他内存相关参数)

### 内存优化技巧

如果遇到内存问题，可以尝试：

1. **降低GPU内存利用率**:
   ```bash
   gpu_memory_utilization=0.6  # 默认是0.8
   ```

2. **启用eager模式**:
   ```bash
   actor_rollout_ref.rollout.enforce_eager=True
   ```

3. **减小batch size**:
   ```bash
   batch_size=64  # 默认是128
   ```

4. **启用offload**:
   ```bash
   do_offload=True  # 已默认启用
   ```

## 工具使用

Pixel-Reasoner模型能够使用以下工具：

1. **crop_image**: 根据边界框坐标裁剪图像
2. **select_frames**: 从视频中选择特定帧
3. **zoom_in**: 放大图像特定区域
4. **crop_image_normalized**: 使用归一化坐标裁剪图像

推理过程中，模型会自动调用这些工具来辅助视觉推理任务。

## 输出结果

推理结果将保存在：
- 控制台输出：实时显示推理进度和结果
- 日志文件：在项目根目录下会生成相应的日志

## 常见问题

### Q1: 如何使用调试模式进行小规模测试？

创建一个小的测试集或修改 `val_data` 为调试数据：
```bash
val_data=[$(pwd)/data/pixel_reasoner/info_vqa_debug/test.parquet]
```

### Q2: 推理速度很慢怎么办？

- 使用异步rollout模式（已默认启用）
- 减少 `n` 参数（每个prompt的采样数）
- 增加 `max_num_batched_tokens`

### Q3: 如何保存推理结果？

推理结果会通过reward_manager自动评估。如果需要保存详细输出，可以修改logger配置：
```bash
trainer.logger=['console','wandb']  # 添加wandb或tensorboard
```

## 参考资料

- [Pixel Reasoner论文](https://arxiv.org/abs/2505.15966)
- [原始PR](https://github.com/TIGER-AI-Lab/verl-tool/pull/63)
- [训练指南](./README.md)

## 下一步

完成推理验证后，你可以：
1. 在自己的数据上进行测试
2. 微调模型以适应特定任务
3. 探索不同的推理策略（温度、top_p等参数）

如有问题，请查阅主仓库的文档或提交issue。
