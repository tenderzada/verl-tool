# Pixel Reasoner 快速推理验证指南

本指南帮助你快速使用预训练的 Pixel-Reasoner 模型进行推理验证。

## 📋 概述

**Pixel-Reasoner** 是一个基于强化学习训练的视觉语言模型，能够使用工具（图像裁剪、帧选择）来解决复杂的视觉推理问题。

**预训练模型**: `VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100`

## 🚀 快速开始（三步走）

### 步骤 1: 安装依赖

```bash
# 安装VerlTool依赖（参考主README）
# 重要：Pixel-Reasoner需要transformers<4.53.0
pip install "transformers<4.53.0"
```

### 步骤 2: 准备评估数据

选择一个数据集进行测试（推荐从InfoVQA开始）：

```bash
# InfoVQA - 信息图表问答
python examples/data_preprocess/pixel_reasoner/infovqa.py \
    --dataset_path=JasperHaozhe/InfoVQA-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/info_vqa
```

其他可选数据集：
- **TallyQA**: 计数问答
- **VStar**: 空间推理
- **MVBench**: 视频理解

<details>
<summary>查看其他数据集准备命令</summary>

```bash
# TallyQA
python examples/data_preprocess/pixel_reasoner/tallyqa.py \
    --dataset_path=JasperHaozhe/TallyQA-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/tallyqa

# VStar
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path=JasperHaozhe/VStar-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/vstar

# MVBench
python examples/data_preprocess/pixel_reasoner/mvbench.py \
    --dataset_path=JasperHaozhe/MVBench-EvalData-PixelReasoner \
    --split=test \
    --local_dir=data/pixel_reasoner/mvbench
```
</details>

### 步骤 3: 根据你的硬件选择推理脚本

#### 如果你有 2x RTX 4090 (推荐) ⭐

```bash
bash examples/train/pixel_reasoner/inference_2x4090.sh
```

#### 如果你有 1x RTX 4090 (快速测试)

```bash
bash examples/train/pixel_reasoner/inference_single_4090.sh
```

#### 如果你有 8x H100/A100 (完整配置)

```bash
bash examples/train/pixel_reasoner/inference.sh
```

**💡 不知道选哪个？** 查看 [RTX4090配置指南](examples/train/pixel_reasoner/RTX4090_CONFIG.md)

## 💻 硬件配置对比

| 配置 | 脚本 | 显存 | Batch Size | 速度 | 适用场景 |
|------|------|------|------------|------|----------|
| **2x RTX 4090** ⭐ | `inference_2x4090.sh` | 48GB | 32 | 中等 | **推荐配置** |
| **1x RTX 4090** | `inference_single_4090.sh` | 24GB | 8 | 慢 | 快速验证 |
| **8x H100/A100** | `inference.sh` | 640GB | 128 | 快 | 生产环境 |

### 方式 1: 2x RTX 4090 批量评估（推荐）⭐

针对消费级GPU优化的配置：

```bash
bash examples/train/pixel_reasoner/inference_2x4090.sh
```

**优势**:
- 专为RTX 4090优化的参数
- 性能与成本的最佳平衡
- 完整的性能指标
- 自动启动工具服务器

**配置**:
- 默认使用 InfoVQA 数据集
- 显存占用: ~15-18GB/卡
- 预期时间: 30-60分钟/数据集
- [详细配置说明](examples/train/pixel_reasoner/RTX4090_CONFIG.md)

### 方式 2: 单样本测试（推荐用于快速验证）

使用 `quick_inference.py` 在单张图片上测试：

```bash
python examples/train/pixel_reasoner/quick_inference.py \
    --model_path VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --image_path /path/to/your/image.jpg \
    --question "What is shown in this image?"
```

**优势**:
- 快速验证模型工作
- 适合调试和理解模型行为
- 单GPU即可运行

**注意**: 此方式不包含工具调用功能，仅做基础推理测试。

## 🛠️ 硬件要求

### ⭐ RTX 4090 配置（消费级GPU，推荐）

**2x RTX 4090** (最佳性价比)
- 显存: 2 × 24GB = 48GB
- 内存: 64GB+ RAM
- 存储: 500GB+ SSD
- 性能: 完整评估约30-60分钟/数据集
- 脚本: `inference_2x4090.sh`

**1x RTX 4090** (最小配置)
- 显存: 24GB
- 内存: 32GB+ RAM
- 存储: 200GB+ SSD
- 性能: 适合快速测试
- 脚本: `inference_single_4090.sh`

### 数据中心配置

**8x H100/A100** (生产环境)
- 显存: 8 × 80GB = 640GB
- 内存: 512GB+ RAM
- 存储: 2TB+ SSD
- 性能: 最快，适合大规模评估
- 脚本: `inference.sh`

## 🔧 常见配置修改

### 修改评估数据集

编辑相应的推理脚本（如 `inference_2x4090.sh`）中的 `val_data` 变量：

```bash
# InfoVQA（默认）
val_data=[$(pwd)/data/pixel_reasoner/info_vqa/test.parquet]

# 或切换到其他数据集
# val_data=[$(pwd)/data/pixel_reasoner/tallyqa/test.parquet]
# val_data=[$(pwd)/data/pixel_reasoner/vstar/test.parquet]
# val_data=[$(pwd)/data/pixel_reasoner/mvbench/test.parquet]
```

### RTX 4090 专属优化

**完整配置说明**: 请查看 [RTX4090_CONFIG.md](examples/train/pixel_reasoner/RTX4090_CONFIG.md)

**快速调优**（如果遇到显存不足）：
```bash
# 编辑 inference_2x4090.sh
batch_size=16              # 从32降到16
gpu_memory_utilization=0.55  # 从0.65降到0.55
max_prompt_length=8192     # 从16384降到8192
```

### 内存优化

如果遇到OOM错误：

```bash
# 降低GPU内存利用率
gpu_memory_utilization=0.6  # 默认0.8

# 减小batch size
batch_size=64  # 默认128

# 启用动态batch size
use_dynamic_bsz=True
```

## 📊 模型能力

Pixel-Reasoner 模型支持以下工具：

1. **crop_image**: 根据边界框裁剪图像
2. **select_frames**: 从视频选择帧
3. **zoom_in**: 放大特定区域
4. **crop_image_normalized**: 归一化坐标裁剪

模型会自动决定是否使用工具来辅助回答问题。

## 📁 文件结构

```
examples/train/pixel_reasoner/
├── README.md              # 训练文档
├── INFERENCE_GUIDE.md     # 详细推理指南（新增）
├── inference.sh           # 批量推理脚本（新增）
├── quick_inference.py     # 单样本测试脚本（新增）
├── eval.sh                # 原始评估脚本
├── train_*.sh             # 训练脚本
└── ...
```

## 🐛 常见问题

### Q: 模型下载速度慢？
A: 使用HuggingFace镜像或手动下载模型到本地，然后修改 `model_name` 为本地路径。

### Q: 推理结果在哪里？
A: 默认输出到控制台。可以通过修改logger配置保存到wandb或tensorboard：
```bash
trainer.logger=['console','wandb']
```

### Q: 如何使用自己的图像？
A: 使用 `quick_inference.py` 脚本，提供你的图像路径和问题。

### Q: 如何只测试几个样本？
A: 创建一个小的测试数据集，或使用 `quick_inference.py` 进行单样本测试。

## 📚 更多资源

- **详细推理指南**: [INFERENCE_GUIDE.md](examples/train/pixel_reasoner/INFERENCE_GUIDE.md)
- **训练指南**: [examples/train/pixel_reasoner/README.md](examples/train/pixel_reasoner/README.md)
- **论文**: [Pixel Reasoner](https://arxiv.org/abs/2505.15966)
- **原始PR**: [#63](https://github.com/TIGER-AI-Lab/verl-tool/pull/63)

## ✅ 验证清单

完成推理验证后，确认：

- [ ] 模型成功加载
- [ ] 工具服务器正常启动
- [ ] 推理产生合理输出
- [ ] 评估指标符合预期
- [ ] 资源使用在可接受范围内

## 🎯 下一步

1. **微调模型**: 在你的数据上进行训练
2. **探索参数**: 调整温度、top_p等生成参数
3. **性能优化**: 根据硬件优化batch size和并行策略
4. **应用开发**: 集成到你的应用中

---

**需要帮助?** 查看主仓库文档或提交issue: https://github.com/TIGER-AI-Lab/verl-tool/issues
