# VStar 快速验证指南（2x RTX 4090，仅推理）

本指南适用于只想快速验证Pixel Reasoner模型在VStar数据集上的表现，不需要完整训练环境的用户。

## ✅ 适用场景

- ✓ 你有 2张 RTX 4090（或类似的24GB显存GPU）
- ✓ 只需要运行推理验证，不需要训练
- ✓ 想快速测试模型效果
- ✓ 网络受限，需要手动下载

## 📊 资源需求

| 项目 | 需求 |
|------|------|
| GPU | 2x RTX 4090 (24GB) |
| 内存 | 64GB+ RAM |
| 存储 | ~10GB (模型6GB + 数据200MB + 依赖) |
| 时间 | 30-45分钟（VStar评估） |

## 🚀 快速开始（4步）

### 步骤1: 安装最小化依赖

```bash
# 创建环境
conda create -n pixel_reasoner python=3.10
conda activate pixel_reasoner

# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装推理依赖（仅需8GB，不需要完整verl-tool）
pip install transformers==4.47.0 accelerate pillow qwen-vl-utils \
    numpy opencv-python regex huggingface_hub vllm==0.11.0
```

**详细说明**: 参考 [MINIMAL_DEPS.md](./MINIMAL_DEPS.md)

### 步骤2: 下载模型和VStar数据集

#### 方式A: 使用huggingface-cli（推荐）

```bash
# 国内用户设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型（~6GB）
huggingface-cli download \
    VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --local-dir ./models/pixel_reasoner_3b \
    --resume-download

# 下载VStar数据集（~200MB）
huggingface-cli download \
    JasperHaozhe/VStar-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/vstar_raw \
    --resume-download
```

#### 方式B: 使用Python脚本

```bash
python examples/train/pixel_reasoner/download_models.py \
    --datasets vstar
```

**详细说明**: 参考 [MANUAL_DOWNLOAD.md](./MANUAL_DOWNLOAD.md)

### 步骤3: 预处理VStar数据集

```bash
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path ./data/pixel_reasoner/vstar_raw \
    --split test \
    --local_dir data/pixel_reasoner/vstar
```

验证生成的文件：
```bash
ls -lh data/pixel_reasoner/vstar/test.parquet
# 应该看到 test.parquet 文件
```

### 步骤4: 运行VStar验证

```bash
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```

## ⏱️ 时间估算

| 步骤 | 时间 |
|------|------|
| 安装依赖 | 5-10分钟 |
| 下载模型 | 10-30分钟（国内镜像） |
| 下载VStar | 1-5分钟 |
| 预处理数据 | 2-5分钟 |
| 运行评估 | 30-45分钟 |
| **总计** | **~1-1.5小时** |

## 📁 目录结构

完成后应该有：

```
verl-tool/
├── models/
│   └── pixel_reasoner_3b/           # 模型文件（~6GB）
│       ├── config.json
│       ├── model.safetensors
│       └── ...
│
└── data/
    └── pixel_reasoner/
        ├── vstar_raw/                # 原始下载
        │   └── ...
        └── vstar/                    # 预处理后
            └── test.parquet          # 用于推理
```

## 🎯 一键脚本（国内用户）

创建 `setup_vstar.sh`:

```bash
#!/bin/bash
set -e

echo "VStar Quick Setup Script"
echo "========================"

# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 1. 下载模型
echo "Step 1/3: Downloading model..."
huggingface-cli download \
    VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --local-dir ./models/pixel_reasoner_3b \
    --resume-download

# 2. 下载数据集
echo "Step 2/3: Downloading VStar dataset..."
huggingface-cli download \
    JasperHaozhe/VStar-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/vstar_raw \
    --resume-download

# 3. 预处理
echo "Step 3/3: Processing dataset..."
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path ./data/pixel_reasoner/vstar_raw \
    --split test \
    --local_dir data/pixel_reasoner/vstar

echo ""
echo "Setup complete! Run inference with:"
echo "bash examples/train/pixel_reasoner/vstar_quick_verify.sh"
```

运行：
```bash
chmod +x setup_vstar.sh
./setup_vstar.sh
```

## 📊 预期结果

运行完成后，你应该看到：

```
========================================
VStar Verification Complete!
========================================

Evaluation Results:
  Total samples: XXX
  Average score: X.XX
  ...
```

## 🐛 常见问题

### Q1: 显存不足（OOM）

**症状**: CUDA out of memory

**解决**: 编辑 `vstar_quick_verify.sh`，降低batch size:
```bash
batch_size=16          # 从32降到16
gpu_memory_utilization=0.55  # 从0.65降到0.55
```

### Q2: 模型下载失败

**症状**: 连接超时或速度很慢

**解决**:
```bash
# 国内用户务必设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用断点续传
huggingface-cli download ... --resume-download
```

### Q3: 数据预处理报错

**症状**: `FileNotFoundError` 或 `KeyError`

**解决**: 检查原始数据是否完整下载
```bash
ls -lh ./data/pixel_reasoner/vstar_raw/
# 应该有数据文件
```

### Q4: VLLM安装失败

**症状**: `pip install vllm` 失败

**解决**:
```bash
# 确保CUDA版本正确
nvidia-smi  # 查看CUDA版本

# 使用对应的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.11.0
```

### Q5: 推理脚本找不到模型

**症状**: `Model not found`

**解决**: 创建本地路径配置
```bash
# 创建 local_paths.sh
cat > local_paths.sh << 'EOF'
#!/bin/bash
export PIXEL_REASONER_MODEL="./models/pixel_reasoner_3b"
export VSTAR_DATA="./data/pixel_reasoner/vstar_raw"
EOF

chmod +x local_paths.sh
source local_paths.sh
```

## 💡 优化技巧

### 1. 监控GPU使用

在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

正常情况：
- GPU利用率: 90-100%
- 显存使用: 15-18GB/卡
- 温度: <80°C

### 2. 保存推理日志

```bash
bash vstar_quick_verify.sh 2>&1 | tee vstar_log_$(date +%Y%m%d_%H%M%S).txt
```

### 3. 后台运行（长时间推理）

```bash
# 使用screen
screen -S vstar
bash vstar_quick_verify.sh
# Ctrl+A, D 分离

# 或使用nohup
nohup bash vstar_quick_verify.sh > vstar.log 2>&1 &
```

### 4. 测试单个样本（调试）

```bash
# 修改数据集为debug模式
# 在预处理脚本中只取前10个样本
python -c "
import pyarrow.parquet as pq
import pyarrow as pa

table = pq.read_table('data/pixel_reasoner/vstar/test.parquet')
small_table = table.slice(0, 10)
pq.write_table(small_table, 'data/pixel_reasoner/vstar/test_debug.parquet')
"

# 修改vstar_quick_verify.sh中的val_data
val_data=[$(pwd)/data/pixel_reasoner/vstar/test_debug.parquet]
```

## 🔍 验证检查清单

运行前检查：

- [ ] 已安装Python 3.10
- [ ] 已安装CUDA和NVIDIA驱动
- [ ] 已安装所有推理依赖
- [ ] 已下载模型到 `./models/pixel_reasoner_3b/`
- [ ] 已下载VStar数据到 `./data/pixel_reasoner/vstar_raw/`
- [ ] 已预处理生成 `./data/pixel_reasoner/vstar/test.parquet`
- [ ] 有足够磁盘空间（10GB+）
- [ ] 2张GPU可用（`nvidia-smi`确认）

## 📈 性能基准

参考性能（2x RTX 4090）:

| 指标 | 预期值 |
|------|--------|
| 推理速度 | 3-5 samples/sec |
| 单卡显存 | 15-18GB |
| GPU利用率 | 90-100% |
| VStar完整评估 | 30-45分钟 |

## 🔄 下一步

完成VStar验证后，可以：

1. **测试其他数据集**
   - InfoVQA（信息图表）
   - TallyQA（计数任务）
   - MVBench（视频理解）

2. **调整推理参数**
   - temperature（生成随机性）
   - top_p（采样策略）
   - max_turns（工具调用次数）

3. **在自己的数据上测试**
   - 使用 `quick_inference.py` 测试单张图片
   - 准备自己的数据集格式

## 📚 相关文档

- [最小化依赖安装](./MINIMAL_DEPS.md) - 仅推理的依赖列表
- [手动下载指南](./MANUAL_DOWNLOAD.md) - 详细下载步骤
- [RTX4090配置](./RTX4090_CONFIG.md) - GPU配置调优
- [完整快速开始](../../../PIXEL_REASONER_QUICKSTART.md) - 所有数据集

## 🎯 最简化流程总结

```bash
# 1. 环境（5分钟）
conda create -n pixel_reasoner python=3.10 -y
conda activate pixel_reasoner
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 accelerate pillow qwen-vl-utils \
    numpy opencv-python regex huggingface_hub vllm==0.11.0

# 2. 下载（15-30分钟，国内用镜像）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --local-dir ./models/pixel_reasoner_3b --resume-download
huggingface-cli download JasperHaozhe/VStar-EvalData-PixelReasoner \
    --repo-type dataset --local-dir ./data/pixel_reasoner/vstar_raw --resume-download

# 3. 预处理（5分钟）
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path ./data/pixel_reasoner/vstar_raw \
    --split test --local_dir data/pixel_reasoner/vstar

# 4. 验证（30-45分钟）
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```

**总时间**: ~1-1.5小时

祝验证顺利！🎉
