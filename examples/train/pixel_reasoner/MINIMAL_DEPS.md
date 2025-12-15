# Pixel Reasoner 最小化依赖安装（仅推理）

如果你只需要运行推理验证，不需要训练，可以使用这个精简的依赖列表。

## 🎯 核心依赖（必需）

```bash
# 基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 模型和推理
pip install transformers==4.47.0  # <4.53.0
pip install accelerate
pip install pillow
pip install qwen-vl-utils

# 推理加速（VLLM）
pip install vllm==0.11.0

# 工具调用支持
pip install numpy
pip install opencv-python
pip install regex

# HuggingFace下载（如果需要）
pip install huggingface_hub
```

## 📦 一键安装（推荐）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 accelerate pillow qwen-vl-utils vllm==0.11.0 \
    numpy opencv-python regex huggingface_hub
```

## ⏭️ 可以跳过的依赖（仅训练需要）

以下依赖仅在训练时需要，推理可以跳过：

```bash
# 训练框架（不需要）
# pip install verl
# pip install deepspeed
# pip install flash-attn

# 数据处理（如果手动下载数据，部分可跳过）
# pip install datasets
# pip install pyarrow
# pip install pandas

# 日志和监控（不需要）
# pip install wandb
# pip install tensorboard

# 分布式训练（不需要）
# pip install ray[default]
```

## 🚀 快速安装流程（仅推理）

### 步骤1: 创建虚拟环境
```bash
conda create -n pixel_reasoner python=3.10
conda activate pixel_reasoner
```

### 步骤2: 安装PyTorch
```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 步骤3: 安装推理依赖
```bash
pip install transformers==4.47.0 accelerate pillow qwen-vl-utils \
    numpy opencv-python regex huggingface_hub
```

### 步骤4: 安装VLLM（推理加速）
```bash
pip install vllm==0.11.0
```

### 步骤5: 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import vllm; print(f'VLLM: {vllm.__version__}')"
```

## 📊 依赖大小对比

| 方案 | 安装大小 | 安装时间 | 适用场景 |
|------|----------|----------|----------|
| **最小化（仅推理）** | ~8GB | 5-10分钟 | ✓ 推荐 |
| 完整verl-tool | ~15GB | 20-30分钟 | 训练+推理 |

## 🔧 可选优化

### Flash Attention（可选，提升速度）
```bash
# 如果想要更快的推理速度，可以安装
pip install flash-attn --no-build-isolation
```

### 数据处理（如果需要预处理数据集）
```bash
pip install datasets pyarrow pandas
```

## ⚠️ 已知限制

使用最小化依赖时：
- ✓ 可以运行推理脚本（`inference_2x4090.sh`）
- ✓ 可以使用工具服务器（图像裁剪、帧选择）
- ✗ 不能运行训练脚本
- ✗ 不能使用完整的verl-tool训练功能

## 🐛 问题排查

### 问题1: `ModuleNotFoundError: No module named 'verl'`
**解决**: 推理不需要verl，忽略此错误。如果脚本报错，使用 `inference_2x4090_minimal.sh`

### 问题2: `transformers版本冲突`
**解决**: 严格使用4.47.0版本
```bash
pip install transformers==4.47.0 --force-reinstall
```

### 问题3: VLLM安装失败
**解决**: 确保CUDA版本正确
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 使用对应的PyTorch版本
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 💡 验证推理环境

创建测试脚本 `test_inference_env.py`:

```python
#!/usr/bin/env python3
import sys

def check_package(name, import_name=None):
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError:
        print(f"✗ {name}: NOT INSTALLED")
        return False

print("=" * 60)
print("Pixel Reasoner Inference Environment Check")
print("=" * 60)
print()

required = [
    ("PyTorch", "torch"),
    ("Transformers", "transformers"),
    ("VLLM", "vllm"),
    ("Pillow", "PIL"),
    ("NumPy", "numpy"),
    ("OpenCV", "cv2"),
]

optional = [
    ("HuggingFace Hub", "huggingface_hub"),
    ("Accelerate", "accelerate"),
    ("qwen-vl-utils", "qwen_vl_utils"),
]

print("Required packages:")
all_ok = all(check_package(name, imp) for name, imp in required)
print()

print("Optional packages:")
for name, imp in optional:
    check_package(name, imp)
print()

# Check CUDA
import torch
print("CUDA Information:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print()

if all_ok:
    print("✓ All required packages installed!")
    print("You can proceed with inference.")
else:
    print("✗ Some required packages missing!")
    print("Please install them first.")
    sys.exit(1)
```

运行验证：
```bash
python test_inference_env.py
```

## 📚 相关文档

- [手动下载指南](./MANUAL_DOWNLOAD.md) - 手动下载模型和数据集
- [VStar快速验证](./VSTAR_QUICKSTART.md) - VStar数据集快速验证
- [RTX4090配置](./RTX4090_CONFIG.md) - GPU配置说明

## 🎯 下一步

安装完依赖后：
1. 下载模型和VStar数据集（见 MANUAL_DOWNLOAD.md）
2. 预处理数据
3. 运行推理验证

```bash
# 快速开始
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```
