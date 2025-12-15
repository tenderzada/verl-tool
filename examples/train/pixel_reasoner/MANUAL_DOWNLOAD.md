# 手动下载模型和VStar数据集

本指南专门为需要手动下载模型和VStar数据集的用户准备。

## 📥 下载清单

| 内容 | 大小 | 仓库地址 |
|------|------|----------|
| Pixel Reasoner 3B 模型 | ~6GB | `VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100` |
| VStar 评估数据集 | ~200MB | `JasperHaozhe/VStar-EvalData-PixelReasoner` |
| **总计** | **~6.2GB** | - |

## 🚀 方式1: huggingface-cli（推荐）

### 安装CLI工具
```bash
pip install -U huggingface_hub[cli]
```

### 登录（可选，用于更快下载）
```bash
huggingface-cli login
# 输入你的HuggingFace token
```

### 🇨🇳 国内用户：设置镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 下载模型
```bash
huggingface-cli download \
    VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --local-dir ./models/pixel_reasoner_3b \
    --resume-download
```

### 下载VStar数据集
```bash
huggingface-cli download \
    JasperHaozhe/VStar-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/vstar_raw \
    --resume-download
```

## 🐍 方式2: Python脚本

### 仅下载模型+VStar
```bash
python examples/train/pixel_reasoner/download_models.py \
    --datasets vstar
```

### 或创建自定义脚本

创建 `download_vstar.py`:

```python
#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os

# 国内用户设置镜像
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("Downloading Pixel Reasoner Model...")
model_path = snapshot_download(
    repo_id="VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100",
    local_dir="./models/pixel_reasoner_3b",
    resume_download=True,
    local_dir_use_symlinks=False,
)
print(f"✓ Model downloaded to: {model_path}")

print("\nDownloading VStar Dataset...")
dataset_path = snapshot_download(
    repo_id="JasperHaozhe/VStar-EvalData-PixelReasoner",
    repo_type="dataset",
    local_dir="./data/pixel_reasoner/vstar_raw",
    resume_download=True,
    local_dir_use_symlinks=False,
)
print(f"✓ Dataset downloaded to: {dataset_path}")

print("\n" + "=" * 60)
print("Download Complete!")
print("=" * 60)
print(f"\nModel: {model_path}")
print(f"Dataset: {dataset_path}")
print("\nNext steps:")
print("1. Process dataset: python examples/data_preprocess/pixel_reasoner/vstar.py \\")
print(f"     --dataset_path {dataset_path} \\")
print("     --split test \\")
print("     --local_dir data/pixel_reasoner/vstar")
print("2. Run inference: bash examples/train/pixel_reasoner/vstar_quick_verify.sh")
```

运行：
```bash
python download_vstar.py
```

## 🌐 方式3: 浏览器下载（备选）

如果CLI和Python都失败，可以通过浏览器手动下载。

### 下载模型

1. 访问：https://huggingface.co/VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100/tree/main

2. 下载必需文件：
   - `config.json`
   - `model.safetensors` 或 `pytorch_model.bin`（~6GB主文件）
   - `tokenizer_config.json`
   - `tokenizer.json`
   - `special_tokens_map.json`
   - `generation_config.json`
   - `preprocessor_config.json`
   - 所有 `.py` 文件（如果有）

3. 保存到目录：
   ```
   ./models/pixel_reasoner_3b/
   ```

### 下载VStar数据集

1. 访问：https://huggingface.co/datasets/JasperHaozhe/VStar-EvalData-PixelReasoner

2. 点击 "Files" 标签

3. 下载所有文件到：
   ```
   ./data/pixel_reasoner/vstar_raw/
   ```

## 🔄 下载后验证

### 验证模型完整性
```bash
ls -lh ./models/pixel_reasoner_3b/

# 应该包含：
# config.json
# model.safetensors (~6GB)
# tokenizer相关文件
```

### 验证数据集
```bash
ls -lh ./data/pixel_reasoner/vstar_raw/

# 应该包含parquet或json格式的数据文件
```

### 快速测试模型加载
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

model_path = "./models/pixel_reasoner_3b"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cpu"  # 先用CPU测试
)

print("✓ Model loaded successfully!")
```

## 📁 目录结构

下载完成后应该有：

```
verl-tool/
├── models/
│   └── pixel_reasoner_3b/
│       ├── config.json
│       ├── model.safetensors (~6GB)
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── ...
│
└── data/
    └── pixel_reasoner/
        └── vstar_raw/
            ├── test.parquet  (或类似文件)
            └── ...
```

## 🔧 设置本地路径

### 创建路径配置文件

创建 `local_paths.sh`:

```bash
#!/bin/bash
# 本地路径配置

export PIXEL_REASONER_MODEL="./models/pixel_reasoner_3b"
export VSTAR_DATA="./data/pixel_reasoner/vstar_raw"

echo "Local paths configured:"
echo "  Model: $PIXEL_REASONER_MODEL"
echo "  VStar: $VSTAR_DATA"
```

### 加载配置
```bash
chmod +x local_paths.sh
source local_paths.sh
```

## 📊 预处理数据集

下载原始数据后，需要预处理成parquet格式：

```bash
# 加载路径
source local_paths.sh

# 预处理VStar数据集
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path $VSTAR_DATA \
    --split test \
    --local_dir data/pixel_reasoner/vstar

# 检查生成的文件
ls -lh data/pixel_reasoner/vstar/
# 应该看到 test.parquet
```

## 🎯 运行推理验证

预处理完成后，运行推理：

```bash
# 使用专门的VStar验证脚本
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```

## ⏱️ 下载时间估算

| 网络速度 | 下载时间 |
|----------|----------|
| 100 Mbps | ~10分钟 |
| 50 Mbps  | ~20分钟 |
| 10 Mbps  | ~1.5小时 |
| 1 Mbps   | ~15小时 |

**国内用户使用镜像**: 通常可以达到10-50 Mbps

## 🐛 常见问题

### Q1: 下载中断怎么办？
A: 使用 `--resume-download` 参数，或重新运行命令，会自动续传

### Q2: 提示认证失败？
A: 运行 `huggingface-cli login` 并输入token

### Q3: 国内下载很慢？
A: 务必设置 `export HF_ENDPOINT=https://hf-mirror.com`

### Q4: 浏览器下载如何加速？
A: 使用下载管理器（如IDM、aria2c）支持断点续传

### Q5: 如何验证下载完整？
A: 运行上面的"快速测试模型加载"代码

## 💡 下载优化技巧

### 1. 使用aria2c多线程下载（高级）

```bash
# 安装aria2c
sudo apt install aria2  # Ubuntu
brew install aria2      # macOS

# 获取文件直链（从HuggingFace页面）
# 然后使用aria2c下载
aria2c -x 16 -s 16 -k 1M \
    "https://huggingface.co/.../model.safetensors" \
    -o model.safetensors
```

### 2. 分时段下载

如果网络高峰期慢，可以在凌晨下载：

```bash
# 使用cron定时任务
# 在凌晨2点开始下载
0 2 * * * /path/to/download_script.sh
```

### 3. 使用代理（如果需要）

```bash
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

huggingface-cli download ...
```

## 📚 下一步

下载完成后：

1. ✓ 验证文件完整性
2. ✓ 创建 `local_paths.sh`
3. ✓ 预处理数据集
4. ✓ 运行推理验证

```bash
# 完整流程
source local_paths.sh
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path $VSTAR_DATA \
    --split test \
    --local_dir data/pixel_reasoner/vstar
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```

## 🎉 总结

**最简单的方式**（国内用户）:

```bash
# 1. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 下载
huggingface-cli download \
    VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --local-dir ./models/pixel_reasoner_3b \
    --resume-download

huggingface-cli download \
    JasperHaozhe/VStar-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/vstar_raw \
    --resume-download

# 3. 预处理
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path ./data/pixel_reasoner/vstar_raw \
    --split test \
    --local_dir data/pixel_reasoner/vstar

# 4. 验证
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```

预计总时间：15-30分钟（取决于网速）
