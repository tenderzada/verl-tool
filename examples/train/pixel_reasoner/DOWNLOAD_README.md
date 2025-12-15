# 模型与数据集下载指南

快速参考：如何下载Pixel Reasoner模型和数据集

## 🚀 快速开始（推荐）

### 方式1: Python脚本（最简单）⭐

```bash
# 安装依赖
pip install -U huggingface_hub

# 下载模型和数据集
python examples/train/pixel_reasoner/download_models.py

# 加载路径配置
source local_paths.sh
```

### 方式2: Bash脚本

```bash
bash examples/train/pixel_reasoner/download_all.sh
source local_paths.sh
```

### 方式3: 仅下载模型（快速测试）

```bash
python examples/train/pixel_reasoner/download_models.py --skip-datasets
# 或
bash examples/train/pixel_reasoner/download_model_only.sh
```

## 🇨🇳 国内用户专属

### 使用HuggingFace镜像

```bash
# 设置镜像环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常下载
python examples/train/pixel_reasoner/download_models.py
```

### 或者在脚本中永久设置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

## 📦 下载内容清单

| 内容 | 大小 | 必需 | 用途 |
|------|------|------|------|
| **模型** | ~6GB | ✓ | 推理必需 |
| InfoVQA数据集 | ~500MB | 推荐 | 信息图表理解评估 |
| TallyQA数据集 | ~300MB | 可选 | 计数任务评估 |
| VStar数据集 | ~200MB | 可选 | 空间推理评估 |
| MVBench数据集 | ~2GB | 可选 | 视频理解评估 |

**总计**: ~9GB（模型+所有评估数据集）

## 🎯 下载选项对比

### 完整下载（推荐用于完整评估）
```bash
python download_models.py
```
- 下载模型 + 所有数据集
- 时间: 30-60分钟（取决于网速）
- 空间: ~9GB

### 最小下载（推荐用于快速验证）
```bash
python download_models.py --skip-datasets
```
- 仅下载模型
- 时间: 10-20分钟
- 空间: ~6GB

### 选择性下载（推荐用于特定评估）
```bash
python download_models.py --datasets infovqa tallyqa
```
- 下载模型 + 指定数据集
- 时间: 15-30分钟
- 空间: ~7GB

## 📁 下载后目录结构

```
project_root/
├── models/
│   └── VerlTool_pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100/
│       ├── config.json
│       ├── model.safetensors
│       └── ...
│
├── data/
│   └── pixel_reasoner/
│       ├── JasperHaozhe_InfoVQA-EvalData-PixelReasoner/
│       ├── JasperHaozhe_TallyQA-EvalData-PixelReasoner/
│       └── ...
│
└── local_paths.sh  # 自动生成的路径配置文件
```

## 🔄 下载后的步骤

### 1. 加载路径配置
```bash
source local_paths.sh
```

### 2. 处理数据集（将原始数据转为parquet格式）

```bash
# InfoVQA
python examples/data_preprocess/pixel_reasoner/infovqa.py \
    --dataset_path $INFOVQA_DATA \
    --split test \
    --local_dir data/pixel_reasoner/info_vqa

# 其他数据集同理...
```

### 3. 运行推理
```bash
bash examples/train/pixel_reasoner/inference_2x4090_local.sh
```

## 🛠️ 下载脚本对比

| 脚本 | 语言 | 优势 | 适用场景 |
|------|------|------|----------|
| `download_models.py` | Python | 进度显示、错误处理、灵活配置 | **推荐** |
| `download_all.sh` | Bash | 简单直接 | 快速一键下载 |
| `download_model_only.sh` | Bash | 最小化 | 只想测试模型 |
| `huggingface-cli` | CLI | 官方工具 | 高级用户 |

## ❓ 常见问题

### Q: 下载速度很慢？
A: 国内用户设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`

### Q: 下载中断了？
A: 重新运行脚本，支持断点续传

### Q: 如何验证下载完整？
A: 运行 `python quick_inference.py` 测试模型加载

### Q: 可以只下载模型吗？
A: 可以！用 `--skip-datasets` 参数

### Q: 磁盘空间不够？
A:
- 仅下载模型: 6GB
- 仅下载InfoVQA: 6.5GB
- 完整下载: 9GB

## 📚 详细文档

- **完整本地部署指南**: [LOCAL_SETUP.md](./LOCAL_SETUP.md)
- **快速开始**: [PIXEL_REASONER_QUICKSTART.md](../../../PIXEL_REASONER_QUICKSTART.md)
- **RTX 4090配置**: [RTX4090_CONFIG.md](./RTX4090_CONFIG.md)

## 💡 推荐工作流

### 新手推荐
```bash
# 1. 先下载模型测试
python download_models.py --skip-datasets

# 2. 快速验证
source local_paths.sh
python quick_inference.py \
    --model_path $PIXEL_REASONER_MODEL \
    --image_path test.jpg \
    --question "test"

# 3. 确认无误后下载数据集
python download_models.py --skip-model --datasets infovqa
```

### 国内用户推荐
```bash
# 1. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 一次性下载
python download_models.py

# 3. 后续使用
source local_paths.sh
bash inference_2x4090_local.sh
```

### 老手直接
```bash
# 一条命令搞定
HF_ENDPOINT=https://hf-mirror.com python download_models.py && \
source local_paths.sh && \
bash inference_2x4090_local.sh
```

## 🎯 脚本使用示例

### Python脚本详细选项

```bash
# 查看帮助
python download_models.py --help

# 仅下载模型
python download_models.py --skip-datasets

# 仅下载数据集
python download_models.py --skip-model

# 选择特定数据集
python download_models.py --datasets infovqa tallyqa

# 自定义目录
python download_models.py \
    --model-dir /data/models \
    --data-dir /data/datasets

# 禁用断点续传
python download_models.py --no-resume
```

### 环境变量配置

```bash
# HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# HuggingFace Token（如果需要）
export HF_TOKEN=hf_xxxxxxxxxxxxx

# 代理设置（如果需要）
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

## ⚡ 性能优化提示

1. **使用镜像**: 国内用户速度提升10倍+
2. **断点续传**: 中断后重新运行，不会重头开始
3. **并行下载**: Python脚本内部已优化
4. **本地缓存**: HuggingFace会缓存到 `~/.cache/huggingface/`

## 🔒 安全提示

- 下载的模型会验证SHA256校验和
- 使用官方HuggingFace Hub API
- 支持使用token进行认证访问
