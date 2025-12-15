# Pixel Reasoner 本地部署指南

本指南适用于需要离线或使用镜像站点下载模型和数据集的用户（如国内用户）。

## 📥 下载方式对比

| 方式 | 优势 | 适用场景 |
|------|------|----------|
| **Python脚本** ⭐ | 进度显示、错误处理好 | 推荐 |
| **Bash脚本** | 简单直接 | 熟悉shell |
| **手动下载** | 完全控制 | 网络极不稳定 |

## 🚀 方式1: Python脚本下载（推荐）

### 安装依赖
```bash
pip install -U huggingface_hub
```

### 下载所有内容
```bash
python examples/train/pixel_reasoner/download_models.py
```

### 仅下载模型（约6GB）
```bash
python examples/train/pixel_reasoner/download_models.py --skip-datasets
```

### 仅下载特定数据集
```bash
# 下载InfoVQA和TallyQA
python examples/train/pixel_reasoner/download_models.py \
    --skip-model \
    --datasets infovqa tallyqa
```

### 自定义目录
```bash
python examples/train/pixel_reasoner/download_models.py \
    --model-dir /data/models \
    --data-dir /data/datasets
```

### 使用镜像站点（国内用户）
```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常下载
python examples/train/pixel_reasoner/download_models.py
```

## 📦 方式2: Bash脚本下载

### 下载所有内容
```bash
bash examples/train/pixel_reasoner/download_all.sh
```

### 仅下载模型
```bash
bash examples/train/pixel_reasoner/download_model_only.sh
```

## 🔧 方式3: 手动下载（huggingface-cli）

### 安装CLI工具
```bash
pip install -U huggingface_hub[cli]
```

### 登录（可选，用于私有模型或更快下载）
```bash
huggingface-cli login
```

### 下载模型
```bash
huggingface-cli download \
    VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --local-dir ./models/pixel_reasoner_3b \
    --resume-download
```

### 下载数据集
```bash
# InfoVQA
huggingface-cli download \
    JasperHaozhe/InfoVQA-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/infovqa_raw \
    --resume-download

# TallyQA
huggingface-cli download \
    JasperHaozhe/TallyQA-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/tallyqa_raw \
    --resume-download

# VStar
huggingface-cli download \
    JasperHaozhe/VStar-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/vstar_raw \
    --resume-download

# MVBench
huggingface-cli download \
    JasperHaozhe/MVBench-EvalData-PixelReasoner \
    --repo-type dataset \
    --local-dir ./data/pixel_reasoner/mvbench_raw \
    --resume-download
```

## 🌐 国内用户专属：使用镜像站

### HuggingFace 镜像
```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 验证
echo $HF_ENDPOINT

# 然后使用任何下载方式
python examples/train/pixel_reasoner/download_models.py
```

### ModelScope 镜像（备选）
```bash
# 安装modelscope
pip install modelscope

# 使用modelscope下载
from modelscope import snapshot_download
model_dir = snapshot_download('VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100')
```

## 📁 下载后的目录结构

```
.
├── models/
│   └── VerlTool_pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── ...
│
├── data/
│   └── pixel_reasoner/
│       ├── TIGER-Lab_PixelReasoner-RL-Data/
│       ├── JasperHaozhe_InfoVQA-EvalData-PixelReasoner/
│       ├── JasperHaozhe_TallyQA-EvalData-PixelReasoner/
│       ├── JasperHaozhe_VStar-EvalData-PixelReasoner/
│       └── JasperHaozhe_MVBench-EvalData-PixelReasoner/
│
└── local_paths.sh  # 自动生成的路径配置
```

## 🔄 数据预处理

下载原始数据后，需要处理成parquet格式：

### 加载本地路径
```bash
source local_paths.sh
```

### 处理评估数据集

```bash
# InfoVQA
python examples/data_preprocess/pixel_reasoner/infovqa.py \
    --dataset_path $INFOVQA_DATA \
    --split test \
    --local_dir data/pixel_reasoner/info_vqa

# TallyQA
python examples/data_preprocess/pixel_reasoner/tallyqa.py \
    --dataset_path $TALLYQA_DATA \
    --split test \
    --local_dir data/pixel_reasoner/tallyqa

# VStar
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path $VSTAR_DATA \
    --split test \
    --local_dir data/pixel_reasoner/vstar

# MVBench
python examples/data_preprocess/pixel_reasoner/mvbench.py \
    --dataset_path $MVBENCH_DATA \
    --split test \
    --local_dir data/pixel_reasoner/mvbench
```

### 处理训练数据（如果需要训练）
```bash
python examples/data_preprocess/pixel_reasoner/prepare_train.py \
    --dataset_path $PIXELREASONER_RL_DATA \
    --local_dir data/pixel_reasoner/train \
    --version max_16384 \
    --include_videos True
```

## 🎯 配置推理脚本使用本地模型

### 方式1: 修改脚本

编辑 `inference_2x4090.sh`，修改模型路径：

```bash
# 原来（从HuggingFace下载）
model_name=VerlTool/pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100

# 改为（使用本地路径）
model_name=./models/VerlTool_pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100
```

### 方式2: 使用环境变量

```bash
# 加载路径配置
source local_paths.sh

# 修改推理脚本使用环境变量
model_name=$PIXEL_REASONER_MODEL
```

### 方式3: 创建专用的本地推理脚本

我已经为你准备了一个使用本地路径的脚本：`inference_2x4090_local.sh`

```bash
bash examples/train/pixel_reasoner/inference_2x4090_local.sh
```

## 📊 存储空间需求

| 内容 | 大小 | 必需 |
|------|------|------|
| 模型 | ~6GB | ✓ |
| InfoVQA | ~500MB | 推荐 |
| TallyQA | ~300MB | 可选 |
| VStar | ~200MB | 可选 |
| MVBench | ~2GB | 可选 |
| 训练数据 | ~10GB | 训练时需要 |
| **总计（评估）** | **~9GB** | - |
| **总计（含训练）** | **~19GB** | - |

## ❓ 常见问题

### Q1: 下载速度很慢怎么办？

**方案1**: 使用镜像站点
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**方案2**: 使用断点续传
```bash
# Python脚本默认支持
python download_models.py  # 中断后重新运行即可继续

# CLI命令
huggingface-cli download ... --resume-download
```

**方案3**: 多线程下载（高级）
```bash
# 使用aria2c
aria2c -x 16 -s 16 <HuggingFace文件直链>
```

### Q2: 下载到一半失败了怎么办？

重新运行下载脚本，会自动从断点继续：
```bash
python download_models.py  # 自动恢复
```

### Q3: 如何验证下载完整性？

```bash
# 检查模型文件
ls -lh models/VerlTool_*/

# 应该包含：
# - config.json
# - model.safetensors 或 pytorch_model.bin
# - tokenizer相关文件

# 快速测试模型加载
python -c "
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(
    './models/VerlTool_pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100',
    trust_remote_code=True
)
print('Model loaded successfully!')
"
```

### Q4: 可以使用软链接吗？

可以，如果你已经在其他位置下载了模型：

```bash
# 创建软链接
ln -s /path/to/existing/model ./models/pixel_reasoner_3b

# 在推理脚本中使用
model_name=./models/pixel_reasoner_3b
```

### Q5: 下载时提示需要token怎么办？

```bash
# 登录HuggingFace
huggingface-cli login

# 或设置token
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

### Q6: 可以只下载模型用quick_inference.py测试吗？

可以！如果只是想快速测试：

```bash
# 1. 只下载模型
python download_models.py --skip-datasets

# 2. 准备一张测试图片
# 3. 运行quick_inference
python examples/train/pixel_reasoner/quick_inference.py \
    --model_path ./models/VerlTool_pixel-reaoner-3b-grpo-n8-b128-t1.0-lr1e-6-complex-reward_global_step_100 \
    --image_path test.jpg \
    --question "描述这张图片"
```

## 🎯 完整工作流程示例

### 场景1: 国内用户首次使用

```bash
# 1. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 下载模型和数据
python examples/train/pixel_reasoner/download_models.py

# 3. 加载路径
source local_paths.sh

# 4. 处理InfoVQA数据集
python examples/data_preprocess/pixel_reasoner/infovqa.py \
    --dataset_path $INFOVQA_DATA \
    --split test \
    --local_dir data/pixel_reasoner/info_vqa

# 5. 运行推理
bash examples/train/pixel_reasoner/inference_2x4090_local.sh
```

### 场景2: 只测试模型不评估

```bash
# 1. 只下载模型
python download_models.py --skip-datasets

# 2. 加载路径
source local_paths.sh

# 3. 测试单张图片
python examples/train/pixel_reasoner/quick_inference.py \
    --model_path $PIXEL_REASONER_MODEL \
    --image_path your_image.jpg \
    --question "What do you see?"
```

## 📚 相关文档

- [快速开始](../../../PIXEL_REASONER_QUICKSTART.md)
- [RTX 4090配置](./RTX4090_CONFIG.md)
- [推理指南](./INFERENCE_GUIDE.md)

## 💡 最佳实践

1. **分步下载**: 先下载模型测试，确认可用后再下载数据集
2. **使用镜像**: 国内用户务必设置 `HF_ENDPOINT`
3. **断点续传**: 遇到中断不要重新开始，直接重新运行脚本
4. **磁盘空间**: 确保至少有20GB可用空间
5. **验证完整**: 下载后运行quick_inference验证模型完整性
