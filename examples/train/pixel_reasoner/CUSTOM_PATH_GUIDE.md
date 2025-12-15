# 使用自定义路径运行 VStar 验证

本指南针对将模型和数据下载到自定义路径的用户。

## 📁 你的路径配置

- **模型路径**: `/mnt/data/pixel_reasoner_3b`
- **原始数据**: `/mnt/data/pixel_reasoner/vstar_raw`
- **处理后数据**: `/mnt/data/pixel_reasoner/vstar` (需要生成)

## ✅ 验证文件完整性

### 1. 检查模型文件

```bash
ls -lh /mnt/data/pixel_reasoner_3b/

# 应该包含:
# - config.json
# - model.safetensors (~6GB)
# - tokenizer相关文件
```

### 2. 检查原始数据

```bash
ls -lh /mnt/data/pixel_reasoner/vstar_raw/

# 应该有数据文件（parquet或json格式）
```

## 🔄 步骤1: 预处理VStar数据集

原始下载的数据需要预处理成推理所需的格式：

```bash
# 预处理VStar数据
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path /mnt/data/pixel_reasoner/vstar_raw \
    --split test \
    --local_dir /mnt/data/pixel_reasoner/vstar
```

**预期输出**:
```
Processing VStar dataset...
Total samples: XXX
Saved to: /mnt/data/pixel_reasoner/vstar/test.parquet
```

### 验证预处理结果

```bash
ls -lh /mnt/data/pixel_reasoner/vstar/test.parquet

# 应该看到 test.parquet 文件（几MB到几十MB）
```

## 🚀 步骤2: 运行推理验证

脚本已自动适配你的路径，直接运行即可：

```bash
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```

脚本会自动：
1. ✓ 检测到 `/mnt/data/pixel_reasoner_3b` 模型
2. ✓ 检测到 `/mnt/data/pixel_reasoner/vstar/test.parquet` 数据
3. ✓ 启动工具服务器
4. ✓ 运行推理评估

## 📊 预期输出

```bash
==========================================
VStar Quick Verification - 2x RTX 4090
==========================================

Using custom model path: /mnt/data/pixel_reasoner_3b
Using custom dataset path: /mnt/data/pixel_reasoner/vstar/test.parquet
Dataset found: /mnt/data/pixel_reasoner/vstar/test.parquet

Configuration Summary:
  Model: /mnt/data/pixel_reasoner_3b
  Dataset: VStar
  GPUs: 2x RTX 4090
  Batch size: 32
  GPU memory: 0.65
  Tool server: http://xxx:xxx/get_observation (pid=xxx)

Starting VStar evaluation...
[推理过程...]

==========================================
VStar Verification Complete!
==========================================
```

## 🔧 方式2: 使用环境变量（可选）

如果你想更灵活地切换路径：

```bash
# 设置环境变量
export PIXEL_REASONER_MODEL="/mnt/data/pixel_reasoner_3b"
export VSTAR_DATA="/mnt/data/pixel_reasoner/vstar_raw"

# 预处理
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path $VSTAR_DATA \
    --split test \
    --local_dir /mnt/data/pixel_reasoner/vstar

# 运行推理
bash examples/train/pixel_reasoner/vstar_quick_verify.sh
```

## 🐛 常见问题

### Q1: 找不到模型

**症状**:
```
Local model not found, using HuggingFace...
```

**解决**:
```bash
# 检查路径是否正确
ls -ld /mnt/data/pixel_reasoner_3b

# 检查目录权限
chmod -R 755 /mnt/data/pixel_reasoner_3b

# 或者使用环境变量
export PIXEL_REASONER_MODEL="/mnt/data/pixel_reasoner_3b"
```

### Q2: 数据集未找到

**症状**:
```
Error: VStar dataset not found!
```

**解决**: 确认已运行预处理步骤
```bash
# 检查原始数据
ls /mnt/data/pixel_reasoner/vstar_raw/

# 重新预处理
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path /mnt/data/pixel_reasoner/vstar_raw \
    --split test \
    --local_dir /mnt/data/pixel_reasoner/vstar

# 验证生成的文件
ls -lh /mnt/data/pixel_reasoner/vstar/test.parquet
```

### Q3: 预处理失败

**症状**: 预处理脚本报错

**解决**:
```bash
# 检查原始数据完整性
python -c "
import os
import glob
raw_dir = '/mnt/data/pixel_reasoner/vstar_raw'
files = glob.glob(os.path.join(raw_dir, '*'))
print(f'Files in {raw_dir}:')
for f in files:
    print(f'  {os.path.basename(f)}')
"

# 检查是否有 parquet 或 json 文件
ls /mnt/data/pixel_reasoner/vstar_raw/*.parquet
# 或
ls /mnt/data/pixel_reasoner/vstar_raw/*.json
```

### Q4: 显存不足

**解决**: 调整脚本参数
```bash
# 编辑 vstar_quick_verify.sh
# 找到这些行并修改:
batch_size=16              # 从32降到16
gpu_memory_utilization=0.55  # 从0.65降到0.55
max_prompt_length=8192     # 从16384降到8192
```

## 💡 快速测试（可选）

在运行完整评估前，可以先测试单个样本：

```bash
# 创建小型测试集
python -c "
import pyarrow.parquet as pq
table = pq.read_table('/mnt/data/pixel_reasoner/vstar/test.parquet')
small_table = table.slice(0, 5)  # 只取5个样本
pq.write_table(small_table, '/mnt/data/pixel_reasoner/vstar/test_small.parquet')
print('Created test_small.parquet with 5 samples')
"

# 修改脚本使用小数据集
# 或者直接编辑 vstar_quick_verify.sh 中的 val_data 路径
```

## 📂 目录结构总览

```
/mnt/data/
├── pixel_reasoner_3b/              # 模型目录
│   ├── config.json
│   ├── model.safetensors (~6GB)
│   └── ...
│
└── pixel_reasoner/
    ├── vstar_raw/                  # 原始下载的数据
    │   └── [原始数据文件]
    │
    └── vstar/                      # 预处理后的数据（需要生成）
        └── test.parquet            # 推理使用这个文件
```

## ⏱️ 时间估算

| 步骤 | 时间 |
|------|------|
| 预处理数据 | 2-5分钟 |
| 推理评估 | 30-45分钟 |
| **总计** | **~35-50分钟** |

## 🎯 完整命令总结

```bash
# 步骤1: 预处理数据
python examples/data_preprocess/pixel_reasoner/vstar.py \
    --dataset_path /mnt/data/pixel_reasoner/vstar_raw \
    --split test \
    --local_dir /mnt/data/pixel_reasoner/vstar

# 步骤2: 验证生成的文件
ls -lh /mnt/data/pixel_reasoner/vstar/test.parquet

# 步骤3: 运行推理
bash examples/train/pixel_reasoner/vstar_quick_verify.sh

# 步骤4: 监控GPU（另一个终端）
watch -n 1 nvidia-smi
```

## 🔍 验证清单

运行前确认：

- [ ] 模型存在: `/mnt/data/pixel_reasoner_3b/`
- [ ] 原始数据存在: `/mnt/data/pixel_reasoner/vstar_raw/`
- [ ] 已运行预处理脚本
- [ ] 生成了: `/mnt/data/pixel_reasoner/vstar/test.parquet`
- [ ] 有2张GPU可用（`nvidia-smi`）
- [ ] 已安装所有依赖

## 🆘 还有问题？

如果脚本仍然找不到你的文件，可以强制指定路径：

```bash
# 方法1: 编辑脚本开头
# 在 vstar_quick_verify.sh 的第18行后添加:
model_name="/mnt/data/pixel_reasoner_3b"
val_data=[/mnt/data/pixel_reasoner/vstar/test.parquet]

# 方法2: 创建软链接到默认位置
mkdir -p ./models ./data/pixel_reasoner
ln -s /mnt/data/pixel_reasoner_3b ./models/pixel_reasoner_3b
ln -s /mnt/data/pixel_reasoner/vstar ./data/pixel_reasoner/vstar
```

## 📝 日志保存

建议保存推理日志以便后续分析：

```bash
bash examples/train/pixel_reasoner/vstar_quick_verify.sh 2>&1 | \
    tee vstar_inference_$(date +%Y%m%d_%H%M%S).log
```

---

**开始验证**: 确认上述步骤后，运行预处理和推理即可！
