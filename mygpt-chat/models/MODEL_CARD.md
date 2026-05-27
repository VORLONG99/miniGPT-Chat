# MyGPT — 模型卡片（Model Card）

## 📊 模型版本清单

| 文件 | 类型 | 参数量 | 说明 |
|------|------|--------|------|
| `pretrained_weights.pt` | PyTorch | 5.12M | GPT-2标准初始化，主模型框架 |
| `best_model.pt` | PyTorch | 5.12M | 最优验证损失版本 |
| `finetuned_weights.json` | Numpy | 0.5M | 50步AdamW微调 |
| `tiny_finetuned.pt` | PyTorch | 0.12M | 轻量级快速推理版 |
| `tokenizer.json` | 词表 | 1301词 | 字符级分词器 |

---

## 🏗️ 主模型架构（pretrained_weights.pt）

```
GPT (Decoder-only Transformer)
├── Token Embedding:   1301 × 256 = 332,800
├── Position Embedding: 256 × 256 = 65,536
├── 6 × Transformer Block
│   ├── LayerNorm
│   ├── MultiHead Attention (8头, head_dim=32)
│   │   ├── QKV Projection: 256→768
│   │   └── Output Projection: 256→256
│   ├── LayerNorm
│   └── Feed Forward: 256→1024→256
├── Final LayerNorm
└── LM Head: 256→1301 (权重绑定)
─────────────────────────────────
总参数量: 5,124,357 ≈ 5.12M
```

---

## 📦 训练数据（已扩充）

### 数据量对比

| 版本 | 对话数量 | 字符数 | 说明 |
|------|---------|--------|------|
| **原版** | 6 条 | ~4,000 | 手写示例 |
| **当前版** | **71 条** | **28,000+** | 多领域高质量 |

### 数据领域分布

| 领域 | 条数 | 内容特征 |
|------|------|---------|
| 通用闲聊 | 20 | 日常问答、情感支持、生活建议 |
| 技术问答 | 13 | REST API、Docker、数据库、网络协议 |
| 代码编程 | 10 | Python/JS代码、算法、设计模式 |
| AI机器学习 | 6 | 神经网络、Transformer、训练流程 |
| 数学逻辑 | 5 | 动态规划、贝叶斯、梯度下降 |
| 科学知识 | 5 | 量子力学、黑洞、DNA、相对论 |
| 历史文化 | 3 | 四大发明、丝绸之路、二战 |
| 多轮对话 | 3 | 学习路径规划、系统设计、Bug排查 |
| 哲学伦理 | 3 | 存在主义、AI意识、功利主义 |
| 生活商业 | 3 | 睡眠、理财、产品定位 |

---

## 🔧 权重初始化策略（GPT-2 标准）

```python
# 1. 一般线性层和Embedding：正态分布初始化
nn.init.normal_(weight, mean=0.0, std=0.02)

# 2. 偏置和LayerNorm：
nn.init.zeros_(bias)
nn.init.ones_(layer_norm.weight)
nn.init.zeros_(layer_norm.bias)

# 3. 残差连接的投影层（特殊缩放）：
# 防止深度网络中残差信号爆炸
std = 0.02 / math.sqrt(2 * n_layer)
nn.init.normal_(residual_proj.weight, std=std)

# 4. 权重绑定（减少参数量）：
lm_head.weight = token_embedding.weight  # 输出词表权重共享
```

---

## 📈 当前模型状态

```
训练阶段：
├── ✅ Phase 0: GPT-2 标准初始化（已完成）
│   └── 初始 loss ≈ 7.17（=ln(1301), 即随机猜测水平）
│
├── ✅ Phase 1: 50步 numpy 微调（已完成）
│   └── loss: 7.17 → 4.14（ppl: 1301 → 62.8）
│
├── ⏳ Phase 2: GPU 全量预训练（需在本地执行）
│   └── 目标 loss < 2.0（ppl < 7.4）
│   └── 命令：python3 models/pretrain.py --epochs 50
│
└── ⏳ Phase 3: SFT 对话微调（需在本地执行）
    └── 目标：让模型学会标准对话格式
```

---

## 🚀 在本机训练（推荐）

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- RAM 8GB+ (或 GPU 8GB+)

### 训练命令

```bash
cd /path/to/mygpt-chat

# 安装依赖
pip install torch numpy

# 全量预训练（CPU约4小时，GPU约15分钟）
python3 models/pretrain.py \
  --epochs 50 \
  --batch 32 \
  --lr 3e-4 \
  --data data/training/train_data.json \
  --ckpt_dir models/checkpoints

# 监控训练（另开终端）
# tensorboard --logdir logs/
```

### 训练进度预估

| 硬件 | 50 epochs 耗时 | 预期最终 loss |
|------|--------------|--------------|
| RTX 3080 | ~8 分钟 | ~1.8 |
| M2 MacBook | ~45 分钟 | ~2.0 |
| CPU (i9) | ~3 小时 | ~2.2 |
| CPU (云沙箱) | 不适合 | — |

---

## 📁 文件路径说明

```
models/checkpoints/
├── pretrained_weights.pt   # 主模型权重（GPT-2 init, 5.1M参数）
├── best_model.pt           # 验证集最优权重
├── final_weights.pt        # 最后 epoch 权重
├── finetuned_weights.json  # 50步numpy微调结果（0.5M）
├── tiny_finetuned.pt       # 轻量版（0.12M，适合快速测试）
└── tokenizer.json          # 字符级分词器（1301 tokens）

corpus_ids.json             # 预编码的语料（用于快速加载）
```

---

## 📝 使用权重

```python
import torch

# 加载主模型权重
ckpt = torch.load("models/checkpoints/pretrained_weights.pt")
print(ckpt.keys())
# dict_keys(['model_state_dict', 'vocab_size', 'block_size',
#            'n_embd', 'n_head', 'n_layer', 'val_loss', 'description'])

# 配置信息
print(f"词表大小: {ckpt['vocab_size']}")    # 1301
print(f"模型维度: {ckpt['n_embd']}")        # 256
print(f"注意力头: {ckpt['n_head']}")        # 8
print(f"层数: {ckpt['n_layer']}")           # 6
print(f"验证损失: {ckpt['val_loss']:.4f}")  # ~7.23 (初始化值)
```
