# MyGPT 模型详细说明

## 📊 当前模型架构

### 模型配置

| 配置项 | 数值 | 说明 |
|--------|------|------|
| **模型名称** | GPT-Mini | 基于 nanoGPT 架构 |
| **词表大小** | 50,257 | 使用 GPT-2 tokenizer |
| **嵌入维度** | 384 | d_model |
| **注意力头数** | 6 | Multi-Head Attention |
| **Transformer 层数** | 6 | 深度 |
| **最大序列长度** | 256 | Block Size |
| **Dropout** | 0.2 | 防止过拟合 |
| **参数总量** | ~10M | 约 1000 万参数 |

### 架构组成

```
GPT-Mini 架构:
├── Token Embedding (50257 × 384)
├── Position Embedding (256 × 384)
├── 6 × Transformer Block
│   ├── LayerNorm
│   ├── Multi-Head Self-Attention (6 heads)
│   ├── LayerNorm
│   └── Feed Forward Network (384 → 1536 → 384)
├── Final LayerNorm
└── Language Model Head (384 → 50257)
```

---

## 📦 当前训练数据

### 数据状态
⚠️ **当前仅有示例数据，未实际训练！**

**现有数据：**
- 📄 `data/training/train_data.json`
- 数量：**6 条示例对话**
- 用途：架构测试、格式演示

**示例内容：**
```json
[
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！很高兴见到你..."},
  
  {"role": "user", "content": "什么是人工智能？"},
  {"role": "assistant", "content": "人工智能（AI）是..."},
  
  // ... 共 6 条
]
```

### 数据来源

当前数据为**手写示例**，用于：
1. 验证数据格式正确性
2. 测试训练流程
3. 演示系统功能

---

## 🎯 如何获取真实训练数据

### 方法一：使用提供的下载脚本

```bash
cd /home/user/mygpt-chat
python download_training_data.py
```

选择下载选项：
- **选项 1**: BELLE 数据（10,000 条中文对话）
- **选项 2**: 多源数据（BELLE + Alpaca，约 8,000 条）

### 方法二：手动下载开源数据集

推荐数据集：

| 数据集 | 规模 | 语言 | 下载链接 |
|--------|------|------|----------|
| BELLE | 200万+ | 中文 | https://huggingface.co/BelleGroup |
| Firefly | 160万+ | 中文 | https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M |
| Alpaca | 52K | 英文 | https://huggingface.co/datasets/tatsu-lab/alpaca |
| ShareGPT | 90K | 英文 | https://huggingface.co/datasets/RyokoAI/ShareGPT52K |

### 方法三：自定义数据

准备自己的对话数据：

```json
[
  {
    "conversations": [
      {"role": "user", "content": "你的问题"},
      {"role": "assistant", "content": "你的回答"}
    ]
  }
]
```

---

## 🏋️ 训练数据量建议

### 不同目标的数据需求

| 目标水平 | 最小数据量 | 推荐数据量 | 预期效果 |
|---------|-----------|-----------|---------|
| **基本对话** | 5,000 条 | 20,000+ | 能进行简单问答 |
| **流畅对话** | 50,000 条 | 200,000+ | 多轮对话流畅 |
| **专业领域** | 100,000 条 | 500,000+ | 领域知识准确 |
| **高质量** | 500,000 条 | 2,000,000+ | 接近 GPT-3.5 水平 |

### 当前模型容量限制

⚠️ **10M 参数的模型建议数据量：**
- 最优：**50,000 - 200,000 条**
- 上限：避免过拟合，不超过 500,000 条

---

## 🚀 训练流程

### 1. 下载训练数据

```bash
# 方式一：使用脚本
python download_training_data.py

# 方式二：手动下载后放入 data/training/ 目录
```

### 2. 配置训练参数

编辑 `config.yaml`:

```yaml
training:
  batch_size: 32
  learning_rate: 3e-4
  max_epochs: 10
```

### 3. 开始训练

```bash
cd models
python trainer.py --config ../config.yaml
```

### 4. 训练监控

```bash
tensorboard --logdir logs/ --port 6006
```

---

## 💡 当前模型状态总结

| 项目 | 状态 | 说明 |
|------|------|------|
| **模型架构** | ✅ 完整实现 | GPT-2 风格，10M 参数 |
| **训练代码** | ✅ 完整实现 | 支持 GPU/CPU |
| **训练数据** | ⚠️ 仅示例 | 需要下载真实数据 |
| **预训练权重** | ❌ 未训练 | 需要运行训练 |
| **推理服务** | ✅ 完整实现 | 支持流式输出 |
| **RAG 增强** | ✅ 完整实现 | 知识库检索 |

---

## 📝 下一步操作

### 最小可行方案（1小时内）

```bash
# 1. 下载 10000 条数据
python download_training_data.py  # 选择选项 1

# 2. 快速训练（CPU 约 2 小时，GPU 约 15 分钟）
cd models
python trainer.py --config ../config.yaml

# 3. 启动服务
cd ../backend
uvicorn app.main:app --reload

# 4. 测试对话
# 访问 http://localhost:3000
```

### 生产级方案

1. **增加模型容量** - 修改 config.yaml:
   ```yaml
   model:
     n_embed: 768    # 从 384 提升到 768
     n_head: 12      # 从 6 提升到 12
     n_layer: 12     # 从 6 提升到 12
     # 参数量将增加到约 100M
   ```

2. **准备大规模数据** - 下载 100万+ 条高质量数据

3. **多 GPU 训练** - 配置分布式训练

4. **持续迭代** - 收集用户反馈数据，持续微调

---

## 🎯 总结

**当前状态：**
- ✅ 完整的模型架构和训练代码
- ✅ 完整的推理服务和前端界面
- ⚠️ **缺少真实训练数据**（仅有 6 条示例）
- ❌ **尚未训练模型**

**模型能力（训练后预期）：**
- 基础对话能力
- 多轮上下文理解
- 代码生成（如果有代码训练数据）
- 知识问答（结合 RAG）

**建议：**
立即运行 `python download_training_data.py` 下载数据，然后开始训练！
