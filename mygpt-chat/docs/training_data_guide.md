# 如何获取 GPT 训练数据 - 完整指南

## 一、开源对话数据集推荐

### 1. 中文对话数据集

| 数据集名称 | 规模 | 链接 | 说明 |
|-----------|------|------|------|
| **BELLE** | 200万+ | https://huggingface.co/BelleGroup | 中文指令微调数据 |
| **Firefly** | 160万+ | https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M | 多任务中文数据 |
| **MOSS** | 100万+ | https://huggingface.co/datasets/fnlp/moss-003-sft-data | 复旦开源对话数据 |
| **LCCC** | 1200万 | https://github.com/thu-coai/LCCC | 大规模中文对话 |
| **STC** | 230万 | https://github.com/MarkWuNLP/MultiTurnResponseSelection | 短文本对话 |

### 2. 英文对话数据集

| 数据集名称 | 规模 | 链接 | 说明 |
|-----------|------|------|------|
| **ShareGPT** | 90万+ | https://huggingface.co/datasets/RyokoAI/ShareGPT52K | 真实用户对话 |
| **Alpaca** | 52K | https://huggingface.co/datasets/tatsu-lab/alpaca | Stanford 指令数据 |
| **Dolly** | 15K | https://huggingface.co/datasets/databricks/databricks-dolly-15k | 高质量指令数据 |
| **OpenAssistant** | 60万+ | https://huggingface.co/datasets/OpenAssistant/oasst1 | 开源助手对话 |

### 3. 代码数据集

| 数据集名称 | 规模 | 说明 |
|-----------|------|------|
| **CodeAlpaca** | 20K | 代码指令数据 |
| **MCoT** | 10万+ | 数学推理+代码 |

---

## 二、数据下载脚本

### 快速下载 BELLE 数据（推荐）

```python
# download_data.py
from datasets import load_dataset
import json

# 下载 BELLE 中文对话数据
print("正在下载 BELLE 数据集...")
dataset = load_dataset("BelleGroup/multiturn_chat_0.8M", split="train")

# 保存为训练格式
output_file = "data/training/belle_train.json"
formatted_data = []

print(f"总数据量: {len(dataset)}")

for item in dataset:
    formatted_data.append({
        "conversations": item["conversations"]
    })

# 保存前 10000 条作为示例
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data[:10000], f, ensure_ascii=False, indent=2)

print(f"✅ 已保存 {min(10000, len(formatted_data))} 条数据到 {output_file}")
```

### 下载并合并多个数据集

```python
# download_multi_datasets.py
from datasets import load_dataset
import json

all_data = []

# 1. BELLE 中文数据
print("下载 BELLE...")
belle = load_dataset("BelleGroup/multiturn_chat_0.8M", split="train[:5000]")
for item in belle:
    all_data.append({"conversations": item["conversations"]})

# 2. Firefly 数据
print("下载 Firefly...")
firefly = load_dataset("YeungNLP/firefly-train-1.1M", split="train[:5000]")
for item in firefly:
    # 转换格式
    conv = [
        {"role": "user", "content": item["input"]},
        {"role": "assistant", "content": item["target"]}
    ]
    all_data.append({"conversations": conv})

# 3. Alpaca 英文数据
print("下载 Alpaca...")
alpaca = load_dataset("tatsu-lab/alpaca", split="train[:3000]")
for item in alpaca:
    conv = [
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": item["output"]}
    ]
    all_data.append({"conversations": conv})

# 保存合并数据
with open("data/training/combined_train.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"✅ 总计 {len(all_data)} 条训练数据")
```

---

## 三、自定义数据准备

### 1. 数据格式要求

```json
[
  {
    "conversations": [
      {"role": "user", "content": "用户问题"},
      {"role": "assistant", "content": "助手回答"}
    ]
  },
  {
    "conversations": [
      {"role": "user", "content": "第一个问题"},
      {"role": "assistant", "content": "第一个回答"},
      {"role": "user", "content": "追问"},
      {"role": "assistant", "content": "追问回答"}
    ]
  }
]
```

### 2. 数据清洗脚本

```python
# clean_data.py
import json
import re

def clean_text(text):
    """清理文本"""
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()

def validate_conversation(conv):
    """验证对话格式"""
    if not isinstance(conv, list):
        return False
    
    for msg in conv:
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ["user", "assistant", "system"]:
            return False
        if len(msg["content"]) < 5 or len(msg["content"]) > 2000:
            return False
    
    return True

# 加载并清洗数据
with open("raw_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = []
for item in raw_data:
    if "conversations" in item:
        conv = item["conversations"]
        # 清理文本
        for msg in conv:
            msg["content"] = clean_text(msg["content"])
        
        # 验证格式
        if validate_conversation(conv):
            cleaned_data.append(item)

print(f"原始数据: {len(raw_data)}, 清洗后: {len(cleaned_data)}")

# 保存清洗后的数据
with open("data/training/cleaned_train.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
```

---

## 四、训练数据量建议

| 目标 | 最小数据量 | 推荐数据量 | 训练时间估计 |
|------|-----------|-----------|-------------|
| 能进行简单对话 | 5,000 条 | 20,000+ | 1-2 小时 (RTX 3090) |
| 流畅多轮对话 | 50,000 条 | 200,000+ | 4-8 小时 |
| 专业领域问答 | 100,000 条 | 500,000+ | 1-2 天 |
| 接近 ChatGPT 水平 | 1,000,000+ | 10,000,000+ | 数周 (多 GPU) |

---

## 五、快速开始训练

### 1. 下载示例数据（1000 条）

```bash
cd /home/user/mygpt-chat
python -c "
from datasets import load_dataset
import json

# 下载少量数据快速测试
dataset = load_dataset('BelleGroup/multiturn_chat_0.8M', split='train[:1000]')

data = [{'conversations': item['conversations']} for item in dataset]

with open('data/training/train_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f'✅ 已下载 {len(data)} 条数据')
"
```

### 2. 开始训练

```bash
cd models
python trainer.py --config ../config.yaml
```

### 3. 训练监控

```bash
# 查看 TensorBoard
tensorboard --logdir logs/ --port 6006
```

---

## 六、数据质量 vs 数量

**质量更重要！**

- ✅ 10,000 条高质量数据 > 100,000 条低质量数据
- ✅ 多样性：涵盖不同话题和风格
- ✅ 准确性：回答正确、无误导信息
- ✅ 格式统一：统一的对话格式

---

## 七、推荐数据组合

**最佳实践：**

```
总数据量：50,000 - 100,000 条

组成：
├── 通用对话（60%）
│   ├── BELLE: 30,000 条
│   └── Firefly: 20,000 条
│
├── 专业知识（20%）
│   ├── 代码数据: 10,000 条
│   └── 数学推理: 5,000 条
│
└── 特定领域（20%）
    ├── 行业知识: 10,000 条
    └── 自定义数据: 5,000 条
```
