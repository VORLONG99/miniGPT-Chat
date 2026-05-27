"""
NanoGPT - 从零构建的 GPT 模型
基于 Transformer Decoder 架构，支持训练与推理
"""
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 模型核心组件
# ============================================================

class CausalSelfAttention(nn.Module):
    """多头因果自注意力机制"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Q, K, V 投影 + 输出投影，合并为一次矩阵运算提高效率
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 因果掩码：防止看到未来信息
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """前馈神经网络 (FFN)"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer 解码器块"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Pre-Norm + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================
# 2. GPT 模型主类
# ============================================================

class GPTConfig:
    """GPT 模型配置"""
    # 小型配置，适合 CPU 训练；生产环境可调大
    block_size: int = 256      # 上下文窗口长度
    vocab_size: int = 32000    # 词表大小 (可对接 tokenizer)
    n_layer: int = 6           # Transformer 层数
    n_head: int = 6            # 注意力头数
    n_embd: int = 384          # 嵌入维度
    dropout: float = 0.1
    bias: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT(nn.Module):
    """GPT 模型 - 基于 Transformer Decoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),       # Token Embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),       # Position Embedding
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重共享: Embedding 层与输出层共享权重
        self.transformer.wte.weight = self.lm_head.weight
        # 初始化权重
        self.apply(self._init_weights)
        # 打印参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT 参数量: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列长度 {T} 超过最大上下文 {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成文本"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================
# 3. 数据集
# ============================================================

class ChatDataset(Dataset):
    """对话数据集 - 支持 SFT 格式"""
    def __init__(self, data_path, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        self._load_data(data_path)

    def _load_data(self, data_path):
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 格式: {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                text = self._format_conversation(item)
                tokens = self.tokenizer.encode(text)
                if len(tokens) > 2:
                    self.examples.append(tokens[:self.block_size + 1])

    def _format_conversation(self, item):
        parts = []
        for conv in item.get("conversations", []):
            role = conv["role"]
            content = conv["content"]
            parts.append(f"<|{role}|>\n{content}</|{role}|>\n")
        return "".join(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ============================================================
# 4. 训练器
# ============================================================

class GPTTrainer:
    """GPT 训练器 - 支持预训练和微调"""
    def __init__(self, model, train_dataset, val_dataset=None, config=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.optimizer = self._configure_optimizer()
        self.iter_num = 0
        self.best_val_loss = float('inf')

    def _configure_optimizer(self, weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95)):
        # 对所有 2D 以上参数做 weight decay，bias 和 LayerNorm 不做
        decay_params = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def train(self, max_steps=1000, eval_interval=100, eval_steps=20, batch_size=8,
              save_path="checkpoints", log_interval=10):
        """执行训练"""
        import os, time
        os.makedirs(save_path, exist_ok=True)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_iter = iter(train_loader)
        losses = []

        t0 = time.time()
        for step in range(max_steps):
            # 获取批次
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(self.device), y.to(self.device)

            # 前向 + 反向
            logits, loss = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

            if step % log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"Step {step:6d} | Loss: {loss.item():.4f} | Time: {dt*1000:.0f}ms")

            if step % eval_interval == 0 and self.val_dataset is not None:
                val_loss = self.evaluate(eval_steps, batch_size)
                print(f"  → Val Loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save(os.path.join(save_path, "best_model.pt"))
                    print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

            self.iter_num += 1

        # 保存最终模型
        self.save(os.path.join(save_path, "final_model.pt"))
        return losses

    @torch.no_grad()
    def evaluate(self, eval_steps, batch_size):
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0
        count = 0
        for i, (x, y) in enumerate(loader):
            if i >= eval_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            count += 1
        self.model.train()
        return total_loss / max(count, 1)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Model loaded from {path}, step={self.iter_num}")


# ============================================================
# 5. 简易 Tokenizer (可替换为 HuggingFace Tokenizer)
# ============================================================

class SimpleTokenizer:
    """基于字级别的简易中文 Tokenizer"""
    def __init__(self):
        self.char2id = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = 4

    def build_vocab(self, texts):
        for text in texts:
            for ch in text:
                if ch not in self.char2id:
                    self.char2id[ch] = self.vocab_size
                    self.id2char[self.vocab_size] = ch
                    self.vocab_size += 1
        print(f"Vocab size: {self.vocab_size}")

    def encode(self, text):
        return [self.char2id.get(ch, 1) for ch in text]

    def decode(self, ids):
        return "".join(self.id2char.get(i, "<unk>") for i in ids)

    def save(self, path):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"char2id": self.char2id, "id2char": {str(k): v for k, v in self.id2char.items()}}, f, ensure_ascii=False)

    def load(self, path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2id = data["char2id"]
            self.id2char = {int(k): v for k, v in data["id2char"].items()}
            self.vocab_size = len(self.char2id)


if __name__ == "__main__":
    # 快速测试
    config = GPTConfig(block_size=128, vocab_size=1000, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)
    x = torch.randint(0, 1000, (2, 32))
    logits, loss = model(x, targets=x)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    print("✅ NanoGPT 模型构建成功！")
