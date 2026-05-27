"""
MyGPT — 预训练权重生成器
按照全栈专家五步框架设计（SOLID + 工程化）：
  Step1 理解需求：生成可用的初始权重，而非随机初始化
  Step2 识别风险：CPU-only 环境，需高效利用现有数据
  Step3 设计方案：GPT-2 标准初始化 + 字符级 tokenizer + 小规模预训练
  Step4 编码实现：清晰分层，可测试，单一职责
  Step5 审查优化：保存检查点，支持断点续训
"""

import json
import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 配置（单一数据类，清晰可读）
# ──────────────────────────────────────────────────────────
@dataclass
class PretrainConfig:
    # 模型结构
    vocab_size:  int   = 5000   # 字符级词表
    block_size:  int   = 256    # 序列长度
    n_embd:      int   = 256    # 嵌入维度（CPU友好）
    n_head:      int   = 8      # 注意力头数
    n_layer:     int   = 6      # Transformer 层数
    dropout:     float = 0.1

    # 训练超参
    batch_size:     int   = 16
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.1
    max_epochs:     int   = 30
    warmup_ratio:   float = 0.05   # 前 5% 步预热
    grad_clip:      float = 1.0
    eval_interval:  int   = 50     # 每50步评估一次

    # 路径
    data_path:  str = "data/training/train_data.json"
    ckpt_dir:   str = "models/checkpoints"
    ckpt_name:  str = "pretrained_weights.pt"

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.ckpt_dir, self.ckpt_name)


# ──────────────────────────────────────────────────────────
# 字符级 Tokenizer（简单、无依赖）
# ──────────────────────────────────────────────────────────
class CharTokenizer:
    """字符级 Tokenizer：将文本映射为整数序列"""
    PAD = 0
    BOS = 1   # begin of sequence
    EOS = 2   # end of sequence
    UNK = 3   # unknown
    SPECIAL_TOKENS = 4

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}
        self.is_built = False

    def build(self, texts: List[str]) -> None:
        """从文本列表构建词表"""
        # 统计字符频率
        freq: dict[str, int] = {}
        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1

        # 取最高频的字符
        sorted_chars = sorted(freq, key=freq.get, reverse=True)
        max_chars = self.vocab_size - self.SPECIAL_TOKENS

        self.char2id = {"<PAD>": self.PAD, "<BOS>": self.BOS,
                        "<EOS>": self.EOS, "<UNK>": self.UNK}
        for ch in sorted_chars[:max_chars]:
            idx = len(self.char2id)
            self.char2id[ch] = idx

        self.id2char = {v: k for k, v in self.char2id.items()}
        self.real_vocab_size = len(self.char2id)
        self.is_built = True
        log.info(f"Tokenizer built: {self.real_vocab_size} tokens")

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids = [self.char2id.get(ch, self.UNK) for ch in text]
        if add_special:
            ids = [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        chars = []
        skip_set = {self.PAD, self.BOS, self.EOS} if skip_special else set()
        for i in ids:
            if i in skip_set:
                continue
            chars.append(self.id2char.get(i, "?"))
        return "".join(chars)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2id": self.char2id,
                       "real_vocab_size": self.real_vocab_size}, f,
                      ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.char2id = data["char2id"]
        self.real_vocab_size = data["real_vocab_size"]
        self.id2char = {int(v): k for k, v in self.char2id.items()}
        self.is_built = True


# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────
class ConversationDataset(Dataset):
    """将对话数据转换为语言模型训练样本"""

    def __init__(self, data_path: str, tokenizer: CharTokenizer, block_size: int):
        self.block_size = block_size
        self.samples: List[List[int]] = []

        with open(data_path, encoding="utf-8") as f:
            records = json.load(f)

        all_texts = []
        for record in records:
            text = self._build_text(record["conversations"])
            all_texts.append(text)

        # 如果 tokenizer 未构建，先构建
        if not tokenizer.is_built:
            tokenizer.build(all_texts)

        # 编码并切割为 block_size 的片段
        for text in all_texts:
            ids = tokenizer.encode(text, add_special=True)
            # 滑动窗口切割
            for start in range(0, max(1, len(ids) - block_size), block_size // 2):
                chunk = ids[start: start + block_size + 1]
                if len(chunk) > 2:
                    # 不足时 pad
                    while len(chunk) < block_size + 1:
                        chunk.append(tokenizer.PAD)
                    self.samples.append(chunk[:block_size + 1])

        log.info(f"Dataset: {len(self.samples)} samples from {len(records)} conversations")

    def _build_text(self, conversations: List[dict]) -> str:
        parts = []
        for msg in conversations:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"用户: {content}")
            elif role == "assistant":
                parts.append(f"助手: {content}")
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        x = ids[:-1]   # 输入：前 N 个 token
        y = ids[1:]    # 目标：后 N 个 token（预测下一个）
        return x, y


# ──────────────────────────────────────────────────────────
# GPT 模型（完整实现，遵循 GPT-2 最佳实践）
# ──────────────────────────────────────────────────────────
class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力"""

    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.head_sz = cfg.n_embd // cfg.n_head

        self.qkv_proj  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj   = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # 因果掩码（注册为 buffer，不参与梯度）
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape 为多头
        def reshape(t):
            return t.view(B, T, self.n_head, self.head_sz).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # 注意力分数
        scale = 1.0 / math.sqrt(self.head_sz)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        y = attn @ v  # (B, n_head, T, head_sz)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.ln1  = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # Residual + Attention
        x = x + self.mlp(self.ln2(x))    # Residual + FFN
        return x


class MiniGPT(nn.Module):
    """
    从零实现的 GPT 模型（遵循 GPT-2 初始化策略）
    参数量约 5-10M，适合 CPU 训练
    """

    def __init__(self, cfg: PretrainConfig, real_vocab_size: int):
        super().__init__()
        self.cfg = cfg
        v = real_vocab_size

        self.tok_emb = nn.Embedding(v, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, v, bias=False)

        # 权重共享（GPT-2 标准做法）
        self.tok_emb.weight = self.lm_head.weight

        # GPT-2 标准初始化
        self.apply(self._init_weights)

        # 缩放残差连接的投影层（GPT-2 Paper 建议）
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"MiniGPT initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """GPT-2 标准权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.block_size

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 忽略 PAD token（id=0）的损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new: int = 200,
                 temperature: float = 0.8, top_k: int = 50) -> List[int]:
        self.eval()
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        generated = list(prompt_ids)

        for _ in range(max_new):
            ids_cond = ids[:, -self.cfg.block_size:]
            logits, _ = self(ids_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

        return generated


# ──────────────────────────────────────────────────────────
# 学习率调度（余弦退火 + 预热）
# ──────────────────────────────────────────────────────────
def get_lr(step: int, total_steps: int, cfg: PretrainConfig) -> float:
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    if step < warmup_steps:
        return cfg.learning_rate * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────
# 训练器（单一职责：只负责训练循环）
# ──────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, cfg: PretrainConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {self.device}")

    def train(self) -> None:
        cfg = self.cfg
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        # ── 1. 数据 ──────────────────────────────────────
        tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)
        dataset = ConversationDataset(cfg.data_path, tokenizer, cfg.block_size)

        # 保存 tokenizer
        tok_path = os.path.join(cfg.ckpt_dir, "tokenizer.json")
        tokenizer.save(tok_path)
        log.info(f"Tokenizer saved: {tok_path}")

        # 划分 train/val（90/10）
        n_val = max(1, len(dataset) // 10)
        n_train = len(dataset) - n_val
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=0)

        log.info(f"Train samples: {n_train}, Val samples: {n_val}")

        # ── 2. 模型 ──────────────────────────────────────
        model = MiniGPT(cfg, tokenizer.real_vocab_size).to(self.device)

        # ── 3. 优化器（AdamW，遵循 GPT-2 最佳实践）──────
        # 不对 bias 和 LayerNorm 参数做 weight decay
        decay_params     = [p for n, p in model.named_parameters()
                            if p.dim() >= 2 and p.requires_grad]
        no_decay_params  = [p for n, p in model.named_parameters()
                            if p.dim() < 2 and p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": decay_params,    "weight_decay": cfg.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8
        )

        total_steps = cfg.max_epochs * len(train_loader)
        log.info(f"Total steps: {total_steps}")

        # ── 4. 训练循环 ───────────────────────────────────
        best_val_loss = float("inf")
        step = 0
        train_losses = []
        val_losses   = []

        for epoch in range(cfg.max_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # 动态调整学习率
                lr = get_lr(step, total_steps, cfg)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                # 前向 + 反向
                _, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（防止爆炸）
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                step += 1

                # 定期评估
                if step % cfg.eval_interval == 0:
                    val_loss = self._evaluate(model, val_loader)
                    val_losses.append(val_loss)
                    ppl = math.exp(min(val_loss, 20))
                    log.info(f"step={step:4d} | lr={lr:.6f} | "
                             f"train_loss={loss.item():.4f} | "
                             f"val_loss={val_loss:.4f} | ppl={ppl:.1f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(model, tokenizer, optimizer,
                                              epoch, step, val_loss, cfg)

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            log.info(f"Epoch {epoch+1}/{cfg.max_epochs} | "
                     f"avg_loss={avg_loss:.4f}")

        # ── 5. 训练结束：保存最终权重 ─────────────────────
        final_path = os.path.join(cfg.ckpt_dir, "final_weights.pt")
        self._save_checkpoint(model, tokenizer, optimizer,
                              cfg.max_epochs, step, best_val_loss,
                              cfg, path=final_path)

        log.info("=" * 50)
        log.info("✅ 预训练完成！")
        log.info(f"   最佳验证损失: {best_val_loss:.4f}")
        log.info(f"   最佳困惑度:   {math.exp(min(best_val_loss, 20)):.1f}")
        log.info(f"   检查点保存:   {cfg.ckpt_dir}/")
        log.info("=" * 50)

        # 生成样例
        log.info("\n生成示例...")
        prompt = "用户: 你好\n助手: "
        prompt_ids = tokenizer.encode(prompt, add_special=True)
        generated = model.generate(prompt_ids, max_new=80, temperature=0.8)
        print("\n" + "=" * 50)
        print("📝 生成示例:")
        print(tokenizer.decode(generated))
        print("=" * 50)

    @torch.no_grad()
    def _evaluate(self, model: MiniGPT, loader: DataLoader) -> float:
        model.eval()
        total_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n += 1
        model.train()
        return total_loss / max(n, 1)

    def _save_checkpoint(self, model: MiniGPT, tokenizer: CharTokenizer,
                          optimizer, epoch: int, step: int,
                          val_loss: float, cfg: PretrainConfig,
                          path: str = None) -> None:
        path = path or cfg.ckpt_path
        torch.save({
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":       epoch,
            "step":        step,
            "val_loss":    val_loss,
            "config":      cfg.__dict__,
            "real_vocab_size": tokenizer.real_vocab_size,
        }, path)
        log.info(f"✅ Checkpoint saved: {path}")


# ──────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MyGPT Pretraining")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--n_embd",    type=int,   default=256)
    parser.add_argument("--n_head",    type=int,   default=8)
    parser.add_argument("--n_layer",   type=int,   default=6)
    parser.add_argument("--data",      type=str,
                        default="data/training/train_data.json")
    parser.add_argument("--ckpt_dir",  type=str,   default="models/checkpoints")
    args = parser.parse_args()

    cfg = PretrainConfig(
        max_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        data_path=args.data,
        ckpt_dir=args.ckpt_dir,
    )

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
