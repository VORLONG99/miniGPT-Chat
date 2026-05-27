"""
GPT 模型架构 - 从零实现
参考 nanoGPT 架构设计
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm(nn.Module):
    """带偏置的 LayerNorm"""
    
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力"""
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        
        # Key, Query, Value 投影
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # 输出投影
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # 正则化
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # 因果掩码：确保只能看到之前的 token
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # 计算 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 注意力计算
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, n_embd: int, dropout: float, bias: bool = True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool = True):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT 模型 - 从零实现"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 256,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        dropout: float = 0.2,
        bias: bool = True
    ):
        super().__init__()
        assert vocab_size > 0
        assert block_size > 0
        
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([
                Block(n_embd, n_head, block_size, dropout, bias)
                for _ in range(n_layer)
            ]),
            ln_f=LayerNorm(n_embd, bias=bias),
        ))
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 打印参数数量
        print(f"模型参数量: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def _init_weights(self, module: nn.Module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            idx: (B, T) 输入 token ids
            targets: (B, T) 目标 token ids
        Returns:
            logits: (B, T, vocab_size)
            loss: 交叉熵损失（如果提供了 targets）
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"序列长度 {T} 超过最大长度 {self.block_size}"
        
        # 位置编码
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Token 嵌入 + 位置嵌入
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # 最终层归一化
        x = self.transformer.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        生成文本
        Args:
            idx: (B, T) 输入序列
            max_new_tokens: 生成的最大 token 数
            temperature: 温度参数
            top_k: Top-k 采样
        Returns:
            生成的序列 (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截断
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            
            # 只取最后一个时间步
            logits = logits[:, -1, :] / temperature
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax 得到概率
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def create_model(
    vocab_size: int = 50257,
    block_size: int = 256,
    n_embd: int = 384,
    n_head: int = 6,
    n_layer: int = 6,
    dropout: float = 0.2
) -> GPT:
    """创建 GPT 模型"""
    return GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )


if __name__ == "__main__":
    # 测试模型
    model = create_model()
    
    # 测试输入
    idx = torch.randint(0, 50257, (2, 32))  # batch_size=2, seq_len=32
    logits, loss = model(idx, targets=idx)
    
    print(f"输入形状: {idx.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"损失: {loss.item() if loss is not None else 'None'}")
    
    # 测试生成
    generated = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=40)
    print(f"生成形状: {generated.shape}")
