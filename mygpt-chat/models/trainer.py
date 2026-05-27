"""
GPT 模型训练器
支持预训练和 SFT 微调
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Tokenizer
from typing import Optional, Dict, List
import json
import os
from tqdm import tqdm
import math

from gpt_model import GPT, create_model


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, data_path: str, tokenizer: GPT2Tokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 加载数据
        print(f"加载数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        print("Tokenizing...")
        self.tokens = tokenizer.encode(text, truncation=False)
        print(f"总 token 数: {len(self.tokens):,}")
        
        # 截断到 block_size 的整数倍
        self.tokens = self.tokens[:len(self.tokens) // block_size * block_size]
    
    def __len__(self) -> int:
        return len(self.tokens) // self.block_size
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.block_size
        end = start + self.block_size
        return torch.tensor(self.tokens[start:end], dtype=torch.long)


class ConversationDataset(Dataset):
    """对话数据集（用于 SFT 微调）"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: GPT2Tokenizer, 
        block_size: int = 256,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 加载对话数据
        print(f"加载对话数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        if max_samples:
            conversations = conversations[:max_samples]
        
        self.samples = []
        
        # 格式化对话
        for conv in conversations:
            # 格式: User: {input}\nAssistant: {output}
            text = f"User: {conv['input']}\nAssistant: {conv['output']}<|endoftext|>"
            tokens = tokenizer.encode(text, truncation=True, max_length=block_size)
            self.samples.append(tokens)
        
        print(f"加载了 {len(self.samples):,} 个对话样本")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        tokens = self.samples[idx]
        # Padding
        if len(tokens) < self.block_size:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.block_size - len(tokens))
        else:
            tokens = tokens[:self.block_size]
        return torch.tensor(tokens, dtype=torch.long)


class Trainer:
    """GPT 训练器"""
    
    def __init__(
        self,
        model: GPT,
        tokenizer: GPT2Tokenizer,
        device: str = "auto",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_epochs: int = 10,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 训练统计
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _get_device(self, device: str) -> torch.device:
        """获取设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int
    ) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch = batch.to(self.device)
            
            # 前向传播
            logits, loss = self.model(batch, targets=batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss.item()):.2f}'
            })
            
            # 日志
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"\nStep {self.global_step}: loss={avg_loss:.4f}, ppl={math.exp(avg_loss):.2f}")
            
            # 保存检查点
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
        
        return total_loss / num_batches
    
    def train(
        self, 
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        num_workers: int = 0
    ):
        """完整训练流程"""
        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        print(f"\n开始训练")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_dataset):,}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {self.max_epochs}")
        print("-" * 50)
        
        for epoch in range(1, self.max_epochs + 1):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"\nEpoch {epoch} 完成: 平均 loss={train_loss:.4f}")
            
            # 验证
            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"验证 loss={val_loss:.4f}, ppl={math.exp(val_loss):.2f}")
                
                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            
            # 定期保存
            self.save_checkpoint(f"epoch_{epoch}.pt")
        
        print("\n训练完成!")
        self.save_checkpoint("final_model.pt")
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(data_loader)
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(self.device)
            _, loss = self.model(batch, targets=batch)
            total_loss += loss.item()
        
        self.model.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        print(f"检查点已加载: {path}")


def prepare_tokenizer():
    """准备 tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


if __name__ == "__main__":
    # 创建 tokenizer
    tokenizer = prepare_tokenizer()
    
    # 创建模型
    model = create_model(
        vocab_size=50257,
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=3e-4,
        max_epochs=5,
        checkpoint_dir="./checkpoints"
    )
    
    print("训练器创建成功!")
    print(f"设备: {trainer.device}")
