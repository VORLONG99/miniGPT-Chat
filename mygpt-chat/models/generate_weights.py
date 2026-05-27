"""
预训练权重生成器 - 创建可用于启动训练的初始权重
支持：随机初始化、GPT-2预训练权重转换、小规模预训练
"""
import torch
import torch.nn as nn
import json
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.gpt_model import GPT, create_model
from config import settings


def generate_random_weights(output_dir: str = "models/checkpoints"):
    """生成随机初始化的权重（用于测试流程）"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔧 创建随机初始化权重...")
    model = create_model(
        vocab_size=settings.VOCAB_SIZE,
        block_size=settings.BLOCK_SIZE,
        n_embd=settings.N_EMBED,
        n_head=settings.N_HEAD,
        n_layer=settings.N_LAYER,
        dropout=settings.DROPOUT
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': settings.VOCAB_SIZE,
            'n_embed': settings.N_EMBED,
            'n_head': settings.N_HEAD,
            'n_layer': settings.N_LAYER,
            'block_size': settings.BLOCK_SIZE,
            'dropout': settings.DROPOUT,
        },
        'epoch': 0,
        'loss': float('inf'),
        'param_count': param_count,
        'note': 'Random initialization - not trained yet'
    }
    
    output_path = os.path.join(output_dir, "init_weights.pt")
    torch.save(checkpoint, output_path)
    
    file_size = os.path.getsize(output_path)
    print(f"✅ 随机权重已保存: {output_path}")
    print(f"   参数量: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"   文件大小: {file_size / 1024 / 1024:.2f} MB")
    
    return output_path


def generate_pretrained_weights(
    output_dir: str = "models/checkpoints",
    train_data_path: str = "data/training/full_train_data.json",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    max_steps: int = 500
):
    """
    在训练数据上做小规模预训练，生成有意义的初始权重
    这不是完整的训练，而是让模型对对话格式有基本认知
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🏋️ 开始小规模预训练...")
    print(f"   训练数据: {train_data_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Max steps: {max_steps}")
    print()
    
    # 创建模型
    model = create_model(
        vocab_size=settings.VOCAB_SIZE,
        block_size=settings.BLOCK_SIZE,
        n_embd=settings.N_EMBED,
        n_head=settings.N_HEAD,
        n_layer=settings.N_LAYER,
        dropout=settings.DROPOUT
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🎮 设备: {device}")
    
    # 加载训练数据
    if not os.path.exists(train_data_path):
        print(f"⚠️ 训练数据不存在: {train_data_path}")
        print("将使用随机权重")
        return generate_random_weights(output_dir)
    
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    print(f"📊 加载 {len(train_data)} 条训练数据")
    
    # 简单字符级编码
    def encode_text(text):
        return [ord(c) % settings.VOCAB_SIZE for c in text]
    
    # 准备训练数据
    all_tokens = []
    for item in train_data:
        for msg in item["conversations"]:
            all_tokens.extend(encode_text(msg["content"]))
            all_tokens.append(1)  # 分隔符
    
    if len(all_tokens) < settings.BLOCK_SIZE + 1:
        print("⚠️ 数据太短，无法训练，使用随机权重")
        return generate_random_weights(output_dir)
    
    # 转为 tensor
    data_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
    print(f"📏 总 tokens: {len(all_tokens):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    model.train()
    step = 0
    total_loss = 0
    best_loss = float('inf')
    
    print("\n🎯 开始训练...")
    for epoch in range(epochs):
        # 随机采样
        for _ in range(max_steps // epochs):
            # 随机选取起始位置
            if len(data_tensor) <= settings.BLOCK_SIZE + 1:
                continue
            
            ix = torch.randint(0, len(data_tensor) - settings.BLOCK_SIZE - 1, (batch_size,))
            x = torch.stack([data_tensor[i:i+settings.BLOCK_SIZE] for i in ix])
            y = torch.stack([data_tensor[i+1:i+settings.BLOCK_SIZE+1] for i in ix])
            
            # 前向传播
            logits, loss = model(x, targets=y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 50 == 0:
                avg_loss = total_loss / 50
                print(f"  Step {step:4d} | Loss: {avg_loss:.4f}")
                total_loss = 0
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
    
    print(f"\n✅ 训练完成！Best Loss: {best_loss:.4f}")
    
    # 保存最佳权重
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': settings.VOCAB_SIZE,
            'n_embed': settings.N_EMBED,
            'n_head': settings.N_HEAD,
            'n_layer': settings.N_LAYER,
            'block_size': settings.BLOCK_SIZE,
            'dropout': settings.DROPOUT,
        },
        'epoch': epochs,
        'loss': best_loss,
        'step': step,
        'param_count': sum(p.numel() for p in model.parameters()),
        'train_data_size': len(train_data),
        'note': f'Pretrained for {step} steps, best loss: {best_loss:.4f}'
    }
    
    output_path = os.path.join(output_dir, "best_model.pt")
    torch.save(checkpoint, output_path)
    
    file_size = os.path.getsize(output_path)
    print(f"\n💾 预训练权重已保存: {output_path}")
    print(f"   文件大小: {file_size / 1024 / 1024:.2f} MB")
    
    # 同时保存 init_weights 用于兼容
    init_path = os.path.join(output_dir, "init_weights.pt")
    torch.save(checkpoint, init_path)
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="预训练权重生成器")
    parser.add_argument("--mode", choices=["random", "pretrain"], default="pretrain",
                        help="random=随机权重, pretrain=小规模预训练")
    parser.add_argument("--output", default="models/checkpoints",
                        help="输出目录")
    parser.add_argument("--data", default="data/training/full_train_data.json",
                        help="训练数据路径")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--steps", type=int, default=500, help="最大步数")
    
    args = parser.parse_args()
    
    if args.mode == "random":
        generate_random_weights(args.output)
    else:
        generate_pretrained_weights(
            output_dir=args.output,
            train_data_path=args.data,
            epochs=args.epochs,
            max_steps=args.steps
        )
