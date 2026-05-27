"""
训练脚本 - 预训练 + SFT 微调
支持从零预训练和基于对话数据的监督微调
"""
import os
import sys
import json
import torch
import argparse
from nano_gpt import GPT, GPTConfig, GPTTrainer, ChatDataset, SimpleTokenizer


def prepare_demo_data(output_dir="data"):
    """准备演示用的对话数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 示例对话数据 (SFT 格式)
    conversations = [
        {"conversations": [{"role": "user", "content": "你好，请问你是谁？"}, {"role": "assistant", "content": "你好！我是一个AI对话助手，很高兴为你服务。"}]},
        {"conversations": [{"role": "user", "content": "什么是人工智能？"}, {"role": "assistant", "content": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统，包括学习、推理、感知和决策等能力。"}]},
        {"conversations": [{"role": "user", "content": "Python是什么？"}, {"role": "assistant", "content": "Python是一种高级编程语言，以简洁易读的语法著称，广泛应用于Web开发、数据科学、人工智能等领域。"}]},
        {"conversations": [{"role": "user", "content": "请解释什么是机器学习"}, {"role": "assistant", "content": "机器学习是人工智能的一个子领域，通过算法让计算机从数据中自动学习和改进，而无需显式编程。主要类型包括监督学习、无监督学习和强化学习。"}]},
        {"conversations": [{"role": "user", "content": "深度学习和机器学习有什么区别？"}, {"role": "assistant", "content": "深度学习是机器学习的一个子集，主要使用多层神经网络来学习数据的复杂表示。相比传统机器学习，深度学习在处理图像、语音和自然语言等非结构化数据时表现更出色，但需要更多的数据和计算资源。"}]},
        {"conversations": [{"role": "user", "content": "什么是Transformer？"}, {"role": "assistant", "content": "Transformer是一种基于自注意力机制的深度学习架构，由Google在2017年提出。它摒弃了传统的循环和卷积结构，完全依赖注意力机制处理序列数据，是GPT、BERT等大语言模型的基础架构。"}]},
        {"conversations": [{"role": "user", "content": "GPT是什么意思？"}, {"role": "assistant", "content": "GPT全称是Generative Pre-trained Transformer，即生成式预训练Transformer。它是一种基于Transformer解码器的大语言模型，通过在海量文本上进行无监督预训练，然后在特定任务上微调来获得强大的语言理解和生成能力。"}]},
        {"conversations": [{"role": "user", "content": "如何学习深度学习？"}, {"role": "assistant", "content": "学习深度学习的建议路径：1）掌握Python编程和数学基础（线性代数、概率论、微积分）；2）学习机器学习基本概念；3）从PyTorch或TensorFlow入门实践；4）系统学习神经网络原理；5）阅读经典论文并复现；6）参与实际项目。"}]},
        {"conversations": [{"role": "user", "content": "什么是RAG？"}, {"role": "assistant", "content": "RAG是Retrieval-Augmented Generation的缩写，即检索增强生成。它通过将外部知识库的检索结果与语言模型的生成能力相结合，使模型能够基于最新的、特定领域的知识来回答问题，有效减少幻觉现象。"}]},
        {"conversations": [{"role": "user", "content": "大语言模型有哪些应用？"}, {"role": "assistant", "content": "大语言模型的应用非常广泛，包括：智能对话助手、文本生成与摘要、代码生成与辅助编程、机器翻译、知识问答、内容审核、教育辅导、医疗辅助诊断等。随着技术进步，应用场景还在不断扩展。"}]},
        {"conversations": [{"role": "user", "content": "什么是注意力机制？"}, {"role": "assistant", "content": "注意力机制是深度学习中的一种技术，让模型能够聚焦于输入中最相关的部分。在Transformer中使用的自注意力机制，允许每个位置直接关注序列中的所有其他位置，从而捕获长距离依赖关系。"}]},
        {"conversations": [{"role": "user", "content": "请介绍一下PyTorch"}, {"role": "assistant", "content": "PyTorch是由Meta开发的开源深度学习框架，以其动态计算图和Python式API著称。它提供了强大的GPU加速张量计算和自动微分功能，是学术界和工业界最流行的深度学习框架之一。"}]},
    ]

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, 'w', encoding='utf-8') as f:
        for conv in conversations[:10]:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for conv in conversations[10:]:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"✅ 演示数据已生成: {train_path}, {val_path}")
    return train_path, val_path


def train(args):
    """主训练流程"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")

    # 1. 准备数据
    if not os.path.exists(args.data_dir):
        train_path, val_path = prepare_demo_data(args.data_dir)
    else:
        train_path = os.path.join(args.data_dir, "train.jsonl")
        val_path = os.path.join(args.data_dir, "val.jsonl")

    # 2. 构建 Tokenizer
    tokenizer = SimpleTokenizer()
    all_texts = []
    for path in [train_path, val_path]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                for conv in item.get("conversations", []):
                    all_texts.append(conv["content"])
    tokenizer.build_vocab(all_texts)
    tokenizer_path = os.path.join(args.data_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # 3. 构建数据集
    train_dataset = ChatDataset(train_path, tokenizer, block_size=args.block_size)
    val_dataset = ChatDataset(val_path, tokenizer, block_size=args.block_size)
    print(f"📊 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    # 4. 构建模型
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config)

    # 5. 训练
    trainer = GPTTrainer(model, train_dataset, val_dataset, config={
        "block_size": args.block_size,
        "vocab_size": tokenizer.vocab_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
    })

    losses = trainer.train(
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        save_path=args.save_path,
        log_interval=args.log_interval,
    )

    # 6. 保存训练信息
    info = {
        "model_config": {
            "block_size": args.block_size,
            "vocab_size": tokenizer.vocab_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
        },
        "tokenizer_path": tokenizer_path,
        "checkpoint_path": os.path.join(args.save_path, "best_model.pt"),
        "final_loss": losses[-1] if losses else None,
        "total_steps": args.max_steps,
    }
    info_path = os.path.join(args.save_path, "training_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 训练完成! 最终 Loss: {losses[-1]:.4f}")
    print(f"📁 模型保存于: {args.save_path}")
    print(f"📄 训练信息: {info_path}")

    return model, tokenizer, losses


def interactive_chat(model, tokenizer, device="cpu"):
    """交互式对话"""
    model.eval()
    model.to(device)
    print("\n" + "="*50)
    print("🤖 GPT 对话模式 (输入 'quit' 退出)")
    print("="*50)

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        prompt = f"<|user|>\n{user_input}</|user|>\n<|assistant|>\n"
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)

        response = tokenizer.decode(output[0].tolist())
        # 提取 assistant 回复
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if "</|assistant|>" in response:
            response = response.split("</|assistant|>")[0]
        print(f"🤖 助手: {response.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoGPT 训练脚本")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="模型保存路径")
    parser.add_argument("--block_size", type=int, default=256, help="上下文窗口长度")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=6, help="注意力头数")
    parser.add_argument("--n_embd", type=int, default=384, help="嵌入维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")
    parser.add_argument("--max_steps", type=int, default=500, help="最大训练步数")
    parser.add_argument("--eval_interval", type=int, default=50, help="评估间隔")
    parser.add_argument("--eval_steps", type=int, default=10, help="评估步数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--chat", action="store_true", help="训练后进入对话模式")

    args = parser.parse_args()
    model, tokenizer, losses = train(args)

    if args.chat:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        interactive_chat(model, tokenizer, device)
