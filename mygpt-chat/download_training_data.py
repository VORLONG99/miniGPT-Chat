"""
快速下载训练数据脚本
运行此脚本下载开源对话数据
"""
import os
import json

def download_sample_data():
    """下载示例训练数据"""
    
    # 尝试导入 datasets 库
    try:
        from datasets import load_dataset
        print("✅ 检测到 datasets 库")
    except ImportError:
        print("❌ 请先安装 datasets 库：")
        print("   pip install datasets")
        return None
    
    print("\n📦 开始下载 BELLE 中文对话数据...")
    
    try:
        # 下载 BELLE 数据集（前 10000 条）
        dataset = load_dataset(
            "BelleGroup/multiturn_chat_0.8M", 
            split="train[:10000]",
            trust_remote_code=True
        )
        
        print(f"✅ 下载成功！共 {len(dataset)} 条数据")
        
        # 转换格式
        formatted_data = []
        for item in dataset:
            formatted_data.append({
                "conversations": item["conversations"]
            })
        
        # 保存
        output_path = "data/training/belle_10k.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存到: {output_path}")
        print(f"📊 数据统计:")
        print(f"   - 总对话数: {len(formatted_data)}")
        print(f"   - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return formatted_data
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n💡 替代方案:")
        print("   1. 手动下载: https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M")
        print("   2. 使用其他数据集: Firefly, Alpaca, ShareGPT")
        return None


def download_multi_sources():
    """从多个数据源下载"""
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 请先安装: pip install datasets")
        return
    
    all_data = []
    
    # 1. BELLE 中文数据
    print("\n1️⃣ 下载 BELLE 数据...")
    try:
        belle = load_dataset("BelleGroup/multiturn_chat_0.8M", split="train[:5000]")
        for item in belle:
            all_data.append({"conversations": item["conversations"]})
        print(f"   ✅ BELLE: 5000 条")
    except Exception as e:
        print(f"   ❌ BELLE 失败: {e}")
    
    # 2. Alpaca 英文数据
    print("\n2️⃣ 下载 Alpaca 数据...")
    try:
        alpaca = load_dataset("tatsu-lab/alpaca", split="train[:3000]")
        for item in alpaca:
            conv = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]
            all_data.append({"conversations": conv})
        print(f"   ✅ Alpaca: 3000 条")
    except Exception as e:
        print(f"   ❌ Alpaca 失败: {e}")
    
    # 保存合并数据
    if all_data:
        output_path = "data/training/combined_train.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 数据合并完成！")
        print(f"   - 总计: {len(all_data)} 条")
        print(f"   - 保存到: {output_path}")
    
    return all_data


if __name__ == "__main__":
    print("=" * 60)
    print("MyGPT 训练数据下载工具")
    print("=" * 60)
    
    print("\n选择下载方式:")
    print("1. 快速下载 (BELLE 10,000 条)")
    print("2. 多源下载 (BELLE + Alpaca, 约 8,000 条)")
    print("3. 退出")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == "1":
        download_sample_data()
    elif choice == "2":
        download_multi_sources()
    else:
        print("👋 已退出")
