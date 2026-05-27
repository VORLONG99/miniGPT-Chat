"""
模型服务 - 管理 GPT 模型的加载、推理和流式输出
"""
import asyncio
import torch
import torch.nn as nn
from typing import Optional, List, Dict, AsyncGenerator
import logging
import json
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from config import settings
from models.gpt_model import GPTModel, GPTConfig

logger = logging.getLogger(__name__)


class ModelService:
    """模型服务类"""
    
    def __init__(self):
        self.model: Optional[GPTModel] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False
        self.tokenizer: Optional[object] = None
        
    async def initialize(self):
        """初始化模型"""
        try:
            # 检测设备
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("💻 Using CPU mode")
            
            # 初始化模型
            config = GPTConfig(
                vocab_size=settings.VOCAB_SIZE,
                n_embed=settings.N_EMBED,
                n_head=settings.N_HEAD,
                n_layer=settings.N_LAYER,
                block_size=settings.BLOCK_SIZE,
                dropout=settings.DROPOUT
            )
            
            self.model = GPTModel(config)
            self.model.to(self.device)
            
            # 尝试加载预训练权重
            checkpoint_path = os.path.join(settings.MODEL_PATH, "best_model.pt")
            if os.path.exists(checkpoint_path):
                logger.info(f"📂 Loading model from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Model weights loaded")
            else:
                logger.info("ℹ️ No pretrained weights found, using random initialization")
            
            self.model.eval()
            self.is_loaded = True
            
            # 初始化简单分词器（实际应用中应使用 tiktoken 或自定义分词器）
            self._init_tokenizer()
            
            logger.info(f"✅ Model initialized: {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise
    
    def _init_tokenizer(self):
        """初始化简单字符级分词器"""
        # 简单的字符级编码，实际应用应使用 BPE 分词器
        self.char_to_idx = {chr(i): i for i in range(256)}
        self.idx_to_char = {i: chr(i) for i in range(256)}
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def encode(self, text: str) -> List[int]:
        """编码文本为 token IDs"""
        # 简单字符编码 + 对话格式
        encoded = []
        for char in text:
            idx = self.char_to_idx.get(char, 0)
            encoded.append(idx)
        return encoded
    
    def decode(self, token_ids: List[int]) -> str:
        """解码 token IDs 为文本"""
        chars = []
        for idx in token_ids:
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """生成文本（非流式）"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # 编码输入
        input_ids = self.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # 生成
        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # 解码输出
        output_text = self.decode(output_ids[0].tolist())
        return output_text[len(prompt):]  # 去掉 prompt 部分
    
    @torch.no_grad()
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> AsyncGenerator[str, None]:
        """流式生成文本"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # 编码输入
        input_ids = self.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # 流式生成
        current_ids = input_tensor.clone()
        generated_tokens = 0
        
        while generated_tokens < max_tokens:
            # 预测下一个 token
            logits = self.model(current_ids)[:, -1, :]
            
            # 温度采样
            logits = logits / temperature
            
            # Top-k 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 解码当前 token
            token_text = self.decode([next_token.item()])
            
            # 检查结束符
            if next_token.item() == self.eos_token_id:
                break
            
            # 返回生成的文本片段
            yield token_text
            
            # 更新序列
            current_ids = torch.cat([current_ids, next_token], dim=1)
            generated_tokens += 1
            
            # 小延迟以实现真实的流式效果
            await asyncio.sleep(0.01)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """对话生成（支持多轮对话上下文）"""
        # 构建对话 prompt
        prompt = self._build_chat_prompt(messages)
        
        if stream:
            async for token in self.generate_stream(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature
            ):
                yield token
        else:
            response = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            yield response
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建对话 prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"系统: {content}\n")
            elif role == "user":
                prompt_parts.append(f"用户: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"助手: {content}\n")
        
        prompt_parts.append("助手: ")
        return "".join(prompt_parts)
    
    def count_tokens(self, text: str) -> int:
        """计算 token 数量"""
        return len(self.encode(text))
    
    async def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False
        logger.info("🧹 Model service cleaned up")
