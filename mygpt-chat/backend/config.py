"""
MyGPT Chat - 配置文件
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    APP_NAME: str = "MyGPT Chat"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API 配置
    API_PREFIX: str = "/api/v1"
    
    # 模型配置
    MODEL_NAME: str = "gpt-mini"  # 模型名称
    MODEL_PATH: str = "./models/checkpoints"  # 模型保存路径
    VOCAB_SIZE: int = 50257  # 词表大小
    N_EMBED: int = 384  # 嵌入维度
    N_HEAD: int = 6  # 注意力头数
    N_LAYER: int = 6  # Transformer 层数
    BLOCK_SIZE: int = 256  # 最大序列长度
    DROPOUT: float = 0.2
    
    # 训练配置
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 3e-4
    MAX_EPOCHS: int = 10
    WARMUP_STEPS: int = 100
    
    # RAG 配置
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_PATH: str = "./data/chroma_db"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # 数据库配置
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/chat.db"
    
    # JWT 配置
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7天
    
    # CORS 配置
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
