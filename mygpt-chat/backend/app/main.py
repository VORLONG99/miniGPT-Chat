"""
MyGPT Chat - FastAPI 主应用
支持流式输出、RAG、用户认证等完整功能
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from app.routers import chat, auth, history, rag, stats
from app.core.database import init_db
from app.services.model_service import ModelService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局模型服务实例
model_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global model_service
    
    # 启动时初始化
    logger.info("🚀 Starting MyGPT Chat Server...")
    
    # 初始化数据库
    await init_db()
    logger.info("✅ Database initialized")
    
    # 初始化模型服务
    model_service = ModelService()
    await model_service.initialize()
    logger.info("✅ Model service initialized")
    
    # 将模型服务注入到 app state
    app.state.model_service = model_service
    
    logger.info(f"🎉 {settings.APP_NAME} v{settings.APP_VERSION} is ready!")
    
    yield
    
    # 关闭时清理
    logger.info("🛑 Shutting down MyGPT Chat Server...")
    if model_service:
        await model_service.cleanup()
    logger.info("👋 Goodbye!")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于自训练 GPT 的智能对话平台",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router, prefix=f"{settings.API_PREFIX}/auth", tags=["认证"])
app.include_router(chat.router, prefix=f"{settings.API_PREFIX}/chat", tags=["对话"])
app.include_router(history.router, prefix=f"{settings.API_PREFIX}/history", tags=["历史"])
app.include_router(rag.router, prefix=f"{settings.API_PREFIX}/rag", tags=["知识库"])
app.include_router(stats.router, prefix=f"{settings.API_PREFIX}/stats", tags=["统计"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}
