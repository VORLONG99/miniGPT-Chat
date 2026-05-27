"""
Pydantic 模型定义
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ==================== 用户相关 ====================

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


class UserCreate(BaseModel):
    """用户注册"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    ZINFOID_08Q: str = Field(..., min_length=6, max_length=100)


class UserLogin(BaseModel):
    """用户登录"""
    username: str
    ZINFOID_09Q: str


class UserResponse(BaseModel):
    """用户信息响应"""
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """令牌响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ==================== 对话相关 ====================

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """聊天消息"""
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[int] = None
    use_rag: bool = False
    stream: bool = True
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(512, ge=1, le=2048)


class ChatResponse(BaseModel):
    """对话响应"""
    conversation_id: int
    message: str
    role: str = "assistant"
    tokens: int
    created_at: datetime
    sources: Optional[List[Dict[str, Any]]] = None  # RAG 来源


class MessageResponse(BaseModel):
    """消息响应"""
    id: int
    conversation_id: int
    role: str
    content: str
    tokens: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    """对话会话响应"""
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    
    class Config:
        from_attributes = True


class ConversationDetail(BaseModel):
    """对话详情"""
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]
    
    class Config:
        from_attributes = True


# ==================== RAG 相关 ====================

class DocumentUpload(BaseModel):
    """文档上传"""
    name: str
    description: Optional[str] = None


class DocumentResponse(BaseModel):
    """文档响应"""
    id: int
    name: str
    description: Optional[str]
    file_type: str
    chunk_count: int
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class RAGSearchRequest(BaseModel):
    """RAG 搜索请求"""
    query: str
    top_k: int = Field(5, ge=1, le=20)


class RAGSearchResult(BaseModel):
    """RAG 搜索结果"""
    content: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = None


# ==================== 统计相关 ====================

class UsageStatsResponse(BaseModel):
    """用量统计响应"""
    date: datetime
    input_tokens: int
    output_tokens: int
    total_tokens: int
    request_count: int


class StatsSummary(BaseModel):
    """统计摘要"""
    total_users: int
    total_conversations: int
    total_messages: int
    total_tokens: int
    avg_tokens_per_message: float
