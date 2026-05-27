"""
对话路由 - 核心聊天功能
支持流式输出、RAG 增强、多轮对话
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict
import json
import logging
from datetime import datetime

from app.core.database import get_db, User, Conversation, Message, UsageStats
from app.core.auth import get_current_active_user
from app.models.schemas import ChatRequest, ChatResponse, MessageResponse
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
    
    async def send_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_text(message)


manager = ConnectionManager()


@router.post("/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """对话完成接口（非流式）"""
    from app.main import app
    
    model_service = app.state.model_service
    
    # 获取或创建对话
    conversation = await _get_or_create_conversation(
        db, current_user.id, request.conversation_id, request.message
    )
    
    # 保存用户消息
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
        tokens=model_service.count_tokens(request.message)
    )
    db.add(user_message)
    
    # 获取对话历史
    history = await _get_conversation_history(db, conversation.id)
    
    # RAG 增强
    sources = None
    if request.use_rag:
        search_results = await rag_service.search(request.message)
        if search_results:
            context = rag_service.build_context(search_results)
            history.insert(0, {
                "role": "system",
                "content": f"参考以下知识库内容回答问题：\n\n{context}"
            })
            sources = search_results
    
    # 生成回复
    response_text = ""
    async for token in model_service.chat(
        history,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=False
    ):
        response_text = token
    
    # 保存助手消息
    assistant_tokens = model_service.count_tokens(response_text)
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=response_text,
        tokens=assistant_tokens,
        metadata={"sources": sources, "use_rag": request.use_rag}
    )
    db.add(assistant_message)
    
    # 更新用量统计
    await _update_usage_stats(
        db, current_user.id, 
        user_message.tokens, 
        assistant_tokens
    )
    
    await db.commit()
    await db.refresh(assistant_message)
    
    return ChatResponse(
        conversation_id=conversation.id,
        message=response_text,
        tokens=assistant_tokens,
        created_at=assistant_message.created_at,
        sources=sources
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """流式对话接口"""
    from app.main import app
    
    model_service = app.state.model_service
    
    async def generate():
        # 获取或创建对话
        conversation = await _get_or_create_conversation(
            db, current_user.id, request.conversation_id, request.message
        )
        
        # 保存用户消息
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            tokens=model_service.count_tokens(request.message)
        )
        db.add(user_message)
        await db.commit()
        
        # 获取对话历史
        history = await _get_conversation_history(db, conversation.id)
        
        # RAG 增强
        sources = None
        if request.use_rag:
            search_results = await rag_service.search(request.message)
            if search_results:
                context = rag_service.build_context(search_results)
                history.insert(0, {
                    "role": "system",
                    "content": f"参考以下知识库内容回答问题：\n\n{context}"
                })
                sources = search_results
        
        # 发送开始标记
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation.id})}\n\n"
        
        # 流式生成
        full_response = ""
        async for token in model_service.chat(
            history,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True
        ):
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        # 保存助手消息
        assistant_tokens = model_service.count_tokens(full_response)
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=full_response,
            tokens=assistant_tokens,
            metadata={"sources": sources, "use_rag": request.use_rag}
        )
        db.add(assistant_message)
        
        # 更新用量统计
        await _update_usage_stats(
            db, current_user.id,
            user_message.tokens,
            assistant_tokens
        )
        
        await db.commit()
        
        # 发送结束标记
        yield f"data: {json.dumps({'type': 'done', 'tokens': assistant_tokens})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.websocket("/ws/{user_id}")
async def websocket_chat(
    websocket: WebSocket,
    user_id: int,
    token: str
):
    """WebSocket 对话接口"""
    from app.main import app
    
    # 验证 token（简化版，实际应使用 JWT 验证）
    try:
        from app.core.auth import get_current_user
        from app.core.database import AsyncSessionLocal
        # 这里应该验证 token，简化处理
        
        await manager.connect(websocket, user_id)
        model_service = app.state.model_service
        
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            use_rag = message_data.get("use_rag", False)
            
            # 构建简单对话
            history = [{"role": "user", "content": user_message}]
            
            # RAG 增强
            if use_rag:
                search_results = await rag_service.search(user_message)
                if search_results:
                    context = rag_service.build_context(search_results)
                    history.insert(0, {
                        "role": "system",
                        "content": f"参考以下知识库内容：\n\n{context}"
                    })
            
            # 流式生成
            async for token in model_service.chat(history, stream=True):
                await websocket.send_text(json.dumps({
                    "type": "token",
                    "content": token
                }))
            
            # 发送完成信号
            await websocket.send_text(json.dumps({
                "type": "done"
            }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)


async def _get_or_create_conversation(
    db: AsyncSession,
    user_id: int,
    conversation_id: int = None,
    first_message: str = ""
) -> Conversation:
    """获取或创建对话"""
    if conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation
    
    # 创建新对话
    title = first_message[:50] + "..." if len(first_message) > 50 else first_message
    conversation = Conversation(
        user_id=user_id,
        title=title or "新对话"
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    
    return conversation


async def _get_conversation_history(
    db: AsyncSession,
    conversation_id: int,
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """获取对话历史"""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(max_messages)
    )
    messages = result.scalars().all()[::-1]  # 按时间正序
    
    return [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]


async def _update_usage_stats(
    db: AsyncSession,
    user_id: int,
    input_tokens: int,
    output_tokens: int
):
    """更新用量统计"""
    today = datetime.utcnow().date()
    
    result = await db.execute(
        select(UsageStats).where(
            UsageStats.user_id == user_id,
            UsageStats.date == today
        )
    )
    stats = result.scalar_one_or_none()
    
    if stats:
        stats.input_tokens += input_tokens
        stats.output_tokens += output_tokens
        stats.total_tokens += input_tokens + output_tokens
        stats.request_count += 1
    else:
        stats = UsageStats(
            user_id=user_id,
            date=today,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            request_count=1
        )
        db.add(stats)
