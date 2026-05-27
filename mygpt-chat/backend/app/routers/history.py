"""
对话历史路由
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from app.core.database import get_db, User, Conversation, Message
from app.core.auth import get_current_active_user
from app.models.schemas import (
    ConversationResponse, 
    ConversationDetail,
    MessageResponse
)

router = APIRouter()


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取对话列表"""
    # 查询对话及消息数量
    result = await db.execute(
        select(
            Conversation,
            func.count(Message.id).label("message_count")
        )
        .outerjoin(Message)
        .where(
            Conversation.user_id == current_user.id,
            Conversation.is_archived == False
        )
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    
    conversations = []
    for conv, msg_count in result:
        conversations.append(ConversationResponse(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=msg_count
        ))
    
    return conversations


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取对话详情"""
    # 查询对话
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 查询消息
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageResponse.model_validate(msg) for msg in messages]
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """删除对话"""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    conversation.is_archived = True
    await db.commit()
    
    return {"message": "对话已删除"}


@router.patch("/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: int,
    title: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """更新对话标题"""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    conversation.title = title
    await db.commit()
    
    return {"message": "标题已更新"}


@router.get("/messages/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取单条消息"""
    result = await db.execute(
        select(Message)
        .join(Conversation)
        .where(
            Message.id == message_id,
            Conversation.user_id == current_user.id
        )
    )
    message = result.scalar_one_or_none()
    
    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")
    
    return MessageResponse.model_validate(message)


@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """删除消息"""
    result = await db.execute(
        select(Message)
        .join(Conversation)
        .where(
            Message.id == message_id,
            Conversation.user_id == current_user.id
        )
    )
    message = result.scalar_one_or_none()
    
    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")
    
    await db.delete(message)
    await db.commit()
    
    return {"message": "消息已删除"}
