"""
用量统计路由
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
from datetime import datetime, timedelta

from app.core.database import get_db, User, Conversation, Message, UsageStats
from app.core.auth import get_current_active_user
from app.models.schemas import UsageStatsResponse, StatsSummary

router = APIRouter()


@router.get("/my-stats", response_model=List[UsageStatsResponse])
async def get_my_stats(
    days: int = 7,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户的用量统计"""
    start_date = datetime.utcnow().date() - timedelta(days=days)
    
    result = await db.execute(
        select(UsageStats)
        .where(
            UsageStats.user_id == current_user.id,
            UsageStats.date >= start_date
        )
        .order_by(UsageStats.date.desc())
    )
    
    stats = result.scalars().all()
    
    return [
        UsageStatsResponse(
            date=stat.date,
            input_tokens=stat.input_tokens,
            output_tokens=stat.output_tokens,
            total_tokens=stat.total_tokens,
            request_count=stat.request_count
        )
        for stat in stats
    ]


@router.get("/my-summary")
async def get_my_summary(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户的统计摘要"""
    # 总对话数
    conv_result = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.user_id == current_user.id)
    )
    total_conversations = conv_result.scalar()
    
    # 总消息数
    msg_result = await db.execute(
        select(func.count(Message.id))
        .join(Conversation)
        .where(Conversation.user_id == current_user.id)
    )
    total_messages = msg_result.scalar()
    
    # 总 token 数
    token_result = await db.execute(
        select(func.sum(UsageStats.total_tokens))
        .where(UsageStats.user_id == current_user.id)
    )
    total_tokens = token_result.scalar() or 0
    
    # 平均每条消息 token 数
    avg_tokens = total_tokens / total_messages if total_messages > 0 else 0
    
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "avg_tokens_per_message": round(avg_tokens, 2)
    }


@router.get("/global-summary", response_model=StatsSummary)
async def get_global_summary(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取全局统计摘要（管理员功能）"""
    # 检查管理员权限
    if not current_user.is_admin:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="需要管理员权限"
        )
    
    # 总用户数
    user_result = await db.execute(
        select(func.count(User.id))
    )
    total_users = user_result.scalar()
    
    # 总对话数
    conv_result = await db.execute(
        select(func.count(Conversation.id))
    )
    total_conversations = conv_result.scalar()
    
    # 总消息数
    msg_result = await db.execute(
        select(func.count(Message.id))
    )
    total_messages = msg_result.scalar()
    
    # 总 token 数
    token_result = await db.execute(
        select(func.sum(UsageStats.total_tokens))
    )
    total_tokens = token_result.scalar() or 0
    
    # 平均每条消息 token 数
    avg_tokens = total_tokens / total_messages if total_messages > 0 else 0
    
    return StatsSummary(
        total_users=total_users,
        total_conversations=total_conversations,
        total_messages=total_messages,
        total_tokens=total_tokens,
        avg_tokens_per_message=round(avg_tokens, 2)
    )


@router.get("/top-users")
async def get_top_users(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """获取用量最高的用户（管理员功能）"""
    if not current_user.is_admin:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="需要管理员权限"
        )
    
    result = await db.execute(
        select(
            User.username,
            func.sum(UsageStats.total_tokens).label("total_tokens"),
            func.sum(UsageStats.request_count).label("total_requests")
        )
        .join(UsageStats)
        .group_by(User.id)
        .order_by(func.sum(UsageStats.total_tokens).desc())
        .limit(limit)
    )
    
    top_users = []
    for username, tokens, requests in result:
        top_users.append({
            "username": username,
            "total_tokens": tokens,
            "total_requests": requests
        })
    
    return top_users
