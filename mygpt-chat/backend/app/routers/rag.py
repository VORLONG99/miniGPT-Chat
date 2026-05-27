"""
RAG 知识库路由
支持文档上传、搜索、管理
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import os
import logging

from app.core.database import get_db, User
from app.core.auth import get_current_active_user
from app.models.schemas import (
    DocumentUpload, 
    DocumentResponse,
    RAGSearchRequest,
    RAGSearchResult
)
from app.services.rag_service import rag_service
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """上传文档到知识库"""
    # 检查文件类型
    allowed_types = ["txt", "pdf", "md", "json"]
    file_ext = file.filename.split(".")[-1].lower()
    
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}. 支持的类型: {allowed_types}"
        )
    
    # 保存文件
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    file_path = os.path.join(
        settings.VECTOR_DB_PATH, 
        f"{current_user.id}_{file.filename}"
    )
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # 添加到知识库
    doc_name = name or file.filename
    doc_id = await rag_service.add_document(
        file_path=file_path,
        name=doc_name,
        description=description,
        file_type=file_ext
    )
    
    return JSONResponse(
        status_code=200,
        content={
            "id": doc_id,
            "name": doc_name,
            "description": description,
            "file_type": file_ext,
            "chunk_count": 0,
            "created_at": None,
            "is_active": True
        }
    )


@router.post("/text")
async def add_text(
    text: str = Form(...),
    source: str = Form("manual"),
    current_user: User = Depends(get_current_active_user)
):
    """直接添加文本到知识库"""
    success = await rag_service.add_text(
        text, 
        metadata={"source": source, "user_id": current_user.id}
    )
    
    if success:
        return {"message": "文本已添加到知识库"}
    else:
        raise HTTPException(
            status_code=500,
            detail="添加文本失败"
        )


@router.post("/search", response_model=List[RAGSearchResult])
async def search_knowledge(
    request: RAGSearchRequest,
    current_user: User = Depends(get_current_active_user)
):
    """搜索知识库"""
    results = await rag_service.search(
        request.query,
        top_k=request.top_k
    )
    
    return [
        RAGSearchResult(
            content=r["content"],
            score=r["score"],
            source=r["source"],
            metadata=r.get("metadata")
        )
        for r in results
    ]


@router.get("/documents", response_model=List[dict])
async def list_documents(
    current_user: User = Depends(get_current_active_user)
):
    """列出知识库文档"""
    documents = await rag_service.list_documents()
    return documents


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """删除文档"""
    success = await rag_service.delete_document(doc_id)
    
    if success:
        return {"message": "文档已删除"}
    else:
        raise HTTPException(
            status_code=404,
            detail="文档不存在"
        )


@router.get("/status")
async def rag_status():
    """RAG 服务状态"""
    return {
        "is_initialized": rag_service.is_initialized,
        "vector_db_path": settings.VECTOR_DB_PATH,
        "embedding_model": settings.EMBEDDING_MODEL
    }
