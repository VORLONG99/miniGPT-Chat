"""
RAG 服务 - 检索增强生成
支持文档上传、向量化、相似度搜索
"""
import os
import logging
from typing import List, Dict, Optional
import asyncio
import hashlib

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain.schema import Document
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logging.warning("LangChain not installed, RAG features will be limited")

from config import settings
from app.core.database import AsyncSessionLocal, KnowledgeBase

logger = logging.getLogger(__name__)


class SimpleEmbedding:
    """简单的词袋嵌入（LangChain 未安装时的备用方案）"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        embeddings = []
        for text in texts:
            # 简单哈希嵌入
            h = hashlib.md5(text.encode()).hexdigest()
            embedding = [float(int(h[i:i+2], 16)) / 255 for i in range(0, 32, 2)]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        return self.embed_documents([text])[0]


class RAGService:
    """RAG 服务类"""
    
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.text_splitter = None
        self.is_initialized = False
        
    async def initialize(self):
        """初始化 RAG 服务"""
        try:
            # 确保目录存在
            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            
            if HAS_LANGCHAIN:
                # 使用 HuggingFace 嵌入模型
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # 文本分割器
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
                )
                
                # 初始化向量存储
                self.vector_store = Chroma(
                    persist_directory=settings.VECTOR_DB_PATH,
                    embedding_function=self.embeddings
                )
                
                logger.info("✅ RAG service initialized with LangChain")
            else:
                # 使用简单嵌入
                self.embeddings = SimpleEmbedding()
                logger.info("✅ RAG service initialized with simple embedding")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG service: {e}")
            raise
    
    async def add_document(
        self, 
        file_path: str, 
        name: str,
        description: Optional[str] = None,
        file_type: str = "txt"
    ) -> int:
        """添加文档到知识库"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 加载文档
            if HAS_LANGCHAIN:
                if file_type == "pdf":
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                documents = loader.load()
                
                # 分割文档
                chunks = self.text_splitter.split_documents(documents)
                
                # 添加到向量存储
                self.vector_store.add_documents(chunks)
                self.vector_store.persist()
                
                # 保存到数据库
                async with AsyncSessionLocal() as db:
                    kb_entry = KnowledgeBase(
                        name=name,
                        description=description,
                        file_path=file_path,
                        file_type=file_type,
                        chunk_count=len(chunks),
                        is_active=True
                    )
                    db.add(kb_entry)
                    await db.commit()
                    await db.refresh(kb_entry)
                    
                    logger.info(f"✅ Added document: {name}, {len(chunks)} chunks")
                    return kb_entry.id
            else:
                logger.warning("LangChain not available, document processing skipped")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            raise
    
    async def add_text(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """直接添加文本到知识库"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if HAS_LANGCHAIN:
                # 分割文本
                chunks = self.text_splitter.split_text(text)
                
                # 创建文档对象
                documents = [
                    Document(page_content=chunk, metadata=metadata or {})
                    for chunk in chunks
                ]
                
                # 添加到向量存储
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
                
                logger.info(f"✅ Added text: {len(chunks)} chunks")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to add text: {e}")
            return False
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """搜索相关文档"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if HAS_LANGCHAIN and self.vector_store:
                # 相似度搜索
                results = self.vector_store.similarity_search_with_score(
                    query, 
                    k=top_k
                )
                
                # 格式化结果
                search_results = []
                for doc, score in results:
                    search_results.append({
                        "content": doc.page_content,
                        "score": float(score),
                        "source": doc.metadata.get("source", "unknown"),
                        "metadata": doc.metadata
                    })
                
                return search_results
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def build_context(self, search_results: List[Dict], max_length: int = 2000) -> str:
        """构建上下文"""
        context_parts = []
        current_length = 0
        
        for result in search_results:
            content = result["content"]
            if current_length + len(content) > max_length:
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n\n".join(context_parts)
    
    async def list_documents(self) -> List[Dict]:
        """列出所有文档"""
        try:
            async with AsyncSessionLocal() as db:
                from sqlalchemy import select
                result = await db.execute(
                    select(KnowledgeBase).where(KnowledgeBase.is_active == True)
                )
                documents = result.scalars().all()
                
                return [
                    {
                        "id": doc.id,
                        "name": doc.name,
                        "description": doc.description,
                        "file_type": doc.file_type,
                        "chunk_count": doc.chunk_count,
                        "created_at": doc.created_at,
                        "is_active": doc.is_active
                    }
                    for doc in documents
                ]
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []
    
    async def delete_document(self, doc_id: int) -> bool:
        """删除文档"""
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(KnowledgeBase).where(KnowledgeBase.id == doc_id)
                )
                doc = result.scalar_one_or_none()
                
                if doc:
                    doc.is_active = False
                    await db.commit()
                    logger.info(f"✅ Deleted document: {doc_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False


# 全局 RAG 服务实例
rag_service = RAGService()
