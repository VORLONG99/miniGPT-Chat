# MyGPT Chat - 完整项目启动指南

## 📁 项目结构

```
mygpt-chat/
├── backend/                    # Python 后端
│   ├── app/
│   │   ├── main.py            # FastAPI 主应用
│   │   ├── config.py          # 配置文件
│   │   ├── core/
│   │   │   ├── database.py    # 数据库模型
│   │   │   └── auth.py        # JWT 认证
│   │   ├── models/
│   │   │   └── schemas.py     # Pydantic 模型
│   │   ├── routers/
│   │   │   ├── auth.py        # 认证路由
│   │   │   ├── chat.py        # 对话路由（流式输出）
│   │   │   ├── history.py     # 历史记录
│   │   │   ├── rag.py         # RAG 知识库
│   │   │   └── stats.py       # 用量统计
│   │   └── services/
│   │       ├── model_service.py  # 模型服务
│   │       └── rag_service.py    # RAG 服务
│   └── requirements.txt
│
├── frontend/                   # Next.js 前端
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx       # 登录/注册页
│   │   │   └── chat/
│   │   │       └── page.tsx   # 对话界面
│   │   └── lib/
│   │       ├── api.ts         # API 调用
│   │       └── store.ts       # 状态管理
│   └── package.json
│
├── models/                     # GPT 模型代码
│   ├── gpt_model.py           # GPT 模型实现
│   └── trainer.py             # 训练脚本
│
└── data/                       # 数据目录
    ├── training/              # 训练数据
    └── rag/                   # RAG 知识库
```

---

## 🚀 快速启动

### 1. 后端启动

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 前端启动

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install
# 或 pnpm install

# 启动开发服务器
npm run dev
```

### 3. 访问应用

- 前端: http://localhost:3000
- 后端 API: http://localhost:8000
- API 文档: http://localhost:8000/docs

---

## 🎯 功能清单

### ✅ 已实现功能

#### 1. 流式输出 ⭐⭐⭐⭐⭐
- WebSocket 实时通信
- SSE (Server-Sent Events) 流式响应
- 逐字显示动画效果
- 打字机光标效果

#### 2. RAG 知识库 ⭐⭐⭐⭐⭐
- 文档上传（支持 TXT、PDF、MD）
- LangChain + Chroma 向量存储
- 语义相似度搜索
- 知识库开关控制

#### 3. 对话历史 ⭐⭐⭐⭐
- 多轮对话上下文保存
- 会话列表管理
- 消息历史查看
- 对话删除功能

#### 4. Markdown 渲染 ⭐⭐⭐⭐
- 支持代码高亮
- 表格、列表、引用
- GFM 语法支持
- 语法高亮 (highlight.js)

#### 5. 用户认证 ⭐⭐⭐
- JWT Token 认证
- 用户注册/登录
- 密码加密存储
- Token 自动刷新

#### 6. 用量统计 ⭐⭐⭐
- Token 消耗统计
- 请求次数统计
- 个人用量报告
- 管理员全局统计

---

## 🔧 配置说明

### 后端配置 (backend/config.py 或 .env)

```env
# 模型配置
MODEL_NAME=gpt-mini
VOCAB_SIZE=50257
N_EMBED=384
N_HEAD=6
N_LAYER=6
BLOCK_SIZE=256

# RAG 配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=./data/chroma_db

# JWT 配置
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# 数据库
DATABASE_URL=sqlite+aiosqlite:///./data/chat.db
```

### 前端配置 (frontend/.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📦 依赖清单

### 后端 Python 依赖

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
sqlalchemy>=2.0.23
aiosqlite>=0.19.0
torch>=2.1.0
numpy>=1.24.0
langchain>=0.1.0
langchain-community>=0.0.10
chromadb>=0.4.22
sentence-transformers>=2.2.0
python-multipart>=0.0.6
httpx>=0.25.2
websockets>=12.0
```

### 前端 Node.js 依赖

```json
{
  "next": "14.0.4",
  "react": "^18.2.0",
  "react-markdown": "^9.0.1",
  "axios": "^1.6.2",
  "zustand": "^4.4.7",
  "react-hot-toast": "^2.4.1",
  "@tanstack/react-query": "^5.14.2"
}
```

---

## 🎓 从零训练 GPT

### 1. 准备训练数据

将对话数据放在 `data/training/` 目录，格式如下：

```json
[
  {
    "conversations": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
    ]
  }
]
```

### 2. 开始训练

```bash
cd models
python trainer.py --config ../config.yaml
```

### 3. 使用训练好的模型

训练完成后，模型会保存在 `models/checkpoints/` 目录，后端会自动加载。

---

## 🔐 安全建议

1. **生产环境必须修改**:
   - `SECRET_KEY` - 使用强随机密钥
   - `DATABASE_URL` - 使用 PostgreSQL/MySQL
   - CORS 配置 - 限制允许的域名

2. **建议添加**:
   - HTTPS 证书
   - Rate Limiting
   - 输入验证和清理
   - 日志审计

---

## 📊 API 接口文档

### 认证接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/auth/register | 用户注册 |
| POST | /api/v1/auth/login | 用户登录 |
| GET | /api/v1/auth/me | 获取当前用户 |

### 对话接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/chat/completions | 对话（非流式） |
| POST | /api/v1/chat/stream | 对话（流式） |
| WS | /api/v1/chat/ws/{user_id} | WebSocket 对话 |

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/v1/history/conversations | 获取对话列表 |
| POST | /api/v1/rag/upload | 上传知识库文档 |
| GET | /api/v1/stats/my-stats | 获取用量统计 |

---

## 🎉 项目特点

1. **完全自训练** - 基于 nanoGPT 从零训练
2. **生产级架构** - FastAPI + Next.js 全栈
3. **流式输出** - 实时逐字显示
4. **RAG 增强** - 知识库问答
5. **完整功能** - 认证、历史、统计

---

## 📞 支持

如有问题，请查看 API 文档: http://localhost:8000/docs
