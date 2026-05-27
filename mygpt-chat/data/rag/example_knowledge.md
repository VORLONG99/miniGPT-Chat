# 这是一个示例知识库文档
# 用于 RAG（检索增强生成）功能演示

## MyGPT Chat 平台介绍

MyGPT Chat 是一个基于自训练 GPT 模型的智能对话平台。平台特点：

1. **从零训练模型** - 使用 nanoGPT 架构，完整的 Transformer 实现
2. **流式输出** - 实时逐字显示生成内容
3. **RAG 增强** - 支持上传知识库文档，让 AI 基于特定知识回答问题
4. **完整功能** - 用户认证、对话历史、用量统计

## 技术架构

### 后端技术栈
- FastAPI - 高性能 Python Web 框架
- SQLAlchemy - 异步 ORM
- PyTorch - 深度学习框架
- LangChain - RAG 应用开发框架
- ChromaDB - 向量数据库

### 前端技术栈
- Next.js 14 - React 全栈框架
- Tailwind CSS - 原子化 CSS
- Zustand - 状态管理
- React Query - 数据请求

### 模型架构
- 基于 GPT-2 架构
- 参数量：约 10M（可配置）
- 层数：6 层 Transformer
- 注意力头：6 个
- 嵌入维度：384

## 如何使用

### 基本对话
1. 注册/登录账号
2. 在对话框输入问题
3. AI 将实时生成回答

### 启用 RAG
1. 点击左侧"启用知识库"开关
2. 上传相关文档（支持 TXT、PDF、MD）
3. 系统自动将文档向量化存储
4. 对话时会检索相关知识辅助回答

### 查看历史
- 左侧显示所有对话历史
- 点击可恢复历史对话

## 常见问题

**Q: 模型训练需要多长时间？**
A: 在单张 RTX 3090 上，训练 10M 参数模型约需 2-4 小时。

**Q: 支持哪些语言的对话？**
A: 主要支持中文和英文，可根据训练数据扩展。

**Q: 如何提高回答质量？**
A: 可以通过以下方式：
- 提供更多训练数据
- 启用 RAG 知识库
- 调整 temperature 参数

## 联系我们

GitHub: https://github.com/yourname/mygpt-chat
文档: https://docs.mygpt-chat.com
