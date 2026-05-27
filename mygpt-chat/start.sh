#!/bin/bash

echo "🚀 MyGPT Chat 启动脚本"
echo "========================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装 Python 3.10+"
    exit 1
fi

# 检查 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js 18+"
    exit 1
fi

echo "✅ 环境检查通过"

# 创建必要的目录
mkdir -p data/training data/rag models/checkpoints logs

# 启动后端
echo ""
echo "📦 启动后端服务..."
cd backend

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
if [ ! -f "deps_installed" ]; then
    echo "安装 Python 依赖..."
    pip install -r requirements.txt
    touch deps_installed
fi

# 启动后端
echo "🌟 后端服务启动在 http://localhost:8000"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

cd ..

# 启动前端
echo ""
echo "📦 启动前端服务..."
cd frontend

# 安装依赖
if [ ! -d "node_modules" ]; then
    echo "安装 Node.js 依赖..."
    npm install
fi

# 启动前端
echo "🌟 前端服务启动在 http://localhost:3000"
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "========================"
echo "✅ MyGPT Chat 启动成功！"
echo ""
echo "🌐 访问地址："
echo "   前端: http://localhost:3000"
echo "   后端: http://localhost:8000"
echo "   API文档: http://localhost:8000/docs"
echo ""
echo "📋 按 Ctrl+C 停止所有服务"
echo ""

# 等待中断信号
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait
