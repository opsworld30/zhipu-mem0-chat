#!/bin/bash

# 系统启动脚本

echo "🚀 启动AI对话系统..."
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 未找到虚拟环境，请先运行: uv sync"
    exit 1
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件"
    echo "📝 正在创建 .env 文件..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ 已创建 .env 文件，请编辑并填入你的 ZHIPU_API_KEY"
        echo ""
        echo "获取 API Key: https://open.bigmodel.cn/"
        exit 1
    else
        echo "❌ 未找到 .env.example 文件"
        exit 1
    fi
fi

# 启动应用
echo "✓ 环境检查通过"
echo "🎨 启动 Streamlit 界面..."
echo ""
echo "📱 应用将在浏览器中打开: http://localhost:8501"
echo ""

# 使用 uv 运行
uv run streamlit run app.py
