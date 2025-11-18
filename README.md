# 🤖 AI对话系统

基于 LangChain 和 Mem0 的智谱AI对话系统，具有智能记忆管理功能。

## ✨ 功能特性

### 核心功能
- 🤖 **智谱AI集成**：使用 GLM-4-Flash 对话模型
- 🧠 **智能记忆**：基于 mem0 的对话记忆管理
- 📝 **向量化**：智谱 Embedding-3 模型
- 💾 **本地存储**：ChromaDB 向量数据库
- 🔍 **网络搜索**：集成 SearXNG MCP 工具，支持实时信息检索
- ⚡ **流式输出**：实时响应

### 界面功能
- 💬 **实时对话**：流式输出
- 🔍 **记忆搜索**：基于语义相似度的记忆检索
- 🌐 **网络搜索**：可选启用 SearXNG 搜索，获取实时信息
- �️* **记忆查看**：可视化展示所有历史记忆
- � ***记忆导出**：导出为 JSON 格式
- 🗑️ **记忆管理**：支持删除单条或全部记忆
- ⚙️ **参数调节**：Temperature、上下文数量等可调
- 👤 **多用户**：支持不同用户 ID 独立记忆

## 📦 安装

### 使用 uv（推荐）

```bash
# 克隆项目
git clone <your-repo>
cd zhipu-mem0

# 安装依赖
uv sync
```

### 使用 pip

```bash
pip install -r requirements.txt
```

## 🔧 配置

### 基础配置

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的智谱 API 密钥：

```env
ZHIPU_API_KEY=your_zhipu_api_key_here
```

> 💡 获取 API Key：访问 [智谱AI开放平台](https://open.bigmodel.cn/)

### 网络搜索配置（可选）

如需启用网络搜索功能，需要配置 SearXNG：

1. 确保 SearXNG 服务运行在 `http://127.0.0.1:8888`
2. 在 `search_tool.py` 中配置 MCP 服务器路径
3. 在界面侧边栏启用"网络搜索"开关

## 🚀 运行

### 快速启动（推荐）

```bash
./run.sh
```

### 方式一：Streamlit 界面

```bash
# 使用 uv
uv run streamlit run app.py

# 或激活虚拟环境
source .venv/bin/activate
streamlit run app.py
```

访问 http://localhost:8501

### 方式二：FastAPI 服务

```bash
uv run python main.py
```

访问 http://localhost:8000

## 📖 使用指南

### Streamlit 界面

1. **开始对话**
   - 在底部输入框输入消息
   - AI 会根据上下文和历史记忆回复

2. **查看记忆**
   - 点击侧边栏 "👁️ 查看记忆" 按钮
   - 可以搜索相关记忆
   - 支持删除单条记忆

3. **导出记忆**
   - 点击 "💾 导出记忆" 按钮
   - 下载 JSON 格式的记忆文件

4. **调整设置**
   - **用户ID**：切换不同用户
   - **记忆功能**：开启/关闭记忆
   - **网络搜索**：启用后 AI 可使用搜索工具获取实时信息
   - **上下文数量**：控制每次使用的记忆条数
   - **Temperature**：控制回复的创造性

### API 接口

#### 对话接口

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "你好",
    "use_memory": true
  }'
```

#### 查询记忆

```bash
curl http://localhost:8000/memory/user123
```

#### 健康检查

```bash
curl http://localhost:8000/health
```

## 🏗️ 项目结构

```
zhipu-mem0/
├── app.py              # Streamlit 前端界面
├── main.py             # FastAPI 后端服务
├── config.py           # 配置管理
├── memory_manager.py   # 记忆管理器
├── search_tool.py      # MCP 搜索工具
├── .env.example        # 环境变量模板
├── .env                # 环境变量（需创建）
├── pyproject.toml      # 项目依赖
└── chroma_db/          # ChromaDB 数据存储（自动创建）
```

## 🧠 记忆管理原理

系统使用 mem0 进行智能记忆管理：

1. **记忆提取**：mem0 自动从对话中提取关键信息
   - 例如："我叫张三" → 提取为 "Name is 张三"

2. **语义搜索**：基于向量相似度检索相关记忆
   - 使用智谱 Embedding-3 模型向量化

3. **上下文整合**：将相关记忆注入到对话上下文中
   - AI 能够记住用户的偏好和历史信息

4. **本地存储**：所有数据存储在本地 ChromaDB
   - 无需额外数据库，开箱即用

## 🎯 特色功能

### 1. 流式输出
实时显示 AI 回复，提供流畅的对话体验。

### 2. 智能记忆提取
mem0 自动分析对话内容，提取关键信息存储为结构化记忆。

### 3. 语义相似度搜索
基于向量相似度的记忆检索，支持语义理解而非简单关键词匹配。

### 4. 记忆管理
- 查看所有记忆及其创建时间
- 按相关度排序搜索结果
- 删除单条或全部记忆
- 导出为 JSON 格式

### 5. 多用户支持
不同用户 ID 拥有独立的记忆空间，数据隔离。

### 6. 网络搜索集成
通过 LangChain MCP 集成 SearXNG 搜索引擎，AI 可根据需要自动搜索实时信息。

## 🔧 技术栈

- **LLM**: 智谱 GLM-4-Flash
- **Embedding**: 智谱 Embedding-3
- **记忆管理**: mem0
- **向量数据库**: ChromaDB
- **搜索引擎**: SearXNG (MCP)
- **Web框架**: Streamlit + FastAPI
- **语言框架**: LangChain

## 📝 开发说明

### 测试 mem0 配置

```bash
python test_mem0.py
```

### 清空记忆数据库

```bash
rm -rf chroma_db/
```

## ⚠️ 注意事项

1. **API 密钥安全**：不要将 `.env` 文件提交到版本控制
2. **记忆管理**：定期清理无用记忆以提高检索效率
3. **用户隔离**：确保不同用户使用不同的 user_id

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
