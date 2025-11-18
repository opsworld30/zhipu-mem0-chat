import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import config
from memory_manager import MemoryManager
from datetime import datetime
import time

# 页面配置
st.set_page_config(
    page_title="AI对话系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
    }

    /* 主标题 */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(102, 126, 234, 0.1);
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #f0f2f6 100%);
        border-right: 1px solid #e0e0e0;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        padding: 0;
    }

    /* 记忆卡片 */
    .memory-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
    }

    .memory-card:hover {
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }

    .user-memory {
        border-left-color: #667eea;
        background: linear-gradient(135deg, #667eea08 0%, transparent 100%);
    }

    .assistant-memory {
        border-left-color: #764ba2;
        background: linear-gradient(135deg, #764ba208 0%, transparent 100%);
    }

    /* 按钮美化 */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .stButton>button[kind="secondary"] {
        background: linear-gradient(135deg, #f0f2f6 0%, #e0e0e0 100%);
        color: #333;
    }

    /* 指标卡片 */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .stMetric label {
        font-size: 0.9rem !important;
        color: #666 !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 输入框美化 */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* 滑块美化 */
    .stSlider>div>div>div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* 聊天消息美化 */
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* 聊天输入框 */
    .stChatInputContainer {
        border-top: 1px solid #e0e0e0;
        padding: 1rem 0;
        background: white;
    }

    /* Expander 美化 */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
        color: #333;
    }

    .streamlit-expanderHeader:hover {
        background: #f8f9ff;
    }

    /* 信息提示框 */
    .stAlert {
        border-radius: 10px;
        border: none;
    }

    /* 下载按钮 */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #52c41a 0%, #73d13d 100%) !important;
        color: white !important;
    }

    /* 分隔线 */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }

    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f0f2f6;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5568d3 0%, #653a8b 100%);
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"

if "llm" not in st.session_state:
    st.session_state.llm = ChatZhipuAI(
        model="glm-4-flash",
        api_key=config.ZHIPU_API_KEY,
        temperature=0.7,
    )

if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager()

if "show_memories" not in st.session_state:
    st.session_state.show_memories = False

if "context_limit" not in st.session_state:
    st.session_state.context_limit = 5

# 主标题
st.markdown('<h1 class="main-header">🤖 AI对话系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">基于 GLM-4-Flash 和 Mem0 的智能对话助手</p>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")

    # 用户信息
    with st.expander("👤 用户信息", expanded=True):
        new_user_id = st.text_input(
            "用户ID",
            value=st.session_state.user_id,
            help="不同的用户ID会有独立的记忆"
        )
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
            st.session_state.messages = []
            st.rerun()

    # 记忆设置
    with st.expander("🧠 记忆设置", expanded=True):
        use_memory = st.checkbox("启用记忆功能", value=True, help="关闭后将不使用历史上下文")
        st.session_state.context_limit = st.slider(
            "上下文记忆数量",
            min_value=1,
            max_value=10,
            value=5,
            help="每次对话使用的历史记忆数量"
        )

    # 模型设置
    with st.expander("🔧 模型设置"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="控制回复的随机性，值越高越有创造性"
        )
        if temperature != st.session_state.llm.temperature:
            st.session_state.llm.temperature = temperature

    st.markdown("---")

    # 统计信息
    memories = st.session_state.memory_manager.get_all_memories(st.session_state.user_id)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 对话轮数", len(st.session_state.messages))
    with col2:
        st.metric("🧠 记忆数量", len(memories))

    st.markdown("---")

    # 功能按钮
    st.subheader("🎯 功能")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("👁️ 查看记忆", use_container_width=True):
            st.session_state.show_memories = not st.session_state.show_memories
            st.rerun()

    col3, col4 = st.columns(2)

    with col3:
        if st.button("💾 导出记忆", use_container_width=True):
            export_data = st.session_state.memory_manager.export_memories(st.session_state.user_id)
            st.download_button(
                label="📥 下载JSON",
                data=export_data,
                file_name=f"memories_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    with col4:
        if st.button("🧹 清空记忆", use_container_width=True, type="secondary"):
            if st.session_state.memory_manager.delete_all_memories(st.session_state.user_id):
                st.success("✓ 记忆已清空")
                st.rerun()
            else:
                st.error("✗ 清空失败")

    st.markdown("---")

    # 关于信息
    with st.expander("ℹ️ 关于"):
        st.markdown("""
        **AI对话系统**

        - 🤖 模型: GLM-4-Flash
        - 🧠 记忆: mem0 + ChromaDB
        - 📝 向量化: Embedding-3
        - 🎨 界面: Streamlit

        由智谱AI提供支持
        """)

# 主界面 - 记忆显示区域
if st.session_state.show_memories:
    st.subheader("🧠 记忆库")

    if memories:
        # 按创建时间分组
        st.info(f"共有 {len(memories)} 条记忆")

        # 搜索功能
        search_query = st.text_input("🔍 搜索记忆", placeholder="输入关键词搜索相关记忆...")

        if search_query:
            search_results = st.session_state.memory_manager.get_context(
                st.session_state.user_id,
                search_query,
                limit=10
            )
            st.success(f"✨ 找到 {len(search_results)} 条相关记忆")

            if search_results:
                for mem in search_results:
                    role = mem.get('role', 'unknown')
                    memory_text = mem.get('memory', '')
                    score = mem.get('score', 0)
                    created_at = mem.get('created_at', '')

                    role_icon = "👤" if role == "user" else "🤖"
                    role_class = "user-memory" if role == "user" else "assistant-memory"
                    role_color = "#667eea" if role == "user" else "#764ba2"

                    st.markdown(f"""
                    <div class="memory-card {role_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong style="color: {role_color};">{role_icon} {role.upper()}</strong>
                            <span style="background: linear-gradient(135deg, {role_color}20 0%, {role_color}10 100%);
                                         padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                                相关度: {score:.1%}
                            </span>
                        </div>
                        <div style="color: #333; line-height: 1.6; margin: 0.5rem 0;">{memory_text}</div>
                        <small style="color: #999; font-size: 0.85rem;">📅 {created_at}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🔍 未找到相关记忆")
        else:
            # 显示所有记忆
            for i, mem in enumerate(memories):
                role = mem.get('role', 'unknown')
                memory_text = mem.get('memory', '')
                created_at = mem.get('created_at', '')
                mem_id = mem.get('id', '')

                role_icon = "👤" if role == "user" else "🤖"
                role_class = "user-memory" if role == "user" else "assistant-memory"
                role_color = "#667eea" if role == "user" else "#764ba2"

                col1, col2 = st.columns([0.92, 0.08])

                with col1:
                    st.markdown(f"""
                    <div class="memory-card {role_class}">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <strong style="color: {role_color};">{role_icon} {role.upper()}</strong>
                            <span style="margin-left: 0.5rem; background: {role_color}15;
                                         padding: 0.1rem 0.5rem; border-radius: 12px; font-size: 0.75rem;">
                                #{i+1}
                            </span>
                        </div>
                        <div style="color: #333; line-height: 1.6; margin: 0.5rem 0;">{memory_text}</div>
                        <small style="color: #999; font-size: 0.85rem;">📅 {created_at}</small>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    if st.button("🗑️", key=f"del_{mem_id}", help="删除此记忆", type="secondary"):
                        if st.session_state.memory_manager.delete_memory(mem_id):
                            st.rerun()
    else:
        st.info("📭 暂无记忆，开始对话即可创建记忆")

    st.markdown("---")

# 对话显示区域
if not st.session_state.messages:
    # 显示欢迎页面
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: white; border-radius: 20px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.08); margin: 2rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">👋</div>
        <h2 style="color: #333; margin-bottom: 1rem;">欢迎使用AI对话系统</h2>
        <p style="color: #666; font-size: 1.1rem; line-height: 1.8; max-width: 600px; margin: 0 auto;">
            我是基于 GLM-4-Flash 的智能助手，具备记忆功能，能够记住我们的对话内容。
            <br><br>
            💡 <strong>提示：</strong>你可以问我任何问题，我会根据我们之前的对话来提供更个性化的回答！
        </p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <div style="background: #f8f9ff; padding: 1rem 1.5rem; border-radius: 12px; border: 2px solid #667eea20;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
                <div style="color: #667eea; font-weight: 600;">智能记忆</div>
            </div>
            <div style="background: #f8f9ff; padding: 1rem 1.5rem; border-radius: 12px; border: 2px solid #764ba220;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚡</div>
                <div style="color: #764ba2; font-weight: 600;">快速响应</div>
            </div>
            <div style="background: #f8f9ff; padding: 1rem 1.5rem; border-radius: 12px; border: 2px solid #667eea20;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="color: #667eea; font-weight: 600;">精准理解</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 显示所有历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 对话输入
if prompt := st.chat_input("💭 输入你的消息..."):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成助手回复
    with st.chat_message("assistant"):
        # 构建消息
        messages = [SystemMessage(content="你是一个友好、专业的AI助手，擅长理解用户需求并提供有帮助的回答。")]

        # 添加记忆上下文
        if use_memory:
            context = st.session_state.memory_manager.get_context(
                st.session_state.user_id,
                prompt,
                limit=st.session_state.context_limit
            )
            if context:
                context_text = "\n".join([f"- {m['memory']}" for m in context])
                messages.append(SystemMessage(content=f"📚 相关历史记忆:\n{context_text}"))

        # 添加最近的对话历史（不包括刚添加的用户消息，因为会在下面单独添加）
        recent_messages = st.session_state.messages[-11:-1] if len(st.session_state.messages) > 1 else []
        for msg in recent_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # 添加当前用户消息
        messages.append(HumanMessage(content=prompt))

        # 调用 LLM - 使用流式输出
        try:
            # 创建一个占位符用于流式显示
            message_placeholder = st.empty()
            full_response = ""

            # 使用 stream 方法进行流式输出
            for chunk in st.session_state.llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    # 添加打字机效果（光标）
                    message_placeholder.markdown(full_response + "▌")

            # 显示最终结果（移除光标）
            message_placeholder.markdown(full_response)

            # 保存助手回复到历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # 保存到记忆
            if use_memory:
                st.session_state.memory_manager.add_message(
                    st.session_state.user_id,
                    prompt,
                    "user"
                )
                st.session_state.memory_manager.add_message(
                    st.session_state.user_id,
                    full_response,
                    "assistant"
                )

        except Exception as e:
            error_msg = f"❌ 发生错误: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
