from langchain_mcp import MCPToolkit
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import config
import asyncio
import nest_asyncio

# 允许嵌套事件循环（Streamlit 需要）
nest_asyncio.apply()

class SearchTool:
    def __init__(self):
        self.enabled = False
        self.toolkit = None
        self.session = None
        self._loop = None

    async def _create_session(self):
        """创建 MCP 会话"""
        try:
            server_params = StdioServerParameters(
                command="uv",
                args=["--directory", config.MCP_SERVER_PATH, "run", "server.py"],
                env={
                    "SEARXNG_BASE_URL": config.SEARXNG_BASE_URL,
                    "REQUEST_TIMEOUT": config.SEARXNG_TIMEOUT,
                    "MAX_RESULTS": config.SEARXNG_MAX_RESULTS
                }
            )

            stdio_transport = stdio_client(server_params)
            read, write = await stdio_transport.__aenter__()

            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            return session, stdio_transport
        except Exception as e:
            raise Exception(f"创建 MCP 会话失败: {e}")

    def initialize(self):
        """初始化搜索工具"""
        try:
            # 获取或创建事件循环
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

            # 创建会话
            self.session, self._stdio_transport = self._loop.run_until_complete(
                self._create_session()
            )

            # 创建工具包
            self.toolkit = MCPToolkit(session=self.session)
            self._loop.run_until_complete(self.toolkit.initialize())

            return True

        except Exception as e:
            print(f"初始化搜索工具失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_tools(self):
        """获取可用的搜索工具"""
        if not self.enabled or not self.toolkit:
            return []
        try:
            return self.toolkit.get_tools()
        except Exception as e:
            print(f"获取工具失败: {e}")
            return []

    def cleanup(self):
        """清理资源"""
        if self.session and self._loop:
            try:
                self._loop.run_until_complete(self.session.__aexit__(None, None, None))
                if hasattr(self, '_stdio_transport'):
                    self._loop.run_until_complete(
                        self._stdio_transport.__aexit__(None, None, None)
                    )
            except Exception as e:
                print(f"清理资源失败: {e}")
