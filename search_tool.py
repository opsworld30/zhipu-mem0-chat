from langchain_mcp import MCPToolkit
from typing import Optional, List

class SearchTool:
    def __init__(self):
        self.enabled = False
        self.toolkit = None
        
    def initialize(self):
        try:
            self.toolkit = MCPToolkit(
                server_params={
                    "command": "uv",
                    "args": ["--directory", "/Users/mac/code/python/searxng-mcp", "run", "server.py"],
                    "env": {
                        "SEARXNG_BASE_URL": "http://127.0.0.1:8888",
                        "REQUEST_TIMEOUT": "30.0",
                        "MAX_RESULTS": "20"
                    }
                }
            )
            return True
        except Exception as e:
            print(f"初始化搜索工具失败: {e}")
            return False
    
    def get_tools(self):
        if not self.enabled or not self.toolkit:
            return []
        return self.toolkit.get_tools()
    
    def format_search_context(self, results: str) -> str:
        if not results:
            return ""
        return f"\n\n🔍 网络搜索结果:\n{results}\n"
