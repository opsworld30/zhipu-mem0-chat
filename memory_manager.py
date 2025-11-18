from mem0 import Memory
import config
import os
from datetime import datetime
import json

class MemoryManager:
    """管理用户对话记忆的类，使用 mem0 和智谱 AI"""

    def __init__(self):
        # 设置环境变量让 mem0 使用智谱 API
        os.environ["OPENAI_API_KEY"] = config.ZHIPU_API_KEY
        os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4/"

        mem_config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "glm-4-flash",
                    "temperature": 0.2,
                    "max_tokens": 1500,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "embedding-3",
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "zhipu_conversations",
                    "path": "./chroma_db",
                }
            }
        }

        self.memory = Memory.from_config(mem_config)

    def add_message(self, user_id: str, message: str, role: str):
        """添加对话消息到记忆中"""
        self.memory.add(
            message,
            user_id=user_id,
            metadata={"role": role}
        )

    def get_context(self, user_id: str, query: str, limit: int = 5):
        """搜索相关的记忆上下文"""
        results = self.memory.search(
            query,
            user_id=user_id,
            limit=limit
        )
        # mem0 返回格式: {'results': [...]}
        return results.get('results', [])

    def get_all_memories(self, user_id: str):
        """获取用户的所有记忆"""
        all_memories = self.memory.get_all(user_id=user_id)
        # mem0 返回格式: {'results': [...]}
        return all_memories.get('results', [])

    def delete_memory(self, memory_id: str):
        """删除指定的记忆"""
        try:
            self.memory.delete(memory_id)
            return True
        except Exception as e:
            print(f"删除记忆失败: {e}")
            return False

    def delete_all_memories(self, user_id: str):
        """删除用户的所有记忆"""
        try:
            self.memory.delete_all(user_id=user_id)
            return True
        except Exception as e:
            print(f"删除所有记忆失败: {e}")
            return False

    def export_memories(self, user_id: str) -> str:
        """导出用户的所有记忆为 JSON 格式"""
        memories = self.get_all_memories(user_id)
        export_data = {
            "user_id": user_id,
            "export_time": datetime.now().isoformat(),
            "total_memories": len(memories),
            "memories": memories
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)
