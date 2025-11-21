import os
from dotenv import load_dotenv

load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

if not ZHIPU_API_KEY:
    raise ValueError("ZHIPU_API_KEY未设置")

LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")

MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "/Users/mac/code/python/searxng-mcp")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8888")
SEARXNG_TIMEOUT = os.getenv("SEARXNG_TIMEOUT", "30.0")
SEARXNG_MAX_RESULTS = os.getenv("SEARXNG_MAX_RESULTS", "20")
