from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import config
from memory_manager import MemoryManager
from intent_analyzer import IntentAnalyzer

app = FastAPI(title="智谱AI对话API")

llm = ChatZhipuAI(
    model=config.LLM_MODEL,
    api_key=config.ZHIPU_API_KEY,
    temperature=config.LLM_TEMPERATURE,
)

memory_manager = MemoryManager()
intent_analyzer = IntentAnalyzer()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    use_memory: bool = True

class ChatResponse(BaseModel):
    response: str
    user_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [SystemMessage(content="你是一个智能助手")]

        if request.use_memory:
            retrieve_needed, store_needed = intent_analyzer.analyze(request.message)
            if retrieve_needed:
                context = memory_manager.get_context(
                    request.user_id,
                    request.message
                )
                if context:
                    context_text = "\n".join([m["memory"] for m in context])
                    messages.append(SystemMessage(content=f"历史上下文:\n{context_text}"))
        else:
            store_needed = False

        messages.append(HumanMessage(content=request.message))

        response = llm.invoke(messages)
        response_text = response.content

        if request.use_memory and store_needed:
            memory_manager.add_message(
                request.user_id,
                request.message,
                "user"
            )
            memory_manager.add_message(
                request.user_id,
                response_text,
                "assistant"
            )
        
        return ChatResponse(
            response=response_text,
            user_id=request.user_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    try:
        memories = memory_manager.get_all_memories(user_id)
        return {"user_id": user_id, "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
