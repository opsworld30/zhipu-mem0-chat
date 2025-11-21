from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import StateGraph, END
import json
import config


class IntentState(TypedDict):
    message: str
    message_type: Literal["question", "statement", "command", "personal_info"]
    retrieve_needed: bool
    store_needed: bool
    confidence: float
    reasoning: str


class IntentAnalyzer:
    def __init__(self):
        self.llm = ChatZhipuAI(
            model=config.LLM_MODEL,
            api_key=config.ZHIPU_API_KEY,
            temperature=0,
        )
        self.graph = self._build_graph()
    
    def _classify_message_type(self, state: IntentState) -> IntentState:
        prompt = [
            SystemMessage(content="""分析用户消息的类型。输出JSON格式：
{
    "type": "question|statement|command|personal_info",
    "confidence": 0.0-1.0
}

类型定义：
- question: 询问、疑问句
- statement: 陈述、观点表达
- command: 指令、要求
- personal_info: 个人信息分享"""),
            HumanMessage(content=state["message"])
        ]
        
        try:
            result = self.llm.invoke(prompt)
            data = json.loads(result.content.strip())
            state["message_type"] = data.get("type", "statement")
            state["confidence"] = data.get("confidence", 0.5)
        except Exception:
            state["message_type"] = "statement"
            state["confidence"] = 0.5
        
        return state
    
    def _analyze_retrieve_intent(self, state: IntentState) -> IntentState:
        if state["message_type"] == "question":
            prompt = [
                SystemMessage(content="""判断这个问题是否需要历史上下文。输出JSON：
{
    "retrieve": bool,
    "reasoning": "判断理由"
}

需要检索的情况：
- 使用指代词（那个、这个、它等）
- 提到时间延续（继续、接着、上次等）
- 询问历史信息"""),
                HumanMessage(content=state["message"])
            ]
        else:
            state["retrieve_needed"] = False
            state["reasoning"] = f"消息类型为{state['message_type']}，通常不需要检索历史"
            return state
        
        try:
            result = self.llm.invoke(prompt)
            data = json.loads(result.content.strip())
            state["retrieve_needed"] = data.get("retrieve", False)
            state["reasoning"] = data.get("reasoning", "")
        except Exception:
            state["retrieve_needed"] = False
            state["reasoning"] = "解析失败，默认不检索"
        
        return state
    
    def _analyze_store_intent(self, state: IntentState) -> IntentState:
        if state["message_type"] == "personal_info":
            state["store_needed"] = True
            state["reasoning"] += " | 个人信息需要存储"
        elif state["message_type"] == "statement":
            prompt = [
                SystemMessage(content="""判断这个陈述是否值得存储。输出JSON：
{
    "store": bool,
    "reasoning": "判断理由"
}

值得存储的情况：
- 表达偏好或观点
- 重要决定或计划
- 有参考价值的信息"""),
                HumanMessage(content=state["message"])
            ]
            
            try:
                result = self.llm.invoke(prompt)
                data = json.loads(result.content.strip())
                state["store_needed"] = data.get("store", True)
                state["reasoning"] += f" | {data.get('reasoning', '')}"
            except Exception:
                state["store_needed"] = True
                state["reasoning"] += " | 默认存储陈述"
        else:
            state["store_needed"] = False
            state["reasoning"] += f" | {state['message_type']}类型通常不需要存储"
        
        return state
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(IntentState)
        
        workflow.add_node("classify", self._classify_message_type)
        workflow.add_node("analyze_retrieve", self._analyze_retrieve_intent)
        workflow.add_node("analyze_store", self._analyze_store_intent)
        
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "analyze_retrieve")
        workflow.add_edge("analyze_retrieve", "analyze_store")
        workflow.add_edge("analyze_store", END)
        
        return workflow.compile()
    
    def analyze(self, message: str) -> tuple[bool, bool]:
        initial_state = {
            "message": message,
            "message_type": "statement",
            "retrieve_needed": False,
            "store_needed": False,
            "confidence": 0.0,
            "reasoning": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result["retrieve_needed"], result["store_needed"]
    
    def analyze_with_details(self, message: str) -> IntentState:
        initial_state = {
            "message": message,
            "message_type": "statement",
            "retrieve_needed": False,
            "store_needed": False,
            "confidence": 0.0,
            "reasoning": ""
        }
        
        return self.graph.invoke(initial_state)
