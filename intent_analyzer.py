from typing import TypedDict, Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import StateGraph, END
import json
import logging
import re
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentState(TypedDict):
    message: str
    message_type: Literal["question", "statement", "command", "personal_info"]
    retrieve_needed: bool
    store_needed: bool
    confidence: float
    reasoning: str
    retrieve_reasoning: Optional[str]
    store_reasoning: Optional[str]
    error: Optional[str]


class IntentAnalyzer:
    def __init__(self):
        self.llm = ChatZhipuAI(
            model=config.LLM_MODEL,
            api_key=config.ZHIPU_API_KEY,
            temperature=0,
        )
        self.graph = self._build_graph()
    
    def _validate_message(self, message: str) -> bool:
        """验证消息是否有效"""
        if not message or not message.strip():
            return False
        
        # 检查是否只包含特殊字符
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff]', '', message.strip())
        return len(cleaned) > 0
    
    def _classify_message_type(self, state: IntentState) -> IntentState:
        """分类消息类型"""
        try:
            if not self._validate_message(state["message"]):
                state["error"] = "消息内容无效或为空"
                state["message_type"] = "statement"
                state["confidence"] = 0.0
                return state
            
            prompt = [
                SystemMessage(content="""分析用户消息的类型。输出JSON格式：
{
    "type": "question|statement|command|personal_info",
    "confidence": 0.0-1.0,
    "reasoning": "分类理由"
}

类型定义：
- question: 询问、疑问句（包含？、吗、呢、如何、为什么等疑问词）
- statement: 陈述、观点表达（描述事实、表达看法、分享经历）
- command: 指令、要求（请、帮我、能否、需要等请求词）
- personal_info: 个人信息分享（姓名、年龄、职业、喜好等个人特征）

请根据消息内容准确判断类型，confidence应该反映判断的确定性。"""),
                HumanMessage(content=state["message"])
            ]
            
            result = self.llm.invoke(prompt)
            content = result.content.strip()
            
            # 提取JSON内容
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                state["message_type"] = data.get("type", "statement")
                state["confidence"] = max(0.0, min(1.0, data.get("confidence", 0.5)))
                state["reasoning"] = data.get("reasoning", "")
            else:
                raise ValueError("无法解析JSON响应")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            state["error"] = f"JSON解析失败: {str(e)}"
            state["message_type"] = "statement"
            state["confidence"] = 0.3
            state["reasoning"] = "解析失败，使用默认类型"
        except Exception as e:
            logger.error(f"消息分类错误: {e}")
            state["error"] = f"分类失败: {str(e)}"
            state["message_type"] = "statement"
            state["confidence"] = 0.3
            state["reasoning"] = "处理失败，使用默认类型"
        
        return state
    
    def _analyze_retrieve_intent(self, state: IntentState) -> IntentState:
        """分析是否需要检索历史信息"""
        try:
            if state["message_type"] != "question":
                state["retrieve_needed"] = False
                state["retrieve_reasoning"] = f"消息类型为{state['message_type']}，通常不需要检索历史"
                return state
            
            prompt = [
                SystemMessage(content="""判断这个问题是否需要历史上下文。输出JSON：
{
    "retrieve": boolean,
    "reasoning": "详细判断理由"
}

需要检索的情况：
1. 使用指代词（那个、这个、它、刚才、前面提到的等）
2. 提到时间延续（继续、接着、上次、之前、后来等）
3. 询问历史信息（还记得吗、我们之前聊过什么、你刚才说等）
4. 对话延续性表达（关于这个、那么、所以等）

不需要检索的情况：
1. 独立的常识性问题
2. 首次提到的全新话题
3. 具体的技术或事实查询
4. 没有任何上下文指向的表达"""),
                HumanMessage(content=state["message"])
            ]
            
            result = self.llm.invoke(prompt)
            content = result.content.strip()
            
            # 提取JSON内容
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                state["retrieve_needed"] = bool(data.get("retrieve", False))
                state["retrieve_reasoning"] = data.get("reasoning", "")
            else:
                raise ValueError("无法解析JSON响应")
                
        except json.JSONDecodeError as e:
            logger.error(f"检索意图JSON解析错误: {e}")
            state["retrieve_needed"] = False
            state["retrieve_reasoning"] = "解析失败，默认不检索"
            if not state.get("error"):
                state["error"] = f"检索意图解析失败: {str(e)}"
        except Exception as e:
            logger.error(f"检索意图分析错误: {e}")
            state["retrieve_needed"] = False
            state["retrieve_reasoning"] = "处理失败，默认不检索"
            if not state.get("error"):
                state["error"] = f"检索意图分析失败: {str(e)}"
        
        return state
    
    def _analyze_store_intent(self, state: IntentState) -> IntentState:
        """分析是否需要存储信息"""
        try:
            if state["message_type"] == "personal_info":
                state["store_needed"] = True
                state["store_reasoning"] = "个人信息需要存储以供后续参考"
            elif state["message_type"] == "statement":
                prompt = [
                    SystemMessage(content="""判断这个陈述是否值得存储。输出JSON：
{
    "store": boolean,
    "reasoning": "详细判断理由"
}

值得存储的情况：
1. 表达个人偏好或观点（喜欢、讨厌、认为、觉得等）
2. 重要决定或计划（打算、准备、决定、计划等）
3. 有参考价值的信息（经验、建议、重要事实等）
4. 情感表达或状态描述（感到、经历、状态等）
5. 用户提供的具体信息（联系方式、地址、时间等）

不值得存储的情况：
1. 简单的应答（好的、嗯、知道了等）
2. 临时性或无意义表达
3. 重复或冗余信息
4. 过于宽泛的描述"""),
                    HumanMessage(content=state["message"])
                ]
                
                result = self.llm.invoke(prompt)
                content = result.content.strip()
                
                # 提取JSON内容
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    state["store_needed"] = bool(data.get("store", True))
                    state["store_reasoning"] = data.get("reasoning", "")
                else:
                    raise ValueError("无法解析JSON响应")
            else:
                state["store_needed"] = False
                state["store_reasoning"] = f"{state['message_type']}类型通常不需要存储"
                
        except json.JSONDecodeError as e:
            logger.error(f"存储意图JSON解析错误: {e}")
            # 对于陈述类型，默认存储；其他类型默认不存储
            state["store_needed"] = (state["message_type"] == "statement")
            state["store_reasoning"] = "解析失败，使用默认策略"
            if not state.get("error"):
                state["error"] = f"存储意图解析失败: {str(e)}"
        except Exception as e:
            logger.error(f"存储意图分析错误: {e}")
            state["store_needed"] = (state["message_type"] == "statement")
            state["store_reasoning"] = "处理失败，使用默认策略"
            if not state.get("error"):
                state["error"] = f"存储意图分析失败: {str(e)}"
        
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
        """分析消息意图，返回是否需要检索和存储"""
        initial_state = {
            "message": message,
            "message_type": "statement",
            "retrieve_needed": False,
            "store_needed": False,
            "confidence": 0.0,
            "reasoning": "",
            "retrieve_reasoning": None,
            "store_reasoning": None,
            "error": None
        }
        
        try:
            result = self.graph.invoke(initial_state)
            logger.info(f"意图分析完成: 类型={result['message_type']}, 检索={result['retrieve_needed']}, 存储={result['store_needed']}")
            return result["retrieve_needed"], result["store_needed"]
        except Exception as e:
            logger.error(f"意图分析失败: {e}")
            return False, False
    
    def analyze_with_details(self, message: str) -> IntentState:
        """分析消息意图，返回详细结果"""
        initial_state = {
            "message": message,
            "message_type": "statement",
            "retrieve_needed": False,
            "store_needed": False,
            "confidence": 0.0,
            "reasoning": "",
            "retrieve_reasoning": None,
            "store_reasoning": None,
            "error": None
        }
        
        try:
            result = self.graph.invoke(initial_state)
            logger.info(f"详细意图分析完成: {result}")
            return result
        except Exception as e:
            logger.error(f"详细意图分析失败: {e}")
            initial_state["error"] = str(e)
            return initial_state
