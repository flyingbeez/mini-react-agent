"""mini_agent —— 从零实现的 ReAct Agent 最小内核（零第三方依赖）。

模块地图：
    llm.py            LLM 客户端抽象（OpenAI 兼容 / 离线 Mock）
    tools.py          工具抽象：@tool 装饰器 + 自动 JSON Schema + 参数校验
    parser.py         解析模型输出：原生 tool_calls 与文本 ReAct 两种模式
    trace.py          轨迹记录（每一步的思考/动作/观察/耗时/token）
    agent.py          核心：ReAct 循环 + 防死循环 + 步数上限
    builtin_tools.py  开箱即用的示例工具
"""

from .agent import AgentResult, ReActAgent, Step
from .llm import LLMResponse, MockLLM, OpenAICompatLLM, ToolCall
from .tools import Tool, ToolError, ToolRegistry, tool
from .trace import AgentTrace

__version__ = "0.1.0"

__all__ = [
    "ReActAgent",
    "AgentResult",
    "Step",
    "LLMResponse",
    "MockLLM",
    "OpenAICompatLLM",
    "ToolCall",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "tool",
    "AgentTrace",
    "__version__",
]
