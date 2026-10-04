"""开箱即用的示例工具。

这四个工具刚好覆盖 Agent 工具设计的四种典型形态：
    calculator        —— 纯计算，必须严格校验（不能 eval 用户字符串）
    now               —— 无参工具，幂等
    text_stats        —— 结构化输出（dict → JSON）
    search_notes      —— 轻量检索，RAG 的雏形（词频打分 + 引用来源）

把它们换掉，就是你的业务工具。工具本身不依赖任何第三方库。
"""

from __future__ import annotations

import ast
import datetime as _dt
import math
import operator
import re
from typing import Any

from .tools import ToolError, tool

__all__ = ["calculator", "now", "text_stats", "search_notes", "load_notes", "NOTES"]

# --------------------------------------------------------------------------- #
# 1) 安全计算器
# --------------------------------------------------------------------------- #

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

_SAFE_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"不支持的常量类型：{type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的运算符：{type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("指数过大（>100），已拒绝计算")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ToolError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        raise ToolError(f"未知标识符 {node.id!r}。可用常量：{sorted(_SAFE_CONSTS)}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
            raise ToolError(
                f"不允许调用该函数。可用函数：{sorted(_SAFE_FUNCS)}"
            )
        if node.keywords:
            raise ToolError("计算器不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        return _SAFE_FUNCS[node.func.id](*args)
    raise ToolError(f"表达式包含不被允许的语法：{type(node).__name__}")


@tool(description="计算一个数学表达式。只支持四则运算、幂、取模，以及 sqrt/log/exp/sin/cos/abs/round/min/max/sum 等安全函数与 pi/e 常量。")
def calculator(expression: str) -> str:
    """计算数学表达式，返回结果字符串。

    Args:
        expression: 数学表达式，例如 "(3+5)*2/4" 或 "sqrt(2)*100"
    """
    expr = (expression or "").strip()
    if not expr:
        raise ToolError("expression 不能为空")
    if len(expr) > 200:
        raise ToolError("表达式过长（>200 字符）")
    if "__" in expr:
        raise ToolError("表达式包含非法字符序列 '__'")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误：{exc.msg}") from exc

    try:
        value = _eval_node(tree)
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("除数不能为零") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"计算失败：{type(exc).__name__}: {exc}") from exc

    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ToolError("计算结果不是有限数")
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


# --------------------------------------------------------------------------- #
# 2) 当前时间
# --------------------------------------------------------------------------- #

@tool(description="获取当前的日期和时间（可指定时区偏移小时数，默认东八区 +8）。")
def now(timezone_offset_hours: float = 8) -> str:
    """返回指定时区的当前时间。

    Args:
        timezone_offset_hours: 相对 UTC 的小时偏移，例如北京为 8
    """
    if not -12 <= timezone_offset_hours <= 14:
        raise ToolError("timezone_offset_hours 必须在 -12 到 14 之间")
    tz = _dt.timezone(_dt.timedelta(hours=timezone_offset_hours))
    current = _dt.datetime.now(tz)
    return current.strftime("%Y-%m-%d %H:%M:%S %Z（周%w）")


# --------------------------------------------------------------------------- #
# 3) 文本统计
# --------------------------------------------------------------------------- #

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_']+")


@tool(description="统计一段文本的字数、词数、行数和高频词，返回 JSON。")
def text_stats(text: str, top_k: int = 5) -> dict:
    """统计文本信息。

    Args:
        text: 待统计的文本
        top_k: 返回前多少个高频词
    """
    if not isinstance(text, str):
        raise ToolError("text 必须是字符串")
    if not 1 <= top_k <= 50:
        raise ToolError("top_k 必须在 1 到 50 之间")

    words = [w.lower() for w in _WORD.findall(text)]
    cjk_chars = _CJK.findall(text)
    counter: dict[str, int] = {}
    for w in words:
        counter[w] = counter.get(w, 0) + 1

    top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return {
        "chars": len(text),
        "chars_no_space": len(re.sub(r"\s", "", text)),
        "cjk_chars": len(cjk_chars),
        "latin_words": len(words),
        "lines": text.count("\n") + 1 if text else 0,
        "top_words": [{"word": w, "count": c} for w, c in top],
    }


# --------------------------------------------------------------------------- #
# 4) 轻量笔记检索（RAG 的雏形，无向量库）
# --------------------------------------------------------------------------- #

NOTES: list[dict[str, str]] = [
    {
        "id": "react",
        "title": "ReAct 范式",
        "text": (
            "ReAct 由 Reasoning 与 Acting 组合而成，2022 年由 Yao 等人提出。"
            "核心循环是 Thought 思考、Action 调用工具、Observation 观察结果，"
            "如此往复直到能给出 Final Answer。它把大模型的推理能力和外部工具的能力结合起来，"
            "解决了纯推理容易幻觉、纯工具调用不会规划的问题。工程上需要加最大步数限制和死循环检测。"
        ),
    },
    {
        "id": "function-calling",
        "title": "Function Calling / Tool Use",
        "text": (
            "Function Calling 是模型原生的结构化工具调用能力。你向模型提供一组工具声明，"
            "每个声明包含 name、description 和 JSON Schema 格式的 parameters；"
            "模型不直接执行函数，而是返回 tool_calls 字段，由你的代码去执行，"
            "再把结果以 role=tool 的消息回传给模型。相比文本 ReAct，它解析更稳定、支持并行调用。"
        ),
    },
    {
        "id": "rag",
        "title": "RAG 检索增强生成",
        "text": (
            "RAG 的流程是：文档切分成 chunk，用 embedding 模型编码成向量存入向量库，"
            "用户提问时把问题也编码成向量并做相似度检索，取回 top-k 片段拼进提示词，"
            "让模型基于这些片段回答并给出引用。它能缓解幻觉、支持私有知识、且不用重新训练模型。"
            "常见优化点包括分块策略、混合检索（向量加关键词）、重排序 rerank 和引用可追溯。"
        ),
    },
    {
        "id": "memory",
        "title": "Agent 记忆",
        "text": (
            "Agent 记忆分短期和长期。短期记忆就是当前对话的上下文窗口，受 token 上限约束；"
            "常见处理方式有滑动窗口、对话摘要压缩、关键信息抽取。长期记忆通常用向量库存储，"
            "按需检索相关历史。设计时要考虑写入策略、遗忘策略和一致性。"
        ),
    },
    {
        "id": "planning",
        "title": "规划与任务分解",
        "text": (
            "Plan-and-Execute 让模型先把任务拆成子任务列表，再逐个执行，"
            "执行中根据结果重新规划 Replan。相比纯 ReAct 逐步试错，它更适合长链条任务，"
            "但需要额外的重规划机制来应对子任务失败。"
        ),
    },
    {
        "id": "mcp",
        "title": "MCP 模型上下文协议",
        "text": (
            "MCP 是 Anthropic 在 2024 年底提出的开放协议，用统一方式把工具、资源和提示词"
            "暴露给任意支持 MCP 的客户端。它把工具提供方和使用方解耦，"
            "核心概念包括 Server、Client、Tools、Resources、Prompts 以及 stdio/SSE 两种传输方式。"
        ),
    },
    {
        "id": "evaluation",
        "title": "Agent 评测",
        "text": (
            "Agent 评测比单轮问答难，因为要评的是轨迹而不是单个答案。"
            "常用指标：任务成功率、工具调用准确率与参数正确率、平均步数、无效调用率、"
            "以及每次任务的平均 token 成本。方法上有规则断言、LLM-as-Judge 以及人工抽检。"
        ),
    },
    {
        "id": "prompt-engineering",
        "title": "Prompt Engineering 要点",
        "text": (
            "提示词工程的核心是控制输出空间：明确角色、给出格式约束、提供少量示例 few-shot、"
            "要求模型先思考再回答。对 Agent 而言，最关键的是让模型稳定输出结构化动作，"
            "并在出错时给出可自我纠正的错误信息。"
        ),
    },
]


def load_notes() -> list[dict[str, str]]:
    """返回内置笔记（可替换成你自己的知识库）。"""
    return NOTES


def _tokenize(text: str) -> list[str]:
    """中英混合的极简分词：英文按词，中文按单字 + 双字组合。"""
    text = text.lower()
    tokens = _WORD.findall(text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@tool(description="在内置的 Agent 知识笔记中做关键词检索，返回最相关的笔记片段及来源 id。适合回答 Agent / RAG / MCP / 评测等概念问题。")
def search_notes(query: str, top_k: int = 2) -> dict:
    """检索内置笔记。

    Args:
        query: 查询关键词或问题
        top_k: 返回几条笔记
    """
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query 不能为空")
    if not 1 <= top_k <= 8:
        raise ToolError("top_k 必须在 1 到 8 之间")

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        raise ToolError("query 中没有可检索的内容")

    scored: list[tuple[float, dict[str, str]]] = []
    for note in NOTES:
        doc_tokens = _tokenize(note["title"] + " " + note["text"])
        if not doc_tokens:
            continue
        doc_set = set(doc_tokens)
        overlap = q_tokens & doc_set
        if not overlap:
            continue
        # 用命中词长度做权重，降低"的/是/在"这类单字噪声的影响
        score = sum(2.0 if len(tok) > 1 else 1.0 for tok in overlap) / math.sqrt(len(doc_set))
        # 标题命中加权
        title_tokens = set(_tokenize(note["title"]))
        score += 1.5 * len(q_tokens & title_tokens)
        scored.append((score, note))

    if not scored:
        return {"query": query, "hits": [], "note": "没有检索到相关笔记，可直接回答或说明无法确认。"}

    scored.sort(key=lambda kv: (-kv[0], kv[1]["id"]))
    hits = [
        {"id": n["id"], "title": n["title"], "score": round(s, 3), "text": n["text"]}
        for s, n in scored[:top_k]
    ]
    return {"query": query, "hits": hits}
