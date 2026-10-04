"""解析模型输出。

Agent 工程里 80% 的 bug 出在"模型没按格式说话"。这个模块专门干脏活：

* **文本 ReAct 模式**：从自由文本里抽出 Thought / Action / Action Input / Final Answer。
  容忍 markdown 加粗、中英文冒号、代码块、大小写、多行 JSON。
* **原生 function calling 模式**：把 OpenAI 风格 tool_calls 归一化成 ToolCall。
* **容错降级**：模型的 Action Input 不是合法 JSON 时，不立刻失败，
  而是把它原样交给上一层；若目标工具只有一个参数，就自动填进去。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = ["ParsedAction", "parse_react_text", "parse_arguments", "strip_code_fence"]

_ACTION_RE = re.compile(
    r"(?:^|\n)[\s>*\-]*(?:Action|行动|动作|工具)\s*[*\s]*[:：]\s*[`\"']?([A-Za-z_][\w.\-]*)[`\"']?",
    re.IGNORECASE,
)
_INPUT_RE = re.compile(
    r"(?:Action\s*Input|行动输入|ActionInput|工具输入|输入)\s*[*\s]*[:：]\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
_THOUGHT_RE = re.compile(
    r"(?:Thought|思考|想法)\s*[*\s]*[:：]\s*(.*?)(?=\n\s*[*>\-]*(?:Action|行动|动作|Final|最终)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_RE = re.compile(
    r"(?:Final\s*Answer|最终答案|最终回答|最终结论)\s*[*\s]*[:：]\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)
# 出现这些标记说明 Action Input 已经结束
_CUT_RE = re.compile(
    r"\n\s*[*>\-]*(?:Observation|观察|Thought|思考|Action\b|行动|Final\s*Answer|最终答案)",
    re.IGNORECASE,
)


@dataclass
class ParsedAction:
    """一次模型输出的结构化结果。"""

    kind: str                                  # "final" | "action" | "unknown"
    thought: str = ""
    action: str = ""
    action_input: dict | None = None           # 解析成 dict 时为 dict，否则 None
    raw_input: str = ""                        # 原始字符串，供降级使用
    final: str = ""
    raw_text: str = ""

    @property
    def is_final(self) -> bool:
        return self.kind == "final"

    @property
    def is_action(self) -> bool:
        return self.kind == "action"


def strip_code_fence(text: str) -> str:
    """去掉模型爱加的代码装饰：```` ``` ```` 围栏，或包住整段的行内反引号。"""
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    if len(t) >= 2 and t.startswith("`") and t.endswith("`") and "\n" not in t:
        return t[1:-1].strip()
    return t


def parse_arguments(raw: Any) -> dict | None:
    """尽最大努力把模型给的参数变成 dict，失败返回 None。"""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = strip_code_fence(raw).strip()
    if not text:
        return {}

    # 1) 标准 JSON
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # 2) 单引号 / 裸键名修补（模型的常见手滑）
    repaired = re.sub(r"'([^'\"\n]*)'", r'"\1"', text)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_]\w*)(\s*:)", r'\1"\2"\3', repaired)
    try:
        value = json.loads(repaired)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # 3) key=value 形式，例如  a=1, b=2
    if "=" in text and ":" not in text.split("=")[0]:
        pairs: dict[str, Any] = {}
        ok = True
        for chunk in re.split(r"[,\n;]+", text):
            if not chunk.strip():
                continue
            if "=" not in chunk:
                ok = False
                break
            k, _, v = chunk.partition("=")
            k = k.strip().strip("\"'")
            v = v.strip().strip("\"'")
            if not k:
                ok = False
                break
            pairs[k] = _guess_scalar(v)
        if ok and pairs:
            return pairs

    return None


def _guess_scalar(text: str) -> Any:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def parse_react_text(text: str) -> ParsedAction:
    """解析一段 ReAct 文本输出。"""
    raw = text or ""
    cleaned = strip_code_fence(raw)

    # 模型有时直接吐一个 JSON 对象当作动作
    if cleaned.startswith("{"):
        obj = parse_arguments(cleaned)
        if isinstance(obj, dict):
            name = obj.get("action") or obj.get("tool") or obj.get("name")
            if isinstance(name, str) and name:
                return ParsedAction(
                    kind="action",
                    thought=str(obj.get("thought") or obj.get("思考") or ""),
                    action=name,
                    action_input=obj.get("action_input", obj.get("arguments", obj.get("input"))),
                    raw_input=json.dumps(obj.get("action_input", obj.get("arguments", "")), ensure_ascii=False),
                    raw_text=raw,
                )
            for key in ("final_answer", "answer", "final", "答案"):
                if key in obj and isinstance(obj[key], str):
                    return ParsedAction(kind="final", final=obj[key], raw_text=raw)

    final_match = _FINAL_RE.search(cleaned)
    thought_match = _THOUGHT_RE.search(cleaned)
    thought = thought_match.group(1).strip() if thought_match else ""

    if final_match and (not _ACTION_RE.search(cleaned) or
                        final_match.start() < (_ACTION_RE.search(cleaned).start() if _ACTION_RE.search(cleaned) else 10**9)):
        return ParsedAction(kind="final", thought=thought, final=final_match.group(1).strip(), raw_text=raw)

    action_match = _ACTION_RE.search(cleaned)
    if not action_match:
        if final_match:
            return ParsedAction(kind="final", thought=thought, final=final_match.group(1).strip(), raw_text=raw)
        return ParsedAction(kind="unknown", thought=thought, final=cleaned.strip(), raw_text=raw)

    action = action_match.group(1).strip()

    input_match = _INPUT_RE.search(cleaned, action_match.end())
    raw_input = ""
    if input_match:
        raw_input = input_match.group(1)
        cut = _CUT_RE.search(raw_input)
        if cut:
            raw_input = raw_input[: cut.start()]
        raw_input = strip_code_fence(raw_input).strip()

    parsed = parse_arguments(raw_input) if raw_input else {}
    return ParsedAction(
        kind="action",
        thought=thought,
        action=action,
        action_input=parsed,
        raw_input=raw_input,
        raw_text=raw,
    )
