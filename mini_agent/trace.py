"""轨迹（trace）记录：Agent 可观测性的最小实现。

为什么必须有 trace？
* **调试**：Agent 出错时你看到的不是异常栈，而是一串"想错了 → 调错了 → 观察错了"。
* **评测**：有了结构化轨迹，才能算"工具调用准确率""平均步数""每步 token 成本"。
* **面试**：能讲清"我怎么定位线上 Agent 的问题"，比背框架 API 值钱得多。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Step", "AgentTrace"]


@dataclass
class Step:
    """Agent 的一步。"""

    index: int
    thought: str = ""
    action: str = ""
    action_input: Any = None
    observation: str = ""
    ok: bool = True
    error: str = ""
    raw_text: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    mode: str = ""  # "native" | "text" | "final"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class AgentTrace:
    """一次 Agent 运行的完整轨迹。"""

    question: str = ""
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    stopped_reason: str = ""     # "final_answer" | "max_steps" | "loop_detected" | "error"
    total_latency_ms: float = 0.0
    model: str = ""

    # ------------------------------------------------------------------ #

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action)

    @property
    def failed_calls(self) -> int:
        return sum(1 for s in self.steps if s.action and not s.ok)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "stopped_reason": self.stopped_reason,
            "model": self.model,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "failed_calls": self.failed_calls,
            "steps": [asdict(s) for s in self.steps],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def render(self) -> str:
        """人类可读的轨迹，用于终端打印或写日志。"""
        lines = [f"Q: {self.question}"]
        for s in self.steps:
            lines.append(f"  [{s.index}] ({s.mode}, {s.latency_ms:.0f}ms, {s.total_tokens} tok)")
            if s.thought:
                lines.append(f"      思考: {_one_line(s.thought)}")
            if s.action:
                lines.append(f"      动作: {s.action}({json.dumps(s.action_input, ensure_ascii=False)})")
                tag = "OK " if s.ok else "ERR"
                lines.append(f"      观察[{tag}]: {_one_line(s.observation or s.error)}")
        lines.append(f"  最终答案: {_one_line(self.final_answer)}")
        lines.append(
            f"  结束原因={self.stopped_reason} 步数={len(self.steps)} "
            f"工具调用={self.tool_calls}(失败 {self.failed_calls}) "
            f"token={self.total_tokens} 总耗时={self.total_latency_ms:.0f}ms"
        )
        return "\n".join(lines)


def _one_line(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
