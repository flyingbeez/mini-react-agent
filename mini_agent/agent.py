"""ReAct Agent 核心循环。

ReAct = Reasoning + Acting（Yao et al., 2022）。一句话概括：

    想一步 → 做一个动作（调工具）→ 看结果 → 再想一步 …… 直到能回答

这个文件把循环写透，并处理了教科书里不写、但工程上必须处理的三件事：
1. **步数上限**：模型可能永远不停，必须兜底。
2. **死循环检测**：连续用同样参数调同一个工具，说明卡住了，要主动打断并提示模型换策略。
3. **工具失败不崩溃**：错误变成 observation 回灌给模型，让它自己纠错（self-healing）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .llm import BaseLLM, LLMError, LLMResponse
from .parser import ParsedAction, parse_react_text
from .tools import ToolError, ToolRegistry
from .trace import AgentTrace, Step

__all__ = ["ReActAgent", "AgentResult", "Step"]

SYSTEM_PROMPT_NATIVE = """你是一个严谨的 AI Agent。你可以调用工具来获取信息或完成计算。

工作原则：
1. 需要外部信息或精确计算时，必须调用工具，不要凭空猜测。
2. 一次可以调用多个工具；能并行就并行。
3. 工具返回报错时，读懂错误信息并修正参数后重试，不要原样重复调用。
4. 不要重复调用同一个工具并传入完全相同的参数——那说明你卡住了，应该换思路。
5. 当信息足够回答用户时，直接给出最终答案，不要再调用工具。
6. 最终答案必须基于工具返回的事实；如果确实是工具无法获取的信息，明确说明"无法确认"。

请用简体中文回答。"""

SYSTEM_PROMPT_TEXT = """你是一个严谨的 AI Agent，遵循 ReAct 范式工作。

你可以使用以下工具：
{tools}

严格按下面格式输出，每次只输出一个步骤：

Thought: 你的推理过程（一句话到三句话）
Action: 要调用的工具名（必须是上面列表中的名字）
Action Input: 调用参数的 JSON 对象，例如 {{"a": 1, "b": 2}}

收到 Observation 后继续下一轮 Thought/Action。当你已经能回答用户时，输出：

Thought: 我已经有足够信息了
Final Answer: 你的最终回答（简体中文）

规则：
- 需要外部信息或精确计算时必须调用工具，不要凭空猜测。
- 不要输出 Observation，那是系统给你的。
- 不要连续用完全相同的参数调用同一个工具，那说明你卡住了。
- Action Input 必须是合法 JSON，不要加注释或多余文字。"""

_NUDGE_TEMPLATE = (
    "注意：你刚才用相同的参数重复调用了工具 `{tool}`，收到的结果也一样。"
    "重复调用不会带来新信息。请换一种做法，或者直接给出最终答案。"
)


@dataclass
class AgentResult:
    """Agent 运行结果。"""

    answer: str
    trace: AgentTrace
    stopped_reason: str = "final_answer"
    error: str = ""

    @property
    def success(self) -> bool:
        return self.stopped_reason == "final_answer" and not self.error

    @property
    def steps(self) -> list[Step]:
        return self.trace.steps

    def __str__(self) -> str:  # pragma: no cover - 方便交互式使用
        return self.answer


@dataclass
class _LoopState:
    """循环过程中需要跨步维护的状态。"""

    repeat_counts: dict[str, int] = field(default_factory=dict)


class ReActAgent:
    """ReAct 智能体。

    Args:
        llm: 模型客户端（BaseLLM 子类）。
        tools: 工具注册表，或可迭代的函数/Tool 集合。
        mode: ``"native"`` 用模型的 function calling；``"text"`` 用文本 ReAct 格式。
        max_steps: 最多几轮模型调用（防止无限循环）。
        loop_threshold: 同一工具+同参数的调用达到几次开始警告。
        repeat_stop_at: 达到几次直接终止并给出诊断。
        verbose: 是否把每一步打印到 stderr。
        timeout_s: 整次运行的墙钟超时（秒），None 表示不限制。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry | Sequence[Any],
        *,
        mode: str = "native",
        max_steps: int = 8,
        loop_threshold: int = 2,
        repeat_stop_at: int = 4,
        verbose: bool = False,
        timeout_s: float | None = None,
        system_prompt: str | None = None,
        max_observation_chars: int = 4000,
    ) -> None:
        if mode not in ("native", "text"):
            raise ValueError("mode 只能是 'native' 或 'text'")
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else _coerce_registry(tools)
        self.mode = mode
        self.max_steps = max_steps
        self.loop_threshold = loop_threshold
        self.repeat_stop_at = repeat_stop_at
        self.verbose = verbose
        self.timeout_s = timeout_s
        self.max_observation_chars = max_observation_chars
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #

    def run(self, question: str, history: Sequence[dict] | None = None) -> AgentResult:
        """执行一次完整的 ReAct 循环。"""
        trace = AgentTrace(question=question, model=getattr(self.llm, "model", ""))
        started = time.time()
        state = _LoopState()

        try:
            answer = self._loop(question, history or [], trace, state, started)
        except LLMError as exc:
            trace.stopped_reason = "error"
            trace.total_latency_ms = (time.time() - started) * 1000
            return AgentResult(answer=f"模型调用失败：{exc}", trace=trace, stopped_reason="error", error=str(exc))
        except KeyboardInterrupt:  # pragma: no cover
            raise
        except Exception as exc:  # noqa: BLE001 - 兜底，保证任何异常都有轨迹可查
            trace.stopped_reason = "error"
            trace.total_latency_ms = (time.time() - started) * 1000
            return AgentResult(
                answer=f"Agent 内部错误：{type(exc).__name__}: {exc}",
                trace=trace,
                stopped_reason="error",
                error=str(exc),
            )

        trace.final_answer = answer
        trace.total_latency_ms = (time.time() - started) * 1000
        return AgentResult(answer=answer, trace=trace, stopped_reason=trace.stopped_reason)

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #

    def _loop(self, question: str, history: Sequence[dict], trace: AgentTrace,
              state: _LoopState, started: float) -> str:
        if self.mode == "native":
            messages: list[dict] = [{"role": "system", "content": self._native_system()}]
            messages.extend(dict(m) for m in history)
            messages.append({"role": "user", "content": question})
            return self._loop_native(messages, trace, state, started)

        messages = [{"role": "system", "content": self._text_system()}]
        messages.extend(dict(m) for m in history)
        messages.append({"role": "user", "content": question})
        return self._loop_text(messages, trace, state, started)

    # -- native function calling ---------------------------------------- #

    def _loop_native(self, messages: list[dict], trace: AgentTrace,
                     state: _LoopState, started: float) -> str:
        for _turn in range(self.max_steps):
            self._check_timeout(started, trace)

            t0 = time.time()
            response = self.llm.chat(messages, tools=self.tools.specs())
            latency = (time.time() - t0) * 1000

            if not response.tool_calls:
                # 收尾也要记一步：否则最后一轮的 token 会从成本统计里漏掉
                trace.add(Step(
                    index=len(trace.steps) + 1,
                    thought="",
                    latency_ms=latency,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    raw_text=response.content or "",
                    mode="final",
                ))
                trace.stopped_reason = "final_answer"
                return (response.content or "").strip() or "（模型没有返回内容）"

            messages.append(response.assistant_message())

            for tc in response.tool_calls:
                step = Step(
                    index=len(trace.steps) + 1,
                    thought=(response.content or "").strip(),
                    action=tc.name,
                    action_input=tc.args() if tc.is_valid_json() else tc.arguments,
                    latency_ms=latency,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    raw_text=response.content or "",
                    mode="native",
                )

                signature = _signature(tc.name, tc.args())
                if not tc.is_valid_json():
                    # arguments 不是合法 JSON —— 直接把错误交给模型
                    observation = (
                        f"参数解析失败：{tc.arguments!r} 不是合法 JSON 对象。"
                        "请重新调用本工具，arguments 必须是 {\"key\": value} 形式的 JSON。"
                    )
                    step.ok = False
                    step.error = observation
                else:
                    observation = self._execute(step, tc.name, tc.args())

                step.observation = observation
                trace.add(step)
                self._log(step)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": observation,
                })

                verdict = self._check_repeat(state, signature, tc.name)
                if verdict == "stop":
                    trace.stopped_reason = "loop_detected"
                    return (
                        f"检测到死循环：`{tc.name}` 被反复以相同参数调用，已强制停止。"
                        f"目前掌握的信息：{_summarize_observations(trace)}"
                    )
                if verdict == "nudge":
                    messages.append({"role": "user", "content": _NUDGE_TEMPLATE.format(tool=tc.name)})

        trace.stopped_reason = "max_steps"
        return (
            f"达到最大步数 {self.max_steps}，任务仍未完成。"
            f"已获得的信息：{_summarize_observations(trace)}"
        )

    # -- 文本 ReAct ------------------------------------------------------ #

    def _loop_text(self, messages: list[dict], trace: AgentTrace,
                   state: _LoopState, started: float) -> str:
        for _turn in range(self.max_steps):
            self._check_timeout(started, trace)

            t0 = time.time()
            response = self.llm.chat(messages)
            latency = (time.time() - t0) * 1000
            text = response.content or ""

            parsed: ParsedAction = parse_react_text(text)

            if parsed.is_final:
                trace.add(Step(
                    index=len(trace.steps) + 1,
                    thought=parsed.thought,
                    latency_ms=latency,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    raw_text=text,
                    mode="final",
                    observation="",
                ))
                trace.stopped_reason = "final_answer"
                return parsed.final

            if not parsed.is_action:
                # 模型没按格式说话：把错误回灌，让它自我修正
                step = Step(
                    index=len(trace.steps) + 1,
                    thought=parsed.thought,
                    latency_ms=latency,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    raw_text=text,
                    mode="text",
                    ok=False,
                    error="输出格式无法解析",
                    observation="输出格式无法解析",
                )
                trace.add(step)
                self._log(step)
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        "你的输出不符合 ReAct 格式。必须包含 `Action: 工具名` 与 "
                        "`Action Input: {\"key\": value}`，或者以 `Final Answer: ...` 开头给出答案。"
                    ),
                })
                continue

            step = Step(
                index=len(trace.steps) + 1,
                thought=parsed.thought,
                action=parsed.action,
                action_input=parsed.action_input if parsed.action_input is not None else parsed.raw_input,
                latency_ms=latency,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                raw_text=text,
                mode="text",
            )

            args = self._prepare_args(step, parsed)
            if args is None:
                observation = step.error
            else:
                observation = self._execute(step, parsed.action, args)

            step.observation = observation
            trace.add(step)
            self._log(step)

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

            signature = _signature(parsed.action, args if args is not None else parsed.raw_input)
            verdict = self._check_repeat(state, signature, parsed.action)
            if verdict == "stop":
                trace.stopped_reason = "loop_detected"
                return (
                    f"检测到死循环：`{parsed.action}` 被反复以相同参数调用，已强制停止。"
                    f"目前掌握的信息：{_summarize_observations(trace)}"
                )
            if verdict == "nudge":
                messages.append({"role": "user", "content": _NUDGE_TEMPLATE.format(tool=parsed.action)})

        trace.stopped_reason = "max_steps"
        return (
            f"达到最大步数 {self.max_steps}，任务仍未完成。"
            f"已获得的信息：{_summarize_observations(trace)}"
        )

    # ------------------------------------------------------------------ #
    # 工具执行与辅助逻辑
    # ------------------------------------------------------------------ #

    def _execute(self, step: Step, name: str, args: Any) -> str:
        """执行工具；任何失败都转成 observation 文本。"""
        try:
            result = self.tools.call(name, args)
            step.ok = True
            return self._truncate(result)
        except ToolError as exc:
            step.ok = False
            step.error = str(exc)
            return f"[工具错误] {exc}"
        except Exception as exc:  # noqa: BLE001
            step.ok = False
            step.error = f"{type(exc).__name__}: {exc}"
            return f"[工具错误] {type(exc).__name__}: {exc}"

    def _prepare_args(self, step: Step, parsed: ParsedAction) -> dict | None:
        """文本模式下的参数兜底：JSON 解析失败时尝试单参数填充。"""
        if parsed.action_input is not None:
            return parsed.action_input

        raw = (parsed.raw_input or "").strip()
        if parsed.action not in self.tools:
            step.ok = False
            step.error = f"不存在工具 {parsed.action!r}。可用：{self.tools.names()}"
            return None

        tool = self.tools.tools[parsed.action]
        required = tool.parameters.get("required", [])
        props = tool.parameters.get("properties", {})
        if len(props) == 1 and raw:
            only = next(iter(props))
            return {only: raw}
        if not required and not raw:
            return {}
        if not required and raw:
            return {next(iter(props)): raw} if len(props) == 1 else {}

        step.ok = False
        step.error = (
            f"Action Input 不是合法 JSON：{raw!r}。"
            f"请输出 JSON 对象，例如 {{\"{required[0] if required else 'input'}\": ...}}"
        )
        return None

    def _check_repeat(self, state: _LoopState, signature: str, tool_name: str) -> str:
        """返回 "" / "nudge" / "stop"。"""
        if not signature:
            return ""
        state.repeat_counts[signature] = state.repeat_counts.get(signature, 0) + 1
        count = state.repeat_counts[signature]

        if count >= self.repeat_stop_at:
            return "stop"
        if count >= self.loop_threshold:
            return "nudge"
        return ""

    def _check_timeout(self, started: float, trace: AgentTrace) -> None:
        if self.timeout_s is not None and (time.time() - started) > self.timeout_s:
            trace.stopped_reason = "timeout"
            raise LLMError(f"运行超时（>{self.timeout_s}s）")

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_observation_chars:
            return text
        return text[: self.max_observation_chars] + f"\n…（结果过长，已截断至 {self.max_observation_chars} 字符）"

    def _native_system(self) -> str:
        return self._system_prompt or SYSTEM_PROMPT_NATIVE

    def _text_system(self) -> str:
        if self._system_prompt:
            return self._system_prompt.replace("{tools}", self.tools.describe())
        return SYSTEM_PROMPT_TEXT.format(tools=self.tools.describe())

    def _log(self, step: Step) -> None:
        if not self.verbose:
            return
        import sys

        flag = "OK " if step.ok else "ERR"
        print(
            f"[step {step.index}] {step.action or '(final)'} -> {flag} "
            f"{_flat(step.observation, 120)}",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------- #
# 模块级小工具
# --------------------------------------------------------------------------- #

def _coerce_registry(tools: Sequence[Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for item in tools:
        registry.register(item)
    return registry


def _signature(name: str, args: Any) -> str:
    if args is None:
        return f"{name}()"
    try:
        return f"{name}({json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)})"
    except (TypeError, ValueError):
        return f"{name}({args!r})"


def _summarize_observations(trace: AgentTrace) -> str:
    items = [s.observation for s in trace.steps if s.ok and s.observation]
    if not items:
        return "（没有成功获得任何工具结果）"
    return " | ".join(_flat(i, 120) for i in items[-3:])


def _flat(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
