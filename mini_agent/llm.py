"""LLM 客户端抽象层。

三个实现：
    BaseLLM          统一接口
    MockLLM          离线脚本化模型，让测试和 demo 不需要 API Key（这是可复现性的关键）
    OpenAICompatLLM  只用标准库 urllib 调用任何 OpenAI 兼容接口
                     （DeepSeek / 通义千问 / 智谱 / Moonshot / vLLM / Ollama / OpenAI）

为什么不用官方 SDK？—— 减少依赖，让你看清 HTTP 层到底发生了什么。
生产代码当然可以用 SDK，但面试时被问到"一次 function calling 的请求长什么样"要能答出来。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "ToolCall",
    "LLMResponse",
    "BaseLLM",
    "MockLLM",
    "OpenAICompatLLM",
    "LLMError",
]


class LLMError(RuntimeError):
    """模型调用失败（网络、鉴权、响应格式）。"""


@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""

    name: str
    arguments: str = "{}"      # 原始字符串，模型的输出不可信，先留着
    id: str = ""
    parsed: dict | None = None  # 解析成功后的字典

    def args(self) -> dict:
        """尽量把 arguments 解析成 dict；失败返回空 dict（由工具层报错给模型）。"""
        if self.parsed is not None:
            return self.parsed
        if not self.arguments or not self.arguments.strip():
            return {}
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def is_valid_json(self) -> bool:
        """arguments 是否是合法的 JSON 对象（用来区分 '{}' 和 '{a: 1'）。"""
        try:
            value = json.loads(self.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(value, dict)

    @classmethod
    def from_openai(cls, raw: dict, index: int = 0) -> "ToolCall":
        fn = raw.get("function") or {}
        return cls(
            name=fn.get("name", ""),
            arguments=fn.get("arguments", "") or "",
            id=raw.get("id") or f"call_{index}",
        )


@dataclass
class LLMResponse:
    """一次模型调用的结果。"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def assistant_message(self) -> dict:
        """转成可回传给模型的 assistant 消息（区分文本模式 / 原生模式）。"""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                }
                for tc in self.tool_calls
            ]
        return msg


class BaseLLM:
    """模型客户端接口：只要实现 chat() 就能接入 Agent。"""

    model: str = "base"

    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    # 方便子类复用
    @staticmethod
    def _validate(messages: Sequence[dict]) -> list[dict]:
        if not messages:
            raise LLMError("messages 不能为空")
        return [dict(m) for m in messages]


class MockLLM(BaseLLM):
    """离线脚本化模型：按顺序吐出预设响应，或按规则函数动态响应。

    用法一（固定脚本，适合测试 Agent 循环）::

        llm = MockLLM([LLMResponse(tool_calls=[ToolCall("add", '{"a":1,"b":2}')]),
                       LLMResponse(content="答案是 3")])

    用法二（规则函数，适合 demo）::

        def brain(messages, tools):
            if "Observation:" not in messages[-1]["content"]:
                return LLMResponse(content="Thought: 我需要算数\\nAction: add\\nAction Input: {\"a\":1,\"b\":2}")
            return LLMResponse(content="Final Answer: 3")

        llm = MockLLM(brain)
    """

    model = "mock-llm"

    def __init__(
        self,
        script: Sequence[LLMResponse | Callable[..., LLMResponse]] | Callable[..., LLMResponse],
        *,
        tokens_per_call: tuple[int, int] = (0, 0),
    ) -> None:
        if callable(script):
            self._responder: Callable[..., LLMResponse] = script
            self._script: list[LLMResponse] = []
        else:
            self._responder = None
            self._script = list(script)
        self._cursor = 0
        self._tokens = tokens_per_call
        self.calls: list[dict] = []  # 记录每次请求，测试里可断言

    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        msgs = self._validate(messages)
        self.calls.append({"messages": msgs, "tools": list(tools or [])})

        if self._responder is not None:
            resp = self._responder(msgs, tools)
        else:
            if self._cursor >= len(self._script):
                raise LLMError(
                    f"MockLLM 脚本已耗尽（第 {self._cursor + 1} 次调用）。"
                    "说明 Agent 循环次数超出预期——通常意味着陷入了死循环。"
                )
            resp = self._script[self._cursor]
            self._cursor += 1

        if not isinstance(resp, LLMResponse):
            raise LLMError(f"MockLLM 期望 LLMResponse，收到 {type(resp).__name__}")
        if not resp.prompt_tokens and not resp.completion_tokens:
            resp.prompt_tokens, resp.completion_tokens = self._tokens
        return resp


class OpenAICompatLLM(BaseLLM):
    """调用任何 OpenAI 兼容的 /chat/completions 接口，仅依赖标准库。"""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 60.0,
        max_retries: int = 2,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("api_key 为空。请设置环境变量，或改用 MockLLM 离线运行。")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}

    # ------------------------------------------------------------------ #

    def _payload(self, messages: Sequence[dict], tools: Sequence[dict] | None,
                 temperature: float) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        return body

    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        msgs = self._validate(messages)
        body = json.dumps(self._payload(msgs, tools, temperature), ensure_ascii=False).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                return self._parse(raw)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                except Exception:  # noqa: BLE001
                    pass
                last_error = LLMError(f"HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 403, 404, 422):  # 重试无意义
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = LLMError(f"{type(exc).__name__}: {exc}")

            if attempt < self.max_retries:
                time.sleep(0.8 * (2 ** attempt))  # 指数退避

        raise last_error or LLMError("未知错误")

    @staticmethod
    def _parse(raw: dict) -> LLMResponse:
        choices = raw.get("choices") or []
        if not choices:
            raise LLMError(f"响应缺少 choices 字段：{json.dumps(raw, ensure_ascii=False)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        usage = raw.get("usage") or {}

        raw_calls = message.get("tool_calls") or []
        return LLMResponse(
            content=message.get("content"),
            tool_calls=[ToolCall.from_openai(c, i) for i, c in enumerate(raw_calls)],
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=raw.get("model", ""),
            finish_reason=choice.get("finish_reason", "") or "",
        )
