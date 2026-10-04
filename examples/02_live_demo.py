#!/usr/bin/env python3
"""接入真实大模型（任何 OpenAI 兼容接口）。

配置环境变量后运行：

    # DeepSeek（推荐，便宜且对中文友好）
    Windows :  $env:DEEPSEEK_API_KEY="sk-xxxx"
    Linux   :  export DEEPSEEK_API_KEY="sk-xxxx"

    python examples/02_live_demo.py "北京到上海高铁大概多久？算一下 4.5 小时是多少分钟"

也支持其他平台（OpenAI / 通义千问 / 智谱 / Moonshot / vLLM / Ollama）：

    $env:LLM_BASE_URL="https://api.moonshot.cn/v1"
    $env:LLM_MODEL="moonshot-v1-8k"
    $env:LLM_API_KEY="sk-xxxx"

不带参数运行会进入交互模式，输入 exit 退出。
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mini_agent import LLMError, OpenAICompatLLM, ReActAgent, ToolRegistry  # noqa: E402
from mini_agent.builtin_tools import calculator, now, search_notes, text_stats  # noqa: E402

LINE = "-" * 78


def build_llm() -> OpenAICompatLLM:
    configs = [
        ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
        ("LLM_API_KEY", os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
         os.environ.get("LLM_MODEL", "gpt-4o-mini")),
    ]
    for env_name, base_url, model in configs:
        key = os.environ.get(env_name)
        if key:
            print(f"[配置] 使用 {env_name}，base_url={base_url}，model={model}")
            return OpenAICompatLLM(key, model=model, base_url=base_url)
    raise SystemExit(
        "没有找到 API Key。请设置 DEEPSEEK_API_KEY 或 LLM_API_KEY 环境变量。\n"
        "如果只是想看 Agent 循环，请运行：python examples/01_offline_demo.py（不需要 Key）"
    )


def main() -> None:
    mode = os.environ.get("AGENT_MODE", "native")
    if mode not in ("native", "text"):
        raise SystemExit("AGENT_MODE 只能是 native 或 text")

    registry = ToolRegistry()
    for func in (calculator, now, text_stats, search_notes):
        registry.register(func)

    agent = ReActAgent(
        build_llm(),
        registry,
        mode=mode,
        max_steps=6,
        timeout_s=90,
        verbose=True,  # 把每一步实时打到 stderr
    )
    print(f"[配置] mode={mode}，可用工具={registry.names()}")
    print(LINE)

    questions = sys.argv[1:]
    if questions:
        for q in questions:
            ask(agent, q)
        return

    print("交互模式：输入问题回车，输入 exit / q 退出。")
    while True:
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit", "q", ""):
            break
        ask(agent, question)


def ask(agent: ReActAgent, question: str) -> None:
    try:
        result = agent.run(question)
    except LLMError as exc:
        print(f"[错误] 模型调用失败：{exc}")
        return

    print(LINE)
    print(result.trace.render())
    print(LINE)
    print(f"Agent > {result.answer}")
    print(f"[用量] token={result.trace.total_tokens}，步数={len(result.steps)}，"
          f"工具调用={result.trace.tool_calls}，结束原因={result.stopped_reason}")


if __name__ == "__main__":
    main()
