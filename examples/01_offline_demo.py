#!/usr/bin/env python3
"""离线演示：不需要任何 API Key，用 MockLLM 把 Agent 循环跑给你看。

三个场景分别演示：
    场景 A  原生 function calling：单工具闭环
    场景 B  工具调用出错 → 模型自我纠正（Agent 的容错能力从哪来）
    场景 C  文本 ReAct：模型的参数写错了 JSON，解析器怎么救回来

运行：python examples/01_offline_demo.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mini_agent import (  # noqa: E402
    LLMResponse,
    MockLLM,
    ReActAgent,
    ToolCall,
    ToolRegistry,
)
from mini_agent.builtin_tools import calculator, now, search_notes, text_stats  # noqa: E402

LINE = "=" * 78


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for func in (calculator, now, text_stats, search_notes):
        registry.register(func)
    return registry


def title(text: str) -> None:
    print(f"\n{LINE}\n{text}\n{LINE}")


def scenario_a() -> None:
    """原生 function calling：模型调一次计算器，然后给出答案。"""
    title("场景 A ｜ 原生 Function Calling：Thought → Action → Observation → Answer")

    llm = MockLLM([
        LLMResponse(
            content="我需要精确计算，先调用计算器。",
            tool_calls=[ToolCall(name="calculator", arguments='{"expression": "(365*24-100)/7"}', id="call_1")],
            prompt_tokens=180, completion_tokens=32,
        ),
        LLMResponse(
            content="(365*24-100)/7 = 1237.14（保留两位小数）",
            prompt_tokens=240, completion_tokens=26,
        ),
    ])
    agent = ReActAgent(llm, build_registry(), mode="native")
    result = agent.run("帮我算一下 (365*24-100)/7，保留两位小数")

    print(result.trace.render())
    print(f"\n>> 最终答案：{result.answer}")
    print(f">> 总 token：{result.trace.total_tokens}（输入 {result.trace.prompt_tokens} / 输出 {result.trace.completion_tokens}）")


def scenario_b() -> None:
    """工具调用出错：模型先调了一个不存在的工具，报错后自己改用正确的工具。"""
    title("场景 B ｜ 错误自愈：工具不存在 → 错误变成 Observation → 模型自己改对")

    llm = MockLLM([
        # 第一轮：模型想当然了，编了一个不存在的工具名
        LLMResponse(
            content="我需要做数学运算。",
            tool_calls=[ToolCall(name="math_solver", arguments='{"expr": "12*12"}', id="call_1")],
        ),
        # 第二轮：看到了「不存在名为 math_solver 的工具」，改用 calculator
        LLMResponse(
            content="工具名不对，改用 calculator。",
            tool_calls=[ToolCall(name="calculator", arguments='{"expression": "12*12"}', id="call_2")],
            prompt_tokens=260, completion_tokens=28,
        ),
        LLMResponse(content="12 × 12 = 144", prompt_tokens=300, completion_tokens=12),
    ])
    agent = ReActAgent(llm, build_registry(), mode="native")
    result = agent.run("12 的平方是多少？")

    print(result.trace.render())
    print(f"\n>> 失败调用次数：{result.trace.failed_calls}，但任务最终成功：{result.success}")
    print(">> 关键点：工具错误没有被抛成异常炸掉循环，而是变成了一条 observation。")


def scenario_c() -> None:
    """文本 ReAct：模型输出的 JSON 用了单引号，解析器自动修好。"""
    title("场景 C ｜ 文本 ReAct：解析器兜住模型的格式手滑")

    def brain(messages, tools):
        last = messages[-1]["content"]
        if "Observation" not in last:
            # 注意：这里故意用了非法 JSON（单引号），看解析器怎么处理
            return LLMResponse(
                content=(
                    "Thought: 我需要检索 Agent 相关知识。\n"
                    "Action: search_notes\n"
                    "Action Input: {'query': 'ReAct 循环', 'top_k': 1}"
                ),
                prompt_tokens=420, completion_tokens=48,
            )
        return LLMResponse(
            content="Thought: 检索到了足够的资料。\nFinal Answer: ReAct 的核心是 Thought→Action→Observation 循环。",
            prompt_tokens=560, completion_tokens=30,
        )

    agent = ReActAgent(MockLLM(brain), build_registry(), mode="text")
    result = agent.run("ReAct 循环是怎么工作的？")

    print(result.trace.render())
    print(f"\n>> 结论：{'单引号 JSON 被自动修复' if result.steps[0].ok else '解析失败'}")


def scenario_d() -> None:
    """死循环防护：模型一直重复同一个调用。"""
    title("场景 D ｜ 死循环防护：模型卡住了，Agent 主动打断并给出诊断")

    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall(name="now", arguments="{}", id=f"call_{i}")])
        for i in range(10)
    ])
    agent = ReActAgent(llm, build_registry(), mode="native", loop_threshold=2, repeat_stop_at=3)
    result = agent.run("现在几点了？（这个模型坏掉了，只会重复调用 now）")

    print(result.trace.render())
    print(f"\n>> 结束原因：{result.stopped_reason}")
    print(">> 关键点：没有无限循环、没有烧 token，而是带着诊断信息优雅退出。")


def main() -> None:
    print("mini-react-agent 离线演示（无需 API Key）\n")
    print("可用工具：", build_registry().names())
    scenario_a()
    scenario_b()
    scenario_c()
    scenario_d()
    print(f"\n{LINE}\n演示结束。接下来：\n"
          f"  * 想看逐模块讲解 → docs/02-代码导读.md\n"
          f"  * 想接真实模型   → python examples/02_live_demo.py \"你的问题\"\n"
          f"  * 想看测试怎么写的 → tests/test_agent.py\n{LINE}")


if __name__ == "__main__":
    main()
