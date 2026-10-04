#!/usr/bin/env python3
"""三分钟学会给自己的业务加一个工具。

这个示例演示三件事：
    1. 用 @tool 装饰器把普通函数变成 Agent 工具（Schema 自动生成）
    2. 工具内部怎么报错才能让模型看懂并自我纠正（抛 ToolError，附带"怎么办"）
    3. 同一份代码在 native / text 两种模式下都能跑

运行：python examples/03_custom_tool.py
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
    ToolError,
    ToolRegistry,
    tool,
)

LINE = "=" * 78

# --------------------------------------------------------------------------- #
# 业务工具：一个"岗位匹配度打分"小工具（和找工作这个场景有关）
# --------------------------------------------------------------------------- #
# 假的岗位库，真实项目里换成你的数据库 / 向量检索
JOBS = {
    "agent-app": {"title": "大模型应用开发实习生", "must": ["python", "llm", "prompt", "rag"]},
    "agent-algo": {"title": "Agent 算法实习生", "must": ["python", "pytorch", "rl", "评测"]},
    "agent-platform": {"title": "AI 平台实习生", "must": ["python", "fastapi", "docker", "向量数据库"]},
}


@tool
def list_jobs() -> dict:
    """列出岗位库里所有岗位的 id 和名称。没有任何参数。"""
    return {jid: job["title"] for jid, job in JOBS.items()}


@tool
def match_job(job_id: str, skills: list[str]) -> dict:
    """计算候选人与某个岗位的技能匹配度。

    Args:
        job_id: 岗位 id，可用 list_jobs 查询
        skills: 候选人掌握的技能关键词列表，例如 ["python", "rag"]
    """
    if job_id not in JOBS:
        # 关键：错误信息要告诉模型"下一步该怎么办"，而不只是"出错了"
        raise ToolError(
            f"岗位 id {job_id!r} 不存在。请先调用 list_jobs 获取合法 id。"
        )
    if not skills:
        raise ToolError("skills 不能为空，请至少列出 1 个技能关键词。")

    job = JOBS[job_id]
    have = {s.strip().lower() for s in skills}
    hit = [s for s in job["must"] if s in have]
    missing = [s for s in job["must"] if s not in have]
    score = round(100 * len(hit) / len(job["must"])) if job["must"] else 0

    return {
        "job": job["title"],
        "score": score,
        "matched": hit,
        "missing": missing,
        "advice": f"优先补 {missing[0]}" if missing else "技能已覆盖，可以准备项目深挖",
    }


def main() -> None:
    registry = ToolRegistry()
    registry.register(list_jobs)
    registry.register(match_job)

    print("自动生成的工具 Schema：")
    for spec in registry.specs():
        fn = spec["function"]
        print(f"  - {fn['name']}: {fn['description']}")
        print(f"      参数：{fn['parameters']}")

    print(f"\n{LINE}\n演示：模型先列出岗位，再打分，中途还用错了一次 id\n{LINE}")

    llm = MockLLM([
        # ① 模型先看看有哪些岗位
        LLMResponse(
            content="先看看岗位库里有什么。",
            tool_calls=[ToolCall(name="list_jobs", arguments="{}")],
        ),
        # ② 模型用错了 id（故意），会收到 ToolError 的提示
        LLMResponse(
            content="我猜 id 是 agent。",
            tool_calls=[ToolCall(name="match_job",
                                 arguments='{"job_id": "agent", "skills": ["python", "rag"]}')],
        ),
        # ③ 看到"请先调用 list_jobs 获取合法 id"后，模型改用合法 id
        LLMResponse(
            content="id 写错了，换成 agent-app。",
            tool_calls=[ToolCall(
                name="match_job",
                arguments='{"job_id": "agent-app", "skills": ["python", "llm", "prompt"]}')],
        ),
        LLMResponse(
            content="你的技能覆盖了 python / llm / prompt，还缺 rag；匹配度 75%。建议补一个 RAG 项目。"
        ),
    ])

    result = ReActAgent(llm, registry, mode="native").run(
        "我叫候选者，会 python、llm、prompt，看看我适合哪个岗位？"
    )
    print(result.trace.render())
    print(f"\n{LINE}\n>> {result.answer}\n{LINE}")
    print("要点回顾：")
    print("  1. @tool + 类型注解 + docstring → Schema 自动生成，不用手写 JSON")
    print("  2. 工具报错时给出『下一步怎么办』，模型才有机会自我纠正")
    print("  3. 工具函数的返回值可以是 dict / list，框架会自动转成 JSON 字符串")


if __name__ == "__main__":
    main()
