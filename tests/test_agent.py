"""Agent 循环测试：用 MockLLM 把"模型行为"变成可复现的脚本。

这是本仓库最重要的测试思路——**不要让测试依赖真实 LLM**。
把模型当成一个可编程的外部依赖，才能稳定地测出循环逻辑、错误恢复和死循环防护。
"""

from __future__ import annotations

import unittest

from mini_agent.agent import ReActAgent
from mini_agent.llm import LLMError, LLMResponse, MockLLM, ToolCall
from mini_agent.tools import ToolRegistry, tool


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool
    def add(a: int, b: int) -> int:
        """两数相加。

        Args:
            a: 第一个加数
            b: 第二个加数
        """
        return a + b

    @tool
    def echo(text: str) -> str:
        """原样返回文本。

        Args:
            text: 要回显的文本
        """
        return text

    @tool
    def boom(x: int) -> str:
        """总是失败的工具。

        Args:
            x: 随便传
        """
        raise RuntimeError("工具内部异常")

    @tool
    def ping() -> str:
        """无参数工具。"""
        return "pong"

    for f in (add, echo, boom, ping):
        registry.register(f)
    return registry


def call(name: str, args: str, cid: str = "c1") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, arguments=args, id=cid)],
                       prompt_tokens=10, completion_tokens=5)


class TestNativeMode(unittest.TestCase):
    def test_single_tool_then_answer(self):
        llm = MockLLM([
            call("add", '{"a": 2, "b": 3}'),
            LLMResponse(content="结果是 5", prompt_tokens=20, completion_tokens=4),
        ])
        result = ReActAgent(llm, build_registry(), mode="native").run("2+3 等于几")

        self.assertTrue(result.success)
        self.assertEqual(result.answer, "结果是 5")
        self.assertEqual(result.stopped_reason, "final_answer")
        self.assertEqual(result.trace.tool_calls, 1)
        self.assertEqual(result.steps[0].observation, "5")
        self.assertTrue(result.steps[0].ok)
        # token 跨轮累加：第一轮 10+5，第二轮 20+4
        self.assertEqual(result.trace.total_tokens, 39)

    def test_parallel_tool_calls_in_one_turn(self):
        llm = MockLLM([
            LLMResponse(tool_calls=[
                ToolCall(name="add", arguments='{"a": 1, "b": 1}', id="c1"),
                ToolCall(name="echo", arguments='{"text": "hi"}', id="c2"),
            ]),
            LLMResponse(content="完成"),
        ])
        result = ReActAgent(llm, build_registry(), mode="native").run("并行调用")
        self.assertEqual(result.trace.tool_calls, 2)
        observations = [s.observation for s in result.steps if s.action]
        self.assertEqual(observations, ["2", "hi"])
        self.assertEqual(result.steps[-1].mode, "final")
        # 第二条消息必须带上两条 tool 结果
        second_call_messages = llm.calls[1]["messages"]
        tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        self.assertEqual({m["tool_call_id"] for m in tool_msgs}, {"c1", "c2"})

    def test_tools_are_advertised_to_model(self):
        llm = MockLLM([LLMResponse(content="ok")])
        ReActAgent(llm, build_registry(), mode="native").run("hi")
        names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
        self.assertEqual(names, {"add", "echo", "boom", "ping"})

    def test_unknown_tool_becomes_observation_then_recovers(self):
        llm = MockLLM([
            call("multiply", '{"a": 1, "b": 2}'),
            call("add", '{"a": 1, "b": 2}', cid="c2"),
            LLMResponse(content="3"),
        ])
        result = ReActAgent(llm, build_registry(), mode="native").run("算 1+2")

        self.assertFalse(result.steps[0].ok)
        self.assertIn("不存在名为", result.steps[0].observation)
        self.assertTrue(result.steps[1].ok)
        self.assertEqual(result.answer, "3")
        self.assertEqual(result.trace.failed_calls, 1)

    def test_tool_exception_becomes_observation_then_recovers(self):
        llm = MockLLM([call("boom", '{"x": 1}'), LLMResponse(content="换了个思路")])
        result = ReActAgent(llm, build_registry(), mode="native").run("调用会失败的工具")
        self.assertIn("工具错误", result.steps[0].observation)
        self.assertIn("工具内部异常", result.steps[0].observation)
        self.assertTrue(result.success)

    def test_invalid_json_arguments_reported_to_model(self):
        llm = MockLLM([call("add", "{a: 1, b: 2"), LLMResponse(content="我改")])
        result = ReActAgent(llm, build_registry(), mode="native").run("x")
        self.assertFalse(result.steps[0].ok)
        self.assertIn("不是合法 JSON", result.steps[0].observation)
        self.assertNotIn("调用了不存在的", result.steps[0].observation)

    def test_max_steps_guard(self):
        # 模型永远只调工具，从不给答案
        script = [call("ping", "{}", cid=f"c{i}") for i in range(6)]
        llm = MockLLM(script)
        result = ReActAgent(llm, build_registry(), mode="native", max_steps=3,
                            repeat_stop_at=99, loop_threshold=99).run("永远不停")
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(len(result.steps), 3)
        self.assertIn("达到最大步数", result.answer)

    def test_loop_detection_stops_repeats(self):
        script = [call("ping", "{}", cid=f"c{i}") for i in range(10)]
        llm = MockLLM(script)
        result = ReActAgent(llm, build_registry(), mode="native", max_steps=8,
                            loop_threshold=2, repeat_stop_at=3).run("卡住了")
        self.assertEqual(result.stopped_reason, "loop_detected")
        self.assertEqual(len(result.steps), 3)
        self.assertIn("死循环", result.answer)

    def test_loop_nudge_is_injected(self):
        script = [call("ping", "{}", cid=f"c{i}") for i in range(10)]
        llm = MockLLM(script)
        ReActAgent(llm, build_registry(), mode="native", max_steps=4,
                   loop_threshold=2, repeat_stop_at=99).run("卡住了")
        nudges = [
            m for m in llm.calls[-1]["messages"]
            if m["role"] == "user" and "重复调用" in str(m.get("content", ""))
        ]
        self.assertTrue(nudges)

    def test_different_args_are_not_treated_as_loop(self):
        script = [
            call("echo", '{"text": "a"}', "c1"),
            call("echo", '{"text": "b"}', "c2"),
            call("echo", '{"text": "c"}', "c3"),
            LLMResponse(content="完成"),
        ]
        result = ReActAgent(MockLLM(script), build_registry(), mode="native",
                            repeat_stop_at=3).run("x")
        self.assertTrue(result.success)
        self.assertEqual(result.trace.tool_calls, 3)

    def test_llm_error_returns_failed_result(self):
        def broken(messages, tools):
            raise LLMError("网络断了")

        result = ReActAgent(MockLLM(broken), build_registry(), mode="native").run("x")
        self.assertFalse(result.success)
        self.assertEqual(result.stopped_reason, "error")
        self.assertIn("网络断了", result.answer)

    def test_empty_model_content_gets_placeholder(self):
        llm = MockLLM([LLMResponse(content="   ")])
        result = ReActAgent(llm, build_registry(), mode="native").run("x")
        self.assertIn("模型没有返回内容", result.answer)

    def test_long_observation_truncated(self):
        llm = MockLLM([
            call("echo", '{"text": "%s"}' % ("x" * 5000)),
            LLMResponse(content="done"),
        ])
        result = ReActAgent(llm, build_registry(), mode="native",
                            max_observation_chars=100).run("x")
        self.assertLess(len(result.steps[0].observation), 200)
        self.assertIn("已截断", result.steps[0].observation)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            ReActAgent(MockLLM([]), build_registry(), mode="magic")


class TestTextMode(unittest.TestCase):
    def test_full_cycle(self):
        def brain(messages, tools):
            if "Observation" not in messages[-1]["content"]:
                return LLMResponse(
                    content='Thought: 我该算一下\nAction: add\nAction Input: {"a": 4, "b": 5}'
                )
            return LLMResponse(content="Thought: 够了\nFinal Answer: 4+5=9")

        agent = ReActAgent(MockLLM(brain), build_registry(), mode="text")
        result = agent.run("4+5 等于几")

        self.assertTrue(result.success)
        self.assertEqual(result.answer, "4+5=9")
        self.assertEqual(result.steps[0].mode, "text")
        self.assertEqual(result.steps[0].observation, "9")
        self.assertEqual(result.steps[-1].mode, "final")
        # 系统提示词里必须包含工具清单
        system = agent.llm.calls[0]["messages"][0]["content"]
        self.assertIn("add(a: integer, b: integer)", system)

    def test_format_violation_then_recovery(self):
        state = {"n": 0}

        def brain(messages, tools):
            state["n"] += 1
            if state["n"] == 1:
                return LLMResponse(content="我不太确定，随便说点什么。")
            return LLMResponse(content="Final Answer: 修正后的答案")

        result = ReActAgent(MockLLM(brain), build_registry(), mode="text").run("x")
        self.assertFalse(result.steps[0].ok)
        self.assertIn("格式", result.steps[0].error)
        self.assertTrue(result.success)
        self.assertEqual(result.answer, "修正后的答案")

    def test_single_param_fallback_from_raw_text(self):
        def brain(messages, tools):
            if "Observation" not in messages[-1]["content"]:
                # 故意不写 JSON
                return LLMResponse(content="Thought: 直接传\nAction: echo\nAction Input: 你好")
            return LLMResponse(content="Final Answer: 收到")

        result = ReActAgent(MockLLM(brain), build_registry(), mode="text").run("x")
        self.assertEqual(result.steps[0].observation, "你好")

    def test_no_arg_tool_with_empty_input(self):
        def brain(messages, tools):
            if "Observation" not in messages[-1]["content"]:
                return LLMResponse(content="Action: ping\nAction Input: {}")
            return LLMResponse(content="Final Answer: pong 可用")

        result = ReActAgent(MockLLM(brain), build_registry(), mode="text").run("x")
        self.assertEqual(result.steps[0].observation, "pong")
        self.assertTrue(result.success)

    def test_unknown_tool_in_text_mode(self):
        def brain(messages, tools):
            if "Observation" not in messages[-1]["content"]:
                return LLMResponse(content='Action: nope\nAction Input: {"a": 1}')
            return LLMResponse(content="Final Answer: 换个工具")

        result = ReActAgent(MockLLM(brain), build_registry(), mode="text").run("x")
        self.assertIn("不存在名为", result.steps[0].error)

    def test_multi_param_unparseable_input_reports_error(self):
        def brain(messages, tools):
            if "Observation" not in messages[-1]["content"]:
                return LLMResponse(content="Action: add\nAction Input: 一加二")
            return LLMResponse(content="Final Answer: ok")

        result = ReActAgent(MockLLM(brain), build_registry(), mode="text").run("x")
        self.assertIn("不是合法 JSON", result.steps[0].error)

    def test_text_mode_loop_detection(self):
        def brain(messages, tools):
            return LLMResponse(content='Action: ping\nAction Input: {}')

        result = ReActAgent(MockLLM(brain), build_registry(), mode="text",
                            max_steps=8, loop_threshold=2, repeat_stop_at=3).run("卡住")
        self.assertEqual(result.stopped_reason, "loop_detected")

    def test_timeout(self):
        class SlowLLM(MockLLM):
            def chat(self, messages, tools=None, temperature=0.0):
                import time
                time.sleep(0.05)
                return LLMResponse(content="Thought: 慢\nAction: ping\nAction Input: {}")

        result = ReActAgent(SlowLLM([]), build_registry(), mode="text",
                            timeout_s=0.01, repeat_stop_at=99).run("x")
        self.assertEqual(result.stopped_reason, "error")
        self.assertIn("超时", result.answer)


class TestTrace(unittest.TestCase):
    def test_trace_json_roundtrip(self):
        import json

        llm = MockLLM([call("add", '{"a": 1, "b": 1}'), LLMResponse(content="2")])
        result = ReActAgent(llm, build_registry(), mode="native").run("1+1")
        data = json.loads(result.trace.to_json())
        self.assertEqual(data["question"], "1+1")
        self.assertEqual(data["final_answer"], "2")
        self.assertEqual(data["tool_calls"], 1)
        self.assertEqual(data["steps"][0]["action"], "add")
        self.assertIn("add", result.trace.render())


if __name__ == "__main__":
    unittest.main()
