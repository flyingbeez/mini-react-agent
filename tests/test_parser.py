"""解析层测试：模型输出千奇百怪，解析器必须扛得住。"""

from __future__ import annotations

import unittest

from mini_agent.parser import parse_arguments, parse_react_text, strip_code_fence


class TestParseArguments(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_arguments('{"a": 1}'), {"a": 1})

    def test_dict_passthrough(self):
        self.assertEqual(parse_arguments({"a": 1}), {"a": 1})

    def test_empty_is_empty_dict(self):
        self.assertEqual(parse_arguments(""), {})

    def test_single_quotes_repaired(self):
        self.assertEqual(parse_arguments("{'city': 'Beijing'}"), {"city": "Beijing"})

    def test_bare_keys_repaired(self):
        self.assertEqual(parse_arguments('{city: "Beijing", days: 3}'),
                         {"city": "Beijing", "days": 3})

    def test_key_value_form(self):
        self.assertEqual(parse_arguments("a=1, b=hello"), {"a": 1, "b": "hello"})

    def test_key_value_with_types(self):
        result = parse_arguments("flag=true, ratio=1.5, name=abc")
        self.assertEqual(result, {"flag": True, "ratio": 1.5, "name": "abc"})

    def test_code_fence_stripped(self):
        self.assertEqual(parse_arguments('```json\n{"a": 1}\n```'), {"a": 1})

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_arguments("这不是参数"))

    def test_json_array_is_not_dict(self):
        self.assertIsNone(parse_arguments("[1, 2]"))


class TestStripCodeFence(unittest.TestCase):
    def test_fence(self):
        self.assertEqual(strip_code_fence("```python\nx = 1\n```"), "x = 1")

    def test_no_fence(self):
        self.assertEqual(strip_code_fence("x = 1"), "x = 1")


class TestParseReactText(unittest.TestCase):
    def test_final_answer(self):
        p = parse_react_text("Thought: 我知道了\nFinal Answer: 答案是 42")
        self.assertTrue(p.is_final)
        self.assertEqual(p.final, "答案是 42")
        self.assertEqual(p.thought, "我知道了")

    def test_action_with_json(self):
        p = parse_react_text(
            'Thought: 需要计算\nAction: calculator\nAction Input: {"expression": "1+1"}'
        )
        self.assertTrue(p.is_action)
        self.assertEqual(p.action, "calculator")
        self.assertEqual(p.action_input, {"expression": "1+1"})
        self.assertEqual(p.thought, "需要计算")

    def test_markdown_bold_and_backticks(self):
        p = parse_react_text(
            "**Thought**: 查一下\n**Action**: `search_notes`\n**Action Input**: `{\"query\": \"rag\"}`"
        )
        self.assertEqual(p.action, "search_notes")
        self.assertEqual(p.action_input, {"query": "rag"})

    def test_fullwidth_colon_and_chinese_markers(self):
        p = parse_react_text('思考：需要检索\n行动：search_notes\n行动输入：{"query": "记忆"}')
        self.assertEqual(p.action, "search_notes")
        self.assertEqual(p.action_input, {"query": "记忆"})

    def test_multiline_json(self):
        text = (
            "Thought: 查\n"
            "Action: search_notes\n"
            'Action Input: {\n  "query": "MCP",\n  "top_k": 3\n}'
        )
        p = parse_react_text(text)
        self.assertEqual(p.action_input, {"query": "MCP", "top_k": 3})

    def test_input_stops_at_observation_marker(self):
        text = (
            'Action: calculator\nAction Input: {"expression": "2*3"}\n'
            "Observation: 6\nThought: 够了"
        )
        p = parse_react_text(text)
        self.assertEqual(p.action_input, {"expression": "2*3"})

    def test_json_object_as_action(self):
        p = parse_react_text('{"thought": "算一下", "action": "calculator", "action_input": {"expression": "2+2"}}')
        self.assertTrue(p.is_action)
        self.assertEqual(p.action, "calculator")
        self.assertEqual(p.action_input, {"expression": "2+2"})

    def test_json_object_final_answer(self):
        p = parse_react_text('{"final_answer": "42"}')
        self.assertTrue(p.is_final)
        self.assertEqual(p.final, "42")

    def test_entire_output_in_code_fence(self):
        p = parse_react_text("```\nThought: 想\nAction: now\nAction Input: {}\n```")
        self.assertEqual(p.action, "now")

    def test_non_json_input_kept_raw(self):
        p = parse_react_text("Action: calculator\nAction Input: 1+1 等于几")
        self.assertEqual(p.action, "calculator")
        self.assertIsNone(p.action_input)
        self.assertEqual(p.raw_input, "1+1 等于几")

    def test_unknown_format(self):
        p = parse_react_text("我觉得这个问题挺难的，容我想想。")
        self.assertEqual(p.kind, "unknown")

    def test_final_wins_over_action_when_earlier(self):
        p = parse_react_text("Thought: 直接回答\nFinal Answer: 好的\nAction: now")
        self.assertTrue(p.is_final)

    def test_empty_text(self):
        self.assertEqual(parse_react_text("").kind, "unknown")


if __name__ == "__main__":
    unittest.main()
