"""内置工具测试，重点是「安全计算器」不能被绕过。"""

from __future__ import annotations

import json
import re
import unittest

from mini_agent.builtin_tools import calculator, now, search_notes, text_stats
from mini_agent.tools import ToolError, ToolRegistry


def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for f in (calculator, now, text_stats, search_notes):
        reg.register(f)
    return reg


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = calculator.__wrapped__ if hasattr(calculator, "__wrapped__") else calculator

    def test_basic(self):
        self.assertEqual(self.calc("1+2*3"), "7")

    def test_parentheses_and_division(self):
        self.assertEqual(self.calc("(3+5)*2/4"), "4")

    def test_integer_float_formatting(self):
        self.assertEqual(self.calc("10/4"), "2.5")
        self.assertEqual(self.calc("4/2"), "2")

    def test_functions(self):
        self.assertTrue(self.calc("sqrt(2)").startswith("1.4142135"))
        self.assertEqual(self.calc("round(sqrt(2), 2)"), "1.41")

    def test_constants(self):
        self.assertEqual(self.calc("floor(pi)"), "3")

    def test_negative_and_power(self):
        self.assertEqual(self.calc("-2**3"), "-8")
        self.assertEqual(self.calc("2**10"), "1024")

    def test_rejects_underscore_sequence(self):
        with self.assertRaises(ToolError):
            self.calc("__import__('os')")

    def test_rejects_attribute_access(self):
        with self.assertRaises(ToolError):
            self.calc("(1).__class__")

    def test_rejects_unknown_function(self):
        with self.assertRaises(ToolError):
            self.calc("open('x')")

    def test_rejects_unknown_name(self):
        with self.assertRaises(ToolError):
            self.calc("answer")

    def test_rejects_lambda(self):
        with self.assertRaises(ToolError):
            self.calc("(lambda: 1)()")

    def test_rejects_list_literal(self):
        with self.assertRaises(ToolError):
            self.calc("[1, 2]")

    def test_rejects_huge_exponent(self):
        with self.assertRaises(ToolError):
            self.calc("2**9999")

    def test_rejects_empty(self):
        with self.assertRaises(ToolError):
            self.calc("")

    def test_rejects_overlong(self):
        with self.assertRaises(ToolError):
            self.calc("1+" * 200)

    def test_division_by_zero_is_a_tool_error(self):
        with self.assertRaises(ToolError):
            self.calc("1/0")

    def test_via_registry_string_args(self):
        self.assertEqual(registry().call("calculator", {"expression": "2+2"}), "4")


class TestNow(unittest.TestCase):
    def test_format(self):
        value = now.__wrapped__() if hasattr(now, "__wrapped__") else now()
        self.assertRegex(value, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_bad_timezone(self):
        f = now.__wrapped__ if hasattr(now, "__wrapped__") else now
        with self.assertRaises(ToolError):
            f(99)


class TestTextStats(unittest.TestCase):
    def test_counts(self):
        f = text_stats.__wrapped__ if hasattr(text_stats, "__wrapped__") else text_stats
        result = f("Hello world hello\n你好世界", top_k=3)
        self.assertEqual(result["latin_words"], 3)
        self.assertEqual(result["cjk_chars"], 4)
        self.assertEqual(result["lines"], 2)
        self.assertEqual(result["top_words"][0], {"word": "hello", "count": 2})

    def test_top_k_bounds(self):
        f = text_stats.__wrapped__ if hasattr(text_stats, "__wrapped__") else text_stats
        with self.assertRaises(ToolError):
            f("abc", top_k=0)

    def test_non_string(self):
        f = text_stats.__wrapped__ if hasattr(text_stats, "__wrapped__") else text_stats
        with self.assertRaises(ToolError):
            f(123)  # type: ignore[arg-type]

    def test_json_serialisable_via_registry(self):
        payload = registry().call("text_stats", {"text": "a b c", "top_k": 2})
        self.assertEqual(json.loads(payload)["latin_words"], 3)


class TestSearchNotes(unittest.TestCase):
    def test_finds_react_note(self):
        f = search_notes.__wrapped__ if hasattr(search_notes, "__wrapped__") else search_notes
        result = f("什么是 ReAct 循环", top_k=2)
        self.assertTrue(result["hits"])
        self.assertEqual(result["hits"][0]["id"], "react")

    def test_finds_rag_note(self):
        f = search_notes.__wrapped__ if hasattr(search_notes, "__wrapped__") else search_notes
        result = f("向量检索 增强生成")
        self.assertEqual(result["hits"][0]["id"], "rag")

    def test_top_k_respected(self):
        f = search_notes.__wrapped__ if hasattr(search_notes, "__wrapped__") else search_notes
        self.assertLessEqual(len(f("Agent 记忆 规划 评测", top_k=2)["hits"]), 2)

    def test_no_hit(self):
        f = search_notes.__wrapped__ if hasattr(search_notes, "__wrapped__") else search_notes
        self.assertEqual(f("zzzqqq")["hits"], [])

    def test_empty_query(self):
        f = search_notes.__wrapped__ if hasattr(search_notes, "__wrapped__") else search_notes
        with self.assertRaises(ToolError):
            f("   ")

    def test_results_are_sorted_desc(self):
        f = search_notes.__wrapped__ if hasattr(search_notes, "__wrapped__") else search_notes
        scores = [h["score"] for h in f("Agent MCP RAG 评测 记忆", top_k=4)["hits"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_registry_payload_contains_source_id(self):
        payload = json.loads(registry().call("search_notes", {"query": "MCP 协议", "top_k": 1}))
        self.assertIn("id", payload["hits"][0])


class TestSchemas(unittest.TestCase):
    def test_all_tools_have_description(self):
        for t in registry().tools.values():
            self.assertTrue(t.description, t.name)
            self.assertTrue(t.description.strip(), t.name)

    def test_required_fields_declared(self):
        reg = registry()
        self.assertIn("expression", reg.tools["calculator"].parameters["required"])
        self.assertIn("query", reg.tools["search_notes"].parameters["required"])
        self.assertNotIn("timezone_offset_hours", reg.tools["now"].parameters["required"])

    def test_no_dunder_leaks_into_schema(self):
        dumped = json.dumps(registry().specs(), ensure_ascii=False)
        self.assertIsNone(re.search(r'"__\w+__"', dumped))


if __name__ == "__main__":
    unittest.main()
