"""工具层测试：schema 生成、参数校验、错误处理。"""

from __future__ import annotations

import unittest
from typing import Annotated, Literal, Optional

from mini_agent.tools import Tool, ToolError, ToolRegistry, function_schema, tool


class TestSchemaGeneration(unittest.TestCase):
    def test_basic_types_and_required(self):
        def f(a: str, b: int, c: float = 1.5, d: bool = False) -> str:
            """演示函数。

            Args:
                a: 第一个参数
                b: 第二个参数
            """
            return ""

        schema = function_schema(f)
        self.assertEqual(schema["name"], "f")
        self.assertEqual(schema["description"], "演示函数。")
        props = schema["parameters"]["properties"]
        self.assertEqual(props["a"], {"type": "string", "description": "第一个参数"})
        self.assertEqual(props["b"], {"type": "integer", "description": "第二个参数"})
        self.assertEqual(props["c"]["type"], "number")
        self.assertEqual(props["c"]["default"], 1.5)
        self.assertEqual(schema["parameters"]["required"], ["a", "b"])

    def test_optional_and_union_strip_none(self):
        def f(x: Optional[int] = None) -> str:
            """doc"""
            return ""

        schema = function_schema(f)
        self.assertEqual(schema["parameters"]["properties"]["x"]["type"], "integer")

    def test_pep604_union(self):
        def f(x: int | None = None) -> str:
            """doc"""
            return ""

        self.assertEqual(function_schema(f)["parameters"]["properties"]["x"]["type"], "integer")

    def test_list_and_literal(self):
        def f(items: list[str], mode: Literal["fast", "slow"] = "fast") -> str:
            """doc"""
            return ""

        props = function_schema(f)["parameters"]["properties"]
        self.assertEqual(props["items"]["type"], "array")
        self.assertEqual(props["items"]["items"]["type"], "string")
        self.assertEqual(props["mode"]["enum"], ["fast", "slow"])

    def test_annotated_description_wins(self):
        def f(x: Annotated[int, "来自 Annotated 的说明"]) -> str:
            """doc

            Args:
                x: 来自 docstring 的说明
            """
            return ""

        self.assertEqual(
            function_schema(f)["parameters"]["properties"]["x"]["description"],
            "来自 Annotated 的说明",
        )

    def test_varargs_ignored(self):
        def f(a: int, *args, **kwargs) -> str:
            """doc"""
            return ""

        props = function_schema(f)["parameters"]["properties"]
        self.assertEqual(list(props), ["a"])

    def test_decorator_name_and_description(self):
        @tool(name="renamed", description="装饰器给的描述")
        def original(x: int) -> int:
            """docstring 描述"""
            return x

        registry = ToolRegistry()
        registry.register(original)
        self.assertEqual(registry.names(), ["renamed"])
        self.assertEqual(registry.tools["renamed"].description, "装饰器给的描述")


class TestValidation(unittest.TestCase):
    def setUp(self):
        def add(a: int, b: int = 10) -> int:
            """相加。

            Args:
                a: 加数
                b: 另一个加数
            """
            return a + b

        self.registry = ToolRegistry()
        self.registry.register(add)

    def test_ok(self):
        self.assertEqual(self.registry.call("add", {"a": 1}), "11")

    def test_missing_required(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.call("add", {})
        self.assertIn("缺少必填参数", str(ctx.exception))

    def test_type_coercion_string_to_int(self):
        self.assertEqual(self.registry.call("add", {"a": "5"}), "15")

    def test_type_coercion_bad(self):
        with self.assertRaises(ToolError):
            self.registry.call("add", {"a": "abc"})

    def test_arguments_as_json_string(self):
        self.assertEqual(self.registry.call("add", '{"a": 2, "b": 3}'), "5")

    def test_arguments_as_invalid_json_string(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.call("add", "{a: 2")
        self.assertIn("JSON", str(ctx.exception))

    def test_unknown_tool(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.call("nope", {})
        self.assertIn("不存在名为", str(ctx.exception))

    def test_extra_args_ignored(self):
        self.assertEqual(self.registry.call("add", {"a": 1, "zzz": 9}), "11")

    def test_tool_exception_becomes_toolerror(self):
        def boom(x: int) -> int:
            """boom"""
            raise ValueError("炸了")

        registry = ToolRegistry()
        registry.register(boom)
        with self.assertRaises(ToolError) as ctx:
            registry.call("boom", {"x": 1})
        self.assertIn("炸了", str(ctx.exception))

    def test_bool_not_accepted_as_int(self):
        def f(n: int) -> str:
            """doc"""
            return str(n)

        registry = ToolRegistry()
        registry.register(f)
        with self.assertRaises(ToolError):
            registry.call("f", {"n": True})

    def test_float_integral_coerced(self):
        def f(n: int) -> str:
            """doc"""
            return str(n)

        registry = ToolRegistry()
        registry.register(f)
        self.assertEqual(registry.call("f", {"n": 3.0}), "3")

    def test_dict_result_serialised(self):
        def f() -> dict:
            """doc"""
            return {"ok": True, "值": 1}

        registry = ToolRegistry()
        registry.register(f)
        self.assertEqual(registry.call("f", {}), '{"ok": true, "值": 1}')


class TestToolObject(unittest.TestCase):
    def test_spec_shape(self):
        def f(a: str) -> str:
            """doc

            Args:
                a: 参数 a
            """
            return a

        spec = Tool.from_function(f).spec()
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], "f")
        self.assertIn("parameters", spec["function"])

    def test_registry_accepts_tool_objects_and_functions(self):
        def f(a: int) -> int:
            """doc"""
            return a

        registry = ToolRegistry()
        registry.register(f)
        registry.register(Tool.from_function(f, name="f2"))
        self.assertEqual(registry.names(), ["f", "f2"])
        self.assertEqual(len(registry), 2)
        self.assertIn("f", registry)

    def test_describe_contains_signature(self):
        def f(a: int, b: str) -> str:
            """说明

            Args:
                a: 数字
                b: 文本
            """
            return ""

        registry = ToolRegistry()
        registry.register(f)
        self.assertIn("f(a: integer, b: string)", registry.describe())


if __name__ == "__main__":
    unittest.main()
