"""工具抽象层：一行装饰器把普通 Python 函数变成 Agent 可调用的工具。

设计要点（面试常问）：
1. **JSON Schema 自动生成**：从类型注解 + docstring 生成，不手写 schema，避免文档与实现漂移。
2. **参数校验与类型纠偏**：模型经常把 5 写成 "5"，这里做保守纠偏；纠偏不了就返回结构化错误，
   让 Agent 自己看到错误并重试，而不是抛异常炸掉整个循环。
3. **错误即观察（error as observation）**：工具失败不终止 Agent，而是变成一条 observation，
   这是 Agent 工程和普通脚本最大的区别之一。
"""

from __future__ import annotations

import inspect
import json
import types as _types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin, get_type_hints

__all__ = ["Tool", "ToolRegistry", "ToolError", "tool", "function_schema"]


class ToolError(Exception):
    """工具执行或参数校验失败。"""


# --------------------------------------------------------------------------- #
# 类型注解 -> JSON Schema
# --------------------------------------------------------------------------- #

_MISSING = object()

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_schema(annotation: Any) -> dict:
    """把 Python 类型注解翻译成 JSON Schema 片段（够用即可，不求完备）。"""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] / Union[X, None] / (X | None)
    if origin is typing.Union or origin is _types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"anyOf": [_annotation_to_schema(a) for a in non_none]}

    # Literal["a", "b"] -> enum
    if origin is Literal:
        if args and all(isinstance(a, str) for a in args):
            return {"type": "string", "enum": list(args)}
        if args and all(isinstance(a, int) for a in args):
            return {"type": "integer", "enum": list(args)}
        return {"enum": list(args)}

    # list[X] / dict[str, X]
    if origin in (list, typing.List):
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _annotation_to_schema(args[0])
        return schema
    if origin in (dict, typing.Dict):
        return {"type": "object"}

    # 普通标量
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    # 兜底：自定义类（如 Pydantic 模型）交给模型自由发挥
    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """解析 docstring：返回 (总体描述, {参数名: 参数说明})。

    支持 Google 风格：

        Args:
            city: 城市名，例如 "Beijing"
            days (int): 天数
    """
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)

    description_lines: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    current: str | None = None

    stop_sections = {"returns", "return", "yields", "raises", "examples", "example", "note", "notes"}

    for raw in doc.splitlines():
        line = raw.strip()
        lowered = line.rstrip(":").lower()
        if lowered in {"args", "arguments", "parameters", "params"}:
            section = "args"
            current = None
            continue
        if lowered in stop_sections:
            section = None
            current = None
            continue
        if section == "args":
            if not line:
                continue
            # "name: desc" 或 "name (int): desc"
            if ":" in line:
                head, _, tail = line.partition(":")
                name = head.split("(")[0].strip().lstrip("*")
                if name.isidentifier():
                    params[name] = tail.strip()
                    current = name
                    continue
            if current:  # 续行
                params[current] = (params[current] + " " + line).strip()
            continue
        if line:
            description_lines.append(line)

    return " ".join(description_lines).strip(), params


def function_schema(func: Callable[..., Any], *, name: str | None = None,
                    description: str | None = None) -> dict:
    """由函数生成 OpenAI 风格的 function schema。"""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:  # 注解里引用了无法解析的名字时降级
        hints = getattr(func, "__annotations__", {}) or {}

    doc_desc, param_docs = _parse_docstring(func.__doc__)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # *args/**kwargs 不进 schema

        annotation = hints.get(pname, param.annotation)
        prop = _annotation_to_schema(annotation)

        # Annotated[X, "说明"] 的说明优先于 docstring
        meta = getattr(annotation, "__metadata__", None)
        if meta:
            texts = [m for m in meta if isinstance(m, str)]
            if texts:
                prop["description"] = texts[0]
        if "description" not in prop and pname in param_docs:
            prop["description"] = param_docs[pname]

        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            prop["default"] = param.default

    return {
        "name": name or func.__name__,
        "description": description or doc_desc or f"调用 {func.__name__}",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# --------------------------------------------------------------------------- #
# Tool / ToolRegistry
# --------------------------------------------------------------------------- #

@dataclass
class Tool:
    """一个可被 Agent 调用的工具。"""

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    @classmethod
    def from_function(cls, func: Callable[..., Any], *, name: str | None = None,
                      description: str | None = None) -> "Tool":
        # 优先复用 @tool 装饰器算好的 schema，避免重复计算、也保证装饰器参数生效
        cached = getattr(func, "__tool_spec__", None)
        if cached is not None and name is None and description is None:
            return cls(
                name=cached["name"],
                description=cached["description"],
                parameters=cached["parameters"],
                func=func,
            )
        schema = function_schema(func, name=name, description=description)
        return cls(
            name=schema["name"],
            description=schema["description"],
            parameters=schema["parameters"],
            func=func,
        )

    def spec(self) -> dict:
        """OpenAI / DeepSeek 兼容的工具声明格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # -- 参数校验 ---------------------------------------------------------- #

    def _check(self, args: Any) -> dict:
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolError(
                    f"参数不是合法 JSON（收到 {args!r}）：{exc}。请输出形如 {{\"key\": value}} 的 JSON 对象。"
                ) from exc
        if not isinstance(args, dict):
            raise ToolError(f"参数必须是 JSON 对象，收到 {type(args).__name__}")

        props: dict = self.parameters.get("properties", {})
        required: list[str] = self.parameters.get("required", [])

        missing = [r for r in required if r not in args or args[r] is None]
        if missing:
            raise ToolError(
                f"缺少必填参数 {missing}。本工具参数：{json.dumps(props, ensure_ascii=False)}"
            )

        cleaned: dict = {}
        for key, value in args.items():
            if key not in props:
                # 多余参数直接忽略，比报错更宽容（模型偶尔会多塞字段）
                continue
            cleaned[key] = _coerce(value, props[key], key)
        return cleaned

    def run(self, args: Any = None) -> str:
        """执行工具，返回字符串结果；失败抛 ToolError。"""
        cleaned = self._check(args)
        try:
            result = self.func(**cleaned)
        except ToolError:
            raise
        except TypeError as exc:  # 参数签名不匹配
            raise ToolError(f"调用 {self.name} 的参数不匹配：{exc}") from exc
        except Exception as exc:  # noqa: BLE001 - 任何工具内部错误都要变成可读观察
            raise ToolError(f"{self.name} 执行出错：{type(exc).__name__}: {exc}") from exc
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def _coerce(value: Any, schema: dict, key: str) -> Any:
    """保守地把模型给的字符串纠正成 schema 声明的类型。"""
    wanted = schema.get("type")
    if wanted == "integer" and isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            raise ToolError(f"参数 {key} 需要整数，收到 {value!r}") from None
    if wanted == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            raise ToolError(f"参数 {key} 需要数字，收到 {value!r}") from None
    if wanted == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        raise ToolError(f"参数 {key} 需要布尔值，收到 {value!r}")
    if wanted == "array" and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise ToolError(f"参数 {key} 需要数组，收到 {value!r}") from None
        if isinstance(parsed, list):
            return parsed
        raise ToolError(f"参数 {key} 需要数组，收到 {value!r}")
    if wanted == "string" and not isinstance(value, str) and value is not None:
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    if wanted == "integer" and isinstance(value, bool):
        raise ToolError(f"参数 {key} 需要整数，收到布尔值")
    if wanted == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass
class ToolRegistry:
    """工具注册表：统一管理、统一生成 schema、统一执行。"""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool_or_func: Tool | Callable[..., Any], **kwargs: Any) -> Tool:
        if isinstance(tool_or_func, Tool):
            t = tool_or_func
        else:
            t = Tool.from_function(tool_or_func, **kwargs)
        self.tools[t.name] = t
        return t

    # 允许 registry.register(func) 与 registry.add(tool) 两种写法
    add = register

    def __contains__(self, name: object) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, arguments: Any) -> str:
        if name not in self.tools:
            raise ToolError(
                f"不存在名为 {name!r} 的工具。可用工具：{self.names()}"
            )
        return self.tools[name].run(arguments)

    def describe(self) -> str:
        """给文本模式 ReAct 提示词用的人类可读工具清单。"""
        lines = []
        for t in self.tools.values():
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in t.parameters.get("properties", {}).items()
            )
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None,
         description: str | None = None) -> Any:
    """装饰器：把函数标记为工具并挂上 schema。

    用法::

        @tool
        def add(a: int, b: int) -> int:
            \"\"\"两数相加。

            Args:
                a: 第一个加数
                b: 第二个加数
            \"\"\"
            return a + b

        @tool(name="weather", description="查询天气")
        def get_weather(city: str) -> str: ...
    """

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__tool_spec__ = function_schema(f, name=name, description=description)  # type: ignore[attr-defined]
        return f

    if func is not None:
        return wrap(func)
    return wrap
