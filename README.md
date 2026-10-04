# mini-react-agent

> 从零实现的 ReAct Agent 内核 —— **零第三方依赖**，300 行核心代码把 Agent 循环讲透，clone 下来不填 API Key 也能跑。

[![tests](https://github.com/flyingbeez/mini-react-agent/actions/workflows/test.yml/badge.svg)](https://github.com/flyingbeez/mini-react-agent/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/dependencies-0-brightgreen)
![tests](https://img.shields.io/badge/tests-103%20passed-success)

---

## 为什么写这个项目

现在有大量 Agent 框架（LangChain、LangGraph、AutoGen……），它们让"跑起来"变得很容易，
也让"理解为什么能跑起来"变得很难。面试里被问到下面这些问题时，光会用框架是答不上来的：

- 一次 function calling 的 HTTP 请求和响应到底长什么样？
- Agent 的循环什么时候该停？模型不停怎么办？
- 工具报错了，为什么有的 Agent 能自己改对，有的直接崩？
- 文本 ReAct 和原生 tool_calls 有什么区别，各自适合什么场景？

这个仓库把这些问题都写成可运行的代码 + 可断言的测试。

## 30 秒看懂 Agent 循环

```
                    ┌──────────────────────────────────────┐
                    │            用户提问                   │
                    └───────────────────┬──────────────────┘
                                        ▼
              ┌─────────────────────────────────────────────┐
              │            组装 messages[]                  │
              │  system(角色+工具清单) + 历史 + user(问题)    │
              └───────────────────┬─────────────────────────┘
                                  ▼
                    ┌─────────────────────────┐
              ┌────▶│     调用 LLM（一次）     │
              │     └───────────┬─────────────┘
              │                 ▼
              │        ┌────────────────┐
              │        │ 有 tool_calls? │
              │        └───┬────────┬───┘
              │         有 │        │ 没有
              │            ▼        ▼
              │   ┌──────────────┐  ┌────────────────────┐
              │   │ 执行工具      │  │ 返回 content 作为   │
              │   │ 结果转字符串  │  │ 最终答案 → 结束     │
              │   └──────┬───────┘  └────────────────────┘
              │          ▼
              │   ┌──────────────────────────────┐
              │   │ 追加 role=tool 消息           │
              │   │ 死循环检测 / 步数检查          │
              │   └──────┬───────────────────────┘
              └──────────┘  （继续下一轮）
```

**核心就一句话**：让模型在"思考 → 调工具 → 看结果"之间循环，直到它不再调工具、直接给出答案。

## 快速开始

```bash
git clone https://github.com/flyingbeez/mini-react-agent.git
cd mini-react-agent

# 1) 跑测试（不需要任何 API Key，秒级完成）
python run_tests.py          # 103 tests, OK

# 2) 跑离线演示（MockLLM，仍然不需要 API Key）
python examples/01_offline_demo.py

# 3) 接真实模型（可选）
#    Windows:  $env:DEEPSEEK_API_KEY="sk-xxx"
#    Linux/macOS: export DEEPSEEK_API_KEY="sk-xxx"
python examples/02_live_demo.py "帮我算一下 (365*24-100)/7，保留两位小数"
```

## 两种模式，一个循环

| | 原生 Function Calling | 文本 ReAct |
|---|---|---|
| 模型输出 | 结构化 `tool_calls` 字段 | 自由文本 `Action: xxx` |
| 解析稳定性 | 高（模型侧保证格式） | 中（要写各种兜底解析） |
| 兼容性 | 需要模型支持 tools 参数 | 任何模型都能用 |
| 并行调用 | 原生支持一轮多工具 | 需要自己约定 |
| 适合场景 | 生产环境首选 | 教学、老模型、可控性要求高 |

同一个 `ReActAgent` 通过 `mode="native"` / `mode="text"` 切换，两种模式的循环逻辑与轨迹格式完全一致，
方便你对照理解"框架到底帮你做了什么"。

## 代码结构

```
mini_agent/
├── llm.py             LLM 抽象：MockLLM（离线）+ OpenAICompatLLM（标准库 urllib，无 SDK 依赖）
├── tools.py           @tool 装饰器 → 自动生成 JSON Schema；参数校验、类型纠偏、错误包装
├── parser.py          解析模型输出：markdown / 中英文冒号 / 单引号 / 多行 JSON / key=value 全兼容
├── trace.py           轨迹记录：每步的思考、动作、观察、耗时、token —— Agent 可观测性的最小实现
├── agent.py           核心循环：步数上限、死循环检测、工具失败自愈、超时
└── builtin_tools.py   4 个示例工具（安全计算器 / 时间 / 文本统计 / 轻量检索）
examples/              3 个可运行示例
tests/                 103 个单元测试
docs/                  原理讲解 + 代码导读 + 面试问答
```

## 自己写一个工具

加上类型注解和 docstring 就够了，Schema 自动生成：

```python
from mini_agent import tool, ReActAgent, ToolRegistry

@tool
def get_weather(city: str, days: int = 1) -> dict:
    """查询指定城市未来几天的天气。

    Args:
        city: 城市名，例如 "Beijing"
        days: 预报天数，1-7
    """
    return {"city": city, "days": days, "temp_c": 26}

registry = ToolRegistry()
registry.register(get_weather)
print(registry.specs()[0])
# {'type': 'function', 'function': {'name': 'get_weather',
#   'description': '查询指定城市未来几天的天气。',
#   'parameters': {'type': 'object',
#     'properties': {'city': {'type': 'string', 'description': '城市名，例如 "Beijing"'},
#                    'days': {'type': 'integer', 'description': '预报天数，1-7', 'default': 1}},
#     'required': ['city']}}}
```

支持的注解：`str/int/float/bool/list[X]/dict`、`Optional[X]`、`X | None`、`Literal[...]`、`Annotated[X, "说明"]`。

## 实现了哪些"教科书不写但生产必须有"的细节

| 问题 | 本项目的处理 | 对应代码 |
|---|---|---|
| 模型永远不停调工具 | `max_steps` 上限 + 返回已完成的信息摘要 | `agent.py::_loop_native` |
| 模型卡在同一个调用上 | 对 `(工具名, 参数)` 做签名计数：先注入提示词纠正，再强制终止并诊断 | `agent.py::_check_repeat` |
| 模型给的参数不是合法 JSON | 转成 observation 回灌，让模型自己修 | `agent.py::_execute` |
| 模型把 `5` 写成 `"5"` | 保守类型纠偏（只在无歧义时转换） | `tools.py::_coerce` |
| 工具内部抛异常 | 包装成 `[工具错误] ...` 观察，循环不崩 | `tools.py::Tool.run` |
| 工具返回超长结果撑爆上下文 | observation 截断并标注 | `agent.py::_truncate` |
| 模型没按 ReAct 格式输出 | 把格式要求回灌并重试 | `agent.py::_loop_text` |
| 成本不可见 | 每步记录 prompt/completion token，轨迹可导出 JSON | `trace.py` |
| 测试依赖真实 LLM、又慢又贵又不稳定 | `MockLLM` 把模型变成可编程依赖 | `llm.py::MockLLM` |
| 输出是自由文本，难断言 | 结构化 `AgentTrace` + `to_dict()/to_json()` | `trace.py` |

## 和 LangChain 的关系（面试怎么答）

> 这个项目不是要替代 LangChain。它是为了搞清楚 LangChain 内部的循环、工具调用和错误处理是怎么实现的。
> 真正做业务我仍然会用框架，因为生态、可观测性、社区支持更成熟；但遇到"Agent 卡住""工具调用失败""成本失控"
> 这类问题，只有理解底层才能定位。

这是面试里比"我会用 LangChain"更有说服力的表述方式。

## 已知局限（主动说出来，比被人问出来好）

- 没有做流式输出（streaming），长回答时会一直等。
- 没有多轮会话管理，`history` 需要调用方自己维护。
- 文本模式的解析靠正则，遇到极端格式仍可能失败（会走"格式纠正"分支，不会崩）。
- 轻量检索是词频打分，不是向量检索 —— 向量检索是下一个仓库的事。
- 没有做并发执行同一轮内的多个工具调用（目前是顺序执行）。

## 延伸阅读

- 原始论文：*ReAct: Synergizing Reasoning and Acting in Language Models* (Yao et al., 2022) —— [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- [docs/01-ReAct原理与Agent循环.md](docs/01-ReAct原理与Agent循环.md) —— 原理讲解
- [docs/02-代码导读.md](docs/02-代码导读.md) —— 逐模块导读
- [docs/03-面试问答.md](docs/03-面试问答.md) —— 围绕本项目的 15 个面试问题与答法

## License

MIT
