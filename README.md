# nano-harness

「Agent Harness 工程」系列的配套代码：用最小实现演示一个可验证的 Agent 运行时。

- 系列导读：[从零打造 Agent Harness：实现篇导读](https://github.com/QingyaFan/blog/issues/259)
- 理论篇：[Harness Engineering：从裸循环到可验证运行时](https://github.com/QingyaFan/blog/issues/231)

## 文章与模块

| 编号 | 文章 | 模块 | 状态 |
| --- | --- | --- | --- |
| I1 | [最小循环：让 Agent 跑起来](https://github.com/QingyaFan/blog/issues/260) | `src/nano_harness/loop.py` | 已实现 |
| I2 | [工具事务：让失败可归因、可重试](https://github.com/QingyaFan/blog/issues/261) | `src/nano_harness/tools/` | 已实现 |
| I3 | [上下文预算：让该看的留下，丢掉的留痕](https://github.com/QingyaFan/blog/issues/262) | `src/nano_harness/context.py` | 已实现 |
| I4 | [生命周期：让中断的运行有交代、能接续](https://github.com/QingyaFan/blog/issues/263) | `src/nano_harness/lifecycle.py` | 已实现 |
| I5 | [权限与沙箱：让否决改变控制流](https://github.com/QingyaFan/blog/issues/264) | `src/nano_harness/policy/` | 已实现 |
| I6 | [技能：让上下文只为用到的那一层付费](https://github.com/QingyaFan/blog/issues/265) | `src/nano_harness/skills.py` | 已实现 |
| I7 | [观测与评测：把运行变成可复现读数](https://github.com/QingyaFan/blog/issues/266) | `src/nano_harness/observability.py` / `eval.py` | 已实现 |
| I8 | [记忆：让跨运行的事实接得上](https://github.com/QingyaFan/blog/issues/267) | `src/nano_harness/memory.py` | 已实现 |
| I9 | [Prompt 装配：把各层拼成一次请求](https://github.com/QingyaFan/blog/issues/268) | `src/nano_harness/prompt.py` | 已实现 |

I1–I9 的实现已落地，配套 82 个测试；每个模块只依赖前序模块，接口在对应文章中展开。

## 环境

- Python 3.11.13，见 `.python-version`
- 依赖见 `pyproject.toml`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## 结构

```text
src/nano_harness/
  messages.py     消息与工具调用
  events.py       事件流
  model.py        Model 协议与 ScriptedModel
  loop.py         I1 最小循环
  tools/          I2 工具规格、错误信封、执行器
  context.py      I3 token 计量、压缩级联、artifact
  lifecycle.py    I4 状态机、检查点、SSE
  policy/         I5 策略、审批、工作区、沙箱、凭证
  skills.py       I6 技能解析、索引、两级加载
  observability.py I7 事件 → span、指标、transcript
  eval.py         I7 fixture 评测、pass^k、回归门
  memory.py       I8 记忆条目、写入/召回/遗忘、隔离
  prompt.py       I9 段装配、信任边界、wire 断言
tests/
examples/
```

运行示例：

```bash
python examples/full_stack.py
python examples/eval_suite.py
python examples/memory_demo.py
python examples/prompt_demo.py
```

## 状态

I1–I9 已实现，82 个测试覆盖循环、工具事务、上下文压缩、生命周期、权限沙箱、技能加载、观测评测、记忆读写与 Prompt 装配。

## License

[MIT](LICENSE)