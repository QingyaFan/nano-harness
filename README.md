# nano-harness

「Agent Harness 工程」系列的配套代码：用最小实现演示一个可验证的 Agent 运行时。

- 系列导读：[从零打造 Agent Harness：实现篇导读](https://github.com/QingyaFan/blog/issues/259)
- 理论篇：[Harness Engineering：从裸循环到可验证运行时](https://github.com/QingyaFan/blog/issues/231)

## 文章与模块

| 编号 | 文章 | 模块 | 状态 |
| --- | --- | --- | --- |
| I1 | [最小循环：让 Agent 跑起来（I1）](https://github.com/QingyaFan/blog/issues/260) | `src/nano_harness/loop.py` | 已实现 |
| I2 | [工具事务：让失败可归因、可重试（I2）](https://github.com/QingyaFan/blog/issues/261) | `src/nano_harness/tools/` | 已实现 |
| I3 | [上下文预算：让该看的留下，丢掉的留痕（I3）](https://github.com/QingyaFan/blog/issues/262) | `src/nano_harness/context.py` | 已实现 |
| I4 | [生命周期：让中断的运行有交代、能接续（I4）](https://github.com/QingyaFan/blog/issues/263) | `src/nano_harness/lifecycle.py` | 已实现 |
| I5 | [权限与沙箱：让否决改变控制流（I5）](https://github.com/QingyaFan/blog/issues/264) | `src/nano_harness/policy/` | 已实现 |
| I6 | [技能：让上下文只为用到的那一层付费（I6）](https://github.com/QingyaFan/blog/issues/265) | `src/nano_harness/skills.py` | 已实现 |
| I7 | [观测与评测：把运行变成可复现读数（I7）](https://github.com/QingyaFan/blog/issues/266) | `src/nano_harness/observability.py` / `eval.py` | 已实现 |
| I8 | [记忆：让跨运行的事实接得上（I8）](https://github.com/QingyaFan/blog/issues/267) | `src/nano_harness/memory.py` | 已实现 |
| I9 | [Prompt 装配：把各层拼成一次请求（I9）](https://github.com/QingyaFan/blog/issues/268) | `src/nano_harness/prompt.py` | 已实现 |
| E1 | [断点续跑实战：一次崩溃恢复的完整时间线（E1）](https://github.com/QingyaFan/blog/issues/269) | `src/nano_harness/lease.py` / `resume.py` | 已实现 |
| E2 | [自进化：从经验沉淀到可回滚发布（E2）](https://github.com/QingyaFan/blog/issues/270) | `src/nano_harness/artifact.py` / `distill.py` / `evolution.py` | 已实现 |
| E3 | [子 Agent：把上下文隔离出去（E3）](https://github.com/QingyaFan/blog/issues/272) | `examples/subagent_review.py` | 示例 |
| E4 | [Prompt Injection：把不可信内容关进数据面（E4）](https://github.com/QingyaFan/blog/issues/273) | `examples/injection_defense.py` | 示例 |

I1–I9、E1 与 E2 的实现已落地，配套 98 个测试；E3 的配套代码是 `examples/subagent_review.py`，E4 的配套代码是 `examples/injection_defense.py`。每个模块只依赖前序模块，接口在对应文章中展开。

## 进阶（可选）

A1–A3 是单 Agent core 之上的编排层，放在 `orchestration/`：它依赖 core，core 不反向依赖它。跨模块边界由 `test_scheduler.py` / `test_graph.py` 的 import 检查守住。

**架构约束**：`src/nano_harness/` 顶层模块即 core；core 不得 import `orchestration`，`orchestration` 只能依赖 core 的公开接口。

| 编号 | 文章 | 模块 | 状态 |
| --- | --- | --- | --- |
| A1 | [Loop Engineering：让循环可信](https://github.com/QingyaFan/blog/issues/254) | `src/nano_harness/orchestration/scheduler.py` | 已实现 |
| A2 | [Graph Engineering：让图可信](https://github.com/QingyaFan/blog/issues/255) | `src/nano_harness/orchestration/graph.py` | 已实现 |
| A3 | [Multi-Agent Engineering：让协作可信](https://github.com/QingyaFan/blog/issues/271) | `src/nano_harness/orchestration/contracts.py` / `agents.py` | 已实现 |

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
  lease.py        E1 run lease 与 fencing
  resume.py       E1 账本重放与 pending 恢复
  artifact.py     E2 产物注册表、状态流转、版本固定
  distill.py      E2 运行轨迹 → 候选、信任与撤销
  evolution.py    E2 评测门、金丝雀、审批、回滚
  orchestration/  A1 外层循环、A2 图执行：节点/边/状态/锚点/checkpoint
tests/
examples/
```

运行示例：

```bash
python examples/full_stack.py
python examples/eval_suite.py
python examples/memory_demo.py
python examples/prompt_demo.py
python examples/resume_crash.py
python examples/evolve_memory.py
python examples/evolve_skill.py
python examples/evolve_policy.py
python examples/loop_scheduler.py
python examples/graph_review.py
python examples/multi_agent_incident.py
python examples/subagent_review.py
python examples/injection_defense.py
```

## 状态

核心 I1–I9、E1 与 E2 已实现，A1 外层调度器、A2 图执行器与 A3 多智能体运行时已落地，E3 提供子 agent 示例、E4 提供注入防御示例与 3 个测试；98 个核心测试、13 个 A1、16 个 A2、19 个 A3、3 个 E4，共 149 个测试，覆盖循环、工具事务、上下文压缩、生命周期、权限沙箱、技能加载、观测评测、记忆读写、Prompt 装配、断点续跑、产物发布、循环调度、图执行、多智能体协作与注入防御。

## License

[MIT](LICENSE)