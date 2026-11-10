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

I1–I6 的实现已落地，配套 50 个测试；每个模块只依赖前序模块，接口在对应文章中展开。

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
tests/
examples/
```

运行示例：

```bash
python examples/full_stack.py
```

## 状态

I1–I6 已实现，50 个测试覆盖循环、工具事务、上下文压缩、生命周期、权限沙箱和技能加载。

## License

[MIT](LICENSE)