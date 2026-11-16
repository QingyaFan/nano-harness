from pathlib import Path

import pytest

from nano_harness.orchestration import (
    ErrorPolicy,
    Graph,
    GraphCheckpointStore,
    GraphConfig,
    GraphError,
    GraphStatus,
    MergeRule,
    NodeResult,
    merge_state,
)


def test_linear_graph_merges_state() -> None:
    graph = Graph("linear")
    graph.add_node(
        "start",
        lambda state: NodeResult.ok_result("started", updates={"count": 1}),
    )
    graph.add_node(
        "finish",
        lambda state: NodeResult.ok_result(
            "done", updates={"count": state["count"] + 1}
        ),
    )
    graph.add_edge("start", "finish")
    graph.set_entry("start")

    result = graph.run()

    assert result.status is GraphStatus.COMPLETED
    assert result.state["count"] == 2
    assert result.outputs == {"start": "started", "finish": "done"}
    assert result.completed == ("start", "finish")
    assert result.steps == 2


def test_conditional_edges_follow_state() -> None:
    graph = Graph("conditional")
    graph.add_node(
        "route",
        lambda state: NodeResult.ok_result(updates={"route": state["route"]}),
    )
    graph.add_node("left", lambda state: NodeResult.ok_result("left", updates={"hit": "left"}))
    graph.add_node("right", lambda state: NodeResult.ok_result("right", updates={"hit": "right"}))
    graph.add_edge("route", "left", when=lambda state: state["route"] == "left")
    graph.add_edge("route", "right", when=lambda state: state["route"] == "right")
    graph.set_entry("route")

    result = graph.run({"route": "right"})

    assert result.status is GraphStatus.COMPLETED
    assert result.completed == ("route", "right")
    assert result.state["hit"] == "right"
    assert "left" not in result.outputs


def test_fanout_and_append_merge() -> None:
    graph = Graph(
        "fanout",
        config=GraphConfig(max_workers=2),
        merge_rules={"findings": MergeRule.APPEND},
    )
    graph.add_node(
        "split",
        lambda state: NodeResult.ok_result(
            updates={"findings": []}, next_nodes=("a", "b")
        ),
    )
    graph.add_node(
        "a",
        lambda state: NodeResult.ok_result("a", updates={"findings": ["a"]}),
    )
    graph.add_node(
        "b",
        lambda state: NodeResult.ok_result("b", updates={"findings": ["b"]}),
    )
    graph.add_node(
        "reduce",
        lambda state: NodeResult.ok_result(
            {"count": len(state["findings"])},
            updates={"reduced": len(state["findings"])},
        ),
    )
    graph.add_edge("a", "reduce")
    graph.add_edge("b", "reduce")
    graph.set_entry("split")

    result = graph.run()

    assert result.status is GraphStatus.COMPLETED
    assert result.state["findings"] == ["a", "b"]
    assert result.state["reduced"] == 2
    assert result.completed.count("reduce") == 1
    assert result.outputs["reduce"] == {"count": 2}


def test_missing_input_fails_closed() -> None:
    graph = Graph("missing")
    graph.add_node(
        "needs_value",
        lambda state: NodeResult.ok_result("ok"),
        requires=("value",),
    )

    result = graph.run()

    assert result.status is GraphStatus.FAILED
    assert result.reason == "needs_value:missing_input:value"
    assert result.steps == 1


def test_partial_graph_keeps_successful_branch() -> None:
    graph = Graph(
        "partial",
        config=GraphConfig(on_error=ErrorPolicy.CONTINUE, max_workers=2),
    )
    graph.add_node(
        "split",
        lambda state: NodeResult.ok_result(next_nodes=("bad", "good")),
    )
    graph.add_node("bad", lambda state: NodeResult.failure("boom"))
    graph.add_node(
        "good",
        lambda state: NodeResult.ok_result("good", updates={"good": True}),
    )
    graph.set_entry("split")

    result = graph.run()

    assert result.status is GraphStatus.PARTIAL
    assert result.state["good"] is True
    assert result.failures == ("bad:boom",)
    assert result.completed == ("split", "bad", "good")


def test_step_budget_and_checkpoint_resume() -> None:
    graph = Graph("resume", config=GraphConfig(max_steps=1))
    graph.add_node("a", lambda state: NodeResult.ok_result("a", updates={"a": True}))
    graph.add_node("b", lambda state: NodeResult.ok_result("b", updates={"b": True}))
    graph.add_edge("a", "b")
    graph.set_entry("a")

    paused = graph.run()
    assert paused.status is GraphStatus.MAX_STEPS
    assert paused.checkpoint is not None
    assert paused.checkpoint.frontier == ("b",)

    resumed_graph = Graph("resume", config=GraphConfig(max_steps=10))
    resumed_graph.add_node("a", lambda state: NodeResult.ok_result("a"))
    resumed_graph.add_node(
        "b", lambda state: NodeResult.ok_result("b", updates={"b": True})
    )
    resumed = resumed_graph.run(checkpoint=paused.checkpoint)
    assert resumed.status is GraphStatus.COMPLETED
    assert resumed.steps == 2
    assert resumed.state == {"a": True, "b": True}


def test_checkpoint_store_persists_across_instances(tmp_path) -> None:
    store = GraphCheckpointStore(tmp_path)
    graph = Graph("stored", config=GraphConfig(max_steps=1), checkpoint_store=store)
    graph.add_node("a", lambda state: NodeResult.ok_result("a", next_nodes=("b",)))
    graph.add_node("b", lambda state: NodeResult.ok_result("b", updates={"done": True}))
    graph.set_entry("a")

    paused = graph.run()
    assert paused.status is GraphStatus.MAX_STEPS
    checkpoint = store.load("stored")
    assert checkpoint is not None
    assert checkpoint.frontier == ("b",)

    resumed_graph = Graph(
        "stored", config=GraphConfig(max_steps=10), checkpoint_store=store
    )
    resumed_graph.add_node("a", lambda state: NodeResult.ok_result("a"))
    resumed_graph.add_node(
        "b", lambda state: NodeResult.ok_result("b", updates={"done": True})
    )
    resumed = resumed_graph.run(checkpoint=checkpoint)
    assert resumed.status is GraphStatus.COMPLETED
    assert resumed.state["done"] is True


def test_fanout_limit_stops_before_running_children() -> None:
    graph = Graph("wide", config=GraphConfig(max_fanout=2))
    graph.add_node(
        "split",
        lambda state: NodeResult.ok_result(next_nodes=("a", "b", "c")),
    )
    graph.add_node("a", lambda state: NodeResult.ok_result("a"))
    graph.add_node("b", lambda state: NodeResult.ok_result("b"))
    graph.add_node("c", lambda state: NodeResult.ok_result("c"))

    result = graph.run()

    assert result.status is GraphStatus.FANOUT_EXCEEDED
    assert result.completed == ("split",)
    assert result.checkpoint is not None
    assert result.checkpoint.frontier == ("a", "b", "c")


def test_anchor_must_pass_before_completion() -> None:
    graph = Graph(
        "anchored",
        anchor=lambda state: None if state.get("tests_passed") else "tests missing",
    )
    graph.add_node(
        "work",
        lambda state: NodeResult.ok_result(updates={"finished": True}),
    )

    failed = graph.run({"tests_passed": False})
    passed = graph.run({"tests_passed": True})

    assert failed.status is GraphStatus.ANCHOR_FAILED
    assert failed.anchor_reason == "tests missing"
    assert passed.status is GraphStatus.COMPLETED


def test_cancel_before_first_node() -> None:
    class Cancelled:
        cancelled = True

    graph = Graph("cancel")
    graph.add_node("a", lambda state: NodeResult.ok_result("a"))

    result = graph.run(cancel=Cancelled())

    assert result.status is GraphStatus.CANCELLED
    assert result.steps == 0


def test_unknown_node_fails() -> None:
    graph = Graph("unknown")
    graph.add_node("a", lambda state: NodeResult.ok_result(next_nodes=("missing",)))

    result = graph.run()

    assert result.status is GraphStatus.FAILED
    assert result.reason == "unknown_node:missing"


def test_edge_error_fails_graph() -> None:
    graph = Graph("edge-error")
    graph.add_node("a", lambda state: NodeResult.ok_result("a"))
    graph.add_node("b", lambda state: NodeResult.ok_result("b"))
    graph.add_edge("a", "b", when=lambda state: 1 / 0)

    result = graph.run()

    assert result.status is GraphStatus.FAILED
    assert result.reason.startswith("edge_error:a:")


def test_merge_state_rules() -> None:
    assert merge_state({"items": [1]}, {"items": [2]}, {"items": MergeRule.APPEND}) == {
        "items": [1, 2]
    }
    assert merge_state({"n": 1}, {"n": 2}, {"n": MergeRule.SUM}) == {"n": 3}
    assert merge_state({"n": 1}, {"n": 2}) == {"n": 2}


def test_checkpoint_from_other_graph_is_rejected() -> None:
    graph = Graph("a")
    graph.add_node("a", lambda state: NodeResult.ok_result("a"))
    other = Graph("b")
    other.add_node("b", lambda state: NodeResult.ok_result("b"))
    checkpoint = graph.run().checkpoint
    assert checkpoint is not None

    with pytest.raises(GraphError):
        other.run(checkpoint=checkpoint)


def test_graph_requires_nodes() -> None:
    with pytest.raises(GraphError):
        Graph("empty").run()


def test_core_does_not_import_graph() -> None:
    core = Path(__file__).resolve().parents[1] / "src" / "nano_harness"
    for path in core.rglob("*.py"):
        if "orchestration" in path.parts:
            continue
        assert "orchestration" not in path.read_text(encoding="utf-8").lower(), path