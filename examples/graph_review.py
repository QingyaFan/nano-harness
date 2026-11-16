"""A2 graph demo: fanout, append merge, reduce, verify, and anchor."""

from nano_harness.orchestration import Graph, GraphConfig, MergeRule, NodeResult


def reviewer(name: str):
    def run(state) -> NodeResult:
        return NodeResult.ok_result(
            f"{name}: 1 finding",
            updates={"findings": [name]},
        )

    return run


def main() -> None:
    graph = Graph(
        "code-review",
        config=GraphConfig(max_workers=3, max_fanout=4),
        merge_rules={"findings": MergeRule.APPEND},
        anchor=lambda state: None
        if state.get("tests_passed")
        else "tests have not passed",
    )
    graph.add_node(
        "split",
        lambda state: NodeResult.ok_result(
            next_nodes=("review_a", "review_b", "review_c"),
            updates={"findings": []},
        ),
    )
    for name in ("review_a", "review_b", "review_c"):
        graph.add_node(name, reviewer(name))
        graph.add_edge(name, "reduce")
    graph.add_node(
        "reduce",
        lambda state: NodeResult.ok_result(
            {"count": len(state["findings"])},
            updates={"summary": f"{len(state['findings'])} findings"},
        ),
        requires=("findings",),
    )
    graph.add_node(
        "verify",
        lambda state: NodeResult.ok_result(
            updates={"verified": True, "tests_passed": True}
        ),
        requires=("summary",),
    )
    graph.add_edge("reduce", "verify")
    graph.set_entry("split")

    result = graph.run()

    print("status:", result.status.value)
    print("completed:", result.completed)
    print("findings:", result.state["findings"])
    print("summary:", result.state["summary"])
    print("anchor:", result.anchor_reason or "passed")
    print("events:", [event.type for event in result.events])


if __name__ == "__main__":
    main()