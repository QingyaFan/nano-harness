"""A2 graph executor: nodes, edges, shared state, fanout, anchors, checkpoints."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..events import EventLog

GRAPH_STARTED = "graph_started"
GRAPH_NODE_STARTED = "graph_node_started"
GRAPH_NODE_FINISHED = "graph_node_finished"
GRAPH_FANOUT = "graph_fanout"
GRAPH_ANCHOR = "graph_anchor"
GRAPH_CHECKPOINT = "graph_checkpoint"
GRAPH_FINISHED = "graph_finished"

_SAFE_GRAPH_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class GraphError(Exception):
    pass


class GraphStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    MAX_STEPS = "max_steps"
    FANOUT_EXCEEDED = "fanout_exceeded"
    ANCHOR_FAILED = "anchor_failed"
    CANCELLED = "cancelled"


class ErrorPolicy(StrEnum):
    FAIL = "fail"
    CONTINUE = "continue"


class MergeRule(StrEnum):
    REPLACE = "replace"
    APPEND = "append"
    SUM = "sum"


@dataclass(frozen=True)
class NodeResult:
    ok: bool = True
    output: Any = None
    updates: dict[str, Any] = field(default_factory=dict)
    next_nodes: tuple[str, ...] | None = None
    error: str | None = None

    @classmethod
    def ok_result(
        cls,
        output: Any = None,
        *,
        updates: dict[str, Any] | None = None,
        next_nodes: tuple[str, ...] | None = None,
    ) -> NodeResult:
        return cls(
            ok=True,
            output=output,
            updates=dict(updates or {}),
            next_nodes=next_nodes,
        )

    @classmethod
    def failure(cls, error: str, *, output: Any = None) -> NodeResult:
        return cls(ok=False, output=output, error=error)


@dataclass(frozen=True)
class Node:
    name: str
    run: Callable[[dict[str, Any]], NodeResult]
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    when: Callable[[dict[str, Any]], bool] | None = None


@dataclass(frozen=True)
class GraphConfig:
    max_steps: int = 100
    max_fanout: int = 8
    max_workers: int = 4
    on_error: ErrorPolicy = ErrorPolicy.FAIL

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_fanout < 1:
            raise ValueError("max_fanout must be at least 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if not isinstance(self.on_error, ErrorPolicy):
            object.__setattr__(self, "on_error", ErrorPolicy(self.on_error))


@dataclass(frozen=True)
class GraphCheckpoint:
    graph_name: str
    step: int
    frontier: tuple[str, ...]
    state: dict[str, Any]
    completed: tuple[str, ...]
    outputs: dict[str, Any]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "step": self.step,
            "frontier": list(self.frontier),
            "state": dict(self.state),
            "completed": list(self.completed),
            "outputs": dict(self.outputs),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphCheckpoint:
        return cls(
            graph_name=str(payload["graph_name"]),
            step=int(payload.get("step", 0)),
            frontier=tuple(payload.get("frontier") or ()),
            state=dict(payload.get("state") or {}),
            completed=tuple(payload.get("completed") or ()),
            outputs=dict(payload.get("outputs") or {}),
            failures=tuple(payload.get("failures") or ()),
        )


class GraphCheckpointStore:
    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self._directory = Path(directory) if directory is not None else None
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, GraphCheckpoint] = {}

    def path(self, graph_name: str) -> Path:
        self._validate(graph_name)
        if self._directory is None:
            raise ValueError("checkpoint store has no directory")
        return self._directory / f"{graph_name}.json"

    def save(self, checkpoint: GraphCheckpoint) -> None:
        self._validate(checkpoint.graph_name)
        if self._directory is None:
            self._memory[checkpoint.graph_name] = checkpoint
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        target = self.path(checkpoint.graph_name)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                checkpoint.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def load(self, graph_name: str) -> GraphCheckpoint | None:
        self._validate(graph_name)
        if self._directory is None:
            return self._memory.get(graph_name)
        target = self.path(graph_name)
        if not target.exists():
            return None
        return GraphCheckpoint.from_dict(
            json.loads(target.read_text(encoding="utf-8"))
        )

    def delete(self, graph_name: str) -> bool:
        self._validate(graph_name)
        if self._directory is None:
            return self._memory.pop(graph_name, None) is not None
        target = self.path(graph_name)
        if not target.exists():
            return False
        target.unlink()
        return True

    @staticmethod
    def _validate(graph_name: str) -> None:
        if not _SAFE_GRAPH_NAME.match(graph_name):
            raise ValueError(
                "graph_name may contain only letters, digits, dot, dash and underscore"
            )


@dataclass(frozen=True)
class GraphResult:
    graph_name: str
    status: GraphStatus
    state: dict[str, Any]
    outputs: dict[str, Any]
    steps: int
    completed: tuple[str, ...]
    failures: tuple[str, ...]
    events: tuple[Any, ...]
    reason: str = ""
    anchor_reason: str = ""
    checkpoint: GraphCheckpoint | None = None


class Graph:
    def __init__(
        self,
        name: str,
        *,
        config: GraphConfig | None = None,
        merge_rules: dict[str, MergeRule | str] | None = None,
        anchor: Callable[[dict[str, Any]], str | None] | None = None,
        checkpoint_store: GraphCheckpointStore | None = None,
        events: EventLog | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("graph name must not be empty")
        self._name = name
        self._config = config or GraphConfig()
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._entry: str | None = None
        self._merge_rules: dict[str, MergeRule] = {
            key: MergeRule(rule) for key, rule in (merge_rules or {}).items()
        }
        self._anchor = anchor
        self._store = checkpoint_store
        self._events = events

    @property
    def name(self) -> str:
        return self._name

    def add_node(
        self,
        name: str,
        run: Callable[[dict[str, Any]], NodeResult],
        *,
        requires: tuple[str, ...] = (),
    ) -> None:
        if not name.strip():
            raise ValueError("node name must not be empty")
        if name in self._nodes:
            raise GraphError(f"duplicate node: {name}")
        self._nodes[name] = Node(name, run, requires)

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        when: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        if not source or not target:
            raise ValueError("edge source and target must not be empty")
        self._edges.append(Edge(source, target, when))

    def set_entry(self, name: str) -> None:
        if name not in self._nodes:
            raise GraphError(f"unknown entry node: {name}")
        self._entry = name

    def run(
        self,
        initial_state: dict[str, Any] | None = None,
        *,
        cancel: Any = None,
        checkpoint: GraphCheckpoint | None = None,
    ) -> GraphResult:
        if not self._nodes:
            raise GraphError("graph has no nodes")

        if checkpoint is not None:
            if checkpoint.graph_name != self._name:
                raise GraphError("checkpoint belongs to another graph")
            state = dict(checkpoint.state)
            frontier = list(checkpoint.frontier)
            completed = list(checkpoint.completed)
            outputs = dict(checkpoint.outputs)
            failures = list(checkpoint.failures)
            steps = checkpoint.step
        else:
            entry = self._entry or next(iter(self._nodes))
            state = dict(initial_state or {})
            frontier = [entry]
            completed = []
            outputs = {}
            failures = []
            steps = 0

        log = self._events or EventLog()
        log.emit(
            GRAPH_STARTED,
            steps,
            graph=self._name,
            frontier=list(frontier),
        )
        return self._drive(
            state, frontier, completed, outputs, failures, steps, log, cancel
        )

    def _drive(
        self,
        state: dict[str, Any],
        frontier: list[str],
        completed: list[str],
        outputs: dict[str, Any],
        failures: list[str],
        steps: int,
        log: EventLog,
        cancel: Any,
    ) -> GraphResult:
        while frontier:
            if _cancelled(cancel):
                return self._finish(
                    GraphStatus.CANCELLED,
                    state,
                    outputs,
                    completed,
                    failures,
                    steps,
                    frontier,
                    log,
                    reason="cancelled",
                )
            if len(frontier) > self._config.max_fanout:
                return self._finish(
                    GraphStatus.FANOUT_EXCEEDED,
                    state,
                    outputs,
                    completed,
                    failures,
                    steps,
                    frontier,
                    log,
                    reason="fanout_exceeded",
                )
            if steps + len(frontier) > self._config.max_steps:
                return self._finish(
                    GraphStatus.MAX_STEPS,
                    state,
                    outputs,
                    completed,
                    failures,
                    steps,
                    frontier,
                    log,
                    reason="step_budget",
                )

            unknown = [name for name in frontier if name not in self._nodes]
            if unknown:
                return self._finish(
                    GraphStatus.FAILED,
                    state,
                    outputs,
                    completed,
                    failures,
                    steps,
                    frontier,
                    log,
                    reason=f"unknown_node:{unknown[0]}",
                )

            log.emit(
                GRAPH_FANOUT,
                steps,
                graph=self._name,
                nodes=list(frontier),
                width=len(frontier),
            )
            for name in frontier:
                log.emit(GRAPH_NODE_STARTED, steps, graph=self._name, node=name)
            results = self._execute_frontier(frontier, state)

            new_failures = []
            for name, result in zip(frontier, results):
                steps += 1
                completed.append(name)
                outputs[name] = result.output
                log.emit(
                    GRAPH_NODE_FINISHED,
                    steps,
                    graph=self._name,
                    node=name,
                    ok=result.ok,
                    error=result.error,
                )
                if result.ok:
                    state = merge_state(state, result.updates, self._merge_rules)
                else:
                    failure = f"{name}:{result.error or 'failed'}"
                    failures.append(failure)
                    new_failures.append(failure)

            if new_failures and self._config.on_error is ErrorPolicy.FAIL:
                return self._finish(
                    GraphStatus.FAILED,
                    state,
                    outputs,
                    completed,
                    failures,
                    steps,
                    [],
                    log,
                    reason=new_failures[0],
                )

            next_frontier: list[str] = []
            edge_error: str | None = None
            for name, result in zip(frontier, results):
                if not result.ok:
                    continue
                try:
                    candidates = self._next_nodes(name, result, state)
                except Exception as exc:  # noqa: BLE001
                    edge_error = f"{name}:{type(exc).__name__}"
                    break
                for target in candidates:
                    if target not in next_frontier:
                        next_frontier.append(target)

            if edge_error:
                return self._finish(
                    GraphStatus.FAILED,
                    state,
                    outputs,
                    completed,
                    failures,
                    steps,
                    [],
                    log,
                    reason=f"edge_error:{edge_error}",
                )

            frontier = next_frontier
            if self._store is not None:
                checkpoint = self._checkpoint(
                    frontier, state, completed, outputs, failures, steps
                )
                self._store.save(checkpoint)
                log.emit(
                    GRAPH_CHECKPOINT,
                    steps,
                    graph=self._name,
                    frontier=list(frontier),
                )

        anchor_reason = ""
        if self._anchor is not None:
            try:
                anchor_reason = self._anchor(dict(state)) or ""
            except Exception as exc:  # noqa: BLE001
                anchor_reason = f"anchor_error:{type(exc).__name__}"
            log.emit(
                GRAPH_ANCHOR,
                steps,
                graph=self._name,
                passed=not anchor_reason,
                reason=anchor_reason,
            )
            if anchor_reason:
                return self._finish(
                    GraphStatus.ANCHOR_FAILED,
                    state,
                    outputs,
                    completed,
                    failures,
                    steps,
                    [],
                    log,
                    reason="anchor_failed",
                    anchor_reason=anchor_reason,
                )

        status = GraphStatus.PARTIAL if failures else GraphStatus.COMPLETED
        return self._finish(
            status,
            state,
            outputs,
            completed,
            failures,
            steps,
            [],
            log,
            reason="failures" if failures else "goal",
        )

    def _execute_frontier(
        self, frontier: list[str], state: dict[str, Any]
    ) -> list[NodeResult]:
        if len(frontier) == 1 or self._config.max_workers == 1:
            return [self._run_node(self._nodes[name], state) for name in frontier]
        workers = min(self._config.max_workers, len(frontier))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(
                pool.map(lambda name: self._run_node(self._nodes[name], state), frontier)
            )

    @staticmethod
    def _run_node(node: Node, state: dict[str, Any]) -> NodeResult:
        missing = [key for key in node.requires if key not in state]
        if missing:
            return NodeResult.failure(f"missing_input:{','.join(missing)}")
        try:
            result = node.run(dict(state))
        except Exception as exc:  # noqa: BLE001
            return NodeResult.failure(f"{type(exc).__name__}: {exc}")
        if not isinstance(result, NodeResult):
            return NodeResult.failure("invalid_result")
        if not isinstance(result.updates, dict):
            return NodeResult.failure("invalid_updates")
        return result

    def _next_nodes(
        self, name: str, result: NodeResult, state: dict[str, Any]
    ) -> list[str]:
        if result.next_nodes is not None:
            return list(result.next_nodes)
        targets = []
        for edge in self._edges:
            if edge.source != name:
                continue
            if edge.when is None or edge.when(dict(state)):
                targets.append(edge.target)
        return targets

    def _checkpoint(
        self,
        frontier: list[str],
        state: dict[str, Any],
        completed: list[str],
        outputs: dict[str, Any],
        failures: list[str],
        steps: int,
    ) -> GraphCheckpoint:
        return GraphCheckpoint(
            graph_name=self._name,
            step=steps,
            frontier=tuple(frontier),
            state=dict(state),
            completed=tuple(completed),
            outputs=dict(outputs),
            failures=tuple(failures),
        )

    def _finish(
        self,
        status: GraphStatus,
        state: dict[str, Any],
        outputs: dict[str, Any],
        completed: list[str],
        failures: list[str],
        steps: int,
        frontier: list[str],
        log: EventLog,
        *,
        reason: str,
        anchor_reason: str = "",
    ) -> GraphResult:
        checkpoint = self._checkpoint(frontier, state, completed, outputs, failures, steps)
        if self._store is not None:
            self._store.save(checkpoint)
        log.emit(
            GRAPH_FINISHED,
            steps,
            graph=self._name,
            status=status.value,
            reason=reason,
            failures=list(failures),
        )
        return GraphResult(
            graph_name=self._name,
            status=status,
            state=dict(state),
            outputs=dict(outputs),
            steps=steps,
            completed=tuple(completed),
            failures=tuple(failures),
            events=log.snapshot(),
            reason=reason,
            anchor_reason=anchor_reason,
            checkpoint=checkpoint,
        )


def merge_state(
    state: dict[str, Any],
    updates: dict[str, Any],
    rules: dict[str, MergeRule] | None = None,
) -> dict[str, Any]:
    merged = dict(state)
    configured = rules or {}
    for key, value in updates.items():
        rule = configured.get(key, MergeRule.REPLACE)
        if rule is MergeRule.APPEND:
            current = list(merged.get(key) or [])
            if isinstance(value, list):
                current.extend(value)
            else:
                current.append(value)
            merged[key] = current
        elif rule is MergeRule.SUM:
            merged[key] = merged.get(key, 0) + value
        else:
            merged[key] = value
    return merged


def _cancelled(token: Any) -> bool:
    return bool(token is not None and getattr(token, "cancelled", False))