"""Optional outer-loop orchestration (A1).

The package depends on the core harness; core modules never import it.
"""

from .model import (
    AgentWorker,
    EventTrigger,
    HumanAction,
    HumanDecision,
    IntervalTrigger,
    LoopBudget,
    LoopError,
    LoopSnapshot,
    LoopState,
    ManualTrigger,
    NeverPause,
    NeverStop,
    PauseWhen,
    PauseWhenContains,
    SchedulerResult,
    SchedulerStatus,
    StepResult,
    StepSummary,
    StopDecision,
    StopWhen,
    StopWhenContains,
)
from .scheduler import Scheduler
from .state import LoopStateStore

__all__ = [
    "AgentWorker",
    "EventTrigger",
    "HumanAction",
    "HumanDecision",
    "IntervalTrigger",
    "LoopBudget",
    "LoopError",
    "LoopSnapshot",
    "LoopState",
    "LoopStateStore",
    "ManualTrigger",
    "NeverPause",
    "NeverStop",
    "PauseWhen",
    "PauseWhenContains",
    "Scheduler",
    "SchedulerResult",
    "SchedulerStatus",
    "StepResult",
    "StepSummary",
    "StopDecision",
    "StopWhen",
    "StopWhenContains",
]