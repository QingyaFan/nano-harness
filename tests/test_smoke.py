import nano_harness
from nano_harness.loop import StopReason


def test_version() -> None:
    assert nano_harness.__version__ == "0.1.0"


def test_stop_reason_values() -> None:
    assert StopReason.COMPLETED == "completed"
    assert StopReason.NO_PROGRESS == "no_progress"