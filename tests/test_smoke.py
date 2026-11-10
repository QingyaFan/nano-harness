import harness
from harness.loop import StopReason


def test_version() -> None:
    assert harness.__version__ == "0.1.0"


def test_stop_reason_values() -> None:
    assert StopReason.COMPLETED == "completed"
    assert StopReason.NO_PROGRESS == "no_progress"