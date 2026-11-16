"""JSON persistence for the A1 scheduler state."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .model import LoopState

_SAFE_LOOP_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class LoopStateStore:
    """Persist one JSON file per loop, or keep states in memory."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self._directory = Path(directory) if directory is not None else None
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, LoopState] = {}

    def path(self, loop_id: str) -> Path:
        self._validate(loop_id)
        if self._directory is None:
            raise ValueError("state store has no directory")
        return self._directory / f"{loop_id}.json"

    def save(self, state: LoopState) -> None:
        self._validate(state.loop_id)
        if self._directory is None:
            self._memory[state.loop_id] = state
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        target = self.path(state.loop_id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def load(self, loop_id: str) -> LoopState | None:
        self._validate(loop_id)
        if self._directory is None:
            return self._memory.get(loop_id)
        target = self.path(loop_id)
        if not target.exists():
            return None
        return LoopState.from_dict(json.loads(target.read_text(encoding="utf-8")))

    def delete(self, loop_id: str) -> bool:
        self._validate(loop_id)
        if self._directory is None:
            return self._memory.pop(loop_id, None) is not None
        target = self.path(loop_id)
        if not target.exists():
            return False
        target.unlink()
        return True

    @staticmethod
    def _validate(loop_id: str) -> None:
        if not _SAFE_LOOP_ID.match(loop_id):
            raise ValueError(
                "loop_id may contain only letters, digits, dot, dash and underscore"
            )