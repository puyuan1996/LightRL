from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RunContext:
    run_id: str
    run_name: str


class Harness(Protocol):
    def run(self, context: RunContext) -> None: ...


class Algorithm(Protocol):
    def configure(self) -> dict: ...
