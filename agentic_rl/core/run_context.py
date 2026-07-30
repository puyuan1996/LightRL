from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RuntimeRunContext:
    run_id: str
    run_name: str
    created_at: str

    @classmethod
    def create(cls, prefix: str) -> "RuntimeRunContext":
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        run_id = f"{prefix}_{timestamp}"
        return cls(run_id=run_id, run_name=run_id, created_at=timestamp)
