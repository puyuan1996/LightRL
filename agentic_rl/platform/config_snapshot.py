from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_snapshot(config: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
