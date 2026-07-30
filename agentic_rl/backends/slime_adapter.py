from __future__ import annotations

from pathlib import Path


def train_entrypoint(repo_root: str | Path) -> str:
    return str(Path(repo_root) / "slime" / "train_async.py")
