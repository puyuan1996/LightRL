from __future__ import annotations

from typing import Any, Protocol


class HarnessFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


class Algorithm(Protocol):
    def configure(self) -> dict[str, Any]: ...
