"""Composable offline evaluation toolkit.

The supported entry point is ``python -m tools.evaluation`` (or the
``lightrl-eval`` console script after installation).  The historical
``python tools/evaluation/eval_cli.py`` form remains supported.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    """Dispatch to the CLI lazily so importing the package stays lightweight."""

    from .eval_cli import main as _main

    return _main()
