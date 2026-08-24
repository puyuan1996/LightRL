"""Append SETA harness support packages without shadowing the image GPU stack.

This module is opt-in through ``LIGHTRL_RUNTIME_HARNESS_OVERLAY``.  Python
loads ``sitecustomize`` after its normal site-packages have been configured,
so appending here lets missing pure-Python packages (terminal-bench/CAMEL) fall
back to a prepared runtime while preserving the RJob image's torch, Ray and
SGLang builds.
"""

from __future__ import annotations

import os
import importlib.machinery
import sys
from pathlib import Path


_support_paths = tuple(
    str(Path(raw_path).resolve())
    for raw_path in os.environ.get("LIGHTRL_SETA_SUPPORT_SITE_PATHS", "").split(os.pathsep)
    if raw_path and Path(raw_path).is_dir()
)
_fallback_paths = tuple(
    str(Path(raw_path).resolve())
    for raw_path in os.environ.get("LIGHTRL_SETA_FALLBACK_SITE_PATHS", "").split(os.pathsep)
    if raw_path and Path(raw_path).is_dir()
)
_fallback_modules = frozenset(
    name.strip()
    for name in os.environ.get("LIGHTRL_SETA_FALLBACK_MODULES", "jmespath").split(",")
    if name.strip()
)


if _support_paths:
    # This prepared site is checked to contain no GPU stack. Appending it makes
    # package metadata visible while preserving every normal image path.
    for support_path in _support_paths:
        if support_path not in sys.path:
            sys.path.append(support_path)


class _AllowlistedFallbackFinder:
    """Resolve a tiny pure-Python allowlist from a broader conda runtime."""

    @staticmethod
    def find_spec(fullname: str, path=None, target=None):  # noqa: ANN001
        if path is not None or fullname not in _fallback_modules:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, _fallback_paths, target)


if _fallback_paths and _fallback_modules:
    sys.meta_path.append(_AllowlistedFallbackFinder())
