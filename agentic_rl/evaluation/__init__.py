"""Evaluation adapters and official-format result exporters."""

from .swebench import build_prediction_coverage, write_official_artifacts

__all__ = ["build_prediction_coverage", "write_official_artifacts"]
