"""Composable math RLVR data, verification, scoring and statistics helpers.

The package deliberately has no model-serving dependency.  ``eval_math.py``
uses it for local scoring, while the training custom RM imports the exact same
``Verifier`` implementation.
"""

from .data import MathExample, data_root_candidates, deduplicate_rows, load_dataset, resolve_data_root, write_manifest
from .extractor import AnswerCandidate, AnswerExtractor, ExtractionResult, extract_answer, extract_answers
from .scorer import ScoreConfig, score_group, score_sample, summarize
from .verifier import VerificationResult, Verifier, verifier_digest

__all__ = [
    "AnswerCandidate",
    "AnswerExtractor",
    "ExtractionResult",
    "MathExample",
    "ScoreConfig",
    "VerificationResult",
    "Verifier",
    "data_root_candidates",
    "deduplicate_rows",
    "extract_answer",
    "extract_answers",
    "load_dataset",
    "resolve_data_root",
    "score_group",
    "score_sample",
    "summarize",
    "verifier_digest",
    "write_manifest",
]
