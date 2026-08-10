#!/usr/bin/env python3
"""Recompute derived metrics from stored per-sample records and patch the summaries.

No generation happens here — it reads `*.detail.json` and adds fields that were
not in the first version of the harness, so every number quoted in the report is
reproducible from the shipped bundle:

  format_penalty_rate   samples that are lenient-correct, NOT truncated, and
                        strict-wrong. i.e. the answer was correct and the response
                        finished, but no `Answer:` line -> scored wrong. This is
                        the clean measure of what the verifier's format rule costs.
  format_penalty_count  the raw numerator, for auditing.
  truncated_no_answer_rate  samples cut off at the cap (finish_reason=="length").
  compliance_rate       samples that emitted a parseable `Answer:` line at all.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RES = Path(os.environ.get("MATH_DATA_ROOT", REPO / "benchmarks" / "math")) / "eval_results"


def main(results_dir: Path) -> None:
    RES = results_dir
    for detail in sorted(RES.glob("*.detail.json")):
        summary = detail.with_name(detail.name.replace(".detail.json", ".summary.json"))
        if not summary.exists():
            continue
        d = json.loads(detail.read_text())
        S = [s for p in d for s in p["samples"]]
        n = len(S)
        if not n:
            continue
        # Format penalty = correct answer + response finished + still scored wrong
        # BECAUSE of the answer format. Samples the verifier could not score at all
        # (non-integer ground truth, see is_correct_minerva) are a different failure
        # and must be excluded, or MATH-500's rate double-counts them.
        pen = [
            s for s in S
            if s.get("acc_lenient") and not s["acc"] and s["finish_reason"] == "stop"
            and not s.get("strict_error") and s["pred"] != "[STRICT_ERROR]"
        ]
        # split the penalty into its two sub-modes
        pen_no_line = [s for s in pen if s["pred"] == "[INVALID]"]
        pen_misparsed = [s for s in pen if s["pred"] != "[INVALID]"]
        # Samples rm_type=dapo could not score at all (non-integer ground truth).
        # They have no extracted pred, so they must not count as "compliant".
        err = [s for s in S if s.get("strict_error") or s["pred"] == "[STRICT_ERROR]"]
        scoreable = [s for s in S if s not in err] if err else S
        comp = [s for s in scoreable if s["pred"] not in ("[INVALID]", "[STRICT_ERROR]")]
        trunc = [s for s in S if s["finish_reason"] == "length"]
        js = json.loads(summary.read_text())
        js["format_penalty_count"] = len(pen)
        js["format_penalty_denominator"] = n
        js["format_penalty_rate"] = round(100 * len(pen) / n, 2)
        js["format_penalty_no_answer_line"] = len(pen_no_line)
        js["format_penalty_misparsed_answer_line"] = len(pen_misparsed)
        js["format_penalty_misparsed_rate"] = round(100 * len(pen_misparsed) / n, 2)
        js["strict_scoring_error_rate"] = round(100 * len(err) / n, 2)
        js["compliance_denominator_scoreable"] = len(scoreable)
        js["compliance_rate"] = round(100 * len(comp) / len(scoreable), 2) if scoreable else None
        js["truncated_no_answer_rate"] = round(100 * len(trunc) / n, 2)
        summary.write_text(json.dumps(js, indent=2))
        print(
            f"{summary.name:<42} format_penalty={len(pen)}/{n} ({100*len(pen)/n:.2f}%)  "
            f"compliance={100*len(comp)/n:.2f}%  truncated={100*len(trunc)/n:.2f}%"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RES,
                    help="directory holding *.detail.json / *.summary.json from eval_math.py")
    main(ap.parse_args().results_dir)
