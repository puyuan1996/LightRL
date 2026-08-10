"""Guards for the math RLVR eval tooling under tools/evaluation/.

The numbers this tooling produces are only interpretable if three things hold:
the two scoring tracks mean what the document says they mean, the eval prompt is
byte-identical to the training prompt, and a dataset the strict verifier cannot
score is reported as such rather than crashing or being silently counted wrong.
Each of those is asserted here.

Background: OpenClaw-RL issues #35, #36 and #37.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "tools" / "evaluation"
TRAIN_SCRIPT = ROOT / "examples" / "training" / "train_qwen3_8b_dapo_math.sh"
SHELL_SCRIPTS = [
    EVAL_DIR / "launch_sglang_math.sh",
    EVAL_DIR / "run_math_base_evals.sh",
    TRAIN_SCRIPT,
]

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# eval_math imports the vendored slime verifier; skip rather than fail if the
# checkout is missing it, so the rest of the suite stays runnable.
eval_math = pytest.importorskip(
    "eval_math", reason="tools/evaluation/eval_math.py needs the vendored slime rm_hub"
)
prepare_math_data = pytest.importorskip("prepare_math_data")


# --- scoring tracks ----------------------------------------------------------

def test_lenient_track_accepts_a_correct_boxed_answer():
    """The whole point of the second track: right answer, wrong output format."""
    assert eval_math.lenient_acc(r"so the total is \boxed{70}", "70") is True
    assert eval_math.lenient_acc(r"so the total is \boxed{71}", "70") is False


def test_lenient_track_needs_a_boxed_answer_to_fire():
    """`Answer: 70` is the strict track's job; lenient only adds the boxed path.

    Reporting them separately is what separates "cannot do the math" from "did
    not follow the output format", so the two must not silently overlap.
    """
    assert eval_math.lenient_acc("Answer: 70", "70") is False


def test_lenient_track_grades_the_full_text_not_a_pre_extracted_answer():
    """grade_answer_verl does its own \\boxed{} extraction.

    Handing it an already-extracted "70" makes it return False for everything,
    which would silently zero the lenient track. This pins the calling contract.
    """
    from slime.rollout.rm_hub import grade_answer_verl

    assert grade_answer_verl(r"\boxed{70}", "70") is True
    assert not grade_answer_verl("70", "70")


def test_strict_track_cannot_score_a_non_integer_answer():
    """rm_type=dapo hard-casts the ground truth with int(float(gt)).

    On MATH-500 that is ~30% of the set (coordinate pairs, fractions, intervals).
    eval_math must report it as strict_scoring_error rather than crash the sweep
    or count it as a wrong answer; see docs/evaluation/math_rlvr.md.
    """
    from slime.rollout.rm_hub import compute_score_dapo

    with pytest.raises(Exception):
        compute_score_dapo(r"Answer: \left(3,\frac{\pi}{2}\right)", r"\left(3,\frac{\pi}{2}\right)")


# --- Avg@k / Pass@k and the token-cap counterfactual -------------------------

def _problem(*samples):
    return {"samples": [{"acc": acc, "completion_tokens": tok} for acc, tok in samples]}


def test_avg_and_pass_at_k_on_a_hand_checked_case():
    per_problem = [_problem((True, 100), (False, 100)), _problem((True, 100), (True, 100))]
    assert eval_math.avg_and_pass_at_k(per_problem, lambda s: True) == (0.75, 1.0)


def test_a_capped_out_sample_scores_wrong_and_stays_in_the_denominator():
    """Dropping over-cap samples instead would shrink the denominator and inflate
    the result, which is the opposite of what the counterfactual is measuring."""
    per_problem = [_problem((True, 100), (False, 100)), _problem((True, 9000), (False, 100))]
    assert eval_math.avg_and_pass_at_k(per_problem, lambda s: True) == (0.5, 1.0)

    capped = eval_math.avg_and_pass_at_k(per_problem, lambda s: s["completion_tokens"] <= 8192)
    assert capped == (0.25, 0.5)


def test_avg_and_pass_at_k_handles_an_empty_run():
    assert eval_math.avg_and_pass_at_k([], lambda s: True) == (0.0, 0.0)


# --- prompt template ---------------------------------------------------------

def test_eval_prompt_wrapper_round_trips():
    """Eval and training must use a byte-identical wrapper.

    If they diverge, the DAPO verifier regex sees a different answer format at
    eval time and under-reports accuracy against the very reward being trained.
    """
    problem = "What is $1+1$?"
    (message,) = prepare_math_data.wrap(problem)
    assert message["role"] == "user"
    assert prepare_math_data.strip_wrapper(message["content"]) == problem


def test_eval_prompt_wrapper_asks_for_the_format_the_strict_verifier_parses():
    (message,) = prepare_math_data.wrap("x")
    assert "Answer: $Answer" in message["content"]
    assert 'after "Answer:"' in message["content"]


def test_math_data_root_is_repo_relative_and_overridable(monkeypatch, tmp_path):
    assert eval_math.DATA_ROOT == ROOT / "benchmarks" / "math"
    monkeypatch.setenv("MATH_DATA_ROOT", str(tmp_path))
    import importlib

    reloaded = importlib.reload(eval_math)
    assert reloaded.DATA_ROOT == tmp_path
    monkeypatch.delenv("MATH_DATA_ROOT")
    importlib.reload(eval_math)


# --- shell entrypoints -------------------------------------------------------

def test_the_expected_shell_scripts_exist():
    """Without this the parametrized checks below would pass on an empty list."""
    assert all(script.is_file() for script in SHELL_SCRIPTS)


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize(
    "path",
    [EVAL_DIR / name for name in (
        "eval_math.py", "prepare_math_data.py", "rescore_math_eval.py",
        "math_paired_stats.py", "launch_sglang_math.sh", "run_math_base_evals.sh",
    )] + [TRAIN_SCRIPT],
    ids=lambda p: p.name,
)
def test_no_site_specific_absolute_paths(path):
    """These were written on one box; a path left behind makes them run nowhere else."""
    offenders = [
        f"{path.name}:{lineno}: {line.strip()}"
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if "/mnt/data/deepghs/" in line or "/nfs/models/" in line or "miniconda3" in line
    ]
    assert not offenders, "site-specific paths must not be hardcoded:\n" + "\n".join(offenders)


def test_training_script_defaults_seed_before_it_is_interpolated():
    """RUN_ID interpolates ${SEED}; under `set -u` a later default aborts the run."""
    lines = TRAIN_SCRIPT.read_text().splitlines()
    seed_default = next(i for i, l in enumerate(lines) if l.startswith('SEED="${SEED:-'))
    run_id_use = next(i for i, l in enumerate(lines) if "${SEED}" in l and "RUN_ID" in l)
    assert seed_default < run_id_use


def test_training_script_requires_the_checkpoint_it_trains():
    source = TRAIN_SCRIPT.read_text()
    assert ': "${HF_CKPT:?' in source
    assert ': "${REF_LOAD:?' in source


def test_training_script_dry_run_resolves_everything_from_the_repo(tmp_path):
    """A dry run must produce a full argv without touching the cluster."""
    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        env={
            "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
            "HF_CKPT": str(tmp_path), "REF_LOAD": str(tmp_path), "DRY_RUN": "1",
        },
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--rm-type math" in result.stdout
    assert str(ROOT / "benchmarks" / "math") in result.stdout
    assert "/mnt/data/deepghs" not in result.stdout


def test_training_script_refuses_to_start_without_the_datasets(tmp_path):
    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        env={
            "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
            "HF_CKPT": str(tmp_path), "REF_LOAD": str(tmp_path),
            "MATH_DATA_ROOT": str(tmp_path / "absent"),
        },
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "prepare_math_data.py" in result.stderr
