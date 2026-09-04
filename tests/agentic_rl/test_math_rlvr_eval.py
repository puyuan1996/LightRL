"""Guards for the math RLVR eval tooling under tools/evaluation/.

The numbers this tooling produces are only interpretable if three things hold:
the two scoring tracks mean what the document says they mean, the eval prompt is
byte-identical to the training prompt, and a dataset the strict verifier cannot
score is reported as such rather than crashing or being silently counted wrong.
Each of those is asserted here.

Background: OpenClaw-RL issues #35, #36 and #37.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def _stub_bin(tmp_path):
    """A PATH with stub ray/pkill so a real launch can be exercised offline."""
    binp = tmp_path / "bin"
    binp.mkdir(exist_ok=True)
    for cmd in ("ray", "pkill", "nvidia-smi"):
        target = binp / cmd
        target.write_text(f'#!/usr/bin/env bash\necho "[stub {cmd}] $*"\nexit 0\n')
        target.chmod(0o755)
    return binp


def _stub_datasets(tmp_path):
    root = tmp_path / "data"
    for name in ("aime-2025", "aime-2024"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.jsonl").write_text('{"prompt": [], "label": "1"}\n')
    return root


@pytest.mark.parametrize("cuda_env_prefix", [None, "set"], ids=["no-cuda-prefix", "cuda-prefix"])
def test_a_real_launch_reaches_ray_job_submit(tmp_path, cuda_env_prefix):
    """The dry run exits before RUNTIME_ENV_JSON is built.

    That block forwards CUDA_HOME, CUDA_PATH, LD_LIBRARY_PATH and HF_HOME
    verbatim, so if any of them is only conditionally defined the launch dies
    under `set -u` right after `ray start` -- with a live Ray head and no job.
    """
    env = {
        "PATH": f"{_stub_bin(tmp_path)}:/usr/bin:/bin", "HOME": str(tmp_path),
        "HF_CKPT": str(tmp_path), "REF_LOAD": str(tmp_path),
        "MATH_DATA_ROOT": str(_stub_datasets(tmp_path)),
    }
    if cuda_env_prefix:
        env["CUDA_ENV_PREFIX"] = str(tmp_path / "cudaenv")

    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)], env=env, capture_output=True, text=True, timeout=180
    )
    assert "unbound variable" not in result.stderr, result.stderr[-1500:]
    assert "job submit" in result.stdout, result.stdout[-1500:]
    assert result.returncode == 0


def test_dry_run_does_not_create_directories_in_the_checkout(tmp_path):
    """RUN_DIR now defaults inside the repo, so it must not be made before the exit."""
    runs = ROOT / "benchmarks" / "math" / "runs"
    before = set(runs.iterdir()) if runs.is_dir() else set()
    subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HF_CKPT": str(tmp_path),
             "REF_LOAD": str(tmp_path), "DRY_RUN": "1"},
        capture_output=True, text=True, timeout=120,
    )
    after = set(runs.iterdir()) if runs.is_dir() else set()
    assert after == before


def test_the_sweep_can_reproduce_the_documented_temperature_comparison(tmp_path):
    """docs/evaluation/math_rlvr.md quotes a T=0.6/top_p=0.95 run; the entrypoint
    hardcoded 1.0/1.0, so that comparison could not be reproduced with it."""
    stub = tmp_path / "pystub"
    stub.write_text('#!/usr/bin/env bash\necho "ARGS: $*"\n')
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(EVAL_DIR / "run_math_base_evals.sh")],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "MODEL": "m",
             "PYTHON": str(stub), "DATASETS": "aime-2025", "TEMPERATURE": "0.6",
             "TOP_P": "0.95", "MATH_DATA_ROOT": str(tmp_path / "data")},
        capture_output=True, text=True, timeout=120,
    )
    assert "--temperature 0.6 --top-p 0.95" in result.stdout
    # The whole tag, not a prefix: "--tag T0.6" also matches "T0.6_p0.95".
    assert "--tag T0.6_p0.95" in result.stdout


class _StubCompletions(BaseHTTPRequestHandler):
    """Minimal /v1/chat/completions that always answers with the same body."""

    reply = ""

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({
            "choices": [{"message": {"content": self.reply}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10, "prompt_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@contextlib.contextmanager
def _stub_endpoint(reply: str):
    _StubCompletions.reply = reply
    server = HTTPServer(("127.0.0.1", 0), _StubCompletions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _run_eval(tmp_path, labels, reply: str, n: int = 2) -> dict:
    """Run the real evaluator against the stub and return its summary."""
    if isinstance(labels, str):
        labels = [labels]
    data = tmp_path / "sets" / "s"
    data.mkdir(parents=True, exist_ok=True)
    (data / "s.jsonl").write_text("".join(
        json.dumps({"prompt": [{"role": "user", "content": "q"}], "label": label}) + "\n"
        for label in labels
    ))
    out = tmp_path / "out"
    with _stub_endpoint(reply) as port:
        assert eval_math.main(
            ["--data", str(data / "s.jsonl"), "--n", str(n), "--model", "m",
             "--port", str(port), "--out", str(out), "--tag", "t"]
        ) is not None
    return json.loads(next(out.glob("*.summary.json")).read_text())


def test_the_default_sweep_tag_is_the_one_paired_stats_looks_up(tmp_path):
    """math_paired_stats.py resolves files by tag; the sweep must not rename them.

    It builds "{stem}_{tag}_n{n}.detail.json" from a hardcoded baseline tag, so a
    sweep that writes a different default name leaves the next pipeline step with
    nothing to find.
    """
    stats_src = (EVAL_DIR / "math_paired_stats.py").read_text()
    baseline_tag = re.search(r'^BASE = "([^"]+)"', stats_src, re.M).group(1)

    stub = tmp_path / "pystub"
    stub.write_text('#!/usr/bin/env bash\necho "ARGS: $*"\n')
    stub.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "MODEL": "m",
           "PYTHON": str(stub), "DATASETS": "aime-2025",
           "MATH_DATA_ROOT": str(tmp_path / "data")}
    default = subprocess.run(
        ["bash", str(EVAL_DIR / "run_math_base_evals.sh")],
        env=env, capture_output=True, text=True, timeout=120,
    )
    # Whole token, so a renamed "T1.0_p1.0" cannot satisfy a prefix match.
    assert re.search(rf"--tag {re.escape(baseline_tag)}(\s|$)", default.stdout), (
        f"the default sweep must still write {baseline_tag}-tagged files"
    )

    # A top_p-only ablation must not land on that same name.
    ablation = subprocess.run(
        ["bash", str(EVAL_DIR / "run_math_base_evals.sh")],
        env={**env, "TOP_P": "0.95"}, capture_output=True, text=True, timeout=120,
    )
    assert re.search(r"--tag T1\.0_p0\.95(\s|$)", ablation.stdout)

    # 1 and 1.00 are the default spelled differently; a string compare would tag
    # them as ablations and orphan a full sweep's output.
    for spelling in ("1", "1.00"):
        same = subprocess.run(
            ["bash", str(EVAL_DIR / "run_math_base_evals.sh")],
            env={**env, "TOP_P": spelling}, capture_output=True, text=True, timeout=120,
        )
        assert re.search(rf"--tag {re.escape(baseline_tag)}(\s|$)", same.stdout), spelling


def test_compliance_rate_uses_the_scoreable_denominator(tmp_path):
    """Counting [STRICT_ERROR] as compliant reports 100% on a set it cannot score.

    Asserted on the computed summary, not on the source text: a mutation that
    keeps the expression and changes only the divisor has to fail here.
    """
    latex_gt = r"\left(3,\frac{\pi}{2}\right)"
    summary = _run_eval(tmp_path / "latex", latex_gt, rf"Answer: {latex_gt}")
    assert summary["strict_scoring_error_rate"] == 100.0
    assert summary["compliance_denominator_scoreable"] == 0
    assert summary["compliance_rate"] is None, "an unscoreable set must not report 100%"

    ok = _run_eval(tmp_path / "int", "70", "Answer: 70")
    assert ok["compliance_denominator_scoreable"] == 2
    assert ok["compliance_rate"] == 100.0

    # The case that separates the two denominators: one scoreable problem and one
    # the strict verifier cannot score. Over all samples this reads 50%; over the
    # scoreable ones, which is what compliance means, it is 100%.
    mixed = _run_eval(tmp_path / "mixed", ["70", r"\frac{1}{2}"], "Answer: 70", n=1)
    assert mixed["compliance_denominator_scoreable"] == 1
    assert mixed["strict_scoring_error_rate"] == 50.0
    assert mixed["compliance_rate"] == 100.0


def test_format_penalty_excludes_unscoreable_samples(tmp_path):
    """§2 defines the penalty as lenient-correct, finished, strict-wrong AND scoreable."""
    latex_gt = r"\frac{1}{2}"
    summary = _run_eval(tmp_path / "pen", latex_gt, rf"so \boxed{{{latex_gt}}}")
    assert summary["strict_scoring_error_rate"] == 100.0
    assert summary["format_penalty_count"] == 0, "unscoreable is a different loss"


def test_slime_root_override_is_honoured(tmp_path, monkeypatch):
    """The error message offers SLIME_ROOT, so it has to be consulted for real."""
    import importlib

    monkeypatch.setenv("SLIME_ROOT", str(tmp_path / "absent"))
    with pytest.raises(SystemExit) as excinfo:
        importlib.reload(eval_math)
    assert str(tmp_path / "absent") in str(excinfo.value)
    assert "PYTHONPATH" not in str(excinfo.value)

    monkeypatch.setenv("SLIME_ROOT", str(ROOT / "slime"))
    assert importlib.reload(eval_math).SLIME_ROOT == ROOT / "slime"
    monkeypatch.delenv("SLIME_ROOT")
    importlib.reload(eval_math)


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
