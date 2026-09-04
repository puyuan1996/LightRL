from pathlib import Path


def test_evaluation_package_and_legacy_cli_are_available():
    import tools.evaluation
    import tools.evaluation.core

    assert callable(tools.evaluation.main)
    assert Path("tools/evaluation/eval_cli.py").is_file()


def test_core_uses_package_imports():
    source = Path("tools/evaluation/core/batch.py").read_text()
    assert "from core." not in source


def test_evaluation_without_output_dir_uses_evaluation_runs_root(monkeypatch, tmp_path):
    from tools.evaluation.core.config import build_specs

    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    spec, _ = build_specs({"job_name": "smoke-1", "dataset": {"path": "tasks"}})
    assert Path(spec.output_dir) == tmp_path / "runs" / "evaluation" / "smoke-1"
