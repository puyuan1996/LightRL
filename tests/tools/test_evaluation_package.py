from pathlib import Path


def test_evaluation_package_and_legacy_cli_are_available():
    import tools.evaluation
    import tools.evaluation.core

    assert callable(tools.evaluation.main)
    assert Path("tools/evaluation/eval_cli.py").is_file()


def test_core_uses_package_imports():
    source = Path("tools/evaluation/core/batch.py").read_text()
    assert "from core." not in source
