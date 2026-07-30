from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
AGENTIC_RL = ROOT / "agentic_rl"


def test_rollout_configs_default_to_camel_agent():
    for name in ("rollout_qwen3.yaml", "rollout_qwen3_think.yaml"):
        config = yaml.safe_load((ROOT / "configs" / "rollout" / name).read_text())
        assert config["harness_option"] == "camel-agent"


def test_slime_backend_routes_registered_harnesses():
    script = (
        AGENTIC_RL / "backends" / "slime" / "runtime" / "train_qwen3_8b.sh"
    ).read_text()
    assert 'HARNESS_OPTION="${HARNESS_OPTION:-camel-agent}"' in script
    assert 'cfg["harness_option"] = harness_option' in script
    assert "claude-code|claude_code)" in script
    assert "mcp__terminal_rl__read_file" in script
    assert "mcp__terminal_rl__write_file" in script
    assert "mcp__terminal_rl__list_dir" in script
    assert '\\"HARNESS_OPTION\\": \\"${HARNESS_OPTION}\\"' in script


def test_experiment_config_selects_harness_without_wrapper_script():
    config = yaml.safe_load(
        (ROOT / "configs" / "experiment" / "dive_po_qwen3_8b_seta.yaml").read_text()
    )
    assert config["extends"] == "../defaults.yaml"
    assert "HARNESS_OPTION" not in config["runtime"]["env"]


def test_dive_po_runtime_accepts_claude_code_alias():
    script = (
        AGENTIC_RL / "backends" / "slime" / "runtime" / "dive_po_qwen3_8b.sh"
    ).read_text()
    assert "claude-code|claude_code)" in script
    assert "Use: camel-agent|claude_code" in script
