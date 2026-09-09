from __future__ import annotations

import json

from tools.analysis.plot_math_rlvr_curves import build_summary, collect_curves


def _write_run(run_dir, *, structured: bool, legacy_log: bool) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    if structured:
        rows = [
            {
                "schema": "terminal_rl.rollout_metrics.v1",
                "phase": "rollout",
                "rollout_id": i,
                "global_step": i,
                "metrics": {"raw_reward": 0.5 + 0.1 * i, "truncated": 0.5},
            }
            for i in range(3)
        ] + [
            {
                "schema": "terminal_rl.eval_dataset_metrics.v1",
                "phase": "eval",
                "rollout_id": 5,
                "global_step": 5,
                "dataset": "aime-2024",
                "reward": 0.65,
                "truncated_ratio": 0.35,
            },
            {
                "schema": "terminal_rl.actor_update_metrics.v1",
                "phase": "actor_train",
                "rollout_id": 0,
                "metrics": {"train_rollout_logprob_abs_diff": 0.012},
            },
        ]
        (logs / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    if legacy_log:
        (logs / "train.log").write_text(
            "[2026-09-08 09:32:27] data.py:241 - rollout 0: "
            "{'rollout/raw_reward': 0.5625, 'rollout/truncated': 0.5625}\n"
            "[2026-09-08 10:33:16] rollout.py:1274 - eval 5: "
            "{'eval/aime-2024': 0.65, 'eval/aime-2024-truncated_ratio': 0.358, "
            "'eval/aime-2024/response_len/mean': 12280.7}\n"
        )


def test_collect_curves_from_structured_jsonl(tmp_path):
    _write_run(tmp_path, structured=True, legacy_log=False)
    curves = collect_curves(tmp_path)
    assert [y for _, y in curves["train"]["raw_reward"]] == [0.5, 0.6, 0.7]
    assert curves["eval"]["aime-2024"]["reward"] == [(5, 0.65)]
    assert curves["eval"]["aime-2024"]["truncated_ratio"] == [(5, 0.35)]
    assert curves["logprob_abs_diff"] == [(0, 0.012)]

    summary = build_summary(curves)
    assert summary["train"]["raw_reward"]["n_points"] == 3
    assert summary["eval"]["aime-2024"]["reward"]["last"] == 0.65


def test_collect_curves_from_legacy_train_log(tmp_path):
    _write_run(tmp_path, structured=False, legacy_log=True)
    curves = collect_curves(tmp_path)
    assert curves["train"]["raw_reward"] == [(0, 0.5625)]
    assert curves["train"]["truncated"] == [(0, 0.5625)]
    assert curves["eval"]["aime-2024"]["reward"] == [(5, 0.65)]
    assert curves["eval"]["aime-2024"]["truncated_ratio"] == [(5, 0.358)]


def test_structured_jsonl_wins_over_log_duplicates(tmp_path):
    _write_run(tmp_path, structured=True, legacy_log=True)
    curves = collect_curves(tmp_path)
    # Structured rollout rows take precedence; the log's rollout 0 is kept only
    # because the structured run stopped at rollout 2... rollout ids 0-2 exist
    # in both sources, so the log copy must not duplicate or overwrite them.
    assert [y for _, y in curves["train"]["raw_reward"]] == [0.5, 0.6, 0.7]
    assert curves["eval"]["aime-2024"]["reward"] == [(5, 0.65)]
