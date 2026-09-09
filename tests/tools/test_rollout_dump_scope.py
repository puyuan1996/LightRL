from slime.ray.rollout import _should_save_debug_rollout_data


def test_debug_rollout_dump_defaults_to_eval_only():
    assert _should_save_debug_rollout_data("eval", evaluation=True)
    assert not _should_save_debug_rollout_data("eval", evaluation=False)


def test_debug_rollout_dump_scope_can_include_train_or_both():
    assert _should_save_debug_rollout_data("train", evaluation=False)
    assert not _should_save_debug_rollout_data("train", evaluation=True)
    assert _should_save_debug_rollout_data("both", evaluation=False)
    assert _should_save_debug_rollout_data("both", evaluation=True)


def test_debug_rollout_dump_rejects_unknown_scope():
    try:
        _should_save_debug_rollout_data("unknown", evaluation=False)
    except ValueError as exc:
        assert "expected one of eval, train, both" in str(exc)
    else:
        raise AssertionError("unknown scope should be rejected")
