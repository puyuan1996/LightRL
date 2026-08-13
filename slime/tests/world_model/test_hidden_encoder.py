import sys
from types import SimpleNamespace

import torch
from torch import nn

from slime.world_model.action_view import render_tool_call_bundle
from slime.world_model.hidden_encoder import PolicyHiddenEncoder
from slime.world_model.result_view import render_result_only_view
from slime.world_model.seta_dataset import TerminalTransition


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        prefix = [self.bos_token_id] if add_special_tokens else []
        return prefix + [3 + (ord(char) % 17) for char in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and not return_dict
        ids = [self.bos_token_id]
        for message in messages:
            if message.get("role") == "assistant":
                ids.append(29)
            ids.extend(self.encode(str(message.get("content", "")), add_special_tokens=False))
        if add_generation_prompt:
            ids.append(29)
        return ids


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=4)
        self.calls = 0

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache, return_dict):
        self.calls += 1
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) * self.scale
        return SimpleNamespace(hidden_states=(hidden, hidden + 1.0))


def _transition():
    return TerminalTransition(
        trajectory_id="t",
        task_name="task",
        data_source="unit",
        turn_idx=0,
        context_messages=[{"role": "user", "content": "state"}],
        action_text="act",
        feedback_text="result",
        next_context_messages=[{"role": "user", "content": "state"}, {"role": "tool", "content": "result"}],
        done=False,
        reward=1.0,
        status="completed",
        source_path="unit",
    )


def test_policy_hidden_encoder_uses_one_forward_for_state_and_action():
    model = _Model()
    encoder = PolicyHiddenEncoder(model, _Tokenizer(), hidden_layer=-1, backprop_to_llm=False)

    output = encoder([_transition()])

    assert model.calls == 3  # current state+action, feedback target, next-state target
    assert output["state_hidden"].shape == (1, 4)
    assert output["action_hidden"].shape == (1, 4)
    assert output["target_hidden"].requires_grad is False
    assert output["has_next"].tolist() == [True]


def test_policy_hidden_encoder_backprop_flag_reaches_backbone():
    model = _Model()
    encoder = PolicyHiddenEncoder(model, _Tokenizer(), hidden_layer=-1, backprop_to_llm=True)

    output = encoder([_transition()])
    (output["state_hidden"].sum() + output["action_hidden"].sum()).backward()

    assert model.scale.grad is not None
    assert output["target_hidden"].requires_grad is False


def test_policy_hidden_encoder_lora_mode_preserves_adapter_freezing():
    model = _Model()
    model.requires_grad_(False)
    encoder = PolicyHiddenEncoder(
        model,
        _Tokenizer(),
        hidden_layer=-1,
        backprop_to_llm=True,
        llm_train_mode="lora",
    )

    assert encoder.llm_train_mode == "lora"
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_policy_hidden_encoder_rejects_ambiguous_action_boundary():
    class DivergentTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt, return_dict)
            if not add_generation_prompt:
                ids.insert(2, 97)
            return ids

    encoder = PolicyHiddenEncoder(_Model(), DivergentTokenizer())

    try:
        encoder([_transition()])
    except ValueError as exc:
        assert "prompt/action token boundary" in str(exc)
    else:
        raise AssertionError("ambiguous action boundary must fail closed")


def test_belief_view_pools_dynamic_suffix_and_conditions_on_exact_action():
    transition = _transition()
    model = _Model()
    tokenizer = _Tokenizer()
    encoder = PolicyHiddenEncoder(
        model,
        tokenizer,
        state_view="belief_view_v1",
        belief_max_events=3,
    )

    output = encoder([transition])
    belief_ids = encoder._belief_ids(transition.context_messages)

    assert torch.equal(
        output["state_hidden"][0],
        torch.full((4,), float(belief_ids[-1] + 1)),
    )
    assert output["action_hidden"].shape == (1, 4)
    assert model.calls == 3


def test_belief_state_does_not_change_when_only_action_changes():
    first = _transition()
    second = TerminalTransition.from_dict(
        {**first.to_dict(), "action_text": "different action"}
    )
    encoder = PolicyHiddenEncoder(
        _Model(),
        _Tokenizer(),
        state_view="belief_view_v1",
    )

    output = encoder([first, second])

    assert torch.equal(output["state_hidden"][0], output["state_hidden"][1])
    assert not torch.equal(output["action_hidden"][0], output["action_hidden"][1])


def test_from_pretrained_uses_safe_hf_defaults(monkeypatch):
    calls = []

    class Factory:
        def __init__(self, value):
            self.value = value

        def from_pretrained(self, path, **kwargs):
            calls.append((path, kwargs))
            return self.value

    fake_transformers = SimpleNamespace(
        AutoTokenizer=Factory(_Tokenizer()),
        AutoModel=Factory(_Model()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    PolicyHiddenEncoder.from_pretrained("local-model", device="cpu")

    assert len(calls) == 2
    for path, kwargs in calls:
        assert path == "local-model"
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
    assert calls[1][1]["low_cpu_mem_usage"] is True
    assert "device_map" not in calls[1][1]

    calls.clear()
    PolicyHiddenEncoder.from_pretrained("local-model", device="cuda:3")
    assert calls[1][1]["low_cpu_mem_usage"] is True
    assert calls[1][1]["device_map"] == {"": 3}


def test_hierarchical_pool_consumes_all_tokens_and_weights_blocks_equally():
    encoder = PolicyHiddenEncoder(
        _Model(),
        _Tokenizer(),
        hidden_layer=-1,
        max_action_tokens=5,
        max_feedback_tokens=5,
        encoder_long_text_mode="hierarchical_chunks_v1",
        chunk_forward_batch_size=2,
    )
    values = ["a", "bbbbbbbbbbbb"]
    pooled = encoder._hierarchical_pool(
        [render_result_only_view(values)],
        kind="target",
        require_grad=False,
    )

    block_values = []
    expected_tokens = 0
    expected_chunks = 0
    for value in values:
        ids = encoder.tokenizer.encode(
            "<environment_observation>\n" + value,
            add_special_tokens=True,
        )
        expected_tokens += len(ids)
        chunks = [ids[start : start + 5] for start in range(0, len(ids), 5)]
        expected_chunks += len(chunks)
        block_values.append(
            sum(sum(token + 1 for token in chunk) / len(chunk) for chunk in chunks)
            / len(chunks)
        )
    expected = sum(block_values) / len(block_values)

    assert torch.allclose(pooled, torch.full((1, 4), expected))
    assert encoder.last_hierarchical_stats["target"] == {
        "record_count": 1,
        "block_count": 2,
        "chunk_count": expected_chunks,
        "token_count": expected_tokens,
    }


def test_hierarchical_forward_requires_canonical_call_and_result_blocks():
    transition = TerminalTransition.from_dict(
        {
            **_transition().to_dict(),
            "action_text": render_tool_call_bundle(
                [{"tool_name": "shell_exec", "args": {"command": "pwd"}}]
            ),
            "feedback_text": render_result_only_view(["/tmp"]),
        }
    )
    encoder = PolicyHiddenEncoder(
        _Model(),
        _Tokenizer(),
        encoder_long_text_mode="hierarchical_chunks_v1",
        chunk_forward_batch_size=2,
    )
    output = encoder([transition])
    assert output["action_hidden"].shape == (1, 4)
    assert output["target_hidden"].shape == (1, 4)
    assert encoder.last_hierarchical_stats["action"]["block_count"] == 1
    assert encoder.last_hierarchical_stats["target"]["block_count"] == 1

    malformed = TerminalTransition.from_dict(
        {**transition.to_dict(), "action_text": "assistant reasoning"}
    )
    try:
        encoder([malformed])
    except ValueError as exc:
        assert "tool-call bundle" in str(exc)
    else:
        raise AssertionError("hierarchical action parsing must fail closed")
