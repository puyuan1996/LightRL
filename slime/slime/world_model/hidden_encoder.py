from __future__ import annotations

from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .action_view import parse_tool_call_bundle
from .result_view import parse_result_only_view
from .seta_dataset import TerminalTransition
from .state_view import BELIEF_VIEW_V1, FULL_CONTEXT_V1, STATE_VIEW_CHOICES, belief_view_parts


@contextmanager
def _optional_model_load_lock():
    """Serialize large checkpoint loads when workers share limited host RAM."""

    raw_path = os.environ.get("WM_MODEL_LOAD_LOCK_PATH")
    if not raw_path:
        yield
        return
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stable_hash_hidden(texts: Sequence[str], hidden_dim: int) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for text in texts:
        seed_bytes = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(seed_bytes, "little") & ((1 << 63) - 1))
        rows.append(F.normalize(torch.randn(hidden_dim, generator=generator), dim=0))
    return torch.stack(rows) if rows else torch.empty(0, hidden_dim)


def hash_hidden_batch(
    transitions: Sequence[TerminalTransition],
    hidden_dim: int,
    *,
    state_view: str = FULL_CONTEXT_V1,
    belief_max_events: int = 3,
) -> dict[str, torch.Tensor]:
    if state_view not in STATE_VIEW_CHOICES:
        raise ValueError(f"Unsupported state_view={state_view!r}")
    if state_view == BELIEF_VIEW_V1:
        state_text = [
            "".join(belief_view_parts(row.context_messages, max_events=belief_max_events))
            for row in transitions
        ]
        next_text = [
            "".join(
                belief_view_parts(
                    row.next_context_messages or row.context_messages,
                    max_events=belief_max_events,
                )
            )
            for row in transitions
        ]
    else:
        state_text = [
            json.dumps(row.context_messages, ensure_ascii=False, sort_keys=True)
            for row in transitions
        ]
        next_text = [
            json.dumps(
                row.next_context_messages or row.context_messages,
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in transitions
        ]
    action_text = [row.action_text for row in transitions]
    feedback_text = [row.feedback_text for row in transitions]
    return {
        "state_hidden": _stable_hash_hidden(state_text, hidden_dim),
        "action_hidden": _stable_hash_hidden(action_text, hidden_dim),
        "target_hidden": _stable_hash_hidden(feedback_text, hidden_dim),
        "next_state_hidden": _stable_hash_hidden(next_text, hidden_dim),
        "has_next": torch.tensor([row.has_next for row in transitions], dtype=torch.bool),
    }


def _longest_common_prefix(first: list[int], second: list[int]) -> int:
    limit = min(len(first), len(second))
    index = 0
    while index < limit and first[index] == second[index]:
        index += 1
    return index


class PolicyHiddenEncoder(nn.Module):
    """Extract state/action/feedback hidden representations from a policy LLM.

    ``state_hidden`` and ``action_hidden`` come from one causal forward over
    ``h_t + a_t``: state uses the prompt-end position and action pools only the
    action span.  Feedback and next-state targets are evaluated on detached
    target branches.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        target_model: nn.Module | None = None,
        hidden_layer: int = -1,
        action_pool: str = "mean",
        max_context_tokens: int = 1536,
        max_action_tokens: int = 512,
        max_feedback_tokens: int = 512,
        backprop_to_llm: bool = False,
        llm_train_mode: str = "full",
        strict_action_boundary: bool = True,
        state_view: str = FULL_CONTEXT_V1,
        belief_max_events: int = 3,
        encoder_long_text_mode: str = "tail_v1",
        chunk_forward_batch_size: int = 16,
        prediction_target: str = "feedback",
    ) -> None:
        super().__init__()
        if action_pool not in {"mean", "last"}:
            raise ValueError(f"Unsupported action_pool={action_pool!r}; expected 'mean' or 'last'")
        self.model = model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.hidden_layer = int(hidden_layer)
        self.action_pool = action_pool
        self.max_context_tokens = int(max_context_tokens)
        self.max_action_tokens = int(max_action_tokens)
        self.max_feedback_tokens = int(max_feedback_tokens)
        self.backprop_to_llm = bool(backprop_to_llm)
        if llm_train_mode not in {"full", "lora"}:
            raise ValueError("llm_train_mode must be 'full' or 'lora'")
        if llm_train_mode == "lora" and not self.backprop_to_llm:
            raise ValueError("LoRA requires backbone backpropagation")
        self.llm_train_mode = llm_train_mode
        if prediction_target not in {"feedback", "next_state"}:
            raise ValueError("prediction_target must be 'feedback' or 'next_state'")
        self.prediction_target = prediction_target
        if self.target_model is not None and not self.backprop_to_llm:
            raise ValueError("a fixed target model is only meaningful with backbone backpropagation")
        self.strict_action_boundary = bool(strict_action_boundary)
        if state_view not in STATE_VIEW_CHOICES:
            raise ValueError(
                f"Unsupported state_view={state_view!r}; expected one of {STATE_VIEW_CHOICES}"
            )
        self.state_view = state_view
        self.belief_max_events = int(belief_max_events)
        if encoder_long_text_mode not in {"tail_v1", "hierarchical_chunks_v1"}:
            raise ValueError(
                "encoder_long_text_mode must be 'tail_v1' or "
                "'hierarchical_chunks_v1'"
            )
        self.encoder_long_text_mode = encoder_long_text_mode
        self.chunk_forward_batch_size = int(chunk_forward_batch_size)
        if min(self.max_context_tokens, self.max_action_tokens, self.max_feedback_tokens) <= 0:
            raise ValueError("token limits must be positive")
        if self.belief_max_events <= 0:
            raise ValueError("belief_max_events must be positive")
        if self.chunk_forward_batch_size <= 0:
            raise ValueError("chunk_forward_batch_size must be positive")
        self.last_hierarchical_stats: dict[str, dict[str, int]] = {}
        self.model.eval()
        if self.llm_train_mode == "full":
            self.model.requires_grad_(self.backprop_to_llm)
        if self.target_model is not None:
            self.target_model.eval()
            self.target_model.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> "PolicyHiddenEncoder":
        from transformers import AutoModel, AutoTokenizer

        fixed_target_backbone = bool(kwargs.pop("fixed_target_backbone", False))
        llm_train_mode = str(kwargs.pop("llm_train_mode", "full"))
        lora_rank = int(kwargs.pop("lora_rank", 16))
        lora_alpha = int(kwargs.pop("lora_alpha", 32))
        lora_dropout = float(kwargs.pop("lora_dropout", 0.05))
        raw_target_modules = kwargs.pop(
            "lora_target_modules",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
        if isinstance(raw_target_modules, str):
            lora_target_modules = [
                value.strip() for value in raw_target_modules.split(",") if value.strip()
            ]
        else:
            lora_target_modules = [str(value).strip() for value in raw_target_modules]
        if llm_train_mode not in {"full", "lora"}:
            raise ValueError("llm_train_mode must be 'full' or 'lora'")
        if llm_train_mode == "lora":
            if not kwargs.get("backprop_to_llm", False):
                raise ValueError("LoRA requires backprop_to_llm=True")
            if lora_rank <= 0 or lora_alpha <= 0 or not 0.0 <= lora_dropout < 1.0:
                raise ValueError("invalid LoRA rank, alpha, or dropout")
            if not lora_target_modules:
                raise ValueError("LoRA target modules cannot be empty")
        torch_dtype: Any = dtype
        if dtype != "auto":
            torch_dtype = getattr(torch, dtype)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        target_device = torch.device(device)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if target_device.type == "cuda":
            # Loading a bf16 8B model on CPU first exceeds small rollout-host
            # memory limits. Accelerate dispatches safetensor shards directly
            # onto the one visible GPU while keeping only a shard-sized CPU peak.
            load_kwargs["device_map"] = {"": target_device.index or 0}
        with _optional_model_load_lock():
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
            if target_device.type != "cuda":
                model = model.to(target_device)
            target_model = None
            if fixed_target_backbone:
                target_model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
                if target_device.type != "cuda":
                    target_model = target_model.to(target_device)
            if llm_train_mode == "lora":
                try:
                    from peft import LoraConfig, TaskType, get_peft_model
                except ImportError as exc:
                    raise RuntimeError("LoRA training requires the peft package") from exc
                model = get_peft_model(
                    model,
                    LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        inference_mode=False,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        target_modules=lora_target_modules,
                    ),
                )
                enable_input_require_grads = getattr(
                    model,
                    "enable_input_require_grads",
                    None,
                )
                if callable(enable_input_require_grads):
                    enable_input_require_grads()
        return cls(
            model,
            tokenizer,
            target_model=target_model,
            llm_train_mode=llm_train_mode,
            **kwargs,
        )

    def merge_lora_for_export(self) -> bool:
        """Merge a trained adapter so the exported backbone is standalone."""

        if self.llm_train_mode != "lora":
            return False
        merge = getattr(self.model, "merge_and_unload", None)
        if not callable(merge):
            raise RuntimeError("LoRA model cannot be merged for export")
        self.model = merge()
        self.model.eval()
        self.llm_train_mode = "full"
        return True

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        value = getattr(self.model.config, "hidden_size", None)
        if value is None:
            value = getattr(self.model.config, "d_model", None)
        if value is None:
            raise AttributeError("Cannot infer hidden size from model config")
        return int(value)

    def _chat_ids(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
        try:
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
            )
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return list(ids)
        except Exception:
            text = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
            return list(self.tokenizer.encode(text, add_special_tokens=True))

    def _current_ids(self, transition: TerminalTransition) -> tuple[list[int], int, list[int]]:
        if self.state_view == BELIEF_VIEW_V1:
            prompt = self._belief_ids(transition.context_messages)
            action = list(
                self.tokenizer.encode(
                    "<ACTION>\n" + transition.action_text,
                    add_special_tokens=False,
                )
            )
            if not action:
                eos = self.tokenizer.eos_token_id
                action = [int(eos if eos is not None else 0)]
            action = action[-self.max_action_tokens :]
            combined = prompt + action
            return combined, len(prompt) - 1, list(range(len(prompt), len(combined)))

        prompt = self._chat_ids(transition.context_messages, add_generation_prompt=True)
        full_messages = list(transition.context_messages) + [
            {"role": "assistant", "content": transition.action_text}
        ]
        full = self._chat_ids(full_messages, add_generation_prompt=False)
        prefix = _longest_common_prefix(prompt, full)
        if prefix == len(prompt):
            action = full[prefix:]
        else:
            if self.strict_action_boundary:
                raise ValueError(
                    "chat template cannot prove an exact prompt/action token boundary; "
                    "use the rollout token IDs or explicitly disable strict_action_boundary for a diagnostic"
                )
            action = list(self.tokenizer.encode(transition.action_text, add_special_tokens=False))
        if not action:
            eos = self.tokenizer.eos_token_id
            action = [int(eos if eos is not None else 0)]
        prompt = prompt[-self.max_context_tokens :]
        action = action[-self.max_action_tokens :]
        if not prompt:
            bos = self.tokenizer.bos_token_id
            prompt = [int(bos if bos is not None else 0)]
        combined = prompt + action
        return combined, len(prompt) - 1, list(range(len(prompt), len(combined)))

    def _target_ids(self, text: str) -> list[int]:
        prefix = "<environment_observation>\n"
        ids = list(self.tokenizer.encode(prefix + text, add_special_tokens=True))
        if not ids:
            eos = self.tokenizer.eos_token_id
            ids = [int(eos if eos is not None else 0)]
        return ids[-self.max_feedback_tokens :]

    def _belief_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        conditioning, suffix = belief_view_parts(
            messages,
            max_events=self.belief_max_events,
        )
        conditioning_ids = list(
            self.tokenizer.encode(conditioning, add_special_tokens=True)
        )
        suffix_ids = list(
            self.tokenizer.encode(suffix, add_special_tokens=False)
        )
        if not suffix_ids:
            raise ValueError("belief_view_v1 produced an empty STATE_VIEW suffix")
        if len(suffix_ids) > self.max_context_tokens:
            raise ValueError(
                "belief_view_v1 STATE_VIEW suffix exceeds max_context_tokens; "
                "reduce event size/count instead of truncating the target block"
            )
        keep_conditioning = self.max_context_tokens - len(suffix_ids)
        ids = conditioning_ids[-keep_conditioning:] + suffix_ids if keep_conditioning else suffix_ids
        if not ids:
            raise ValueError("belief_view_v1 produced no encoder tokens")
        return ids

    def _next_ids(self, transition: TerminalTransition) -> list[int]:
        messages = transition.next_context_messages or transition.context_messages
        if self.state_view == BELIEF_VIEW_V1:
            return self._belief_ids(messages)
        ids = self._chat_ids(messages, add_generation_prompt=True)
        if not ids:
            bos = self.tokenizer.bos_token_id
            ids = [int(bos if bos is not None else 0)]
        return ids[-self.max_context_tokens :]

    def _pad(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(row) for row in rows)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            pad = 0
        input_ids = torch.full((len(rows), max_len), int(pad), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=self.device)
        for index, row in enumerate(rows):
            length = len(row)
            input_ids[index, :length] = torch.tensor(row, dtype=torch.long, device=self.device)
            attention_mask[index, :length] = 1
        return input_ids, attention_mask

    def _forward_hidden(
        self,
        rows: list[list[int]],
        *,
        require_grad: bool,
        target_branch: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_branch and require_grad:
            raise ValueError("target branches must never require gradients")
        input_ids, attention_mask = self._pad(rows)
        context = nullcontext() if require_grad else torch.no_grad()
        model = self.target_model if target_branch and self.target_model is not None else self.model
        with context:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("Policy model did not return hidden_states")
            hidden = hidden_states[self.hidden_layer].float()
        return hidden, attention_mask

    def _hierarchical_blocks(
        self,
        text: str,
        *,
        kind: str,
    ) -> list[list[int]]:
        if kind == "action":
            calls = parse_tool_call_bundle(text)
            rendered = [
                json.dumps(
                    call,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for call in calls
            ]
            prefix = "<tool_call>\n"
            token_limit = self.max_action_tokens
        elif kind == "target":
            rendered = list(parse_result_only_view(text))
            prefix = "<environment_observation>\n"
            token_limit = self.max_feedback_tokens
        else:
            raise ValueError(f"unsupported hierarchical block kind={kind!r}")

        blocks: list[list[int]] = []
        for value in rendered:
            ids = list(
                self.tokenizer.encode(
                    prefix + value,
                    add_special_tokens=True,
                )
            )
            if not ids:
                eos = self.tokenizer.eos_token_id
                ids = [int(eos if eos is not None else 0)]
            blocks.append(ids)
        return blocks

    def _hierarchical_pool(
        self,
        texts: Sequence[str],
        *,
        kind: str,
        require_grad: bool,
        target_branch: bool = False,
    ) -> torch.Tensor:
        token_limit = (
            self.max_action_tokens if kind == "action" else self.max_feedback_tokens
        )
        record_blocks = [
            self._hierarchical_blocks(text, kind=kind) for text in texts
        ]
        chunks: list[list[int]] = []
        ownership: list[tuple[int, int]] = []
        token_count = 0
        for record_index, blocks in enumerate(record_blocks):
            for block_index, ids in enumerate(blocks):
                token_count += len(ids)
                for start in range(0, len(ids), token_limit):
                    chunks.append(ids[start : start + token_limit])
                    ownership.append((record_index, block_index))

        pooled_chunks: list[torch.Tensor] = []
        for start in range(0, len(chunks), self.chunk_forward_batch_size):
            hidden, mask = self._forward_hidden(
                chunks[start : start + self.chunk_forward_batch_size],
                require_grad=require_grad,
                target_branch=target_branch,
            )
            weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            pooled_chunks.extend(pooled.unbind(0))

        block_chunks: dict[tuple[int, int], list[torch.Tensor]] = {}
        for owner, pooled in zip(ownership, pooled_chunks, strict=True):
            block_chunks.setdefault(owner, []).append(pooled)
        record_rows: list[torch.Tensor] = []
        for record_index, blocks in enumerate(record_blocks):
            block_rows = [
                torch.stack(block_chunks[(record_index, block_index)]).mean(dim=0)
                for block_index in range(len(blocks))
            ]
            record_rows.append(torch.stack(block_rows).mean(dim=0))
        self.last_hierarchical_stats[kind] = {
            "record_count": len(record_blocks),
            "block_count": sum(len(blocks) for blocks in record_blocks),
            "chunk_count": len(chunks),
            "token_count": token_count,
        }
        return torch.stack(record_rows)

    def forward(
        self,
        transitions: Sequence[TerminalTransition],
        *,
        include_auxiliary_targets: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not transitions:
            raise ValueError("PolicyHiddenEncoder requires at least one transition")
        current_rows: list[list[int]] = []
        state_positions: list[int] = []
        action_positions: list[list[int]] = []
        for transition in transitions:
            row, state_position, action_position = self._current_ids(transition)
            if self.encoder_long_text_mode == "hierarchical_chunks_v1":
                # The separately pooled canonical action branch below carries
                # the action gradient.  Tokens after the causal state position
                # cannot influence the state representation and are omitted.
                row = row[: state_position + 1]
                action_position = []
            current_rows.append(row)
            state_positions.append(state_position)
            action_positions.append(action_position)

        current_hidden, _ = self._forward_hidden(current_rows, require_grad=self.backprop_to_llm)
        state_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        for index, (state_position, action_position) in enumerate(zip(state_positions, action_positions)):
            state_rows.append(current_hidden[index, state_position])
            if action_position:
                action_span = current_hidden[index, action_position]
                action_rows.append(
                    action_span[-1] if self.action_pool == "last" else action_span.mean(dim=0)
                )

        if self.encoder_long_text_mode == "hierarchical_chunks_v1":
            action_pooled = self._hierarchical_pool(
                [transition.action_text for transition in transitions],
                kind="action",
                require_grad=self.backprop_to_llm,
            )
            if self.prediction_target == "feedback" or include_auxiliary_targets:
                target_pooled = self._hierarchical_pool(
                    [transition.feedback_text for transition in transitions],
                    kind="target",
                    require_grad=False,
                    target_branch=True,
                )
            else:
                target_pooled = torch.zeros_like(action_pooled).detach()
        else:
            action_pooled = torch.stack(action_rows)
            if self.prediction_target == "feedback" or include_auxiliary_targets:
                target_rows = [self._target_ids(transition.feedback_text) for transition in transitions]
                target_hidden, target_mask = self._forward_hidden(
                    target_rows, require_grad=False, target_branch=True
                )
                target_pooled = (target_hidden * target_mask.unsqueeze(-1)).sum(dim=1) / target_mask.sum(
                    dim=1, keepdim=True
                ).clamp_min(1)
            else:
                target_pooled = torch.zeros_like(action_pooled).detach()

        next_rows = [self._next_ids(transition) for transition in transitions]
        next_hidden, next_mask = self._forward_hidden(
            next_rows, require_grad=False, target_branch=True
        )
        next_lengths = next_mask.sum(dim=1).clamp_min(1) - 1
        next_pooled = next_hidden[
            torch.arange(next_hidden.size(0), device=next_hidden.device),
            next_lengths,
        ]
        result = {
            "state_hidden": torch.stack(state_rows),
            "action_hidden": action_pooled,
            "target_hidden": target_pooled.detach(),
            "next_state_hidden": next_pooled.detach(),
            "has_next": torch.tensor([row.has_next for row in transitions], dtype=torch.bool, device=self.device),
        }
        if not self.backprop_to_llm:
            result["state_hidden"] = result["state_hidden"].detach()
            result["action_hidden"] = result["action_hidden"].detach()
        return result
