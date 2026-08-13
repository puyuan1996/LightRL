from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .metadata import redact_sensitive_jsonable, redact_sensitive_text, stable_hash, truncate_head_tail_text


_SUPPORTED_RECORD_SCHEMAS = {
    "openclaw_text_jepa_world_model_v1",
    "openclaw_text_jepa_world_model_v2",
    "openclaw_text_jepa_world_model_v3",
    "openclaw_terminal_latent_world_model_v2",
    "openclaw_terminal_transition_v3",
}
_CANONICAL_RESULT_SOURCE = "result_only_v1"
_CANONICAL_ACTION_VIEW = "tool_call_bundle_v1"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _redact_jsonable(value: Any) -> Any:
    return redact_sensitive_jsonable(value)


def _bounded_text(value: Any, max_chars: int) -> str:
    return truncate_head_tail_text(redact_sensitive_text(value), max_chars)


def _validated_canonical_action_text(
    action_text: Any,
    *,
    max_chars: int | None = None,
    allow_invalid: bool = False,
) -> tuple[str, bool]:
    """Validate an already-redacted action view without outer-string redaction."""

    from .action_view import parse_tool_call_bundle

    action = str(action_text or "")
    if max_chars is not None and len(action) > max_chars:
        raise ValueError(
            "canonical action view cannot be truncated; increase max_text_chars"
        )
    try:
        parse_tool_call_bundle(action)
        return action, True
    except ValueError:
        if allow_invalid:
            return action, False
        raise



def _validated_result_only_text(
    feedback_text: Any,
    *,
    max_chars: int | None = None,
    allow_invalid: bool = False,
) -> tuple[str, bool]:
    """Validate an already-redacted result view without outer-string redaction."""

    from .result_view import parse_result_only_view, render_result_only_view

    feedback = str(feedback_text or "")
    if max_chars is not None and len(feedback) > max_chars:
        raise ValueError(
            "canonical result view cannot be truncated; increase max_text_chars"
        )
    try:
        results = parse_result_only_view(feedback)
        if render_result_only_view(
            [redact_sensitive_text(result) for result in results]
        ) != feedback:
            raise ValueError("result_only_v1 target is not canonically redacted")
        return feedback, True
    except ValueError:
        if allow_invalid:
            return feedback, False
        raise



def _redacted_transition_value(
    transition: "TerminalTransition",
) -> dict[str, Any]:
    value = asdict(transition)
    canonical_action = (
        transition.action_view_schema == _CANONICAL_ACTION_VIEW
    )
    canonical_feedback = (
        transition.feedback_source == _CANONICAL_RESULT_SOURCE
    )
    if not canonical_action and not canonical_feedback:
        return _redact_jsonable(value)
    action_text: str | None = None
    feedback_text: str | None = None
    if canonical_action:
        action_text, _ = _validated_canonical_action_text(
            transition.action_text,
            max_chars=None,
            allow_invalid=False,
        )
    if canonical_feedback:
        feedback_text, _ = _validated_result_only_text(
            transition.feedback_text,
            max_chars=None,
            allow_invalid=False,
        )
    # Redact metadata and context recursively, but keep strict structured
    # views out of regex-based outer-string redaction.
    if canonical_action:
        value["action_text"] = ""
    if canonical_feedback:
        value["feedback_text"] = ""
    value = _redact_jsonable(value)
    if action_text is not None:
        value["action_text"] = action_text
    if feedback_text is not None:
        value["feedback_text"] = feedback_text
    return value


def _as_messages(value: Any, *, max_chars: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            message = _redact_jsonable(item)
        else:
            message = {"role": "user", "content": _bounded_text(item, max_chars)}
        if "content" in message:
            message["content"] = _bounded_text(message.get("content"), max_chars)
        messages.append(message)
    return messages


def _messages_text(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class TerminalTransition:
    """A redacted turn-level transition used by the offline latent model."""

    trajectory_id: str
    task_name: str | None
    data_source: str | None
    turn_idx: int
    context_messages: list[dict[str, Any]]
    action_text: str
    feedback_text: str
    next_context_messages: list[dict[str, Any]] | None
    done: bool
    reward: float | None
    status: str | None
    source_path: str
    task_id: str | None = None
    task_cluster_id: str | None = None
    rollout_id: int | None = None
    train_step: int | None = None
    group_index: int | None = None
    sample_index: int | None = None
    has_tool_result: bool = False
    feedback_source: str = "unknown"
    action_view_schema: str = "unknown"
    tool_names: tuple[str, ...] = ()
    reward_label_scope: str | None = None
    reward_label_source: str | None = None
    reward_label_semantics: str | None = None
    reward_label_is_execution_outcome: bool | None = None
    reward_label_terminal: bool = False

    @property
    def context_text(self) -> str:
        return _messages_text(_redact_jsonable(self.context_messages))

    @property
    def next_context_text(self) -> str | None:
        if not self.next_context_messages:
            return None
        return _messages_text(_redact_jsonable(self.next_context_messages))

    @property
    def context_hash(self) -> str:
        return stable_hash(self.context_text)

    @property
    def transition_id(self) -> str:
        action_text = (
            _validated_canonical_action_text(self.action_text)[0]
            if self.action_view_schema == _CANONICAL_ACTION_VIEW
            else redact_sensitive_text(self.action_text)
        )
        feedback_text = (
            _validated_result_only_text(self.feedback_text)[0]
            if self.feedback_source == _CANONICAL_RESULT_SOURCE
            else redact_sensitive_text(self.feedback_text)
        )
        return stable_hash(
            {
                "trajectory_id": redact_sensitive_text(self.trajectory_id),
                "turn_idx": self.turn_idx,
                "context_hash": self.context_hash,
                "action_text": action_text,
                "feedback_text": feedback_text,
            }
        )

    @property
    def has_next(self) -> bool:
        return bool(self.next_context_messages)

    def to_dict(self) -> dict[str, Any]:
        value = _redacted_transition_value(self)
        action_text = str(value.get("action_text") or "")
        feedback_text = str(value.get("feedback_text") or "")
        trajectory_id = str(value.get("trajectory_id") or "")
        value.update(
            {
                "schema": "openclaw_terminal_transition_v3",
                "transition_id": self.transition_id,
                "trajectory_id": trajectory_id,
                "uid": trajectory_id,
                "has_next": self.has_next,
                "context_text": self.context_text,
                "context_hash": self.context_hash,
                "context_hash_schema": "canonical_redacted_text_v1",
                "action_hash": stable_hash(action_text),
                "next_observation_text": feedback_text,
                "next_observation_hash": stable_hash(feedback_text),
                "next_context_text": self.next_context_text,
                "next_context_hash": stable_hash(self.next_context_text) if self.next_context_text else None,
                "reward_score": self.reward,
                "redaction_applied": True,
            }
        )
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TerminalTransition":
        fields = cls.__dataclass_fields__
        kwargs = {key: value[key] for key in fields if key in value}
        kwargs["tool_names"] = tuple(kwargs.get("tool_names") or ())
        return cls(**kwargs)


def _canonicalize_transition(transition: TerminalTransition) -> TerminalTransition:
    value = _redacted_transition_value(transition)
    return TerminalTransition.from_dict(value)


def _tool_names(turn: dict[str, Any]) -> tuple[str, ...]:
    names = []
    for call in turn.get("tool_calls") or []:
        if isinstance(call, dict):
            names.append(str(call.get("tool_name") or call.get("name") or "tool"))
    return tuple(names)


def _action_text(turn: dict[str, Any], *, max_chars: int) -> str:
    parts: list[str] = []
    assistant_output = _bounded_text(turn.get("assistant_output") or "", max_chars).strip()
    if assistant_output:
        parts.append(assistant_output)
    for call in turn.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name") or call.get("name") or "tool")
        args = _redact_jsonable(call.get("args", call.get("arguments", {})))
        rendered = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        signature = f"{name}({rendered})"
        if signature not in assistant_output:
            parts.append(signature)
    return _bounded_text("\n".join(parts), max_chars)


def _feedback_text(
    turn: dict[str, Any],
    *,
    status: Any,
    is_terminal: bool,
    max_chars: int,
) -> tuple[str, bool, str]:
    parts: list[str] = []
    for call in turn.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("result") is None:
            continue
        name = str(call.get("tool_name") or call.get("name") or "tool")
        result = _bounded_text(call.get("result"), max_chars)
        parts.append(f"<tool_result name={name}>\n{result}\n</tool_result>")
    if parts:
        return _bounded_text("\n\n".join(parts), max_chars), True, "tool_result"
    if is_terminal:
        payload = {"observation_type": "terminal_status", "status": status}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str), False, "terminal_status"
    return '{"status":"no_tool_result"}', False, "no_tool_result"


def _turn_reward(
    reward: dict[str, Any],
    turn_idx: int,
    *,
    is_terminal: bool,
) -> tuple[float | None, dict[str, Any]]:
    for row in reward.get("per_turn_scores") or []:
        if not isinstance(row, dict) or int(row.get("turn_idx", -1)) != turn_idx:
            continue
        value = _finite_float(row.get("score"))
        if value is not None:
            return value, {
                "reward_label_scope": "per_turn",
                "reward_label_source": "reward.per_turn_scores.score",
                "reward_label_semantics": str(row.get("semantics") or "training_step_score_unspecified"),
                "reward_label_is_execution_outcome": (
                    row.get("reward_label_is_execution_outcome")
                    if isinstance(row.get("reward_label_is_execution_outcome"), bool)
                    else None
                ),
                "reward_label_terminal": is_terminal,
            }
    if not is_terminal:
        return None, {
            "reward_label_scope": "missing_nonterminal",
            "reward_label_source": None,
            "reward_label_semantics": None,
            "reward_label_is_execution_outcome": None,
            "reward_label_terminal": False,
        }
    for key in ("score", "base_score", "raw_score"):
        value = _finite_float(reward.get(key))
        if value is not None:
            return value, {
                "reward_label_scope": "trajectory_terminal",
                "reward_label_source": f"reward.{key}",
                "reward_label_semantics": "trajectory_training_reward_unspecified",
                "reward_label_is_execution_outcome": (
                    reward.get("reward_label_is_execution_outcome")
                    if isinstance(reward.get("reward_label_is_execution_outcome"), bool)
                    else None
                ),
                "reward_label_terminal": True,
            }
    return None, {
        "reward_label_scope": "missing",
        "reward_label_source": None,
        "reward_label_semantics": None,
        "reward_label_is_execution_outcome": None,
        "reward_label_terminal": is_terminal,
    }


def transitions_from_seta_trajectory(
    payload: dict[str, Any],
    *,
    source_path: str,
    max_text_chars: int = 4096,
) -> list[TerminalTransition]:
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    reward = payload.get("reward") if isinstance(payload.get("reward"), dict) else {}
    turns = [turn for turn in (payload.get("turns") or []) if isinstance(turn, dict)]
    if any(len(turn.get("sdk_model_turns") or []) > 1 for turn in turns):
        raise ValueError("multi-interaction SETA turn requires an explicit harness adapter")
    trajectory_id = str(info.get("uid") or Path(source_path).parent.name or stable_hash(source_path))
    terminal_status = str(info.get("status")) if info.get("status") is not None else None
    transitions: list[TerminalTransition] = []
    for index, turn in enumerate(turns):
        turn_idx = int(turn.get("turn_idx", index))
        is_terminal = index == len(turns) - 1
        next_messages = None
        if not is_terminal:
            next_messages = _as_messages(
                turns[index + 1].get("context_messages"),
                max_chars=max_text_chars,
            )
        feedback_text, has_tool_result, feedback_source = _feedback_text(
            turn,
            status=terminal_status if is_terminal else "in_progress",
            is_terminal=is_terminal,
            max_chars=max_text_chars,
        )
        reward_value, reward_contract = _turn_reward(reward, turn_idx, is_terminal=is_terminal)
        transitions.append(
            TerminalTransition(
                trajectory_id=trajectory_id,
                task_name=str(info.get("task_name")) if info.get("task_name") is not None else None,
                data_source=str(info.get("data_source")) if info.get("data_source") is not None else None,
                turn_idx=turn_idx,
                context_messages=_as_messages(turn.get("context_messages"), max_chars=max_text_chars),
                action_text=_action_text(turn, max_chars=max_text_chars),
                feedback_text=feedback_text,
                next_context_messages=next_messages,
                done=is_terminal,
                reward=reward_value,
                status=terminal_status if is_terminal else "in_progress",
                source_path=source_path,
                task_id=str(info.get("task_id")) if info.get("task_id") is not None else None,
                task_cluster_id=(
                    str(info.get("task_cluster_id"))
                    if info.get("task_cluster_id") is not None
                    else None
                ),
                rollout_id=info.get("rollout_id"),
                train_step=info.get("train_step"),
                group_index=info.get("group_index"),
                sample_index=info.get("sample_index"),
                has_tool_result=has_tool_result,
                feedback_source=feedback_source,
                tool_names=_tool_names(turn),
                **reward_contract,
            )
        )
    return transitions


def _messages_from_record(record: dict[str, Any], *, max_chars: int) -> list[dict[str, Any]]:
    if record.get("context_messages"):
        return _as_messages(record["context_messages"], max_chars=max_chars)
    text = record.get("context_text")
    if not text:
        return []
    try:
        parsed = json.loads(str(text))
    except (TypeError, json.JSONDecodeError):
        return [{"role": "user", "content": _bounded_text(text, max_chars)}]
    if isinstance(parsed, dict) and "context_messages" in parsed:
        return _as_messages(parsed["context_messages"], max_chars=max_chars)
    return _as_messages(parsed, max_chars=max_chars)


def transition_from_world_model_record(
    record: dict[str, Any],
    *,
    source_path: str,
    max_text_chars: int = 4096,
    allow_unverified_world_model_views: bool = False,
) -> TerminalTransition:
    schema = record.get("schema")
    if schema is not None and schema not in _SUPPORTED_RECORD_SCHEMAS:
        raise ValueError(f"unsupported world-model record schema: {schema!r}")
    if record.get("world_model_skipped"):
        raise ValueError("world-model skipped records cannot be converted into transitions")
    status = str(record.get("status")) if record.get("status") is not None else None
    reward = _finite_float(record.get("reward_score", record.get("reward")))
    feedback_source = str(
        record.get("feedback_source") or "world_model_record"
    )
    action_view_schema = str(
        record.get("action_view_schema") or "unknown"
    )
    if action_view_schema == _CANONICAL_ACTION_VIEW:
        action_text, canonical_action_view = _validated_canonical_action_text(
            record.get("action_text"),
            max_chars=max_text_chars,
            allow_invalid=allow_unverified_world_model_views,
        )
        if allow_unverified_world_model_views and not canonical_action_view:
            action_view_schema = f"{_CANONICAL_ACTION_VIEW}_unverified"
    else:
        action_text = _bounded_text(
            record.get("action_text") or "",
            max_text_chars,
        )
    if feedback_source == _CANONICAL_RESULT_SOURCE:
        feedback_raw = record.get("next_observation_text") or record.get("feedback_text")
        feedback_text, canonical_feedback_view = _validated_result_only_text(
            feedback_raw,
            max_chars=max_text_chars,
            allow_invalid=allow_unverified_world_model_views,
        )
        if allow_unverified_world_model_views and not canonical_feedback_view:
            feedback_source = f"{_CANONICAL_RESULT_SOURCE}_unverified"
    else:
        feedback_text = _bounded_text(
            record.get("next_observation_text")
            or record.get("feedback_text")
            or "",
            max_text_chars,
        )
    next_context_messages = None
    if record.get("next_context_messages"):
        next_context_messages = _as_messages(record.get("next_context_messages"), max_chars=max_text_chars)
    elif record.get("next_context_text"):
        next_context_messages = _messages_from_record(
            {"context_text": record.get("next_context_text")},
            max_chars=max_text_chars,
        )
    return TerminalTransition(
        trajectory_id=str(record.get("uid") or record.get("trajectory_id") or stable_hash(record)),
        task_name=str(record.get("task_name")) if record.get("task_name") is not None else None,
        data_source=str(record.get("data_source")) if record.get("data_source") is not None else None,
        turn_idx=int(record.get("turn_idx", 0) or 0),
        context_messages=_messages_from_record(record, max_chars=max_text_chars),
        action_text=action_text,
        feedback_text=feedback_text,
        next_context_messages=next_context_messages,
        done=bool(record.get("done", False)),
        reward=reward,
        status=status,
        source_path=str(record.get("source_path") or source_path),
        task_id=str(record.get("task_id")) if record.get("task_id") is not None else None,
        task_cluster_id=(
            str(record.get("task_cluster_id"))
            if record.get("task_cluster_id") is not None
            else None
        ),
        rollout_id=record.get("rollout_id"),
        train_step=record.get("train_step"),
        group_index=record.get("group_index"),
        sample_index=record.get("sample_index"),
        has_tool_result=bool(record.get("has_tool_result", False)),
        feedback_source=feedback_source,
        action_view_schema=action_view_schema,
        tool_names=tuple(record.get("tool_names") or ()),
        reward_label_scope=record.get("reward_label_scope"),
        reward_label_source=record.get("reward_label_source"),
        reward_label_semantics=record.get("reward_label_semantics"),
        reward_label_is_execution_outcome=(
            record.get("reward_label_is_execution_outcome")
            if isinstance(record.get("reward_label_is_execution_outcome"), bool)
            else None
        ),
        reward_label_terminal=bool(record.get("reward_label_terminal", False)),
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"world-model JSONL rows must be objects: {path}")
                yield value


def _load_pt_records(path: Path, *, allow_unverified_replay: bool) -> list[dict[str, Any]]:
    from .replay_buffer import TrajectoryReplayBuffer

    replay = TrajectoryReplayBuffer.load(
        path,
        require_verified=not allow_unverified_replay,
        allow_unverified_records=allow_unverified_replay,
    )
    return replay.records()


def load_terminal_transitions(
    input_path: str | Path,
    *,
    max_trajectories: int | None = None,
    max_transitions: int | None = None,
    require_tool_feedback: bool = False,
    allow_unverified_replay: bool = False,
    allow_unverified_world_model_views: bool = False,
    max_text_chars: int = 4096,
) -> list[TerminalTransition]:
    """Load raw SETA ``traj.json``, records JSONL, or verified replay snapshots."""

    root = Path(input_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"World-model input does not exist: {root}")
    if root.is_dir():
        paths = sorted(root.rglob("traj.json"))
        if not paths:
            paths = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.pt"))
    else:
        paths = [root]
    if max_trajectories is not None:
        paths = paths[: max(0, int(max_trajectories))]

    transitions: list[TerminalTransition] = []
    seen_transition_ids: set[str] = set()
    for path in paths:
        if path.suffix in {".pt", ".pth"}:
            rows = _load_pt_records(path, allow_unverified_replay=allow_unverified_replay)
            batch = [
                transition_from_world_model_record(
                    row,
                    source_path=str(path),
                    max_text_chars=max_text_chars,
                    allow_unverified_world_model_views=allow_unverified_replay,
                )
                for row in rows
            ]
        elif path.suffix == ".jsonl":
            batch = [
                transition_from_world_model_record(
                    row,
                    source_path=str(path),
                    max_text_chars=max_text_chars,
                    allow_unverified_world_model_views=allow_unverified_world_model_views,
                )
                for row in _iter_jsonl(path)
            ]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError(f"SETA trajectory must be a JSON object: {path}")
            batch = transitions_from_seta_trajectory(
                payload,
                source_path=str(path),
                max_text_chars=max_text_chars,
            )
        for transition in batch:
            transition = _canonicalize_transition(transition)
            if require_tool_feedback and not transition.has_tool_result:
                continue
            if not transition.action_text or not transition.feedback_text:
                raise ValueError(
                    f"transition has an empty action or feedback payload: {path}"
                )
            if transition.transition_id in seen_transition_ids:
                raise ValueError(
                    f"duplicate transition_id in world-model input: {transition.transition_id}"
                )
            seen_transition_ids.add(transition.transition_id)
            transitions.append(transition)
            if max_transitions is not None and len(transitions) >= int(max_transitions):
                return transitions
    return transitions
