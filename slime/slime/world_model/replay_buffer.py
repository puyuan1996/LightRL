from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch

from .metadata import redact_sensitive_jsonable, stable_hash


_REPLAY_SCHEMA_V1 = "openclaw_terminal_wm_replay_v1"
_REPLAY_SCHEMA_V2 = "openclaw_terminal_wm_replay_v2"


def _records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _redact_jsonable(value: Any) -> Any:
    return redact_sensitive_jsonable(value)


def _redact_replay_record(
    record: dict[str, Any],
    *,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Redact a record without corrupting canonical structured text views.

    In strict mode, canonical schemas must remain parseable; in diagnostic mode,
    malformed canonical views are preserved as redacted non-canonical variants.
    """

    protected: dict[str, str] = {}
    value = dict(record)
    if value.get("action_view_schema") == "tool_call_bundle_v1":
        from .action_view import parse_tool_call_bundle, render_tool_call_bundle

        action = str(value.get("action_text") or "")
        try:
            parsed_action = parse_tool_call_bundle(action)
        except ValueError:
            if not allow_unverified:
                raise
            value["action_view_schema"] = "tool_call_bundle_v1_unverified"
            protected["action_text"] = _redact_jsonable(action)
            value["action_text"] = ""
        else:
            protected["action_text"] = render_tool_call_bundle(parsed_action)
            value["action_text"] = ""
    if value.get("feedback_source") == "result_only_v1":
        from .result_view import parse_result_only_view, render_result_only_view

        for field in ("feedback_text", "next_observation_text"):
            text = value.get(field)
            if text is None:
                continue
            feedback_text = str(text)
            try:
                results = parse_result_only_view(feedback_text)
            except ValueError:
                if not allow_unverified:
                    raise
                value["feedback_source"] = "result_only_v1_unverified"
                protected[field] = _redact_jsonable(feedback_text)
                value[field] = ""
            else:
                protected[field] = render_result_only_view(
                    [redact_sensitive_jsonable(result) for result in results]
                )
                value[field] = ""
    sanitized = _redact_jsonable(value)
    sanitized.update(protected)
    return sanitized


class TrajectoryReplayBuffer:
    """Fixed-capacity replay buffer for DAPO-collected world-model records.

    The public ``push(entries, current_step)`` / ``sample(n, current_step,
    baseline_reward)`` shape follows the replay interface used by local PR #16.
    Unlike the PR's SIL buffer, this buffer defaults to admitting both success
    and failure transitions because latent dynamics need the full outcome
    distribution.
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        *,
        score_threshold: float | None = None,
        seed: int = 42,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        self.buffer_size = int(buffer_size)
        self.score_threshold = score_threshold
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._records: deque[dict[str, Any]] = deque(maxlen=self.buffer_size)
        self._ids: set[str] = set()
        self.total_admitted = 0
        self.total_rejected = 0
        self.total_sampled = 0
        self.total_evicted = 0
        self.provenance_verified = True

    @staticmethod
    def _record_id(record: dict[str, Any]) -> str:
        return str(
            record.get("transition_id")
            or stable_hash(
                {
                    "uid": record.get("uid") or record.get("trajectory_id"),
                    "turn_idx": record.get("turn_idx"),
                    "context_hash": record.get("context_hash"),
                    "action_hash": record.get("action_hash") or record.get("action_text"),
                }
            )
        )

    def push(self, entries: Iterable[dict[str, Any] | Any], current_step: int = 0) -> int:
        admitted_before = self.total_admitted
        for entry in entries:
            if hasattr(entry, "to_dict"):
                record = dict(entry.to_dict())
            elif isinstance(entry, dict):
                record = dict(entry)
            else:
                raise TypeError(f"Replay entries must be dict-like, got {type(entry).__name__}")
            record = _redact_replay_record(record)
            record["redaction_applied"] = True
            if record.get("world_model_skipped"):
                self.total_rejected += 1
                continue
            reward = record.get("reward_score", record.get("reward"))
            if self.score_threshold is not None and (reward is None or float(reward) < self.score_threshold):
                self.total_rejected += 1
                continue
            record_id = self._record_id(record)
            if record_id in self._ids:
                self.total_rejected += 1
                continue
            if len(self._records) == self.buffer_size:
                evicted = self._records[0]
                self._ids.discard(self._record_id(evicted))
                self.total_evicted += 1
            record["transition_id"] = record_id
            record["step_collected"] = int(current_step)
            self._records.append(record)
            self._ids.add(record_id)
            self.total_admitted += 1
        return self.total_admitted - admitted_before

    def sample(
        self,
        n: int,
        current_step: int = 0,
        baseline_reward: float | None = None,
    ) -> list[dict[str, Any]]:
        del current_step, baseline_reward
        if n <= 0 or not self._records:
            return []
        sampled = self._rng.sample(list(self._records), min(int(n), len(self._records)))
        self.total_sampled += len(sampled)
        return copy.deepcopy(sampled)

    def records(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._records))

    def state_dict(self) -> dict[str, Any]:
        records = self.records()
        return {
            "schema_version": _REPLAY_SCHEMA_V2,
            "buffer_size": self.buffer_size,
            "score_threshold": self.score_threshold,
            "seed": self.seed,
            "records": records,
            "records_sha256": _records_sha256(records),
            "rng_state": self._rng.getstate(),
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
            "total_sampled": self.total_sampled,
            "total_evicted": self.total_evicted,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        require_verified: bool = True,
        allow_unverified_records: bool = False,
    ) -> None:
        schema = state.get("schema_version")
        if schema not in {_REPLAY_SCHEMA_V1, _REPLAY_SCHEMA_V2}:
            raise ValueError(f"unsupported world-model replay schema: {schema!r}")
        records = state.get("records") or []
        if not isinstance(records, list):
            raise TypeError("world-model replay records must be a list")
        if len(records) > self.buffer_size:
            raise ValueError(
                "world-model replay record count exceeds its declared buffer capacity"
            )
        expected_digest = state.get("records_sha256")
        if schema == _REPLAY_SCHEMA_V2:
            if not expected_digest or expected_digest != _records_sha256(records):
                raise ValueError("world-model replay records digest mismatch")
            self.provenance_verified = not allow_unverified_records
        else:
            self.provenance_verified = False
            if require_verified:
                raise ValueError(
                    "legacy replay has no records digest; rebuild it or explicitly allow an unverified diagnostic"
                )
        self._records.clear()
        self._ids.clear()
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                if schema == _REPLAY_SCHEMA_V2:
                    raise TypeError(f"verified world-model replay record {index} is not an object")
                continue
            record = dict(record)
            sanitized = _redact_replay_record(
                record,
                allow_unverified=allow_unverified_records and schema == _REPLAY_SCHEMA_V2,
            )
            sanitized["redaction_applied"] = True
            if schema == _REPLAY_SCHEMA_V2 and sanitized != record and not allow_unverified_records:
                raise ValueError("verified world-model replay contains an unredacted record")
            record = sanitized
            record_id = self._record_id(record)
            record["transition_id"] = record_id
            if record_id in self._ids:
                if schema == _REPLAY_SCHEMA_V2:
                    raise ValueError(
                        f"verified world-model replay contains duplicate transition_id: {record_id}"
                    )
                continue
            if len(self._records) == self.buffer_size:
                evicted = self._records[0]
                self._ids.discard(self._record_id(evicted))
            self._records.append(record)
            self._ids.add(record_id)
        if schema == _REPLAY_SCHEMA_V2 and len(self._records) != len(records):
            raise RuntimeError("verified world-model replay load changed the record count")
        self.total_admitted = int(state.get("total_admitted", len(self._records)))
        self.total_rejected = int(state.get("total_rejected", 0))
        self.total_sampled = int(state.get("total_sampled", 0))
        self.total_evicted = int(state.get("total_evicted", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.setstate(rng_state)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_verified: bool = True,
        allow_unverified_records: bool = False,
    ) -> "TrajectoryReplayBuffer":
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "world_model_replay" in state:
            state = state["world_model_replay"]
        if not isinstance(state, dict):
            raise TypeError(f"Expected replay state dict, got {type(state).__name__}")
        buffer = cls(
            buffer_size=int(state.get("buffer_size", max(1, len(state.get("records") or [])))),
            score_threshold=state.get("score_threshold"),
            seed=int(state.get("seed", 42)),
        )
        buffer.load_state_dict(
            state,
            require_verified=require_verified,
            allow_unverified_records=allow_unverified_records,
        )
        return buffer

    def stats(self) -> dict[str, float]:
        attempted = self.total_admitted + self.total_rejected
        return {
            "wm_replay_size": float(len(self)),
            "wm_replay_capacity": float(self.buffer_size),
            "wm_replay_total_admitted": float(self.total_admitted),
            "wm_replay_total_rejected": float(self.total_rejected),
            "wm_replay_total_sampled": float(self.total_sampled),
            "wm_replay_total_evicted": float(self.total_evicted),
            "wm_replay_admit_rate": float(self.total_admitted) / max(attempted, 1),
            "wm_replay_provenance_verified": float(self.provenance_verified),
        }

    def __len__(self) -> int:
        return len(self._records)


def world_model_records_from_samples(
    samples: Iterable[Any],
    *,
    require_nonempty: bool = False,
) -> list[dict[str, Any]]:
    """Extract attached world-model metadata from flat or grouped samples."""

    records: list[dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, (list, tuple)):
            records.extend(world_model_records_from_samples(sample))
            continue
        train_metadata = getattr(sample, "train_metadata", None)
        metadata = getattr(sample, "metadata", None)
        train_metadata = train_metadata if isinstance(train_metadata, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        record = train_metadata.get("world_model") or metadata.get("world_model")
        if isinstance(record, dict):
            records.append(dict(record))
    if require_nonempty and not records:
        raise RuntimeError(
            "world-model replay collection produced no transition metadata; "
            "verify --world-model-enable and the rollout harness adapter"
        )
    return records
