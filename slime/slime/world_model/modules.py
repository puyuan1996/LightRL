from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .metrics import action_delta, effective_rank, mean_cosine_distance


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() != 3:
        raise ValueError(f"Expected hidden tensor with 2 or 3 dims, got {tuple(x.shape)}")
    if mask is None:
        return x.mean(dim=1)
    mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer adapted for the text-latent probe."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)
        if proj.size(-2) < 2:
            return proj.sum() * 0.0
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
        # Projectors emit unit-normalized samples. A random unit direction has
        # variance 1 / D on the unit sphere, so sqrt(D) is required before
        # comparing its characteristic function with N(0, 1).
        standardized = (proj @ a) * math.sqrt(proj.size(-1))
        x_t = standardized.unsqueeze(-1) * self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class StableProjector(nn.Module):
    """Normalize and project LLM hidden states into a controlled latent space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        *,
        clip_value: float = 30.0,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim, latent_dim)
        self.clip_value = float(clip_value)
        self.output_norm = output_norm
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(min=-self.clip_value, max=self.clip_value)
        z = self.net(x)
        if self.output_norm:
            z = F.normalize(z, dim=-1)
        return z


class ActionConditionedPredictor(nn.Module):
    """Legacy concat-MLP predictor kept for checkpoint compatibility/ablation."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        *,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or latent_dim * 4
        self.output_norm = bool(output_norm)
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, state_latent: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        pred = self.net(torch.cat([state_latent, action_latent], dim=-1))
        return F.normalize(pred, dim=-1) if self.output_norm else pred


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def _action_contrast_terms(
    pred: torch.Tensor,
    shuffled_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a dimension-independent cosine hinge and target-distance gap."""

    pos_dist = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)
    neg_dist = 1.0 - F.cosine_similarity(shuffled_pred.float(), target.float(), dim=-1)
    return F.relu(float(margin) + pos_dist - neg_dist).mean(), (neg_dist - pos_dist).mean()


class ActionAdaLNBlock(nn.Module):
    """Transformer block whose normalization and residual gates are action-conditioned.

    Observation/state latents are the only self-attention tokens.  The action
    latent is mapped to AdaLN shift/scale/gate parameters and is never appended
    to the token sequence.
    """

    def __init__(self, latent_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(f"latent_dim={latent_dim} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(latent_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, latent_dim),
        )
        self.action_to_adaln = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim * 6))

    def forward(
        self,
        state_tokens: torch.Tensor,
        action_condition: torch.Tensor,
        *,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.action_to_adaln(
            action_condition
        ).chunk(6, dim=-1)
        attn_input = _modulate(self.norm1(state_tokens), shift_attn, scale_attn)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        state_tokens = state_tokens + gate_attn * attn_out
        mlp_input = _modulate(self.norm2(state_tokens), shift_mlp, scale_mlp)
        return state_tokens + gate_mlp * self.mlp(mlp_input)


class ActionConditionedTransformerPredictor(nn.Module):
    """LeWM-style latent predictor with action AdaLN conditioning.

    Inputs may be independent transitions ``(B, D)`` or turn sequences
    ``(B, T, D)``.  In both cases, only state latents enter self-attention.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_turns: int = 64,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        self.max_turns = int(max_turns)
        self.output_norm = bool(output_norm)
        self.position = nn.Parameter(torch.zeros(1, self.max_turns, latent_dim))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            [ActionAdaLNBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim * 2))
        self.output = nn.Linear(latent_dim, latent_dim)

    def forward(self, state_latent: torch.Tensor, action_latent: torch.Tensor) -> torch.Tensor:
        squeeze_turn = state_latent.dim() == 2
        if squeeze_turn:
            state_latent = state_latent.unsqueeze(1)
        if action_latent.dim() == 2:
            action_latent = action_latent.unsqueeze(1)
        if state_latent.dim() != 3 or action_latent.dim() != 3:
            raise ValueError(
                "state_latent and action_latent must have shape (B,D) or (B,T,D); "
                f"got {tuple(state_latent.shape)} and {tuple(action_latent.shape)}"
            )
        if state_latent.shape != action_latent.shape:
            raise ValueError(
                f"state/action latent shapes must match, got {tuple(state_latent.shape)} and "
                f"{tuple(action_latent.shape)}"
            )
        turns = state_latent.size(1)
        if turns > self.max_turns:
            raise ValueError(f"turn sequence length {turns} exceeds max_turns={self.max_turns}")

        x = state_latent + self.position[:, :turns].to(dtype=state_latent.dtype)
        causal_mask = None
        if turns > 1:
            causal_mask = torch.triu(
                torch.ones(turns, turns, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        for block in self.blocks:
            x = block(x, action_latent, causal_mask=causal_mask)
        shift, scale = self.final_adaln(action_latent).chunk(2, dim=-1)
        pred = self.output(_modulate(self.final_norm(x), shift, scale))
        if self.output_norm:
            pred = F.normalize(pred, dim=-1)
        return pred.squeeze(1) if squeeze_turn else pred


@dataclass
class TextLatentWorldModelConfig:
    state_hidden_dim: int
    action_hidden_dim: int
    target_hidden_dim: int
    latent_dim: int = 1024
    adapter_dim: int | None = None
    projector_hidden_dim: int | None = None
    predictor_hidden_dim: int | None = None
    # ``None`` identifies PR #19 checkpoints, whose config predates this field.
    # New trainers must set predictor_type explicitly and use the shared-latent
    # architecture below.
    predictor_type: str | None = None
    architecture_version: str | None = None
    # Keep the established feedback objective as the compatibility default.
    # ``next_state`` selects the LeWM-style direct future-state objective.
    prediction_target: str = "feedback"
    # The ablation modes preserve the full model layout while removing either
    # the action or state signal from the predictor.
    predictor_input_mode: str = "observed"
    # ``residual`` predicts an unconstrained update and reconstructs the next
    # latent from the current state. ``direct`` preserves the established path.
    prediction_form: str = "direct"
    predictor_depth: int = 2
    predictor_num_heads: int = 1
    predictor_mlp_ratio: float = 4.0
    predictor_max_turns: int = 64
    clip_value: float = 30.0
    sigreg_num_proj: int = 1024
    action_contrast_margin: float = 0.05
    value_head: bool = True
    uncertainty_head: bool = True
    # Offline LeWM follows the upstream joint-embedding objective by default.
    # Online callers can enable stop-gradient/EMA targets explicitly.
    stop_grad_target: bool = False
    target_ema_decay: float = 0.996
    # ``frozen_random_orthogonal_v1`` provides one data-independent target
    # geometry shared by JEPA and direct baselines.
    target_geometry: str = "learned_shared_v2"
    fixed_target_seed: int = 20260731
    # Select which learned manifold receives anti-collapse regularization.
    # ``state`` preserves existing checkpoints and training behavior.
    sigreg_scope: str = "state"
    # Detached cross-microbatch samples keep contrastive and SIGReg terms
    # defined when end-to-end Qwen training requires batch_size=1.
    microbatch_queue_size: int = 0


class TextLatentWorldModel(nn.Module):
    """JEPA-style text world model for terminal-agent replay data.

    The module treats policy/frozen-encoder hidden states as raw material only.
    All branches pass through explicit projectors before entering the shared
    latent space, matching the controlled hidden-to-belief-latent design.
    """

    def __init__(self, config: TextLatentWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        architecture = config.architecture_version
        if architecture is None:
            architecture = "legacy_mlp_v1" if config.predictor_type is None else "shared_latent_v2"
        if architecture not in {"legacy_mlp_v1", "shared_latent_v2"}:
            raise ValueError(f"Unknown architecture_version={architecture!r}")
        self.architecture_version = architecture
        self.target_geometry = getattr(
            config, "target_geometry", "learned_shared_v2"
        )
        if self.target_geometry not in {
            "learned_shared_v2",
            "frozen_random_orthogonal_v1",
        }:
            raise ValueError(f"Unknown target_geometry={self.target_geometry!r}")
        if (
            self.target_geometry == "frozen_random_orthogonal_v1"
            and architecture != "shared_latent_v2"
        ):
            raise ValueError(
                "frozen_random_orthogonal_v1 requires shared_latent_v2"
            )
        if (
            self.target_geometry == "frozen_random_orthogonal_v1"
            and config.latent_dim > config.target_hidden_dim
        ):
            raise ValueError(
                "frozen target latent_dim cannot exceed target_hidden_dim"
            )
        if config.prediction_target not in {"feedback", "next_state"}:
            raise ValueError(
                f"Unknown prediction_target={config.prediction_target!r}; "
                "expected 'feedback' or 'next_state'"
            )
        if config.predictor_input_mode not in {"observed", "state_only", "action_only"}:
            raise ValueError(
                f"Unknown predictor_input_mode={config.predictor_input_mode!r}; "
                "expected 'observed', 'state_only', or 'action_only'"
            )
        if config.prediction_form not in {"direct", "residual"}:
            raise ValueError(
                f"Unknown prediction_form={config.prediction_form!r}; "
                "expected 'direct' or 'residual'"
            )
        if config.sigreg_scope not in {"state", "pred", "state_pred"}:
            raise ValueError(
                f"Unknown sigreg_scope={config.sigreg_scope!r}; "
                "expected 'state', 'pred', or 'state_pred'"
            )
        if int(config.microbatch_queue_size) < 0:
            raise ValueError("microbatch_queue_size must be non-negative")

        predictor_type = config.predictor_type or "mlp"
        if architecture == "legacy_mlp_v1" and predictor_type != "mlp":
            raise ValueError("legacy_mlp_v1 supports only predictor_type='mlp'")

        self.action_projector = StableProjector(
            config.action_hidden_dim,
            config.latent_dim,
            config.projector_hidden_dim,
            clip_value=config.clip_value,
        )
        if architecture == "legacy_mlp_v1":
            self.state_projector = StableProjector(
                config.state_hidden_dim,
                config.latent_dim,
                config.projector_hidden_dim,
                clip_value=config.clip_value,
            )
            self.target_projector = StableProjector(
                config.target_hidden_dim,
                config.latent_dim,
                config.projector_hidden_dim,
                clip_value=config.clip_value,
            )
        else:
            adapter_dim = config.adapter_dim or config.latent_dim
            self.state_adapter = StableProjector(
                config.state_hidden_dim,
                adapter_dim,
                config.projector_hidden_dim,
                clip_value=config.clip_value,
                output_norm=False,
            )
            self.target_adapter: StableProjector | None
            if self.target_geometry == "learned_shared_v2":
                self.target_adapter = StableProjector(
                    config.target_hidden_dim,
                    adapter_dim,
                    config.projector_hidden_dim,
                    clip_value=config.clip_value,
                    output_norm=False,
                )
            else:
                self.target_adapter = None
            self.shared_projector = StableProjector(
                adapter_dim,
                config.latent_dim,
                config.projector_hidden_dim,
                clip_value=config.clip_value,
            )
            self.target_shared_projector: StableProjector | None = None
            if config.stop_grad_target and self.target_geometry == "learned_shared_v2":
                if config.state_hidden_dim != config.target_hidden_dim:
                    raise ValueError(
                        "EMA target requires matching state_hidden_dim and target_hidden_dim"
                    )
                if not 0.0 <= config.target_ema_decay < 1.0:
                    raise ValueError("target_ema_decay must be in [0, 1)")
                self.target_adapter.load_state_dict(self.state_adapter.state_dict())
                self.target_shared_projector = copy.deepcopy(self.shared_projector)
                self.target_adapter.requires_grad_(False)
                self.target_shared_projector.requires_grad_(False)
            if self.target_geometry == "frozen_random_orthogonal_v1":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    int(getattr(config, "fixed_target_seed", 20260731))
                )
                matrix = torch.randn(
                    config.target_hidden_dim,
                    config.latent_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
                matrix, _ = torch.linalg.qr(matrix, mode="reduced")
                self.register_buffer(
                    "fixed_target_projection",
                    matrix.contiguous(),
                    persistent=True,
                )

        if predictor_type == "mlp":
            self.predictor = ActionConditionedPredictor(
                config.latent_dim,
                config.predictor_hidden_dim,
                output_norm=config.prediction_form == "direct",
            )
        elif predictor_type == "adaln":
            self.predictor = ActionConditionedTransformerPredictor(
                config.latent_dim,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                max_turns=config.predictor_max_turns,
                output_norm=config.prediction_form == "direct",
            )
        else:
            raise ValueError(f"Unknown predictor_type={predictor_type!r}; expected 'adaln' or 'mlp'")

        head_dim = config.latent_dim * 2 if architecture == "legacy_mlp_v1" else config.latent_dim
        self.value_head = nn.Linear(head_dim, 1) if config.value_head else None
        self.uncertainty_head = nn.Linear(head_dim, 1) if config.uncertainty_head else None
        self.sigreg = SIGReg(num_proj=config.sigreg_num_proj)
        self._queued_state_latent: torch.Tensor | None = None
        self._queued_action_latent: torch.Tensor | None = None
        self._queued_pred_latent: torch.Tensor | None = None

    def reset_microbatch_queue(self) -> None:
        self._queued_state_latent = None
        self._queued_action_latent = None
        self._queued_pred_latent = None

    def _append_microbatch_queue(self, name: str, value: torch.Tensor) -> None:
        limit = int(self.config.microbatch_queue_size)
        if limit <= 0:
            return
        current = getattr(self, name)
        detached = value.detach()
        combined = detached if current is None else torch.cat([current, detached], dim=0)
        setattr(self, name, combined[-limit:])

    def _predict_next(
        self,
        state_latent: torch.Tensor,
        action_latent: torch.Tensor,
    ) -> torch.Tensor:
        update = self.predictor(state_latent, action_latent)
        if self.config.prediction_form == "residual":
            return F.normalize(state_latent + update, dim=-1)
        return update

    def map_fixed_target_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map raw hidden states into the shared frozen target geometry."""

        if self.target_geometry != "frozen_random_orthogonal_v1":
            raise RuntimeError(
                "map_fixed_target_hidden requires frozen_random_orthogonal_v1"
            )
        if hidden.size(-1) != self.config.target_hidden_dim:
            raise ValueError(
                "fixed target hidden width mismatch: "
                f"expected={self.config.target_hidden_dim} actual={hidden.size(-1)}"
            )
        normalized = F.layer_norm(hidden.float(), (hidden.size(-1),))
        projection = self.fixed_target_projection.to(
            device=normalized.device,
            dtype=normalized.dtype,
        )
        return F.normalize(normalized @ projection, dim=-1)

    def map_target_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map target hidden states using the configured target geometry."""

        if self.target_geometry == "frozen_random_orthogonal_v1":
            return self.map_fixed_target_hidden(hidden)
        if self.architecture_version == "legacy_mlp_v1":
            return self.target_projector(hidden)
        if self.target_adapter is None:
            raise RuntimeError("learned target adapter is missing")
        target_projector = self.target_shared_projector or self.shared_projector
        return target_projector(self.target_adapter(hidden))

    def forward(
        self,
        *,
        state_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        target_hidden: torch.Tensor | None = None,
        next_state_hidden: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        next_state_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        state_feat = _masked_mean(state_hidden, state_mask)
        action_feat = _masked_mean(action_hidden, action_mask)
        if self.config.predictor_input_mode == "state_only":
            action_feat = torch.zeros_like(action_feat)
        elif self.config.predictor_input_mode == "action_only":
            state_feat = torch.zeros_like(state_feat)
        if self.architecture_version == "legacy_mlp_v1":
            state_latent = self.state_projector(state_feat)
        else:
            state_latent = self.shared_projector(self.state_adapter(state_feat))
        action_latent = self.action_projector(action_feat)
        pred_latent = self._predict_next(state_latent, action_latent)
        target_latent = None
        if target_hidden is not None:
            target_feat = _masked_mean(target_hidden, target_mask)
            target_latent = self.map_target_hidden(target_feat)

        next_state_latent = None
        next_state_target_latent = None
        if next_state_hidden is not None:
            next_state_feat = _masked_mean(next_state_hidden, next_state_mask)
            if self.architecture_version == "legacy_mlp_v1":
                next_state_latent = self.state_projector(next_state_feat)
                next_state_target_latent = next_state_latent
            else:
                next_state_latent = self.shared_projector(self.state_adapter(next_state_feat))
                if self.target_geometry == "frozen_random_orthogonal_v1":
                    next_state_target_latent = self.map_fixed_target_hidden(
                        next_state_feat
                    )
                elif self.config.prediction_target == "next_state" and self.config.stop_grad_target:
                    if self.target_shared_projector is None:
                        raise RuntimeError("EMA target projector is missing")
                    next_state_target_latent = self.target_shared_projector(
                        self.target_adapter(next_state_feat)
                    )
                else:
                    next_state_target_latent = next_state_latent

        head_input = (
            torch.cat([state_latent, action_latent], dim=-1)
            if self.architecture_version == "legacy_mlp_v1"
            else pred_latent
        )
        value = self.value_head(head_input).squeeze(-1) if self.value_head is not None else None
        uncertainty = None
        if self.uncertainty_head is not None:
            uncertainty = F.softplus(self.uncertainty_head(head_input).squeeze(-1))

        return {
            "state_latent": state_latent,
            "action_latent": action_latent,
            "pred_latent": pred_latent,
            "target_latent": target_latent,
            "next_state_latent": next_state_latent,
            "next_state_target_latent": next_state_target_latent,
            "value": value,
            "uncertainty": uncertainty,
        }

    @torch.no_grad()
    def update_target_ema(self) -> bool:
        """Update the shared-architecture target encoder after an optimizer step."""

        if (
            self.architecture_version != "shared_latent_v2"
            or not self.config.stop_grad_target
            or self.target_geometry == "frozen_random_orthogonal_v1"
        ):
            return False
        if self.target_shared_projector is None:
            raise RuntimeError("EMA target projector is missing")
        decay = float(self.config.target_ema_decay)
        pairs = (
            (self.target_adapter, self.state_adapter),
            (self.target_shared_projector, self.shared_projector),
        )
        for target_module, online_module in pairs:
            for target_parameter, online_parameter in zip(
                target_module.parameters(), online_module.parameters(), strict=True
            ):
                target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)
            for target_buffer, online_buffer in zip(
                target_module.buffers(), online_module.buffers(), strict=True
            ):
                target_buffer.copy_(online_buffer)
        return True

    def compute_loss(
        self,
        *,
        state_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        next_state_hidden: torch.Tensor | None = None,
        has_next: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
        reward_mask: torch.Tensor | None = None,
        pred_loss_type: str = "mse",
        sigreg_coef: float = 0.1,
        action_contrast_coef: float = 0.1,
        alignment_coef: float = 0.1,
        feedback_aux_coef: float = 0.0,
        value_coef: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(
            state_hidden=state_hidden,
            action_hidden=action_hidden,
            target_hidden=target_hidden,
            next_state_hidden=next_state_hidden,
        )
        pred = out["pred_latent"]
        feedback_target = out["target_latent"]
        next_state_target = out["next_state_target_latent"]
        prediction_mask = torch.ones(pred.size(0), dtype=torch.bool, device=pred.device)
        if self.config.prediction_target == "next_state":
            if next_state_target is None:
                raise ValueError("prediction_target='next_state' requires next_state_hidden")
            if has_next is None:
                raise ValueError("prediction_target='next_state' requires an explicit has_next mask")
            prediction_mask = has_next.to(device=pred.device, dtype=torch.bool).view(-1)
            if prediction_mask.numel() != pred.size(0):
                raise ValueError("has_next length must match the prediction batch")
            if not bool(prediction_mask.any().item()):
                raise ValueError("prediction_target='next_state' batch has no has_next=true samples")
            selected_pred = pred[prediction_mask]
            selected_target = next_state_target[prediction_mask]
        else:
            selected_pred = pred
            selected_target = feedback_target
        loss_target = selected_target.detach() if self.config.stop_grad_target else selected_target
        def prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if pred_loss_type == "cosine":
                return mean_cosine_distance(prediction, target)
            if pred_loss_type == "smooth_l1":
                return F.smooth_l1_loss(prediction, target)
            if pred_loss_type == "scaled_mse":
                return F.mse_loss(prediction, target) * prediction.size(-1)
            if pred_loss_type == "mse":
                return F.mse_loss(prediction, target)
            raise ValueError(f"Unknown pred_loss_type={pred_loss_type!r}")

        pred_loss = prediction_loss(selected_pred, loss_target)
        feedback_aux_loss = pred_loss * 0.0
        if feedback_aux_coef != 0.0:
            if self.config.prediction_target != "next_state":
                raise ValueError(
                    "feedback auxiliary supervision requires prediction_target='next_state'"
                )
            selected_feedback_target = feedback_target[prediction_mask]
            if self.config.stop_grad_target:
                selected_feedback_target = selected_feedback_target.detach()
            feedback_aux_loss = prediction_loss(
                selected_pred,
                selected_feedback_target,
            )

        selected_state = out["state_latent"][prediction_mask]
        selected_action = out["action_latent"][prediction_mask]
        queued_state = self._queued_state_latent if self.training else None
        queued_pred = self._queued_pred_latent if self.training else None
        sigreg_state = (
            torch.cat([queued_state, selected_state], dim=0)
            if queued_state is not None
            else selected_state
        )
        sigreg_pred = (
            torch.cat([queued_pred, selected_pred], dim=0)
            if queued_pred is not None
            else selected_pred
        )
        if self.config.sigreg_scope == "pred":
            sigreg_loss = self.sigreg(sigreg_pred)
        elif self.config.sigreg_scope == "state_pred":
            if int(self.config.microbatch_queue_size) > 0:
                sigreg_loss = self.sigreg(torch.cat([sigreg_state, sigreg_pred], dim=0))
            else:
                sigreg_loss = self.sigreg(torch.stack([selected_state, selected_pred], dim=0))
        elif self.architecture_version == "legacy_mlp_v1":
            sigreg_loss = self.sigreg(
                torch.stack([out["state_latent"][prediction_mask], loss_target], dim=0)
            )
        elif (
            self.config.stop_grad_target
            or self.target_geometry == "frozen_random_orthogonal_v1"
        ):
            # A frozen target branch cannot respond to regularization. Keep the
            # anti-collapse gradient on the learned state manifold only.
            sigreg_loss = self.sigreg(sigreg_state)
        else:
            # Jointly learned targets otherwise have a trivial solution: the
            # target adapter can collapse while the state manifold stays full
            # rank. Regularize both learned views symmetrically.
            sigreg_loss = self.sigreg(
                torch.stack([out["state_latent"][prediction_mask], loss_target], dim=0)
            )
        contrast_loss = pred_loss * 0.0
        delta = pred_loss * 0.0
        action_target_gap = pred_loss * 0.0
        negative_action_count = 0
        if selected_action.size(0) > 1:
            shuffled_action = torch.roll(selected_action, shifts=1, dims=0)
            negative_action_count = int(selected_action.size(0))
        elif self.training and self._queued_action_latent is not None:
            shuffled_action = self._queued_action_latent[-selected_action.size(0):]
            negative_action_count = int(shuffled_action.size(0))
        else:
            shuffled_action = None
        if shuffled_action is not None:
            if action_contrast_coef != 0.0:
                shuffled_pred = self._predict_next(selected_state, shuffled_action)
            else:
                with torch.no_grad():
                    shuffled_pred = self._predict_next(selected_state, shuffled_action)
            contrast_loss, action_target_gap = _action_contrast_terms(
                selected_pred,
                shuffled_pred,
                loss_target,
                margin=float(self.config.action_contrast_margin),
            )
            delta = action_delta(selected_pred, shuffled_pred)

        if self.training and int(self.config.microbatch_queue_size) > 0:
            self._append_microbatch_queue("_queued_state_latent", selected_state)
            self._append_microbatch_queue("_queued_action_latent", selected_action)
            self._append_microbatch_queue("_queued_pred_latent", selected_pred)

        alignment_loss = pred_loss * 0.0
        next_state_latent = out["next_state_latent"]
        if next_state_latent is not None and alignment_coef != 0.0:
            align_target = feedback_target.detach()
            if has_next is None:
                alignment_loss = F.mse_loss(next_state_latent, align_target)
            else:
                mask = has_next.to(device=pred.device, dtype=torch.bool).view(-1)
                if mask.any():
                    alignment_loss = F.mse_loss(next_state_latent[mask], align_target[mask])

        value_loss = pred_loss * 0.0
        if reward is not None and out["value"] is not None and value_coef != 0.0:
            reward = reward.float().view_as(out["value"])
            if reward_mask is not None:
                mask = reward_mask.to(device=out["value"].device, dtype=torch.bool).view_as(out["value"])
                if mask.any():
                    if self.architecture_version == "legacy_mlp_v1":
                        value_loss = F.mse_loss(out["value"][mask], reward[mask])
                    else:
                        value_loss = F.smooth_l1_loss(out["value"][mask], reward[mask])
            else:
                if self.architecture_version == "legacy_mlp_v1":
                    value_loss = F.mse_loss(out["value"], reward)
                else:
                    value_loss = F.smooth_l1_loss(out["value"], reward)

        loss = (
            pred_loss
            + sigreg_coef * sigreg_loss
            + action_contrast_coef * contrast_loss
            + alignment_coef * alignment_loss
            + feedback_aux_coef * feedback_aux_loss
            + value_coef * value_loss
        )
        next_metric_mask = (
            has_next.to(device=pred.device, dtype=torch.bool).view(-1)
            if has_next is not None
            else prediction_mask
        )
        has_next_metric = bool(next_metric_mask.any().item())
        metrics = {
            "wm/pred_loss": pred_loss.detach(),
            "wm/feedback_pred_mse": F.mse_loss(pred, feedback_target).detach(),
            "wm/next_state_pred_mse": (
                F.mse_loss(pred[next_metric_mask], next_state_target[next_metric_mask]).detach()
                if next_state_target is not None and has_next_metric
                else pred_loss.detach() * 0.0
            ),
            "wm/prediction_mask_count": prediction_mask.float().sum().detach(),
            "wm/sigreg_loss": sigreg_loss.detach(),
            "wm/action_contrast_loss": contrast_loss.detach(),
            "wm/alignment_loss": alignment_loss.detach(),
            "wm/feedback_aux_loss": feedback_aux_loss.detach(),
            "wm/value_loss": value_loss.detach(),
            "wm/value_mask_count": (
                reward_mask.to(device=pred_loss.device, dtype=torch.float32).sum().detach()
                if reward_mask is not None
                else torch.as_tensor(0.0 if reward is None else reward.numel(), device=pred_loss.device)
            ),
            "wm/effective_rank": effective_rank(out["state_latent"]).detach(),
            "wm/pred_effective_rank": effective_rank(selected_pred).detach(),
            "wm/target_effective_rank": effective_rank(loss_target).detach(),
            "wm/action_delta": delta.detach(),
            "wm/action_target_gap": action_target_gap.detach(),
            "wm/contrast_negative_count": torch.as_tensor(
                float(negative_action_count), device=pred_loss.device
            ),
            "wm/sigreg_sample_count": torch.as_tensor(
                float(sigreg_pred.size(0)), device=pred_loss.device
            ),
        }
        return loss, metrics
