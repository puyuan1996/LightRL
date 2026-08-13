import pytest
import torch
from torch import nn

from slime.world_model.modules import (
    SIGReg,
    TextLatentWorldModel,
    TextLatentWorldModelConfig,
    _action_contrast_terms,
)


class _CaptureSIGReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = None

    def forward(self, value):
        self.shape = tuple(value.shape)
        return value.sum() * 0.0


def test_text_latent_world_model_loss_backward():
    torch.manual_seed(0)
    config = TextLatentWorldModelConfig(
        state_hidden_dim=8,
        action_hidden_dim=6,
        target_hidden_dim=10,
        latent_dim=4,
        sigreg_num_proj=8,
    )
    model = TextLatentWorldModel(config)
    loss, metrics = model.compute_loss(
        state_hidden=torch.randn(5, 3, 8),
        action_hidden=torch.randn(5, 2, 6),
        target_hidden=torch.randn(5, 4, 10),
        reward=torch.randn(5),
        value_coef=0.1,
    )
    assert loss.ndim == 0
    assert "wm/pred_loss" in metrics
    assert "wm/action_delta" in metrics
    assert "wm/pred_effective_rank" in metrics
    assert "wm/target_effective_rank" in metrics
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())
    assert model.architecture_version == "legacy_mlp_v1"
    assert any(param.grad is not None for param in model.target_projector.parameters())


@pytest.mark.parametrize(
    ("scope", "expected_shape"),
    [("state", (5, 4)), ("pred", (5, 4)), ("state_pred", (2, 5, 4))],
)
def test_sigreg_scope_selects_the_registered_manifold(scope, expected_shape):
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=4,
            predictor_type="adaln",
            predictor_num_heads=2,
            target_geometry="frozen_random_orthogonal_v1",
            sigreg_scope=scope,
        )
    )
    capture = _CaptureSIGReg()
    model.sigreg = capture
    loss, _ = model.compute_loss(
        state_hidden=torch.randn(5, 4),
        action_hidden=torch.randn(5, 4),
        target_hidden=torch.randn(5, 4),
        sigreg_coef=0.1,
        action_contrast_coef=0.0,
        alignment_coef=0.0,
    )
    loss.backward()
    assert capture.shape == expected_shape


def test_invalid_sigreg_scope_is_rejected():
    with pytest.raises(ValueError, match="sigreg_scope"):
        TextLatentWorldModel(
            TextLatentWorldModelConfig(
                state_hidden_dim=4,
                action_hidden_dim=4,
                target_hidden_dim=4,
                latent_dim=4,
                predictor_type="adaln",
                predictor_num_heads=2,
                sigreg_scope="invalid",
            )
        )


def test_microbatch_queue_enables_batch_one_regularization_and_contrast():
    torch.manual_seed(7)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=4,
            predictor_type="adaln",
            predictor_num_heads=2,
            target_geometry="frozen_random_orthogonal_v1",
            sigreg_scope="pred",
            sigreg_num_proj=16,
            microbatch_queue_size=4,
        )
    )
    model.train()
    common = {
        "state_hidden": torch.randn(1, 4),
        "target_hidden": torch.randn(1, 4),
        "sigreg_coef": 0.1,
        "action_contrast_coef": 0.2,
        "alignment_coef": 0.0,
    }
    _, first = model.compute_loss(action_hidden=torch.randn(1, 4), **common)
    _, second = model.compute_loss(action_hidden=torch.randn(1, 4), **common)

    assert first["wm/contrast_negative_count"] == 0
    assert first["wm/sigreg_sample_count"] == 1
    assert second["wm/contrast_negative_count"] == 1
    assert second["wm/sigreg_sample_count"] == 2
    assert second["wm/action_delta"] > 0
    assert second["wm/sigreg_loss"] > 0

    model.reset_microbatch_queue()
    _, reset = model.compute_loss(action_hidden=torch.randn(1, 4), **common)
    assert reset["wm/contrast_negative_count"] == 0


def test_adaln_predictor_keeps_action_out_of_attention_tokens():
    torch.manual_seed(0)
    config = TextLatentWorldModelConfig(
        state_hidden_dim=4,
        action_hidden_dim=4,
        target_hidden_dim=4,
        latent_dim=4,
        predictor_type="adaln",
        predictor_num_heads=2,
        sigreg_num_proj=4,
    )
    model = TextLatentWorldModel(config)
    state = torch.randn(2, 3, 4)
    action = torch.randn(2, 3, 4)

    first = model.predictor(state, action)
    second = model.predictor(state, torch.roll(action, shifts=1, dims=0))

    assert first.shape == state.shape
    assert not torch.allclose(first, second)
    assert model.predictor.blocks[0].attn.embed_dim == config.latent_dim


def test_adaln_predictor_is_causal_across_turns():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=4,
            predictor_type="adaln",
            predictor_num_heads=2,
            sigreg_num_proj=4,
        )
    )
    state = torch.randn(2, 3, 4)
    action = torch.randn(2, 3, 4)
    changed_state = state.clone()
    changed_action = action.clone()
    changed_state[:, 2] += 100.0
    changed_action[:, 2] -= 100.0

    first = model.predictor(state, action)
    changed = model.predictor(changed_state, changed_action)

    assert torch.allclose(first[:, :2], changed[:, :2], atol=1e-6, rtol=1e-6)


def test_legacy_config_preserves_pr19_state_dict_layout():
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=3,
        )
    )

    keys = set(model.state_dict())
    assert model.architecture_version == "legacy_mlp_v1"
    assert any(key.startswith("state_projector.") for key in keys)
    assert any(key.startswith("target_projector.") for key in keys)
    assert not any(key.startswith("state_adapter.") for key in keys)


def test_text_latent_world_model_value_loss_honors_reward_mask():
    torch.manual_seed(0)
    config = TextLatentWorldModelConfig(
        state_hidden_dim=4,
        action_hidden_dim=4,
        target_hidden_dim=4,
        latent_dim=3,
        sigreg_num_proj=4,
    )
    model = TextLatentWorldModel(config)
    _loss, metrics = model.compute_loss(
        state_hidden=torch.randn(3, 4),
        action_hidden=torch.randn(3, 4),
        target_hidden=torch.randn(3, 4),
        reward=torch.tensor([100.0, -100.0, 3.0]),
        reward_mask=torch.tensor([False, False, False]),
        value_coef=1.0,
    )

    assert torch.allclose(metrics["wm/value_loss"], torch.tensor(0.0))
    assert torch.allclose(metrics["wm/value_mask_count"], torch.tensor(0.0))


def test_stop_grad_target_freezes_target_projector_for_all_losses():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=3,
            sigreg_num_proj=4,
            stop_grad_target=True,
        )
    )

    loss, _ = model.compute_loss(
        state_hidden=torch.randn(4, 4),
        action_hidden=torch.randn(4, 4),
        target_hidden=torch.randn(4, 4),
        sigreg_coef=0.1,
        action_contrast_coef=0.1,
    )
    loss.backward()

    assert all(parameter.grad is None for parameter in model.target_projector.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())


def test_stop_grad_target_freezes_shared_target_adapter_with_alignment():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=4,
            predictor_type="adaln",
            predictor_num_heads=2,
            sigreg_num_proj=4,
            stop_grad_target=True,
        )
    )

    loss, _ = model.compute_loss(
        state_hidden=torch.randn(4, 4),
        action_hidden=torch.randn(4, 4),
        target_hidden=torch.randn(4, 4),
        next_state_hidden=torch.randn(4, 4),
        has_next=torch.ones(4, dtype=torch.bool),
        sigreg_coef=0.1,
        action_contrast_coef=0.1,
        alignment_coef=0.1,
    )
    loss.backward()

    assert all(parameter.grad is None for parameter in model.target_adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())


def test_shared_joint_target_regularizes_both_views_but_stop_grad_does_not():
    def build(stop_grad_target):
        model = TextLatentWorldModel(
            TextLatentWorldModelConfig(
                state_hidden_dim=4,
                action_hidden_dim=4,
                target_hidden_dim=4,
                latent_dim=4,
                predictor_type="adaln",
                predictor_num_heads=2,
                sigreg_num_proj=4,
                stop_grad_target=stop_grad_target,
            )
        )
        model.sigreg = _CaptureSIGReg()
        model.compute_loss(
            state_hidden=torch.randn(4, 4),
            action_hidden=torch.randn(4, 4),
            target_hidden=torch.randn(4, 4),
            sigreg_coef=0.1,
            action_contrast_coef=0.0,
            alignment_coef=0.0,
        )
        return model

    joint = build(False)
    stopped = build(True)

    assert joint.sigreg.shape == (2, 4, 4)
    assert stopped.sigreg.shape == (4, 4)


def test_sigreg_standardizes_unit_sphere_and_penalizes_collapse():
    torch.manual_seed(0)
    regularizer = SIGReg(knots=17, num_proj=256)
    isotropic = torch.nn.functional.normalize(torch.randn(512, 128), dim=-1)
    collapsed = isotropic[:1].expand_as(isotropic)

    torch.manual_seed(7)
    isotropic_loss = regularizer(isotropic)
    torch.manual_seed(7)
    collapsed_loss = regularizer(collapsed)

    assert isotropic_loss < collapsed_loss
    assert isotropic_loss < 2.0

    torch.manual_seed(1)
    smaller_batch = torch.nn.functional.normalize(torch.randn(64, 128), dim=-1)
    torch.manual_seed(7)
    smaller_loss = regularizer(smaller_batch)
    assert 0.5 < float(smaller_loss / isotropic_loss) < 2.0


def test_cosine_action_contrast_has_a_feasible_zero_loss():
    target = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss, gap = _action_contrast_terms(target, -target, target, margin=0.05)

    assert loss.item() == 0.0
    assert gap.item() == pytest.approx(2.0)


def test_action_diagnostics_are_computed_when_contrast_loss_is_disabled():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            predictor_num_heads=2,
            sigreg_num_proj=8,
        )
    )
    _, metrics = model.compute_loss(
        state_hidden=torch.randn(8, 8),
        action_hidden=torch.randn(8, 8),
        target_hidden=torch.randn(8, 8),
        action_contrast_coef=0.0,
    )

    assert metrics["wm/action_delta"] > 0.0
    assert torch.isfinite(metrics["wm/action_target_gap"])


def test_input_ablation_modes_are_invariant_and_parameter_matched():
    torch.manual_seed(0)
    observed = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            predictor_num_heads=2,
            sigreg_num_proj=8,
            predictor_input_mode="observed",
        )
    )
    state_only = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            predictor_num_heads=2,
            sigreg_num_proj=8,
            predictor_input_mode="state_only",
        )
    )
    state_only.load_state_dict(observed.state_dict())
    state = torch.randn(6, 8)
    first_action = torch.randn(6, 8)
    second_action = torch.randn(6, 8)
    target = torch.randn(6, 8)

    first = state_only(
        state_hidden=state,
        action_hidden=first_action,
        target_hidden=target,
    )
    second = state_only(
        state_hidden=state,
        action_hidden=second_action,
        target_hidden=target,
    )

    assert torch.equal(first["pred_latent"], second["pred_latent"])
    assert torch.equal(first["action_latent"], second["action_latent"])
    assert sum(parameter.numel() for parameter in observed.parameters()) == sum(
        parameter.numel() for parameter in state_only.parameters()
    )

    action_only = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            predictor_num_heads=2,
            sigreg_num_proj=8,
            predictor_input_mode="action_only",
        )
    )
    action_only.load_state_dict(observed.state_dict())
    first = action_only(
        state_hidden=state,
        action_hidden=first_action,
        target_hidden=target,
    )
    second = action_only(
        state_hidden=torch.randn_like(state),
        action_hidden=first_action,
        target_hidden=target,
    )

    assert torch.equal(first["pred_latent"], second["pred_latent"])
    assert torch.equal(first["state_latent"], second["state_latent"])
    assert sum(parameter.numel() for parameter in observed.parameters()) == sum(
        parameter.numel() for parameter in action_only.parameters()
    )


def test_residual_prediction_form_reconstructs_state_for_zero_update():
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=4,
            predictor_type="mlp",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            prediction_form="residual",
            value_head=False,
            uncertainty_head=False,
        )
    )
    for parameter in model.predictor.parameters():
        parameter.data.zero_()
    output = model(
        state_hidden=torch.randn(3, 8),
        action_hidden=torch.randn(3, 8),
        target_hidden=torch.randn(3, 8),
        next_state_hidden=torch.randn(3, 8),
    )

    assert torch.allclose(output["pred_latent"], output["state_latent"], atol=1e-6)
    assert torch.allclose(output["pred_latent"].norm(dim=-1), torch.ones(3), atol=1e-6)


def test_invalid_predictor_input_mode_fails_closed():
    with pytest.raises(ValueError, match="Unknown predictor_input_mode"):
        TextLatentWorldModel(
            TextLatentWorldModelConfig(
                state_hidden_dim=8,
                action_hidden_dim=8,
                target_hidden_dim=8,
                latent_dim=8,
                predictor_type="adaln",
                predictor_num_heads=2,
                predictor_input_mode="invalid",
            )
        )


def test_invalid_prediction_form_fails_closed():
    with pytest.raises(ValueError, match="Unknown prediction_form"):
        TextLatentWorldModel(
            TextLatentWorldModelConfig(
                state_hidden_dim=8,
                action_hidden_dim=8,
                target_hidden_dim=8,
                latent_dim=8,
                predictor_type="adaln",
                predictor_num_heads=2,
                prediction_form="invalid",
            )
        )


def test_shared_stop_grad_target_is_initialized_from_online_and_updated_by_ema():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            predictor_num_heads=2,
            stop_grad_target=True,
            target_ema_decay=0.5,
        )
    )
    assert all(
        torch.equal(target, online)
        for target, online in zip(
            model.target_adapter.parameters(), model.state_adapter.parameters(), strict=True
        )
    )
    before = [parameter.detach().clone() for parameter in model.target_adapter.parameters()]
    with torch.no_grad():
        next(model.state_adapter.parameters()).add_(1.0)

    assert model.update_target_ema() is True
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.target_adapter.parameters(), strict=True)
    )
    assert all(not parameter.requires_grad for parameter in model.target_adapter.parameters())


def test_next_state_objective_uses_direct_future_latent_and_has_next_mask():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            predictor_num_heads=2,
            sigreg_num_proj=8,
        )
    )
    state = torch.randn(4, 8)
    action = torch.randn(4, 8)
    feedback = torch.randn(4, 8)
    next_state = torch.randn(4, 8)
    has_next = torch.tensor([True, False, True, False])

    first_loss, first_metrics = model.compute_loss(
        state_hidden=state,
        action_hidden=action,
        target_hidden=feedback,
        next_state_hidden=next_state,
        has_next=has_next,
        sigreg_coef=0.0,
        action_contrast_coef=0.0,
        alignment_coef=0.0,
    )
    second_loss, _ = model.compute_loss(
        state_hidden=state,
        action_hidden=action,
        target_hidden=torch.randn_like(feedback),
        next_state_hidden=next_state,
        has_next=has_next,
        sigreg_coef=0.0,
        action_contrast_coef=0.0,
        alignment_coef=0.0,
    )

    assert torch.allclose(first_loss, second_loss)
    assert first_metrics["wm/prediction_mask_count"].item() == 2
    assert torch.allclose(first_metrics["wm/pred_loss"], first_metrics["wm/next_state_pred_mse"])


def test_next_state_objective_fails_closed_without_valid_future_rows():
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=4,
            action_hidden_dim=4,
            target_hidden_dim=4,
            latent_dim=4,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            predictor_num_heads=2,
        )
    )
    kwargs = {
        "state_hidden": torch.randn(2, 4),
        "action_hidden": torch.randn(2, 4),
        "target_hidden": torch.randn(2, 4),
        "next_state_hidden": torch.randn(2, 4),
    }

    with pytest.raises(ValueError, match="explicit has_next"):
        model.compute_loss(**kwargs)
    with pytest.raises(ValueError, match="no has_next=true"):
        model.compute_loss(**kwargs, has_next=torch.zeros(2, dtype=torch.bool))


def test_next_state_feedback_auxiliary_adds_result_alignment_loss():
    torch.manual_seed(17)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            predictor_num_heads=2,
            target_geometry="frozen_random_orthogonal_v1",
        )
    )
    kwargs = {
        "state_hidden": torch.randn(4, 8),
        "action_hidden": torch.randn(4, 8),
        "target_hidden": torch.randn(4, 8),
        "next_state_hidden": torch.randn(4, 8),
        "has_next": torch.tensor([True, True, False, True]),
        "pred_loss_type": "scaled_mse",
        "sigreg_coef": 0.0,
        "action_contrast_coef": 0.0,
        "alignment_coef": 0.0,
    }

    base, _ = model.compute_loss(**kwargs, feedback_aux_coef=0.0)
    combined, metrics = model.compute_loss(**kwargs, feedback_aux_coef=0.3)

    assert metrics["wm/feedback_aux_loss"].item() > 0.0
    assert torch.allclose(
        combined,
        base + 0.3 * metrics["wm/feedback_aux_loss"],
    )


def test_next_state_ema_target_is_frozen_and_tracks_online_state_encoder():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            predictor_num_heads=2,
            stop_grad_target=True,
            target_ema_decay=0.5,
        )
    )
    loss, _ = model.compute_loss(
        state_hidden=torch.randn(4, 8),
        action_hidden=torch.randn(4, 8),
        target_hidden=torch.randn(4, 8),
        next_state_hidden=torch.randn(4, 8),
        has_next=torch.ones(4, dtype=torch.bool),
        sigreg_coef=0.1,
        action_contrast_coef=0.0,
        alignment_coef=0.0,
    )
    loss.backward()

    assert all(parameter.grad is None for parameter in model.target_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.target_shared_projector.parameters())
    assert any(parameter.grad is not None for parameter in model.state_adapter.parameters())


def test_scaled_mse_preserves_unit_latent_geometry_scale():
    torch.manual_seed(0)
    model = TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=8,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            prediction_target="next_state",
            predictor_num_heads=2,
        )
    )
    kwargs = {
        "state_hidden": torch.randn(4, 8),
        "action_hidden": torch.randn(4, 8),
        "target_hidden": torch.randn(4, 8),
        "next_state_hidden": torch.randn(4, 8),
        "has_next": torch.ones(4, dtype=torch.bool),
        "sigreg_coef": 0.0,
        "action_contrast_coef": 0.0,
        "alignment_coef": 0.0,
    }
    mse, _ = model.compute_loss(**kwargs, pred_loss_type="mse")
    scaled, _ = model.compute_loss(**kwargs, pred_loss_type="scaled_mse")

    assert scaled.item() == pytest.approx(mse.item() * 8)


def _fixed_geometry_model(seed: int = 7) -> TextLatentWorldModel:
    return TextLatentWorldModel(
        TextLatentWorldModelConfig(
            state_hidden_dim=8,
            action_hidden_dim=8,
            target_hidden_dim=8,
            latent_dim=4,
            predictor_type="adaln",
            architecture_version="shared_latent_v2",
            predictor_num_heads=2,
            sigreg_num_proj=8,
            value_head=False,
            uncertainty_head=False,
            target_geometry="frozen_random_orthogonal_v1",
            fixed_target_seed=seed,
        )
    )


def test_frozen_target_geometry_is_deterministic_and_data_independent():
    first = _fixed_geometry_model(7)
    second = _fixed_geometry_model(7)
    different = _fixed_geometry_model(11)

    assert torch.equal(
        first.fixed_target_projection,
        second.fixed_target_projection,
    )
    assert not torch.equal(
        first.fixed_target_projection,
        different.fixed_target_projection,
    )
    gram = first.fixed_target_projection.T @ first.fixed_target_projection
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)


def test_frozen_target_forward_uses_common_geometry_without_target_params():
    model = _fixed_geometry_model()
    hidden = torch.randn(6, 8)
    action = torch.randn(6, 8)
    output = model(
        state_hidden=hidden,
        action_hidden=action,
        target_hidden=hidden,
    )

    assert model.target_adapter is None
    assert model.target_shared_projector is None
    assert torch.equal(
        output["target_latent"],
        model.map_fixed_target_hidden(hidden),
    )
    assert not output["target_latent"].requires_grad

    torch.nn.functional.mse_loss(
        output["pred_latent"],
        output["target_latent"],
    ).backward()
    assert any(
        parameter.grad is not None for parameter in model.state_adapter.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in model.action_projector.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())


def test_frozen_target_geometry_state_dict_roundtrip_and_no_ema():
    first = _fixed_geometry_model()
    second = _fixed_geometry_model(seed=99)
    second.load_state_dict(first.state_dict())
    hidden = torch.randn(3, 8)

    assert torch.equal(
        first.map_fixed_target_hidden(hidden),
        second.map_fixed_target_hidden(hidden),
    )
    assert first.update_target_ema() is False


def test_frozen_target_geometry_rejects_incompatible_architecture_and_width():
    with pytest.raises(ValueError, match="requires shared_latent_v2"):
        TextLatentWorldModel(
            TextLatentWorldModelConfig(
                state_hidden_dim=8,
                action_hidden_dim=8,
                target_hidden_dim=8,
                latent_dim=4,
                architecture_version="legacy_mlp_v1",
                target_geometry="frozen_random_orthogonal_v1",
            )
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        TextLatentWorldModel(
            TextLatentWorldModelConfig(
                state_hidden_dim=4,
                action_hidden_dim=4,
                target_hidden_dim=4,
                latent_dim=8,
                predictor_type="adaln",
                architecture_version="shared_latent_v2",
                target_geometry="frozen_random_orthogonal_v1",
            )
        )
