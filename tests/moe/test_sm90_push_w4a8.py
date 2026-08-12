import pytest
import torch

from flashinfer.fused_moe.nvfp4_checkpoint import NVFP4Checkpoint
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MoEEpConfigError,
    Sm90PushNvFp4MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.backend import (
    Sm90PushNvFp4MegaKernelBackend,
)
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
    Sm90PushNvFp4Weights,
    make_sm90_push_nvfp4_weights_from_checkpoints,
)


def _checkpoint(experts: int, n: int, k: int) -> NVFP4Checkpoint:
    return NVFP4Checkpoint(
        torch.zeros(experts, n, k // 2, dtype=torch.uint8),
        torch.ones(experts, n, k // 16).to(torch.float8_e4m3fn),
        torch.ones(experts, dtype=torch.float32),
        (experts, n, k),
        tuple(range(experts)),
        "flashinfer.sm90_push.nvfp4.test",
    )


def test_e2m1_rne_midpoints_use_even_codes():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.weights import (
        _e2m1_codes,
    )

    values = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    expected = torch.tensor([0, 0, 2, 2, 4, 4, 6, 6, 7], dtype=torch.uint8)
    assert torch.equal(_e2m1_codes(values), expected)
    assert torch.equal(_e2m1_codes(-values[1:]), expected[1:] | 0x08)


def test_w4a8_checkpoint_conversion_is_typed_and_versioned():
    weights = make_sm90_push_nvfp4_weights_from_checkpoints(
        _checkpoint(2, 256, 128),
        _checkpoint(2, 128, 128),
        nvfp4_mode="w4a8",
        group_size=64,
        residual_scheme="pow2",
    )
    assert isinstance(weights, Sm90PushNvFp4Weights)
    assert weights.nvfp4_mode == "w4a8"
    for view in (weights.w13, weights.w2):
        assert view.manifest.layout_version == 3
        assert view.manifest.group_size == 64
        assert view.manifest.residual_scheme == "pow2"
        assert view.manifest.padded_shape[2] % 128 == 0
        view.verify_checksums()


def test_rs_checkpoint_conversion_keeps_exact_w4a16_streams():
    weights = make_sm90_push_nvfp4_weights_from_checkpoints(
        _checkpoint(2, 256, 128),
        _checkpoint(2, 128, 128),
        nvfp4_mode="w4a16_rs",
    )
    assert weights.nvfp4_mode == "w4a16_rs"
    assert weights.w13.payload.shape[:3] == (2, 4, 8)
    assert weights.w2.payload.shape[:3] == (2, 2, 8)


def test_rs_checkpoint_conversion_rejects_nonidentity_expert_mapping():
    w13 = _checkpoint(2, 256, 128)
    w2 = _checkpoint(2, 128, 128)
    w13 = NVFP4Checkpoint(
        w13.packed_e2m1,
        w13.scale_e4m3_per16,
        w13.global_alpha,
        w13.logical_shape,
        (4, 7),
        w13.source_format_version,
    )
    w2 = NVFP4Checkpoint(
        w2.packed_e2m1,
        w2.scale_e4m3_per16,
        w2.global_alpha,
        w2.logical_shape,
        (4, 7),
        w2.source_format_version,
    )
    with pytest.raises(ValueError, match="identity-ordered"):
        make_sm90_push_nvfp4_weights_from_checkpoints(w13, w2, nvfp4_mode="w4a16_rs")


def test_weight_mode_mismatch_is_rejected():
    w13 = _checkpoint(1, 256, 128)
    w2 = _checkpoint(1, 128, 128)
    rs = make_sm90_push_nvfp4_weights_from_checkpoints(w13, w2, nvfp4_mode="w4a16_rs")
    with pytest.raises(TypeError, match="NVFP4SM90WeightViewV3"):
        Sm90PushNvFp4Weights("w4a8", rs.w13, rs.w2)


def test_modelopt_loader_requires_cuda_target_for_cpu_tensors():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.weights import (
        load_modelopt_transformed_weights,
    )

    state_dict = {}
    for prefix, rows in (("w13", 256), ("w2", 128)):
        state_dict[f"{prefix}.weight"] = torch.zeros(1, rows, 64, dtype=torch.uint8)
        state_dict[f"{prefix}.weight_scale"] = torch.ones(
            1, rows, 8, dtype=torch.float32
        ).to(torch.float8_e4m3fn)
        state_dict[f"{prefix}.weight_scale_2"] = torch.ones(1, dtype=torch.float32)

    with pytest.raises(ValueError, match="pass device='cuda:<index>'"):
        load_modelopt_transformed_weights(
            state_dict,
            w13_prefix="w13",
            w2_prefix="w2",
        )


def _validate(config: Sm90PushNvFp4MegaMoeConfig, monkeypatch) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import backend

    monkeypatch.setattr(backend, "_validate_sm90_arch", lambda: None)
    instance = Sm90PushNvFp4MegaKernelBackend(config)
    instance.validate_init(
        BootstrapConfig(world_size=1, rank=0),
        FleetParams(num_experts=2, max_tokens_per_rank=8, token_hidden_size=128),
    )


def test_w4a8_accepts_bf16_wire_with_local_a8_quantization(monkeypatch):
    _validate(
        Sm90PushNvFp4MegaMoeConfig(
            intermediate_size=128,
            top_k=1,
            nvfp4_mode="w4a8",
            payload_dtype="bf16",
            combine_dtype="bf16",
            grouped_combine=False,
        ),
        monkeypatch,
    )


def test_rs_rejects_fp8_combine_wire(monkeypatch):
    with pytest.raises(
        MoEEpConfigError, match="w4a16_rs requires combine_dtype='bf16'"
    ):
        _validate(
            Sm90PushNvFp4MegaMoeConfig(
                intermediate_size=128,
                top_k=1,
                nvfp4_mode="w4a16_rs",
                fuse_act=False,
                combine_dtype="fp8",
                grouped_combine=False,
            ),
            monkeypatch,
        )


def test_rs_rejects_fused_activation(monkeypatch):
    with pytest.raises(MoEEpConfigError, match="w4a16_rs requires fuse_act=False"):
        _validate(
            Sm90PushNvFp4MegaMoeConfig(
                intermediate_size=128,
                top_k=1,
                nvfp4_mode="w4a16_rs",
                fuse_act=True,
                combine_dtype="bf16",
                grouped_combine=False,
            ),
            monkeypatch,
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("group_size", 16, "group_size must be 32, 64, or 128"),
        ("residual_scheme", "invalid", "residual_scheme must be 'generic' or 'pow2'"),
    ],
)
def test_rs_rejects_invalid_w4a8_layout_knobs(monkeypatch, field, value, match):
    config = Sm90PushNvFp4MegaMoeConfig(
        intermediate_size=128,
        top_k=1,
        nvfp4_mode="w4a16_rs",
        fuse_act=False,
        combine_dtype="bf16",
        grouped_combine=False,
    )
    setattr(config, field, value)
    with pytest.raises(MoEEpConfigError, match=match):
        _validate(config, monkeypatch)


@pytest.mark.parametrize(
    "field,value",
    [("group_size", 64), ("residual_scheme", "pow2")],
)
def test_rs_rejects_unused_w4a8_layout_knobs(monkeypatch, field, value):
    config = Sm90PushNvFp4MegaMoeConfig(
        intermediate_size=128,
        top_k=1,
        nvfp4_mode="w4a16_rs",
        fuse_act=False,
        combine_dtype="bf16",
        grouped_combine=False,
    )
    setattr(config, field, value)
    with pytest.raises(
        MoEEpConfigError,
        match="w4a16_rs requires group_size=128 and residual_scheme='generic'",
    ):
        _validate(config, monkeypatch)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("rs_n_tactic", 128, "rs_n_tactic must be 64"),
        ("rs_stages", 2, "rs_stages must be 3"),
        ("rs_stage_k", 128, "rs_stage_k must be 64"),
    ],
)
def test_rs_rejects_unsupported_tactic(monkeypatch, field, value, match):
    config = Sm90PushNvFp4MegaMoeConfig(
        intermediate_size=128,
        top_k=1,
        nvfp4_mode="w4a16_rs",
        fuse_act=False,
        combine_dtype="bf16",
        grouped_combine=False,
    )
    setattr(config, field, value)
    with pytest.raises(MoEEpConfigError, match=match):
        _validate(config, monkeypatch)


def test_rs_accepts_fp8_payload_and_bf16_combine(monkeypatch):
    _validate(
        Sm90PushNvFp4MegaMoeConfig(
            intermediate_size=128,
            top_k=1,
            nvfp4_mode="w4a16_rs",
            fuse_act=False,
            payload_dtype="fp8",
            combine_dtype="bf16",
            grouped_combine=False,
        ),
        monkeypatch,
    )
