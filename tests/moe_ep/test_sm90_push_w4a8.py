import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_checkpoint import (
    NVFP4Checkpoint,
)
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda.backend import (
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
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda.weights import (
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
        group_size=64,
        residual_scheme="pow2",
    )
    assert isinstance(weights, Sm90PushNvFp4Weights)
    for view in (weights.w13, weights.w2):
        assert view.manifest.layout_version == 4
        assert view.manifest.group_size == 64
        assert view.manifest.residual_scheme == "pow2"
        assert view.manifest.padded_shape[2] % 128 == 0
        view.verify_checksums()


def test_modelopt_loader_requires_cuda_target_for_cpu_tensors():
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda.weights import (
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


def test_modelopt_loader_builds_stage_contiguous_v4_views(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_repack import (
        NVFP4SM90WeightViewV4,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda.weights import (
        load_modelopt_transformed_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_weights as nvfp4_weight_impl,
    )

    state_dict = {}
    for prefix, rows in (("w13", 256), ("w2", 128)):
        checkpoint = _checkpoint(2, rows, 128)
        state_dict[f"{prefix}.weight"] = checkpoint.packed_e2m1
        state_dict[f"{prefix}.weight_scale"] = checkpoint.scale_e4m3_per16
        state_dict[f"{prefix}.weight_scale_2"] = checkpoint.global_alpha

    monkeypatch.setattr(
        nvfp4_weight_impl,
        "_move_modelopt_checkpoint",
        lambda checkpoint, _device: checkpoint,
    )
    weights = load_modelopt_transformed_weights(
        state_dict,
        w13_prefix="w13",
        w2_prefix="w2",
        group_size=64,
        residual_scheme="pow2",
        payload_layout=4,
        device="cuda:0",
    )

    for view in (weights.w13, weights.w2):
        assert isinstance(view, NVFP4SM90WeightViewV4)
        assert view.manifest.layout_version == 4
        assert view.manifest.group_size == 64
        assert view.manifest.residual_scheme == "pow2"
        view.verify_checksums()


def _validate(config: Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig, monkeypatch) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        backend,
    )

    monkeypatch.setattr(backend, "_validate_sm90_arch", lambda: None)
    instance = Sm90PushNvFp4MegaKernelBackend(config)
    instance.validate_init(
        BootstrapConfig(world_size=1, rank=0),
        FleetParams(num_experts=2, max_tokens_per_rank=8, token_hidden_size=128),
    )


def test_w4a8_accepts_bf16_wire_with_local_a8_quantization(monkeypatch):
    _validate(
        Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig(
            intermediate_size=128,
            top_k=1,
            payload_dtype="bf16",
            combine_dtype="bf16",
            grouped_combine=False,
        ),
        monkeypatch,
    )
