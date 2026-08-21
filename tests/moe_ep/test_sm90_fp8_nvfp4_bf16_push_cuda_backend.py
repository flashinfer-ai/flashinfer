"""Public MegaKernelBackend correctness gates for SM90 push NVFP4."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest import mock

import pytest
import torch

from flashinfer.fused_moe.nvfp4_checkpoint import reference_dequantize_nvfp4

from ._sm90_push_fp8_reference import reference_moe


def _sm90_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least
        from flashinfer.utils import is_sm90a_supported

        return is_cuda_version_at_least("12.0") and is_sm90a_supported(
            torch.device("cuda")
        )
    except Exception:
        return False


def _sm90_fp8_cuda_available() -> bool:
    if not _sm90_cuda_available():
        return False
    try:
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least

        return is_cuda_version_at_least("12.8")
    except Exception:
        return False


_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
requires_sm90 = pytest.mark.skipif(
    not _sm90_cuda_available() or _WORLD > 1,
    reason="requires one SM90 GPU and CUDA Toolkit 12.0+ outside torchrun",
)
requires_dist = pytest.mark.skipif(
    _WORLD < 2 or not _sm90_cuda_available(),
    reason="requires torchrun with at least two SM90 GPUs and CUDA Toolkit 12.0+",
)
requires_sm90_fp8 = pytest.mark.skipif(
    not _sm90_fp8_cuda_available() or _WORLD > 1,
    reason="requires one SM90 GPU and CUDA Toolkit 12.8+ outside torchrun",
)

HIDDEN = 256
INTERMEDIATE = 256
LOCAL_EXPERTS = 2
TOP_K = 2
TOKEN_CAPACITY = 32

_KEEP_ALIVE: list[object] = []


@pytest.fixture(autouse=True)
def _destroy_retained_layers():
    try:
        yield
    finally:
        retained_layers = tuple(reversed(_KEEP_ALIVE))
        _KEEP_ALIVE.clear()
        for layer in retained_layers:
            destroy = getattr(layer, "destroy", None)
            if destroy is not None:
                destroy()


def test_compute_rejects_weight_mismatch_after_completing_the_round() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda.backend import (
        Sm90PushNvFp4MegaKernelBackend,
        _Sm90PushNvFp4Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda.config import (
        Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushNvFp4MegaKernelBackend(
        Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig(intermediate_size=128, top_k=1)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock(state="idle")
    workspace = _Sm90PushNvFp4Workspace(
        pipe=object(),
        runner=runner,
        active_weights=transformed,
        staged_weights=transformed,
        staged_tokens=1,
    )

    output = object()
    runner.compute.return_value = output
    with pytest.raises(RuntimeError, match="different weight bundle"):
        backend.compute(workspace, object(), output=output)

    runner.compute.assert_called_once_with(output=output)
    assert workspace.staged_weights is None
    assert workspace.staged_tokens is None


def _make_weights(
    num_experts: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (
        torch.randn(
            num_experts,
            2 * INTERMEDIATE,
            HIDDEN,
            generator=generator,
        )
        * HIDDEN**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    w2 = (
        torch.randn(
            num_experts,
            HIDDEN,
            INTERMEDIATE,
            generator=generator,
        )
        * INTERMEDIATE**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    return w13, w2


def _make_inputs(
    num_tokens: int,
    num_experts: int,
    seed: int,
    device: torch.device,
    *,
    mode: str = "random",
    rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    if mode == "hot":
        ids = torch.zeros(num_tokens, TOP_K, dtype=torch.int32)
    else:
        logits = torch.randn(num_tokens, num_experts, generator=generator)
        if mode == "all_remote" and num_experts > LOCAL_EXPERTS:
            begin = rank * LOCAL_EXPERTS
            logits[:, begin : begin + LOCAL_EXPERTS] = float("-inf")
        ids = logits.topk(TOP_K, dim=1).indices.to(torch.int32)
    weights = torch.rand(num_tokens, TOP_K, generator=generator) + 0.1
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (
        x,
        ids.to(device),
        weights.to(device=device, dtype=torch.float32),
    )


def _build_layer(
    world_size: int,
    rank: int,
    device: torch.device,
    *,
    nvfp4_mode: str,
    payload_dtype: str,
    combine_dtype: str,
    grouped_combine: bool,
    capacity_factor: float = 1.0,
    transformed: bool = False,
    dedup_dispatch: bool = True,
    fuse_act: bool | None = None,
    weight_seed: int = 17,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig,
    )

    total_experts = LOCAL_EXPERTS * world_size
    w13, w2 = _make_weights(total_experts, weight_seed, device)
    begin = rank * LOCAL_EXPERTS
    end = begin + LOCAL_EXPERTS
    local_w13 = w13[begin:end].contiguous()
    local_w2 = w2[begin:end].contiguous()
    transformed_weights = None
    if transformed:
        from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
            load_modelopt_transformed_weights,
            quantize_bf16_to_nvfp4_checkpoint,
        )

        w13_checkpoint = quantize_bf16_to_nvfp4_checkpoint(local_w13)
        w2_checkpoint = quantize_bf16_to_nvfp4_checkpoint(local_w2)
        state_dict = {
            "w13.weight": w13_checkpoint.packed_e2m1,
            "w13.weight_scale": w13_checkpoint.scale_e4m3_per16,
            "w13.weight_scale_2": w13_checkpoint.global_alpha,
            "w2.weight": w2_checkpoint.packed_e2m1,
            "w2.weight_scale": w2_checkpoint.scale_e4m3_per16,
            "w2.weight_scale_2": w2_checkpoint.global_alpha,
        }
        transformed_weights = load_modelopt_transformed_weights(
            state_dict,
            w13_prefix="w13",
            w2_prefix="w2",
            nvfp4_mode=nvfp4_mode,
            device=device,
        )
    process_group = None
    if world_size > 1:
        import torch.distributed as dist

        process_group = dist.group.WORLD
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            process_group=process_group,
        ),
        fleet_params=FleetParams(
            num_experts=total_experts,
            max_tokens_per_rank=TOKEN_CAPACITY,
            token_hidden_size=HIDDEN,
        ),
        weights=MoEWeightPack(
            w13=local_w13,
            w2=local_w2,
        ),
        backend=MegaConfig(
            megakernel=Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOP_K,
                nvfp4_mode=nvfp4_mode,
                payload_dtype=payload_dtype,
                combine_dtype=combine_dtype,
                grouped_combine=grouped_combine,
                dedup_dispatch=dedup_dispatch,
                fuse_act=(nvfp4_mode == "w4a8" if fuse_act is None else fuse_act),
                capacity_factor=capacity_factor,
            ),
            quantize_input=True,
            preprocess_weights=not transformed,
            transformed_weights=transformed_weights,
        ),
    )
    _KEEP_ALIVE.append(layer)
    return layer, w13, w2


def _forward(layer, x, topk_ids, topk_weights):
    from flashinfer.moe_ep import MoEEpTensors

    return layer(
        MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
    )


def _build_prepared_layer(megakernel, transformed_weights):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=FleetParams(
            num_experts=LOCAL_EXPERTS,
            max_tokens_per_rank=TOKEN_CAPACITY,
            token_hidden_size=HIDDEN,
        ),
        weights=None,
        backend=MegaConfig(
            megakernel=megakernel,
            quantize_input=True,
            preprocess_weights=False,
            transformed_weights=transformed_weights,
        ),
    )


def _reference(x, topk_ids, topk_weights, w13, w2):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        quantize_bf16_to_nvfp4_checkpoint,
    )

    w13_dequant = reference_dequantize_nvfp4(quantize_bf16_to_nvfp4_checkpoint(w13))
    w2_dequant = reference_dequantize_nvfp4(quantize_bf16_to_nvfp4_checkpoint(w2))
    return reference_moe(
        x,
        w13_dequant,
        w2_dequant,
        topk_ids,
        topk_weights,
    )


def _normalized_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-6)
    numerator = (output.float() - reference.float()).square().mean().sqrt()
    return float(numerator / denominator)


def _cosine(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
    )


def _assert_close(mode: str, output: torch.Tensor, reference: torch.Tensor) -> None:
    assert torch.isfinite(output.float()).all()
    normalized_l2 = _normalized_l2(output, reference)
    cosine = _cosine(output, reference)
    if mode == "w4a8":
        # This end-to-end gate includes routing, two quantized GEMMs, activation
        # quantization, and combine. Direct W4A8 tests enforce tighter byte and
        # numerical oracles for the individual GEMM path.
        assert normalized_l2 <= 0.35, f"W4A8 normalized L2={normalized_l2:.6f}"
        assert cosine >= 0.95, f"W4A8 cosine={cosine:.6f}"
    else:
        assert normalized_l2 <= 0.12
        assert cosine >= 0.99


@requires_sm90
@pytest.mark.parametrize(
    "nvfp4_mode,payload_dtype,combine_dtype,grouped_combine,dedup_dispatch,fuse_act",
    [
        ("w4a8", "fp8", "fp8", True, True, True),
        ("w4a8", "fp8", "bf16", False, False, True),
        ("w4a8", "bf16", "fp8", True, True, True),
        ("w4a8", "bf16", "bf16", False, True, True),
        ("w4a8", "fp8", "fp8", True, True, False),
        ("w4a16_rs", "fp8", "bf16", False, True, False),
    ],
)
def test_public_ep1_forward_configs(
    nvfp4_mode: str,
    payload_dtype: str,
    combine_dtype: str,
    grouped_combine: bool,
    dedup_dispatch: bool,
    fuse_act: bool,
) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1,
        0,
        device,
        nvfp4_mode=nvfp4_mode,
        payload_dtype=payload_dtype,
        combine_dtype=combine_dtype,
        grouped_combine=grouped_combine,
        dedup_dispatch=dedup_dispatch,
        fuse_act=fuse_act,
    )
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 23, device)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    reference = _reference(x, ids, weights, w13, w2)
    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    _assert_close(nvfp4_mode, output, reference)


@requires_sm90
def test_public_ep1_capacity_factor_quarter_happy_path() -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1,
        0,
        device,
        nvfp4_mode="w4a8",
        payload_dtype="fp8",
        combine_dtype="fp8",
        grouped_combine=True,
        capacity_factor=0.25,
    )
    x, ids, weights = _make_inputs(8, LOCAL_EXPERTS, 29, device)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _assert_close("w4a8", output, _reference(x, ids, weights, w13, w2))


@requires_sm90
def test_modelopt_transformed_weights_run_without_preprocessing() -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1,
        0,
        device,
        nvfp4_mode="w4a8",
        payload_dtype="bf16",
        combine_dtype="bf16",
        grouped_combine=False,
        transformed=True,
    )
    x, ids, weights = _make_inputs(8, LOCAL_EXPERTS, 31, device)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _assert_close("w4a8", output, _reference(x, ids, weights, w13, w2))


@requires_sm90_fp8
@pytest.mark.parametrize("fuse_fc1_epilogue", [False, True])
def test_modelopt_folded_fp8_weights_run_on_fp8_backend(
    fuse_fc1_epilogue: bool,
) -> None:
    from flashinfer.moe_ep import (
        Sm90PushFp8MegaMoeConfig,
        load_sm90_push_nvfp4_modelopt_folded_fp8_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        quantize_bf16_to_nvfp4_checkpoint,
    )

    device = torch.device("cuda", 0)
    w13, w2 = _make_weights(LOCAL_EXPERTS, 37, device)
    w13_checkpoint = quantize_bf16_to_nvfp4_checkpoint(w13)
    w2_checkpoint = quantize_bf16_to_nvfp4_checkpoint(w2)
    transformed = load_sm90_push_nvfp4_modelopt_folded_fp8_weights(
        {
            "w13.weight": w13_checkpoint.packed_e2m1.cpu(),
            "w13.weight_scale": w13_checkpoint.scale_e4m3_per16.cpu(),
            "w13.weight_scale_2": w13_checkpoint.global_alpha.cpu(),
            "w2.weight": w2_checkpoint.packed_e2m1.cpu(),
            "w2.weight_scale": w2_checkpoint.scale_e4m3_per16.cpu(),
            "w2.weight_scale_2": w2_checkpoint.global_alpha.cpu(),
        },
        w13_prefix="w13",
        w2_prefix="w2",
        interleave_gate_up=fuse_fc1_epilogue,
        device=device,
    )
    layer = _build_prepared_layer(
        Sm90PushFp8MegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            payload_dtype="fp8",
            combine_dtype="fp8",
            grouped_combine=True,
            fuse_fc1_epilogue=fuse_fc1_epilogue,
        ),
        transformed,
    )
    try:
        x, ids, weights = _make_inputs(8, LOCAL_EXPERTS, 41, device)
        output = _forward(layer, x, ids, weights)
        torch.cuda.synchronize()
        reference = reference_moe(
            x,
            reference_dequantize_nvfp4(w13_checkpoint),
            reference_dequantize_nvfp4(w2_checkpoint),
            ids,
            weights,
        )
        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output.float()).all()
        assert _normalized_l2(output, reference) < 0.10
        assert _cosine(output, reference) > 0.997
    finally:
        layer.destroy()


@requires_sm90_fp8
def test_folded_fp8_error_matches_online_w4a8() -> None:
    from flashinfer.moe_ep import (
        Sm90PushFp8MegaMoeConfig,
        Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig,
        make_sm90_push_nvfp4_folded_fp8_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        make_transformed_weights_from_checkpoints,
        quantize_bf16_to_nvfp4_checkpoint,
    )

    device = torch.device("cuda", 0)
    w13, w2 = _make_weights(LOCAL_EXPERTS, 47, device)
    w13_checkpoint = quantize_bf16_to_nvfp4_checkpoint(w13)
    w2_checkpoint = quantize_bf16_to_nvfp4_checkpoint(w2)
    online_weights = make_transformed_weights_from_checkpoints(
        w13_checkpoint,
        w2_checkpoint,
        nvfp4_mode="w4a8",
        group_size=128,
        residual_scheme="generic",
    )
    folded_weights = make_sm90_push_nvfp4_folded_fp8_weights(
        w13_checkpoint,
        w2_checkpoint,
    )
    x, ids, topk_weights = _make_inputs(16, LOCAL_EXPERTS, 53, device)
    online_layer = _build_prepared_layer(
        Sm90_Fp8_Nvfp4_Bf16_PushCuda_MegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            nvfp4_mode="w4a8",
            group_size=128,
            residual_scheme="generic",
            payload_dtype="bf16",
            combine_dtype="bf16",
            grouped_combine=False,
            fuse_act=False,
        ),
        online_weights,
    )
    try:
        online_output = _forward(online_layer, x, ids, topk_weights).clone()
        torch.cuda.synchronize()
    finally:
        online_layer.destroy()

    folded_layer = _build_prepared_layer(
        Sm90PushFp8MegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            payload_dtype="bf16",
            combine_dtype="bf16",
            grouped_combine=False,
            fuse_fc1_epilogue=False,
        ),
        folded_weights,
    )
    try:
        folded_output = _forward(folded_layer, x, ids, topk_weights).clone()
        torch.cuda.synchronize()
    finally:
        folded_layer.destroy()

    reference = reference_moe(
        x,
        reference_dequantize_nvfp4(w13_checkpoint),
        reference_dequantize_nvfp4(w2_checkpoint),
        ids,
        topk_weights,
    )
    online_l2 = _normalized_l2(online_output, reference)
    folded_l2 = _normalized_l2(folded_output, reference)
    online_cosine = _cosine(online_output, reference)
    folded_cosine = _cosine(folded_output, reference)
    online_cosine_error = max(0.0, 1.0 - online_cosine)
    folded_cosine_error = max(0.0, 1.0 - folded_cosine)

    _assert_close("w4a8", online_output, reference)
    assert torch.isfinite(folded_output.float()).all()
    assert folded_l2 < 0.10
    assert folded_cosine > 0.997
    assert folded_l2 <= online_l2 * 1.25 + 0.005, (
        f"folded L2 {folded_l2:.6f} exceeds online W4A8 {online_l2:.6f}; "
        f"ratio={folded_l2 / max(online_l2, 1e-12):.3f}"
    )
    assert folded_cosine_error <= online_cosine_error * 1.25 + 0.0005, (
        f"folded cosine {folded_cosine:.6f} is below online W4A8 {online_cosine:.6f}"
    )


@requires_sm90_fp8
@pytest.mark.parametrize("interleave_gate_up", [False, True])
def test_folded_fp8_layout_mismatch_is_rejected(
    interleave_gate_up: bool,
) -> None:
    from flashinfer.moe_ep import (
        MoEEpConfigError,
        make_sm90_push_nvfp4_folded_fp8_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.weights import (
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        quantize_bf16_to_nvfp4_checkpoint,
    )

    device = torch.device("cuda", 0)
    w13, w2 = _make_weights(LOCAL_EXPERTS, 43, device)
    transformed = make_sm90_push_nvfp4_folded_fp8_weights(
        quantize_bf16_to_nvfp4_checkpoint(w13),
        quantize_bf16_to_nvfp4_checkpoint(w2),
        interleave_gate_up=interleave_gate_up,
    )

    with pytest.raises(MoEEpConfigError, match="weight layout does not match"):
        validate_transformed_mega_weights(
            transformed,
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            num_local_experts=LOCAL_EXPERTS,
            fuse_fc1_epilogue=not interleave_gate_up,
        )


@requires_sm90
@pytest.mark.parametrize(
    "case,marker",
    [
        ("compact_record", "sm90_push: compact record"),
        ("combine_row", "sm90_push: combine row"),
    ],
)
def test_padded_a2a_contract_traps_in_subprocess(case: str, marker: str) -> None:
    code = f"""\
import torch
from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
    Sm90PushCombine, Sm90PushConfig, Sm90PushPayload, Sm90PushPipe,
)
H, E, K, T = 256, 2, 1, 4
pipe = Sm90PushPipe(
    ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
    token_capacity=T, device_index=0,
    config=Sm90PushConfig(
        payload_dtype=Sm90PushPayload.BF16,
        combine_dtype=Sm90PushCombine.BF16,
    ),
)
pipe.proto_begin_round()
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.zeros(T, K, dtype=torch.int32, device="cuda")
weights = torch.ones(T, K, dtype=torch.float32, device="cuda")
pipe.proto_dispatch(x, ids, weights)
pipe.proto_wait_prefix()
a = torch.empty(128, H, dtype=torch.bfloat16, device="cuda")
meta = torch.empty(pipe.meta_rows, 4, dtype=torch.int32, device="cuda")
row_map = torch.empty(pipe.meta_rows, dtype=torch.int32, device="cuda")
offsets = torch.empty(E + 1, dtype=torch.int64, device="cuda")
tile_prefix = torch.empty(E + 1, dtype=torch.int64, device="cuda")
padded_m = torch.empty(1, dtype=torch.int32, device="cuda")
if {case!r} == "compact_record":
    pipe._seg_src_base[0] = pipe.meta_rows
pipe.proto_compact_bf16_padded(a, meta, row_map, offsets, tile_prefix, padded_m, 64)
if {case!r} == "combine_row":
    row_map[0] = a.shape[0]
    pipe.proto_combine_mapped(a, meta, row_map)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined[-1500:]
    assert "UNEXPECTED-SURVIVAL" not in result.stdout, combined[-1500:]
    assert marker in combined, combined[-1500:]


@requires_sm90
def test_bf16_checkpoint_quantization_is_chunk_invariant(monkeypatch) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_nvfp4_bf16_push_cuda import (
        weights as nvfp4_weights,
    )

    device = torch.device("cuda", 0)
    source = (
        _make_weights(2, 71, device)[0][:, :67, :]
        .contiguous()
        .detach()
        .requires_grad_()
    )
    monkeypatch.setattr(nvfp4_weights, "_NVFP4_QUANT_CHUNK_VALUES", 1 << 30)
    full = nvfp4_weights.quantize_bf16_to_nvfp4_checkpoint(source)
    monkeypatch.setattr(nvfp4_weights, "_NVFP4_QUANT_CHUNK_VALUES", 3 * HIDDEN)
    chunked = nvfp4_weights.quantize_bf16_to_nvfp4_checkpoint(source)
    assert torch.equal(chunked.packed_e2m1, full.packed_e2m1)
    assert torch.equal(
        chunked.scale_e4m3_per16.view(torch.uint8),
        full.scale_e4m3_per16.view(torch.uint8),
    )
    assert torch.equal(chunked.global_alpha, full.global_alpha)
    assert not full.packed_e2m1.requires_grad
    assert not full.scale_e4m3_per16.requires_grad
    assert not full.global_alpha.requires_grad


@requires_sm90
@pytest.mark.parametrize(
    "nvfp4_mode,fuse_act",
    [("w4a8", True), ("w4a8", False), ("w4a16_rs", False)],
)
def test_public_ep1_graph_replay(nvfp4_mode: str, fuse_act: bool) -> None:
    device = torch.device("cuda", 0)
    rs_mode = nvfp4_mode == "w4a16_rs"
    layer, _, _ = _build_layer(
        1,
        0,
        device,
        nvfp4_mode=nvfp4_mode,
        payload_dtype="fp8",
        combine_dtype="bf16" if rs_mode else "fp8",
        grouped_combine=not rs_mode,
        fuse_act=fuse_act,
    )
    inputs = [
        _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 81 + index, device)
        for index in range(2)
    ]
    eager = []
    for x, ids, weights in inputs:
        eager.append(_forward(layer, x, ids, weights).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(inputs[0][0]).copy_(inputs[0][0])
    static_ids = torch.empty_like(inputs[0][1]).copy_(inputs[0][1])
    static_weights = torch.empty_like(inputs[0][2]).copy_(inputs[0][2])
    side_stream = torch.cuda.Stream()
    for _ in range(2):
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, static_x, static_ids, static_weights)
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, static_x, static_ids, static_weights)
    replayed = []
    for (x, ids, weights), expected in zip(inputs, eager, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_weights.copy_(weights)
        graph.replay()
        torch.cuda.synchronize()
        replayed.append(static_output.clone())
        assert torch.equal(static_output, expected)
    assert not torch.equal(replayed[0], replayed[1])


@requires_sm90
@pytest.mark.parametrize("nvfp4_mode", ["w4a8", "w4a16_rs"])
def test_public_ep1_two_layers_share_workspace_and_graph_replay(
    nvfp4_mode: str,
) -> None:
    device = torch.device("cuda", 0)
    rs_mode = nvfp4_mode == "w4a16_rs"
    config = {
        "nvfp4_mode": nvfp4_mode,
        "payload_dtype": "fp8",
        "combine_dtype": "bf16" if rs_mode else "fp8",
        "grouped_combine": not rs_mode,
        "fuse_act": not rs_mode,
    }
    first, _, _ = _build_layer(1, 0, device, weight_seed=101, **config)
    second, _, _ = _build_layer(1, 0, device, weight_seed=102, **config)
    inputs = [
        _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 103 + index, device)
        for index in range(2)
    ]
    eager = []
    for x, ids, weights in inputs:
        first_output = _forward(first, x, ids, weights).clone()
        second_output = _forward(second, x, ids, weights).clone()
        torch.cuda.synchronize()
        assert not torch.equal(first_output, second_output)
        eager.append((first_output, second_output))

    assert first._workspace is second._workspace
    static_x = torch.empty_like(inputs[0][0]).copy_(inputs[0][0])
    static_ids = torch.empty_like(inputs[0][1]).copy_(inputs[0][1])
    static_weights = torch.empty_like(inputs[0][2]).copy_(inputs[0][2])
    side_stream = torch.cuda.Stream()
    for _ in range(2):
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(first, static_x, static_ids, static_weights)
            _forward(second, static_x, static_ids, static_weights)
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first_static_output = _forward(first, static_x, static_ids, static_weights)
        second_static_output = _forward(second, static_x, static_ids, static_weights)
    for index, (x, ids, weights) in enumerate(inputs):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_weights.copy_(weights)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(first_static_output, eager[index][0])
        assert torch.equal(second_static_output, eager[index][1])


@requires_sm90
@pytest.mark.parametrize("nvfp4_mode", ["w4a8", "w4a16_rs"])
def test_public_ep1_destroy_is_idempotent(nvfp4_mode: str) -> None:
    device = torch.device("cuda", 0)
    rs_mode = nvfp4_mode == "w4a16_rs"
    layer, _, _ = _build_layer(
        1,
        0,
        device,
        nvfp4_mode=nvfp4_mode,
        payload_dtype="fp8",
        combine_dtype="bf16" if rs_mode else "fp8",
        grouped_combine=not rs_mode,
        fuse_act=not rs_mode,
    )
    x, ids, weights = _make_inputs(8, LOCAL_EXPERTS, 97, device)
    _forward(layer, x, ids, weights)
    layer.destroy()
    layer.destroy()
    assert layer._workspace is None


def _dist_setup() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


@requires_dist
@pytest.mark.parametrize(
    "nvfp4_mode,payload_dtype,combine_dtype,grouped_combine,route_mode",
    [
        ("w4a8", "fp8", "fp8", True, "all_remote"),
        ("w4a16_rs", "bf16", "bf16", False, "random"),
    ],
)
def test_public_multirank_forward_configs(
    nvfp4_mode: str,
    payload_dtype: str,
    combine_dtype: str,
    grouped_combine: bool,
    route_mode: str,
) -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    layer, w13, w2 = _build_layer(
        world_size,
        rank,
        device,
        nvfp4_mode=nvfp4_mode,
        payload_dtype=payload_dtype,
        combine_dtype=combine_dtype,
        grouped_combine=grouped_combine,
    )
    x, ids, weights = _make_inputs(
        TOKEN_CAPACITY,
        LOCAL_EXPERTS * world_size,
        41 + rank,
        device,
        mode=route_mode,
        rank=rank,
    )
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _assert_close(nvfp4_mode, output, _reference(x, ids, weights, w13, w2))
    dist.barrier()


@requires_dist
def test_public_multirank_uneven_empty_and_recovery() -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    layer, w13, w2 = _build_layer(
        world_size,
        rank,
        device,
        nvfp4_mode="w4a8",
        payload_dtype="fp8",
        combine_dtype="fp8",
        grouped_combine=True,
    )
    num_tokens = 0 if rank == 1 else max(TOKEN_CAPACITY - 7 * rank, 1)
    x, ids, weights = _make_inputs(
        num_tokens,
        LOCAL_EXPERTS * world_size,
        53 + rank,
        device,
        rank=rank,
    )
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    assert output.shape == x.shape
    if num_tokens:
        _assert_close("w4a8", output, _reference(x, ids, weights, w13, w2))
    x, ids, weights = _make_inputs(
        TOKEN_CAPACITY,
        LOCAL_EXPERTS * world_size,
        67 + rank,
        device,
        rank=rank,
    )
    recovered = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _assert_close("w4a8", recovered, _reference(x, ids, weights, w13, w2))
    dist.barrier()
