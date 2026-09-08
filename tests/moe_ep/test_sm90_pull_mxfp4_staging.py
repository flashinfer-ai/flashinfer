"""CPU unit tests for SM90 Humming MXFP4 activation staging."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from flashinfer.moe_ep import FleetParams, MoEEpConfigError
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.staging import (
    stage_mega_moe_inputs,
    staged_tokens,
    validate_sm90_mxfp4_forward_inputs,
)


_VENDOR_ROOT = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/src"
)
_MXFP4_CODEGEN_ENV_VARS = {
    "MEGA_MXFP4_FC2_SCALE_LOAD_OVERLAP",
    "MEGA_MXFP4_FC2_STREAM_SCALE_PROMOTION",
    "MEGA_MXFP4_FC1_K64_PIPELINE",
    "MEGA_MXFP4_K256_FC1_TWO_GROUP_PIPELINE",
    "MEGA_MXFP4_FC1_SPLIT_FINAL_SCALE",
    "MEGA_MXFP4_K256_FC2_HALF_FRAGMENT",
    "MEGA_MXFP4_REUSE_FC1_ACCUM_SCRATCH",
    "MEGA_MXFP4_SPLIT_FC1_STORE_OVERLAP",
}


def _literal_top_level_assignments(path: Path) -> dict[str, object]:
    assignments: dict[str, object] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                value = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
            assignments[node.targets[0].id] = value
    return assignments


def _workspace(*, capacity: int = 4, hidden: int = 128, top_k: int = 2):
    return (
        torch.empty((capacity, hidden), dtype=torch.float8_e4m3fn),
        torch.empty((capacity, 4), dtype=torch.float32),
        torch.empty((capacity, top_k), dtype=torch.int64),
        torch.empty((capacity, top_k), dtype=torch.float32),
    )


def _routing(num_tokens: int, *, top_k: int = 2):
    ids = torch.arange(num_tokens * top_k, dtype=torch.int64).reshape(num_tokens, top_k)
    weights = torch.full((num_tokens, top_k), 1.0 / top_k, dtype=torch.float32)
    return ids, weights


def test_mxfp4_codegen_policy_is_fixed_and_has_no_hidden_environment_reads():
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for package in ("common", "src", "moe_nvfp4_swapab", "moe_hopper_fp8")
        for path in (_VENDOR_ROOT / package).rglob("*.py")
    )
    assert not (_MXFP4_CODEGEN_ENV_VARS & set(source_text.split()))
    for variable in _MXFP4_CODEGEN_ENV_VARS:
        assert variable not in source_text

    kernel_constants = _literal_top_level_assignments(
        _VENDOR_ROOT / "moe_hopper_fp8/kernel_mxfp4_fp8_glu_fc12_swapab.py"
    )
    assert {
        name: kernel_constants[name]
        for name in (
            "Mxfp4Fc2ScaleLoadOverlap",
            "Mxfp4Fc2StreamScalePromotion",
            "Mxfp4Fc1K64Pipeline",
            "Mxfp4K256Fc1TwoGroupPipeline",
            "Mxfp4Fc1SplitFinalScale",
            "Mxfp4K256Fc2HalfFragment",
        )
    } == {
        "Mxfp4Fc2ScaleLoadOverlap": 0,
        "Mxfp4Fc2StreamScalePromotion": 1,
        "Mxfp4Fc1K64Pipeline": 0,
        "Mxfp4K256Fc1TwoGroupPipeline": 0,
        "Mxfp4Fc1SplitFinalScale": None,
        "Mxfp4K256Fc2HalfFragment": None,
    }

    epilogue_constants = _literal_top_level_assignments(
        _VENDOR_ROOT / "moe_hopper_fp8/epilogue_fp8_swapab.py"
    )
    assert epilogue_constants["SwapABMxfp4ReuseFc1AccumScratch"] == 1
    assert epilogue_constants["SwapABMxfp4SplitFc1StoreOverlap"] == "none"


def test_quantized_stage_uses_one_full_hidden_scale_replicated_four_times():
    hidden = torch.zeros((2, 128), dtype=torch.bfloat16)
    hidden[0, 0] = 448.0
    hidden[0, 1] = -224.0
    hidden[1, 0] = 112.0
    hidden[1, 1] = -56.0
    ids, routing_weights = _routing(2)
    x, x_sf, out_ids, out_weights = _workspace()

    stage_mega_moe_inputs(
        hidden,
        routing_weights,
        ids,
        x,
        x_sf,
        out_ids,
        out_weights,
        quantize_input=True,
    )

    expected_scale = torch.tensor([[1.0], [0.25]], dtype=torch.float32)
    torch.testing.assert_close(x_sf[:2], expected_scale.expand(2, 4))
    torch.testing.assert_close(
        x[:2].float() * x_sf[:2, :1], hidden.float(), rtol=0.0, atol=0.0
    )
    assert torch.equal(out_ids[:2], ids)
    assert torch.equal(out_weights[:2], routing_weights)
    assert torch.equal(out_ids[2:], torch.full((2, 2), -1, dtype=torch.int64))
    assert staged_tokens(out_ids) == 2


def test_prestaged_payload_is_byte_exact_and_logical_scale_is_replicated():
    payload = torch.tensor(
        [[0.0, 0.5, -1.5, 448.0] + [0.0] * 124],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.125]], dtype=torch.float32)
    ids, routing_weights = _routing(1)
    x, x_sf, out_ids, out_weights = _workspace()

    stage_mega_moe_inputs(
        payload,
        routing_weights,
        ids,
        x,
        x_sf,
        out_ids,
        out_weights,
        quantize_input=False,
        scales=scales,
    )

    assert torch.equal(x[:1].view(torch.uint8), payload.view(torch.uint8))
    assert torch.equal(x_sf[:1], scales.expand(1, 4))
    assert staged_tokens(out_ids) == 1


def test_empty_stage_clears_routing_sentinel_and_records_zero():
    hidden = torch.empty((0, 128), dtype=torch.bfloat16)
    ids, routing_weights = _routing(0)
    x, x_sf, out_ids, out_weights = _workspace()
    out_ids.fill_(17)

    stage_mega_moe_inputs(
        hidden,
        routing_weights,
        ids,
        x,
        x_sf,
        out_ids,
        out_weights,
        quantize_input=True,
    )

    assert torch.equal(out_ids, torch.full_like(out_ids, -1))
    assert staged_tokens(out_ids) == 0


@pytest.mark.parametrize(
    ("payload_dtype", "scale_shape", "scale_dtype", "message"),
    [
        (torch.float16, (2, 1), torch.float32, "torch.float8_e4m3fn"),
        (torch.float8_e4m3fn, (2, 4), torch.float32, r"shape \(2, 1\)"),
        (torch.float8_e4m3fn, (2, 1), torch.float16, "torch.float32"),
    ],
)
def test_prestaged_validation_rejects_noncanonical_abi(
    payload_dtype, scale_shape, scale_dtype, message
):
    fleet = FleetParams(
        num_experts=8,
        max_tokens_per_rank=4,
        token_hidden_size=128,
    )
    hidden = torch.empty((2, 128), dtype=payload_dtype)
    ids, routing_weights = _routing(2)
    scales = torch.empty(scale_shape, dtype=scale_dtype)

    with pytest.raises(MoEEpConfigError, match=message):
        validate_sm90_mxfp4_forward_inputs(
            hidden,
            ids,
            routing_weights,
            fleet,
            top_k=2,
            quantize_input=False,
            scales=scales,
        )


def test_forward_validation_requires_cuda_after_shape_and_dtype_contracts():
    fleet = FleetParams(
        num_experts=8,
        max_tokens_per_rank=4,
        token_hidden_size=128,
    )
    hidden = torch.empty((2, 128), dtype=torch.bfloat16)
    ids, routing_weights = _routing(2)

    with pytest.raises(MoEEpConfigError, match="hidden_states must be a CUDA tensor"):
        validate_sm90_mxfp4_forward_inputs(
            hidden,
            ids,
            routing_weights,
            fleet,
            top_k=2,
            quantize_input=True,
        )
