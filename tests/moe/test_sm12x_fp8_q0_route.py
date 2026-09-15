"""SM12x FP8 Q0 route against a pure-Torch reference."""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp8_q0_route_triton import (
    fp8_q0_route_triton,
    fp8_q0_route_workspace_shapes,
)
from flashinfer.utils import is_sm120a_supported


SF_M_ALIGN = 4
pytestmark = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="cute_dsl not available"
)


def _skip_if_not_sm120():
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")


def _padded_offset(offset, expert):
    return (offset + expert * (SF_M_ALIGN - 1)) // SF_M_ALIGN * SF_M_ALIGN


def _make_inputs(num_tokens, hidden_size, top_k, num_experts):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.randn(
        num_tokens,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    ids = torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
    ids = (ids % num_experts).reshape(num_tokens, top_k)
    weights = torch.rand(num_tokens, top_k, generator=generator, device="cuda")
    return x, ids, weights


def _quantize_reference(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scales = torch.clamp(blocks.abs().amax(dim=2), min=1.0e-4) / 448.0
    quant = (blocks / scales[:, :, None]).to(torch.float8_e4m3fn).reshape_as(x)
    return quant, scales


def _assert_output(x, topk_ids, topk_weights, num_experts, output):
    offsets, token_map, token_weights, q_out, scale_out = output
    counts = torch.bincount(topk_ids.flatten().long(), minlength=num_experts)
    offsets_ref = torch.zeros(num_experts + 1, dtype=torch.int32, device=x.device)
    offsets_ref[1:] = counts.cumsum(0, dtype=torch.int32)
    assert torch.equal(offsets, offsets_ref)
    q_ref, scale_ref = _quantize_reference(x)
    seen = torch.zeros_like(topk_ids, dtype=torch.bool)
    for expert in range(num_experts):
        begin, end = map(int, offsets[expert : expert + 2].tolist())
        scale_begin = _padded_offset(begin, expert)
        for row in range(begin, end):
            token = int(token_map[row].item())
            slot = int(
                ((topk_ids[token] == expert) & ~seen[token])
                .nonzero(as_tuple=False)[0, 0]
                .item()
            )
            seen[token, slot] = True
            assert torch.equal(token_weights[row], topk_weights[token, slot])
            assert torch.equal(
                q_out[row].view(torch.uint8), q_ref[token].view(torch.uint8)
            )
            scale_row = scale_begin + row - begin
            assert torch.equal(scale_out[:, scale_row], scale_ref[token])
    assert bool(seen.all().item())


@pytest.mark.parametrize(
    "num_tokens,hidden_size,top_k,num_experts",
    ((3, 1152, 1, 7), (65, 256, 4, 7)),
)
def test_fp8_q0_route_matches_torch(num_tokens, hidden_size, top_k, num_experts):
    _skip_if_not_sm120()
    x, topk_ids, topk_weights = _make_inputs(
        num_tokens, hidden_size, top_k, num_experts
    )
    kwargs = {}
    if num_tokens != 3:
        shape13, shape2 = fp8_q0_route_workspace_shapes(
            num_tokens, hidden_size, top_k, num_experts
        )
        kwargs = {
            "workspace13": torch.empty(shape13, dtype=torch.bfloat16, device="cuda"),
            "workspace2": torch.empty(shape2, dtype=torch.bfloat16, device="cuda"),
        }
    output = fp8_q0_route_triton(x, topk_ids, topk_weights, num_experts, **kwargs)
    _assert_output(x, topk_ids, topk_weights, num_experts, output)


@pytest.mark.parametrize("workspace_name", ("workspace13", "workspace2"))
def test_fp8_q0_route_rejects_non_1d_workspace(workspace_name):
    _skip_if_not_sm120()
    num_tokens, hidden_size, top_k, num_experts = 3, 256, 1, 7
    x, topk_ids, topk_weights = _make_inputs(
        num_tokens, hidden_size, top_k, num_experts
    )
    shape13, shape2 = fp8_q0_route_workspace_shapes(
        num_tokens, hidden_size, top_k, num_experts
    )
    sizes = {"workspace13": shape13[0], "workspace2": shape2[0]}
    kwargs = {
        workspace_name: torch.empty(
            (1, sizes[workspace_name]), dtype=torch.bfloat16, device="cuda"
        )
    }
    with pytest.raises(ValueError, match="1D buffer"):
        fp8_q0_route_triton(x, topk_ids, topk_weights, num_experts, **kwargs)
