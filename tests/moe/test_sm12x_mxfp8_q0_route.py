"""SM12x MXFP8 Q0 route against a pure-Torch reference."""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_q0_route_triton import (
    mxfp8_q0_route_triton,
    mxfp8_q0_route_workspace_shapes,
    make_mxfp8_q0_route_workspace,
)
from flashinfer.utils import is_sm120a_supported
from tests.moe.test_cute_dsl_sm12x_mxfp8_mxfp4 import (
    compute_padded_offset,
    mxfp8_act_quantize,
    pack_ue8m0_to_int32,
)


def _cuda_13_or_newer() -> bool:
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available"),
    pytest.mark.skipif(
        not _cuda_13_or_newer(),
        reason="SM12x MXFP8 Q0 route requires CUDA 13 or later",
    ),
]


def _skip_if_not_sm120():
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")


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


def _assert_output(x, topk_ids, topk_weights, num_experts, output):
    offsets, token_map, token_weights, q_out, scale_out = output
    routed = topk_ids >= 0
    counts = torch.bincount(topk_ids[routed].long(), minlength=num_experts)
    offsets_ref = torch.zeros(num_experts + 1, dtype=torch.int32, device=x.device)
    offsets_ref[1:] = counts.cumsum(0, dtype=torch.int32)
    assert torch.equal(offsets, offsets_ref)
    q_ref, scale_ref = mxfp8_act_quantize(x)
    scale_ref = pack_ue8m0_to_int32(scale_ref)
    seen = torch.zeros_like(topk_ids, dtype=torch.bool)
    for expert in range(num_experts):
        begin, end = map(int, offsets[expert : expert + 2].tolist())
        scale_begin = compute_padded_offset(begin, expert)
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
    assert torch.equal(seen, routed)


@pytest.mark.parametrize(
    "routing,num_tokens,top_k,num_experts",
    (
        ("balanced", 64, 16, 896),
        ("skewed", 64, 16, 896),
        ("large_expert_fallback", 65, 4, 1025),
    ),
)
def test_mxfp8_q0_route_prefill_distributions(routing, num_tokens, top_k, num_experts):
    _skip_if_not_sm120()
    hidden_size = 512
    x, topk_ids, topk_weights = _make_inputs(
        num_tokens, hidden_size, top_k, num_experts
    )
    if routing == "skewed":
        topk_ids = torch.arange(top_k, dtype=torch.int32, device="cuda")
        topk_ids = topk_ids.expand(num_tokens, -1).contiguous()
    shape13, shape2 = mxfp8_q0_route_workspace_shapes(
        num_tokens, hidden_size, top_k, num_experts
    )
    output = mxfp8_q0_route_triton(
        x,
        topk_ids,
        topk_weights,
        num_experts,
        workspace13=torch.empty(shape13, dtype=torch.bfloat16, device="cuda"),
        workspace2=torch.empty(shape2, dtype=torch.bfloat16, device="cuda"),
    )
    _assert_output(x, topk_ids, topk_weights, num_experts, output)


def _mask_unrouted(topk_ids, pattern):
    """Mark CUDA-graph padding the way vLLM does: expert id -1, weight kept."""
    num_tokens, top_k = topk_ids.shape
    if pattern == "tail":
        masked = torch.zeros_like(topk_ids, dtype=torch.bool)
        masked[num_tokens - max(1, num_tokens // 4) :] = True
    elif pattern == "all":
        masked = torch.ones_like(topk_ids, dtype=torch.bool)
    elif pattern == "mixed":
        token = torch.arange(num_tokens, device=topk_ids.device)[:, None]
        slot = torch.arange(top_k, device=topk_ids.device)[None, :]
        masked = (token + slot) % 2 == 0
    else:
        raise ValueError(pattern)
    return topk_ids.masked_fill(masked, -1)


@pytest.mark.parametrize("pattern", ("tail", "all", "mixed"))
@pytest.mark.parametrize(
    "path,num_tokens,top_k,num_experts",
    (
        ("decode", 5, 4, 7),
        ("prefill", 65, 4, 7),
        ("large_expert_fallback", 65, 4, 1025),
    ),
)
def test_mxfp8_q0_route_skips_unrouted(path, num_tokens, top_k, num_experts, pattern):
    """A negative expert id gets no routed row; the routed pairs are unchanged."""
    _skip_if_not_sm120()
    hidden_size = 512
    x, topk_ids, topk_weights = _make_inputs(
        num_tokens, hidden_size, top_k, num_experts
    )
    topk_ids = _mask_unrouted(topk_ids, pattern)
    shape13, shape2 = mxfp8_q0_route_workspace_shapes(
        num_tokens, hidden_size, top_k, num_experts
    )
    # Dirty external workspaces: stale rows must not leak into the result.
    workspace13 = torch.full(shape13, 3.0, dtype=torch.bfloat16, device="cuda")
    workspace2 = torch.full(shape2, 3.0, dtype=torch.bfloat16, device="cuda")
    workspace = make_mxfp8_q0_route_workspace(
        x, topk_ids, num_experts, workspace13=workspace13, workspace2=workspace2
    )
    output = mxfp8_q0_route_triton(
        x, topk_ids, topk_weights, num_experts, workspace=workspace
    )
    _assert_output(x, topk_ids, topk_weights, num_experts, output)
    unrouted = topk_ids < 0
    assert bool((workspace.dst_rows[unrouted] == -1).all().item())
    assert bool((workspace.scale_dst_rows[unrouted] == -1).all().item())
