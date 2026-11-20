"""SM12x FP8 Q0 route against a pure-Torch reference."""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp8_q0_route_triton import (
    fp8_q0_route_triton,
    fp8_q0_route_workspace_shapes,
    make_fp8_q0_route_workspace,
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
    routed = topk_ids >= 0
    counts = torch.bincount(topk_ids[routed].long(), minlength=num_experts)
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
    assert torch.equal(seen, routed)


def _assert_token_output(x, ids, weights, experts, output):
    offsets, mapping, routed_weights, q, sf = output
    q_ref, sf_ref = _quantize_reference(x)
    assert q.shape == x.shape and sf.shape == (sf_ref.shape[1], x.shape[0])
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref.t().contiguous())
    routed_q = torch.empty(mapping.numel(), x.shape[1], dtype=q.dtype, device=x.device)
    routed_sf = torch.zeros(
        sf.shape[0], _padded_offset(mapping.numel(), experts), device=x.device
    )
    for expert in range(experts):
        begin, end = map(int, offsets[expert : expert + 2].tolist())
        if begin < end:
            tokens = mapping[begin:end].long()
            routed_q.view(torch.uint8)[begin:end] = q.view(torch.uint8)[tokens]
            padded = _padded_offset(begin, expert)
            routed_sf[:, padded : padded + end - begin] = sf[:, tokens]
    _assert_output(
        x,
        ids,
        weights,
        experts,
        (offsets, mapping, routed_weights, routed_q, routed_sf),
    )


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


@pytest.mark.parametrize(
    "routing,num_tokens,top_k,num_experts",
    (
        ("balanced", 64, 16, 896),
        ("skewed", 64, 16, 896),
        ("large_expert_fallback", 65, 4, 1025),
    ),
)
def test_fp8_q0_route_prefill_distributions(routing, num_tokens, top_k, num_experts):
    _skip_if_not_sm120()
    hidden_size = 256
    x, topk_ids, topk_weights = _make_inputs(
        num_tokens, hidden_size, top_k, num_experts
    )
    if routing == "skewed":
        topk_ids = torch.arange(top_k, dtype=torch.int32, device="cuda")
        topk_ids = topk_ids.expand(num_tokens, -1).contiguous()
    shape13, shape2 = fp8_q0_route_workspace_shapes(
        num_tokens, hidden_size, top_k, num_experts
    )
    output = fp8_q0_route_triton(
        x,
        topk_ids,
        topk_weights,
        num_experts,
        workspace13=torch.empty(shape13, dtype=torch.bfloat16, device="cuda"),
        workspace2=torch.empty(shape2, dtype=torch.bfloat16, device="cuda"),
    )
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
@pytest.mark.parametrize("token_only", (False, True))
@pytest.mark.parametrize(
    "path,num_tokens,top_k,num_experts",
    (
        ("decode", 5, 4, 7),
        ("prefill", 65, 4, 7),
        ("large_expert_fallback", 65, 4, 1025),
    ),
)
def test_fp8_q0_route_skips_unrouted(
    path, num_tokens, top_k, num_experts, pattern, token_only
):
    """A negative expert id gets no routed row; the routed pairs are unchanged."""
    _skip_if_not_sm120()
    hidden_size = 256
    x, topk_ids, topk_weights = _make_inputs(
        num_tokens, hidden_size, top_k, num_experts
    )
    topk_ids = _mask_unrouted(topk_ids, pattern)
    shape13, shape2 = fp8_q0_route_workspace_shapes(
        num_tokens, hidden_size, top_k, num_experts
    )
    # Dirty external workspaces: stale rows must not leak into the result.
    workspace13 = torch.full(shape13, 3.0, dtype=torch.bfloat16, device="cuda")
    workspace2 = torch.full(shape2, 3.0, dtype=torch.bfloat16, device="cuda")
    workspace = make_fp8_q0_route_workspace(
        x, topk_ids, num_experts, workspace13=workspace13, workspace2=workspace2
    )
    output = fp8_q0_route_triton(
        x,
        topk_ids,
        topk_weights,
        num_experts,
        workspace=workspace,
        token_only=token_only,
    )
    if token_only:
        _assert_token_output(x, topk_ids, topk_weights, num_experts, output)
    else:
        _assert_output(x, topk_ids, topk_weights, num_experts, output)
    unrouted = topk_ids < 0
    assert bool((workspace.dst_rows[unrouted] == -1).all().item())
    assert bool((workspace.scale_dst_rows[unrouted] == -1).all().item())


@pytest.mark.parametrize(
    "tokens,experts,caller_workspace",
    ((1, 7, True), (64, 7, False), (65, 7, True), (65, 1025, False)),
)
def test_fp8_token_only_q0(tokens, experts, caller_workspace):
    _skip_if_not_sm120()
    x, ids, weights = _make_inputs(tokens, 1152, 4, experts)
    kwargs = {}
    if caller_workspace:
        shape13, shape2 = fp8_q0_route_workspace_shapes(tokens, 1152, 4, experts)
        kwargs = dict(
            workspace13=torch.empty(shape13, dtype=torch.bfloat16, device="cuda"),
            workspace2=torch.empty(shape2, dtype=torch.bfloat16, device="cuda"),
        )
    output = fp8_q0_route_triton(x, ids, weights, experts, token_only=True, **kwargs)
    _assert_token_output(x, ids, weights, experts, output)
    if caller_workspace:
        assert output[3].data_ptr() == kwargs["workspace13"].data_ptr()
        assert (
            output[4].untyped_storage().data_ptr()
            == kwargs["workspace2"].untyped_storage().data_ptr()
        )


@pytest.mark.parametrize("tokens", (1, 65))
def test_fp8_workspace_reusable_across_token_only_and_routed(tokens):
    _skip_if_not_sm120()
    x, ids, weights = _make_inputs(tokens, 1152, 4, 7)
    workspace = make_fp8_q0_route_workspace(x, ids, 7)
    layout = {
        name: (tensor.shape, tensor.stride(), tensor.data_ptr())
        for name, tensor in vars(workspace).items()
    }
    output = fp8_q0_route_triton(
        x, ids, weights, 7, workspace=workspace, token_only=True
    )
    _assert_token_output(x, ids, weights, 7, output)
    output = fp8_q0_route_triton(x, ids, weights, 7, workspace=workspace)
    assert layout == {
        name: (tensor.shape, tensor.stride(), tensor.data_ptr())
        for name, tensor in vars(workspace).items()
    }
    _assert_output(x, ids, weights, 7, output)


@pytest.mark.parametrize("token_only", (False, True))
def test_fp8_q0_large_input_row_stride(token_only):
    _skip_if_not_sm120()
    row_stride = 1 << 30
    storage = torch.empty(2 * row_stride + 128, dtype=torch.bfloat16, device="cuda")
    x = storage.as_strided((3, 128), (row_stride, 1))
    x.copy_(torch.arange(384, device="cuda").reshape(3, 128) / 13)
    ids = torch.arange(3, dtype=torch.int32, device="cuda")[:, None]
    weights = torch.ones(3, 1, device="cuda")
    output = fp8_q0_route_triton(x, ids, weights, 3, token_only=token_only)
    if token_only:
        _assert_token_output(x, ids, weights, 3, output)
    else:
        _assert_output(x, ids, weights, 3, output)


def test_fp8_q0_rejects_workspace_overlapping_strided_input():
    _skip_if_not_sm120()
    storage = torch.empty(1024, dtype=torch.bfloat16, device="cuda")
    x = storage.as_strided((3, 128), (256, 1))
    ids = torch.arange(3, dtype=torch.int32, device="cuda")[:, None]
    weights = torch.ones(3, 1, device="cuda")
    shape13, shape2 = fp8_q0_route_workspace_shapes(3, 128, 1, 3)
    with pytest.raises(ValueError, match="overlap|alias"):
        fp8_q0_route_triton(
            x,
            ids,
            weights,
            3,
            workspace13=storage[512 : 512 + shape13[0]],
            workspace2=torch.empty(shape2, dtype=torch.bfloat16, device="cuda"),
            token_only=True,
        )
