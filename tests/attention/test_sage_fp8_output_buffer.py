"""Sage output views preserve numerics, storage boundaries and copy fallbacks."""

import pytest
import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100/SM103",
)


def inputs(batch, seq, heads, topk):
    torch.manual_seed(42)
    q, k, v = [
        torch.randn(batch, seq, heads, 128, device="cuda", dtype=torch.bfloat16)
        .mul_(0.5)
        .to(torch.float8_e4m3fn)
        for _ in range(3)
    ]
    blocks = (seq + 63) // 64
    index = (
        (
            (
                torch.arange(blocks, device="cuda")[:, None]
                + torch.arange(topk, device="cuda")
            )
            % blocks
        )
        .to(torch.int32)[None, None]
        .expand(batch, heads, -1, -1)
        .contiguous()
    )
    scales = dict(
        q_scale=torch.ones(batch, heads, seq, device="cuda"),
        k_scale=torch.ones(batch, heads, (seq + 15) // 16, device="cuda"),
        v_scale=torch.ones(heads, 128, device="cuda"),
    )
    return (q, k, v, index, topk), scales


@pytest.mark.parametrize(
    "batch,seq,heads,topk",
    [(1, 65, 3, 2), (2, 67, 3, 2), (1, 652, 14, 8), (1, 39756, 14, 160)],
)
@pytest.mark.parametrize(
    "layout", ["packed", "head_padding", "channel_stride", "unaligned"]
)
def test_output_views_and_guards(batch, seq, heads, topk, layout):
    args, kwargs = inputs(batch, seq, heads, topk)
    expected = bsa_attn_sm100_blk64_fwd(*args, **kwargs, kv_splits=1, use_clc=False)[0]
    if layout == "unaligned":
        backing = torch.full(
            (batch * seq * heads * 128 + 1,), -123, device="cuda", dtype=torch.bfloat16
        )
        out = backing[1:].view(batch, seq, heads, 128)
    else:
        width = (
            256
            if layout == "channel_stride"
            else 136
            if layout == "head_padding"
            else 128
        )
        backing = torch.full(
            (batch, seq + 2, heads, width), -123, device="cuda", dtype=torch.bfloat16
        )
        out = (
            backing[:, 1:-1, :, ::2]
            if layout == "channel_stride"
            else backing[:, 1:-1, :, :128]
        )
    actual = bsa_attn_sm100_blk64_fwd(
        *args, **kwargs, kv_splits=1, use_clc=False, out=out
    )[0]
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if layout == "unaligned":
        assert backing[0] == -123
    else:
        assert torch.all(backing[:, 0] == -123) and torch.all(backing[:, -1] == -123)
        if layout == "channel_stride":
            assert torch.all(backing[..., 1::2] == -123)
        if layout == "head_padding":
            assert torch.all(backing[..., 128:] == -123)


def test_split_kv_output_fallback():
    args, kwargs = inputs(1, 652, 14, 8)
    expected = bsa_attn_sm100_blk64_fwd(*args, **kwargs, kv_splits=2, use_clc=False)[0]
    out = torch.empty_like(expected)
    actual = bsa_attn_sm100_blk64_fwd(
        *args, **kwargs, kv_splits=2, use_clc=False, out=out
    )[0]
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_direct_output_has_no_copy_kernel():
    args, kwargs = inputs(1, 652, 14, 8)
    out = torch.empty((1, 652, 14, 128), device="cuda", dtype=torch.bfloat16)
    bsa_attn_sm100_blk64_fwd(*args, **kwargs, out=out, kv_splits=1, use_clc=False)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        bsa_attn_sm100_blk64_fwd(*args, **kwargs, out=out, kv_splits=1, use_clc=False)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
    assert names
    assert not [n for n in names if "copy_kernel" in n or "elementwise_kernel" in n]


@pytest.mark.parametrize("operand_index", [1, 2])
def test_output_aliases_kv_uses_copyback(operand_index):
    args, kwargs = inputs(1, 652, 14, 8)
    out = torch.empty((1, 652, 14, 128), device="cuda", dtype=torch.bfloat16)
    alias = out.view(torch.float8_e4m3fn)[..., :128]
    alias.copy_(args[operand_index])
    aliased_args = list(args)
    aliased_args[operand_index] = alias
    expected = bsa_attn_sm100_blk64_fwd(
        *aliased_args, **kwargs, kv_splits=1, use_clc=False
    )[0]
    actual = bsa_attn_sm100_blk64_fwd(
        *aliased_args, **kwargs, kv_splits=1, use_clc=False, out=out
    )[0]
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
