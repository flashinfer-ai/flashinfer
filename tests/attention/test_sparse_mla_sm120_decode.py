# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public scheduling choices, fused split finalization and graph replay."""

import pytest
import torch
from tests.attention.sparse_mla_prefill_utils import (
    fixture,
    numerical_metrics,
    decode_call as call,
    rope_reference,
)


@pytest.mark.parametrize("impl", ["auto", "decode-splitkv", "decode"])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize(
    "tokens,heads", [(1, 8), (7, 16), (7, 32), (7, 64), (28, 64), (98, 64)]
)
@pytest.mark.parametrize("ragged", [False, True])
def test_decode_math_and_fusion(impl, tokens, heads, ragged, inline):
    f = fixture(tokens, heads, "v41_fp8", ragged=ragged, inline=inline)
    lse = call(f, impl)
    metric = numerical_metrics(f, lse)
    assert metric["finite_output"] and metric["relative_l2"] < 0.07, metric
    if not ragged:
        assert metric["lse_max_abs"] < 0.06, metric
    # Very short ragged rows expose Q's FP8 error directly. Both the unchanged
    # prefill and decode have 0.0643 BF16-reference LSE error on H32/seed42.
    # Check the actual FP8 QK arithmetic tightly instead of loosening that gate.
    torch.testing.assert_close(
        lse.double(), fp8_query_lse_reference(f), atol=2e-5, rtol=0
    )
    # Independent attention accuracy was checked above. This second check
    # isolates the fusion contract: separate Q RoPE -> the same attention core
    # -> inverse output RoPE/FP8 must match the fused composition, including
    # BF16 rounding boundaries and output scales. It is not an independent
    # attention oracle, because the fused and separate paths share that core.
    expected = [x.clone() for x in (f["out"].view(torch.uint8), f["scales"], lse)]
    lse = call(f, impl, False)
    for a, b in zip(
        (f["out"].view(torch.uint8), f["scales"], lse), expected, strict=True
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("impl", ["decode-splitkv", "decode", "auto"])
@pytest.mark.parametrize("empty", ["lengths", "indices"])
@pytest.mark.parametrize("inline", [False, True])
def test_decode_empty_and_mutable_graph(impl, empty, inline):
    f = fixture(7, 64, "v41_fp8", with_sink=False, inline=inline)
    lse = call(f, impl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(f, impl)
    for _ in range(10):
        for ids in f["indices"]:
            ids[:, ::7] = -1
        graph.replay()
    for x in f[empty]:
        x.fill_(0 if empty == "lengths" else -1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isneginf(lse).all()
    assert not torch.count_nonzero(f["out"].float())
    for lens, ids in zip(f["lengths"], f["indices"], strict=True):
        lens.fill_(ids.shape[1])
        ids.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    expected = [x.clone() for x in (f["out"].view(torch.uint8), f["scales"], lse)]
    actual = call(f, impl)
    for a, b in zip(
        (f["out"].view(torch.uint8), f["scales"], actual), expected, strict=True
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("bad", ["invalid", "LOW-LATENCY", None])
def test_decode_invalid_impl(bad):
    f = fixture(7, 64, "v41_fp8")
    with pytest.raises(ValueError, match="impl"):
        call(f, bad)


@pytest.mark.parametrize("inline", [False, True])
def test_decode_explicit_selection(inline):
    f = fixture(7, 64, "v41_fp8", inline=inline)
    for impl in ["decode-splitkv", "decode"]:
        call(f, impl)
        facts = [
            dict(p.plan.inspect())
            for key, p in f["wrapper"]._prepared_calls.items()
            if key[1] == impl
        ]
        assert len(facts) == 1 and facts[0]["implementation"] == (
            "dsv4_fp8_splitkv_decode" if impl == "decode-splitkv" else "dsv4_fp8_decode"
        )
        if impl == "decode":
            assert facts[0]["partial_bytes"] == 0


def test_footer_auto_keeps_small_workloads_on_legacy_decode():
    f = fixture(1, 8, "v41_fp8", inline=False, extra_topk=0)
    call(f, "auto")
    plan = next(iter(f["wrapper"]._prepared_calls.values())).plan.inspect()
    assert plan["implementation"] == "ordinary"
    assert plan["variant"] == 0
    # The automatic cutoff must not restrict either explicit implementation.
    for impl, expected in (
        ("decode-splitkv", "dsv4_fp8_splitkv_decode"),
        ("decode", "dsv4_fp8_decode"),
    ):
        call(f, impl)
        plan = next(
            p.plan.inspect()
            for key, p in f["wrapper"]._prepared_calls.items()
            if key[1] == impl
        )
        assert plan["implementation"] == expected


def fp8_query_lse_reference(f):
    q = rope_reference(f["q"], f["positions"], f["cs"]).bfloat16().float()
    grouped = q.reshape(*q.shape[:2], 16, 32)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    bits = (scale.view(torch.int32) + 0x007FFFFF) & 0x7F800000
    scale = bits.view(torch.float32)
    q = (
        ((grouped / scale).to(torch.float8_e4m3fn).float() * scale)
        .reshape_as(q)
        .double()
    )
    kv = torch.cat(
        [
            d[ix.clamp_min(0).long()].double()
            for d, ix in zip(f["decoded"], f["indices"], strict=True)
        ],
        1,
    )
    mask = torch.cat(
        [
            (ix >= 0)
            & (torch.arange(ix.shape[1], device=ix.device)[None] < ln[:, None])
            for ix, ln in zip(f["indices"], f["lengths"], strict=True)
        ],
        1,
    )
    scores = (q @ kv.transpose(1, 2)) * 512**-0.5
    scores.masked_fill_(~mask[:, None], -torch.inf)
    lse = torch.logsumexp(scores, -1)
    return (
        torch.logaddexp(lse, f["sink"].double()[None]) if f["sink"] is not None else lse
    )


@pytest.mark.parametrize("impl", ["auto", "decode-splitkv", "decode"])
@pytest.mark.parametrize("inline", [False, True])
def test_decode_single_cache(impl, inline):
    f = fixture(7, 64, "v41_fp8", extra_topk=0, inline=inline)
    lse = call(f, impl)
    metric = numerical_metrics(f, lse)
    assert metric["relative_l2"] < 0.07 and metric["lse_max_abs"] < 0.06


@pytest.mark.parametrize("impl", ["auto", "decode-splitkv", "decode"])
@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("extra_page_size", [2, 3, 64])
def test_decode_footer_page_addressing(impl, heads, extra_page_size):
    """Gather across page boundaries with padded pages and a nonzero origin."""
    f = fixture(
        7,
        heads,
        "v41_fp8",
        inline=False,
        pool_tokens=4032,
        extra_page_size=extra_page_size,
        topk=65,
        extra_topk=97,
    )
    # Include both sides of page boundaries, invalid indices, partial tiles,
    # and an empty row. Main and extra caches have independent page geometry.
    for ids, lengths, page in zip(
        f["indices"], f["lengths"], (64, extra_page_size), strict=True
    ):
        ids[:, :4] = torch.tensor(
            [page - 1, page, 2 * page - 1, 2 * page], device="cuda"
        )
        ids[:, 5::11] = -1
        lengths[0] = 0
        lengths[1] = 33
    lse = call(f, impl)
    torch.testing.assert_close(
        lse.double(), fp8_query_lse_reference(f), atol=2e-5, rtol=0
    )
    expected = f["out"].float().clone(), f["scales"].clone(), lse.clone()
    backing = []
    for i, cache in enumerate(f["caches"]):
        flat = cache.reshape(cache.shape[0], -1)
        padded = torch.full(
            (flat.shape[0], flat.shape[1] + 48), 0xA5, dtype=torch.uint8, device="cuda"
        )
        view = padded[:, 16:-32].view_as(cache)
        view.copy_(cache)
        f["caches"][i] = view
        backing.append(padded)
    call(f, impl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_lse = call(f, impl)
    f["out"].fill_(float("nan"))
    f["scales"].fill_(255)
    actual_lse.fill_(float("nan"))
    graph.replay()
    for actual, wanted in zip(
        (f["out"].float(), f["scales"], actual_lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
    for padded in backing:
        assert (padded[:, :16] == 0xA5).all() and (padded[:, -32:] == 0xA5).all()


def test_decode_parallelism_policy():
    from flashinfer.mla._sparse_mla_sm120._execution import get_sparse_mla_sm120_module
    from flashinfer.mla._sparse_mla_sm120._prepared import validate_metadata

    f = fixture(7, 64, "v41_fp8")
    m = validate_metadata(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        f["context"],
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        None,
        None,
        None,
        model=6,
        is_dsv4_nvfp4=False,
        extra_fp4=False,
        value_dim=512,
    )
    module = get_sparse_mla_sm120_module()
    for sms, expected in [(188, "decode-splitkv"), (28, "decode")]:
        plan = module.resolve_decode_attention(m, 0, sms, 101376)
        assert plan.inspect()["implementation"] == (
            "dsv4_fp8_splitkv_decode"
            if expected == "decode-splitkv"
            else "dsv4_fp8_decode"
        )
    with pytest.raises(RuntimeError):
        module.resolve_decode_attention(m, 0, 0, 101376)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (7, "decode-splitkv"),
        (49, "decode"),
        (91, "decode"),
        (98, "decode-splitkv"),
        (119, "decode-splitkv"),
        (126, "decode"),
    ],
)
def test_decode_wave_policy(tokens, expected):
    from flashinfer.mla._sparse_mla_sm120._execution import get_sparse_mla_sm120_module
    from flashinfer.mla._sparse_mla_sm120._prepared import validate_metadata

    f = fixture(1, 64, "v41_fp8")
    m = validate_metadata(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        f["context"],
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        None,
        None,
        None,
        model=6,
        is_dsv4_nvfp4=False,
        extra_fp4=False,
        value_dim=512,
    )
    p = get_sparse_mla_sm120_module().resolve_decode_attention(
        m._replace(tokens=tokens), 0, 188, 101376
    )
    assert p.inspect()["implementation"] == (
        "dsv4_fp8_splitkv_decode" if expected == "decode-splitkv" else "dsv4_fp8_decode"
    )


@pytest.mark.parametrize("impl", ["decode-splitkv", "decode"])
@pytest.mark.parametrize("inline", [False, True])
def test_decode_single_kernel_boundary(impl, tmp_path, inline):
    import re

    f = fixture(7, 64, "v41_fp8", inline=inline)
    call(f, impl)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        call(f, impl)
    graph.replay()
    torch.cuda.synchronize()
    path = tmp_path / "decode.dot"
    graph.debug_dump(str(path))
    nodes = re.findall(r'"graph_\d+_node_\d+"\[.*?\];', path.read_text(), re.DOTALL)
    assert len(nodes) == 1 and 'label="{KERNEL' in nodes[0], nodes
    assert (
        "dsv4_fp8_splitkv_decode" if impl == "decode-splitkv" else "cute_prefill"
    ) in nodes[0]
