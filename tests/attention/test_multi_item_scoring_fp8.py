# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""MIS tile jumps must prefetch FP8 scales from the same physical KV tile."""

import functools
import importlib.util
import math
from pathlib import Path

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


@functools.cache
def _reference_case_maker():
    """Load the shared dense-case builder for standalone and importlib tests."""
    # Support pytest's importlib mode and running this file on its own.
    spec = importlib.util.spec_from_file_location(
        "_mis_fp8_reference",
        Path(__file__).with_name("test_multi_item_scoring_tile_skip.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._make_case


@pytest.mark.parametrize(
    "q_dtype,kv_dtype",
    [(torch.float16, torch.float8_e4m3fn), (torch.bfloat16, torch.float8_e5m2)],
)
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("group", [1, 2, 3])
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("policy", ["default", "no_split", "fixed_3_pages"])
@pytest.mark.parametrize("prefix,items,query_tokens", [(17, 16, 71), (129, 8, None)])
def test_mis_fp8_physical_scale_bounds(
    q_dtype, kv_dtype, layout, group, head_dim, policy, prefix, items, query_tokens
):
    """Match quantized MIS references with token/head scales across tile jumps."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if (
        q_dtype == torch.bfloat16
        and get_compute_capability(torch.device("cuda:0"))[0] < 8
    ):
        pytest.skip("bf16 Q is not supported with FP8 token-head scale on SM75")

    q, kv16, meta, _, _ = _reference_case_maker()(
        [(prefix, [31] * items)],
        dtype=q_dtype,
        dim=head_dim,
        group=group,
        query_tokens=query_tokens,
    )
    # A wide, deterministic token/head scale range makes using a skipped
    # physical tile's scales produce a visible numerical error.
    exponent = (
        torch.arange(kv16.numel() // head_dim, device="cuda").reshape(kv16.shape[:-1])
        % 9
        - 4
    )
    kv16 = kv16 * exponent.float().exp2().unsqueeze(-1).to(q_dtype)
    maximum = 448.0 if kv_dtype == torch.float8_e4m3fn else 57344.0
    scale = (kv16.float().abs().amax(-1) / maximum).clamp(min=1e-12)
    assert float(scale.max() / scale.min()) > 32
    kv = (kv16.float() / scale.unsqueeze(-1)).to(kv_dtype)
    # CPU float32 reference starts from the quantized values and exact scales;
    # it does not call any FlashInfer reference kernel.
    reference, ref_lse = _dense_reference(
        q, kv, scale, meta, prefix, items, head_dim, group
    )
    sf = (scale[:, 0], scale[:, 1])
    if layout == "HND":
        kv = kv.permute(0, 1, 3, 2, 4).contiguous()
        sf = tuple(x.permute(0, 2, 1).contiguous() for x in sf)
    meta.update(kv_data_type=kv_dtype, use_token_head_sf=True)
    options = (
        {"disable_split_kv": True}
        if policy == "no_split"
        else ({"fixed_split_size": 3} if policy == "fixed_3_pages" else {})
    )
    workspace = torch.full((128 * 1024 * 1024,), 255, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper.plan(**meta, **options)
    out, lse = wrapper.run(q, kv, kv_cache_sf=sf, return_lse=True)
    torch.testing.assert_close(out.cpu().float(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)


def _dense_reference(q, kv, scale, meta, prefix, items, head_dim, group):
    """Reference quantized inputs on CPU without a FlashInfer reference kernel."""
    dequant = kv.cpu().float()
    if scale is not None:
        dequant = dequant * scale.cpu().unsqueeze(-1)
    physical = meta["paged_kv_indices"].cpu().long()
    klen = prefix + 32 * items
    cache = dequant[physical]
    k = cache[:, 0].reshape(-1, 4, head_dim)[:klen].repeat_interleave(group, 1)
    v = cache[:, 1].reshape(-1, 4, head_dim)[:klen].repeat_interleave(group, 1)
    query = q.cpu().float()
    keys = torch.arange(klen, device="cpu")
    queries = torch.arange(query.shape[0], device="cpu") + klen - query.shape[0]
    item_ids = torch.tensor(
        [-1] * prefix + [i for i in range(items) for _ in range(32)], device="cpu"
    )
    content = torch.tensor(
        [False] * prefix + [p > 0 for _ in range(items) for p in range(32)],
        device="cpu",
    )
    allowed = (keys[None, :] <= queries[:, None]) & (
        (keys[None, :] < prefix)
        | (
            (item_ids[queries, None] == item_ids[None, :])
            & content[queries, None]
            & content[None, :]
        )
    )
    scores = torch.einsum("qhd,khd->hqk", query, k) / math.sqrt(head_dim)
    scores.masked_fill_(~allowed.unsqueeze(0), -torch.inf)
    reference = torch.einsum("hqk,khd->qhd", scores.softmax(-1).nan_to_num(0), v)
    ref_lse = scores.logsumexp(-1).T / math.log(2)
    return reference, ref_lse


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("group", [1, 2, 3])
@pytest.mark.parametrize("policy", ["default", "no_split", "fixed_3_pages"])
@pytest.mark.parametrize("prefix,items,query_tokens", [(17, 16, 71), (129, 8, None)])
def test_mis_fp8_query_prefetch_bounds(
    layout, group, policy, prefix, items, query_tokens
):
    """Check FP8-query MIS prefetch against a dense reference on SM120."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if get_compute_capability(torch.device("cuda:0")) != (12, 0):
        pytest.skip("FA2 FP8 query public API requires SM120")
    q16, kv16, meta, _, _ = _reference_case_maker()(
        [(prefix, [31] * items)],
        dtype=torch.float16,
        dim=256,
        group=group,
        query_tokens=query_tokens,
    )
    q, kv = q16.to(torch.float8_e4m3fn), kv16.to(torch.float8_e4m3fn)
    reference, ref_lse = _dense_reference(q, kv, None, meta, prefix, items, 256, group)
    if layout == "HND":
        kv = kv.permute(0, 1, 3, 2, 4).contiguous()
    meta.update(q_data_type=q.dtype, kv_data_type=kv.dtype, o_data_type=torch.float16)
    options = (
        {"disable_split_kv": True}
        if policy == "no_split"
        else ({"fixed_split_size": 3} if policy == "fixed_3_pages" else {})
    )
    workspace = torch.full((128 * 1024 * 1024,), 255, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2"
    )
    wrapper.plan(**meta, **options)
    out, lse = wrapper.run(q, kv, return_lse=True)
    torch.testing.assert_close(out.cpu().float(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)


def test_mis_fp8_query_rejects_token_head_scales():
    """Reject token/head KV scales when the query dtype is FP8."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _, _, meta, _, _ = _reference_case_maker()([(17, [31] * 2)])
    meta.update(
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
        use_token_head_sf=True,
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), backend="fa2"
    )
    with pytest.raises(ValueError, match="requires an fp16/bf16 query"):
        wrapper.plan(**meta)
