"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

HEAD_DIM = 128
PAGE_SIZE = 16


def _skip_unless_fa2_jit():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    if get_compute_capability(torch.device("cuda")) < (7, 5):
        pytest.skip("FA2 prefill JIT requires SM75 or newer.")


def _dense_reference(
    q,
    k,
    v,
    qo_lens,
    kv_lens,
    ranges,
    causal_window_left,
    range_window_left,
    sm_scale,
):
    """Materialize the mask and run attention in float32."""
    outputs = []
    q_off = 0
    kv_off = 0
    for qo_len, kv_len in zip(qo_lens, kv_lens, strict=True):
        qi = q[q_off : q_off + qo_len].float()
        ki = k[kv_off : kv_off + kv_len].float()
        vi = v[kv_off : kv_off + kv_len].float()
        ri = ranges[q_off : q_off + qo_len]

        num_qo_heads = qi.shape[1]
        num_kv_heads = ki.shape[1]
        group = num_qo_heads // num_kv_heads

        q_abs = torch.arange(kv_len - qo_len, kv_len, device=qi.device)
        kv_idx = torch.arange(kv_len, device=qi.device)
        dist = q_abs[:, None] - kv_idx[None, :]

        causal = dist >= 0
        if causal_window_left >= 0:
            # DefaultAttention keeps a key when
            #   kv_idx + qo_len + window_left >= kv_len + qo_idx,
            # which is q_abs - kv <= window_left. Written from that inequality
            # rather than from the variant, so the two cannot agree by copying
            # the same mistake.
            causal = causal & (dist <= causal_window_left)

        start = ri[:, 0][:, None]
        end = ri[:, 1][:, None]
        in_range = (kv_idx[None, :] >= start) & (kv_idx[None, :] <= end)
        if range_window_left > 0:
            in_range = in_range & (dist < range_window_left)

        mask = causal | in_range

        out = torch.empty_like(qi)
        for h in range(num_qo_heads):
            kh = ki[:, h // group]
            vh = vi[:, h // group]
            logits = (qi[:, h] @ kh.transpose(0, 1)) * sm_scale
            logits = logits.masked_fill(~mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            out[:, h] = probs @ vh
        outputs.append(out)
        q_off += qo_len
        kv_off += kv_len
    return torch.cat(outputs, dim=0)


def _build_paged(k, v, kv_lens, device, dtype):
    """Pack per-request K/V into pages whose ids are shuffled.

    The pages themselves are a dense tensor; what is deliberately not a
    contiguous run is the page mapping, so the kernel has to follow
    ``paged_kv_indices`` instead of striding through the cache.
    """
    pages_per_req = [(l + PAGE_SIZE - 1) // PAGE_SIZE for l in kv_lens]
    total_pages = sum(pages_per_req)
    num_kv_heads = k.shape[1]
    # Shuffle page ids so the indices are not a contiguous run.
    page_order = torch.randperm(total_pages, device=device)
    kv_cache = torch.zeros(
        total_pages, 2, PAGE_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=dtype
    )
    indptr = [0]
    indices = []
    last_page_len = []
    kv_off = 0
    cursor = 0
    for kv_len, npages in zip(kv_lens, pages_per_req, strict=True):
        req_pages = page_order[cursor : cursor + npages]
        cursor += npages
        for p in range(npages):
            lo = p * PAGE_SIZE
            hi = min(lo + PAGE_SIZE, kv_len)
            n = hi - lo
            kv_cache[req_pages[p], 0, :n] = k[kv_off + lo : kv_off + hi]
            kv_cache[req_pages[p], 1, :n] = v[kv_off + lo : kv_off + hi]
        indices.append(req_pages)
        indptr.append(indptr[-1] + npages)
        last = kv_len - (npages - 1) * PAGE_SIZE
        last_page_len.append(last)
        kv_off += kv_len
    return (
        kv_cache,
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.cat(indices).to(torch.int32),
        torch.tensor(last_page_len, dtype=torch.int32, device=device),
    )


def _run_case(
    qo_lens,
    kv_lens,
    ranges_per_req,
    *,
    causal_window_left=-1,
    range_window_left=-1,
    num_qo_heads=8,
    num_kv_heads=2,
    seed=0,
    o_dtype=None,
    q_scale=None,
    k_scale=None,
    v_scale=None,
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(seed)

    total_q = sum(qo_lens)
    total_kv = sum(kv_lens)
    q = torch.randn(total_q, num_qo_heads, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(total_kv, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(total_kv, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)

    ranges = torch.full((total_q, 2), -1, dtype=torch.int32, device=device)
    off = 0
    for qo_len, spans in zip(qo_lens, ranges_per_req, strict=True):
        for q_local, (start, end) in spans:
            ranges[off + q_local, 0] = start
            ranges[off + q_local, 1] = end
        off += qo_len

    kv_cache, kv_indptr, kv_indices, last_page_len = _build_paged(
        k, v, kv_lens, device, dtype
    )
    qo_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_dtype,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=o_dtype,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    out = wrapper.run(
        q,
        kv_cache,
        ranges,
        causal_window_left=causal_window_left,
        range_window_left=range_window_left,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if o_dtype is not None:
        assert out.dtype == o_dtype, (
            f"output dtype {out.dtype} ignored the o_data_type {o_dtype} "
            "the wrapper was built with"
        )
    # q/k scales are folded into sm_scale by the parent; v_scale is applied to
    # the output after the kernel returns, so the reference has to do the same.
    sm_scale = sm_scale * (q_scale or 1.0) * (k_scale or 1.0)
    ref = _dense_reference(
        q,
        k,
        v,
        qo_lens,
        kv_lens,
        ranges,
        causal_window_left,
        range_window_left,
        sm_scale,
    )
    if v_scale is not None:
        ref = ref * v_scale
    return out, ref, (q, k, v, qo_lens, kv_lens, ranges, sm_scale)


def test_bidirectional_ranges_matches_dense_reference():
    """Multiple requests, mixed shapes, GQA, shuffled page ids, sentinels."""
    _skip_unless_fa2_jit()
    qo_lens = [96, 48, 17]
    kv_lens = [96, 64, 40]
    ranges_per_req = [
        # A span wider than one KV tile, on the queries that fall inside it.
        [(i, (16, 60)) for i in range(16, 61)],
        [(i, (4, 40)) for i in range(4, 41)],
        # Third request is all sentinel: plain causal.
        [],
    ]
    out, ref, _ = _run_case(qo_lens, kv_lens, ranges_per_req)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_bidirectional_ranges_differs_from_causal_only():
    """The spans must actually change the result, not be silently ignored."""
    _skip_unless_fa2_jit()
    qo_lens = [96]
    kv_lens = [96]
    spans = [[(i, (16, 60)) for i in range(16, 61)]]
    out, ref, ctx = _run_case(qo_lens, kv_lens, spans)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    q, k, v, qo_lens_, kv_lens_, ranges, sm_scale = ctx
    causal_only = torch.full_like(ranges, -1)
    causal_ref = _dense_reference(
        q, k, v, qo_lens_, kv_lens_, causal_only, -1, -1, sm_scale
    )
    assert not torch.allclose(ref, causal_ref, atol=1e-2, rtol=1e-2), (
        "the bidirectional spans did not change the reference, so this case "
        "cannot detect a mask that ignores them"
    )


def test_bidirectional_ranges_window_semantics():
    """The causal window and the range clamp are separate knobs."""
    _skip_unless_fa2_jit()
    qo_lens = [80]
    kv_lens = [80]
    spans = [[(i, (8, 70)) for i in range(8, 71)]]

    unclamped, ref_unclamped, _ = _run_case(
        qo_lens, kv_lens, spans, causal_window_left=16, range_window_left=-1
    )
    torch.testing.assert_close(unclamped.float(), ref_unclamped, atol=2e-2, rtol=2e-2)

    clamped, ref_clamped, _ = _run_case(
        qo_lens, kv_lens, spans, causal_window_left=16, range_window_left=24
    )
    torch.testing.assert_close(clamped.float(), ref_clamped, atol=2e-2, rtol=2e-2)

    assert not torch.allclose(ref_unclamped, ref_clamped, atol=1e-2, rtol=1e-2), (
        "the range clamp did not change the reference for this shape"
    )


def test_causal_window_zero_keeps_only_the_diagonal():
    """``0`` is a real window here, not "off".

    With every range a sentinel, a window of ``0`` leaves each query attending
    exactly one key, so softmax is a no-op and the output is that key's value
    row. That expectation comes from the contract rather than from the dense
    reference, so it holds even if both were wrong in the same direction.
    """
    _skip_unless_fa2_jit()
    qo_lens = [64]
    kv_lens = [64]
    num_qo_heads, num_kv_heads = 8, 2
    out, _, (_, _, v, _, _, _, _) = _run_case(
        qo_lens,
        kv_lens,
        [[]],
        causal_window_left=0,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
    )
    group = num_qo_heads // num_kv_heads
    expected = torch.stack(
        [v[:, h // group] for h in range(num_qo_heads)], dim=1
    ).float()
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("window", [0, 1, 7, 16])
def test_causal_window_keeps_window_plus_one_keys(window):
    """``N`` keeps ``N + 1`` keys, counted from the attention itself.

    All logits are equal, so softmax is uniform over whatever was kept, and a
    value that carries the key index makes the output the mean of the kept
    indices. For a window of ``N`` the kept span is ``[max(q - N, 0), q]``, so
    that mean is a closed form neither mask is consulted for.
    """
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    kv_len = qo_len = 48

    q = torch.zeros(qo_len, 1, HEAD_DIM, device=device, dtype=dtype)
    k = torch.zeros(kv_len, 1, HEAD_DIM, device=device, dtype=dtype)
    v = torch.zeros(kv_len, 1, HEAD_DIM, device=device, dtype=dtype)
    v[:, 0, 0] = torch.arange(kv_len, device=device, dtype=torch.float32).to(dtype)

    kv_cache, kv_indptr, kv_indices, last_page_len = _build_paged(
        k, v, [kv_len], device, dtype
    )
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    ranges = torch.full((qo_len, 2), -1, dtype=torch.int32, device=device)

    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        q_data_type=dtype,
        kv_data_type=dtype,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        1,
        1,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out = wrapper.run(q, kv_cache, ranges, causal_window_left=window)

    q_abs = torch.arange(kv_len, device=device, dtype=torch.float32)
    first_kept = torch.clamp(q_abs - window, min=0)
    expected_mean = (q_abs + first_kept) / 2.0
    torch.testing.assert_close(
        out[:, 0, 0].float(), expected_mean, atol=5e-2, rtol=5e-2
    )


def test_causal_window_boundary_n_differs_from_n_plus_one():
    """``N`` and ``N + 1`` must not produce the same attention."""
    _skip_unless_fa2_jit()
    qo_lens = [64]
    kv_lens = [64]
    out_n, _, _ = _run_case(qo_lens, kv_lens, [[]], causal_window_left=8, seed=3)
    out_n1, _, _ = _run_case(qo_lens, kv_lens, [[]], causal_window_left=9, seed=3)
    assert not torch.allclose(out_n.float(), out_n1.float(), atol=1e-3, rtol=1e-3), (
        "window N and N + 1 kept the same keys"
    )


@pytest.mark.parametrize("off_value", [-1, -8])
def test_causal_window_negative_is_unbounded(off_value):
    """Only a negative value disables the window, and every one behaves alike."""
    _skip_unless_fa2_jit()
    qo_lens = [64]
    kv_lens = [64]
    out_off, _, _ = _run_case(
        qo_lens, kv_lens, [[]], causal_window_left=off_value, seed=5
    )
    out_full, _, _ = _run_case(
        qo_lens, kv_lens, [[]], causal_window_left=kv_lens[0], seed=5
    )
    torch.testing.assert_close(out_off.float(), out_full.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("qo_len", [1, 7, 15, 31, 33, 65])
def test_bidirectional_ranges_padded_query_lanes(qo_len):
    """Tails that do not fill a CTA query tile must not read out of bounds."""
    _skip_unless_fa2_jit()
    kv_len = qo_len + 23
    spans = [[(i, (2, min(kv_len - 1, 2 + qo_len))) for i in range(qo_len)]]
    out, ref, _ = _run_case([qo_len], [kv_len], spans, seed=qo_len)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_bidirectional_ranges_rejects_bad_ranges():
    """Shape, dtype and device are checked before the kernel runs."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    q = torch.randn(8, 8, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = torch.zeros(
        1, 2, PAGE_SIZE, 2, HEAD_DIM, device=device, dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="int32"):
        wrapper.run(q, cache, torch.zeros(8, 2, dtype=torch.int64, device=device))
    with pytest.raises(ValueError, match=r"\[total_q, 2\]"):
        wrapper.run(q, cache, torch.zeros(16, dtype=torch.int32, device=device))
    with pytest.raises(ValueError, match="one row per query token"):
        wrapper.run(q, cache, torch.zeros(4, 2, dtype=torch.int32, device=device))
    with pytest.raises(ValueError, match="query device"):
        wrapper.run(q, cache, torch.zeros(8, 2, dtype=torch.int32))


def test_bidirectional_ranges_rejects_multi_item_scoring():
    """prefix_len_ptr owns the mask itself and cannot be combined with this."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([8], dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            8,
            2,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            prefix_len_ptr=torch.tensor([4], dtype=torch.uint32, device=device),
        )


def test_bidirectional_ranges_packed_nvfp4_kv():
    """The variant reads a packed NVFP4 cache through its scale factors."""
    _skip_unless_fa2_jit()
    pytest.importorskip("tests.test_helpers.utils_fp4")
    from tests.test_helpers.utils_fp4 import create_nvfp4_kv, nvfp4_to_float

    device = torch.device("cuda")
    q_dtype = torch.bfloat16
    qo_len = kv_len = 96
    num_qo_heads, num_kv_heads = 8, 2
    num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE

    torch.manual_seed(7)
    kv_shape = (num_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM // 2)
    k_packed, k_sf, k_gs = create_nvfp4_kv(kv_shape, str(device))
    v_packed, v_sf, v_gs = create_nvfp4_kv(kv_shape, str(device))
    kv_cache = torch.stack([k_packed, v_packed], dim=1)
    kv_cache_sf = torch.stack([k_sf, v_sf], dim=1)

    # Dequantized reference in the query dtype.
    k_dq = (
        nvfp4_to_float(k_packed, k_sf, k_gs)
        .to(q_dtype)
        .reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len]
    )
    v_dq = (
        nvfp4_to_float(v_packed, v_sf, v_gs)
        .to(q_dtype)
        .reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len]
    )

    q = torch.randn(qo_len, num_qo_heads, HEAD_DIM, device=device, dtype=q_dtype)
    ranges = torch.full((qo_len, 2), -1, dtype=torch.int32, device=device)
    ranges[16:61, 0] = 16
    ranges[16:61, 1] = 60

    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [(kv_len - 1) % PAGE_SIZE + 1], dtype=torch.int32, device=device
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        q_data_type=q_dtype,
        kv_data_type=torch.uint8,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=q_dtype,
        kv_data_type=torch.uint8,
    )
    out = wrapper.run(q, kv_cache, ranges, kv_cache_sf=kv_cache_sf)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    ref = _dense_reference(q, k_dq, v_dq, [qo_len], [kv_len], ranges, -1, -1, sm_scale)
    causal_ref = _dense_reference(
        q,
        k_dq,
        v_dq,
        [qo_len],
        [kv_len],
        torch.full_like(ranges, -1),
        -1,
        -1,
        sm_scale,
    )
    assert not torch.allclose(ref, causal_ref, atol=1e-2, rtol=1e-2), (
        "this NVFP4 case does not exercise the bidirectional branch"
    )
    # NVFP4 storage is lossy, so the tolerance is wider than the BF16 cases.
    torch.testing.assert_close(out.float(), ref, atol=1e-1, rtol=1e-1)


def test_bidirectional_ranges_separate_output_dtype():
    """o_data_type is specialized independently of q_data_type."""
    _skip_unless_fa2_jit()
    qo_lens = [48]
    kv_lens = [48]
    spans = [[(i, (8, 40)) for i in range(8, 41)]]
    out, ref, _ = _run_case(qo_lens, kv_lens, spans, o_dtype=torch.float16)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_bidirectional_ranges_value_scale():
    """v_scale is applied to the output after the kernel, and must survive."""
    _skip_unless_fa2_jit()
    qo_lens = [48]
    kv_lens = [48]
    spans = [[(i, (8, 40)) for i in range(8, 41)]]
    scaled, ref_scaled, _ = _run_case(qo_lens, kv_lens, spans, v_scale=0.5)
    torch.testing.assert_close(scaled.float(), ref_scaled, atol=2e-2, rtol=2e-2)

    unscaled, ref_unscaled, _ = _run_case(qo_lens, kv_lens, spans)
    assert not torch.allclose(scaled.float(), unscaled.float(), atol=1e-2, rtol=1e-2), (
        "v_scale did not change the output, so this case cannot detect a drop"
    )
    torch.testing.assert_close(ref_scaled, ref_unscaled * 0.5, atol=1e-3, rtol=1e-3)


def test_bidirectional_ranges_query_key_scales():
    """Non-unit q_scale/k_scale reach the variant through sm_scale."""
    _skip_unless_fa2_jit()
    qo_lens = [48]
    kv_lens = [48]
    spans = [[(i, (8, 40)) for i in range(8, 41)]]
    scaled, ref_scaled, _ = _run_case(qo_lens, kv_lens, spans, q_scale=2.0, k_scale=3.0)
    torch.testing.assert_close(scaled.float(), ref_scaled, atol=2e-2, rtol=2e-2)

    unscaled, ref_unscaled, _ = _run_case(qo_lens, kv_lens, spans)
    assert not torch.allclose(ref_scaled, ref_unscaled, atol=1e-2, rtol=1e-2), (
        "the scales did not change the reference, so this case cannot detect a "
        "wrapper that drops them"
    )


def _plan_only_wrapper(device, **ctor_kwargs):
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    kwargs = dict(
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    kwargs.update(ctor_kwargs)
    return flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace, **kwargs
    )


def _plan_args(device):
    return (
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([8], dtype=torch.int32, device=device),
        8,
        2,
        HEAD_DIM,
        PAGE_SIZE,
    )


@pytest.mark.parametrize("method", ["plan", "workspace_size"])
@pytest.mark.parametrize(
    "override, match",
    [
        ({"q_data_type": torch.float16}, "q_data_type"),
        ({"kv_data_type": torch.float16}, "kv_data_type"),
        ({"o_data_type": torch.float16}, "o_data_type"),
        ({"head_dim_qk": 64}, "head_dim_qk"),
        ({"head_dim_vo": 64}, "head_dim_vo"),
    ],
)
def test_bidirectional_ranges_rejects_specialization_mismatch(method, override, match):
    """A configuration the constructor did not compile must not be planned."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    wrapper = _plan_only_wrapper(device)
    args = list(_plan_args(device))
    kwargs = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    # pytest hands the same dict object to every method parametrization, so
    # copy before consuming the positional override.
    override = dict(override)
    if "head_dim_qk" in override:
        args[6] = override.pop("head_dim_qk")
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        getattr(wrapper, method)(*args, **kwargs)


@pytest.mark.parametrize(
    "override, match",
    [
        ({"causal": True}, "causal=True"),
        ({"window_left": 16}, "window_left is not supported"),
        ({"logits_soft_cap": 30.0}, "logits_soft_cap"),
        ({"use_fp16_qk_reduction": True}, "use_fp16_qk_reduction"),
        ({"token_pos_in_items_len": 4}, "token_pos_in_items_len"),
    ],
)
def test_bidirectional_ranges_rejects_inherited_options(override, match):
    """Inherited options the variant makes meaningless are rejected, not ignored."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    wrapper = _plan_only_wrapper(device)
    kwargs = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        wrapper.plan(*_plan_args(device), **kwargs)


def test_bidirectional_ranges_rejects_custom_mask():
    """A caller-supplied mask would be silently dropped, so it is refused."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    wrapper = _plan_only_wrapper(device)
    mask = torch.ones(8 * 8, dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="custom_mask"):
        wrapper.plan(
            *_plan_args(device),
            custom_mask=mask,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="packed_custom_mask"):
        wrapper.plan(
            *_plan_args(device),
            packed_custom_mask=torch.ones(8, dtype=torch.uint8, device=device),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )


def test_bidirectional_ranges_rejects_rotary_specialization():
    """The variant declares no rope scalars, so a rotary mode cannot compile."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    with pytest.raises(NotImplementedError, match="pos_encoding_mode"):
        _plan_only_wrapper(device, pos_encoding_mode="ROPE_LLAMA")


def test_bidirectional_ranges_rejects_non_contiguous_and_tensor_scales():
    """Inputs that would break a captured graph or the sm_scale fold are refused."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    wrapper = _plan_only_wrapper(device)
    q = torch.randn(8, 8, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = torch.zeros(
        1, 2, PAGE_SIZE, 2, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    ranges = torch.zeros(8, 2, dtype=torch.int32, device=device)

    # A column slice has the right shape and dtype but a row stride of 4.
    non_contiguous = torch.zeros(8, 4, dtype=torch.int32, device=device)[:, :2]
    assert not non_contiguous.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.run(q, cache, non_contiguous)

    for name in ("q_scale", "k_scale", "v_scale"):
        with pytest.raises(ValueError, match=name):
            wrapper.run(q, cache, ranges, **{name: torch.ones(8, device=device)})

    with pytest.raises(ValueError, match="dtype"):
        wrapper.run(q.to(torch.float16), cache, ranges)


def test_bidirectional_ranges_generator_matches_wrapper():
    """The public generator and the wrapper build the same module."""
    _skip_unless_fa2_jit()
    from flashinfer.jit.attention.modules import (
        gen_batch_prefill_bidirectional_ranges_module,
        get_batch_prefill_bidirectional_ranges_spec,
    )

    spec = get_batch_prefill_bidirectional_ranges_spec(
        torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, HEAD_DIM, HEAD_DIM
    )
    device = torch.device("cuda")
    wrapper = _plan_only_wrapper(device)
    assert wrapper._spec == spec, (
        "the wrapper and the spec builder disagree, so the generator and the "
        "wrapper would compile different modules under the same URI"
    )

    module = gen_batch_prefill_bidirectional_ranges_module(
        torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, HEAD_DIM, HEAD_DIM
    )
    assert module.name == spec["uri"]
    # Already built by the wrapper above, so this only proves the public
    # generator reaches the same cached module instead of a second one.
    built = module.build_and_load()
    for symbol in ("plan", "paged_run", "ragged_run"):
        assert hasattr(built, symbol), f"generated module is missing {symbol}"


def test_bidirectional_ranges_cuda_graph():
    """Capture and replay, with the range buffer rewritten between replays."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    qo_len = kv_len = 64
    num_qo_heads, num_kv_heads = 8, 2
    num_pages = kv_len // PAGE_SIZE

    torch.manual_seed(13)
    q = torch.randn(qo_len, num_qo_heads, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    kv_cache = torch.zeros(
        num_pages, 2, PAGE_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=dtype
    )
    kv_cache[:, 0] = k.reshape(num_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM)
    kv_cache[:, 1] = v.reshape(num_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM)

    qo_indptr_buf = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr_buf = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices_buf = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_len_buf = torch.tensor([PAGE_SIZE], dtype=torch.int32, device=device)
    ranges_buf = torch.full((qo_len, 2), -1, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr_buf,
        paged_kv_indptr_buf=kv_indptr_buf,
        paged_kv_indices_buf=kv_indices_buf,
        paged_kv_last_page_len_buf=last_page_len_buf,
        q_data_type=dtype,
        kv_data_type=dtype,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    wrapper.plan(
        qo_indptr_buf,
        kv_indptr_buf,
        kv_indices_buf,
        last_page_len_buf,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out_buf = torch.empty(qo_len, num_qo_heads, HEAD_DIM, device=device, dtype=dtype)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        wrapper.run(q, kv_cache, ranges_buf, out=out_buf)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, kv_cache, ranges_buf, out=out_buf)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    for start, end in ((-1, -1), (8, 40), (0, 63)):
        ranges_buf[:, 0] = -1
        ranges_buf[:, 1] = -1
        if start >= 0:
            ranges_buf[start : end + 1, 0] = start
            ranges_buf[start : end + 1, 1] = end
        graph.replay()
        torch.cuda.synchronize()
        ref = _dense_reference(
            q, k, v, [qo_len], [kv_len], ranges_buf, -1, -1, sm_scale
        )
        torch.testing.assert_close(out_buf.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("method", ["plan", "workspace_size"])
def test_bidirectional_ranges_dtypes_default_to_constructor(method):
    """An omitted dtype means "what this wrapper was built for", not float16."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    wrapper = _plan_only_wrapper(
        device, q_data_type=torch.float16, kv_data_type=torch.float16
    )
    # No dtype arguments at all: the float16 specialization has to be inherited.
    getattr(wrapper, method)(*_plan_args(device))
    # Restating it is still accepted, and a different one is still refused.
    getattr(wrapper, method)(
        *_plan_args(device), q_data_type=torch.float16, kv_data_type=torch.float16
    )
    with pytest.raises(ValueError, match="q_data_type"):
        getattr(wrapper, method)(*_plan_args(device), q_data_type=torch.bfloat16)


@pytest.mark.parametrize("method", ["plan", "workspace_size"])
def test_bidirectional_ranges_output_dtype_defaults_to_constructor(method):
    """A separate output dtype does not have to be repeated on every plan."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    wrapper = _plan_only_wrapper(device, o_data_type=torch.float16)
    getattr(wrapper, method)(
        *_plan_args(device), q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16
    )
    # The default is the constructor's float16, so naming the query dtype here
    # is a real mismatch rather than a harmless restatement.
    with pytest.raises(ValueError, match="o_data_type"):
        getattr(wrapper, method)(*_plan_args(device), o_data_type=torch.bfloat16)


def _causal_definition_name(num_qo_heads, num_kv_heads, head_dim, page_size):
    """The name a plain causal prefill at these axes registers under."""
    from flashinfer.trace.templates.attention import gqa_paged_prefill_trace

    return gqa_paged_prefill_trace.definition_name(
        {
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
        }
    )


def test_specialized_run_never_reaches_the_parent_class_attribute():
    """Nothing installed on the parent's ``run`` can see this wrapper.

    Both the trace auto-dump and Trace Apply work by owning that attribute, so
    a call that never reads it cannot be traced as a causal prefill nor
    substituted for one. Replacing the attribute with a tripwire is the
    strongest statement of that: the call has to keep working.
    """
    _skip_unless_fa2_jit()
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    tripped = []

    def _tripwire(self, *args, **kwargs):
        tripped.append(True)
        raise AssertionError(
            "the specialized wrapper went through the parent's run attribute"
        )

    original = BatchPrefillWithPagedKVCacheWrapper.run
    BatchPrefillWithPagedKVCacheWrapper.run = _tripwire
    try:
        out, ref, _ = _run_case([48], [48], [[(i, (4, 30)) for i in range(4, 31)]])
    finally:
        BatchPrefillWithPagedKVCacheWrapper.run = original

    assert not tripped
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_trace_apply_solution_for_causal_does_not_capture_this_wrapper():
    """A causal solution at the same axes must not be substituted here."""
    _skip_unless_fa2_jit()
    from flashinfer import trace_apply
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    name = _causal_definition_name(8, 2, HEAD_DIM, PAGE_SIZE)
    fired = []

    def _solution(*args, **kwargs):
        fired.append(name)
        raise AssertionError("a causal solution was applied to the ranges wrapper")

    spans = [[(i, (4, 30)) for i in range(4, 31)]]
    wrapped = trace_apply.enable_apply({name: _solution})
    try:
        # Without this the test could pass because nothing was installed at
        # all. The stronger statement is
        # test_specialized_run_never_reaches_the_parent_class_attribute: this
        # one only says a live installation did not fire here.
        assert wrapped > 0, "Trace Apply installed nothing for this definition"
        assert getattr(
            BatchPrefillWithPagedKVCacheWrapper.run, "_trace_apply", False
        ), "the parent's run was not patched, so this proves nothing"
        out, ref, _ = _run_case([48], [48], spans)
    finally:
        trace_apply.disable_apply()

    assert not fired
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_trace_apply_does_patch_the_parent_while_enabled():
    """The exclusion is this wrapper's, not a hole in Trace Apply.

    Without this, the test above would pass just as well if the mechanism had
    installed nothing at all. What it asserts is the precondition: with a
    solution registered for the causal definition name, the parent's ``run``
    attribute really is replaced for the duration -- and that attribute is
    exactly the one the specialized wrapper no longer reads.
    """
    _skip_unless_fa2_jit()
    from flashinfer import trace_apply
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    name = _causal_definition_name(8, 2, HEAD_DIM, PAGE_SIZE)
    before = BatchPrefillWithPagedKVCacheWrapper.run

    wrapped = trace_apply.enable_apply({name: lambda *a, **k: None})
    try:
        during = BatchPrefillWithPagedKVCacheWrapper.run
        assert wrapped > 0, "Trace Apply wrapped nothing for this definition name"
        assert during is not before, "the parent's run attribute was not patched"
        assert getattr(during, "_trace_apply", False)
    finally:
        trace_apply.disable_apply()

    assert BatchPrefillWithPagedKVCacheWrapper.run is before


def test_non_sentinel_ranges_still_take_the_specialized_kernel():
    """Excluding the wrapper must not quietly turn it into a causal prefill."""
    _skip_unless_fa2_jit()
    spans = [[(i, (4, 40)) for i in range(4, 41)]]
    out, ref, _ = _run_case([48], [48], spans)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    causal_only, causal_ref, _ = _run_case([48], [48], [[]])
    assert not torch.allclose(out.float(), causal_only.float(), atol=1e-2, rtol=1e-2), (
        "the ranges had no effect: the call fell back to a plain causal prefill"
    )


# PrefillPlanInfo::ToVector order; split_kv is the last entry and cta_tile_q
# the fourth. Reading it is how these tests show the split path actually ran
# rather than assuming a long KV implies it.
_PLAN_INFO_CTA_TILE_Q = 3
_PLAN_INFO_SPLIT_KV = 14


def _split_case(
    qo_lens,
    kv_lens,
    ranges_per_req,
    *,
    fixed_split_size=None,
    disable_split_kv=False,
    max_token_per_sequence=None,
    max_sequence_kv=None,
    num_qo_heads=8,
    num_kv_heads=2,
    seed=11,
):
    """Plan with the split knobs, return (out, ref, plan_info)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(seed)

    total_q, total_kv = sum(qo_lens), sum(kv_lens)
    q = torch.randn(total_q, num_qo_heads, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(total_kv, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(total_kv, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)

    ranges = torch.full((total_q, 2), -1, dtype=torch.int32, device=device)
    off = 0
    for qo_len, spans in zip(qo_lens, ranges_per_req, strict=True):
        for q_local, (start, end) in spans:
            ranges[off + q_local, 0] = start
            ranges[off + q_local, 1] = end
        off += qo_len

    kv_cache, kv_indptr, kv_indices, last_page_len = _build_paged(
        k, v, kv_lens, device, dtype
    )
    qo_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        q_data_type=dtype,
        kv_data_type=dtype,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=dtype,
        kv_data_type=dtype,
        max_token_per_sequence=max_token_per_sequence,
        max_sequence_kv=max_sequence_kv,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )
    out, lse = wrapper.run(q, kv_cache, ranges, return_lse=True)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    ref = _dense_reference(q, k, v, qo_lens, kv_lens, ranges, -1, -1, sm_scale)
    return out, lse, ref, list(wrapper._plan_info), wrapper


# 17 and 33 straddle the CTA_TILE_Q choices, so neither is a multiple of the
# tile the scheduler picks.
@pytest.mark.parametrize("qo_len", [17, 33])
# fixed_split_size is a page count, not a token count: the scheduler's kv
# lengths come from the paged indptr, so a 256-token request is 16 pages and a
# split size of 16 would leave it in one chunk.
@pytest.mark.parametrize("fixed_split_size", [2, 4])
def test_fixed_split_size_takes_the_split_path(qo_len, fixed_split_size):
    """``fixed_split_size`` has to reach the scheduler and change the plan."""
    _skip_unless_fa2_jit()
    kv_len = 256
    spans = [[(i, (8, 200)) for i in range(min(qo_len, 8), qo_len)]]
    out, lse, ref, plan_info, _ = _split_case(
        [qo_len], [kv_len], spans, fixed_split_size=fixed_split_size
    )
    assert plan_info[_PLAN_INFO_SPLIT_KV] == 1, (
        f"fixed_split_size={fixed_split_size} did not produce a split plan: "
        f"plan_info={plan_info}"
    )
    assert plan_info[_PLAN_INFO_CTA_TILE_Q] > 0
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(lse.float()).all()


def test_disable_split_kv_keeps_the_unsplit_plan():
    """The control: the same shape refuses to split when told to."""
    _skip_unless_fa2_jit()
    qo_len, kv_len = 33, 256
    spans = [[(i, (8, 200)) for i in range(8, qo_len)]]
    _, _, _, split_info, _ = _split_case([qo_len], [kv_len], spans, fixed_split_size=4)
    out, _, ref, unsplit_info, _ = _split_case(
        [qo_len], [kv_len], spans, fixed_split_size=4, disable_split_kv=True
    )
    assert split_info[_PLAN_INFO_SPLIT_KV] == 1
    assert unsplit_info[_PLAN_INFO_SPLIT_KV] == 0, (
        f"disable_split_kv was ignored: plan_info={unsplit_info}"
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_split_and_unsplit_agree():
    """Whatever the split does, the answer may not depend on it."""
    _skip_unless_fa2_jit()
    qo_len, kv_len = 33, 256
    spans = [[(i, (8, 200)) for i in range(8, qo_len)]]
    split_out, split_lse, ref, split_info, _ = _split_case(
        [qo_len], [kv_len], spans, fixed_split_size=4
    )
    plain_out, plain_lse, _, plain_info, _ = _split_case(
        [qo_len], [kv_len], spans, fixed_split_size=4, disable_split_kv=True
    )
    assert split_info[_PLAN_INFO_SPLIT_KV] == 1
    assert plain_info[_PLAN_INFO_SPLIT_KV] == 0
    torch.testing.assert_close(split_out.float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        split_out.float(), plain_out.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        split_lse.float(), plain_lse.float(), atol=2e-2, rtol=2e-2
    )


def test_plan_forwards_max_token_per_sequence():
    """The knob has to reach the parent, not just leave the answer intact.

    The parent stores it as ``_max_q_len`` and otherwise derives that from the
    qo_indptr, so reading it back is what separates a forwarded argument from a
    dropped one: for these shapes the derived value would be 33, not 64.
    """
    _skip_unless_fa2_jit()
    qo_len, kv_len = 33, 256
    spans = [[(i, (8, 200)) for i in range(8, qo_len)]]

    _, _, _, _, derived = _split_case([qo_len], [kv_len], spans)
    assert derived._max_q_len == qo_len

    out, lse, ref, plan_info, wrapper = _split_case(
        [qo_len], [kv_len], spans, max_token_per_sequence=64
    )
    assert wrapper._max_q_len == 64, (
        "max_token_per_sequence never reached the parent: _max_q_len is "
        f"{wrapper._max_q_len}, the value derived from qo_indptr"
    )
    assert len(plan_info) == 15
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    assert torch.isfinite(lse.float()).all()


def test_max_sequence_kv_is_refused_by_both_planning_entry_points():
    """Refused here rather than failing inside the parent.

    ``BatchPrefillWithPagedKVCacheWrapper.plan`` skips binding the host-side kv
    arrays when this is set and then reads them, so the call dies with an
    ``UnboundLocalError`` several frames away. Both planning entry points of
    this wrapper close it with a message that names the argument.
    """
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    qo_len = kv_len = 32
    num_qo_heads, num_kv_heads = 8, 2

    k = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    _, kv_indptr, kv_indices, last_page_len = _build_paged(
        k, v, [kv_len], device, dtype
    )
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)

    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        q_data_type=dtype,
        kv_data_type=dtype,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    args = (
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
    )
    kwargs = dict(q_data_type=dtype, kv_data_type=dtype)

    for entry in (wrapper.plan, wrapper.workspace_size):
        with pytest.raises(NotImplementedError, match="max_sequence_kv"):
            entry(*args, max_sequence_kv=512, **kwargs)

    # Both still work without it, so the refusal is the argument's and not the
    # entry point's.
    wrapper.workspace_size(*args, **kwargs)
    wrapper.plan(*args, **kwargs)


def test_runtime_shape_mismatch_closes_before_dispatch():
    """The module's compile-time dims and plan's head counts are contracts."""
    _skip_unless_fa2_jit()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    qo_len = kv_len = 32
    num_qo_heads, num_kv_heads = 8, 2

    k = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    kv_cache, kv_indptr, kv_indices, last_page_len = _build_paged(
        k, v, [kv_len], device, dtype
    )
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    ranges = torch.full((qo_len, 2), -1, dtype=torch.int32, device=device)

    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        q_data_type=dtype,
        kv_data_type=dtype,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    good = torch.randn(qo_len, num_qo_heads, HEAD_DIM, device=device, dtype=dtype)
    wrapper.run(good, kv_cache, ranges)

    wrong_heads = torch.randn(qo_len, 4, HEAD_DIM, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="query heads"):
        wrapper.run(wrong_heads, kv_cache, ranges)

    wrong_width = torch.randn(qo_len, num_qo_heads, 64, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="head_dim_qk"):
        wrapper.run(wrong_width, kv_cache, ranges)

    num_pages = kv_cache.shape[0]
    narrow_cache = torch.randn(
        num_pages, 2, PAGE_SIZE, num_kv_heads, 64, device=device, dtype=dtype
    )
    with pytest.raises(ValueError, match="width"):
        wrapper.run(good, narrow_cache, ranges)

    extra_cache = torch.randn(
        num_pages, 2, PAGE_SIZE, 4, HEAD_DIM, device=device, dtype=dtype
    )
    with pytest.raises(ValueError, match="kv heads"):
        wrapper.run(good, extra_cache, ranges)


def _planned_wrapper_and_inputs(qo_len=48, kv_len=48, num_qo_heads=8, num_kv_heads=2):
    """A planned wrapper plus q, the paged cache, the ranges and a reference."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(17)

    q = torch.randn(qo_len, num_qo_heads, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(kv_len, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)

    ranges = torch.full((qo_len, 2), -1, dtype=torch.int32, device=device)
    for row in range(4, 31):
        ranges[row, 0] = 4
        ranges[row, 1] = 30

    kv_cache, kv_indptr, kv_indices, last_page_len = _build_paged(
        k, v, [kv_len], device, dtype
    )
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)

    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
        workspace,
        kv_layout="NHD",
        q_data_type=dtype,
        kv_data_type=dtype,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        PAGE_SIZE,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    ref = _dense_reference(q, k, v, [qo_len], [kv_len], ranges, -1, -1, sm_scale)
    return wrapper, q, kv_cache, ranges, ref


def test_run_return_lse_applies_the_override_checks():
    """``run_return_lse`` has to reach this class's ``run``.

    ``functools.partialmethod`` captures the function it is given, so an
    inherited binding sends the call into the parent body and skips every check
    the override exists to make. Both halves are asserted: the ranges are
    honoured, and a range tensor the override refuses is still refused.
    """
    _skip_unless_fa2_jit()
    wrapper, q, kv_cache, ranges, ref = _planned_wrapper_and_inputs()

    out, lse = wrapper.run_return_lse(q, kv_cache, ranges)
    assert out.shape == q.shape
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    assert lse.shape[0] == q.shape[0]
    assert torch.isfinite(lse.float()).all()

    padded = torch.full((ranges.size(0), 4), -1, dtype=torch.int32, device=q.device)
    padded[:, :2] = ranges
    non_contiguous = padded[:, :2]
    assert not non_contiguous.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        wrapper.run_return_lse(q, kv_cache, non_contiguous)

    with pytest.raises(ValueError, match="scalar"):
        wrapper.run_return_lse(
            q, kv_cache, ranges, q_scale=torch.ones(1, device=q.device)
        )


def test_deprecated_forward_entry_points_take_the_ranges():
    """``forward`` and ``forward_return_lse`` carry this wrapper's contract.

    The parent's versions have no parameter for the ranges and set plan state
    this wrapper refuses, so reaching them would assign ``_causal`` and friends
    and only then fail inside ``prepare_jit_additional_args``.
    """
    _skip_unless_fa2_jit()
    wrapper, q, kv_cache, ranges, ref = _planned_wrapper_and_inputs()

    out = wrapper.forward(q, kv_cache, ranges)
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    pair = wrapper.forward_return_lse(q, kv_cache, ranges)
    assert isinstance(pair, tuple) and len(pair) == 2
    lse_out, lse = pair
    torch.testing.assert_close(lse_out.float(), ref, atol=2e-2, rtol=2e-2)
    assert lse.shape[0] == q.shape[0]
    assert torch.isfinite(lse.float()).all()

    # The plan state the parent's forward would have written is untouched.
    assert getattr(wrapper, "_causal", False) is False

    with pytest.raises(ValueError, match="contiguous"):
        padded = torch.full((ranges.size(0), 4), -1, dtype=torch.int32, device=q.device)
        padded[:, :2] = ranges
        wrapper.forward(q, kv_cache, padded[:, :2])


_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def _spec_kwargs(**overrides):
    base = dict(
        dtype_q=torch.bfloat16,
        dtype_kv=torch.bfloat16,
        dtype_o=torch.bfloat16,
        dtype_idx=torch.int32,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize("fp8", _FP8_DTYPES, ids=["e4m3fn", "e5m2"])
@pytest.mark.parametrize("field", ["dtype_q", "dtype_o"])
def test_fp8_is_refused_by_spec_and_generator(fp8, field):
    """fa2 has no fp8 tensor-core path and no fp8 output path.

    The public generators assert on both before rendering; this variant reaches
    ``gen_customize_batch_prefill_module`` directly, so the refusal lives in the
    shared spec that the generator and the wrapper both pass through. Reaching
    nvcc instead fails on ``static_assert(sizeof(DTypeQ) == 2)``, which names
    nothing the caller passed.
    """
    from flashinfer.jit.attention.modules import (
        gen_batch_prefill_bidirectional_ranges_module,
        get_batch_prefill_bidirectional_ranges_spec,
    )

    expected = "fp8 tensor core" if field == "dtype_q" else "FP8 output"
    with pytest.raises(ValueError, match=expected):
        get_batch_prefill_bidirectional_ranges_spec(**_spec_kwargs(**{field: fp8}))
    with pytest.raises(ValueError, match=expected):
        gen_batch_prefill_bidirectional_ranges_module(**_spec_kwargs(**{field: fp8}))


@pytest.mark.parametrize("fp8", _FP8_DTYPES, ids=["e4m3fn", "e5m2"])
@pytest.mark.parametrize("field", ["q_data_type", "o_data_type"])
def test_fp8_is_refused_by_the_wrapper_constructor(fp8, field):
    """The constructor shares the spec's refusal rather than repeating it."""
    workspace = torch.empty(1024, dtype=torch.uint8)
    expected = "fp8 tensor core" if field == "q_data_type" else "FP8 output"
    kwargs = dict(
        kv_layout="NHD",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
    )
    kwargs[field] = fp8
    with pytest.raises(ValueError, match=expected):
        flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(workspace, **kwargs)


@pytest.mark.parametrize("fp8", _FP8_DTYPES, ids=["e4m3fn", "e5m2"])
def test_kv_only_fp8_is_not_refused(fp8):
    """The positive control: a KV-only quantization is deliberately allowed.

    ``fp8_enabled`` in the public generators is a property of the query dtype;
    quantizing only the cache does not select the fp8 tensor-core template, so
    the refusal above must not spread to it.
    """
    from flashinfer.jit.attention.modules import (
        get_batch_prefill_bidirectional_ranges_spec,
    )

    spec = get_batch_prefill_bidirectional_ranges_spec(**_spec_kwargs(dtype_kv=fp8))
    assert spec["backend"] == "fa2"
    assert spec["dtype_kv"] == fp8
    assert spec["packed_fp4_kv"] is False
