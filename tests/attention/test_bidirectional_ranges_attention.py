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
        if causal_window_left > 0:
            causal = causal & (dist < causal_window_left)

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
