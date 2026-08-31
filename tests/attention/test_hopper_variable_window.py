"""FA3 Hopper VariableWindow tests for batch prefill (ragged and paged).

Per-token inclusive KV start/end bounds, compiled as a sibling of sliding
window. Skip unless SM90A; force backend=fa3. Do not import PrimTS tests.
"""

from typing import List, Optional, Sequence, Tuple

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported

E4M3_MAX = 448.0
DT = torch.float8_e4m3fn


def _require_sm90():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")


def _ws():
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def gemma_variable_window_bounds(
    seq_len: int,
    left_window: int,
    right_window: int,
    image_segs: Sequence[Tuple[int, int]],
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.arange(seq_len, device=device)
    starts = torch.clamp(q - left_window, min=0)
    ends = q.clone()
    for seg_start, seg_end in image_segs:
        in_seg = (q >= seg_start) & (q < seg_end)
        ends = torch.where(in_seg, torch.clamp(q + right_window, max=seg_end - 1), ends)
    return starts.to(torch.int32), ends.to(torch.int32)


def causal_swa_bounds(
    qo_len: int, kv_len: int, window_left: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.arange(qo_len, device=device)
    q_pos = q + kv_len - qo_len
    starts = torch.clamp(q_pos - window_left, min=0)
    ends = q_pos
    return starts.to(torch.int32), ends.to(torch.int32)


def variable_window_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Inclusive [start, end] mask, GQA by repeating KV heads. FP32 softmax."""
    hq, hkv = q.shape[1], k.shape[1]
    if hq % hkv != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * sm_scale
    kv_idx = torch.arange(k.shape[0], device=q.device).view(1, -1)
    keep = (kv_idx >= starts.view(-1, 1)) & (kv_idx <= ends.view(-1, 1))
    scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, v.float())
    return out.to(q.dtype)


def ragged_variable_window_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    batch = qo_indptr.numel() - 1
    for b in range(batch):
        qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
        ks, ke = int(kv_indptr[b]), int(kv_indptr[b + 1])
        outs.append(
            variable_window_attention_ref(
                q[qs:qe], k[ks:ke], v[ks:ke], starts[qs:qe], ends[qs:qe]
            )
        )
    return torch.cat(outs, dim=0)


def _paged_gather(
    cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
    batch: int,
) -> torch.Tensor:
    chunks = []
    for b in range(batch):
        ps, pe = int(kv_indptr[b]), int(kv_indptr[b + 1])
        pages = kv_indices[ps:pe]
        last = int(kv_last_page_len[b])
        seq = []
        for i, page in enumerate(pages.tolist()):
            n = last if i == pages.numel() - 1 else page_size
            seq.append(cache[page, :n])
        chunks.append(torch.cat(seq, dim=0))
    return torch.cat(chunks, dim=0)


def per_head_symmetric_quant(
    x: torch.Tensor, quant_dtype: torch.dtype = DT
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_max = x.abs().amax(dim=(0, 2)).to(torch.float32)
    s = torch.clamp(x_max / E4M3_MAX, min=1e-6)
    q = torch.clamp(x / s.view(1, -1, 1), min=-E4M3_MAX, max=E4M3_MAX).to(quant_dtype)
    return q, s


def _run_ragged_vw(
    q, k, v, qo_indptr, kv_indptr, starts, ends, num_qo, num_kv, head_dim
):
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _ws(), "NHD", backend="fa3"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo,
        num_kv,
        head_dim,
        causal=False,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=torch.half if q.dtype in (DT, torch.float8_e5m2) else q.dtype,
    )
    return wrapper.run(q, k, v)


@pytest.mark.parametrize("seq_len", [128, 257, 512])
@pytest.mark.parametrize("window_left", [64, 128])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_uniform_window_matches_swa(seq_len, window_left, head_dim):
    _require_sm90()
    torch.manual_seed(0)
    hq, hkv = 32, 8
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = qo_indptr.clone()
    q = torch.randn(seq_len, hq, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, hkv, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, hkv, head_dim, dtype=torch.half, device="cuda")
    starts, ends = causal_swa_bounds(seq_len, seq_len, window_left, q.device)

    swa = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_ws(), "NHD", backend="fa3")
    swa.plan(
        qo_indptr, kv_indptr, hq, hkv, head_dim, causal=True, window_left=window_left
    )
    o_swa = swa.run(q, k, v)

    o_vw = _run_ragged_vw(
        q, k, v, qo_indptr, kv_indptr, starts, ends, hq, hkv, head_dim
    )
    torch.testing.assert_close(o_vw, o_swa, rtol=1e-3, atol=1e-3)


def test_gemma_text_image_pattern():
    _require_sm90()
    torch.manual_seed(1)
    seq_len, left, right = 2560, 128, 128
    hq, hkv, d = 32, 8, 128
    starts, ends = gemma_variable_window_bounds(
        seq_len, left, right, ((256, 1280), (1536, 2560)), "cuda"
    )
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = qo_indptr.clone()
    q = torch.randn(seq_len, hq, d, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, hkv, d, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, hkv, d, dtype=torch.half, device="cuda")
    o = _run_ragged_vw(q, k, v, qo_indptr, kv_indptr, starts, ends, hq, hkv, d)
    o_ref = variable_window_attention_ref(q, k, v, starts, ends)
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)


def test_chunked_prefill_absolute_kv_indices():
    _require_sm90()
    torch.manual_seed(2)
    qo_len, kv_len, window_left = 128, 512, 64
    hq, hkv, d = 32, 8, 128
    starts, ends = causal_swa_bounds(qo_len, kv_len, window_left, "cuda")
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
    q = torch.randn(qo_len, hq, d, dtype=torch.half, device="cuda")
    k = torch.randn(kv_len, hkv, d, dtype=torch.half, device="cuda")
    v = torch.randn(kv_len, hkv, d, dtype=torch.half, device="cuda")
    o = _run_ragged_vw(q, k, v, qo_indptr, kv_indptr, starts, ends, hq, hkv, d)
    o_ref = variable_window_attention_ref(q, k, v, starts, ends)
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)


def test_ragged_batch_size_2():
    _require_sm90()
    torch.manual_seed(3)
    lens = [96, 160]
    hq, hkv, d = 32, 8, 128
    qo_indptr = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32, device="cuda")
    kv_indptr = qo_indptr.clone()
    nnz = int(qo_indptr[-1])
    q = torch.randn(nnz, hq, d, dtype=torch.half, device="cuda")
    k = torch.randn(nnz, hkv, d, dtype=torch.half, device="cuda")
    v = torch.randn(nnz, hkv, d, dtype=torch.half, device="cuda")
    start_chunks, end_chunks = [], []
    for seq in lens:
        s, e = causal_swa_bounds(seq, seq, 48, "cuda")
        start_chunks.append(s)
        end_chunks.append(e)
    starts = torch.cat(start_chunks)
    ends = torch.cat(end_chunks)
    o = _run_ragged_vw(q, k, v, qo_indptr, kv_indptr, starts, ends, hq, hkv, d)
    o_ref = ragged_variable_window_ref(q, k, v, qo_indptr, kv_indptr, starts, ends)
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("page_size", [16, 64])
def test_paged_variable_window(page_size):
    _require_sm90()
    torch.manual_seed(4)
    seq_len, hq, hkv, d = 257, 32, 8, 128
    num_pages = (seq_len + page_size - 1) // page_size
    last = seq_len % page_size or page_size
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    kv_last = torch.tensor([last], dtype=torch.int32, device="cuda")
    q = torch.randn(seq_len, hq, d, dtype=torch.half, device="cuda")
    pk = torch.randn(num_pages, page_size, hkv, d, dtype=torch.half, device="cuda")
    pv = torch.randn(num_pages, page_size, hkv, d, dtype=torch.half, device="cuda")
    starts, ends = causal_swa_bounds(seq_len, seq_len, 128, "cuda")

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _ws(), "NHD", backend="fa3"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        hq,
        hkv,
        d,
        page_size,
        causal=False,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
    )
    o = wrapper.run(q, (pk, pv))
    k = _paged_gather(pk, kv_indptr, kv_indices, kv_last, page_size, 1)
    v = _paged_gather(pv, kv_indptr, kv_indices, kv_last, page_size, 1)
    o_ref = variable_window_attention_ref(q, k, v, starts, ends)
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)


def test_non_monotonic_starts_inside_q_tile():
    _require_sm90()
    torch.manual_seed(5)
    seq_len, hq, hkv, d = 128, 8, 2, 128
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = qo_indptr.clone()
    q = torch.randn(seq_len, hq, d, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, hkv, d, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, hkv, d, dtype=torch.half, device="cuda")
    # Most rows start at 64; one later row in the same Q tile jumps to 0 so
    # tile skip cannot use the first row's start. Clamp ends so start <= end
    # (rows 0-63 would otherwise be empty windows and softmax to NaN).
    starts = torch.full((seq_len,), 64, dtype=torch.int32, device="cuda")
    starts[80] = 0
    ends = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    ends = torch.maximum(ends, starts)
    o = _run_ragged_vw(q, k, v, qo_indptr, kv_indptr, starts, ends, hq, hkv, d)
    o_ref = variable_window_attention_ref(q, k, v, starts, ends)
    torch.testing.assert_close(o.float(), o_ref.float(), rtol=2e-2, atol=2e-2)


def _fp8_ragged_vs_fp16(seq_len: int):
    hq, hkv, d = 32, 8, 128
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = qo_indptr.clone()
    q = torch.randn(seq_len, hq, d, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, hkv, d, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, hkv, d, dtype=torch.half, device="cuda")
    starts, ends = causal_swa_bounds(seq_len, seq_len, 128, "cuda")
    o_ref = _run_ragged_vw(q, k, v, qo_indptr, kv_indptr, starts, ends, hq, hkv, d)

    q8, sq = per_head_symmetric_quant(q)
    k8, sk = per_head_symmetric_quant(k)
    v8, sv = per_head_symmetric_quant(v)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _ws(), "NHD", backend="fa3"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        hq,
        hkv,
        d,
        causal=False,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
        q_data_type=DT,
        kv_data_type=DT,
        o_data_type=torch.half,
    )
    o_fp8 = wrapper.run(q8, k8, v8, sq, sk, sv)
    torch.cuda.synchronize()
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2).item()
    assert mse < 1.0, f"MSE too high: {mse}"


@pytest.mark.timeout(300)
@pytest.mark.parametrize("seq_len", [257, 512])
def test_fp8_ragged_variable_window(seq_len):
    _require_sm90()
    torch.manual_seed(6)
    _fp8_ragged_vs_fp16(seq_len)


@pytest.mark.timeout(300)
def test_fp8_paged_variable_window():
    _require_sm90()
    torch.manual_seed(7)
    seq_len, page_size, hq, hkv, d = 257, 32, 32, 8, 128
    num_pages = (seq_len + page_size - 1) // page_size
    last = seq_len % page_size or page_size
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    kv_last = torch.tensor([last], dtype=torch.int32, device="cuda")
    q = torch.randn(seq_len, hq, d, dtype=torch.half, device="cuda")
    pk = torch.randn(num_pages, page_size, hkv, d, dtype=torch.half, device="cuda")
    pv = torch.randn(num_pages, page_size, hkv, d, dtype=torch.half, device="cuda")
    starts, ends = causal_swa_bounds(seq_len, seq_len, 128, "cuda")

    ref_w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_ws(), "NHD", backend="fa3")
    ref_w.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        hq,
        hkv,
        d,
        page_size,
        causal=False,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
    )
    o_ref = ref_w.run(q, (pk, pv))

    q8, sq = per_head_symmetric_quant(q)
    k8, sk = per_head_symmetric_quant(pk.view(-1, hkv, d))
    v8, sv = per_head_symmetric_quant(pv.view(-1, hkv, d))
    k8 = k8.view(num_pages, page_size, hkv, d)
    v8 = v8.view(num_pages, page_size, hkv, d)
    fp8_w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_ws(), "NHD", backend="fa3")
    fp8_w.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        hq,
        hkv,
        d,
        page_size,
        causal=False,
        variable_window_token_starts=starts,
        variable_window_token_ends=ends,
        q_data_type=DT,
        kv_data_type=DT,
        o_data_type=torch.half,
    )
    o_fp8 = fp8_w.run(q8, (k8, v8), sq, sk, sv)
    torch.cuda.synchronize()
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2).item()
    assert mse < 1.0, f"MSE too high: {mse}"


def test_rejects_fa2_backend():
    _require_sm90()
    seq_len = 32
    starts, ends = causal_swa_bounds(seq_len, seq_len, 8, "cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _ws(), "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="fa3"):
        wrapper.plan(
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            8,
            2,
            128,
            causal=False,
            variable_window_token_starts=starts,
            variable_window_token_ends=ends,
        )


def test_rejects_missing_ends():
    _require_sm90()
    seq_len = 32
    starts, _ = causal_swa_bounds(seq_len, seq_len, 8, "cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _ws(), "NHD", backend="fa3"
    )
    with pytest.raises(ValueError, match="both"):
        wrapper.plan(
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            8,
            2,
            128,
            causal=False,
            variable_window_token_starts=starts,
        )


def test_rejects_window_left_combo():
    _require_sm90()
    seq_len = 32
    starts, ends = causal_swa_bounds(seq_len, seq_len, 8, "cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _ws(), "NHD", backend="fa3"
    )
    with pytest.raises(ValueError, match="window_left"):
        wrapper.plan(
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            8,
            2,
            128,
            causal=False,
            window_left=8,
            variable_window_token_starts=starts,
            variable_window_token_ends=ends,
        )


def test_rejects_causal_true_combo():
    _require_sm90()
    seq_len = 32
    starts, ends = causal_swa_bounds(seq_len, seq_len, 8, "cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _ws(), "NHD", backend="fa3"
    )
    with pytest.raises(ValueError, match="causal"):
        wrapper.plan(
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            8,
            2,
            128,
            causal=True,
            variable_window_token_starts=starts,
            variable_window_token_ends=ends,
        )


def test_rejects_custom_mask_combo():
    _require_sm90()
    seq_len = 32
    starts, ends = causal_swa_bounds(seq_len, seq_len, 8, "cuda")
    mask = torch.ones(seq_len * seq_len, dtype=torch.bool, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _ws(), "NHD", backend="fa3"
    )
    with pytest.raises(ValueError, match="custom_mask"):
        wrapper.plan(
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            8,
            2,
            128,
            causal=False,
            custom_mask=mask,
            variable_window_token_starts=starts,
            variable_window_token_ends=ends,
        )
