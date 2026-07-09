"""
Copyright (c) 2025 by FlashInfer team.

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
"""Exhaustive correctness tests for the FlashInfer GVR Top-K kernel API.

Requires a Blackwell GPU (sm_100 / B200) and ``nvidia-cutlass-dsl``.

Test matrix
-----------
test_gvr_topk_decode          — 256 knob combinations (dtype × K × N × batch ×
                                 use_256bit_load × num_threads × warp_parallel ×
                                 cluster_size)
test_gvr_topk_decode_next_n   — next_n=2 path (temporal stride)
test_gvr_topk_decode_values   — return_output_values=True verification
test_gvr_topk_sort_prepare    — seqlen_sorted=True dispatch
test_gvr_topk_compress_ratio  — compress_ratio=4 (DSv4 mode)
test_gvr_topk_lb_prepare      — LB classifier smoke test
test_gvr_topk_lb_decode       — LB hybrid kernel correctness
test_gvr_topk_lb_roundtrip    — prepare + decode full roundtrip
test_can_use_gvr_topk         — hardware-gate API
test_gvr_topk_preallocated    — pre-allocated output buffers
test_gvr_topk_bad_device      — RuntimeError on non-Blackwell
test_gvr_topk_bad_dtype       — TypeError on unsupported dtype
"""

import pytest
import torch

try:
    import flashinfer

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE, reason="flashinfer not installed"
)

# ---------------------------------------------------------------------------
# Skip if not Blackwell
# ---------------------------------------------------------------------------

_IS_BLACKWELL = (
    torch.cuda.is_available()
    and _FLASHINFER_AVAILABLE
    and flashinfer.can_use_gvr_topk(torch.device("cuda"))
)

requires_blackwell = pytest.mark.skipif(
    not _IS_BLACKWELL,
    reason="GVR Top-K requires a Blackwell GPU (sm_100+) and nvidia-cutlass-dsl",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    num_rows: int,
    N: int,
    top_k: int,
    dtype: torch.dtype,
    seed: int,
    next_n: int = 1,
    compress_ratio: int = 1,
):
    """Build (logits, pre_idx, seq_lens) for a multi-row test.

    Shapes:
      logits  : [num_rows, N]                 — compressed-token-index space
      pre_idx : [num_rows // next_n, top_k]   — argmax in slot 0
      seq_lens: [num_rows // next_n]          — uncompressed-token space

    ``seq_lens = N * compress_ratio`` makes the kernel's N_eff match the
    reference torch.topk effective length for next_n in {1, 2}.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logits_f32 = torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0
    logits = logits_f32.to(dtype)
    num_groups = num_rows // next_n

    effective_len = N - next_n + 1
    argmax_idx = logits[::next_n, :effective_len].argmax(dim=-1).int()
    pre_idx = torch.zeros(num_groups, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j

    seq_lens_val = N * compress_ratio
    seq_lens = torch.full((num_groups,), seq_lens_val, dtype=torch.int32, device="cuda")
    return logits, pre_idx, seq_lens


def _tie_aware_correct(
    kernel_idxs: torch.Tensor,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int,
    compress_ratio: int = 1,
):
    """Multi-row tie-aware correctness check.

    Per row r:
        actual_kv_len = seq_lens[r // next_n] - next_n + (r % next_n) + 1
        N_eff = actual_kv_len // compress_ratio

    Reference ``torch.topk`` is masked to this range so reference and
    kernel scan exactly the same columns.

    Returns (ok: bool, message: str).
    """
    num_rows = kernel_idxs.shape[0]
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()

    for row in range(num_rows):
        ofs = row % next_n
        actual_kv_len = int(seq_lens_host[row // next_n]) - next_n + ofs + 1
        N_eff = actual_kv_len // compress_ratio
        if N_eff < top_k:
            continue  # degenerate — skip

        row_logits = logits_f32[row, :N_eff]
        topk_vals, _ = torch.topk(row_logits, k=top_k, largest=True, sorted=True)
        kth_value = topk_vals[-1].item()

        sel = [int(i) for i in kernel_idxs[row].cpu().tolist() if i >= 0]
        if any(i >= N_eff for i in sel):
            return False, f"row={row}: out-of-range index"
        if len(set(sel)) != len(sel):
            return False, f"row={row}: duplicate indices"
        if len(sel) != top_k:
            return False, f"row={row}: returned {len(sel)} indices, expected {top_k}"

        sel_vals = row_logits[torch.tensor(sel, device=logits.device, dtype=torch.long)]
        n_below = int((sel_vals < kth_value).sum().item())
        if n_below > 0:
            return False, f"row={row}: {n_below} selected values < Kth-rank ({kth_value:.6f})"

        sel_sorted, _ = sel_vals.sort(descending=True)
        if not torch.allclose(sel_sorted, topk_vals, rtol=1e-5, atol=1e-5):
            max_diff = (sel_sorted - topk_vals).abs().max().item()
            return False, f"row={row}: sorted-value mismatch (max diff {max_diff:.4e})"

    return True, "ok"


# ---------------------------------------------------------------------------
# test_gvr_topk_decode — main knob sweep (256 combinations)
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [4096, 65536])
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("use_256bit_load", [False, True])
@pytest.mark.parametrize("num_threads_per_block", [512, 1024])
@pytest.mark.parametrize("enable_warp_parallel_reduce", [False, True])
@pytest.mark.parametrize("cluster_size", [1, 4])
def test_gvr_topk_decode(
    dtype,
    top_k,
    N,
    batch_size,
    use_256bit_load,
    num_threads_per_block,
    enable_warp_parallel_reduce,
    cluster_size,
):
    """Correctness sweep over all knob combinations."""
    next_n = 1
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k: degenerate path")

    seed = 42
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(num_rows, N, top_k, dtype, seed, next_n=next_n)

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    _, out_idxs = flashinfer.gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        top_k,
        next_n=next_n,
        num_sms=num_sms,
        use_256bit_load=use_256bit_load,
        num_threads_per_block=num_threads_per_block,
        enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        return_output_values=False,
        cluster_size=cluster_size,
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n)
    assert ok, (
        f"dtype={dtype} K={top_k} N={N} batch={batch_size} "
        f"256bit={use_256bit_load} T={num_threads_per_block} "
        f"warp_par={enable_warp_parallel_reduce} cs={cluster_size}: {msg}"
    )


# ---------------------------------------------------------------------------
# test_gvr_topk_decode_next_n — temporal stride next_n=2
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("N", [4096, 16384])
@pytest.mark.parametrize("batch_size", [2, 16])
def test_gvr_topk_decode_next_n(dtype, top_k, N, batch_size):
    """Verify next_n=2 path (temporal stride used in V3.2 speculative decode)."""
    next_n = 2
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k: degenerate path")

    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(num_rows, N, top_k, dtype, seed=7, next_n=next_n)

    _, out_idxs = flashinfer.gvr_topk_decode(
        logits, pre_idx, seq_lens, top_k, next_n=next_n, return_output_values=False
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n)
    assert ok, f"dtype={dtype} K={top_k} N={N} batch={batch_size} next_n={next_n}: {msg}"


# ---------------------------------------------------------------------------
# test_gvr_topk_decode_values — return_output_values=True
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("N", [8192, 32768])
def test_gvr_topk_decode_values(dtype, top_k, N):
    """Verify that returned top-k values match logits[row, indices]."""
    batch_size = 4
    next_n = 1
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13, next_n=next_n)

    out_vals, out_idxs = flashinfer.gvr_topk_decode(
        logits, pre_idx, seq_lens, top_k, return_output_values=True
    )
    torch.cuda.synchronize()

    assert out_vals is not None
    assert out_vals.shape == (batch_size, top_k)
    assert out_idxs.shape == (batch_size, top_k)

    # Values at returned indices must equal out_vals (within dtype precision).
    logits_f32 = logits.float()
    for row in range(batch_size):
        idxs = out_idxs[row].long()
        expected_vals = logits_f32[row][idxs]
        actual_vals = out_vals[row].float()
        assert torch.allclose(expected_vals, actual_vals, rtol=1e-3, atol=1e-3), (
            f"row={row}: value mismatch at indices"
        )

    # Correctness check on indices.
    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n)
    assert ok, f"dtype={dtype} K={top_k} N={N}: {msg}"


# ---------------------------------------------------------------------------
# test_gvr_topk_sort_prepare — seqlen_sorted dispatch
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("N", [4096, 16384])
@pytest.mark.parametrize("batch_size", [4, 32])
def test_gvr_topk_sort_prepare(top_k, N, batch_size):
    """Verify seqlen_sorted=True with order_row from gvr_topk_sort_prepare."""
    dtype = torch.bfloat16
    next_n = 1
    logits, pre_idx, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=99, next_n=next_n
    )

    # Mix up seq_lens so sorting does something observable.
    for i in range(batch_size):
        seq_lens[i] = N - i * (N // batch_size // 4)
    seq_lens = seq_lens.clamp(min=top_k + next_n)

    order_row = flashinfer.gvr_topk_sort_prepare(seq_lens)
    assert order_row.dtype == torch.int32
    assert order_row.shape == seq_lens.shape

    _, out_idxs = flashinfer.gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        top_k,
        next_n=next_n,
        seqlen_sorted=True,
        order_row=order_row,
        return_output_values=False,
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n)
    assert ok, f"K={top_k} N={N} batch={batch_size} seqlen_sorted=True: {msg}"


# ---------------------------------------------------------------------------
# test_gvr_topk_sort_prepare_ordering — order_row is longest-first
# ---------------------------------------------------------------------------


@requires_blackwell
def test_gvr_topk_sort_prepare_ordering():
    """order_row from gvr_topk_sort_prepare must be descending in seq_lens."""
    seq_lens = torch.tensor([100, 500, 200, 800, 50], dtype=torch.int32, device="cuda")
    order_row = flashinfer.gvr_topk_sort_prepare(seq_lens)
    order_row_cpu = order_row.cpu()
    seq_lens_cpu = seq_lens.cpu()

    # The reordered lengths must be descending.
    reordered = seq_lens_cpu[order_row_cpu.long()]
    for i in range(len(reordered) - 1):
        assert reordered[i] >= reordered[i + 1], (
            f"order_row is not longest-first at position {i}: "
            f"{reordered[i]} < {reordered[i+1]}"
        )


# ---------------------------------------------------------------------------
# test_gvr_topk_compress_ratio — compress_ratio=4 (DSv4 mode)
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("N", [4096])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_gvr_topk_compress_ratio(dtype, top_k, N, batch_size):
    """Verify compress_ratio=4 (KV-indexer compression, DSv4 mode)."""
    next_n = 1
    compress_ratio = 4
    logits, pre_idx, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=55, next_n=next_n, compress_ratio=compress_ratio
    )

    _, out_idxs = flashinfer.gvr_topk_decode(
        logits, pre_idx, seq_lens, top_k, compress_ratio=compress_ratio
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(
        out_idxs, logits, seq_lens, top_k, next_n, compress_ratio=compress_ratio
    )
    assert ok, f"dtype={dtype} K={top_k} N={N} batch={batch_size} cr={compress_ratio}: {msg}"


# ---------------------------------------------------------------------------
# test_gvr_topk_preallocated — pre-allocated output buffers
# ---------------------------------------------------------------------------


@requires_blackwell
def test_gvr_topk_preallocated():
    """Test that pre-allocated out_values and out_indices are used correctly."""
    dtype = torch.bfloat16
    top_k = 512
    N = 4096
    batch_size = 4

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_vals = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")
    out_idxs = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")

    returned_vals, returned_idxs = flashinfer.gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        top_k,
        return_output_values=True,
        out_values=out_vals,
        out_indices=out_idxs,
    )
    torch.cuda.synchronize()

    # Should return the same buffers (same data_ptr).
    assert returned_vals is out_vals
    assert returned_idxs is out_idxs

    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n=1)
    assert ok, msg


# ---------------------------------------------------------------------------
# test_gvr_topk_lb_prepare — LB classifier smoke test
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("max_batch_size", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_gvr_topk_lb_prepare(max_batch_size, batch_size):
    """Verify that the LB prepare kernel writes valid order_row and counters."""
    if batch_size > max_batch_size:
        pytest.skip("batch_size > max_batch_size")

    N = 8192
    long_threshold = 4096
    seq_lens = torch.randint(low=1024, high=N * 2, size=(batch_size,), dtype=torch.int32, device="cuda")

    order_row, counters = flashinfer.gvr_topk_lb_prepare(
        seq_lens,
        max_batch_size=max_batch_size,
        long_threshold=long_threshold,
    )
    torch.cuda.synchronize()

    assert order_row.shape == (max_batch_size,)
    assert counters.shape == (2,)
    assert order_row.dtype == torch.int32
    assert counters.dtype == torch.int32

    n_long = int(counters[0].item())
    n_short = int(counters[1].item())
    assert n_long + n_short == batch_size, (
        f"n_long={n_long} + n_short={n_short} != batch_size={batch_size}"
    )
    assert n_long >= 0 and n_short >= 0

    # All valid order_row entries (first batch_size) should be in [0, batch_size).
    valid_entries = order_row[:batch_size].cpu()
    for i, entry in enumerate(valid_entries.tolist()):
        assert 0 <= entry < batch_size, f"order_row[{i}]={entry} out of range"


# ---------------------------------------------------------------------------
# test_gvr_topk_lb_prepare_invalid — bad max_batch_size raises
# ---------------------------------------------------------------------------


@requires_blackwell
def test_gvr_topk_lb_prepare_invalid():
    """Bad max_batch_size values must raise ValueError."""
    seq_lens = torch.full((4,), 1024, dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="power of 2"):
        flashinfer.gvr_topk_lb_prepare(seq_lens, max_batch_size=100)  # not power-of-2

    with pytest.raises(ValueError, match="power of 2"):
        flashinfer.gvr_topk_lb_prepare(seq_lens, max_batch_size=32)  # below 64

    with pytest.raises(ValueError):
        flashinfer.gvr_topk_lb_prepare(seq_lens, max_batch_size=2048)  # above 1024

    with pytest.raises(ValueError, match="max_batch_size"):
        # batch_size > max_batch_size
        seq_lens_big = torch.full((200,), 1024, dtype=torch.int32, device="cuda")
        flashinfer.gvr_topk_lb_prepare(seq_lens_big, max_batch_size=64)


# ---------------------------------------------------------------------------
# test_gvr_topk_lb_roundtrip — full LB prepare + decode correctness
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("N", [4096, 32768])
@pytest.mark.parametrize("batch_size", [4, 32])
def test_gvr_topk_lb_roundtrip(dtype, top_k, N, batch_size):
    """Full LB pipeline: prepare → decode. Verify result against torch.topk."""
    next_n = 1
    compress_ratio = 1
    max_batch_size = 128
    cluster_size = 4
    num_threads = 512

    if batch_size > max_batch_size:
        pytest.skip("batch_size > max_batch_size for this test")

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=77, next_n=next_n)

    order_row, counters = flashinfer.gvr_topk_lb_prepare(
        seq_lens,
        max_batch_size=max_batch_size,
        long_threshold=N // 2,
    )

    _, out_idxs = flashinfer.gvr_topk_lb_decode(
        logits,
        pre_idx,
        seq_lens,
        order_row,
        counters,
        top_k,
        next_n=next_n,
        compress_ratio=compress_ratio,
        cluster_size=cluster_size,
        max_batch_size=max_batch_size,
        num_threads=num_threads,
        return_output_values=False,
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n, compress_ratio)
    assert ok, (
        f"dtype={dtype} K={top_k} N={N} batch={batch_size} "
        f"LB roundtrip: {msg}"
    )


# ---------------------------------------------------------------------------
# test_gvr_topk_lb_decode_values — LB with return_output_values=True
# ---------------------------------------------------------------------------


@requires_blackwell
def test_gvr_topk_lb_decode_values():
    """Verify LB decode returns correct values when return_output_values=True."""
    dtype = torch.bfloat16
    top_k = 512
    N = 4096
    batch_size = 8
    max_batch_size = 128

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=21)

    order_row, counters = flashinfer.gvr_topk_lb_prepare(
        seq_lens, max_batch_size=max_batch_size
    )

    out_vals, out_idxs = flashinfer.gvr_topk_lb_decode(
        logits,
        pre_idx,
        seq_lens,
        order_row,
        counters,
        top_k,
        max_batch_size=max_batch_size,
        return_output_values=True,
    )
    torch.cuda.synchronize()

    assert out_vals is not None
    assert out_vals.shape == (batch_size, top_k)

    logits_f32 = logits.float()
    for row in range(batch_size):
        idxs = out_idxs[row].long()
        expected = logits_f32[row][idxs]
        actual = out_vals[row].float()
        assert torch.allclose(expected, actual, rtol=1e-3, atol=1e-3), (
            f"row={row}: LB output values do not match logits[row, indices]"
        )


# ---------------------------------------------------------------------------
# test_can_use_gvr_topk — hardware gate
# ---------------------------------------------------------------------------


def test_can_use_gvr_topk_cuda():
    """can_use_gvr_topk returns a bool on any CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    result = flashinfer.can_use_gvr_topk(torch.device("cuda"))
    assert isinstance(result, bool)


def test_can_use_gvr_topk_cpu():
    """can_use_gvr_topk returns False for CPU device."""
    assert flashinfer.can_use_gvr_topk(torch.device("cpu")) is False


# ---------------------------------------------------------------------------
# test_gvr_topk_bad_dtype — unsupported dtype
# ---------------------------------------------------------------------------


@requires_blackwell
def test_gvr_topk_bad_dtype():
    """Passing an unsupported dtype must raise TypeError."""
    batch_size, N, top_k = 2, 4096, 512
    # int32 is not a supported logit dtype — cast from float to avoid torch.randn limitation
    logits = torch.randn(batch_size, N, device="cuda").to(torch.int32)
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((batch_size,), N, dtype=torch.int32, device="cuda")

    with pytest.raises((TypeError, ValueError)):
        flashinfer.gvr_topk_decode(logits, pre_idx, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_gvr_topk_large — single large-batch stress test
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("top_k", [1024])
def test_gvr_topk_large(dtype, top_k):
    """Stress test: large batch × large sequence length."""
    N = 65536
    batch_size = 128

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=9)

    _, out_idxs = flashinfer.gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        top_k,
        cluster_size=4,
        return_output_values=False,
    )
    torch.cuda.synchronize()

    ok, msg = _tie_aware_correct(out_idxs, logits, seq_lens, top_k, next_n=1)
    assert ok, f"large stress test dtype={dtype} K={top_k} N={N} batch={batch_size}: {msg}"


# ---------------------------------------------------------------------------
# test_gvr_topk_stream_safety — output is ready after synchronize
# ---------------------------------------------------------------------------


@requires_blackwell
def test_gvr_topk_stream_safety():
    """Kernel must be stream-safe: second call reuses compiled kernel."""
    dtype = torch.bfloat16
    top_k = 512
    N = 4096
    batch_size = 2

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=3)

    # First call — triggers compilation.
    _, idxs1 = flashinfer.gvr_topk_decode(logits, pre_idx, seq_lens, top_k)
    torch.cuda.synchronize()

    # Second call — must reuse compiled kernel (no error, same result).
    _, idxs2 = flashinfer.gvr_topk_decode(logits, pre_idx, seq_lens, top_k)
    torch.cuda.synchronize()

    # Both calls on the same input must produce the same top-k set per row.
    for row in range(batch_size):
        s1 = set(idxs1[row].cpu().tolist())
        s2 = set(idxs2[row].cpu().tolist())
        assert s1 == s2, f"row={row}: repeated calls produced different top-k sets"


# ---------------------------------------------------------------------------
# test_gvr_topk_seqlen_sorted_requires_order_row — input validation
# ---------------------------------------------------------------------------


@requires_blackwell
def test_gvr_topk_seqlen_sorted_requires_order_row():
    """seqlen_sorted=True without order_row must raise AssertionError."""
    dtype = torch.bfloat16
    top_k = 512
    N = 4096
    batch_size = 4

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=5)

    with pytest.raises(AssertionError):
        flashinfer.gvr_topk_decode(
            logits, pre_idx, seq_lens, top_k, seqlen_sorted=True, order_row=None
        )
