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
"""Correctness tests for flashinfer.top_k_decode.

On Blackwell (sm_100+) with ``pre_idx`` supplied the GVR fast path runs.
On other hardware the radix fallback is used; most tests still execute.

Test matrix
-----------
test_basic_decode             — dtype × top_k × N × batch; works on all GPUs
test_return_values            — return_values=True correctness
test_next_n                   — next_n=2 (V3.2 speculative-decode stride)
test_compress_ratio           — compress_ratio=4 (DSv4 KV compression)
test_preallocated_outputs     — pre-allocated out_indices / out_values
test_large_batch              — stress: large batch × long rows
test_repeated_calls           — same inputs twice → same top-K set
test_no_pre_idx_fallback      — pre_idx=None always goes to radix path
test_lb_config_validation     — GvrTopKLBConfig bad args raise at construction
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.cute_dsl.top_k.config import GvrTopKLBConfig
    from flashinfer.cute_dsl.utils import is_cute_dsl_available
    from flashinfer.utils import get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False
    GvrTopKLBConfig = None

pytestmark = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE, reason="flashinfer not installed"
)

# True only on Blackwell (sm_100+) with nvidia-cutlass-dsl installed.
# Use the public is_backend_supported() method exposed by @backend_requirement.
def _gvr_hw_supported() -> bool:
    if not torch.cuda.is_available() or not _FLASHINFER_AVAILABLE:
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    cc = major * 10 + minor
    return (
        flashinfer.top_k_decode.is_backend_supported("gvr", cc)
        and is_cute_dsl_available()
    )

_IS_BLACKWELL = _gvr_hw_supported()

requires_blackwell = pytest.mark.skipif(
    not _IS_BLACKWELL,
    reason="GVR fast path requires Blackwell (sm_100+) and nvidia-cutlass-dsl",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(num_rows, N, top_k, dtype, seed, next_n=1, compress_ratio=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0).to(dtype)
    num_groups = num_rows // next_n
    effective_len = N - next_n + 1
    argmax_idx = logits[::next_n, :effective_len].argmax(dim=-1).int()
    pre_idx = torch.zeros(num_groups, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = argmax_idx
    for j in range(1, top_k):
        pre_idx[:, j] = j
    seq_lens = torch.full(
        (num_groups,), N * compress_ratio, dtype=torch.int32, device="cuda"
    )
    return logits, pre_idx, seq_lens


def _check_correct(indices, logits, seq_lens, top_k, next_n=1, compress_ratio=1):
    """Every selected value must be >= the k-th largest in its row."""
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    for row in range(indices.shape[0]):
        ofs = row % next_n
        actual_kv_len = int(seq_lens_host[row // next_n]) - next_n + ofs + 1
        N_eff = actual_kv_len // compress_ratio
        if N_eff < top_k:
            continue
        row_logits = logits_f32[row, :N_eff]
        kth_value = torch.topk(row_logits, k=top_k).values[-1].item()
        sel = [int(i) for i in indices[row].cpu().tolist() if i >= 0]
        assert len(sel) == top_k, f"row={row}: got {len(sel)} indices, want {top_k}"
        assert len(set(sel)) == len(sel), f"row={row}: duplicate indices"
        assert all(i < N_eff for i in sel), f"row={row}: out-of-range index"
        sel_vals = row_logits[torch.tensor(sel, device=logits.device, dtype=torch.long)]
        assert (sel_vals < kth_value).sum() == 0, (
            f"row={row}: some selected values below kth-rank ({kth_value:.6f})"
        )


# ---------------------------------------------------------------------------
# test_basic_decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("N", [4096, 32768])
@pytest.mark.parametrize("batch_size", [1, 32])
def test_basic_decode(dtype, top_k, N, batch_size):
    """top_k_decode with pre_idx: works on Blackwell (GVR) and any GPU (radix)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if N < top_k:
        pytest.skip("N < top_k")

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=42)
    pre_idx_arg = pre_idx if _IS_BLACKWELL else None

    indices = flashinfer.top_k_decode(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    if _IS_BLACKWELL:
        _check_correct(indices, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_return_values
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_return_values(dtype, top_k):
    """Returned values must equal logits[row, indices]."""
    N, batch_size = 8192, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13)

    indices, values = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=pre_idx, return_values=True
    )
    torch.cuda.synchronize()

    assert values.shape == (batch_size, top_k)
    logits_f32 = logits.float()
    for row in range(batch_size):
        expected = logits_f32[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


# ---------------------------------------------------------------------------
# test_next_n
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("batch_size", [2, 16])
def test_next_n(dtype, top_k, batch_size):
    """next_n=2: two rows share one pre_idx / seq_len entry."""
    next_n, N = 2, 8192
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k")
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=7, next_n=next_n
    )

    indices = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=pre_idx, next_n=next_n
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


# ---------------------------------------------------------------------------
# test_compress_ratio
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_compress_ratio(dtype, top_k):
    """compress_ratio=4: seq_lens in uncompressed-token space."""
    compress_ratio, N, batch_size = 4, 4096, 8
    logits, pre_idx, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=55, compress_ratio=compress_ratio
    )

    indices = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=pre_idx, compress_ratio=compress_ratio
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, compress_ratio=compress_ratio)


# ---------------------------------------------------------------------------
# test_preallocated_outputs
# ---------------------------------------------------------------------------


@requires_blackwell
def test_preallocated_outputs():
    """out_indices and out_values passed by caller are written in-place."""
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")

    ret_i, ret_v = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=pre_idx,
        out_indices=out_i, return_values=True, out_values=out_v,
    )
    torch.cuda.synchronize()

    assert ret_i is out_i
    assert ret_v is out_v
    _check_correct(out_i, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_large_batch
# ---------------------------------------------------------------------------


@requires_blackwell
def test_large_batch():
    """128 rows × 65536 cols stress test."""
    dtype, top_k, N, batch_size = torch.bfloat16, 1024, 65536, 128
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=9)

    indices = flashinfer.top_k_decode(logits, seq_lens, top_k, pre_idx=pre_idx)
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_repeated_calls
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_repeated_calls():
    """Two identical calls must return the same top-K index set per row."""
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=3)
    pre_idx_arg = pre_idx if _IS_BLACKWELL else None

    idx1 = flashinfer.top_k_decode(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()
    idx2 = flashinfer.top_k_decode(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()

    for row in range(batch_size):
        assert set(idx1[row].cpu().tolist()) == set(idx2[row].cpu().tolist()), (
            f"row={row}: repeated calls returned different top-k sets"
        )


# ---------------------------------------------------------------------------
# test_no_pre_idx_fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_no_pre_idx_fallback():
    """pre_idx=None must always use the radix fallback on any GPU."""
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=77)

    indices = flashinfer.top_k_decode(logits, seq_lens, top_k, pre_idx=None)
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32


# ---------------------------------------------------------------------------
# Radix-backend tests (run on any GPU, backend="radix" forced explicitly)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_return_values(dtype, top_k):
    """Radix backend: returned values must equal logits[row, indices]."""
    N, batch_size = 8192, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13)

    indices, values = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=None, return_values=True, backend="radix"
    )
    torch.cuda.synchronize()

    assert values.shape == (batch_size, top_k)
    logits_f32 = logits.float()
    for row in range(batch_size):
        expected = logits_f32[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("batch_size", [2, 16])
def test_radix_next_n(dtype, top_k, batch_size):
    """Radix backend: next_n=2 — two rows share one seq_len entry."""
    next_n, N = 2, 8192
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k")
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=7, next_n=next_n
    )

    indices = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=None, next_n=next_n, backend="radix"
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_compress_ratio(dtype, top_k):
    """Radix backend: compress_ratio=4 — seq_lens in uncompressed-token space."""
    compress_ratio, N, batch_size = 4, 4096, 8
    logits, _, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=55, compress_ratio=compress_ratio
    )

    indices = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=None,
        compress_ratio=compress_ratio, backend="radix"
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, compress_ratio=compress_ratio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_radix_preallocated_outputs():
    """Radix backend: out_indices and out_values are written in-place."""
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")

    ret_i, ret_v = flashinfer.top_k_decode(
        logits, seq_lens, top_k, pre_idx=None,
        out_indices=out_i, return_values=True, out_values=out_v, backend="radix"
    )
    torch.cuda.synchronize()

    assert ret_i is out_i
    assert ret_v is out_v
    _check_correct(out_i, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_lb_config_validation
# ---------------------------------------------------------------------------


def test_lb_config_validation():
    """GvrTopKLBConfig raises ValueError on invalid arguments."""
    with pytest.raises(ValueError, match="power of 2"):
        GvrTopKLBConfig(max_batch_size=100)
    with pytest.raises(ValueError, match="power of 2"):
        GvrTopKLBConfig(max_batch_size=32)
    with pytest.raises(ValueError, match="power of 2"):
        GvrTopKLBConfig(max_batch_size=2048)
    with pytest.raises(ValueError, match="cluster_size"):
        GvrTopKLBConfig(cluster_size=0)
    with pytest.raises(ValueError, match="num_threads"):
        GvrTopKLBConfig(num_threads=256)
