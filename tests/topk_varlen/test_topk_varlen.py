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

"""Correctness tests for flashinfer.top_k_varlen.

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
test_no_pre_idx_selects_radix — pre_idx=None → a radix backend (never GVR), correct
test_lb_config_validation     — GvrTopKLBConfig bad args raise at construction
test_load_balance_modes       — True/False GVR paths correct
test_gvr_row_width_alignment  — GVR rejects non-vec-aligned N
test_radix_cutlass_*          — masked CUTLASS radix (any GPU) coverage
test_auto_gvr_knobs_256bit_alignment_gate  — 256-bit gated on 32B N alignment
test_lb_256bit_misaligned_no_crash  — N=4104 LB regression (latent crash fixed)
test_auto_gvr_knobs_shape_aware  — auto() picks shape-appropriate launch config

radix (CuTe DSL) backend — Blackwell only
-----------------------------------------
test_radix_basic              — single-CTA correctness across dtype/K/batch
test_radix_multi_cta_regime   — ctas_per_group > 1 (SMEM split + small-batch fan-out;
                                covers the N=131072 SMEM-overflow regression)
test_radix_next_n / _compress_ratio / _return_values / _preallocated_outputs
test_varlen_ragged            — distinct per-row seq_lens (radix + radix_cutlass)
test_seq_len_equals_top_k     — degenerate seq_len == top_k selects all valid indices

Cross-cutting
-------------
test_cuda_graph_radix_multi_cta — capture/replay incl. fresh-data replay (row_states guard)
test_cuda_graph_gvr           — GVR under CUDA graph
test_backend_heuristic_priority — auto priority gvr > radix > radix_cutlass
test_heuristic_signature_mirrors_api — every API parameter is accepted by the auto heuristic
test_cross_backend_value_consistency — all backends select the same value multiset
test_unknown_backend_rejected — unregistered / pre-rename backend names rejected
test_input_validation         — 1-D logits / non-int32 seq_lens rejected
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.topk_varlen.kernels.config import GvrTopKLBConfig
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
        flashinfer.top_k_varlen.is_backend_supported("gvr", cc)
        and is_cute_dsl_available()
    )


_IS_BLACKWELL = _gvr_hw_supported()

requires_blackwell = pytest.mark.skipif(
    not _IS_BLACKWELL,
    reason="GVR fast path requires Blackwell (sm_100+) and nvidia-cutlass-dsl",
)

# Backends compiled by nvcc; every other backend is a CuTe-DSL kernel and needs
# nvidia-cutlass-dsl at call time, which is_backend_supported() (a static
# compute-capability list) does not know about.
_NVCC_BACKENDS = ("radix_cutlass", "sglang")


def _backend_hw_supported(backend: str) -> bool:
    """is_backend_supported(backend, cc) on the current device, plus the DSL
    package for the CuTe-DSL backends."""
    if not torch.cuda.is_available() or not _FLASHINFER_AVAILABLE:
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    if not flashinfer.top_k_varlen.is_backend_supported(backend, major * 10 + minor):
        return False
    return backend in _NVCC_BACKENDS or is_cute_dsl_available()


def _skip_unless_backend(backend: str) -> None:
    if not _backend_hw_supported(backend):
        pytest.skip(f"{backend} unsupported on this device")


# radix_primitives is Ampere+ and compiles distinct kernels there (the
# warp-aggregated walker on sm_8x, no PDL below sm_90): gating its tests on
# the GVR predicate would skip those paths everywhere but datacentre Blackwell.
requires_radix_primitives = pytest.mark.skipif(
    not _backend_hw_supported("radix_primitives"),
    reason="radix_primitives requires Ampere+ (sm_80+) and nvidia-cutlass-dsl",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(num_rows, N, top_k, dtype, seed, next_n=1, compress_ratio=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
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


def _check_correct(
    indices,
    logits,
    seq_lens,
    top_k,
    next_n=1,
    compress_ratio=1,
    require_all_checked=False,
):
    """Every selected value must be >= the k-th largest in its row.

    With ``require_all_checked=True`` every row must be non-degenerate
    (``N_eff >= top_k``) and actually verified — this turns the otherwise-silent
    "skip degenerate row" branch into a hard failure, guarding against a
    mis-parametrized test that quietly checks nothing.
    """
    logits_f32 = logits.to(torch.float32)
    seq_lens_host = seq_lens.cpu().tolist()
    n_checked = 0
    for row in range(indices.shape[0]):
        ofs = row % next_n
        actual_kv_len = int(seq_lens_host[row // next_n]) - next_n + ofs + 1
        N_eff = actual_kv_len // compress_ratio
        if N_eff < top_k:
            if require_all_checked:
                raise AssertionError(
                    f"row={row}: N_eff={N_eff} < top_k={top_k} — degenerate row "
                    f"not allowed under require_all_checked"
                )
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
        n_checked += 1
    if require_all_checked:
        assert n_checked == indices.shape[0], (
            f"only {n_checked}/{indices.shape[0]} rows were verified"
        )


def _make_varlen_inputs(seq_len_list, N, dtype, seed):
    """Ragged batch: per-row seq_lens vary; no pre_idx (radix backends).

    Returns ``(logits[batch, N], seq_lens[batch] int32)`` where
    ``seq_lens[i] = seq_len_list[i]``.
    """
    batch_size = len(seq_len_list)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
    seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device="cuda")
    return logits, seq_lens


def _radix_ctas(N, dtype, batch_size):
    """ctas_per_group the radix (CuTe DSL) backend will use for this shape."""
    from flashinfer.topk_varlen.topk_varlen import _radix_get_chunk_config
    from flashinfer.utils import get_device_sm_count, get_shared_bytes_per_block_optin

    device = torch.device("cuda")
    num_sms = get_device_sm_count(device)
    smem_capacity = get_shared_bytes_per_block_optin(device)
    ctas, _chunk = _radix_get_chunk_config(N, dtype, batch_size, num_sms, smem_capacity)
    return ctas


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
    """top_k_varlen with pre_idx: works on Blackwell (GVR) and any GPU (radix)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if top_k > N:
        pytest.skip("N < top_k")

    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=42)
    pre_idx_arg = pre_idx if _IS_BLACKWELL else None

    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    # Correctness is verifiable on any GPU: Blackwell runs GVR (pre_idx), other
    # hardware runs the masked radix_cutlass fallback — both produce a valid top-K.
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

    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, return_values=True
    )
    torch.cuda.synchronize()

    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype  # auto-allocated values keep the logits dtype
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

    indices, _ = flashinfer.top_k_varlen(
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

    indices, _ = flashinfer.top_k_varlen(
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

    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        out_indices=out_i,
        return_values=True,
        out_values=out_v,
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

    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx)
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# test_repeated_calls
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_repeated_calls():
    """Repeated identical calls each return a valid top-K (no state corruption).

    Results need not be bit-identical: the radix_cutlass fallback runs with
    deterministic=False, so BF16 values that tie at the K-th boundary let two
    correct calls select different (equally valid) tied indices. Assert each
    call is a correct top-K rather than requiring identical index sets.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=3)
    pre_idx_arg = pre_idx if _IS_BLACKWELL else None

    idx1, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()
    idx2, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=pre_idx_arg)
    torch.cuda.synchronize()

    _check_correct(idx1, logits, seq_lens, top_k, require_all_checked=True)
    _check_correct(idx2, logits, seq_lens, top_k, require_all_checked=True)


# ---------------------------------------------------------------------------
# test_no_pre_idx_selects_radix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_no_pre_idx_selects_radix():
    """pre_idx=None resolves auto to a radix backend (never GVR) and is correct.

    On Blackwell auto picks ``radix`` (CuTe DSL); on other hardware it picks
    ``radix_cutlass`` (masked CUTLASS). GVR requires pre_idx, so it is never
    selected here.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=77)

    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=None)
    torch.cuda.synchronize()

    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    # auto without pre_idx must resolve to a radix backend, never gvr.
    assert flashinfer.top_k_varlen.suitable_auto_backends[0] in (
        "radix",
        "radix_cutlass",
    )
    _check_correct(indices, logits, seq_lens, top_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_skip_check_auto_backend():
    """skip_check=True with backend="auto" must not raise TypeError.

    When skip_check=True the decorator still calls heuristic_func with positional
    args (*args from the caller).  The heuristic's old **kwargs signature caused
    TypeError because the positional logits/seq_lens/top_k arguments overflowed
    the single 'suitable_backends' slot.  Spelling out the full signature fixes it.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=77)
    # Must not raise TypeError regardless of hardware.
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, backend="auto", skip_check=True
    )
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)
    _check_correct(indices, logits, seq_lens, top_k)


def test_heuristic_signature_mirrors_api():
    """Every ``top_k_varlen`` parameter must be a parameter of the auto heuristic.

    The decorator binds the API defaults and forwards them all to the heuristic
    as keyword arguments, so a parameter added to the API but not to the
    heuristic makes every ``backend="auto"`` call raise TypeError (the paged
    output parameters did exactly that).  Hardware-independent.
    """
    import inspect

    from flashinfer.topk_varlen.topk_varlen import _top_k_varlen_heuristic

    api = inspect.signature(flashinfer.top_k_varlen).parameters
    heuristic = inspect.signature(_top_k_varlen_heuristic).parameters
    missing = [name for name in api if name not in heuristic]
    assert not missing, f"auto heuristic lacks API parameters: {missing}"


# ---------------------------------------------------------------------------
# Radix-backend tests (run on any GPU, backend="radix_cutlass" forced explicitly)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_cutlass_return_values(dtype, top_k):
    """radix_cutlass backend: returned values must equal logits[row, indices]."""
    N, batch_size = 8192, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13)

    indices, values = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        return_values=True,
        backend="radix_cutlass",
    )
    torch.cuda.synchronize()

    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype  # auto-allocated values keep the logits dtype
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
def test_radix_cutlass_next_n(dtype, top_k, batch_size):
    """radix_cutlass backend: next_n=2 — two rows share one seq_len entry."""
    next_n, N = 2, 8192
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k")
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(num_rows, N, top_k, dtype, seed=7, next_n=next_n)

    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, next_n=next_n, backend="radix_cutlass"
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_cutlass_compress_ratio(dtype, top_k):
    """radix_cutlass backend: compress_ratio=4 — seq_lens in uncompressed-token space."""
    compress_ratio, N, batch_size = 4, 4096, 8
    logits, _, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=55, compress_ratio=compress_ratio
    )

    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        compress_ratio=compress_ratio,
        backend="radix_cutlass",
    )
    torch.cuda.synchronize()

    _check_correct(indices, logits, seq_lens, top_k, compress_ratio=compress_ratio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_radix_cutlass_preallocated_outputs():
    """radix_cutlass backend: out_indices and out_values are written in-place."""
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")

    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        out_indices=out_i,
        return_values=True,
        out_values=out_v,
        backend="radix_cutlass",
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


# ---------------------------------------------------------------------------
# test_load_balance_modes — True / False correct
# ---------------------------------------------------------------------------


def _make_ragged_gvr_inputs(top_k, dtype=torch.bfloat16, next_n=1):
    """4 long requests (> 64K threshold) + 12 short requests: a ragged batch.

    With ``next_n > 1``, each request contributes ``next_n`` logit rows, so
    ``logits`` has shape ``(batch_size * next_n, N)``.
    """
    N = 128 * 1024
    seq_len_list = [N] * 4 + [2048] * 12
    batch_size = len(seq_len_list)
    num_rows = batch_size * next_n
    torch.manual_seed(7)
    logits = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 2.0).to(
        dtype
    )
    seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device="cuda")
    logits_f32 = logits.to(torch.float32)
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
    for r in range(batch_size):
        # Primary row (nn=0): effective range is [0, seq_len - next_n + 1).
        effective_len = seq_len_list[r] - next_n + 1
        pre_idx[r, 0] = int(logits_f32[r * next_n, :effective_len].argmax().item())
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")
    return logits, seq_lens, pre_idx


@requires_blackwell
@pytest.mark.parametrize(
    "load_balance,next_n", [(True, 1), (False, 1), (True, 2), (False, 2)]
)
def test_load_balance_modes(load_balance, next_n):
    """load_balance=True/False, next_n=1/2 all produce correct GVR top-K on a ragged batch.

    The next_n=2, load_balance=False combination specifically exercises _run_gvr
    (the single-CTA path) whose order_row was previously constructed with a
    [::next_n] slice bug that made it too short when next_n > 1.
    """
    top_k = 512
    logits, seq_lens, pre_idx = _make_ragged_gvr_inputs(top_k, next_n=next_n)
    num_rows = logits.shape[0]
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        next_n=next_n,
        backend="gvr",
        load_balance=load_balance,
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@requires_blackwell
def test_gvr_lb_workspace_reuse():
    """Caller-provided workspace buffers are reused across GVR LB calls.

    Verifies that passing a pre-allocated workspace dict produces the same
    correct results as the default (locally-allocated) path, and that the
    same buffers can be safely reused across multiple calls.
    """
    top_k, batch_size = 512, 8
    logits, seq_lens, pre_idx = _make_ragged_gvr_inputs(top_k)
    batch_size = seq_lens.shape[0]

    # Compute max_batch_size: smallest power-of-2 in [64, 1024] >= batch_size.
    max_batch_size = next(m for m in (64, 128, 256, 512, 1024) if m >= batch_size)
    workspace = {
        "gvr_order_row": torch.empty(max_batch_size, dtype=torch.int32, device="cuda"),
        "gvr_counters": torch.empty(2, dtype=torch.int32, device="cuda"),
    }

    # First call — workspace gets populated.
    indices0, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", workspace=workspace
    )
    torch.cuda.synchronize()
    _check_correct(indices0, logits, seq_lens, top_k)

    # Second call with same workspace — must give identical correct results.
    indices1, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", workspace=workspace
    )
    torch.cuda.synchronize()
    _check_correct(indices1, logits, seq_lens, top_k)
    assert torch.equal(indices0, indices1), "workspace reuse changed the result"


@requires_blackwell
def test_gvr_no_lb_next_n():
    """load_balance=False with next_n=2: order_row must be request-level, not row-level.

    Regression test for the [::next_n] slice bug: seq_lens already has shape
    (num_requests,), so slicing it with [::next_n] produced an order_row that was
    next_n times too short, causing out-of-bounds kernel accesses when next_n > 1.
    """
    top_k, next_n, N = 512, 2, 8192
    batch_size = 8  # requests
    num_rows = batch_size * next_n
    logits, pre_idx, seq_lens = _make_inputs(
        num_rows, N, top_k, torch.bfloat16, seed=11, next_n=next_n
    )
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        next_n=next_n,
        backend="gvr",
        load_balance=False,
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


# ---------------------------------------------------------------------------
# test_gvr_row_width_alignment — GVR N must be vec-aligned; radix is unconstrained
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,align", [(torch.bfloat16, 8), (torch.float16, 8), (torch.float32, 4)]
)
def test_gvr_row_width_alignment(dtype, align):
    """GVR rejects misaligned N for explicit backend="gvr" and auto-routes past it.

    GVR uses 128-bit vectorized loads, so each row must be 16-byte aligned.
    The suitability check catches this and:
      - raises ValueError for explicit backend="gvr"
      - falls back to radix_cutlass for backend="auto" (pre_idx provided)
    """
    top_k, batch_size = 512, 4
    N_bad = 4096 + 1  # not a multiple of 4 or 8 for any supported dtype
    logits = torch.randn(batch_size, N_bad, dtype=dtype, device="cuda")
    seq_lens = torch.full((batch_size,), N_bad, dtype=torch.int32, device="cuda")
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")

    # Explicit backend="gvr" must fail (alignment check fires in the suitability
    # function; the decorator raises the generic problem-size error).
    with pytest.raises(ValueError, match="not supported"):
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", load_balance=False
        )

    # backend="auto" must succeed by routing to radix_cutlass (no alignment constraint).
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="auto"
    )
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_radix_cutlass_row_width_no_alignment_constraint():
    """radix_cutlass backend accepts any N (no vectorized-load alignment requirement)."""
    top_k, batch_size, N_bad = 512, 4, 4097
    logits = torch.randn(batch_size, N_bad, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.full((batch_size,), N_bad, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_cutlass"
    )
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)


# ---------------------------------------------------------------------------
# Shape-aware launch config (GvrTopKConfig.auto) + 256-bit N-alignment gate
# ---------------------------------------------------------------------------


def test_auto_gvr_knobs_256bit_alignment_gate():
    """_auto_gvr_knobs force-disables 256-bit loads unless N is 32-byte aligned.

    256-bit loads assume 32B-aligned rows (N*itemsize % 32); the up-front N check
    only guarantees 16B. The gate keeps a 256-bit kernel from being selected for a
    16B-but-not-32B-aligned N (which would fault). No GPU needed beyond dtype size.
    """
    from flashinfer.topk_varlen.topk_varlen import _n_is_256bit_aligned

    # bf16 itemsize 2 -> 256-bit needs N % 16 == 0.
    assert _n_is_256bit_aligned(torch.bfloat16, 4096)
    assert not _n_is_256bit_aligned(torch.bfloat16, 4104)  # %16 == 8
    # fp32 itemsize 4 -> 256-bit needs N % 8 == 0.
    assert _n_is_256bit_aligned(torch.float32, 8192)
    assert not _n_is_256bit_aligned(torch.float32, 8196)  # %8 == 4


@requires_blackwell
def test_lb_256bit_misaligned_no_crash():
    """LB on N=4104 bf16 (16B-aligned, NOT 32B) runs correctly, not fault.

    Regression for a latent bug: the LB kernel defaulted to 256-bit loads (32B
    alignment) for all dtypes, faulting on 16B-but-not-32B-aligned N. auto() now
    gates 256-bit off for such N and the 128-bit path runs correctly.
    """
    top_k, N, batch_size = 512, 4104, 16
    assert N % 8 == 0 and (N * 2) % 32 != 0  # 128-bit OK, 256-bit would fault
    torch.manual_seed(31)
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2).to(
        torch.bfloat16
    )
    seq_lens = torch.tensor(
        [N] * 4 + [2048] * (batch_size - 4), dtype=torch.int32, device="cuda"
    )
    lf = logits.float()
    pre_idx = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
    for r in range(batch_size):
        pre_idx[r, 0] = int(lf[r, : int(seq_lens[r])].argmax().item())
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")

    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr", load_balance=True
    )
    torch.cuda.synchronize()  # would surface a misaligned-address fault
    _check_correct(indices, logits, seq_lens, top_k)


@requires_blackwell
def test_auto_gvr_knobs_shape_aware():
    """auto() picks a shape-appropriate config: large-N fp32 small-batch -> 1024
    threads + 256-bit + low min_blocks (vs the frozen 512/mb3 old default)."""
    from flashinfer.topk_varlen.topk_varlen import _auto_gvr_knobs

    logits = torch.randn(8, 131072, dtype=torch.float32, device="cuda")
    num_threads, knobs = _auto_gvr_knobs(logits, is_lb=False)
    assert num_threads == 1024
    assert knobs["use_256bit_load"] is True  # fp32, N>=16384, 32B-aligned
    assert knobs["min_blocks_per_mp"] <= 1


# ---------------------------------------------------------------------------
# radix (CuTe DSL) backend — Blackwell only
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
@pytest.mark.parametrize("batch_size", [1, 8])
def test_radix_basic(dtype, top_k, batch_size):
    """radix (CuTe DSL) single-CTA correctness across dtype/K/batch."""
    N = 8192  # < max_chunk for all dtypes -> single-CTA
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=42)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix")
    torch.cuda.synchronize()
    assert indices.shape == (batch_size, top_k)
    assert indices.dtype == torch.int32
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size",
    [
        # SMEM-forced split — the N=131072 shared-memory-overflow regression.
        (torch.bfloat16, 1024, 131072, 64),
        # Small-batch fan-out: one row split across many CTAs to fill the machine.
        (torch.bfloat16, 1024, 65536, 1),
        # fp32 has a smaller max_chunk (57536), so N=65536 forces a split too.
        (torch.float32, 2048, 65536, 32),
        (torch.float32, 2048, 131072, 32),
    ],
)
def test_radix_multi_cta_regime(dtype, top_k, N, batch_size):
    """radix multi-CTA path (ctas_per_group > 1): SMEM split + small-batch fan-out.

    This is the coverage the perf work most needs: the single-CTA-only path
    faulted on rows too large for shared memory (N=131072), and the multi-CTA
    split + global-histogram merge had no committed correctness test.
    """
    ctas = _radix_ctas(N, dtype, batch_size)
    assert ctas > 1, (
        f"expected multi-CTA, got ctas_per_group={ctas} for N={N} batch={batch_size}"
    )
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=101)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix")
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [512, 1024])
def test_radix_next_n(dtype, top_k):
    """radix backend: next_n=2 (two rows share one seq_len entry)."""
    next_n, N, batch_size = 2, 8192, 8
    if N - next_n + 1 < top_k:
        pytest.skip("N_eff < top_k")
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(num_rows, N, top_k, dtype, seed=7, next_n=next_n)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, next_n=next_n, backend="radix"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@requires_blackwell
@pytest.mark.parametrize("top_k,next_n", [(512, 1), (1024, 1), (512, 2), (1024, 2)])
def test_radix_compress_ratio(top_k, next_n):
    """radix backend: compress_ratio=4, varied top_k and next_n.

    Tests both axes independently covered by test_radix_compress_ratio and
    test_radix_next_n, plus their interaction (next_n > 1 with compress_ratio > 1)
    which is where _run_radix's kernel length formula must apply compress_ratio
    after the next_n adjustment, not before.
    """
    dtype, compress_ratio, N, batch_size = torch.bfloat16, 4, 4096, 8
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=55, next_n=next_n, compress_ratio=compress_ratio
    )
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        next_n=next_n,
        compress_ratio=compress_ratio,
        backend="radix",
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(
        indices, logits, seq_lens, top_k, next_n=next_n, compress_ratio=compress_ratio
    )


@requires_blackwell
def test_radix_next_n_compress_ratio():
    """radix backend: next_n=2 combined with compress_ratio=4.

    Regression test: the next_n per-row adjustment (in token units) must happen
    before dividing by compress_ratio, not after. Pre-dividing seq_lens and then
    subtracting next_n in compressed-index units gives the wrong column bound
    (off by up to compress_ratio-1 columns per row).
    """
    dtype, top_k, next_n, compress_ratio = torch.bfloat16, 512, 2, 4
    N, batch_size = 4096, 8
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=17, next_n=next_n, compress_ratio=compress_ratio
    )
    # _make_inputs fills seq_lens with N*compress_ratio (16384), which is divisible
    # by compress_ratio: there adjust-before-divide and the buggy divide-before-
    # adjust agree (both give 4095), so the regression would slip through. Override
    # to N*compress_ratio + 1 (16385), where the two orders diverge:
    # (16385-2+0+1)//4 = 4096 (correct, all N columns) vs
    # (16385//4)-2+0+1 = 4095 (buggy, one column short).
    seq_lens = torch.full_like(seq_lens, N * compress_ratio + 1)
    # Make the last column a guaranteed top-1 value so the off-by-one is
    # observable. Correct adjust-before-divide gives N_eff = N for every row, so
    # column N-1 is in range and must be selected. The buggy divide-before-adjust
    # gives N_eff = N-1 for the ofs=0 rows, dropping column N-1 from their search
    # window — so its absence from the selected indices flags the regression
    # deterministically (a k-th-value check misses it here: at seed=17 no
    # divergence row otherwise places column N-1 in the top-k).
    logits[:, N - 1] = logits.float().abs().max().item() + 1.0
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        next_n=next_n,
        compress_ratio=compress_ratio,
        backend="radix",
    )
    torch.cuda.synchronize()
    assert indices.shape == (num_rows, top_k)
    _check_correct(
        indices, logits, seq_lens, top_k, next_n=next_n, compress_ratio=compress_ratio
    )
    # Every row's correct N_eff == N, so the boosted last column must appear in
    # every row's selected indices; a divide-before-adjust bound would drop it
    # from the ofs=0 rows.
    assert (indices == (N - 1)).any(dim=1).all(), (
        "column N-1 (a guaranteed top-1 value) is missing from some row's "
        "top-k — the next_n adjustment was applied after the compress_ratio "
        "divide (divide-before-adjust regression)"
    )


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_return_values(dtype):
    """radix backend: returned values equal logits[row, indices]."""
    top_k, N, batch_size = 512, 8192, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=13)
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=None, return_values=True, backend="radix"
    )
    torch.cuda.synchronize()
    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype  # auto-allocated values keep the logits dtype
    lf = logits.float()
    for row in range(batch_size):
        expected = lf[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: values do not match logits[row, indices]"
        )


@requires_blackwell
@pytest.mark.parametrize("return_values", [True, False])
def test_radix_preallocated_outputs(return_values):
    """radix backend: out_indices written in-place; out_values honoured iff return_values=True.

    Covers the return_values=False + out_values-supplied case: _run_radix must pass
    None to the kernel (compiled with return_output_values=False) even when the caller
    has pre-allocated a values buffer.  Before the fix, the real tensor was forwarded
    unconditionally into a kernel compiled to expect None.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 8192, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=11)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")
    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=None,
        out_indices=out_i,
        return_values=return_values,
        out_values=out_v,
        backend="radix",
    )
    torch.cuda.synchronize()
    assert ret_i is out_i
    if return_values:
        assert ret_v is out_v
    else:
        assert ret_v is None
    _check_correct(out_i, logits, seq_lens, top_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "backend,load_balance",
    [
        ("radix", None),
        ("radix_primitives", None),
        ("radix_cutlass", None),
        ("gvr", True),
        ("gvr", False),
        ("gvr_2", None),
    ],
)
def test_out_values_ignored_when_return_values_false(backend, load_balance):
    """out_values supplied but return_values=False must not corrupt the kernel call.

    _compile_radix specialises the kernel on return_output_values: when False the
    compiled signature has None for the values slot.  Passing a real tensor there
    (without the 'out_values if return_output_values else None' guard) causes a
    type mismatch.  Covers all backends plus both GVR load-balance paths.
    """
    _skip_unless_backend(backend)
    # gvr_2 is fp32-only; the other backends keep the original bf16 coverage.
    dtype = torch.float32 if backend == "gvr_2" else torch.bfloat16
    top_k, N, batch_size = 512, 8192, 4
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=99)
    # Pre-allocate a values buffer but deliberately do NOT set return_values=True.
    out_v = torch.full((batch_size, top_k), float("nan"), dtype=dtype, device="cuda")

    kwargs = dict(return_values=False, out_values=out_v, backend=backend)
    if backend == "gvr":
        kwargs["pre_idx"] = pre_idx
        kwargs["load_balance"] = load_balance
    elif backend == "gvr_2":
        kwargs["pre_idx"] = pre_idx

    ret_i, ret_v = flashinfer.top_k_varlen(logits, seq_lens, top_k, **kwargs)
    torch.cuda.synchronize()
    # return_values=False → second element must be None regardless of out_values.
    assert ret_v is None
    # Indices must still be correct.
    _check_correct(ret_i, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# Variable-length (true varlen) + degenerate seq_len coverage
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("backend", ["radix", "radix_primitives", "radix_cutlass"])
def test_varlen_ragged(backend):
    """Distinct per-row seq_lens: every row is masked to its own length.

    ``_make_inputs`` uses a uniform length, so this is the primary test of the
    varlen masking that ``top_k_varlen`` exists for. All rows are >= top_k so
    ``require_all_checked`` verifies every one.
    """
    _skip_unless_backend(backend)
    dtype, top_k, N = torch.bfloat16, 512, 8192
    seq_len_list = [top_k, top_k + 1, 1024, 2048, 4096, 6000, 8000, N]
    logits, seq_lens = _make_varlen_inputs(seq_len_list, N, dtype, seed=88)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend=backend)
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("backend", ["radix", "radix_primitives", "radix_cutlass"])
def test_seq_len_equals_top_k(backend):
    """Degenerate seq_len == top_k: the top-K is exactly all valid indices [0, top_k)."""
    _skip_unless_backend(backend)
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, seq_lens = _make_varlen_inputs([top_k] * batch_size, N, dtype, seed=64)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend=backend)
    torch.cuda.synchronize()
    for row in range(batch_size):
        sel = set(int(i) for i in indices[row].cpu().tolist() if i >= 0)
        assert sel == set(range(top_k)), (
            f"row={row}: seq_len==top_k must select all [0,{top_k}); got {len(sel)} unique"
        )


# ---------------------------------------------------------------------------
# CUDA graph capture / replay
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_graph_radix_multi_cta():
    """radix multi-CTA under CUDA graph capture/replay.

    Specifically exercises the row_states zero-init + kernel self-reset
    guardrail: a second replay with *fresh* input data must stay correct, i.e.
    the inter-CTA arrival counter must not carry stale state across replays.
    """
    _skip_unless_backend("radix")
    dtype, top_k, N, batch_size = torch.bfloat16, 1024, 131072, 8
    assert _radix_ctas(N, dtype, batch_size) > 1  # ensure the multi-CTA path
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2).to(
        dtype
    )
    seq_lens = torch.full((batch_size,), N, dtype=torch.int32, device="cuda")
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix", out_indices=out_i
        )

    # Warmup on a side stream (JIT compile + row_states alloc) before capture,
    # so capture itself performs no allocation.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)

    # Overwrite the captured input buffer with fresh data and replay again.
    # Zeroing out_i first means a no-op replay (or stale row_states) would leave
    # zeros and fail the check — so passing proves the kernel truly re-executes.
    fresh = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 3).to(
        dtype
    )
    logits.copy_(fresh)
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


@requires_blackwell
@pytest.mark.parametrize("load_balance", [False, True])
def test_cuda_graph_gvr(load_balance):
    """GVR under CUDA graph capture/replay — both single-CTA and LB paths.

    ``load_balance=True`` is the documented default whose docstring promises
    CUDA-graph safety; it runs the two-kernel prepare+main path with device-side
    counters/order_row. A ragged batch (long + short rows) exercises both LB
    branches. Zeroing ``out_i`` before the second replay proves the kernel
    re-executes rather than passing on stale warmup output.
    """
    top_k = 512
    logits, seq_lens, pre_idx = _make_ragged_gvr_inputs(top_k)
    batch_size = seq_lens.shape[0]
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            pre_idx=pre_idx,
            backend="gvr",
            load_balance=load_balance,
            out_indices=out_i,
        )

    # Warmup on a side stream so the first LB allocation (order_row / counters)
    # happens outside capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)

    # Zero the output and replay again on the same inputs: a no-op replay (or a
    # counter/order_row that carried stale state) would leave zeros and fail.
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


# ---------------------------------------------------------------------------
# Auto-selection, cross-backend consistency, and input validation
# ---------------------------------------------------------------------------


def test_backend_heuristic_priority():
    """Auto-selection is shape/dtype-aware and tracks the measured winners.

    Hardware-independent: exercises the heuristic directly with meta tensors
    (only .dtype/.shape are read) so a regression in the decision rules is
    caught even off-GPU. The boundaries are grounded in the B200 sweep
    documented on the heuristic itself.
    """
    from flashinfer.topk_varlen.topk_varlen import _top_k_varlen_heuristic

    def order(suitable, dtype, batch, n_cols):
        logits = torch.empty(batch, n_cols, dtype=dtype, device="meta")
        seq_lens = torch.empty(batch, dtype=torch.int32, device="meta")
        return _top_k_varlen_heuristic(suitable, logits, seq_lens, 1024)

    all4 = ["radix_cutlass", "radix", "gvr_2", "gvr"]  # unordered on purpose

    # fp32 + hint: gvr_2 always first; small problems rank radix over gvr.
    assert order(all4, torch.float32, 1, 8192) == [
        "gvr_2",
        "radix",
        "gvr",
        "radix_cutlass",
    ]
    # fp32 large batch x long rows: gvr ahead of radix; the fp32 big corner
    # (N >= 64K and B*N >= 2^23) ranks radix_cutlass over radix.
    assert order(all4, torch.float32, 256, 131072) == [
        "gvr_2",
        "gvr",
        "radix_cutlass",
        "radix",
    ]
    # B*N = 2^23 but N < 64K: gvr first, radix over radix_cutlass.
    assert order(all4, torch.float32, 256, 32768) == [
        "gvr_2",
        "gvr",
        "radix",
        "radix_cutlass",
    ]
    # bf16 (gvr_2 never suitable): radix wins everywhere below B*N = 2^23...
    assert order(["gvr", "radix", "radix_cutlass"], torch.bfloat16, 64, 65536) == [
        "radix",
        "gvr",
        "radix_cutlass",
    ]
    # ...and gvr only above it; radix_cutlass never leads in half precision.
    assert order(["gvr", "radix", "radix_cutlass"], torch.bfloat16, 256, 131072) == [
        "gvr",
        "radix",
        "radix_cutlass",
    ]
    # no-hint fallbacks
    assert order(["radix", "radix_cutlass"], torch.bfloat16, 256, 131072) == [
        "radix",
        "radix_cutlass",
    ]
    assert order(["radix", "radix_cutlass"], torch.float32, 256, 131072) == [
        "radix_cutlass",
        "radix",
    ]
    assert order(["radix_cutlass"], torch.float32, 1, 4096) == ["radix_cutlass"]

    # radix_filter admission (auto-vs-oracle study, PR #4811): hint-free fp32
    # from 32K columns up, except the single-row case at >= 512K, and at every
    # N once B >= 256 (ahead of the fp32 radix_cutlass corner).
    hf = ["radix_cutlass", "radix", "radix_filter"]
    assert order(hf, torch.float32, 16, 32768) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.float32, 16, 8192) == ["radix", "radix_cutlass"]
    assert order(hf, torch.float32, 1, 65536) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.float32, 1, 524288) == ["radix", "radix_cutlass"]
    assert order(hf, torch.float32, 256, 8192) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.float32, 256, 131072) == [
        "radix_filter",
        "radix_cutlass",
        "radix",
    ]
    # fp32 with a hint but gvr_2 unsuitable: radix_filter ranks ahead of gvr.
    assert order(hf + ["gvr"], torch.float32, 64, 131072)[:2] == ["radix_filter", "gvr"]
    # half precision: gvr only for B >= 256 with 32K-512K columns (the old
    # B*N >= 2^23 rule picked it at B=16 x 2M, a 7x loss to radix); radix_filter
    # in the mid band for small batches and from 128K up for B >= 64.
    hh = ["gvr", "radix", "radix_cutlass", "radix_filter"]
    assert order(hh, torch.bfloat16, 16, 2097152) == ["radix", "gvr", "radix_cutlass"]
    assert order(hh, torch.bfloat16, 256, 131072) == [
        "gvr",
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hh, torch.bfloat16, 256, 32768) == ["gvr", "radix", "radix_cutlass"]
    assert order(hh, torch.bfloat16, 64, 524288) == [
        "radix_filter",
        "radix",
        "gvr",
        "radix_cutlass",
    ]
    assert order(hf, torch.float16, 16, 65536) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    assert order(hf, torch.bfloat16, 64, 8192) == ["radix", "radix_cutlass"]
    assert order(hf, torch.bfloat16, 256, 8192) == [
        "radix_filter",
        "radix",
        "radix_cutlass",
    ]
    # hint-free fp32 (gvr_2 runs its hint-free engines): gvr_2 still first
    # everywhere except the one measured loss — K >= 2048, N <= 4096, single
    # row — where radix_filter leads and gvr_2 follows; a real hint restores
    # gvr_2 to the front there.
    all5 = all4 + ["radix_filter"]

    def order_k(suitable, batch, n_cols, top_k, hinted):
        logits = torch.empty(batch, n_cols, dtype=torch.float32, device="meta")
        seq_lens = torch.empty(batch, dtype=torch.int32, device="meta")
        pre_idx = (
            torch.empty(batch, top_k, dtype=torch.int32, device="meta")
            if hinted
            else None
        )
        return _top_k_varlen_heuristic(suitable, logits, seq_lens, top_k, pre_idx)

    assert order_k(all5, 16, 32768, 512, False)[0] == "gvr_2"
    assert order_k(all5, 1, 4096, 1024, False)[0] == "gvr_2"
    assert order_k(all5, 2, 4096, 2048, False)[0] == "gvr_2"
    assert order_k(all5, 1, 4096, 2048, False)[:2] == ["radix_filter", "gvr_2"]
    # N <= top_k: every row is the identity answer; radix_filter emits it
    # cheaper than gvr_2's short-path launch, with or without a hint
    assert order_k(all5, 16, 2048, 2048, False)[:2] == ["radix_filter", "gvr_2"]
    assert order_k(all5, 16, 2048, 2048, True)[:2] == ["radix_filter", "gvr_2"]
    assert order_k(all5, 4, 1000, 1024, True)[:2] == ["radix_filter", "gvr_2"]
    assert order_k(all5, 4, 1025, 1024, True)[0] == "gvr_2"
    assert order_k(all4, 1, 4096, 2048, False)[0] == "gvr_2"  # no radix_filter
    # (a meta pre_idx fails the CUDA-device check and counts as absent; a hint
    # on the logits device is exercised by the GPU tests)
    assert order_k(all5, 1, 4096, 2048, True)[:2] == ["radix_filter", "gvr_2"]

    # None tensors (skip_check / doc examples): static fallback order.
    assert _top_k_varlen_heuristic(all4, None, None, None) == [
        "gvr_2",
        "gvr",
        "radix",
        "radix_cutlass",
    ]


@requires_blackwell
def test_cross_backend_value_consistency():
    """radix, radix_cutlass, gvr, and gvr_2 select the same top-K *value* multiset.

    Compares sorted selected values (not indices) so ties don't cause spurious
    failures. fp32 keeps ties rare; any real divergence between backends fails.
    """
    dtype, top_k, N, batch_size = torch.float32, 1024, 8192, 8
    logits, pre_idx, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=123)
    idx_r, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix")
    idx_p, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    idx_c, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, backend="radix_cutlass")
    idx_g, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr"
    )
    idx_g2, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr_2"
    )
    torch.cuda.synchronize()
    lf = logits.float()
    for row in range(batch_size):
        vr = lf[row][idx_r[row].long()].sort(descending=True).values
        vp = lf[row][idx_p[row].long()].sort(descending=True).values
        assert torch.allclose(vr, vp, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs radix_primitives value multisets differ"
        )
        vc = lf[row][idx_c[row].long()].sort(descending=True).values
        vg = lf[row][idx_g[row].long()].sort(descending=True).values
        vg2 = lf[row][idx_g2[row].long()].sort(descending=True).values
        assert torch.allclose(vr, vc, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs radix_cutlass value multisets differ"
        )
        assert torch.allclose(vr, vg, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs gvr value multisets differ"
        )
        assert torch.allclose(vr, vg2, rtol=1e-4, atol=1e-4), (
            f"row={row}: radix vs gvr_2 value multisets differ"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("N,batch", [(262144, 1), (262144, 8), (1048576, 64)])
@pytest.mark.parametrize("hint_kind", ["identity", "random"])
def test_walkfirst_garbage_hint_no_fallback(N, batch, hint_kind):
    """A garbage pre_idx must never send a row to the exact fallback: the
    walk-first kernel is hint-free and ignores pre_idx, so the call is
    identical to the hintless one.  (An earlier hint rung replaced the sample,
    and identity hints on the multi-CTA forms overflowed staging: 6-8x slower
    at 1M.)  Checks exactness and that no row reports the fallback path
    (status block 0)."""
    from flashinfer.topk_varlen.topk_varlen import _prim_status

    _skip_unless_backend("walkfirst_primitives")
    top_k = 512 if batch == 1 else 1024
    torch.manual_seed(N // 1024 + batch)
    logits = (torch.randn(batch, N, device="cuda") * 2.0).contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    if hint_kind == "identity":
        pre_idx = (
            torch.arange(top_k, dtype=torch.int32, device="cuda")
            .expand(batch, top_k)
            .contiguous()
        )
    else:
        pre_idx = torch.randint(0, N, (batch, top_k), dtype=torch.int32, device="cuda")
    out = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        out_indices=out,
        pre_idx=pre_idx,
        backend="walkfirst_primitives",
    )
    torch.cuda.synchronize()
    ref = torch.topk(logits, top_k, dim=1).values
    got = torch.sort(logits.gather(1, out.long()), dim=1, descending=True).values
    assert torch.equal(got, ref), "garbage hint changed the selected value multiset"
    status = _prim_status(batch, logits.device)
    fallback_rows = int((status[:batch] != 0).sum())
    assert fallback_rows == 0, f"{fallback_rows}/{batch} rows took the exact fallback"


def _gvr2_check_complete_exact(out, logits, kv, top_k, ref_vals, sentinel):
    """Every output slot written and the selected value multiset equals the
    reference (rows masked to their own length)."""
    torch.cuda.synchronize()
    assert int((out == sentinel).sum()) == 0, "unwritten output slots"
    for r in range(out.shape[0]):
        idx = out[r].long()
        assert bool((idx >= 0).all()) and bool((idx < int(kv[r])).all()), (
            f"row={r}: index outside the valid window"
        )
        assert idx.numel() == torch.unique(idx).numel(), f"row={r}: duplicate indices"
        got = logits[r][idx].sort(descending=True).values
        assert torch.equal(got, ref_vals[r]), f"row={r}: value multiset differs"


@requires_blackwell
@pytest.mark.parametrize("n_valid", [3072, 4096], ids=["n3072", "n4096"])
def test_gvr2_high_anchor_hint_completeness(n_valid):
    """Port of TensorRT-LLM PR #18501's regression: anchor-only hints whose
    gathered values all sit ABOVE the true k-th value (an argmax anchor over
    the all-zero cold-start buffer, with row[0] = second-max) bracket the
    sampling band so it holds fewer than top_k entries.  The register-family
    kernel must then escape to the key-space ranking instead of stopping at
    the histogram total (pre-fix: out[tot:k) left unwritten -- 130,304 of
    131,072 slots per cell here)."""
    top_k, bs = 512, 256
    gen = torch.Generator(device="cuda").manual_seed(top_k + n_valid)
    logits = torch.randn(
        (bs, n_valid), generator=gen, dtype=torch.float32, device="cuda"
    )
    logits[:, 0] = torch.topk(logits, 2, dim=1).values[:, 1]
    ref_vals = torch.topk(logits, top_k, dim=1).values
    pre_idx = torch.zeros((bs, top_k), dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = logits.argmax(dim=1).to(torch.int32)
    kv = torch.full((bs,), n_valid, dtype=torch.int32, device="cuda")
    out = torch.full((bs, top_k), -7, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits, kv, top_k, out_indices=out, pre_idx=pre_idx, backend="gvr_2"
    )
    _gvr2_check_complete_exact(out, logits, kv, top_k, ref_vals, -7)


@requires_blackwell
def test_gvr2_neginf_tail_completeness():
    """Port of TensorRT-LLM PR #18501's second regression: an in-window -inf in
    the row's tail column (n_valid % 4 == 1) drags the hint-free bracket to
    -inf, every classify product becomes NaN and the histogram total is zero
    (pre-fix: whole rows unwritten).  Odd rows keep fewer than top_k finite
    entries so the -inf tie class exercises the escape's fill-lane bound
    (pre-fix: duplicate indices)."""
    top_k, bs, npad, n_valid = 1024, 256, 4096, 4093
    gen = torch.Generator(device="cuda").manual_seed(top_k + n_valid)
    logits = torch.randn((bs, npad), generator=gen, dtype=torch.float32, device="cuda")
    logits[:, n_valid:] = 3e38  # poison past the window
    logits[:, n_valid - 1] = float("-inf")  # in-window -inf in the tail column
    logits[1::2, 500:n_valid] = float("-inf")  # odd rows: n_finite < top_k
    masked = logits.clone()
    masked[:, n_valid:] = float("-inf")
    ref_vals = torch.topk(masked, top_k, dim=1).values
    pre_idx = torch.zeros((bs, top_k), dtype=torch.int32, device="cuda")
    kv = torch.full((bs,), n_valid, dtype=torch.int32, device="cuda")
    out = torch.full((bs, top_k), -7, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits, kv, top_k, out_indices=out, pre_idx=pre_idx, backend="gvr_2"
    )
    _gvr2_check_complete_exact(out, logits, kv, top_k, ref_vals, -7)


@requires_blackwell
def test_gvr2_plus_inf_selected():
    """A single +inf must be in the top-k (finite + inf inputs are inside the
    kernel's exactness contract).  Was a strict xfail (DKG issue #58) until the
    TensorRT-LLM #18625 port landed upstream with the gvr_2 backend."""
    top_k, n = 1024, 4096
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((1, n), generator=gen, dtype=torch.float32, device="cuda")
    logits[0, 17] = float("inf")
    kv = torch.full((1,), n, dtype=torch.int32, device="cuda")
    pre_idx = (
        torch.arange(top_k, dtype=torch.int32, device="cuda").view(1, -1).contiguous()
    )
    out, _ = flashinfer.top_k_varlen(
        logits, kv, top_k, pre_idx=pre_idx, backend="gvr_2"
    )
    torch.cuda.synchronize()
    assert bool((out == 17).any()), "+inf entry missing from the top-k"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "backend",
    [
        "radix",
        "radix_cutlass",
        "radix_filter",
        "gvr",
        "gvr_2",
        "radix_primitives",
        "cutlass_primitives",
    ],
)
def test_seq_len_below_next_n_all_backends(backend):
    """seq_len < next_n - t makes ``seq_len - next_n + t + 1`` negative.

    Padded / evicted requests reach every backend with such rows. Each must
    treat them as empty (all -1) and stay inside its own row: a negative
    length is neither an unsigned count (radix_cutlass) nor a write offset
    before the row (radix / gvr identity epilogues). The caller buffer is
    over-allocated and pre-filled with a sentinel so any write that lands
    outside a row is detected, and full rows preceding the empty rows are
    checked exactly so a write into a neighbour's tail is caught too.
    """
    _skip_unless_backend(backend)
    if backend == "radix_filter":
        # is_backend_supported() is static (registration + CC lists) and stays
        # True on Blackwell even when the installed nvidia-cutlass-dsl is < 4.8,
        # where the vendored radix_filter kernels cannot compile and the API's
        # fail-closed probe rejects every call ("Problem size is not supported").
        # Gate on the same dynamic probe test_radix_filter.py uses: skip, not fail.
        from flashinfer.topk_varlen.topk_varlen import _radix_filter_kernel_dsl_ok

        if not _radix_filter_kernel_dsl_ok():
            pytest.skip("radix_filter requires nvidia-cutlass-dsl >= 4.8")
    top_k, N, next_n = 512, 8192, 3
    # request seq_lens: three that make some numerators negative
    # (0 -> -2,-1,0; 1 -> -1,0,1; 2 -> 0,1,2), one straddling top_k, two full.
    # The zero-length request goes FIRST: an unclamped kernel writes row 0's
    # `-1` padding from offset -2, i.e. into the guard row, where nothing
    # overwrites it. Behind a full row the same stray writes land in that
    # row's tail and are racily repaired by the full row's own kernel.
    req_lens = [0, 1, 2, top_k + 2, N, N]
    num_req = len(req_lens)
    num_rows = num_req * next_n
    torch.manual_seed(7)
    logits = torch.randn(num_rows, N, dtype=torch.float32, device="cuda")
    seq_lens = torch.tensor(req_lens, dtype=torch.int32, device="cuda")
    pre_idx = torch.arange(top_k, dtype=torch.int32, device="cuda").repeat(num_req, 1)
    sentinel = 0x7EADBEEF
    # one guard row before and after the caller's buffer
    arena = torch.full(
        (num_rows + 2, top_k), sentinel, dtype=torch.int32, device="cuda"
    )
    out = arena[1 : num_rows + 1]
    kwargs = {"backend": backend, "next_n": next_n, "out_indices": out}
    if backend in ("gvr", "gvr_2"):
        kwargs["pre_idx"] = pre_idx
    idx, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, **kwargs)
    torch.cuda.synchronize()
    assert idx.data_ptr() == out.data_ptr()
    assert (arena[0] == sentinel).all() and (arena[-1] == sentinel).all(), (
        "write landed outside the caller's buffer"
    )
    arena_cpu = arena.cpu()
    for r in range(num_rows):
        length = max(0, req_lens[r // next_n] - next_n + (r % next_n) + 1)
        row = arena_cpu[r + 1].tolist()
        assert sentinel not in row, f"row={r}: slot never written"
        valid = [i for i in row if i >= 0]
        assert row.count(-1) == top_k - len(valid), f"row={r}: bad -1 padding"
        if length <= top_k:
            assert sorted(valid) == list(range(length)), (
                f"row={r}: length={length} must select exactly [0,{length}); "
                f"got {len(valid)} indices"
            )
        else:
            assert len(valid) == top_k and max(valid) < length, (
                f"row={r}: index past length={length}"
            )
            got = logits[r][torch.tensor(valid, device="cuda")].sort().values
            ref = logits[r, :length].topk(top_k).values.sort().values
            assert torch.equal(got, ref), f"row={r}: wrong top-k value set"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "backend", ["radix_primitives", "cutlass_primitives", "sglang", "radix_cutlass"]
)
def test_grouped_rows_non_divisible_compress_ratio(backend):
    """next_n=3 with compress_ratio=4 on token counts that are NOT multiples
    of 4: row t of request q ranks exactly max(0, (seq_len - next_n + t + 1)
    // 4) columns.  Dividing before the next_n adjustment, or truncating a
    negative numerator toward zero, moves some row's bound by one column
    here, which the fully divisible lengths of ``_make_inputs`` cannot show.
    Sentinel arena and padding checks as in
    test_seq_len_below_next_n_all_backends; values are compared as sorted
    multisets against torch.topk of the row's valid prefix.
    """
    _skip_unless_backend(backend)
    top_k, N, next_n, cr = 512, 8192, 3, 4
    req_lens = [0, 1, 5, 4 * 2000 + 1, 4 * top_k + 1, 4 * top_k + 6, 4 * N - 3, 4 * N]
    num_req = len(req_lens)
    num_rows = num_req * next_n
    gen = torch.Generator(device="cuda").manual_seed(34)
    logits = torch.randn(num_rows, N, dtype=torch.float32, device="cuda", generator=gen)
    seq_lens = torch.tensor(req_lens, dtype=torch.int32, device="cuda")
    sentinel = 0x7EADBEEF
    arena = torch.full(
        (num_rows + 2, top_k), sentinel, dtype=torch.int32, device="cuda"
    )
    out = arena[1 : num_rows + 1]
    idx, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        backend=backend,
        next_n=next_n,
        compress_ratio=cr,
        out_indices=out,
    )
    torch.cuda.synchronize()
    assert idx.data_ptr() == out.data_ptr()
    assert (arena[0] == sentinel).all() and (arena[-1] == sentinel).all(), (
        "write landed outside the caller's buffer"
    )
    arena_cpu = arena.cpu()
    for r in range(num_rows):
        length = min(
            N, max(0, (req_lens[r // next_n] - next_n + (r % next_n) + 1) // cr)
        )
        kk = min(top_k, length)
        row = arena_cpu[r + 1].tolist()
        assert sentinel not in row, f"row={r}: slot never written"
        valid = sorted(i for i in row if i >= 0)
        assert row.count(-1) == top_k - kk, f"row={r}: length={length}: bad -1 padding"
        assert len(valid) == kk == len(set(valid)), (
            f"row={r}: length={length}: {len(valid)} indices"
        )
        if kk == 0:
            continue
        assert valid[-1] < length, f"row={r}: index past length={length}"
        got = logits[r][torch.tensor(valid, device="cuda")].sort().values
        ref = logits[r, :length].topk(kk).values.sort().values
        assert torch.equal(got, ref), f"row={r}: length={length}: wrong value multiset"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_unknown_backend_rejected():
    """Unregistered backend names — including the pre-rename 'radix_cutedsl' — raise.

    Matches the specific rejection error (not a bare Exception) so an unrelated
    failure — OOM, a missing dependency, an input assertion — cannot satisfy it.
    """
    from flashinfer.utils import BackendSupportedError

    dtype, top_k, N, batch_size = torch.bfloat16, 512, 4096, 4
    logits, _, seq_lens = _make_inputs(batch_size, N, top_k, dtype, seed=5)
    for bad in ("radix_cutedsl", "not_a_backend"):
        with pytest.raises((BackendSupportedError, ValueError), match=bad):
            flashinfer.top_k_varlen(logits, seq_lens, top_k, backend=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_input_validation():
    """1-D logits and non-int32 seq_lens are rejected with ValueErrors (real
    exceptions with a message, so the checks also hold under ``python -O``)."""
    top_k = 512
    logits = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.full((4,), 4096, dtype=torch.int32, device="cuda")
    # logits must be 2-D
    with pytest.raises(ValueError, match="2-D CUDA"):
        flashinfer.top_k_varlen(logits[0], seq_lens[:1], top_k)
    # seq_lens must be int32
    with pytest.raises(ValueError, match="int32"):
        flashinfer.top_k_varlen(logits, seq_lens.long(), top_k)


def _malformed_hints(batch, top_k):
    dev = "cuda"
    return {
        "transposed": torch.zeros(top_k, batch, dtype=torch.int32, device=dev).t(),
        "wrong_batch": torch.zeros(batch // 2, top_k, dtype=torch.int32, device=dev),
        "wrong_width": torch.zeros(batch, top_k // 2, dtype=torch.int32, device=dev),
        "int64": torch.zeros(batch, top_k, dtype=torch.int64, device=dev),
        "cpu": torch.zeros(batch, top_k, dtype=torch.int32),
        "misaligned": torch.zeros(batch * top_k + 4, dtype=torch.int32, device=dev)[
            1 : 1 + batch * top_k
        ].view(batch, top_k),
        "three_d": torch.zeros(batch, top_k, 1, dtype=torch.int32, device=dev),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "kind",
    [
        "transposed",
        "wrong_batch",
        "wrong_width",
        "int64",
        "cpu",
        "misaligned",
        "three_d",
    ],
)
def test_malformed_hint_is_discarded_with_warning(kind):
    """A malformed ``pre_idx`` (wrong shape, dtype, device, layout or
    alignment) is dropped with a RuntimeWarning and the call runs hint-free
    and exact, under ``auto``, under an explicit hint-free backend, and under
    ``gvr_2`` (which runs its hint-free engines). ``gvr`` cannot run
    without a hint and refuses the call up front instead of failing inside
    the kernel."""
    from flashinfer.utils import BackendSupportedError

    batch, n, top_k = 4, 8192, 1024
    torch.manual_seed(11)
    logits = torch.randn(batch, n, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), n, dtype=torch.int32, device="cuda")
    bad = _malformed_hints(batch, top_k)[kind]
    ref = torch.sort(torch.topk(logits, top_k, dim=1).values, dim=1).values
    with pytest.warns(RuntimeWarning, match="pre_idx"):
        indices, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=bad)
    assert bool(((indices >= 0) & (indices < n)).all()), "index out of range"
    assert torch.equal(torch.sort(logits.gather(1, indices.long()), dim=1).values, ref)
    # explicit hint-free backend (radix on Blackwell+, radix_cutlass elsewhere):
    # the hint is dropped with the same warning and the result stays exact
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    hint_free = (
        "radix"
        if flashinfer.top_k_varlen.is_backend_supported("radix", cc)
        else "radix_cutlass"
    )
    with pytest.warns(RuntimeWarning, match="pre_idx"):
        indices, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, pre_idx=bad, backend=hint_free
        )
    assert bool(((indices >= 0) & (indices < n)).all()), "index out of range"
    assert torch.equal(torch.sort(logits.gather(1, indices.long()), dim=1).values, ref)
    # gvr_2: the hint is dropped with the warning and the call runs exact on
    # the hint-free engines (checked and skip_check paths alike)
    if flashinfer.top_k_varlen.is_backend_supported("gvr_2", cc):
        for skip in (False, True):
            with pytest.warns(
                RuntimeWarning, match="falls back to its hint-free engines"
            ):
                indices, _ = flashinfer.top_k_varlen(
                    logits,
                    seq_lens,
                    top_k,
                    pre_idx=bad,
                    backend="gvr_2",
                    skip_check=skip,
                )
            assert torch.equal(
                torch.sort(logits.gather(1, indices.long()), dim=1).values, ref
            )
    # gvr (V1) consumes the hint: refused by its checker up front (the
    # @backend_requirement decorator reports a failed explicit-backend check
    # as ValueError("Problem size is not supported ..."))
    if flashinfer.top_k_varlen.is_backend_supported("gvr", cc):
        with pytest.raises((BackendSupportedError, ValueError), match="not supported"):
            flashinfer.top_k_varlen(logits, seq_lens, top_k, pre_idx=bad, backend="gvr")
        # skip_check=True bypasses the checkers: the body must still refuse
        # instead of handing the discarded hint (None) to the kernel host
        with (
            pytest.warns(RuntimeWarning, match="pre_idx"),
            pytest.raises(BackendSupportedError, match="well-formed"),
        ):
            flashinfer.top_k_varlen(
                logits, seq_lens, top_k, pre_idx=bad, backend="gvr", skip_check=True
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_output_buffer_contract():
    """Caller-provided ``out_indices`` / ``out_values`` must be contiguous, on
    the logits device, of the right dtype and exactly ``[num_rows, top_k]``;
    anything else is a ValueError. The gvr_2 host used to accept a wider
    buffer and pack the result into it at stride ``top_k``, so rows 0 and 1
    of the caller's buffer held two result rows each and the rest stayed
    untouched."""
    batch, n, top_k = 4, 8192, 1024
    logits = torch.randn(batch, n, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), n, dtype=torch.int32, device="cuda")
    bad = {
        "wider": torch.empty(batch, 2 * top_k, dtype=torch.int32, device="cuda"),
        "taller": torch.empty(2 * batch, top_k, dtype=torch.int32, device="cuda"),
        "short_flat": torch.empty(batch * top_k - 1, dtype=torch.int32, device="cuda"),
        # right element count, wrong 2-D shape: would be a silent re-layout
        "transposed_shape": torch.empty(top_k, batch, dtype=torch.int32, device="cuda"),
        "misaligned": torch.empty(batch * top_k + 4, dtype=torch.int32, device="cuda")[
            1 : 1 + batch * top_k
        ].view(batch, top_k),
        "int64": torch.empty(batch, top_k, dtype=torch.int64, device="cuda"),
        "non_contiguous": torch.empty(
            top_k, batch, dtype=torch.int32, device="cuda"
        ).t(),
        "cpu": torch.empty(batch, top_k, dtype=torch.int32),
    }
    for buf in bad.values():
        with pytest.raises(ValueError, match="out_indices"):
            flashinfer.top_k_varlen(logits, seq_lens, top_k, out_indices=buf)
    with pytest.raises(ValueError, match="out_values"):
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            return_values=True,
            out_values=torch.empty(batch, top_k, dtype=torch.bfloat16, device="cuda"),
        )
    # a flat buffer with exactly num_rows * top_k elements is viewed in place
    # (the contract the radix_filter in-place test relies on), for every backend
    flat_i = torch.full((batch * top_k,), -7, dtype=torch.int32, device="cuda")
    idx, _ = flashinfer.top_k_varlen(logits, seq_lens, top_k, out_indices=flat_i)
    assert idx.data_ptr() == flat_i.data_ptr() and tuple(idx.shape) == (batch, top_k)
    assert int((flat_i == -7).sum()) == 0
    good_i = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
    good_v = torch.empty(batch, top_k, dtype=torch.float32, device="cuda")
    idx, vals = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        return_values=True,
        out_indices=good_i,
        out_values=good_v,
    )
    assert idx.data_ptr() == good_i.data_ptr() and vals.data_ptr() == good_v.data_ptr()


# ---------------------------------------------------------------------------
# Coverage hardening (from the critical review): multi-CTA values, LB caps,
# degenerate short rows, radix_cutlass under CUDA graph.
# ---------------------------------------------------------------------------


@requires_blackwell
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size",
    [
        (torch.bfloat16, 1024, 131072, 64),  # SMEM-split multi-CTA
        (torch.float32, 2048, 65536, 32),  # fp32 multi-CTA
    ],
)
def test_radix_multi_cta_return_values(dtype, top_k, N, batch_size):
    """radix return_values on the multi-CTA path: the inter-CTA histogram-merge
    value-gather is otherwise unverified (single-CTA value tests don't cover it)."""
    assert _radix_ctas(N, dtype, batch_size) > 1
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=202)
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix", return_values=True
    )
    torch.cuda.synchronize()
    assert values.shape == (batch_size, top_k)
    assert values.dtype == dtype
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    lf = logits.float()
    for row in range(batch_size):
        expected = lf[row][indices[row].long()]
        assert torch.allclose(expected, values[row].float(), rtol=1e-3, atol=1e-3), (
            f"row={row}: multi-CTA values do not match logits[row, indices]"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("backend", ["radix", "radix_primitives", "radix_cutlass"])
def test_seq_len_less_than_top_k(backend):
    """Rows with seq_len < top_k: every valid index [0, seq_len) is selected.

    The two backends pad the surplus slots differently — ``radix`` writes the
    ``-1`` sentinel, ``radix_cutlass`` leaves masked-region indices (>= seq_len)
    — so this asserts the backend-agnostic guarantee (all valid entries chosen,
    unique, in-range) rather than a specific padding representation.
    """
    _skip_unless_backend(backend)
    dtype, top_k, N = torch.bfloat16, 512, 4096
    seq_len_list = [top_k - 1, top_k // 2, 17, 1]  # all strictly < top_k
    logits, seq_lens = _make_varlen_inputs(seq_len_list, N, dtype, seed=71)
    # return_values=True exercises the value-gather path. With seq_len < top_k the
    # kernel writes the -1 sentinel into surplus slots, so radix_cutlass's gather
    # must clamp the index (a raw -1 trips a device-side bounds assert) and zero
    # those slots — this combination guards that regression.
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend=backend, return_values=True
    )
    torch.cuda.synchronize()
    lf = logits.float()
    for row, sl in enumerate(seq_len_list):
        idx = indices[row]
        in_range = sorted(i for i in idx.cpu().tolist() if 0 <= i < sl)
        assert in_range == list(range(sl)), (
            f"{backend} row={row} seq_len={sl}: expected all valid indices "
            f"[0,{sl}); got {len(in_range)} unique in-range"
        )
        # Values at valid slots must equal logits[row, idx].
        valid = (idx >= 0) & (idx < sl)
        if valid.any():
            got = values[row][valid].float()
            exp = lf[row][idx[valid].long()]
            assert torch.allclose(got, exp, rtol=1e-3, atol=1e-3), (
                f"{backend} row={row}: gathered values mismatch logits[row, idx]"
            )
        # radix_cutlass zeros the -1 sentinel slots; assert it did.
        if backend == "radix_cutlass":
            sentinel = idx < 0
            if sentinel.any():
                assert (values[row][sentinel] == 0).all(), (
                    f"{backend} row={row}: sentinel (-1) value slots not zeroed"
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("next_n", [1, 2])
def test_cuda_graph_radix_cutlass(next_n):
    """radix_cutlass (the non-Blackwell auto default) under CUDA graph replay.

    Also exercises next_n>1 (the repeat_interleave/arange masking branch) under
    capture. Fresh-data replay proves re-execution.
    """
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 8192, 8
    num_rows = batch_size * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=44, next_n=next_n
    )
    out_i = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            next_n=next_n,
            backend="radix_cutlass",
            out_indices=out_i,
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(
        out_i, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )

    fresh = (torch.randn(num_rows, N, dtype=torch.float32, device="cuda") * 3).to(dtype)
    logits.copy_(fresh)
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(
        out_i, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )


def test_lb_max_batch_size_boundaries():
    """_lb_max_batch_size rounds up to the next power-of-2 cap in [64, 1024]."""
    from flashinfer.topk_varlen.topk_varlen import _lb_max_batch_size

    assert _lb_max_batch_size(1) == 64
    assert _lb_max_batch_size(64) == 64
    assert _lb_max_batch_size(65) == 128
    assert _lb_max_batch_size(256) == 256
    assert _lb_max_batch_size(512) == 512
    assert _lb_max_batch_size(1024) == 1024
    with pytest.raises(ValueError):
        _lb_max_batch_size(1025)


# ---------------------------------------------------------------------------
# radix_primitives (CuTe DSL primitives API, coarse-histogram) backend
# ---------------------------------------------------------------------------


@requires_radix_primitives
@pytest.mark.parametrize(
    "dtype,top_k",
    [
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float16, 1024),
        (torch.float32, 2048),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_radix_primitives_basic(dtype, top_k, batch_size):
    N = 8192
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=31)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize(
    "dtype,top_k,N,batch_size",
    [
        # Shapes that force the *radix* backend multi-CTA.  radix_primitives
        # streams the batch-32/64 rows on one CTA each (no SMEM staging); the
        # single-row 64K cell forms an 8-CTA group on 148-SM parts.
        (torch.bfloat16, 1024, 131072, 64),
        (torch.bfloat16, 1024, 65536, 1),
        (torch.float32, 2048, 65536, 32),
        (torch.float32, 2048, 131072, 32),
    ],
)
def test_radix_primitives_large_n(dtype, top_k, N, batch_size):
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=33)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k,next_n", [(512, 2), (1024, 2)])
def test_radix_primitives_next_n(dtype, top_k, next_n):
    N, num_rows = 8192, 8
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=35, next_n=next_n
    )
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, next_n=next_n, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, next_n=next_n)


@requires_radix_primitives
def test_radix_primitives_compress_ratio():
    dtype, top_k, N, batch_size, cr = torch.bfloat16, 512, 8192, 4, 4
    logits, _, seq_lens = _make_inputs(
        batch_size, N, top_k, dtype, seed=37, compress_ratio=cr
    )
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, compress_ratio=cr, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, compress_ratio=cr)


@requires_radix_primitives
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", ["single_cta", "multi_cta"])
def test_radix_primitives_return_values(dtype, shape):
    """values equal logits[row, indices] exactly; rows shorter than top_k
    (the degenerate identity branch) pad the indices with -1 AND the values
    with 0, in the single-CTA kernel and in the multi-CTA group kernel (two
    65536-column rows form a group on every part with >= 16 SMs)."""
    from flashinfer.topk_varlen.topk_varlen import _prim_get_group_config
    from flashinfer.utils import get_device_sm_count

    top_k = 1024
    if shape == "single_cta":
        N = 8192
        lens = [N, N - 1, top_k - 1, top_k // 2, 17, 1, 0, N]
    else:
        N = 65536
        lens = [N, top_k // 2]
    cpg, _chunk = _prim_get_group_config(
        N, dtype, len(lens), get_device_sm_count(torch.device("cuda"))
    )
    if (cpg > 1) != (shape == "multi_cta"):
        pytest.skip(f"{shape}: this device forms ctas_per_group={cpg} here")
    logits, seq_lens = _make_varlen_inputs(lens, N, dtype, seed=39)
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives", return_values=True
    )
    torch.cuda.synchronize()
    assert values.shape == (len(lens), top_k) and values.dtype == dtype
    lf = logits.float()
    for r, L in enumerate(lens):
        kk = min(top_k, L)
        row = indices[r]
        assert bool((row[:kk] >= 0).all()) and bool((row[kk:] == -1).all()), (
            f"row={r}: pads"
        )
        assert bool((values[r][kk:] == 0).all()), f"row={r}: values at -1 slots not 0"
        if kk == 0:
            continue
        sel = row[:kk].long()
        assert int(sel.max()) < L and sel.unique().numel() == kk, f"row={r}: dup/oor"
        assert torch.equal(values[r][:kk].float(), lf[r][sel]), (
            f"row={r}: values do not match logits[row, indices]"
        )
        got = torch.sort(lf[r][sel], descending=True).values
        assert torch.equal(got, torch.topk(lf[r, :L], kk).values), f"row={r}: values"


@requires_radix_primitives
def test_radix_primitives_unaligned_rows():
    """Row width not a multiple of the 16B vector: exercises the scalar
    prologue/tail split (row byte address changes alignment per row)."""
    dtype, top_k, N = torch.bfloat16, 512, 8190  # N*2 % 16 != 0
    seq_len_list = [N, 7000, 4096, 513]
    logits, seq_lens = _make_varlen_inputs(seq_len_list, N, dtype, seed=41)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_primitives_heavy_ties(dtype):
    """Coarsely quantized logits: the threshold bin holds many exact
    duplicates, exercising the in-smem exact tie select (eq_count > remaining
    but <= TIE_CAP)."""
    top_k, N, batch_size = 512, 8192, 4
    torch.manual_seed(43)
    logits = (
        torch.randint(0, 64, (batch_size, N), device="cuda").to(torch.float32) / 8.0
    ).to(dtype)
    seq_lens = torch.full((batch_size,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_primitives_tie_overflow(dtype):
    """Near-constant rows: >TIE_CAP elements share the threshold bin, forcing
    the exact gmem refinement path.  Boosted columns must still be selected."""
    top_k, N = 512, 8192
    logits = torch.zeros(2, N, dtype=dtype, device="cuda")
    logits[0, 100] = 5.0
    logits[0, 7000] = 4.0
    seq_lens = torch.full((2,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    for row in range(2):
        sel = indices[row].long()
        assert sel.unique().numel() == top_k, f"row={row}: duplicate indices"
        assert (sel >= 0).all() and (sel < N).all(), f"row={row}: out-of-range"
    picked = set(indices[0].cpu().tolist())
    assert 100 in picked and 7000 in picked, "boosted columns must be selected"


@requires_radix_primitives
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_radix_primitives_16bit_inbin_tie_overflow(dtype):
    """Regression: 16-bit tie-overflow refinement must OR the coarse-bin top
    bits into the final pivot key.

    All elements share ONE coarse bin (values are 8 consecutive ULPs of 1.0;
    a 13-bit bin leaves exactly 3 key bits free), so eq_count = N > TIE_CAP
    forces the exact refinement path.  The buggy pivot held only the refined
    low byte, so ``key > pivot`` admitted nearly every bin member and the row
    filled in atomic arrival order, dropping genuinely-larger values.
    """
    top_k, N = 1024, 8192
    ulp = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-10
    torch.manual_seed(45)
    logits = (
        1.0 + torch.randint(0, 8, (3, N), device="cuda").to(torch.float32) * ulp
    ).to(dtype)
    seq_lens = torch.tensor([N, N - 1, top_k + 9], dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_radix_primitives_multi_cta_overflow_cleanup(dtype):
    """Regression: multi-CTA FINAL_MC emission must be fenced from the rank-0
    row cleanup.

    Two-valued rows overflow the tie stage (half the row shares the threshold
    bin), sending every group through the refinement + FINAL_MC path where
    all ranks emit through the shared g_out/g_eqf atomics.  Without that
    barrier, rank 0's cleanup raced those emissions: counters restarted
    mid-row (duplicate indices) and leaked nonzero into the NEXT call, whose
    first output slots then kept stale garbage.  Repeated fresh-data calls on
    the shared group state make the race bite reliably.
    """
    top_k, N = 512, 131072
    seq_lens = torch.tensor([N, top_k + 33], dtype=torch.int32, device="cuda")
    for it in range(6):
        torch.manual_seed(47 + it)
        logits = torch.where(torch.rand(2, N, device="cuda") < 0.5, 1.0, -1.0).to(dtype)
        indices, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix_primitives"
        )
        torch.cuda.synchronize()
        _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
def test_radix_primitives_inf_flood_multirow():
    """Regression: > top_k in-range +inf values must classify as TIES.

    The fp32 float-boundary collect gave the above-top-bin boundary as +inf,
    so ``v >= hi_b`` classified every +inf as GREATER-THAN -- more gt hits
    than the histogram promised.  The scan-collect's positional stores had no
    top_k cap, so the excess spilled past the row's output slots into the
    NEXT row's indices (row 0 always looked fine; every later row picked
    ~half random values, some duplicated).  Multi-row is essential: a single
    row hides the bug (the spill lands out-of-tensor and the first top_k
    emissions happen to be infs, i.e. a correct answer).
    """
    top_k, N, batch = 2048, 32768, 16
    torch.manual_seed(101)
    logits = torch.randn(batch, N, device="cuda")
    logits = torch.where(
        torch.rand(batch, N, device="cuda") < 0.2,
        torch.full_like(logits, float("inf")),
        logits,
    ).contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize(
    "nnan,negative,N,batch",
    [
        (10, False, 32768, 4),  # few NaNs: used to be silently dropped
        (100, False, 32768, 4),  # > tie slack: used to UNDERFILL (-1 pads)
        (3000, False, 32768, 4),  # > top_k: used to return an EMPTY row
        (10, True, 32768, 4),  # negative-sign NaN patterns
        (1500, False, 65536, 2),  # multi-CTA group path
    ],
)
def test_radix_primitives_nan_inputs(nnan, negative, N, batch):
    """Regression: in-range NaNs must rank top (torch.topk semantics).

    The fp32 float-boundary collect classified with ordered compares, which
    every NaN fails, so NaNs counted by the (integer-bin) histogram were
    dropped by the collect: rows underfilled with -1 pads once the NaN count
    exceeded the threshold-bin slack, and came back empty with > top_k NaNs.
    The classify branch is now inverted around a strict-GT threshold
    (coarse_bin_gt_threshold_f32) so NaNs of either sign land in the gt arm.
    torch.equal is NaN-hostile: compare NaN counts + finite values.
    """
    top_k = 2048
    torch.manual_seed(303)
    logits = torch.randn(batch, N, device="cuda")
    for r in range(batch):
        idx = torch.randperm(N, device="cuda")[:nnan]
        logits[r, idx] = float("nan")
        if negative:
            lb = logits[r].view(torch.int32)
            lb[idx] = lb[idx] | torch.tensor(
                -0x80000000, device="cuda", dtype=torch.int32
            )
    logits = logits.contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    for r in range(batch):
        sel = indices[r]
        assert int((sel < 0).sum()) == 0, f"row{r}: -1 pads in a full row"
        sel = sel.long()
        assert int(sel.max()) < N and sel.unique().numel() == top_k, f"row{r}: dup/oor"
        got = logits[r][sel]
        ref = torch.topk(logits[r], top_k).values
        assert int(torch.isnan(got).sum()) == min(nnan, top_k), f"row{r}: NaN count"
        got_fin = got[~torch.isnan(got)].sort(descending=True).values
        ref_fin = ref[~torch.isnan(ref)]
        assert torch.equal(got_fin, ref_fin), f"row{r}: finite part mismatch"


def _inject_nans(row, cols, payloads, negative):
    """Write quiet NaNs with the given payloads (sign bit per ``negative``)
    into the row at ``cols``, in the row's own dtype; payloads wrap into the
    dtype's payload field (22 bits fp32, 9 bits fp16, 6 bits bf16)."""
    base, width, pbits = {
        torch.float32: (0x7FC00000, 32, 22),
        torch.float16: (0x7E00, 16, 9),
        torch.bfloat16: (0x7FC0, 16, 6),
    }[row.dtype]
    if negative:
        base |= 1 << (width - 1)
    half = 1 << (width - 1)
    bits = [
        ((base | (1 + (p - 1) % ((1 << pbits) - 1))) + half) % (1 << width) - half
        for p in payloads
    ]
    bits_dtype = torch.int32 if width == 32 else torch.int16
    row.view(bits_dtype)[cols] = torch.tensor(bits, dtype=bits_dtype, device=row.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16], ids=["f32", "bf16", "f16"]
)
@pytest.mark.parametrize("backend", ["walkfirst_primitives", "radix_primitives"])
def test_nan_ranking_primitives(backend, dtype):
    """NaN placement of the two primitives kernels on rows of 2048 / 8000 /
    20000 columns -- walk-first's census, register and walk arms.

    Quiet +NaN ranks above every finite value in both kernels (torch.topk
    semantics) and the rest of the selection is the finite top-k multiset.
    Sign-set NaN depends on the key space: fp32 ranks EVERY NaN top in both
    kernels (walk-first's census converts through cvt.rn.f16.f32, which
    canonicalises NaN; radix_primitives' fp32 collect sends every NaN to the
    greater-than arm).  The 16-bit dtypes rank keys as signed integer
    patterns: walk-first's census / register arms (rows <= 16384) put a
    sign-set NaN at the BOTTOM (never selected while finite values remain)
    and its walk pipeline (20000 columns) at the top; radix_primitives'
    to_key16 flips sign-set patterns, so a sign-set 16-bit NaN ranks bottom
    on every width.  A flood of more than top_k +NaNs fills the row with
    NaNs only.
    """
    _skip_unless_backend(backend)
    top_k, N, nnan = 512, 20000, 7
    lens = [2048, 8000, 20000]
    gen = torch.Generator(device="cuda").manual_seed(0x7FC0)
    base = torch.randn(len(lens), N, device="cuda", generator=gen).to(dtype)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    for negative in (False, True):
        x = base.clone()
        for r, L in enumerate(lens):
            cols = torch.randperm(L, device="cuda", generator=gen)[:nnan]
            _inject_nans(x[r], cols, range(1, nnan + 1), negative)
        idx, _ = flashinfer.top_k_varlen(x, seq_lens, top_k, backend=backend)
        torch.cuda.synchronize()
        for r, L in enumerate(lens):
            sel = idx[r].long()
            assert int((sel < 0).sum()) == 0, f"negative={negative} row{r}: -1 pads"
            assert int(sel.max()) < L and sel.unique().numel() == top_k, (
                f"negative={negative} row{r}: dup/oor"
            )
            got = x[r][sel]
            ranks_top = (
                not negative
                or dtype == torch.float32
                or (backend == "walkfirst_primitives" and L > 16384)
            )
            want_nan = nnan if ranks_top else 0
            assert int(torch.isnan(got).sum()) == want_nan, (
                f"negative={negative} row{r}: NaN count, expected {want_nan}"
            )
            finite = x[r, :L][~torch.isnan(x[r, :L])]
            ref = torch.topk(finite, top_k - want_nan).values
            got_fin = torch.sort(got[~torch.isnan(got)], descending=True).values
            assert torch.equal(got_fin, ref), f"negative={negative} row{r}: finite part"
    # flood: more than top_k +NaNs (distinct payloads where the dtype has
    # them), two rows of 16384
    n2, k2, nflood = 16384, 2048, 2100
    x2 = torch.randn(2, n2, device="cuda", generator=gen).to(dtype)
    for r in range(2):
        cols = torch.randperm(n2, device="cuda", generator=gen)[:nflood]
        _inject_nans(x2[r], cols, range(1, nflood + 1), False)
    seq2 = torch.full((2,), n2, dtype=torch.int32, device="cuda")
    idx2, _ = flashinfer.top_k_varlen(x2, seq2, k2, backend=backend)
    torch.cuda.synchronize()
    for r in range(2):
        sel = idx2[r].long()
        assert int((sel < 0).sum()) == 0, f"flood row{r}: -1 pads"
        assert int(sel.max()) < n2 and sel.unique().numel() == k2, (
            f"flood row{r}: dup/oor"
        )
        assert bool(torch.isnan(x2[r][sel]).all()), (
            f"flood row{r}: finite value selected"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "backend,dtype",
    [
        ("radix_primitives", torch.float32),
        ("walkfirst_primitives", torch.float32),
        ("walkfirst_primitives", torch.bfloat16),
        ("walkfirst_primitives", torch.float16),
    ],
    ids=["radix_f32", "walkfirst_f32", "walkfirst_bf16", "walkfirst_f16"],
)
@pytest.mark.parametrize("top_k", [512, 2048])
def test_signed_zero_crossing(backend, dtype, top_k):
    """Regression: the rank-k value is -0.0 while +0.0 values rank above it.

    radix_primitives' fp32 float-boundary collect tested ``v <= T`` with T =
    -0.0 (the ordered-key predecessor of the +0.0 bin's bound), which holds
    for +0.0 as well, while the fp16 coarse histogram kept the two zeros in
    different bins: every +0.0 winner vanished and the row came back
    underfilled (3000 valid columns, K=2048: 154 positives, 1346 +0.0 and
    1360 -0.0; 1346 slots stayed unwritten).  Both zeros are one bin and one
    key now, and a zero threshold steps down to the largest negative float.
    -0.0 and +0.0 are equal as values, so the reference multiset compares
    with torch.equal.  The 16-bit dtypes run walk-first's 8-element vector /
    integer-key path, where the two zeros are distinct keys of equal value.
    """
    _skip_unless_backend(backend)
    N = 16384
    lens = [3000, 2600, 700, 4096, 8192, 12000, 16384]
    torch.manual_seed(5)
    logits = torch.relu(torch.randn(len(lens), N, device="cuda") - 1.2816).to(dtype)
    logits[:, 1::2] = logits[:, 1::2] * -1.0  # odd columns: -0.0 zeros and negatives
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = torch.full((len(lens), top_k), 0x7EADBEEF, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend=backend, out_indices=out
    )
    torch.cuda.synchronize()
    for r, L in enumerate(lens):
        kk = min(top_k, L)
        row = indices[r]
        assert int((row == 0x7EADBEEF).sum()) == 0, f"row{r}: unwritten slots"
        assert bool((row[:kk] >= 0).all()) and bool((row[kk:] == -1).all()), (
            f"row{r}: pads"
        )
        sel = row[:kk].long()
        assert int(sel.max()) < L and sel.unique().numel() == kk, f"row{r}: dup/oor"
        got = torch.sort(logits[r, :L][sel], descending=True).values
        assert torch.equal(got, torch.topk(logits[r, :L], kk).values), f"row{r}: values"


@requires_radix_primitives
@pytest.mark.parametrize("edge", [2.0, -2.0, 8.0, 0.5])
def test_radix_primitives_binade_edge_tie_overflow(edge):
    """Regression: fp32 coarse bins that straddle a 2^24 ordered-key
    boundary (fp16 binade edges: values near +/-2^m, m odd) must run the
    high-byte refinement round.

    The overflow refinement skips the (24, 8) round when the coarse bin
    provably pins key bits [24, 32); 30 binade-edge bins violate that bound
    (found by adversarial review + exhaustive host enumeration,
    proto_wide_predicate.py).  With the round wrongly skipped, values on
    opposite sides of the boundary (e.g. 2.0 vs its fp32 predecessor, both
    in one coarse bin) get misordered by the masked low-bit compares and
    the strictly-larger values are dropped.
    """
    top_k, N = 2048, 16384
    torch.manual_seed(7)
    below = torch.nextafter(
        torch.tensor(edge, device="cuda"), torch.tensor(0.0, device="cuda")
    )
    logits = torch.full((2, N), -100.0 if edge > 0 else -1e9, device="cuda")
    logits[:, :3000] = below if edge > 0 else edge
    logits[:, 3000:3500] = edge if edge > 0 else below
    for r in range(2):
        logits[r] = logits[r][torch.randperm(N, device="cuda")]
    logits = logits.contiguous()
    seq_lens = torch.full((2,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize("N,batch", [(8192, 4), (65536, 2), (32768, 16)])
def test_radix_primitives_approx_ties(N, batch):
    """approx_ties=True: sglang-compatible tie-truncation semantics.

    A row whose threshold coarse bin holds > TIE_CAP candidates (``1.0 +
    rand * 2**-12`` has 2048 distinct fp32 values, each repeated) is filled
    with an arbitrary first-arrival subset of that bin instead of the exact
    smallest-key refinement.  Contract checked here: full row of
    unique in-range indices, every selected value from the tie bin or above
    (>= 1.0 in this construction), and approx_ties=True on rows without tie
    overflow is still exact (value multiset equals torch.topk).
    """
    top_k = 2048
    torch.manual_seed(11)
    # rows 0..: half in-bin (1.0 + eps), half fill at -5.0 -> tie overflow
    logits = torch.full((batch, N), -5.0, device="cuda")
    m = N // 2
    logits[:, :m] = 1.0 + torch.rand(batch, m, device="cuda") * 2**-12
    for r in range(batch):
        logits[r] = logits[r][torch.randperm(N, device="cuda")]
    logits = logits.contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives", approx_ties=True
    )
    torch.cuda.synchronize()
    for r in range(batch):
        sel = indices[r]
        assert int((sel < 0).sum()) == 0, f"row{r}: pads in a full row"
        sel = sel.long()
        assert int(sel.max()) < N and sel.unique().numel() == top_k, f"row{r} dup/oor"
        assert bool((logits[r][sel] >= 1.0).all()), f"row{r}: picked below the tie bin"

    # no-overflow rows: approx_ties=True stays exact (value multiset == torch.topk)
    torch.manual_seed(12)
    xr = (torch.randn(batch, N, device="cuda") * 2.0).contiguous()
    ia, _ = flashinfer.top_k_varlen(
        xr, seq_lens, top_k, backend="radix_primitives", approx_ties=True
    )
    torch.cuda.synchronize()
    _check_correct(ia, xr, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize(
    "top_k,pattern,dtype",
    [
        (4096, "randn", torch.float32),
        (4096, "randn", torch.bfloat16),
        (3000, "randn", torch.float32),
        (4096, "constant", torch.float32),  # uniform ties, remaining > TIE_CAP
        # one bin: 2048 distinct fp32 values, each repeated; EQFILL after rounds
        (4096, "one_bin", torch.float32),
        (4096, "constant", torch.bfloat16),
    ],
)
def test_radix_primitives_large_top_k(top_k, pattern, dtype):
    """top_k > TIE_CAP (2048): the staged tie machinery caps at TIE_CAP, so
    remaining > TIE_CAP rows resolve through the masked EQFILL arm (exact:
    survivors of all refinement rounds are provably key-identical).  Also
    covers the multi-CTA shape and short rows."""
    torch.manual_seed(21)
    N, batch = 16384, 3
    if pattern == "randn":
        logits = torch.randn(batch, N, device="cuda") * 2.0
    elif pattern == "constant":
        logits = torch.full((batch, N), 1.5, device="cuda")
    else:  # one_bin
        logits = 1.0 + torch.rand(batch, N, device="cuda") * 2**-12
    logits = logits.to(dtype).contiguous()
    # row 2 is short but non-degenerate (N_eff must stay >= top_k for the
    # strict checker); the degenerate short-row -1 fill is covered by
    # test_top_k_varlen's short-row cases at small k.
    seq_lens = torch.tensor([N, N - 3, top_k + 21], dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    # multi-CTA shape (same pattern, so constant/one_bin also exercise the
    # EQFILL_MC arm)
    if pattern == "randn":
        logits2 = torch.randn(1, 65536, device="cuda").to(dtype).contiguous()
    elif pattern == "constant":
        logits2 = torch.full((1, 65536), 1.5, device="cuda").to(dtype).contiguous()
    else:  # one_bin
        logits2 = (
            (1.0 + torch.rand(1, 65536, device="cuda") * 2**-12).to(dtype).contiguous()
        )
    seq2 = torch.full((1,), 65536, dtype=torch.int32, device="cuda")
    idx2, _ = flashinfer.top_k_varlen(logits2, seq2, top_k, backend="radix_primitives")
    torch.cuda.synchronize()
    _check_correct(idx2, logits2, seq2, top_k, require_all_checked=True)


@requires_radix_primitives
def test_radix_primitives_large_top_k_exact_fill_tie_arm():
    """Regression: top_k > TIE_CAP with a threshold bin that EXACTLY fills the
    remainder (gt_count + eq_count == top_k, eq_count > TIE_CAP).

    The collect stages at most TIE_CAP tie slots, but the direct-copy arm
    (``eq_count <= remaining``) copied ``eq_count`` entries from the stage,
    reading past it into the histogram / counter smem: garbage or duplicate
    indices among the 1000 + 3096 = 4096 winners.  1000 distinct values in
    [10, 20) are strictly greater than the 3096 copies of 5.0; the rest of
    the row is negative, so nothing else reaches the tie bin.
    """
    top_k, N, gt_count = 4096, 16384, 1000
    eq_count = top_k - gt_count
    gen = torch.Generator(device="cuda").manual_seed(4096)
    logits = -torch.rand(2, N, device="cuda", generator=gen) * 50.0 - 1.0
    greater = 10.0 + torch.arange(gt_count, device="cuda") * (10.0 / gt_count)
    for r in range(2):
        perm = torch.randperm(N, device="cuda", generator=gen)
        logits[r, perm[:gt_count]] = greater
        logits[r, perm[gt_count:top_k]] = 5.0
    seq_lens = torch.full((2,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    for r in range(2):
        sel = indices[r].long()
        assert int((sel < 0).sum()) == 0, f"row{r}: -1 pads in a full row"
        assert int(sel.max()) < N and sel.unique().numel() == top_k, f"row{r}: dup/oor"
        got = torch.sort(logits[r][sel], descending=True).values
        assert torch.equal(got, torch.topk(logits[r], top_k).values), f"row{r}: values"
        assert int((got == 5.0).sum()) == eq_count, f"row{r}: tie-bin count"


@requires_radix_primitives
def test_radix_primitives_large_top_k_multi_cta_sorted_rows():
    """Regression: top_k > TIE_CAP on the multi-CTA group shape (N=65536,
    small batch) with one chunk holding more than TIE_CAP strictly-greater
    elements.

    The multi-CTA collect staged each CTA's strictly-greater hits in the
    TIE_CAP-entry stage and clamped the batch reservation to TIE_CAP, so a
    descending row (every winner in the first chunk) lost the excess: rank 0
    back-filled from the tie bin and padded the rest with -1.  The ascending
    row puts the winners in the LAST chunk; the random rows spread them.
    The contract is the exact value multiset whether the host runs the shape
    as a group or forces one CTA for top_k > TIE_CAP.
    """
    top_k, N = 4096, 65536
    gen = torch.Generator(device="cuda").manual_seed(65536)
    logits = torch.randn(4, N, device="cuda", generator=gen) * 2.0
    logits[0] = torch.arange(N, 0, -1, device="cuda", dtype=torch.float32)
    logits[1] = torch.arange(1, N + 1, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((4,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    for r in range(4):
        sel = indices[r].long()
        assert int((sel < 0).sum()) == 0, f"row{r}: -1 pads in a full row"
        assert int(sel.max()) < N and sel.unique().numel() == top_k, f"row{r}: dup/oor"
        got = torch.sort(logits[r][sel], descending=True).values
        assert torch.equal(got, torch.topk(logits[r], top_k).values), f"row{r}: values"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "backend", ["radix_primitives", "walkfirst_primitives", "sglang"]
)
def test_empty_batch(backend):
    """Zero rows must early-return (a (0, top_k) int32 result) instead of
    launching a zero-block grid, which is a CUDA error."""
    _skip_unless_backend(backend)
    logits = torch.empty(0, 4096, dtype=torch.float32, device="cuda")
    seq_lens = torch.empty(0, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, 512, backend=backend)
    torch.cuda.synchronize()
    assert indices.shape == (0, 512) and indices.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "backend", ["walkfirst_primitives", "sglang", "radix_primitives"]
)
def test_top_k_exceeds_width(backend):
    """top_k > N: every row is shorter than k, so the whole batch is the
    identity answer ([0, len) then -1 pads) and any staging sized from top_k
    against N must not misbehave.  A backend may instead refuse the shape up
    front with the API's ValueError (reported as a skip); silent garbage or a
    device fault is the failure."""
    from flashinfer.utils import BackendSupportedError

    _skip_unless_backend(backend)
    N, top_k = 1024, 2048
    lens = [0, 1, 700, N]
    gen = torch.Generator(device="cuda").manual_seed(2048)
    logits = torch.randn(
        len(lens), N, dtype=torch.float32, device="cuda", generator=gen
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    sentinel = 0x7EADBEEF
    out = torch.full((len(lens), top_k), sentinel, dtype=torch.int32, device="cuda")
    try:
        idx, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend=backend, out_indices=out
        )
    except (ValueError, BackendSupportedError) as e:
        pytest.skip(f"{backend} refuses top_k > N up front: {e}")
    torch.cuda.synchronize()
    for r, L in enumerate(lens):
        row = idx[r]
        assert int((row == sentinel).sum()) == 0, f"row={r}: unwritten slots"
        assert bool((row[L:] == -1).all()), f"row={r}: pads"
        assert sorted(row[:L].tolist()) == list(range(L)), f"row={r}: identity [0,{L})"


@requires_radix_primitives
def test_radix_primitives_multi_cta_mixed_dtypes():
    """bf16 and fp32 multi-CTA groups back-to-back in one process.

    Regression: the two dtypes have different row_states layouts (histogram
    sizes), and the kernel's end-of-launch self-reset only zeroes its own
    layout's offsets.  A shared scratch buffer let bf16's (deliberately
    un-reset) stale tie buffer alias into fp32's group-1 histogram,
    corrupting every row handled by group >= 1.  Buffers are now keyed by
    layout; this test locks that in.  batch=2 ensures group 1 is exercised.
    """
    from flashinfer.topk_varlen.topk_varlen import _prim_get_group_config
    from flashinfer.utils import get_device_sm_count

    nsms = get_device_sm_count(torch.device("cuda"))
    for dtype, N, top_k in (
        (torch.bfloat16, 65536, 1024),
        (torch.float32, 65536, 2048),
        (torch.bfloat16, 131072, 1024),
        (torch.float32, 131072, 2048),
    ):
        cpg, _chunk = _prim_get_group_config(N, dtype, 2, nsms)
        assert cpg > 1, f"expected multi-CTA at batch=2 N={N}, got cpg={cpg}"
        logits, seq_lens = _make_varlen_inputs([N, N - 12345], N, dtype, seed=53)
        indices, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix_primitives"
        )
        torch.cuda.synchronize()
        _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_radix_primitives_multi_cta_tie_overflow(dtype):
    """Multi-CTA group + tie overflow: the cooperative gmem refinement path
    (per-round global 256-bin merges) on a near-constant long row."""
    top_k, N = 1024, 65536
    logits = torch.zeros(1, N, dtype=dtype, device="cuda")
    logits[0, 123] = 5.0
    seq_lens = torch.full((1,), N, dtype=torch.int32, device="cuda")
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="radix_primitives"
    )
    torch.cuda.synchronize()
    sel = indices[0].long()
    assert sel.unique().numel() == top_k
    assert (sel >= 0).all() and (sel < N).all()
    assert 123 in indices[0].cpu().tolist(), "boosted column must be selected"


@requires_radix_primitives
@pytest.mark.parametrize("return_values", [True, False])
def test_radix_primitives_preallocated_outputs(return_values):
    dtype, top_k, N, batch_size = torch.bfloat16, 512, 8192, 4
    logits, seq_lens = _make_varlen_inputs([N] * batch_size, N, dtype, seed=45)
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch_size, top_k, dtype=dtype, device="cuda")
    ret_i, ret_v = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        backend="radix_primitives",
        return_values=return_values,
        out_indices=out_i,
        out_values=out_v,
    )
    torch.cuda.synchronize()
    assert ret_i is out_i
    if return_values:
        assert ret_v is out_v
    else:
        assert ret_v is None
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


@requires_radix_primitives
@pytest.mark.parametrize(
    "dtype,N,batch_size",
    [(torch.bfloat16, 131072, 8), (torch.float32, 16384, 16)],
    ids=["group_bf16_128k", "solo_f32_16k"],
)
def test_cuda_graph_radix_primitives(dtype, N, batch_size):
    """radix_primitives under CUDA graph capture/replay, including with fresh
    input data.  The bf16 128K x 8 cell takes the multi-CTA group kernel on
    148-SM parts, whose self-resetting row_states must survive replay (a
    stale arrival counter would corrupt the second replay); the fp32 16K cell
    is the stateless single-CTA kernel."""
    top_k = 1024
    torch.manual_seed(N // 1024 + batch_size)
    logits = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 2).to(
        dtype
    )
    seq_lens = torch.full((batch_size,), N, dtype=torch.int32, device="cuda")
    out_i = torch.empty(batch_size, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, backend="radix_primitives", out_indices=out_i
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)

    fresh = (torch.randn(batch_size, N, dtype=torch.float32, device="cuda") * 3).to(
        dtype
    )
    logits.copy_(fresh)
    out_i.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out_i, logits, seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("N", [16384, 65536])
def test_cuda_graph_walkfirst_primitives(N):
    """walkfirst_primitives under CUDA graph replay with the lengths changing
    between replays: the backend keeps per-(rows, stream) device state (the
    self-resetting slab, status, mc_state), so one graph must serve rows that
    grow from the identity / census arm (700 columns) into the walk arm (the
    full width; split across CTAs at 64K) and shrink back, plus a mixed batch
    touching every arm boundary.  Captured on the stream the one eager
    warm-up call ran on, as the paged graph test does."""
    _skip_unless_backend("walkfirst_primitives")
    rows, top_k, short = 16, 512, 700
    gen = torch.Generator(device="cuda").manual_seed(N // 1024)
    logits = torch.randn(rows, N, dtype=torch.float32, device="cuda", generator=gen)
    seq_lens = torch.full((rows,), short, dtype=torch.int32, device="cuda")
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device="cuda")
    kw = dict(backend="walkfirst_primitives", out_indices=out)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # compile + per-stream scratch, outside capture
        flashinfer.top_k_varlen(logits, seq_lens, top_k, **kw)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(logits, seq_lens, top_k, **kw)
    torch.cuda.current_stream().wait_stream(s)
    mixed = [0, 1, top_k - 1, top_k, top_k + 1, short, 4095, 4096]
    mixed += [4097, 8191, 8192, 8193, 12000, N - 5, N - 1, N]
    for lengths in ([N] * rows, [short] * rows, mixed, [N] * rows):
        seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        out.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        for r, L in enumerate(lengths):
            kk = min(top_k, L)
            row = out[r]
            assert int((row == -7).sum()) == 0, f"len={L} row{r}: unwritten slots"
            assert bool((row[:kk] >= 0).all()) and bool((row[kk:] == -1).all()), (
                f"len={L} row{r}: pads"
            )
            if kk == 0:
                continue
            sel = row[:kk].long()
            assert int(sel.max()) < L and sel.unique().numel() == kk, (
                f"len={L} row{r}: dup/oor"
            )
            got = torch.sort(logits[r, :L][sel], descending=True).values
            assert torch.equal(got, torch.topk(logits[r, :L], kk).values), (
                f"len={L} row{r}: values"
            )


# ---------------------------------------------------------------------------
# cutlass_primitives: the vendored library as one backend
# ---------------------------------------------------------------------------


def _cutlass_primitives_available():
    return (
        _FLASHINFER_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
        and is_cute_dsl_available()
    )


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16], ids=["f32", "bf16", "f16"]
)
@pytest.mark.parametrize(
    "N,batch,top_k",
    [
        (4096, 256, 512),
        (16384, 64, 1024),
        (16384, 8, 2048),
        (65536, 8, 512),
        (65536, 148, 1024),
        (65536, 256, 2048),
        (262144, 64, 1024),
        (1048576, 8, 1024),
    ],
)
def test_cutlass_primitives_exact(N, batch, top_k, dtype):
    """Value multiset equals torch.topk on every row, full and ragged lengths; every kernel
    the library's router can pick is covered by the (N, batch) grid."""
    torch.manual_seed(N // 1024 + batch)
    logits = (torch.randn(batch, N, device="cuda") * 2.0).to(dtype).contiguous()
    for seq_lens in (
        torch.full((batch,), N, dtype=torch.int32, device="cuda"),
        torch.randint(top_k + 1, N + 1, (batch,), dtype=torch.int32, device="cuda"),
    ):
        out = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, out_indices=out, backend="cutlass_primitives"
        )
        torch.cuda.synchronize()
        _check_correct(out, logits, seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("kind", ["constant", "two_values", "short_rows"])
def test_cutlass_primitives_degenerate_rows(kind):
    """Low-entropy rows take the exact fallback; rows shorter than k pad with -1."""
    batch, N, top_k = 16, 65536, 1024
    torch.manual_seed(23)
    if kind == "constant":
        logits = torch.full((batch, N), 0.5, device="cuda")
    elif kind == "two_values":
        logits = torch.where(torch.rand(batch, N, device="cuda") < 0.001, 3.0, -1.0)
    else:
        logits = torch.randn(batch, N, device="cuda")
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    if kind == "short_rows":
        seq_lens = torch.tensor(
            [0, 1, 100, 1023, 1024, 1025, 4096, 16384] * 2,
            dtype=torch.int32,
            device="cuda",
        )
    out = torch.full((batch, top_k), -7, dtype=torch.int32, device="cuda")
    flashinfer.top_k_varlen(
        logits, seq_lens, top_k, out_indices=out, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    for r in range(batch):
        n_eff = int(seq_lens[r])
        valid = min(n_eff, top_k)
        assert (out[r, valid:] == -1).all(), f"row {r}: padding"
        idx = out[r, :valid].long()
        assert (
            idx.numel() == torch.unique(idx).numel()
            and bool((idx >= 0).all())
            and bool((idx < n_eff).all())
        )
        if valid == top_k:
            got = logits[r, idx].sort(descending=True).values
            ref = torch.topk(logits[r, :n_eff], top_k).values
            assert torch.equal(got, ref), f"row {r}: values differ"


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
def test_cuda_graph_cutlass_primitives():
    """Capture and replay: the library launches on the caller's stream through the TVM-FFI
    environment stream, so it must capture like any other backend."""
    batch, N, top_k = 64, 65536, 1024
    torch.manual_seed(29)
    logits = (torch.randn(batch, N, device="cuda") * 2.0).contiguous()
    seq_lens = torch.randint(
        top_k + 1, N + 1, (batch,), dtype=torch.int32, device="cuda"
    )
    out = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")

    def call():
        flashinfer.top_k_varlen(
            logits, seq_lens, top_k, out_indices=out, backend="cutlass_primitives"
        )

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out, logits, seq_lens, top_k, require_all_checked=True)
    logits.copy_((torch.randn(batch, N, device="cuda") * 3.0))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    _check_correct(out, logits, seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 1024])
@pytest.mark.parametrize("next_n", [2, 3])
@pytest.mark.parametrize("N", [8192, 65536])
def test_cutlass_primitives_next_n(dtype, top_k, next_n, N):
    """next_n rows share one seq_len entry; row i of a group sees i % next_n more tokens."""
    num_rows = 8 * next_n
    logits, _, seq_lens = _make_inputs(
        num_rows, N, top_k, dtype, seed=41, next_n=next_n
    )
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, next_n=next_n, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(
        indices, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("cr", [2, 4])
@pytest.mark.parametrize("N", [8192, 65536, 262144])
def test_cutlass_primitives_compress_ratio(cr, N):
    """compress_ratio divides the token length into compressed-block units."""
    dtype, top_k, batch_size = torch.bfloat16, 512, 6
    logits, _, _ = _make_inputs(batch_size, N, top_k, dtype, seed=43, compress_ratio=cr)
    # ragged token lengths, all long enough for a full top-k in block units
    g = torch.Generator(device="cuda").manual_seed(43)
    seq_lens = torch.randint(
        (top_k + 1) * cr, N * cr + 1, (batch_size,), device="cuda", generator=g
    ).to(torch.int32)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, compress_ratio=cr, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(
        indices, logits, seq_lens, top_k, compress_ratio=cr, require_all_checked=True
    )


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N, batch", [(8192, 8), (65536, 8), (262144, 16)])
def test_cutlass_primitives_return_values(dtype, N, batch):
    """values equal logits[row, indices] exactly; padding slots carry -inf values."""
    top_k = 1024
    logits, seq_lens = _make_varlen_inputs([N] * batch, N, dtype, seed=45)
    seq_lens[0] = top_k // 2  # a padded row
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    torch.cuda.synchronize()
    assert values.shape == (batch, top_k) and values.dtype == dtype
    _check_correct(indices, logits, seq_lens, top_k)
    for row in range(batch):
        valid = min(top_k, int(seq_lens[row]))
        expected = logits[row][indices[row, :valid].long()]
        assert torch.equal(expected, values[row, :valid]), f"row={row}: values differ"
        assert torch.isneginf(values[row, valid:]).all(), f"row={row}: padding values"
    # preallocated outputs are written in place
    out_i = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch, top_k, dtype=dtype, device="cuda")
    ri, rv = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        backend="cutlass_primitives",
        return_values=True,
        out_indices=out_i,
        out_values=out_v,
    )
    torch.cuda.synchronize()
    assert ri.data_ptr() == out_i.data_ptr() and rv.data_ptr() == out_v.data_ptr()
    # order within a row is unspecified: compare the rows as sorted multisets
    assert torch.equal(rv.float().sort(dim=1).values, values.float().sort(dim=1).values)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N, batch", [(8192, 8), (65536, 8), (262144, 16)])
def test_cutlass_primitives_paged_rows(dtype, N, batch):
    """Logits living in a wider arena (row stride > N) and a column-sliced view: same answers
    as the contiguous copy; a misaligned slice is refused."""
    top_k, pad = 1024, 64
    arena = (torch.randn(batch, N + pad, device="cuda") * 2.0).to(dtype)
    logits = arena[:, :N]
    assert not logits.is_contiguous()
    seq_lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    ref, _ = flashinfer.top_k_varlen(
        logits.contiguous(), seq_lens, top_k, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(
        indices, logits.contiguous(), seq_lens, top_k, require_all_checked=True
    )
    lf = logits.float()
    for row in range(batch):
        got = lf[row][indices[row].long()].sort().values
        want = lf[row][ref[row].long()].sort().values
        assert torch.equal(got, want), f"row={row}: arena and contiguous answers differ"
        assert torch.equal(lf[row][indices[row].long()], values[row].float())
    shifted = arena[
        :, 16 : 16 + N
    ]  # a slice starting inside the arena, rows still aligned
    out, _ = flashinfer.top_k_varlen(
        shifted, seq_lens, top_k, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(out, shifted.contiguous(), seq_lens, top_k, require_all_checked=True)
    odd = arena[
        :, 1 : 1 + N
    ]  # 4-byte offset: misaligned rows, copied into a padded arena
    out, _ = flashinfer.top_k_varlen(odd, seq_lens, top_k, backend="cutlass_primitives")
    torch.cuda.synchronize()
    _check_correct(out, odd.contiguous(), seq_lens, top_k, require_all_checked=True)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize(
    "N, dtype", [(4100, torch.float32), (16386, torch.bfloat16), (65541, torch.float32)]
)
def test_cutlass_primitives_unaligned_row_length(N, dtype):
    """Row lengths that are not whole 16-byte vectors: exact, ragged, with values."""
    top_k, batch = 512, 6
    logits, seq_lens = _make_varlen_inputs([N] * batch, N, dtype, seed=49)
    g = torch.Generator(device="cuda").manual_seed(49)
    seq_lens = torch.randint(top_k + 1, N + 1, (batch,), device="cuda", generator=g).to(
        torch.int32
    )
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    for row in range(batch):
        assert torch.equal(logits[row][indices[row].long()], values[row])


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
@pytest.mark.parametrize(
    "top_k, N, dtype",
    [
        (5000, 16384, torch.float32),
        (8192, 65536, torch.float32),
        (8192, 262144, torch.float32),
        (
            16384,
            32768,
            torch.float32,
        ),  # register kernel beyond its tie stage: radix refine
        (20000, 32768, torch.bfloat16),
        (65535, 65536, torch.float32),  # k = N - 1
        (16384, 16384, torch.float32),  # k = N: every row is the identity
        (12000, 262144, torch.float32),  # streaming: stage grown or split widened
        (65536, 262144, torch.float32),
        (
            200000,
            1 << 20,
            torch.float32,
        ),  # wide slab split, or the exact select on 99 KB parts
        (
            900000,
            1 << 20,
            torch.bfloat16,
        ),  # nothing holds it: the exact select for every row
    ],
    ids=[
        "5000",
        "8192_64k",
        "8192_256k",
        "reg16k",
        "reg20k_bf16",
        "n_minus_1",
        "k_eq_n",
        "str12k",
        "str64k",
        "1M_200k",
        "1M_900k_bf16",
    ],
)
def test_cutlass_primitives_large_k(top_k, N, dtype):
    """Any k up to and including N is eligible: exact on full rows, padded on rows shorter
    than k, with values."""
    batch = 4
    logits, seq_lens = _make_varlen_inputs([N] * batch, N, dtype, seed=47)
    from flashinfer.topk_varlen.topk_varlen import (
        _cutlass_primitives_top_k_varlen_check,
    )

    # the checker answers False for backend="auto" without consulting the
    # router (the backend is explicit-only unpaged); ask as an explicit call
    assert _cutlass_primitives_top_k_varlen_check(
        logits, seq_lens, top_k, backend="cutlass_primitives"
    )
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k, require_all_checked=True)
    for row in range(batch):
        assert torch.equal(logits[row][indices[row].long()], values[row]), f"row={row}"
    seq_lens[0] = top_k // 2  # a row shorter than k pads with -1 / -inf
    indices, values = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives", return_values=True
    )
    torch.cuda.synchronize()
    _check_correct(indices, logits, seq_lens, top_k)
    valid = top_k // 2
    assert (indices[0, valid:] == -1).all() and torch.isneginf(values[0, valid:]).all()
    assert torch.equal(logits[0][indices[0, :valid].long()], values[0, :valid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "backend",
    ["cutlass_primitives", "radix_primitives", "walkfirst_primitives", "sglang"],
)
def test_primitives_checkers_refuse_strided_seq_lens_and_zero_width(backend):
    """The single-launch backends' checkers encode what their kernels cannot
    take -- a strided seq_lens view (the kernels index it as a dense int32
    array) and zero-width logits -- so an explicit call raises the API's
    ValueError instead of launching on a malformed view."""
    _skip_unless_backend(backend)
    rows, N, top_k = 4, 4096, 512
    gen = torch.Generator(device="cuda").manual_seed(2)
    logits = torch.randn(rows, N, device="cuda", generator=gen)
    meta = torch.full((rows, 2), N, dtype=torch.int32, device="cuda")
    strided = meta[:, 0]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="Problem size"):
        flashinfer.top_k_varlen(logits, strided, top_k, backend=backend)
    with pytest.raises(ValueError):
        flashinfer.top_k_varlen(
            torch.empty(rows, 0, device="cuda"),
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            top_k,
            backend=backend,
        )


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
def test_cutlass_primitives_arena_sliced_rows_and_columns():
    """A view sliced on rows AND columns of a wider arena (row stride > N and
    an offset base) gives the value multisets of its contiguous copy, with
    the -1 padding of rows shorter than k."""
    N, top_k = 8192, 512
    gen = torch.Generator(device="cuda").manual_seed(3)
    arena = torch.randn(8, N + 64, device="cuda", generator=gen) * 2.0
    logits = arena[2:6, 16 : 16 + N]
    assert not logits.is_contiguous()
    lens = [N, 6000, 700, top_k]
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    idx, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives"
    )
    ref, _ = flashinfer.top_k_varlen(
        logits.contiguous(), seq_lens, top_k, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    for r, L in enumerate(lens):
        kk = min(top_k, L)
        assert bool((idx[r, kk:] == -1).all()) and bool((ref[r, kk:] == -1).all())
        got = torch.sort(logits[r, :L][idx[r, :kk].long()]).values
        want = torch.sort(logits[r, :L][ref[r, :kk].long()]).values
        assert torch.equal(got, want), f"row={r}: sliced arena differs from the copy"
        assert torch.equal(got, torch.topk(logits[r, :L], kk).values.sort().values)


@pytest.mark.skipif(
    not _cutlass_primitives_available(), reason="cutlass_primitives needs CUDA SM80+"
)
def test_cutlass_primitives_exported_helpers():
    """The lazily exported helpers on ``flashinfer.topk_varlen``:
    cutlass_primitives_row_order builds a permutation for a second batch
    size without recompiling (well under 2 s), and
    release_cutlass_primitives_resources reports the freed entries and
    leaves the backend usable."""
    import time

    from flashinfer import topk_varlen as tv

    N, top_k = 65536, 512
    gen = torch.Generator(device="cuda").manual_seed(5)
    for rows in (256, 96):
        seq_lens = torch.randint(
            top_k + 1, N + 1, (rows,), generator=gen, device="cuda"
        ).to(torch.int32)
        t0 = time.perf_counter()
        order = tv.cutlass_primitives_row_order(seq_lens, N)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        assert order.dtype == torch.int32 and tuple(order.shape) == (rows,)
        assert torch.equal(
            torch.sort(order).values,
            torch.arange(rows, device="cuda", dtype=torch.int32),
        )
        if rows == 96:
            assert elapsed < 2.0, (
                f"row_order recompiled for a new batch size ({elapsed:.1f}s)"
            )
    freed = tv.release_cutlass_primitives_resources()
    assert isinstance(freed, int) and freed >= 0
    logits = torch.randn(96, N, device="cuda", generator=gen) * 2.0
    idx, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="cutlass_primitives"
    )
    torch.cuda.synchronize()
    _check_correct(idx, logits, seq_lens, top_k, require_all_checked=True)


# ---------------------------------------------------------------------------
# cutlass_primitives: memory ownership and stream isolation
#
# Adversarial by design: garbage-filled workspaces, one workspace reused across every shape,
# graphs replayed after the workspace was scribbled on, several streams running one shape at
# once through replayed graphs (the only way to make launches overlap from Python), and a check
# that a call with a workspace and preallocated outputs touches the allocator not at all.
# Every combination of the API's options (next_n, compress_ratio, return_values, layout,
# dtype, preallocated or returned outputs, workspace or default) is exercised.
# ---------------------------------------------------------------------------

_CP_K = 512
_CP_OPTIONS = [  # (next_n, compress_ratio, return_values)
    (1, 1, False),
    (2, 1, True),
    (1, 4, False),
    (4, 2, True),
]
_CP_LAYOUTS = ["contiguous", "paged", "misaligned"]


def _cp_cells():
    """One (N, batch) per kernel the library's router can pick on this device: register,
    clustered register (or streaming where clusters are unavailable), streaming with the cluster
    merge, and streaming with the slab merge when some shape takes it here."""
    if not _cutlass_primitives_available():
        return {}
    try:
        from flashinfer.topk_varlen.cutlass_primitives.dispatch.device import (
            device_facts,
        )
        from flashinfer.topk_varlen.cutlass_primitives.topk.dispatch.router import (
            choose,
        )
    except ImportError:  # the router imports cutlass at module scope
        return {}

    cells = {"register": (4096, 8), "cluster": (65536, 8), "streaming": (262144, 64)}
    facts = device_facts(torch.device("cuda"))
    for n, rows in ((1 << 20, 8), (1 << 20, 16), (1 << 18, 8), (1 << 18, 64)):
        kind, config = choose(facts, torch.float32, _CP_K, n, rows)
        if kind == "streaming" and config.merge == "slab" and config.splits > 1:
            cells["slab"] = (n, rows)
            break
    return cells


_CP_CELLS = _cp_cells()


def _cp_logits(cell, dtype, layout, seed):
    """Logits for a cell in one of the three layouts the backend distinguishes: contiguous
    (read in place), a paged arena with wider rows (read in place through a storage view), and
    a slice at an odd column (copied into a padded arena)."""
    n, batch = _CP_CELLS[cell]
    g = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "contiguous":
        return (torch.randn(batch, n, device="cuda", generator=g) * 2.0).to(dtype)
    arena = (torch.randn(batch, n + 64, device="cuda", generator=g) * 2.0).to(dtype)
    return arena[:, :n] if layout == "paged" else arena[:, 1 : 1 + n]


def _cp_seq_lens(cell, next_n, compress_ratio, seed):
    """Ragged token lengths under which every row still holds a full top-k."""
    n, batch = _CP_CELLS[cell]
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = _CP_K * compress_ratio + next_n, n * compress_ratio + 1
    return torch.randint(lo, hi, (batch // next_n,), device="cuda", generator=g).to(
        torch.int32
    )


def _cp_workspace(logits, top_k=_CP_K, fill=0xA5, slack=0):
    from flashinfer.topk_varlen.kernels.cutlass_primitives_backend import (
        cutlass_primitives_workspace_bytes,
    )

    ws = torch.empty(
        cutlass_primitives_workspace_bytes(logits, top_k) + slack,
        dtype=torch.uint8,
        device="cuda",
    )
    return ws.fill_(fill)  # the backend must not rely on any prior content


def _cp_verify(
    indices, values, logits, seq_lens, next_n, compress_ratio, return_values
):
    _check_correct(
        indices,
        logits,
        seq_lens,
        _CP_K,
        next_n=next_n,
        compress_ratio=compress_ratio,
        require_all_checked=True,
    )
    if return_values:
        assert values is not None and values.dtype == logits.dtype
        for row in range(indices.shape[0]):
            assert torch.equal(logits[row][indices[row].long()], values[row]), (
                f"row={row}"
            )
    else:
        assert values is None


def _cp_run(
    logits, seq_lens, next_n, compress_ratio, return_values, workspace=None, **kw
):
    return flashinfer.top_k_varlen(
        logits,
        seq_lens,
        _CP_K,
        next_n=next_n,
        compress_ratio=compress_ratio,
        return_values=return_values,
        backend="cutlass_primitives",
        workspace=workspace,
        **kw,
    )


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("cell", list(_CP_CELLS))
@pytest.mark.parametrize("layout", _CP_LAYOUTS)
@pytest.mark.parametrize(
    "next_n, compress_ratio, return_values",
    _CP_OPTIONS,
    ids=[f"nn{a}_cr{b}_{'vals' if c else 'idx'}" for a, b, c in _CP_OPTIONS],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_cutlass_primitives_workspace_every_option(
    cell, layout, next_n, compress_ratio, return_values, dtype
):
    """Every option combination, once through the default caches and once through a
    garbage-filled caller workspace with preallocated outputs; both exact, and identical as
    multisets."""
    if dtype == torch.bfloat16 and cell not in ("register", "slab"):
        pytest.skip("16-bit covered on the register and slab cells")
    logits = _cp_logits(cell, dtype, layout, seed=11)
    seq_lens = _cp_seq_lens(cell, next_n, compress_ratio, seed=12)
    idx_a, val_a = _cp_run(logits, seq_lens, next_n, compress_ratio, return_values)
    ws = _cp_workspace(logits, fill=0xFF)
    rows = logits.shape[0]
    out_i = torch.full((rows, _CP_K), -9, dtype=torch.int32, device="cuda")
    out_v = torch.empty(rows, _CP_K, dtype=dtype, device="cuda")
    idx_b, val_b = _cp_run(
        logits,
        seq_lens,
        next_n,
        compress_ratio,
        return_values,
        workspace={"cutlass_primitives_workspace": ws, "gvr2_workspace": None},
        out_indices=out_i,
        out_values=out_v,
    )
    torch.cuda.synchronize()
    assert idx_b.data_ptr() == out_i.data_ptr()
    for idx, val in ((idx_a, val_a), (idx_b, val_b)):
        _cp_verify(idx, val, logits, seq_lens, next_n, compress_ratio, return_values)
    lf = logits.float()
    for row in range(rows):
        got = lf[row][idx_b[row].long()].sort().values
        want = lf[row][idx_a[row].long()].sort().values
        assert torch.equal(got, want), (
            f"row={row}: workspace and default answers differ"
        )


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("cell", list(_CP_CELLS))
def test_cutlass_primitives_workspace_bytes_and_validation(cell):
    """The reported size works and one byte less is refused before any launch; wrong device,
    non-contiguous, misaligned and non-tensor workspaces are refused; any dtype is accepted;
    a dict without our key, or with only other backends' keys, takes the default path."""
    from flashinfer.topk_varlen.kernels.cutlass_primitives_backend import (
        cutlass_primitives_workspace_bytes,
    )

    logits = _cp_logits(cell, torch.float32, "contiguous", seed=13)
    seq_lens = _cp_seq_lens(cell, 1, 1, seed=13)
    nbytes = cutlass_primitives_workspace_bytes(logits, _CP_K)
    assert nbytes > 0 and nbytes % 256 == 0
    # the misaligned layout needs the arena on top
    odd = _cp_logits(cell, torch.float32, "misaligned", seed=13)
    assert cutlass_primitives_workspace_bytes(odd, _CP_K) >= nbytes + odd.numel() * 4

    def run(ws):
        return _cp_run(logits, seq_lens, 1, 1, False, workspace=ws)

    idx, _ = run({"cutlass_primitives_workspace": _cp_workspace(logits)})
    torch.cuda.synchronize()
    _check_correct(idx, logits, seq_lens, _CP_K, require_all_checked=True)
    with pytest.raises(ValueError, match="needed"):
        run(
            {
                "cutlass_primitives_workspace": torch.empty(
                    nbytes - 1, dtype=torch.uint8, device="cuda"
                )
            }
        )
    with pytest.raises(ValueError, match="live on"):
        run({"cutlass_primitives_workspace": torch.empty(nbytes, dtype=torch.uint8)})
    with pytest.raises(ValueError, match="contiguous"):
        run(
            {
                "cutlass_primitives_workspace": torch.empty(
                    2 * nbytes, dtype=torch.uint8, device="cuda"
                )[::2]
            }
        )
    with pytest.raises(ValueError, match="aligned"):
        run(
            {
                "cutlass_primitives_workspace": torch.empty(
                    nbytes + 16, dtype=torch.uint8, device="cuda"
                )[4:]
            }
        )
    with pytest.raises(TypeError):
        run({"cutlass_primitives_workspace": bytearray(nbytes)})
    for dtype in (torch.float32, torch.int64, torch.bfloat16):
        esize = torch.tensor([], dtype=dtype).element_size()
        ws = torch.empty(-(-nbytes // esize), dtype=dtype, device="cuda")
        idx, _ = run({"cutlass_primitives_workspace": ws})
        torch.cuda.synchronize()
        _check_correct(idx, logits, seq_lens, _CP_K, require_all_checked=True)
    big = torch.empty(1024 + nbytes, dtype=torch.uint8, device="cuda")
    idx, _ = run(
        {"cutlass_primitives_workspace": big[768:]}
    )  # a 256-byte multiple into a larger buffer
    torch.cuda.synchronize()
    _check_correct(idx, logits, seq_lens, _CP_K, require_all_checked=True)
    for ws in ({}, {"gvr2_workspace": torch.empty(8, device="cuda")}, None):
        idx, _ = run(ws)
        torch.cuda.synchronize()
        _check_correct(idx, logits, seq_lens, _CP_K, require_all_checked=True)
    # zero rows: nothing needed, nothing launched
    empty = logits[:0]
    assert cutlass_primitives_workspace_bytes(empty, _CP_K) == 0
    idx, _ = _cp_run(
        empty,
        seq_lens[:0],
        1,
        1,
        False,
        workspace={"cutlass_primitives_workspace": big},
    )
    assert idx.shape == (0, _CP_K)


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("cell", list(_CP_CELLS))
@pytest.mark.parametrize("layout", _CP_LAYOUTS)
def test_cutlass_primitives_workspace_call_allocates_nothing(cell, layout):
    """With outputs and workspace supplied, a warmed call touches the allocator not at all: the
    caller fully controls device memory, including the padded copy of a misaligned input."""
    logits = _cp_logits(cell, torch.float32, layout, seed=14)
    seq_lens = _cp_seq_lens(cell, 1, 1, seed=14)
    rows = logits.shape[0]
    ws = {"cutlass_primitives_workspace": _cp_workspace(logits)}
    out_i = torch.empty(rows, _CP_K, dtype=torch.int32, device="cuda")
    out_v = torch.empty(rows, _CP_K, dtype=torch.float32, device="cuda")
    kw = dict(workspace=ws, out_indices=out_i, out_values=out_v)
    _cp_run(logits, seq_lens, 1, 1, True, **kw)  # compiles
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        _cp_run(logits, seq_lens, 1, 1, True, **kw)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    _cp_verify(out_i, out_v, logits, seq_lens, 1, 1, True)


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
def test_cutlass_primitives_one_workspace_serves_every_shape():
    """A workspace sized for the largest need serves every cell, dtype, layout and option in
    any order, with garbage written between calls."""
    from flashinfer.topk_varlen.kernels.cutlass_primitives_backend import (
        cutlass_primitives_workspace_bytes,
    )

    problems = []
    for cell in _CP_CELLS:
        for dtype in (torch.float32, torch.bfloat16):
            for layout in ("contiguous", "misaligned"):
                problems.append((cell, _cp_logits(cell, dtype, layout, seed=15)))
    need = max(cutlass_primitives_workspace_bytes(x, _CP_K) for _, x in problems)
    ws = torch.empty(need, dtype=torch.uint8, device="cuda")
    for i, (cell, logits) in enumerate(problems + problems[::-1]):
        next_n, compress_ratio, return_values = _CP_OPTIONS[i % len(_CP_OPTIONS)]
        seq_lens = _cp_seq_lens(cell, next_n, compress_ratio, seed=16 + i)
        ws.fill_(0xC3)
        idx, val = _cp_run(
            logits,
            seq_lens,
            next_n,
            compress_ratio,
            return_values,
            workspace={"cutlass_primitives_workspace": ws},
        )
        torch.cuda.synchronize()
        _cp_verify(idx, val, logits, seq_lens, next_n, compress_ratio, return_values)


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("cell", list(_CP_CELLS))
@pytest.mark.parametrize("with_workspace", [False, True], ids=["cached", "workspace"])
def test_cutlass_primitives_cuda_graph_scribbled_workspace(cell, with_workspace):
    """A graph captured with a workspace re-zeroes the counters on every replay, so scribbling
    on the workspace between replays cannot break the merge; the cached path replays too."""
    logits = _cp_logits(cell, torch.float32, "paged", seed=17)
    seq_lens = _cp_seq_lens(cell, 2, 1, seed=17)
    rows = logits.shape[0]
    ws = _cp_workspace(logits) if with_workspace else None
    wsd = {"cutlass_primitives_workspace": ws} if with_workspace else None
    out_i = torch.empty(rows, _CP_K, dtype=torch.int32, device="cuda")
    out_v = torch.empty(rows, _CP_K, dtype=torch.float32, device="cuda")

    def call():
        _cp_run(
            logits,
            seq_lens,
            2,
            1,
            True,
            workspace=wsd,
            out_indices=out_i,
            out_values=out_v,
        )

    s = torch.cuda.Stream()
    torch.cuda.synchronize()  # inputs and workspace come from the default stream
    with torch.cuda.stream(s):
        call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        call()
    for fill in (0xFF, 0x01, 0x80):
        if ws is not None:
            ws.fill_(fill)
        out_i.fill_(-9)
        g.replay()
        torch.cuda.synchronize()
        _cp_verify(out_i, out_v, logits, seq_lens, 2, 1, True)
        logits.copy_(_cp_logits(cell, torch.float32, "paged", seed=fill))


def _cp_concurrent_streams(cell, nstreams, workspaces, launches=12):
    """nstreams graphs of one shape, each replayed on its own stream at the same time; every
    stream's answer must be exact.  Replayed graphs are the only launches cheap enough to
    overlap from Python; they exercise exactly the overlap the per-stream buffers must survive."""
    xs = [
        _cp_logits(cell, torch.float32, "contiguous", seed=100 + i)
        for i in range(nstreams)
    ]
    seq_lens = _cp_seq_lens(cell, 1, 1, seed=100)
    rows = xs[0].shape[0]
    outs = [torch.full((rows, _CP_K), -9, dtype=torch.int32, device="cuda") for _ in xs]
    wsds = [
        {"cutlass_primitives_workspace": _cp_workspace(x)} if workspaces else None
        for x in xs
    ]
    streams = [torch.cuda.Stream() for _ in xs]
    # the inputs were produced on the default stream: order the side streams behind it (the
    # usual cross-stream rule; without it a busy GPU lets a launch read lengths still being
    # generated and pad the row)
    torch.cuda.synchronize()
    graphs = []
    for x, out, wsd, s in zip(xs, outs, wsds, streams, strict=True):
        with torch.cuda.stream(s):
            _cp_run(
                x, seq_lens, 1, 1, False, workspace=wsd, out_indices=out
            )  # this stream's cache
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(launches):
                _cp_run(x, seq_lens, 1, 1, False, workspace=wsd, out_indices=out)
        graphs.append(g)
    for _ in range(3):
        for out in outs:
            out.fill_(-9)
        torch.cuda.synchronize()  # the fills ran on the default stream
        for g, s in zip(graphs, streams, strict=True):
            with torch.cuda.stream(s):
                g.replay()
        torch.cuda.synchronize()
        for x, out in zip(xs, outs, strict=True):
            _check_correct(out, x, seq_lens, _CP_K, require_all_checked=True)


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("cell", list(_CP_CELLS))
@pytest.mark.parametrize("nstreams", [2, 4])
def test_cutlass_primitives_default_buffers_are_stream_private(cell, nstreams):
    """The default path with no caller involvement: two to four streams running the same shape
    at once must not share slab, counters or status (they did before the per-stream keying:
    the slab cell then merged mixed segments and returned wrong indices)."""
    _cp_concurrent_streams(cell, nstreams, workspaces=False)


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("cell", list(_CP_CELLS))
def test_cutlass_primitives_caller_workspaces_one_per_stream(cell):
    _cp_concurrent_streams(cell, 3, workspaces=True)


def _cp_ordered_cell():
    """A shape the router sends to the one-CTA-per-row streaming kernel, the only one that takes
    a caller row order; None when this device routes every candidate elsewhere."""
    from flashinfer.topk_varlen.cutlass_primitives.dispatch.device import device_facts
    from flashinfer.topk_varlen.cutlass_primitives.topk.dispatch.router import choose

    facts = device_facts(torch.device("cuda"))
    for n, rows in ((65536, 256), (131072, 256), (262144, 512), (32768, 512)):
        kind, config = choose(facts, torch.float32, _CP_K, n, rows)
        if kind == "streaming" and config.splits == 1:
            return n, rows
    return None


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("order", ["longest_first", "random", "reversed"])
def test_cutlass_primitives_row_order(order):
    """``workspace["cutlass_primitives_row_order"]`` sets the order in which the streaming
    kernel's CTAs take rows: the result is exact under any permutation, the same call without
    the key still runs, the tensor is validated, and a graph captured with the key replays."""
    cell = _cp_ordered_cell()
    if cell is None:
        pytest.skip("no shape routes to the unsplit streaming kernel on this device")
    n, batch = cell
    logits, _, seq_lens = _make_inputs(batch, n, _CP_K, torch.float32, seed=71)
    if order == "longest_first":
        perm = torch.argsort(seq_lens, descending=True).to(torch.int32)
    elif order == "random":
        perm = torch.randperm(batch, device="cuda").to(torch.int32)
    else:
        perm = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
    key = "cutlass_primitives_row_order"
    out, _ = _cp_run(logits, seq_lens, 1, 1, False, workspace={key: perm})
    _check_correct(out, logits, seq_lens, _CP_K, require_all_checked=True)
    # with a caller workspace too, and again without the key (the unordered kernel)
    ws = {key: perm, "cutlass_primitives_workspace": _cp_workspace(logits)}
    out2, _ = _cp_run(logits, seq_lens, 1, 1, False, workspace=ws)
    _check_correct(out2, logits, seq_lens, _CP_K, require_all_checked=True)
    out3, _ = _cp_run(logits, seq_lens, 1, 1, False)
    _check_correct(out3, logits, seq_lens, _CP_K, require_all_checked=True)
    for bad in (perm.to(torch.int64), perm[:-1], perm.cpu()):
        with pytest.raises(ValueError, match="row_order"):
            _cp_run(logits, seq_lens, 1, 1, False, workspace={key: bad})
    # a CUDA graph holding the ordered kernel replays with fresh data
    out_g = torch.empty(batch, _CP_K, dtype=torch.int32, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        _cp_run(logits, seq_lens, 1, 1, False, workspace={key: perm}, out_indices=out_g)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            _cp_run(
                logits, seq_lens, 1, 1, False, workspace={key: perm}, out_indices=out_g
            )
    torch.cuda.synchronize()
    for seed in (72, 73):
        new, _, _ = _make_inputs(batch, n, _CP_K, torch.float32, seed=seed)
        logits.copy_(new)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        _check_correct(out_g, logits, seq_lens, _CP_K, require_all_checked=True)


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("balanced", [True, False], ids=["balanced", "longest_first"])
@pytest.mark.parametrize("next_n", [1, 2])
def test_cutlass_primitives_row_order_helper(balanced, next_n):
    """``cutlass_primitives_row_order`` builds the row-order key from the lengths alone: a
    permutation, longest first (the SM-count longest rows first, then shortest first when
    balanced), one entry per ``next_n`` rows with the radix backends' effective lengths, exact
    results under it, in-place refresh into ``out`` so a captured graph sees new lengths."""
    from flashinfer.topk_varlen.cutlass_primitives.dispatch.device import device_facts
    from flashinfer.topk_varlen.kernels.cutlass_primitives_backend import (
        cutlass_primitives_row_order,
    )

    cell = _cp_ordered_cell()
    if cell is None:
        pytest.skip("no shape routes to the unsplit streaming kernel on this device")
    n, batch = cell
    batch -= batch % next_n
    logits, _, seq_lens = _make_inputs(batch, n, _CP_K, torch.float32, seed=74)
    req_lens = seq_lens[::next_n].contiguous()  # one length per request
    order = cutlass_primitives_row_order(req_lens, n, next_n=next_n, balanced=balanced)
    assert order.dtype == torch.int32 and tuple(order.shape) == (batch,)
    assert torch.equal(
        torch.sort(order).values, torch.arange(batch, device="cuda", dtype=torch.int32)
    )
    eff = (
        req_lens.repeat_interleave(next_n)
        - next_n
        + torch.arange(batch, device="cuda") % next_n
        + 1
    ).clamp(0, n)
    ranked = eff[order.long()]
    sms = device_facts(torch.device("cuda")).sm_count
    if balanced and batch > sms:
        # the first SM-count rows are the longest (non-increasing up to bucket ties), the rest
        # non-decreasing, and the shortest rows pair with the longest
        assert ranked[:sms].min() >= ranked[sms:].max() - n // 256
        assert (ranked[sms:].diff() >= -(n // 256)).all()
    else:
        assert (
            ranked.diff() <= n // 256
        ).all()  # non-increasing up to one bucket's width
    key = "cutlass_primitives_row_order"
    out, _ = _cp_run(logits, req_lens, next_n, 1, False, workspace={key: order})
    # the checker takes per-row lengths: the effective lengths under next_n, not seq_lens
    _check_correct(out, logits, eff.to(torch.int32), _CP_K, require_all_checked=True)
    # in-place refresh keeps the buffer a graph captured; rows within one length bucket come
    # out in atomic (arbitrary) order, so two calls agree on the bucket sequence, not bitwise
    buf = torch.empty(batch, dtype=torch.int32, device="cuda")
    ret = cutlass_primitives_row_order(
        req_lens, n, next_n=next_n, balanced=balanced, out=buf
    )
    assert ret.data_ptr() == buf.data_ptr()
    assert torch.equal(torch.sort(buf).values, torch.sort(order).values)
    bucket = lambda o: (eff[o.long()] * 255) // n  # noqa: E731
    assert torch.equal(bucket(buf), bucket(order))
    with pytest.raises(ValueError, match="num_rows"):
        cutlass_primitives_row_order(
            req_lens, n, next_n=next_n, num_rows=batch + next_n
        )
    with pytest.raises(ValueError, match="out must"):
        cutlass_primitives_row_order(req_lens, n, next_n=next_n, out=buf[:-1])


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_cutlass_primitives_census_split_cell(dtype):
    """Large k routes to the library's census split kernel (two launches): exact with values,
    next_n, a garbage-filled caller workspace with preallocated outputs leaving the allocator
    untouched, and four streams replaying graphs of the shape at once."""
    from flashinfer.topk_varlen.cutlass_primitives.dispatch.device import device_facts
    from flashinfer.topk_varlen.cutlass_primitives.topk.dispatch.router import choose
    from flashinfer.topk_varlen.kernels.cutlass_primitives_backend import (
        cutlass_primitives_workspace_bytes,
    )

    N, batch, top_k, next_n = 262144, 8, 100000, 2
    kind, _ = choose(device_facts(torch.device("cuda")), dtype, top_k, N, batch)
    assert kind == "census_split"
    logits, _, seq_lens = _make_inputs(batch, N, top_k, dtype, seed=61, next_n=next_n)
    ws = torch.empty(
        cutlass_primitives_workspace_bytes(logits, top_k),
        dtype=torch.uint8,
        device="cuda",
    ).fill_(0xE7)
    out_i = torch.empty(batch, top_k, dtype=torch.int32, device="cuda")
    out_v = torch.empty(batch, top_k, dtype=dtype, device="cuda")
    kw = dict(
        next_n=next_n,
        return_values=True,
        backend="cutlass_primitives",
        workspace={"cutlass_primitives_workspace": ws},
        out_indices=out_i,
        out_values=out_v,
    )
    flashinfer.top_k_varlen(logits, seq_lens, top_k, **kw)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    ws.fill_(0x11)
    flashinfer.top_k_varlen(logits, seq_lens, top_k, **kw)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    _check_correct(
        out_i, logits, seq_lens, top_k, next_n=next_n, require_all_checked=True
    )
    for row in range(batch):
        assert torch.equal(logits[row][out_i[row].long()], out_v[row])
    # the default caches, four streams at once
    xs = [_cp_logits_shape(N, batch, dtype, seed=70 + i) for i in range(4)]
    lens = torch.full((batch,), N, dtype=torch.int32, device="cuda")
    outs = [
        torch.full((batch, top_k), -9, dtype=torch.int32, device="cuda") for _ in xs
    ]
    streams = [torch.cuda.Stream() for _ in xs]
    torch.cuda.synchronize()
    graphs = []
    for x, out, s in zip(xs, outs, streams, strict=True):
        with torch.cuda.stream(s):
            flashinfer.top_k_varlen(
                x, lens, top_k, backend="cutlass_primitives", out_indices=out
            )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(6):
                flashinfer.top_k_varlen(
                    x, lens, top_k, backend="cutlass_primitives", out_indices=out
                )
        graphs.append(g)
    for _ in range(3):
        for out in outs:
            out.fill_(-9)
        torch.cuda.synchronize()
        for g, s in zip(graphs, streams, strict=True):
            with torch.cuda.stream(s):
                g.replay()
        torch.cuda.synchronize()
        for x, out in zip(xs, outs, strict=True):
            _check_correct(out, x, lens, top_k, require_all_checked=True)


def _cp_logits_shape(N, batch, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(batch, N, device="cuda", generator=g) * 2.0).to(dtype)


@pytest.mark.skipif(not _CP_CELLS, reason="cutlass_primitives needs CUDA SM80+")
def test_cutlass_primitives_streams_and_shapes_interleaved():
    """Shapes, options and streams interleaved in one loop through the default caches: no
    buffer is ever picked up by the wrong shape or stream."""
    from flashinfer.topk_varlen.kernels import cutlass_primitives_backend as CPB

    streams = [torch.cuda.Stream() for _ in range(2)]
    problems = []
    for i, cell in enumerate(_CP_CELLS):
        next_n, compress_ratio, return_values = _CP_OPTIONS[i % len(_CP_OPTIONS)]
        logits = _cp_logits(cell, torch.float32, "contiguous", seed=20 + i)
        seq_lens = _cp_seq_lens(cell, next_n, compress_ratio, seed=20 + i)
        problems.append(
            (cell, logits, seq_lens, next_n, compress_ratio, return_values, [])
        )
    torch.cuda.synchronize()  # inputs come from the default stream; the launches go to others
    for _ in range(3):
        for (
            _,
            logits,
            seq_lens,
            next_n,
            compress_ratio,
            return_values,
            results,
        ) in problems:
            for s in streams:
                with torch.cuda.stream(s):
                    results.append(
                        _cp_run(logits, seq_lens, next_n, compress_ratio, return_values)
                    )
    torch.cuda.synchronize()
    for (
        cell,
        logits,
        seq_lens,
        next_n,
        compress_ratio,
        return_values,
        results,
    ) in problems:
        for j, (idx, val) in enumerate(results):
            try:
                _cp_verify(
                    idx, val, logits, seq_lens, next_n, compress_ratio, return_values
                )
            except AssertionError as e:
                pytest.fail(
                    f"{cell} {_CP_CELLS[cell]} nn={next_n} cr={compress_ratio} "
                    f"vals={return_values} launch {j}: {e}"
                )
    # the status cache holds one entry per stream for each (rows, words)
    handles = {s.cuda_stream for s in streams}
    for _, logits, *_ in problems:
        rows = logits.shape[0]
        seen = {key[1] for key in CPB._status if key[2] == rows}
        assert handles <= seen, f"rows={rows}: status buffers not keyed per stream"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("return_values", [False, True], ids=["idx", "vals"])
@pytest.mark.parametrize("next_n,compress_ratio", [(1, 1), (3, 1), (1, 4), (3, 4)])
def test_sglang_backend_grouped_rows(next_n, compress_ratio, return_values):
    """backend="sglang": the launcher kernel derives every row's length on
    device from the request-level ``seq_lens`` (next_n / compress_ratio /
    clamp to [0, N]) -- the same grouped-row convention as the other
    backends, with no per-call host tensor ops.  Padded or evicted requests
    (seq_len below next_n) give all -1 rows, over-long requests clamp to N,
    and a sentinel arena checks that nothing lands outside the caller's
    buffer.  With ``return_values`` the dispatcher gathers the selected
    logits and writes 0 at the -1 slots."""
    _skip_unless_backend("sglang")
    top_k, N = 512, 8192
    cr = compress_ratio
    req_lens = [0, 1, 2, next_n - 1, top_k * cr + 2, N * cr, N * cr + 7]
    num_req = len(req_lens)
    num_rows = num_req * next_n
    torch.manual_seed(7)
    logits = torch.randn(num_rows, N, dtype=torch.float32, device="cuda")
    seq_lens = torch.tensor(req_lens, dtype=torch.int32, device="cuda")
    sentinel = 0x7EADBEEF
    arena = torch.full(
        (num_rows + 2, top_k), sentinel, dtype=torch.int32, device="cuda"
    )
    out = arena[1 : num_rows + 1]
    idx, vals = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        backend="sglang",
        next_n=next_n,
        compress_ratio=cr,
        return_values=return_values,
        out_indices=out,
    )
    torch.cuda.synchronize()
    assert idx.data_ptr() == out.data_ptr()
    assert (arena[0] == sentinel).all() and (arena[-1] == sentinel).all(), (
        "write landed outside the caller's buffer"
    )
    if return_values:
        assert vals.shape == (num_rows, top_k) and vals.dtype == torch.float32
    else:
        assert vals is None
    arena_cpu = arena.cpu()
    for r in range(num_rows):
        length = min(
            N, max(0, (req_lens[r // next_n] - next_n + (r % next_n) + 1) // cr)
        )
        row = arena_cpu[r + 1].tolist()
        assert sentinel not in row, f"row={r}: slot never written"
        valid = sorted(i for i in row if i >= 0)
        assert row.count(-1) == top_k - len(valid), f"row={r}: bad -1 padding"
        if length <= top_k:
            assert valid == list(range(length)), f"row={r}: length={length}"
        else:
            assert len(valid) == top_k and valid[-1] < length, (
                f"row={r}: index past length"
            )
            got = logits[r][torch.tensor(valid, device="cuda")].sort().values
            ref = logits[r, :length].topk(top_k).values.sort().values
            assert torch.equal(got, ref), f"row={r}: wrong top-k value set"
        if return_values:
            pad = idx[r] < 0
            assert bool((vals[r][pad] == 0).all()), f"row={r}: padded values not 0"
            assert torch.equal(vals[r][~pad], logits[r][idx[r][~pad].long()]), (
                f"row={r}: values do not match logits[row, indices]"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_sglang_backend_rejects_unsupported_inputs():
    """Explicit backend="sglang" refuses what its kernel cannot run -- fp32
    only, rows of whole 16-byte vectors, top_k <= 2048, contiguous and
    16-byte-aligned 2-D logits, CUDA seq_lens -- with the API's ValueError /
    BackendSupportedError, never a binding-level error or a device fault
    (a misaligned LDG.128 is a sticky CUDA error for the whole process)."""
    from flashinfer.utils import BackendSupportedError

    _skip_unless_backend("sglang")
    rows, N, top_k = 4, 4096, 512
    gen = torch.Generator(device="cuda").manual_seed(4098)
    lens = torch.full((rows,), N, dtype=torch.int32, device="cuda")

    def lg(n, dtype=torch.float32):
        x = torch.randn(rows, n, dtype=torch.float32, device="cuda", generator=gen)
        return x.to(dtype)

    wide = lg(N + 64)
    buf = torch.randn(rows * N + 4, device="cuda", generator=gen)
    cases = {
        "width_not_whole_vectors": (lg(4098), torch.full_like(lens, 4098), top_k),
        "top_k_above_2048": (lg(N), lens, 2049),
        "bfloat16": (lg(N, torch.bfloat16), lens, top_k),
        "non_contiguous": (wide[:, :N], lens, top_k),
        "shifted_4_bytes": (buf[1 : 1 + rows * N].view(rows, N), lens, top_k),
        "one_dimensional": (lg(N)[0], lens[:1], top_k),
        "seq_lens_on_cpu": (lg(N), lens.cpu(), top_k),
    }
    assert not cases["non_contiguous"][0].is_contiguous()
    assert cases["shifted_4_bytes"][0].is_contiguous()
    assert cases["shifted_4_bytes"][0].data_ptr() & 15
    for name, (x, sl, k) in cases.items():
        try:
            flashinfer.top_k_varlen(x, sl, k, backend="sglang")
            torch.cuda.synchronize()
        except (ValueError, BackendSupportedError):
            continue
        pytest.fail(f"{name}: backend='sglang' accepted an input its kernel cannot run")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16], ids=["f32", "bf16", "f16"]
)
@pytest.mark.parametrize("N", [16392, 65536])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_walkfirst_register_arm_variable_lengths(N, top_k, dtype):
    """walkfirst_primitives, rows far shorter than the padded width: the
    census arm (rows <= 4K) and the register-resident arm (4K < rows <= 16K,
    one read of the row bounded by the REAL length) must be exact at every
    arm boundary (identity at length <= k, the 4K vector boundary, the 16K
    arm cutoff and the walk pipeline just above it -- rows 16385 and 16392
    at the narrow width), pad with -1, stay inside the caller's buffer
    (sentinel arena), and certify every row (status 0: no exact fallback, no
    flood refine).  The 16-bit dtypes take the 8-element vector / 13-bit
    coarse-bin path with its 8-element tails (N stays a multiple of 8)."""
    _skip_unless_backend("walkfirst_primitives")
    from flashinfer.topk_varlen.topk_varlen import _prim_status

    lens = [0, 1, top_k - 1, top_k, top_k + 1, 700, 2047, 2048, 2049, 4095, 4096, 4097]
    lens += [8191, 8192, 8193, 12000, 16383, 16384, 16385, N]
    lens = [min(v, N) for v in lens]
    rows = len(lens)
    torch.manual_seed(3)
    logits = torch.randn(rows, N, dtype=torch.float32, device="cuda").to(dtype)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    sentinel = 0x7EADBEEF
    arena = torch.full((rows + 2, top_k), sentinel, dtype=torch.int32, device="cuda")
    out = arena[1 : rows + 1]
    idx, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, backend="walkfirst_primitives", out_indices=out
    )
    torch.cuda.synchronize()
    assert idx.data_ptr() == out.data_ptr()
    assert (arena[0] == sentinel).all() and (arena[-1] == sentinel).all(), (
        "write landed outside the caller's buffer"
    )
    status = _prim_status(rows, logits.device)
    assert int((status[:rows] != 0).sum()) == 0, "a row took the exact fallback"
    arena_cpu = arena.cpu()
    for r, length in enumerate(lens):
        row = arena_cpu[r + 1].tolist()
        assert sentinel not in row, f"row={r}: slot never written"
        valid = sorted(i for i in row if i >= 0)
        kk = min(top_k, length)
        assert row.count(-1) == top_k - kk, f"row={r}: bad -1 padding"
        assert len(valid) == kk and len(set(valid)) == kk, f"row={r}: dup/short"
        if kk == 0:
            continue
        assert valid[-1] < length, f"row={r}: index past length={length}"
        got = logits[r][torch.tensor(valid, device="cuda")].sort().values
        ref = logits[r, :length].topk(kk).values.sort().values
        assert torch.equal(got, ref), f"row={r}: wrong top-k value set"
