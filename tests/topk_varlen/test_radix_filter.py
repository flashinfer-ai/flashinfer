# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Focused regression tests for the `radix_filter` top-k backend."""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


def _radix_filter_dsl_ok() -> bool:
    """Dynamic probe: does the installed CuTe DSL support the vendored kernels?

    is_backend_supported() is static (registration + CC lists only), so it
    stays True on a supported arch even when nvidia-cutlass-dsl < 4.8 -- and
    the API then rejects every call via the backend's fail-closed DSL check.
    Tests must skip on such environments, not fail.
    """
    from flashinfer.topk_varlen.topk_varlen import _radix_filter_kernel_dsl_ok

    return _radix_filter_kernel_dsl_ok()


def _skip_unless_radix_filter(device: torch.device) -> None:
    cc = get_compute_capability(device)
    if cc[0] * 10 + cc[1] not in (100, 103, 107):
        pytest.skip("radix_filter requires SM100/SM103/SM107")
    if not flashinfer.top_k_varlen.is_backend_supported(
        "radix_filter", cc[0] * 10 + cc[1]
    ):
        pytest.skip("radix_filter not supported in this environment")
    if not _radix_filter_dsl_ok():
        pytest.skip("radix_filter requires nvidia-cutlass-dsl >= 4.8")


@pytest.mark.parametrize(
    "dtype,N",
    [
        # Row stride (N * element_size) deliberately NOT a multiple of the
        # kernel's 32-byte copy alignment, so successive rows start at varying
        # misalignments and the scalar prologue spans up to 7 (fp32) or 15
        # (fp16/bf16) elements.
        (torch.float32, 101),
        (torch.float16, 105),
        (torch.bfloat16, 105),
    ],
)
def test_short_misaligned_rows(dtype, N):
    """Short misaligned rows must not scan past their valid length.

    ``prologue_elems`` is derived from address alignment alone; on a row with
    ``top_k < seq_len < prologue span`` an unclamped prologue reads elements
    beyond ``seq_len`` into the coarse histogram, so indices past the row's
    valid length can be emitted as top-k results (and the final row can read
    past the allocation). Elements beyond each row's length are planted with a
    large sentinel so any over-read is guaranteed to surface in the output
    rather than depending on random values.

    Regression for PR #4621 review (prologue clamp in every mode).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    torch.manual_seed(7)
    top_k = 4
    # Lengths straddle the maximum possible prologue span, all > top_k so the
    # trivial-case shortcut cannot mask the scan path. Kept at 5/6 so that for
    # every tested (dtype, N) at least one row has prologue span > length
    # (e.g. fp32/N=101: row 5 starts 4 bytes past a 32-byte boundary, giving a
    # 7-element prologue against a 6-element row).
    lens = [5, 6, 5, 6, 5, 6, 5, 6]
    B = len(lens)
    logits = torch.randn(B, N, dtype=dtype, device=device)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    # Sentinel: anything the kernel reads beyond a row's valid length would
    # dominate the top-k and show up as an out-of-range index.
    for b, n in enumerate(lens):
        logits[b, n:] = 100.0

    idx, vals = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, return_values=True, backend="radix_filter"
    )
    idx = idx.view(B, top_k)
    vals = vals.view(B, top_k)

    for b, n in enumerate(lens):
        sel = idx[b][idx[b] >= 0]
        assert sel.numel() == top_k, f"row {b}: expected {top_k} indices"
        assert int(sel.max()) < n, (
            f"row {b} (len={n}): index {int(sel.max())} beyond the valid "
            f"length -- the prologue scanned past the row"
        )
        assert sel.unique().numel() == sel.numel()
        ref = torch.topk(logits[b, :n].float(), top_k).values
        got = torch.sort(vals[b].float(), descending=True).values
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)


def test_num_sms_cache_is_per_device():
    """The SM-count memo must be keyed by device, not first-caller-wins.

    On a heterogeneous multi-GPU host, a process-wide scalar would pin the
    first device's SM count and mis-size the persistent grid / occupancy-mode
    decision for every other GPU. Host-only (no kernels launched), so it runs
    on any multi-GPU box; it has real teeth wherever the visible devices have
    differing SM counts (the unfixed code returns one count for all devices).

    Regression for PR #4621 review (per-device SM-count cache).
    """
    if not _radix_filter_dsl_ok():
        pytest.skip("radix_filter kernels require nvidia-cutlass-dsl >= 4.8")

    from flashinfer.topk_varlen.kernels.filtered_topk_decode import _get_num_sms

    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 visible CUDA devices")

    for i in range(torch.cuda.device_count()):
        expected = torch.cuda.get_device_properties(i).multi_processor_count
        got = _get_num_sms(torch.device("cuda", i))
        assert got == expected, (
            f"device {i}: cached SM count {got} != actual {expected} -- "
            f"the cache is not keyed by device"
        )


def test_compile_uses_persistent_jit_cache():
    """radix_filter compiles must go through FlashInfer's persistent JIT cache.

    A bare ``cute.compile`` keeps the kernel only in process memory: every new
    process (e.g. each serving worker) recompiles on first use, and nothing
    honors architecture/DSL/source invalidation or FLASHINFER_DISABLE_JIT.
    After one call through the public API, the exported artifact must exist in
    the on-disk module directory (tagged by compile architecture). Fails on
    the unrouted code, which writes no artifact.

    Regression for PR #4621 review (persistent JIT routing + arch in key).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    from flashinfer.jit import env as jit_env

    torch.manual_seed(3)
    logits = torch.randn(4, 8192, dtype=torch.float32, device=device)
    seq_lens = torch.full((4,), 8192, dtype=torch.int32, device=device)
    flashinfer.top_k_varlen(logits, seq_lens, 512, backend="radix_filter")

    module_dirs = list(jit_env.FLASHINFER_JIT_DIR.glob("radix_filter_topk_*_cute_dsl"))
    assert module_dirs, (
        f"no radix_filter_topk module directory under "
        f"{jit_env.FLASHINFER_JIT_DIR} -- compiles are not routed through the "
        f"persistent JIT cache"
    )
    objs = [o for d in module_dirs for o in d.glob("*.o")]
    assert objs, f"module dir {module_dirs} contains no exported kernel objects"


def test_strided_and_misaligned_inputs():
    """Padded row views are zero-copy ABI; misaligned bases are materialized.

    A framework score buffer sliced to the vocab width (leading stride wider
    than the row) must produce identical results to the compact tensor -- the
    kernel ABI now declares a symbolic leading stride, so this pins declared
    behavior rather than an accident of the runtime. A base pointer sliced off
    32-byte alignment previously failed late with an opaque FFI alignment
    error; the wrapper now materializes it.

    Regression for PR #4621 review (strided/offset input ABI).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    torch.manual_seed(11)
    B, N, k = 4, 8192, 512
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    # Padded view: rows 1+ distinguish the true stride from a compact
    # misinterpretation (row 0 is at offset 0 either way).
    buf = torch.randn(B, 2 * N, dtype=torch.float32, device=device)
    view = buf[:, :N]
    assert not view.is_contiguous()
    idx, _ = flashinfer.top_k_varlen(view, seq_lens, k, backend="radix_filter")
    idx = idx.view(B, k)
    for b in range(B):
        sel = idx[b][idx[b] >= 0]
        kth = torch.topk(view[b], k).values.min()
        assert bool((view[b][sel.long()] >= kth - 1e-4).all()), (
            f"row {b}: padded-view results do not match the view's own data"
        )

    # Misaligned base: previously an opaque late ValueError from the FFI.
    base = torch.randn(B * N + 1, dtype=torch.float32, device=device)
    mis = base[1:].view(B, N)
    assert mis.data_ptr() % 32 != 0
    idx, _ = flashinfer.top_k_varlen(mis, seq_lens, k, backend="radix_filter")
    idx = idx.view(B, k)
    for b in range(B):
        sel = idx[b][idx[b] >= 0]
        kth = torch.topk(mis[b], k).values.min()
        assert bool((mis[b][sel.long()] >= kth - 1e-4).all())


def test_truncate_policy_rejects_topk_equal_to_smem_capacity():
    """TRUNCATE must require top_k strictly below the SMEM candidate capacity.

    The fine-threshold search selects the bin whose inclusive cumulative count
    STRICTLY exceeds the remaining k; truncation to exactly top_k candidates
    makes the total equal k, so no bin qualifies and refinement consumes stale
    control state. Host-only (constructor validation; nothing is compiled).

    Regression for PR #4621 review (TRUNCATE boundary).
    """
    if not _radix_filter_dsl_ok():
        pytest.skip("radix_filter kernels require nvidia-cutlass-dsl >= 4.8")
    import cutlass

    from flashinfer.topk_varlen.kernels.filtered_topk_util import (
        FilteredTopKKernelVarlen,
    )

    probe = FilteredTopKKernelVarlen(cutlass.Float32, 1 << 20, 512)
    S = probe.filtered_topk_smem_input_size
    assert S >= 512

    # Equality must now be rejected up front...
    with pytest.raises(ValueError, match=r"requires top_k \(\d+\) <"):
        FilteredTopKKernelVarlen(
            cutlass.Float32, 1 << 20, S, overflow_policy="TRUNCATE"
        )
    # ...while strictly-below remains accepted.
    FilteredTopKKernelVarlen(
        cutlass.Float32, 1 << 20, S - 256, overflow_policy="TRUNCATE"
    )


def test_bounded_spill_policy_correctness():
    """BOUNDED_SPILL must clear the overflow flag it consumes.

    The overflow flag was initialized only for REREAD while BOUNDED_SPILL both
    increments and reads it; stale shared memory could select the
    non-overflow refinement with more candidates than the bounded buffer.
    Exercises the kernel-level wrapper (the public API pins REREAD) with a
    spill capacity small enough that rows overflow into the reread fallback,
    checked against a torch reference. Note: the unfixed behavior depends on
    residual SMEM contents, so this is functional coverage of the fixed path
    rather than a deterministic reproduction of the stale read.

    Regression for PR #4621 review (overflow-flag initialization).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    from flashinfer.topk_varlen.kernels.filtered_topk_decode import (
        cute_dsl_radix_filter_topk_wrapper,
    )

    torch.manual_seed(5)
    B, N, k = 8, 65536, 2048
    logits = torch.randn(B, N, dtype=torch.float32, device=device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    idx, _ = cute_dsl_radix_filter_topk_wrapper(
        logits,
        seq_lens,
        k,
        1,
        return_val=False,
        overflow_policy="BOUNDED_SPILL",
        spill_capacity=k + 256,  # small: forces overflow -> reread fallback
    )
    idx = idx.view(B, k)
    for b in range(B):
        sel = idx[b][idx[b] >= 0]
        assert sel.numel() == k and sel.unique().numel() == k
        kth = torch.topk(logits[b], k).values.min()
        assert bool((logits[b][sel.long()] >= kth - 1e-4).all()), f"row {b}"


@pytest.mark.parametrize("backend", ["radix_filter", "radix", "radix_cutlass"])
def test_next_n_group_abi_validated(backend):
    """next_n < 1 and row/group mismatches must fail fast at the API.

    Every backend maps row r to sequence r // next_n, so next_n == 0 divides
    by zero and logits.shape[0] != seq_lens.numel() * next_n silently reads
    seq_lens out of bounds on device (or applies the wrong grouping) -- easy
    to hit for an adapter that supplies expanded per-row lengths. Message-
    matched so the pre-fix behavior (a deeper, different error or silent
    corruption) cannot satisfy the test.

    Regression for PR #4621 review (grouped next_n ABI validation).
    """
    device = torch.device("cuda")
    if backend == "radix_filter":
        _skip_unless_radix_filter(device)
    else:
        # @backend_requirement rejects an unregistered backend/CC combination
        # before the API body's next_n validation can run.
        cc = get_compute_capability(device)
        if not flashinfer.top_k_varlen.is_backend_supported(
            backend, cc[0] * 10 + cc[1]
        ):
            pytest.skip(f"{backend} not supported on SM{cc[0]}{cc[1]}")

    logits = torch.randn(5, 4096, dtype=torch.float32, device=device)
    seq_lens = torch.full((2,), 4096, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match=r"next_n must be >= 1"):
        flashinfer.top_k_varlen(logits, seq_lens, 512, next_n=0, backend=backend)

    # 5 rows cannot be 2 groups of next_n=2.
    with pytest.raises(ValueError, match=r"row // next_n"):
        flashinfer.top_k_varlen(logits, seq_lens, 512, next_n=2, backend=backend)


def test_out_buffers_written_in_place():
    """Caller-supplied out_indices/out_values must be kernel destinations.

    Previously the wrapper always allocated its own outputs and the public
    path copied into the caller's buffers -- an extra num_rows x top_k
    allocation and D2D copy per call, and unstable destinations for CUDA
    graphs. The profiler assertion has teeth: the old path launches
    aten::copy_, the threaded path launches none.

    Regression for PR #4621 review (out-buffer threading).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(13)
    B, N, k = 8, 16384, 1024
    logits = torch.randn(B, N, dtype=torch.float32, device=device)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)
    out_i = torch.empty(B * k, dtype=torch.int32, device=device)
    out_v = torch.empty(B * k, dtype=torch.float32, device=device)

    # warm-up (compile outside the profiled region)
    flashinfer.top_k_varlen(
        logits,
        seq_lens,
        k,
        return_values=True,
        out_indices=out_i,
        out_values=out_v,
        backend="radix_filter",
    )
    # CPU activity alone records aten::copy_ at dispatch level; CUDA activity
    # would pull in CUPTI, which is not stable on every pre-release stack.
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        idx, vals = flashinfer.top_k_varlen(
            logits,
            seq_lens,
            k,
            return_values=True,
            out_indices=out_i,
            out_values=out_v,
            backend="radix_filter",
        )
        torch.cuda.synchronize()

    assert idx.data_ptr() == out_i.data_ptr()
    assert vals is not None and vals.data_ptr() == out_v.data_ptr()
    copies = [e.key for e in prof.key_averages() if "aten::copy_" in e.key]
    assert not copies, f"output copy still present: {copies}"

    # and the results in the caller's buffers are correct
    oi = out_i.view(B, k)
    for b in range(B):
        sel = oi[b][oi[b] >= 0]
        kth = torch.topk(logits[b], k).values.min()
        assert bool((logits[b][sel.long()] >= kth - 1e-4).all())


def test_checker_rejects_pre_idx_and_oob_top_k():
    """Explicit radix_filter calls must fail backend validation, not deeper.

    The checker docstring advertises pre_idx as a hard exclusion and the
    vendored kernel only supports top_k in [1, 16384], but neither was
    enforced: a non-None pre_idx was silently ignored (the kernel ran without
    the hint) and an oversized top_k surfaced as the kernel constructor's
    ValueError instead of a backend-validation error. Message-matched to the
    @backend_requirement rejection so the pre-fix behaviors cannot pass.

    Regression for PR #4621 review round 2 (checker constraints).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    torch.manual_seed(21)
    logits = torch.randn(8, 32768, dtype=torch.float32, device=device)
    seq_lens = torch.full((8,), 32768, dtype=torch.int32, device=device)

    pre = torch.zeros(8, 4096, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match=r"Problem size is not supported"):
        flashinfer.top_k_varlen(
            logits, seq_lens, 1024, pre_idx=pre, backend="radix_filter"
        )
    with pytest.raises(ValueError, match=r"Problem size is not supported"):
        flashinfer.top_k_varlen(logits, seq_lens, 16385, backend="radix_filter")


def test_tma_forced_on_misaligned_stride_fails_fast():
    """Forced TMA with a 16-byte-misaligned leading stride must fail fast.

    With the symbolic leading stride, num_cols divisibility no longer implies
    the row byte-stride alignment cuTensorMapEncodeTiled requires; a padded
    view (stride(0) = N + 1) previously reached the descriptor and failed with
    an opaque error after compiling. Now the wrapper raises an actionable
    ValueError before compilation. Runs on any radix_filter arch because
    enable_tma_load=True bypasses the tuned default.

    Regression for PR #4621 review round 2 (TMA stride gate).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    from flashinfer.topk_varlen.kernels.filtered_topk_decode import (
        _get_num_sms,
        cute_dsl_radix_filter_topk_wrapper,
    )

    sms = _get_num_sms(device)
    B, N, k = sms + 2, 4096, 512  # rows > SMs => large-occupancy TMA path
    base = torch.randn(B, N + 1, dtype=torch.float32, device=device)
    view = base[:, :N]
    assert view.stride(0) % 4 != 0  # fp32 _tma_div == 4
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match=r"16-byte"):
        cute_dsl_radix_filter_topk_wrapper(
            view,
            seq_lens,
            k,
            1,
            return_val=False,
            cluster_size=1,
            enable_tma_load=True,
        )


def test_tma_auto_falls_back_on_misaligned_stride():
    """The tuned auto-TMA default must fall back to LDG on misaligned strides.

    On the arch where the tuned default fires (fp32, Rubin, large N,
    rows > SMs), a padded view with stride(0) % 4 != 0 previously auto-enabled
    TMA and failed inside cuTensorMapEncodeTiled; the stride-aware gate keeps
    the call on the LDG path, which must produce correct top-k.

    Regression for PR #4621 review round 2 (TMA stride gate, auto path).
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    import cutlass

    from flashinfer.topk_varlen.kernels.filtered_topk_util import (
        get_topk_architecture_config,
        tma_tuned_default,
    )

    architecture, _ = get_topk_architecture_config()
    if not tma_tuned_default(cutlass.Float32, architecture, 131072):
        pytest.skip("async-TMA tuned default not active on this arch")

    torch.manual_seed(23)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    B, N, k = sms + 2, 131072, 2048
    base = torch.randn(B, N + 1, dtype=torch.float32, device=device)
    view = base[:, :N]
    assert view.stride(0) % 4 != 0
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=device)

    idx, _ = flashinfer.top_k_varlen(view, seq_lens, k, backend="radix_filter")
    idx = idx.view(B, k)
    for b in (0, B // 2, B - 1):
        sel = idx[b][idx[b] >= 0]
        assert sel.numel() == k and sel.unique().numel() == k
        kth = torch.topk(view[b], k).values.min()
        assert bool((view[b][sel.long()] >= kth - 1e-4).all()), f"row {b}"


def test_out_buffers_foreign_device_rejected():
    """Caller out-buffers on a different device than the input must be rejected.

    The kernel launches on input_values.device; a foreign-device buffer passes
    dtype/shape validation but hands the kernel a pointer it cannot legally
    write. No pre-fix teeth run here on purpose: executing the unfixed path
    performs a cross-device write that can poison the CUDA context, so this
    is functional coverage of the rejection only.

    Regression for PR #4621 review round 2 (out-buffer device validation).
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 visible CUDA devices")

    in_dev = None
    for i in range(torch.cuda.device_count()):
        d = torch.device("cuda", i)
        cc = get_compute_capability(d)
        if cc[0] * 10 + cc[1] in (100, 103, 107):
            in_dev = d
            break
    if in_dev is None:
        pytest.skip("no radix_filter-capable device visible")
    _skip_unless_radix_filter(in_dev)
    other = torch.device("cuda", (in_dev.index + 1) % torch.cuda.device_count())

    B, N, k = 4, 4096, 512
    logits = torch.randn(B, N, dtype=torch.float32, device=in_dev)
    seq_lens = torch.full((B,), N, dtype=torch.int32, device=in_dev)
    foreign = torch.empty(B * k, dtype=torch.int32, device=other)

    with (
        torch.cuda.device(in_dev),
        pytest.raises(ValueError, match=r"input device"),
    ):
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            k,
            backend="radix_filter",
            out_indices=foreign,
        )


@pytest.mark.xfail(
    strict=False,
    reason="upstream DKG multi-pass merge scans the padded candidate width; "
    "padded (-1, -inf) entries tie with genuinely valid -inf elements, so "
    "slots that should hold valid indices can come back as -1. strict=False "
    "because the leak depends on tie-collection order, which is not "
    "deterministic across runs/architectures. Preserved as xfail until the "
    "upstream fix lands (see the PR #4621 review thread).",
)
def test_multipass_merge_padding_vs_valid_neg_inf():
    """xfail: multi-pass merge can emit -1 despite >= k valid elements.

    Two chunks per row (N = 2 * 16384) with a nearly-empty second chunk
    (1 valid element < k), so stage one pads that chunk's candidate list
    with k - 1 (-1, -inf) entries; only 8 values are finite and the rest of
    the row is genuinely valid -inf. The row length (16385) exceeds k, so
    every output slot must be a valid index -- but padded and valid -inf
    candidates share a radix key, and the merge's fixed-width scan selects
    pads (measured ~1087/2048 slots on SM100 with this shape; a fully-valid
    second chunk produces no padding, which is why that case passes). Three
    seeds are tried because tie-collection order varies run to run. NOT
    reachable from public top_k_varlen.
    """
    device = torch.device("cuda")
    _skip_unless_radix_filter(device)

    from flashinfer.topk_varlen.kernels.filtered_topk_decode import (
        cute_dsl_radix_filter_topk_multi_cta_wrapper,
    )

    B, N, k = 2, 32768, 1024
    total_neg = 0
    for trial in range(3):
        vals = torch.full((B, N), float("-inf"), dtype=torch.float32, device=device)
        vals[:, :8] = torch.arange(
            8 + trial, trial, -1, dtype=torch.float32, device=device
        )
        # 16384 + 1: the second chunk holds one valid element < k, forcing
        # k - 1 stage-one pads into the merge width. Total valid still >= k.
        seq_lens = torch.full((B,), 16384 + 1, dtype=torch.int32, device=device)
        idx, _ = cute_dsl_radix_filter_topk_multi_cta_wrapper(
            vals, seq_lens, k, 1, return_val=False
        )
        total_neg += int((idx.view(B, k) < 0).sum())
    assert total_neg == 0, (
        "merge emitted padded -1 for rows with >= k valid elements: "
        f"{total_neg} invalid slots across 3 trials"
    )
