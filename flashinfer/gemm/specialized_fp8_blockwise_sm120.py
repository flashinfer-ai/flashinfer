"""Specialized SM120 backend for fp8 1x128/128x128 blockwise-scale GEMM.

An agent-generated, shape-specialized kernel set for the dense decode GEMMs
of fp8-blockwise-quantized serving workloads (per-token 1x128 fp32
activation scales, 128x128 fp32 weight block scales, NT layout, bf16 out).
Exposed through ``gemm_fp8_nt_groupwise`` as the ``"specialized"`` backend
(PR #5667 pattern):

- ``backend="auto"`` routes to ``"specialized"`` only on the allowlisted
  shapes/contracts below; everything else resolves as before.
- An explicitly selected backend is NEVER silently swapped: explicit
  ``backend="specialized"`` that cannot serve a call falls back to the
  composable torch path (dequant + matmul), not to another fused backend.
- Kill switch: ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` forces the
  composable/stock path.
- ``fallback_reasons``/``dispatch_stats`` record why calls were (not)
  served.

Allowlist (every admitted row measured on RTX PRO 6000, two independent
runs; shipped in ``specialized_fp8_blockwise_sm120_allowlist.json``):
(N, K) in {(64, 2048), (1024, 2048), (2048, 512), (2048, 4096)} with
1 <= M <= 32, plus (9216, 2048) and (12288, 2048) with 1 <= M <= 16.
The N=64 class is 128-unaligned: the stock SM120 cutlass groupwise kernel
accepts it and returns silently wrong output, so there the backend is also
a correctness fix.

CUDA-graph rules: the host entry takes the ambient stream per call
(tvm-ffi); under stream capture only (M, N, K) shapes already launched on
that device (all of them after :func:`warmup`) dispatch, so no lazy
module/kernel load happens under capture. The key is the exact shape, not
the (N, K) class, because the dispatcher picks a different kernel per M
range. Autotune mode falls back.
"""

import functools
import json
import os
from collections import Counter, deque
from pathlib import Path
from typing import Set, Tuple

import torch

from ..jit.cpp_ext import is_cuda_version_at_least
from ..jit.gemm.specialized_fp8_blockwise_sm120 import (
    gen_specialized_gemm_fp8_blockwise_sm120_module,
)

_HERE = Path(__file__).parent


@functools.cache
def _allowlist() -> dict:
    """(N, K) -> max admitted M, loaded from the shipped allowlist JSON."""
    with open(_HERE / "specialized_fp8_blockwise_sm120_allowlist.json") as f:
        raw = json.load(f)
    return {
        tuple(int(x) for x in k.split("x")): int(v)
        for k, v in raw["auto_route"]["nk_to_m_max"].items()
    }


# telemetry: why calls fell back, and per-backend dispatch counts
fallback_reasons: deque = deque(maxlen=256)
dispatch_stats: Counter = Counter()

# (device_index, M, N, K) calls already launched on that device (CUDA-graph
# readiness). Keyed by the full shape, NOT by (N, K) class: combo_entry.cu
# picks a different kernel per M range inside a class (with exact-M cluster
# forms, e.g. at N=64), so one launched M says nothing about the kernel a
# different M would lazily load under capture. A given (M, N, K) always takes
# the same dispatch route, so this key identifies the kernel actually loaded;
# warmup() launches every admitted M and therefore warms every route.
_warmed: Set[Tuple[int, int, int, int]] = set()


@functools.cache
def get_specialized_gemm_fp8_blockwise_sm120_module():
    # The JIT spec lives with its siblings in flashinfer/jit/gemm/ and is
    # registered in flashinfer/aot.py (exact sm_120a target), so jit-cache
    # installs carry a prebuilt copy instead of compiling on first call.
    return gen_specialized_gemm_fp8_blockwise_sm120_module().build_and_load()


def _is_sm120_exactly(device: torch.device) -> bool:
    major, minor = torch.cuda.get_device_capability(device)
    return major == 12 and minor == 0


def is_specialized_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode,
    scale_granularity_mnk,
    out: torch.Tensor = None,
    out_dtype: torch.dtype = None,
) -> bool:
    """Contract guard. Guard order per the specialized-backend spec:
    GPU-free checks (kill switch, geometry, dtype, allowlist, contiguity)
    first, then device guards. Returns False with the reason recorded."""

    def _no(reason):
        fallback_reasons.append(("specialized", reason))
        return False

    # --- GPU-free checks
    if os.environ.get("FLASHINFER_SPECIALIZED_KERNEL_DISABLE") == "1":
        return _no("kill_switch")
    if a.dim() != 2 or b.dim() != 2:
        return _no("geometry:not_2d")
    # The kernel launches on A's device (combo_entry makes it current) and
    # reads every operand through raw pointers: an operand on another device
    # would be an illegal cross-device access, not a torch error.
    devices = {a.device, b.device, a_scale.device, b_scale.device}
    if out is not None:
        devices.add(out.device)
    if len(devices) != 1:
        return _no("device:operand_mismatch")
    M, K = a.shape
    N = b.shape[0]
    m_max = _allowlist().get((N, K))
    if m_max is None or not (1 <= M <= m_max):
        return _no(f"allowlist:m{M}n{N}k{K}")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return _no("dtype:operands")
    if scale_major_mode not in (None, "K"):
        return _no("layout:scale_major_mode")
    if tuple(scale_granularity_mnk) != (1, 128, 128):
        return _no("layout:scale_granularity")
    KB = K // 128
    NB = (N + 127) // 128
    if a_scale.shape != (M, KB) or b_scale.shape != (NB, KB):
        return _no("layout:scale_shape")
    if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        return _no("dtype:scales")
    if out is None:
        if out_dtype not in (None, torch.bfloat16):
            return _no("layout:out_dtype")
    elif out.shape != (M, N) or out.dtype != torch.bfloat16 or not out.is_contiguous():
        return _no("layout:out")
    if not (
        a.is_contiguous()
        and b.is_contiguous()
        and a_scale.is_contiguous()
        and b_scale.is_contiguous()
    ):
        return _no("layout:contiguity")
    # The kernels use 256-bit (v8.b32) operand loads and 32-bit packed bf16
    # output stores: operand base pointers must be 32B aligned and the output
    # base 4B aligned (offset storage views can violate this; fresh torch
    # allocations are always 256B aligned). Scales are read as f32 and are
    # naturally aligned for any fp32 tensor.
    if a.data_ptr() % 32 != 0 or b.data_ptr() % 32 != 0:
        return _no("layout:operand_alignment")
    if out is not None and out.data_ptr() % 4 != 0:
        return _no("layout:out_alignment")
    # 256-bit global loads need PTX ISA 8.8, first shipped with CUDA 12.9;
    # older toolkits would fail JIT compilation instead of running.
    try:
        if not is_cuda_version_at_least("12.9"):
            return _no("toolkit:cuda_below_12.9")
    except Exception:  # noqa: BLE001 - no detectable toolkit: cannot JIT
        return _no("toolkit:version_unknown")
    # --- device guards
    if not _is_sm120_exactly(a.device):  # sm_121 unmeasured -> stock
        return _no("device:not_sm120")
    try:
        from ..autotuner import AutoTuner

        if AutoTuner.get().is_tuning_mode:
            return _no("autotune_mode")
    except Exception:  # noqa: BLE001 - autotuner surface moved: be safe
        return _no("autotune_probe_failed")
    if (
        torch.cuda.is_current_stream_capturing()
        and (a.device.index, M, N, K) not in _warmed
    ):
        return _no("capture:unwarmed")
    return True


def run_specialized_gemm_fp8_blockwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """DPS entry: Y = dequant(A) @ dequant(B).T, written into ``out``."""
    module = get_specialized_gemm_fp8_blockwise_sm120_module()
    module.run(a, b, a_scale, b_scale, out)
    _warmed.add((a.device.index, a.shape[0], b.shape[0], a.shape[1]))
    dispatch_stats["specialized"] += 1
    return out


def run_composable_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    scale_major_mode=None,
    scale_granularity_mnk=(1, 128, 128),
) -> torch.Tensor:
    """Composable torch path (dequant + matmul) for an EXPLICIT
    ``backend="specialized"`` call the kernel cannot serve. Correct for any
    shape (including 128-unaligned N), either scale layout, and any scale
    granularity; not performance-tuned."""
    M, K = a.shape
    N = b.shape[0]
    gm, gn, gk = (int(x) for x in scale_granularity_mnk)
    a_s = a_scale.to(torch.float32)
    b_s = b_scale.to(torch.float32)
    if scale_major_mode == "MN":  # (k_blocks, *) -> (*, k_blocks)
        a_s = a_s.transpose(0, 1)
        b_s = b_s.transpose(0, 1)
    a_full = a_s.repeat_interleave(gm, dim=0)[:M].repeat_interleave(gk, dim=1)[:, :K]
    b_full = b_s.repeat_interleave(gn, dim=0)[:N].repeat_interleave(gk, dim=1)[:, :K]
    a32 = a.to(torch.float32) * a_full
    b32 = b.to(torch.float32) * b_full
    out.copy_((a32 @ b32.T).to(out.dtype))
    dispatch_stats["specialized_composable_fallback"] += 1
    return out


def warmup(device: torch.device = None) -> None:
    """Build the module and launch every admitted M of every allowlisted
    (N, K) class so each dispatched kernel specialization loads before
    CUDA-graph capture (the per-class dispatcher picks different kernels
    for different M ranges; lazy kernel load happens here, not under
    capture). Warmup is per device."""
    dev = device or torch.device("cuda")
    for (N, K), m_max in sorted(_allowlist().items()):
        for M in range(1, m_max + 1):
            a = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device=dev)
            b = torch.zeros(N, K, dtype=torch.float8_e4m3fn, device=dev)
            a_s = torch.ones(M, K // 128, dtype=torch.float32, device=dev)
            b_s = torch.ones(
                (N + 127) // 128, K // 128, dtype=torch.float32, device=dev
            )
            y = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
            run_specialized_gemm_fp8_blockwise(a, b, a_s, b_s, y)
    torch.cuda.synchronize()


def check_specialized_requirement(
    a,
    b,
    a_scale,
    b_scale,
    scale_major_mode=None,
    mma_sm=1,
    scale_granularity_mnk=(1, 128, 128),
    out=None,
    out_dtype=None,
    backend="specialized",
    batch_invariant=False,
):
    """``backend_requirement`` checker signature adapter (out may be None
    at check time; the dispatch body re-validates with the real tensor)."""
    if batch_invariant:
        return False
    if backend == "specialized":
        # Explicitly selected: always accepted — the dispatch body serves it
        # with the kernel when the contract guard passes, else with the
        # composable torch fallback (never another fused backend). A module
        # build/load failure on an explicit request RAISES (an explicit
        # backend is never silently substituted).
        return True
    if not is_specialized_supported(
        a,
        b,
        a_scale,
        b_scale,
        scale_major_mode,
        scale_granularity_mnk,
        out,
        out_dtype,
    ):
        return False
    # Auto admission additionally requires the module to actually build and
    # load: a missing toolkit, a jit-cache install without this module, or a
    # compile failure must not raise out of a ``backend="auto"`` call. The
    # failure is recorded and auto resolves to the stock backend instead.
    try:
        get_specialized_gemm_fp8_blockwise_sm120_module()
    except Exception as e:  # noqa: BLE001 - any build/load failure falls back
        fallback_reasons.append(
            ("specialized", f"module_unavailable:{type(e).__name__}")
        )
        return False
    return True
