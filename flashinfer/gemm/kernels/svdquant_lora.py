# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host glue for the fused NVFP4 SVDQuant kernel (Sm100BlockScaledLoRADenseGemmKernel).

Framework-agnostic (no framework imports): the caller supplies the already-swizzled weight
scale-factor blob and the externally-quantized NVFP4 activation (ext_xq/ext_sf), both consumed
by the CuTe-DSL kernel as a raw data_ptr() reinterpreted via the block-scaled scale-factor
layout. The BF16 down-projection D = X @ L2_merged^T is computed here with torch.matmul;
the framework merges any activation smoothing scale into L2 once while loading weights.

Public API:
    state = prepare_svdquant_state(M, Rq, w_sf_swizzled, L1, L2_merged,
                                   gscale_x, gscale_w, r)
    y = mm_fp4_svdquant(x, state, ext_xq=..., ext_sf=...)   # y is [M, O] bf16
"""

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from .dense_blockscaled_gemm_lora_sm100 import Sm100BlockScaledLoRADenseGemmKernel

_V2_KERNEL_CACHE = {}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _build_compiled(N, K, M, r, out_dtype=torch.bfloat16, tactic=None):
    """cute.compile the kernel's fused residual + LoRA-up wrapper for a fixed shape.

    Returns (compiled_callable, sf_m, sf_n, sf_k, lora_k). Cached per
    (N,K,M,r,out_dtype,tactic).

    tactic: optional (mma_tiler_mn, cluster_shape_mn, use_prefetch) selected by the
    private prepare-time autotuner.
    """
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    # The compiled kernel is symbolic in M (sf_m/sf_n/sf_k + grid are runtime args), so a fixed
    # A selected tactic compiles once and is reused across all M -- critical to avoid recompiling per
    # batch during e2e warmup. Only the None path (analytical heuristic, capture fallback) is
    # M-dependent, so it keeps M in the key.
    # The compiled persistent grid contains a device-specific residency limit.
    # Keep separate entries when multiple CUDA devices are present.
    device_index = torch.cuda.current_device()
    key = (
        device_index,
        N,
        K,
        r,
        out_dtype,
        tactic,
        M if tactic is None else None,
    )
    if key in _V2_KERNEL_CACHE:
        # The compiled callable is symbolic in M, but the activation scale-factor
        # extent is a runtime property of the current M.  Do not return the sf_m
        # from whichever batch populated the compile cache first.
        compiled, cached_sf_n, cached_sf_k, cached_lora_k = _V2_KERNEL_CACHE[key]
        return compiled, _ceil_div(M, 128), cached_sf_n, cached_sf_k, cached_lora_k

    sf_dtype = cutlass.Float8E4M3FN
    c_dtype = cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
    lora_k = _ceil_div(r, 16) * 16
    sf_m = _ceil_div(M, 128)
    sf_n = _ceil_div(N, 128)
    sf_k = _ceil_div(_ceil_div(K, 16), 4)

    # Tile/cluster: autotuned tactic > analytical heuristic, falling back to 128x128 (1,1).
    # The fused-LoRA path now supports 2-CTA (256-wide) tiles + multicast D/L1, so any non-swap
    # sm100 tactic is allowed (2-CTA 256-tiles are 1.3-1.6x faster at large M = high batch). swap_ab
    # is still unsupported (D is M-indexed / L1 N-indexed; swap would transpose those roles).
    mma_tiler_mn, cluster_shape_mn, use_prefetch = (128, 128), (1, 1), False
    if tactic is not None:
        mma_tiler_mn, cluster_shape_mn, use_prefetch = tactic
    else:
        try:
            from flashinfer.gemm.kernels.utils import (
                _select_sm100_mm_fp4_cute_dsl_tactic,
            )

            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            tac = _select_sm100_mm_fp4_cute_dsl_tactic(M, N, K, sm_count)
            t_mma, t_cluster, t_swap, t_prefetch, t_kernel, _ = tac
            if (not t_swap) and t_kernel == "sm100":
                mma_tiler_mn, cluster_shape_mn, use_prefetch = (
                    t_mma,
                    t_cluster,
                    t_prefetch,
                )
        except Exception:
            pass

    gemm = Sm100BlockScaledLoRADenseGemmKernel(
        16,
        mma_tiler_mn,
        cluster_shape_mn,
        use_prefetch,
        True,
        lora_rank=r,
    )
    sym_m = cute.sym_int()
    sym_k = cute.sym_int()
    sym_n = cute.sym_int()
    a_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, sym_k), stride_order=(1, 0), assumed_align=32
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_n, sym_k), stride_order=(1, 0), assumed_align=32
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        c_dtype, (sym_m, sym_n), stride_order=(1, 0), assumed_align=16
    )
    a_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    # The persistent scheduler consumes a cluster count, not a CTA count.  Passing
    # the 1-CTA residency for a 2/4-CTA tactic overstates how many clusters can be
    # resident and produces the wrong persistent grid.  Match mm_fp4's builder.
    mac = get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    opts = "--opt-level 2 --enable-tvm-ffi"

    d_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, cute.sym_int()), stride_order=(1, 0), assumed_align=16
    )
    l1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_n, cute.sym_int()), stride_order=(1, 0), assumed_align=16
    )
    compiled = cute.compile(
        gemm.wrapper_lora,
        a_fake,
        b_fake,
        c_fake,
        d_fake,
        l1_fake,
        sf_m,
        sf_n,
        sf_k,
        1,
        a_sf_ptr,
        b_sf_ptr,
        alpha_fake,
        mac,
        stream_fake,
        options=opts,
    )
    _V2_KERNEL_CACHE[key] = (compiled, sf_n, sf_k, lora_k)
    return compiled, sf_m, sf_n, sf_k, lora_k


# Candidate tactics for the prepare-time autotuner. Non-swap sm100 only (the LoRA path supports
# 1-CTA and 2-CTA 256-wide tiles, not swap_ab). 2-CTA 256-tiles are 1.3-1.6x faster at large M;
# the analytical heuristic under-selects them, so we time-select at prepare (warmup) like the
# production TunableRunner. (tile, cluster, use_prefetch).
_AUTOTUNE_CANDIDATES = [
    # 1-CTA (win at small M, where trtllm is overhead-bound and a single CTA suffices)
    ((128, 128), (1, 1), False),
    ((128, 256), (1, 1), False),
    ((128, 128), (1, 2), False),
    ((128, 192), (1, 2), False),
    # 2-CTA 256-wide (win at large M = high batch; 1.3-1.6x faster residual). Unsupported combos
    # (e.g. tile_n the shape can't tile) are skipped by the try/except in _autotune_tactic.
    ((256, 128), (2, 1), False),
    ((256, 256), (2, 1), False),
    ((256, 192), (2, 1), False),
    ((256, 256), (2, 2), False),
    ((256, 192), (2, 2), False),
    ((256, 256), (4, 1), False),
]
_TACTIC_CHOICE_CACHE = {}


def _time_call(call, iters=10, warmup=3):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        call()
        e1[i].record()
    torch.cuda.synchronize()
    return sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))[iters // 2] * 1000.0


def _autotune_tactic(N, K, M, r, device=None):
    """Time-select the fastest tactic for (N,K,timing-M,r). Returns a (tile,cluster,prefetch) tuple,
    or None if timing is unavailable (CUDA-graph capture) -> caller falls back to the heuristic.
    Random correctly-sized operands (timing is shape-, not value-, dependent)."""
    try:
        if torch.cuda.is_current_stream_capturing():
            return None
    except Exception:
        return None
    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    # Preserve exact-M choices through the allocation cap. Power-of-two bucketing aliases materially
    # different shapes (for example M=6889 and M=8192), so a startup tune can otherwise select the
    # wrong tactic for a later batch. Shapes above the cap intentionally share its measured tactic.
    timing_m = min(M, 32768)
    key = (dev.index, N, K, timing_m, r)
    if key in _TACTIC_CHOICE_CACHE:
        return _TACTIC_CHOICE_CACHE[key]

    Kp = K // 2
    lora_k = _ceil_div(r, 16) * 16
    # Cap the timing-M to avoid huge autotune allocations (out[M,N] is ~3.6GB at M=147k/b16 ->
    # OOM). The best tactic is stable in the large-M (compute-bound) regime, and the
    # compiled kernel is symbolic in M, so timing at a capped M picks the same tactic.
    M = timing_m
    sf_m = _ceil_div(M, 128)
    sf_n = _ceil_div(N, 128)
    sf_k = _ceil_div(_ceil_div(K, 16), 4)
    a = torch.randint(0, 256, (M, Kp), dtype=torch.uint8, device=dev)
    wq = torch.randint(0, 256, (N, Kp), dtype=torch.uint8, device=dev)
    a_sf = torch.randint(
        0, 256, (sf_m * sf_k * 512,), dtype=torch.uint8, device=dev
    ).view(torch.float8_e4m3fn)
    w_sf = torch.randint(
        0, 256, (sf_n * sf_k * 512,), dtype=torch.uint8, device=dev
    ).view(torch.float8_e4m3fn)
    alpha = torch.tensor([0.5], dtype=torch.float32, device=dev)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    d = torch.randn(M, lora_k, dtype=torch.bfloat16, device=dev)
    l1 = torch.randn(N, lora_k, dtype=torch.bfloat16, device=dev)

    best_t, best_us = None, None
    for tac in _AUTOTUNE_CANDIDATES:
        try:
            compiled, sfm, sfn, sfk, _ = _build_compiled(N, K, M, r, tactic=tac)
            call = lambda c=compiled: c(
                a,
                wq,
                out,
                d,
                l1,
                sfm,
                sfn,
                sfk,
                a_sf.data_ptr(),
                w_sf.data_ptr(),
                alpha,
            )
            us = _time_call(call)
            if best_us is None or us < best_us:
                best_us, best_t = us, tac
        except Exception:
            # A candidate tile/cluster the kernel can't codegen for this shape (e.g. a tile_n the N
            # can't tile): skip it. best_t only ever holds a tactic that built+ran successfully.
            continue
    # Defensive default: if every candidate failed to build (should not happen), fall back to the
    # universally-supported 128x128 1-CTA tile rather than None (which would route prepare() to the
    # analytical heuristic, whose mm_fp4-oriented pick is not guaranteed LoRA-codegen-safe).
    if best_t is None:
        best_t = ((128, 128), (1, 1), False)
    _TACTIC_CHOICE_CACHE[key] = best_t
    return best_t


@torch.compiler.disable
def prepare_svdquant_state(
    M,
    Rq,
    w_sf_swizzled,
    L1,
    L2_merged,
    gscale_x,
    gscale_w,
    r,
):
    """Build a reusable dispatch state for one (M, I, O, r) shape (injection path).

      Rq: prepacked NVFP4 weight [O, I//2] uint8 (E2M1, K-major).
      w_sf_swizzled: weight block-scales already in the NVFP4 MMA swizzle (uint8/e4m3 blob, the layout
                     trtllm fp4_quantize / swizzle_sf produce -- passed to the kernel as raw data_ptr).
      L1: up-proj [O, r] bf16.
      L2_merged: down-proj [r, I] bf16, already merged with any pre_quant_scale by the caller.
      gscale_x, gscale_w: static global scales; the kernel applies alpha = gscale_x * gscale_w.
    The activation is externally quantized; this kernel has no internal quantizer.
    """
    O = int(Rq.shape[0])
    I = int(Rq.shape[1]) * 2
    lora_k = _ceil_div(r, 16) * 16
    if not Rq.is_cuda:
        raise ValueError("Rq must be a CUDA tensor")
    dev = Rq.device

    # Time-select the tile/cluster once per (shape, capped exact M) during warmup. Under CUDA-graph
    # capture the private autotuner returns None and _build_compiled uses its analytical heuristic.
    with torch.cuda.device(dev):
        tactic = _autotune_tactic(O, I, M, r, device=dev)
        compiled, sf_m, sf_n, sf_k, _ = _build_compiled(O, I, M, r, tactic=tactic)

    wq = Rq.to(device=dev).view(torch.uint8).contiguous()
    w_sf = w_sf_swizzled.to(device=dev).contiguous().view(torch.float8_e4m3fn)

    # Fold 1/alpha into L1 so the LoRA-up D@L1^T accumulates into the MAIN block-scaled accumulator:
    # the (unchanged) epilogue computes out = alpha*acc = alpha*(residual + D@(L1/alpha)^T)
    # = alpha*residual + D@L1^T. This avoids a 2nd TMEM accumulator => keeps the base num_acc_stage
    # MMA/epilogue overlap (the perf). alpha~3e-4 => L1/alpha~O(1e2), fine in bf16 + the f32 acc.
    alpha_f = float(gscale_x) * float(gscale_w)
    L1d = (L1.to(device=dev, dtype=torch.float32) / alpha_f).to(torch.bfloat16)
    if L1d.shape[0] != O:
        L1d = F.pad(L1d, (0, 0, 0, O - L1d.shape[0]))
    if L1d.shape[1] != lora_k:
        L1d = F.pad(L1d, (0, lora_k - L1d.shape[1]))  # [O, lora_k]
    L1d = L1d.contiguous()

    L2d = L2_merged.to(device=dev, dtype=torch.bfloat16)
    if L2d.shape[1] != I:
        L2d = F.pad(L2d, (0, I - L2d.shape[1]))  # [r, I]
    # Any activation smoothing scale is merged into L2 once by the framework's
    # weight-preparation path, not here per shape-specific dispatch state.
    L2_t = L2d.t().contiguous()  # [I, r]

    # Preallocated LoRA-down output (graph-safe: no per-call alloc). D_buf is padded to lora_k with
    # zero columns; D_active is the contiguous [M, r] write target.
    D_buf = torch.zeros((M, lora_k), dtype=torch.bfloat16, device=dev)
    D_active = (
        D_buf if lora_k == r else torch.empty((M, r), dtype=torch.bfloat16, device=dev)
    )

    return {
        "M": M,
        "I": I,
        "O": O,
        "r": r,
        "lora_k": lora_k,
        "compiled": compiled,
        "sf_m": sf_m,
        "sf_n": sf_n,
        "sf_k": sf_k,
        "wq": wq,
        "w_sf": w_sf,
        "L1": L1d,
        "L2_t": L2_t,
        "D_buf": D_buf,
        "D_active": D_active,
        "alpha": torch.tensor([alpha_f], dtype=torch.float32, device=dev),
        # `out` is allocated per call (not cached): under CUDA-graph capture it comes from the
        # graph's private pool, and caching a [Mpad, O] buffer per shape is prohibitive at large M.
    }


@torch.compiler.disable
def mm_fp4_svdquant(x, state, ext_xq, ext_sf):
    """Run the fused NVFP4 SVDQuant linear. x: [M, I] (M padded to 128). Returns [M, O] bf16.

    Wrapped in @torch.compiler.disable because Dynamo cannot trace the CuTe-DSL JIT objects; this
    is kept as an opaque eager region (tracing into it otherwise causes graph breaks).

    ext_xq: NVFP4 activation [M, I//2] uint8 (E2M1); ext_sf: its swizzled block-scale blob.
    The BF16 down-proj uses raw x because pre_quant_scale is already merged into L2 by the caller.
    """
    c = state
    out = torch.empty((x.shape[0], c["O"]), dtype=torch.bfloat16, device=x.device)
    xb = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    torch.mm(xb, c["L2_t"], out=c["D_active"])  # [M, r]
    if c["D_active"] is not c["D_buf"]:
        c["D_buf"][:, : c["r"]].copy_(c["D_active"])
    c["compiled"](
        ext_xq.view(torch.uint8),
        c["wq"],
        out,
        c["D_buf"],
        c["L1"],
        c["sf_m"],
        c["sf_n"],
        c["sf_k"],
        ext_sf.data_ptr(),
        c["w_sf"].data_ptr(),
        c["alpha"],
    )
    return out
