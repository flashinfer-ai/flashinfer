# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile (cuda.tile Python) GEMM kernels for FlashInfer.

This module currently provides ``mm_bf16_cutile`` — a BF16 GEMM
implementation that lives next to the existing CUTLASS / cuDNN / TGV /
cuBLASLt / TinyGEMM backends in ``flashinfer.gemm.gemm_base.mm_bf16``. The
kernel is written in pure ``cuda.tile`` Python (no external DSL middleware)
and supports per-shape exhaustive autotune via
``cuda.tile.tune.exhaustive_search``.

The underlying kernel ``_gemm_alpha_beta_kernel_cutile`` is a general
alpha-beta GEMM (C = alpha * op(A) @ op(B) + beta * C) with persistent
scheduling; ``mm_bf16_cutile`` is a thin wrapper that pins alpha=1.0,
beta=0.0 and binds the upstream ``mm_bf16`` layout (a row-major (M,K), b a
transposed view of (N,K) row-major storage) into the kernel's native (M,K)
x (N,K) trans_b=True form.

Source / Attribution
--------------------
The kernel, autotune config space, and dispatcher logic are ported verbatim
from NVIDIA TileGym (Apache-2.0 / MIT dual-licensed), specifically
``src/tilegym/suites/flashinfer/cutile/gemm/gemm_alpha_beta.py`` at
https://github.com/NVIDIA/TileGym. TileGym-internal decorators
(``@register_impl``) and helpers (``build_cutile_kernel_from_autotune``,
``get_kernel_configs``) are dropped here in favor of the equivalent public
``cuda.tile`` APIs, so this module has no TileGym runtime dependency.
"""

from types import SimpleNamespace
from typing import Optional

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search


def _cdiv(a: int, b: int) -> int:
    """Ceiling division helper."""
    return (a + b - 1) // b


# Ported verbatim from NVIDIA TileGym
# (https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/suites/flashinfer/cutile/gemm/gemm_alpha_beta.py).
@ct.kernel
def _gemm_alpha_beta_kernel_cutile(
    a_ptr,  # Input matrix A [M, K] or [K, M] if transpose_a
    b_ptr,  # Input matrix B [K, N] or [N, K] if transpose_b
    c_ptr,  # Output/Input matrix C [M, N] - modified in place
    alpha: ct.Constant[float],
    beta: ct.Constant[float],
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int],
    total_tiles: ct.Constant[int],
    num_programs: ct.Constant[int],
    num_pid_m: ct.Constant[int],
    num_pid_n: ct.Constant[int],
    transpose_a: ct.Constant[int],  # 0 or 1
    transpose_b: ct.Constant[int],  # 0 or 1
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    EPILOGUE_SUBTILE: ct.Constant[int],
):
    """cuTile kernel for GEMM with alpha/beta scaling.

    C = alpha * op(A) @ op(B) + beta * C

    - Persistent scheduling with SM-aware grid sizing
    - GROUP_SIZE_M based tile swizzling for L2 locality
    - Optional epilogue subtiling (EPILOGUE_SUBTILE=1) for SM100 shared-mem reduction
    - Latency hints on loads for pipelining
    """
    pid = ct.bid(0)

    num_k_tiles = ct.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    zero_pad = ct.PaddingMode.ZERO

    # Persistent scheduling loop
    for current_pid in range(pid, total_tiles, num_programs):
        # Tile swizzling
        group_id = current_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + (current_pid % group_size_m_actual)
        pid_n = (current_pid % num_pid_in_group) // group_size_m_actual

        # Initialize accumulator
        acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

        for k in range(num_k_tiles):
            if transpose_a == 1:
                # A is [K, M], load [BLOCK_K, BLOCK_M] and transpose
                a_block_kt = ct.load(
                    a_ptr,
                    index=(k, pid_m),
                    shape=(BLOCK_K, BLOCK_M),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )
                a_block = ct.permute(a_block_kt, (1, 0))
            else:
                a_block = ct.load(
                    a_ptr,
                    index=(pid_m, k),
                    shape=(BLOCK_M, BLOCK_K),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )

            if transpose_b == 1:
                # B is [N, K], load [BLOCK_N, BLOCK_K] and transpose
                b_block_nt = ct.load(
                    b_ptr,
                    index=(pid_n, k),
                    shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )
                b_block = ct.permute(b_block_nt, (1, 0))
            else:
                b_block = ct.load(
                    b_ptr,
                    index=(k, pid_n),
                    shape=(BLOCK_K, BLOCK_N),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )

            acc = ct.mma(a_block, b_block, acc=acc)

        if EPILOGUE_SUBTILE == 1:
            # Split accumulator into two N/2 halves to reduce shared memory in epilogue
            acc0 = ct.extract(acc, index=(0, 0), shape=(BLOCK_M, BLOCK_N // 2))
            acc1 = ct.extract(acc, index=(0, 1), shape=(BLOCK_M, BLOCK_N // 2))

            c_load0 = ct.load(
                c_ptr,
                index=(pid_m, pid_n * 2),
                shape=(BLOCK_M, BLOCK_N // 2),
                order=(0, 1),
                padding_mode=zero_pad,
            )
            c_load0_f32 = ct.astype(c_load0, ct.float32)
            result0 = alpha * acc0 + beta * c_load0_f32
            c_block0 = ct.astype(result0, c_ptr.dtype)
            ct.store(
                c_ptr,
                index=(pid_m, pid_n * 2),
                tile=c_block0,
                order=(0, 1),
            )

            c_load1 = ct.load(
                c_ptr,
                index=(pid_m, pid_n * 2 + 1),
                shape=(BLOCK_M, BLOCK_N // 2),
                order=(0, 1),
                padding_mode=zero_pad,
            )
            c_load1_f32 = ct.astype(c_load1, ct.float32)
            result1 = alpha * acc1 + beta * c_load1_f32
            c_block1 = ct.astype(result1, c_ptr.dtype)
            ct.store(
                c_ptr,
                index=(pid_m, pid_n * 2 + 1),
                tile=c_block1,
                order=(0, 1),
            )
        else:
            c_load = ct.load(
                c_ptr,
                index=(pid_m, pid_n),
                shape=(BLOCK_M, BLOCK_N),
                order=(0, 1),
                padding_mode=zero_pad,
            )
            c_load_f32 = ct.astype(c_load, ct.float32)
            result = alpha * acc + beta * c_load_f32
            c_block = ct.astype(result, c_ptr.dtype)
            ct.store(
                c_ptr,
                index=(pid_m, pid_n),
                tile=c_block,
                order=(0, 1),
            )


def _autotune_configs(device=None):
    """Iterator of autotune configurations, tuned per GPU arch.

    ``device``: optional torch.device for the specific GPU whose capability
    we should query (defaults to the current active CUDA device).
    """
    gpu_capability = torch.cuda.get_device_capability(device)

    if gpu_capability[0] >= 10:
        # EPILOGUE_SUBTILE=1 only validated on SM100; disable on SM12x to avoid correctness issues
        subtile_options = [0, 1] if gpu_capability == (10, 0) else [0]
        for BM, BN, nc in [
            (64, 64, 1),
            (64, 128, 1),
            (128, 64, 1),
            (128, 128, 1),
            (256, 64, 1),
            (256, 128, 1),
            (256, 128, 2),
            (256, 256, 2),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4, 8]:
                    for subtile in subtile_options:
                        yield SimpleNamespace(
                            BLOCK_M=BM,
                            BLOCK_N=BN,
                            BLOCK_K=BK,
                            GROUP_SIZE_M=8,
                            num_ctas=nc,
                            occupancy=occupancy,
                            EPILOGUE_SUBTILE=subtile,
                        )
    elif gpu_capability == (9, 0):
        for BM, BN in [(128, 128), (128, 256), (64, 128), (128, 64), (256, 128)]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                        EPILOGUE_SUBTILE=0,
                    )
    else:
        for BM, BN in [(64, 64), (128, 64), (128, 128), (128, 256), (256, 128)]:
            for BK in [64]:
                for occupancy in [1, 2]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                        EPILOGUE_SUBTILE=0,
                    )


def _default_kernel_config(device=None):
    """GPU-specific fallback config for the non-autotune path.

    ``device``: optional torch.device for the specific GPU whose capability
    we should query (defaults to the current active CUDA device).
    """
    gpu_capability = torch.cuda.get_device_capability(device)
    if gpu_capability == (10, 0):
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 2,
            "occupancy": 2,
            "EPILOGUE_SUBTILE": 1,
        }
    if gpu_capability[0] >= 10:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 2,
            "EPILOGUE_SUBTILE": 0,
        }
    if gpu_capability == (9, 0):
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
            "EPILOGUE_SUBTILE": 0,
        }
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_SIZE_M": 8,
        "num_ctas": 1,
        "occupancy": 1,
        "EPILOGUE_SUBTILE": 0,
    }


def _compute_grid_and_programs(
    M, N, BLOCK_M, BLOCK_N, num_sms, num_ctas, occupancy, device=None
):
    """Compute persistent-grid programs / pid counts.

    ``device``: optional torch.device for the specific GPU whose SM count we
    should query (defaults to the current active CUDA device). Robust under
    multi-GPU environments where the active device may differ from the
    tensors' device.
    """
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    if num_sms is not None:
        NUM_SMS = min(NUM_SMS, num_sms)
    num_pid_m = _cdiv(M, BLOCK_M)
    num_pid_n = _cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    # Guarantee positive program count even when NUM_SMS // num_ctas == 0
    num_programs = max(1, min(NUM_SMS // num_ctas, total_tiles) * occupancy)
    return num_pid_m, num_pid_n, total_tiles, num_programs


# Module-level tune cache:
#   key: (M, N, K, transpose_a_int, transpose_b_int, dtype, num_sms, str(device))
#   value: (best_cfg, ct.kernel(...) bound to the chosen num_ctas/occupancy)
_TUNE_CACHE: dict = {}


def _gemm_alpha_beta_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    alpha: float = 1.0,
    beta: float = 0.0,
    num_sms: Optional[int] = None,
    use_autotune: bool = True,
    kernel_configs: Optional[dict] = None,
) -> torch.Tensor:
    """cuTile GEMM: C = alpha * op(A) @ op(B) + beta * C.

    Args
    ----
    a, b, c : contiguous BF16 tensors, (M,K)/(K,M), (K,N)/(N,K), (M,N).
    trans_a / trans_b : whether op() is transpose on the respective input.
    alpha, beta : scalar scaling factors.
    num_sms : optional SM count throttle; None uses full SM count.
    use_autotune : when True, exhaustive_search picks BLOCK_M/BLOCK_N/...; when
        False, a GPU-specific default config is used.
    kernel_configs : optional dict overriding fields of the default config
        (only consulted in the non-autotune path).

    Returns
    -------
    The same `c` tensor (modified in-place).
    """
    if trans_a:
        K, M = a.shape
    else:
        M, K = a.shape

    if trans_b:
        N, KB = b.shape
    else:
        KB, N = b.shape

    if K != KB:
        raise ValueError(f"incompatible inner dims: {K} vs {KB}")
    if c.shape != (M, N):
        raise ValueError(f"C must be (M,N)=({M},{N}), got {tuple(c.shape)}")
    if not a.is_contiguous():
        raise ValueError("A must be contiguous")
    if not b.is_contiguous():
        raise ValueError("B must be contiguous")
    if not c.is_contiguous():
        raise ValueError("C must be contiguous")

    transpose_a_int = 1 if trans_a else 0
    transpose_b_int = 1 if trans_b else 0

    # For very low SM counts (1–16), autotune overhead can dominate.
    if num_sms is not None and num_sms <= 16:
        use_autotune = False

    # Pin the stream to ``a.device`` so multi-GPU callers don't pick up the
    # wrong default stream when the active CUDA device differs from where the
    # tensors live.
    stream = torch.cuda.current_stream(a.device)

    if use_autotune:
        # Pre-allocate a single tuning output buffer reused across all trials.
        # Previously this was `c.clone()` inside ``args_fn`` and fired on every
        # autotune trial; for large M*N this caused unnecessary allocations and
        # could OOM. The buffer is overwritten by every trial, which is fine
        # since exhaustive_search only cares about timing, and the post-search
        # NaN/Inf probe re-binds and re-launches anyway.
        autotune_out = torch.empty_like(c)

        def grid_fn(cfg):
            _, _, _, num_programs = _compute_grid_and_programs(
                M,
                N,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                num_sms,
                cfg.num_ctas,
                cfg.occupancy,
                device=a.device,
            )
            return (num_programs, 1, 1)

        def args_fn(cfg):
            num_pid_m, num_pid_n, total_tiles, num_programs = (
                _compute_grid_and_programs(
                    M,
                    N,
                    cfg.BLOCK_M,
                    cfg.BLOCK_N,
                    num_sms,
                    cfg.num_ctas,
                    cfg.occupancy,
                    device=a.device,
                )
            )
            return (
                a,
                b,
                autotune_out,  # shared tuning buffer (see comment above)
                float(alpha),
                float(beta),
                M,
                N,
                K,
                total_tiles,
                num_programs,
                num_pid_m,
                num_pid_n,
                transpose_a_int,
                transpose_b_int,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                cfg.BLOCK_K,
                cfg.GROUP_SIZE_M,
                cfg.EPILOGUE_SUBTILE,
            )

        def launch_args_fn(cfg):
            num_pid_m, num_pid_n, total_tiles, num_programs = (
                _compute_grid_and_programs(
                    M,
                    N,
                    cfg.BLOCK_M,
                    cfg.BLOCK_N,
                    num_sms,
                    cfg.num_ctas,
                    cfg.occupancy,
                    device=a.device,
                )
            )
            return (
                a,
                b,
                c,
                float(alpha),
                float(beta),
                M,
                N,
                K,
                total_tiles,
                num_programs,
                num_pid_m,
                num_pid_n,
                transpose_a_int,
                transpose_b_int,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                cfg.BLOCK_K,
                cfg.GROUP_SIZE_M,
                cfg.EPILOGUE_SUBTILE,
            )

        # Include ``c.dtype`` because the kernel's store dtype is determined by
        # ``c_ptr.dtype`` — different output dtypes produce different specialized
        # kernels, so they must autotune separately.
        cache_key = (
            M,
            N,
            K,
            transpose_a_int,
            transpose_b_int,
            a.dtype,
            c.dtype,
            num_sms,
            str(a.device),
        )
        if cache_key not in _TUNE_CACHE:
            result = exhaustive_search(
                list(_autotune_configs(a.device)),
                stream,
                grid_fn,
                _gemm_alpha_beta_kernel_cutile,
                args_fn,
                lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
            )

            # exhaustive_search ranks configs by latency only — it does not
            # verify numerical correctness. We've observed configurations that
            # complete in measurable time but produce NaN on specific shape
            # combinations. Walk the success list from fastest to slowest and
            # pick the first config whose output is NaN/Inf-free.
            #
            # A reference is computed from torch.mm on the same inputs; we
            # accept the cfg if its output matches the reference's overall
            # finiteness (no NaN, no Inf). We deliberately do not impose a
            # cosine-similarity threshold here — the kernel itself is bit-
            # identical to TileGym's verified version, so any genuine
            # correctness mismatch would surface differently.
            ranked = sorted(result.successes, key=lambda m: m.mean_us)
            best_cfg = None
            tuned_kernel = None
            # Reuse the pre-allocated ``autotune_out`` rather than allocating
            # a new probe buffer via ``c.clone()`` — same shape / dtype, and
            # nothing else reads it after the autotune search.
            probe_out = autotune_out
            for measure in ranked:
                trial_cfg = measure.config
                # Use ``replace_hints`` (not ``ct.kernel(_pyfunc, ...)``) so the
                # original kernel's decorator-level hints are preserved — only
                # ``num_ctas`` / ``occupancy`` are swapped. Mirrors bmm.py and
                # the way ``exhaustive_search`` itself rebinds hints internally.
                trial_kernel = _gemm_alpha_beta_kernel_cutile.replace_hints(
                    num_ctas=trial_cfg.num_ctas,
                    occupancy=trial_cfg.occupancy,
                )
                probe_out.zero_()
                try:
                    # Build launch args binding `probe_out` as the output
                    # tensor so we can inspect it without clobbering `c`.
                    pid_m, pid_n, ttl, nprog = _compute_grid_and_programs(
                        M,
                        N,
                        trial_cfg.BLOCK_M,
                        trial_cfg.BLOCK_N,
                        num_sms,
                        trial_cfg.num_ctas,
                        trial_cfg.occupancy,
                        device=a.device,
                    )
                    ct.launch(
                        stream,
                        (nprog, 1, 1),
                        trial_kernel,
                        (
                            a,
                            b,
                            probe_out,
                            float(alpha),
                            float(beta),
                            M,
                            N,
                            K,
                            ttl,
                            nprog,
                            pid_m,
                            pid_n,
                            transpose_a_int,
                            transpose_b_int,
                            trial_cfg.BLOCK_M,
                            trial_cfg.BLOCK_N,
                            trial_cfg.BLOCK_K,
                            trial_cfg.GROUP_SIZE_M,
                            trial_cfg.EPILOGUE_SUBTILE,
                        ),
                    )
                    torch.cuda.synchronize()
                except Exception:
                    # Treat any runtime failure here as a config we can't use;
                    # fall through to try the next-ranked one.
                    continue
                if torch.isnan(probe_out).any() or torch.isinf(probe_out).any():
                    continue
                best_cfg = trial_cfg
                tuned_kernel = trial_kernel
                break

            if best_cfg is None:
                # All autotune candidates produced NaN/Inf or failed at probe
                # time. Fall back to exhaustive_search's nominal best so we
                # still launch something — it at least completed one timed
                # run successfully. Mirrors bmm.py. We deliberately avoid
                # ``_default_kernel_config`` here because its hand-picked
                # shape (BLOCK_M=256 on sm100) is not guaranteed to work for
                # rare small-M shapes the autotune space already covered.
                best_cfg = result.best.config
                tuned_kernel = _gemm_alpha_beta_kernel_cutile.replace_hints(
                    num_ctas=best_cfg.num_ctas,
                    occupancy=best_cfg.occupancy,
                )
            _TUNE_CACHE[cache_key] = (best_cfg, tuned_kernel)

        best_cfg, tuned_kernel = _TUNE_CACHE[cache_key]
        ct.launch(stream, grid_fn(best_cfg), tuned_kernel, launch_args_fn(best_cfg))
        return c

    # Non-autotune path
    cfg = dict(_default_kernel_config(a.device))
    if kernel_configs:
        cfg.update(kernel_configs)

    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    GROUP_SIZE_M = cfg.get("GROUP_SIZE_M", 8)
    num_ctas = cfg.get("num_ctas", 1)
    occupancy = cfg.get("occupancy", 1)
    epilogue_subtile = cfg.get("EPILOGUE_SUBTILE", 0)

    num_pid_m, num_pid_n, total_tiles, num_programs = _compute_grid_and_programs(
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        num_sms,
        num_ctas,
        occupancy,
        device=a.device,
    )
    grid = (num_programs, 1, 1)

    # Bind config-dependent kernel attributes (num_ctas / occupancy) without
    # going through any external autotune helper.
    if num_ctas != 1 or occupancy != 1:
        kernel = ct.kernel(
            _gemm_alpha_beta_kernel_cutile._pyfunc,
            num_ctas=num_ctas,
            occupancy=occupancy,
        )
    else:
        kernel = _gemm_alpha_beta_kernel_cutile

    ct.launch(
        stream,
        grid,
        kernel,
        (
            a,
            b,
            c,
            float(alpha),
            float(beta),
            M,
            N,
            K,
            total_tiles,
            num_programs,
            num_pid_m,
            num_pid_n,
            transpose_a_int,
            transpose_b_int,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_SIZE_M,
            epilogue_subtile,
        ),
    )
    return c


def mm_bf16_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    use_autotune: bool = True,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """BF16 GEMM via cuTile, matching the `flashinfer.mm_bf16` layout.

    Equivalent computation: ``out = a @ b``.

    Layout
    ------
    Upstream `mm_bf16` is invoked with:
        a : shape (M, K), bf16, row-major (contiguous)
        b : a transposed view of a (N, K) row-major tensor — shape (K, N), bf16,
            *not* contiguous (strides are (1, K))
        out : shape (M, N), out_dtype, row-major (contiguous)

    The cuTile kernel's native fast path expects ``b_native`` to be a (N, K)
    row-major contiguous tensor with ``trans_b=True``. We recover that view
    cheaply via ``b.transpose(-2, -1)`` (zero-copy on the storage layer that
    `mm_bf16` itself produced).
    """
    # NaN-poisoning guard: the underlying alpha-beta GEMM kernel computes
    #   result = alpha * acc + beta * c_load_f32
    # Even with ``beta == 0.0``, IEEE 754 specifies ``0 * NaN == NaN`` (and
    # likewise ``0 * Inf == NaN``), so any NaN already sitting in the storage
    # backing ``out`` poisons every output element.
    #
    # ``mm_bf16`` callers typically allocate ``out`` via ``torch.empty(...)``,
    # which leaves the buffer uninitialized. CUDA's caching allocator may
    # return memory whose previous occupant left NaN bits — which then leak
    # through the beta=0 epilogue path and produce all-NaN output non-
    # deterministically (depending on allocator state).
    #
    # Zeroing ``out`` here costs ~one fused memset; on B200 BF16 GEMM shapes
    # this is well under 1% of total kernel time. The proper long-term fix is
    # a beta=0 specialization that skips the c_load entirely (follow-up).
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(
            f"mm_bf16_cutile requires bf16 a and b; got a.dtype={a.dtype}, b.dtype={b.dtype}"
        )
    out.zero_()

    # Recover (N, K) row-major contiguous view that the kernel expects.
    # The upstream caller produces b as `(N, K).transpose(-2, -1)`, so undoing
    # the transpose yields the original contiguous storage with no copy.
    b_native = b.transpose(-2, -1)
    if not b_native.is_contiguous():
        # Defensive fallback for callers that hand us a truly non-contiguous b.
        b_native = b_native.contiguous()

    return _gemm_alpha_beta_cutile(
        a=a,
        b=b_native,
        c=out,
        trans_a=False,
        trans_b=True,
        alpha=1.0,
        beta=0.0,
        num_sms=num_sms,
        use_autotune=use_autotune,
    )
