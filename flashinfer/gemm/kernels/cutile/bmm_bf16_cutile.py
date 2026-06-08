# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile (cuda.tile Python) BF16 batched matrix multiplication for FlashInfer.

This module provides ``bmm_bf16_cutile`` — a BF16 batched matrix multiplication
that plugs into the existing ``flashinfer.gemm.gemm_base.bmm_bf16`` dispatcher
alongside the ``cudnn`` and ``cutlass`` backends. It targets standard BMM
workloads:

    out[b] = a[b] @ b[b]
    where
        a is (B, M, K) BF16 row-major
        b is (B, K, N) BF16 column-major (caller's view; physically (B, N, K) row-major)

The cuTile kernel and autotune logic are ported verbatim from NVIDIA TileGym
(https://github.com/NVIDIA/TileGym), specifically the standard (non-swap_ab)
``_ragged_bmm_kernel`` in
``src/tilegym/suites/flashinfer/cutile/gemm/ragged_bmm.py``. TileGym's kernel
operates on a flattened ``(total_m, K)`` A with an ``m_indptr`` boundary
tensor (designed for ragged / grouped BMM workloads). For the regular
batched mm case where every batch has the same M, the FlashInfer wrapper
constructs ``m_indptr = arange(0, (B+1)*M, M)`` and reshapes ``A`` to
``(B*M, K)`` before invoking the kernel — this lets us reuse a single
proven kernel rather than authoring a separate regular-bmm variant.

Lessons applied from the BF16 cuTile port (MR adding ``mm_bf16(cutile)``):

* ``from __future__ import annotations`` is NOT used — it would convert the
  ``ct.Constant[int]`` annotations into strings at function-definition time
  and break ``cuda.tile``'s runtime introspection of the
  ``Annotated[int, ConstantAnnotation()]`` metadata.

* No TMA / mma_scaled variant in v1 — the manual ct.mma path is simpler to
  verify and matches the ``mm_bf16`` and ``gemm_fp8_nt_groupwise`` cuTile
  paths' design.

* No swap_ab variant in v1 — the standard kernel covers the common
  ``M >= 128`` workloads. The swap_ab variant (better for small-M cases)
  is a follow-up MR.
"""

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search


def make_bmm_bf16_tune_cache() -> dict:
    """Create a fresh tune cache for :func:`bmm_bf16_cutile`.

    The cache maps ``(B, M, N, K, transpose_a, transpose_b, dtype, device)``
    shape tuples to the best cuTile tile config found by exhaustive search.
    Pass the same dict across multiple calls to avoid re-tuning shapes already
    seen — the first call for a new shape runs exhaustive search (typically a
    few seconds); subsequent calls return immediately from the cache.

    Example::

        cache = make_bmm_bf16_tune_cache()

        # First call — runs exhaustive_search for this (B, M, N, K) shape.
        out1 = bmm_bf16_cutile(A, B, out, tune_cache=cache)

        # Second call at the same shape — hits cache, near-zero overhead.
        out2 = bmm_bf16_cutile(A, B, out, tune_cache=cache)

    Leave *tune_cache* unset (or ``None``) to use the module-level default
    cache, which persists for the lifetime of the process.
    """
    return {}


# Module-level default tune cache.  Key: (B, M, N, K, transpose_a,
# transpose_b, a_dtype, device) — output shape and types determine kernel
# specialization.  Use :func:`make_bmm_bf16_tune_cache` to create an
# independent cache (e.g. in tests or multi-tenant serving scenarios).
_BMM_BF16_TUNE_CACHE: dict = make_bmm_bf16_tune_cache()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# Ported verbatim from NVIDIA TileGym
# (https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/suites/flashinfer/cutile/gemm/ragged_bmm.py).
# Standard (non-swap_ab) kernel — single outer K loop, no nested scale loop.
@ct.kernel
def _bmm_bf16_kernel_cutile(
    a,  # Input matrix A [total_m, K] or [K, total_m] if transpose_a
    b,  # Input matrix B [Q, N, K] or [Q, K, N] if not transpose_b
    c,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    q,  # Number of batches
    max_m,  # Max segment size
    n,  # Output N dimension
    TRANSPOSE_A: ct.Constant[int],
    TRANSPOSE_B: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """cuTile BMM via the ragged-bmm interface.

    Performs ``A @ B`` or ``A @ B^T`` (per TRANSPOSE_B) where:
    - A is flattened with segment offsets ``m_indptr`` (regular BMM uses
      a fixed-stride indptr to express equal-sized batches).
    - B is batched ``(Q, N, K)`` or ``(Q, K, N)``.
    - Output C is ``(total_m, N)``.

    Uses persistent scheduling with static grid and GROUP_SIZE_M tile
    swizzling for L2 locality.
    """
    pid = ct.bid(0)

    if TRANSPOSE_A == 1:
        num_k_tiles = ct.num_tiles(a, axis=0, shape=(BLOCK_K, BLOCK_M))
    else:
        num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K))
    num_pid_m = ct.cdiv(max_m, BLOCK_M)
    num_pid_n = ct.cdiv(n, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * q
    num_programs = ct.num_blocks(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for current_pid in range(pid, total_tiles, num_programs):
        pid_q = current_pid // tiles_per_batch
        pid_in_batch = current_pid % tiles_per_batch

        group_id = pid_in_batch // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + (pid_in_batch % group_size_m_actual)
        pid_n = (pid_in_batch % num_pid_in_group) // group_size_m_actual

        # Load segment boundaries
        m_start_tile = ct.load(m_indptr, index=(pid_q,), shape=(1,))
        m_start = m_start_tile.item()
        m_end_tile = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,))
        m_end = m_end_tile.item()
        valid_m = m_end - m_start

        if pid_m * BLOCK_M < valid_m:
            # Slice A and C along the M axis for this segment.
            if TRANSPOSE_A == 1:
                Ai = a.slice(axis=1, start=m_start, stop=m_end)
            else:
                Ai = a.slice(axis=0, start=m_start, stop=m_end)
            Ci = c.slice(axis=0, start=m_start, stop=m_end)

            dot_acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            for k in range(num_k_tiles):
                # ─── A load ───
                if TRANSPOSE_A == 1:
                    a_block_kt = ct.load(
                        Ai,
                        index=(k, pid_m),
                        shape=(BLOCK_K, BLOCK_M),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    a_block = ct.permute(a_block_kt, (1, 0))
                else:
                    a_block = ct.load(
                        Ai,
                        index=(pid_m, k),
                        shape=(BLOCK_M, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                # ─── B load ───
                if TRANSPOSE_B == 1:
                    # B is [Q, N, K] — N-major within batch.
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_block_nk = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))
                    b_block_t = ct.permute(b_block_nk, (1, 0))
                else:
                    # B is [Q, K, N] — K-major within batch.
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, k, pid_n),
                        shape=(1, BLOCK_K, BLOCK_N),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_block_t = ct.reshape(b_block_3d, (BLOCK_K, BLOCK_N))

                dot_acc = ct.mma(a_block, b_block_t, acc=dot_acc)

            c_block = ct.astype(dot_acc, c.dtype)
            ct.store(Ci, index=(pid_m, pid_n), tile=c_block)


def _bmm_bf16_autotune_configs(device=None):
    """Yield autotune configurations.

    Ported from TileGym's ``_ragged_bmm_autotune_configs_standard``.
    Blackwell (SM10x/SM12x) gets the expanded config grid; older arches use
    a conservative subset.

    ``device``: optional torch.device for the specific GPU whose capability
    we should query (defaults to the current active CUDA device).
    """
    gpu_capability = torch.cuda.get_device_capability(device)

    if gpu_capability[0] >= 10:
        # Blackwell B200/B300/B100 (sm_100/103/120/121)
        for BM, BN in [
            (256, 256),
            (128, 256),
            (128, 128),
            (256, 128),
            (64, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    else:
        # Conservative fallback (Hopper/Ampere/Ada — not officially supported
        # but kept so cuda.tile autotune still has valid configs).
        for BM, BN in [(128, 128), (64, 128), (128, 64), (128, 256)]:
            for BK in [64]:
                for occupancy in [1, 2]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )


def _bmm_bf16_autotune_and_launch(
    stream,
    a_flat,  # (B*M, K) row-major bf16
    b_3d,  # (B, N, K) row-major bf16 (= transpose of caller's (B, K, N) col-major)
    c_flat,  # (B*M, N) row-major output
    m_indptr,  # (B+1,) int32, regular stride
    B,
    M,
    N,
    K,
    transpose_a_int,
    transpose_b_int,
    tune_cache: dict | None = None,
):
    """Launch BMM kernel with exhaustive_search autotuning.

    *tune_cache* is the dict to read/write tuned configs.  ``None`` falls
    back to the module-level :data:`_BMM_BF16_TUNE_CACHE`.
    """
    if tune_cache is None:
        tune_cache = _BMM_BF16_TUNE_CACHE
    NUM_SMS = torch.cuda.get_device_properties(a_flat.device).multi_processor_count
    # Include ``c_flat.dtype`` because the kernel epilogue's store dtype is
    # ``c.dtype`` — different output dtypes produce different specialized kernels.
    cache_key = (
        B,
        M,
        N,
        K,
        transpose_a_int,
        transpose_b_int,
        a_flat.dtype,
        c_flat.dtype,
        str(a_flat.device),
    )

    if cache_key not in tune_cache:
        configs = list(_bmm_bf16_autotune_configs(a_flat.device))

        # Pre-allocate a single tuning output buffer reused across all trials —
        # avoids `c_flat.clone()` per trial (gemm.py uses the same pattern; see
        # comment there for rationale).
        autotune_out = torch.empty_like(c_flat)

        def grid_fn(cfg):
            num_pid_m = _cdiv(M, cfg.BLOCK_M)
            num_pid_n = _cdiv(N, cfg.BLOCK_N)
            tiles_per_batch = num_pid_m * num_pid_n
            total_tiles = tiles_per_batch * B
            num_programs = min(NUM_SMS // cfg.num_ctas, total_tiles) * cfg.occupancy
            return (num_programs, 1, 1)

        def args_fn(cfg):
            return (
                a_flat,
                b_3d,
                autotune_out,
                m_indptr,
                B,
                M,
                N,
                transpose_a_int,
                transpose_b_int,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                cfg.BLOCK_K,
                cfg.GROUP_SIZE_M,
            )

        def hints_fn(cfg):
            return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

        result = exhaustive_search(
            configs,
            stream,
            grid_fn,
            _bmm_bf16_kernel_cutile,
            args_fn,
            hints_fn,
        )

        # exhaustive_search ranks configs by latency only — verify correctness
        # (no NaN, no Inf) by re-running each ranked config in order and
        # accepting the first one whose output is finite. Mirrors gemm.py.
        ranked = sorted(result.successes, key=lambda m: m.mean_us)
        best_cfg = None
        for measure in ranked:
            trial_cfg = measure.config
            trial_kernel = _bmm_bf16_kernel_cutile.replace_hints(**hints_fn(trial_cfg))
            try:
                autotune_out.zero_()
                ct.launch(stream, grid_fn(trial_cfg), trial_kernel, args_fn(trial_cfg))
                torch.cuda.synchronize()
            except Exception:
                continue
            if torch.isnan(autotune_out).any() or torch.isinf(autotune_out).any():
                continue
            best_cfg = trial_cfg
            break
        if best_cfg is None:
            # All autotune candidates produced NaN/Inf or failed; fall back to
            # exhaustive_search's nominal best so we still launch something.
            best_cfg = result.best.config

        tune_cache[cache_key] = (
            best_cfg,
            _bmm_bf16_kernel_cutile.replace_hints(**hints_fn(best_cfg)),
        )

    best_cfg, tuned_kernel = tune_cache[cache_key]
    BM = best_cfg.BLOCK_M
    BN = best_cfg.BLOCK_N
    num_pid_m = _cdiv(M, BM)
    num_pid_n = _cdiv(N, BN)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * B
    num_programs = min(NUM_SMS // best_cfg.num_ctas, total_tiles) * best_cfg.occupancy
    ct.launch(
        stream,
        (num_programs, 1, 1),
        tuned_kernel,
        (
            a_flat,
            b_3d,
            c_flat,
            m_indptr,
            B,
            M,
            N,
            transpose_a_int,
            transpose_b_int,
            best_cfg.BLOCK_M,
            best_cfg.BLOCK_N,
            best_cfg.BLOCK_K,
            best_cfg.GROUP_SIZE_M,
        ),
    )


def bmm_bf16_cutile(
    A: torch.Tensor,  # (B, M, K) bf16, row-major
    B: torch.Tensor,  # (B, K, N) bf16, column-major (caller's view)
    out: torch.Tensor,  # (B, M, N) bf16, row-major
    tune_cache: dict | None = None,
) -> torch.Tensor:
    """BF16 batched matrix multiplication via cuTile.

    Computes ``out[b] = A[b] @ B[b]`` for each batch ``b``.

    The kernel re-uses TileGym's ragged-bmm kernel by:
    1. Flattening ``A`` from ``(B, M, K)`` to ``(B*M, K)`` (preserves
       row-major contig).
    2. Materializing ``B`` as ``(B, N, K)`` row-major via the caller's
       transposed view — flashinfer's ``bmm_bf16`` documents ``B`` as
       column-major ``(B, K, N)``, which is the same memory as
       ``(B, N, K)`` row-major.
    3. Building ``m_indptr = arange(0, (Bs+1)*M, M)`` so every "segment"
       has size M — i.e. the regular BMM case.
    4. Flattening ``out`` to ``(B*M, N)`` row-major.

    Returns the input ``out`` (modified in place).

    Restrictions:
    * BF16 inputs only (the kernel uses ``ct.mma`` which accepts bf16).
    * Output dtype must be bf16 (caller-side requirement; can be relaxed
      via ``ct.astype`` in the kernel if needed).
    """
    if A.dim() != 3 or B.dim() != 3:
        raise ValueError(
            f"bmm_bf16_cutile expects 3D inputs; got A.shape={A.shape}, B.shape={B.shape}"
        )

    Bs, M, K = A.shape
    Bs_b, Kb, N = B.shape
    if Bs != Bs_b:
        raise ValueError(f"Batch dim mismatch: A.shape[0]={Bs}, B.shape[0]={Bs_b}")
    if Kb != K:
        raise ValueError(f"K dim mismatch: A.shape[2]={K}, B.shape[1]={Kb}")
    if out.shape != (Bs, M, N):
        raise ValueError(f"out.shape must be {(Bs, M, N)}; got {tuple(out.shape)}")

    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise ValueError(
            f"bmm_bf16_cutile requires bf16 A and B; got A.dtype={A.dtype}, B.dtype={B.dtype}"
        )
    if out.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            f"bmm_bf16_cutile supports bfloat16 / float16 / float32 out; got {out.dtype}"
        )

    # B in flashinfer is documented as column-major (B, K, N) — same memory
    # as row-major (B, N, K). The kernel wants (Q, N, K) for TRANSPOSE_B=1.
    # B.transpose(-2, -1) gives a (B, N, K) view that is row-major contig
    # when the original B was column-major contig (which is the common
    # case after the caller does e.g. ``weight.transpose(-2, -1)``).
    B_kernel = B.transpose(-2, -1)
    if not B_kernel.is_contiguous():
        B_kernel = B_kernel.contiguous()

    # A must be contiguous (B, M, K) row-major; flatten to (B*M, K).
    if not A.is_contiguous():
        A = A.contiguous()
    A_flat = A.reshape(Bs * M, K)

    # Output flat view (B*M, N) row-major.
    if not out.is_contiguous():
        raise ValueError("out must be contiguous (B, M, N) row-major")
    out_flat = out.reshape(Bs * M, N)

    # Regular-stride m_indptr: every "segment" is exactly M tokens.
    m_indptr = torch.arange(
        0,
        (Bs + 1) * M,
        M,
        device=A.device,
        dtype=torch.int32,
    )

    # Pin the stream to ``A.device`` for multi-GPU correctness — see gemm.py
    # for the same fix.
    _bmm_bf16_autotune_and_launch(
        torch.cuda.current_stream(A.device),
        A_flat,
        B_kernel,
        out_flat,
        m_indptr,
        Bs,
        M,
        N,
        K,
        transpose_a_int=0,
        transpose_b_int=1,
        tune_cache=tune_cache,
    )
    return out
