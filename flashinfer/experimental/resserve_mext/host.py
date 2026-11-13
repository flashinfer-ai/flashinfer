# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""SM100 Fold kernel compilation and launch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import functools

import torch

_KERNEL_CACHE: dict[tuple, tuple] = {}


STORE_MODE_TO_MODE = {
    "tma": "fold_tma",
    "tma_mext": "fold_tma_mext",
    "direct_mext": "fold_mext_direct",
}


DEFAULT_MMA_TILER_MN = (128, 128)
DEFAULT_CLUSTER_SHAPE_MN = (1, 1)
SF_VEC_SIZE = 16

# Row-pair / tma candidates. Tile8 families are not part of this entry.
# Entries that share a tile differ by the AB pipeline stage count.
ROW_PAIR_TACTICS = (
    ((128, 8), (1, 1), 5),
    ((128, 16), (1, 1), 6),
    ((128, 16), (1, 1), 5),
    ((128, 24), (1, 1), None),
    ((128, 32), (1, 1), 5),
    ((128, 32), (1, 1), None),
    ((128, 64), (1, 1), None),
    ((128, 128), (1, 1), None),
    ((256, 64), (2, 1), None),
    ((256, 128), (2, 1), None),
    ((256, 256), (2, 1), None),
)
# Widths with per-tile SFB selection and TMEM addressing.
_ADDRESSED_WIDTHS = (64, 128, 192, 256)
# Widths below 64 are valid only when every activation row fits in tile 0.
_NARROW_TMA_WIDTHS = (8, 16, 24, 32)


@functools.lru_cache(maxsize=1)
def _cutlass_modules():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .kernel_sm100_fold import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    return (
        cutlass,
        cute,
        make_ptr,
        get_max_active_clusters,
        {
            "sm100": Sm100BlockScaledPersistentDenseGemmKernel,
        },
    )


def _compile_fold_gemm(
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    out_dtype: torch.dtype,
    mode: str = "fold_tma",
    kernel_arch: str = "sm100",
    ab_stage_override: int | None = None,
    pi_via_layout: bool = False,
    strided_weight: bool = False,
    fp4_format: str = "nvfp4",
    compile_if_missing: bool = True,
):
    """Compile or retrieve an SM100 Fold kernel."""
    assert mode in (
        "fold_tma",
        "fold_tma_mext",
        "fold_mext_direct",
        "kloop_tma",
        "plain_mn",
    ), mode
    if kernel_arch != "sm100":
        raise ValueError("Fold supports SM100; SM103 Fold is not supported")

    assert mode != "kloop_tma" or (kernel_arch == "sm100" and not strided_weight), (
        "kloop_tma: sm100, single-alpha"
    )

    if fp4_format not in ("nvfp4", "mxfp4"):
        raise ValueError(f"unknown FP4 format: {fp4_format}")
    sf_vec_size = 32 if fp4_format == "mxfp4" else SF_VEC_SIZE
    key = (
        torch.cuda.current_device(),
        fp4_format,
        mma_tiler_mn,
        cluster_shape_mn,
        out_dtype,
        mode,
        kernel_arch,
        ab_stage_override,
        pi_via_layout,
        strided_weight,
    )
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    if not compile_if_missing:
        return None

    cutlass, cute, make_ptr, get_max_active_clusters, kernel_classes = (
        _cutlass_modules()
    )
    kernel_cls = kernel_classes[kernel_arch]
    c_cutlass_dtype = (
        cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
    )

    gemm = kernel_cls(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        enable_pdl=True,
        fold_pairs_tma=(mode in ("fold_tma", "fold_tma_mext", "fold_mext_direct")),
        fold_on_m=(mode in ("fold_tma_mext", "fold_mext_direct")),
        mext_direct_store=(mode == "fold_mext_direct"),
        pi_via_layout=pi_via_layout,
        ab_stage_override=ab_stage_override,
        k_loop=(mode == "kloop_tma"),
    )

    _a6 = mode == "fold_mext_direct"
    sym_m = cute.sym_int(divisibility=16 if _a6 else 1)
    sym_n = cute.sym_int(divisibility=16 if _a6 else 1)
    sym_k = cute.sym_int()
    sym_mtok = cute.sym_int(divisibility=8 if _a6 else 1)

    _a_rows, _b_rows = (
        (sym_n, sym_m)
        if mode in ("fold_tma_mext", "fold_mext_direct", "kloop_tma", "plain_mn")
        else (sym_m, sym_n)
    )

    _w_is_a = mode not in ("fold_tma_mext", "fold_mext_direct", "kloop_tma", "plain_mn")

    def _operand(rows, is_weight):
        if strided_weight and is_weight:
            return cute.runtime.make_fake_tensor(
                cutlass.Uint8,
                (rows, sym_k),
                (cute.sym_int(), 1),
                assumed_align=32,
            )
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (rows, sym_k), stride_order=(1, 0), assumed_align=32
        )

    a_fake = _operand(_a_rows, _w_is_a)
    b_fake = _operand(_b_rows, not _w_is_a)
    if mode == "kloop_tma":
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (_a_rows, cute.sym_int()),
            stride_order=(1, 0),
            assumed_align=32,
        )
    if mode in ("kloop_tma", "plain_mn"):
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_n, sym_m),
            stride_order=(1, 0),
            assumed_align=16,
        )
    elif mode in ("fold_tma", "fold_tma_mext", "fold_mext_direct"):
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_mtok, sym_m),
            stride_order=(1, 0),
            assumed_align=16,
        )
    else:
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_m, sym_n),
            stride_order=(1, 0),
            assumed_align=16,
        )
    sf_dtype = cutlass.Float8E8M0FNU if fp4_format == "mxfp4" else cutlass.Float8E4M3FN
    a_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = (
        None
        if fp4_format == "mxfp4" and kernel_arch == "sm100"
        else cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )
    )
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    def compile_kernel():
        return cute.compile(
            gemm.wrapper,
            a_fake,
            b_fake,
            c_fake,
            1,
            1,
            1,
            1,
            a_sf_ptr,
            b_sf_ptr,
            alpha_fake,
            max_active_clusters,
            stream_fake,
            False,
            lambda x: x,
            options="--opt-level 2 --enable-tvm-ffi",
        )

    specialization = (key[1:], max_active_clusters)
    kernel_name = "kernel_" + hashlib.sha256(repr(specialization).encode()).hexdigest()
    compiled = build_and_load_cute_dsl_kernel(
        "resserve_mext",
        kernel_name,
        compile_kernel,
        extra_key_files=[
            __file__,
            str(Path(__file__).with_name("kernel_sm100_fold.py")),
            str(Path(__file__).parents[2] / "cute_dsl" / "utils.py"),
        ],
    )
    _KERNEL_CACHE[key] = compiled
    return compiled


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def select_row_pair_tactic(m_tok: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the static row-pair tile and cluster for a token count.

    Default pipeline stages. Tile8 is not in this table.
    """
    if m_tok <= 8:
        wanted, cluster = (128, 24), (1, 1)
    elif m_tok <= 16:
        wanted, cluster = (128, 32), (1, 1)
    elif m_tok < 256:
        wanted, cluster = (128, 64), (1, 1)
    else:
        wanted, cluster = (256, 128), (2, 1)
    for tiler, tactic_cluster, stages in ROW_PAIR_TACTICS:
        if tiler == wanted and tactic_cluster == cluster and stages is None:
            return tiler, tactic_cluster
    raise RuntimeError(f"no static row-pair tactic for M={m_tok}")


def _check_tile_width(mma_tiler_mn: tuple[int, int], store_mode: str, m_tok: int):
    """Reject tiles whose SFB and TMEM addressing does not cover this launch.

    Widths 64, 128, 192 and 256 have per-tile scale selection. A narrower tma
    tile has none, so it is limited to launches where tile 0 holds every
    activation row (``2 * m_tok <= width``). Tile 0 then uses the same scale
    slice and TMEM offset as the addressed widths. M-fold modes stay on the
    addressed widths.
    """
    width = mma_tiler_mn[1]
    if width in _ADDRESSED_WIDTHS:
        return
    if store_mode == "tma" and width in _NARROW_TMA_WIDTHS and 2 * m_tok <= width:
        return
    raise ValueError(
        f"Fold tile width {width} is unsupported for store_mode={store_mode!r} "
        f"with {m_tok} token rows; narrow tma tiles require 2*M <= width"
    )


def mext_fold_gemm(
    weight_fp4: torch.Tensor,
    act_fp4_rowpair: torch.Tensor,
    weight_sf: torch.Tensor,
    act_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    mma_tiler_mn: tuple[int, int] = DEFAULT_MMA_TILER_MN,
    cluster_shape_mn: tuple[int, int] = DEFAULT_CLUSTER_SHAPE_MN,
    store_mode: str = "tma",
    kernel_arch: str = "sm100",
    ab_stage_override: int | None = None,
    pi_via_layout: bool = False,
    fp4_format: str = "nvfp4",
) -> torch.Tensor:
    """Compute GEMM with full-channel residue compensation."""
    n_w, k_packed = weight_fp4.shape
    m2 = act_fp4_rowpair.shape[0]
    assert m2 % 2 == 0, "activation must be row-pair interleaved (even rows)"
    m_tok = m2 // 2
    k = k_packed * 2
    _check_tile_width(mma_tiler_mn, store_mode, m_tok)

    _w_ld = weight_fp4.stride(0)
    _strided_w = _w_ld != k_packed
    if _strided_w:
        assert weight_fp4.stride(1) == 1, (
            "weight must stay K-contiguous within a row; got stride "
            f"{weight_fp4.stride()}"
        )
        assert _w_ld > k_packed, (
            f"weight row stride {_w_ld} < width {k_packed}: rows would overlap"
        )

        assert _w_ld % 32 == 0, (
            f"strided weight row stride {_w_ld} is not 32B-aligned; "
            "materialise a contiguous base-K prefix for this layer"
        )
    if act_fp4_rowpair.stride(0) != k_packed or act_fp4_rowpair.stride(1) != 1:
        raise ValueError(
            f"activation must be compact [{m2}, {k_packed}]; got stride "
            f"{act_fp4_rowpair.stride()}"
        )

    if store_mode in ("tma_mext", "direct_mext") and m_tok % 8:
        raise ValueError(
            "M-fold requires tile8 activations padded to a multiple of 8 rows"
        )

    _mode = STORE_MODE_TO_MODE[store_mode]

    no_alpha = fp4_format == "mxfp4" and kernel_arch == "sm100"
    if not no_alpha and alpha.numel() != 1:
        raise ValueError("alpha must contain one element")

    compiled = _compile_fold_gemm(
        mma_tiler_mn,
        cluster_shape_mn,
        out_dtype,
        mode=_mode,
        kernel_arch=kernel_arch,
        ab_stage_override=ab_stage_override,
        pi_via_layout=pi_via_layout,
        strided_weight=_strided_w,
        fp4_format=fp4_format,
    )

    if no_alpha:
        alpha_t = None
    else:
        alpha_t = alpha.reshape(1).to(torch.float32)
    sf_m = _ceil_div(n_w, 128)
    sf_n = _ceil_div(m2, 128)
    sf_k = _ceil_div(k, (32 if fp4_format == "mxfp4" else SF_VEC_SIZE) * 4)

    if store_mode in ("tma_mext", "direct_mext"):
        assert store_mode != "direct_mext" or m2 >= mma_tiler_mn[0], (
            f"direct_mext needs 2*m_tok >= mma_tiler M "
            f"({m2} < {mma_tiler_mn[0]}); use a 1SM (128,x) tile at small m"
        )
        assert store_mode != "direct_mext" or (n_w % 32 == 0 and m_tok % 8 == 0), (
            n_w,
            m_tok,
        )

        d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)
        compiled(
            act_fp4_rowpair,
            weight_fp4,
            d_t,
            sf_n,
            sf_m,
            sf_k,
            act_sf.data_ptr(),
            weight_sf.data_ptr(),
            alpha_t,
        )
        return d_t

    if store_mode == "tma":
        d_t = torch.empty(m_tok, n_w, dtype=out_dtype, device=weight_fp4.device)
        compiled(
            weight_fp4,
            act_fp4_rowpair,
            d_t,
            sf_m,
            sf_n,
            sf_k,
            weight_sf.data_ptr(),
            act_sf.data_ptr(),
            alpha_t,
        )
        return d_t

    raise ValueError(f"unknown store_mode {store_mode!r}")
