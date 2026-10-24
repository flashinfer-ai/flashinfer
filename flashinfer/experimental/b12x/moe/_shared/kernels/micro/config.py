"""Small-batch routed MoE configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


_BLOCK_SIZE = 16

# MX-FP6 block scales use 32 elements per vector (see b12x._lib.fp6).
MXFP6_BLOCK_SIZE = 32

_FP8_E4M3_MAX = 448.0

_FC2_TILE_RECIP_GS_NUM = 6.0 * _FP8_E4M3_MAX

_NUM_WARPS = 16

_BLOCK_DIM = _NUM_WARPS * 32

_K_PER_CTA = 16

_MAX_DIRECT_K_SEGMENTS = 12

_W4A16_PACKED_E4M3_SCALE_CACHE_VERSION = 1


def _direct_k_segments_supported(k_segments: int) -> bool:
    return 1 <= int(k_segments) <= _MAX_DIRECT_K_SEGMENTS


def _align_up(value: int, align: int) -> int:
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def _direct_k_segments_for_k(k: int) -> int:
    return _align_up(k, 32 * _BLOCK_SIZE) // (32 * _BLOCK_SIZE)


def _fc1_chunks_for_m(m: int, n: int) -> int:
    rows_per_warp = max(1, int(m))
    rows_per_chunk = max(_BLOCK_SIZE, _NUM_WARPS * rows_per_warp)
    chunks = max(1, int(n) // rows_per_chunk)
    while chunks > 1:
        i_chunk = int(n) // chunks
        if int(n) % chunks == 0 and i_chunk % _BLOCK_SIZE == 0:
            return chunks
        chunks -= 1
    return 1


# --- MX-FP6 (w6a8_mx) decode micro path ------------------------------------
# MX-FP6 weights use 32-element UE8M0 block scales (vs FP4's 16-element E4M3),
# and are stored 3:4-packed (3 bytes per 4 codes). The decode micro path
# mirrors the FP4 micro GEMV but resolves codes through the shared decode LUT
# primitive (b12x._lib.fp6) instead of the FP4 software dot.
_MXFP6_ELEMS_PER_SEGMENT = 32 * MXFP6_BLOCK_SIZE  # 32 scale blocks per K segment


def _fp6_micro_enabled() -> bool:
    """Opt-in gate for the MX-FP6 decode micro path (default off).

    Keeps the validated dynamic FP6 fused path as the only FP6 backend until
    the micro kernel is wired into dispatch and numerically proven.
    """
    return os.environ.get("B12X_ENABLE_FP6_MICRO", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _mxfp6_micro_k_segments_for_k(k: int) -> int:
    return _align_up(k, _MXFP6_ELEMS_PER_SEGMENT) // _MXFP6_ELEMS_PER_SEGMENT


def _mxfp6_micro_shapes_ok(
    m: int, k: int, n: int, num_topk: int, weight_E: int
) -> bool:
    """Format-fundamental shape predicate for the MX-FP6 decode micro path.

    Mirrors ``MoEMicroKernelBackend.is_supported`` but at the FP6 32-element
    block granularity. FC1-chunk feasibility for a specific ``(m, n)`` is
    validated by ``configure`` once the kernel geometry is built; this gate
    covers the constraints that hold independent of the kernel inner-loop
    tiling.
    """
    if m not in (1, 2, 4, 8):
        return False
    if k <= 0 or k % MXFP6_BLOCK_SIZE != 0 or k % 128 != 0:
        return False
    if _mxfp6_micro_k_segments_for_k(k) > _MAX_DIRECT_K_SEGMENTS:
        return False
    if n <= 0 or n % MXFP6_BLOCK_SIZE != 0:
        return False
    return 0 < num_topk <= 32 and weight_E > 0


@dataclass(frozen=True)
class _ShapeConfig:
    """Compile-time shape constants."""

    k_dim: int
    n: int
    two_n: int
    weight_E: int
    num_topk: int
    k_half: int
    n_half: int
    w1_sf_rows: int
    w1_sf_cols: int
    w2_sf_rows: int
    w2_sf_cols: int
    k_blocks: int
    k_segments: int
    k_segments_aligned: bool
    fc1_chunks: int
    fc1_chunks_per_block: int
    i_chunk: int
    rows_per_warp_fc1: int
    smem_xh_stride: int
    smem_xh_size: int
    inter_blocks: int
    inter_u32: int
    fc2_n_chunks: int


def _make_shape_config(
    *,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    weight_E: int,
    is_gated: bool = True,
) -> _ShapeConfig:
    k_half = k // 2
    n_half = n // 2
    two_n = 2 * n if is_gated else n
    w1_sf_rows = _align_up(two_n, 128)
    w1_sf_cols = _align_up(k // _BLOCK_SIZE, 4)
    w2_sf_rows = _align_up(k, 128)
    w2_sf_cols = _align_up(n // _BLOCK_SIZE, 4)
    k_blocks = k // _BLOCK_SIZE
    k_segments = _direct_k_segments_for_k(k)
    k_segments_aligned = k_blocks == k_segments * 32
    fc1_chunks = _fc1_chunks_for_m(m, n)
    fc1_chunks_per_block = 16  # chunks per 128-wide swizzle block
    i_chunk = n // fc1_chunks
    rows_per_warp_fc1 = i_chunk // _NUM_WARPS
    smem_xh_stride = k_segments * (_BLOCK_SIZE // 2) + 1
    smem_xh_size = k_half + (k // (_BLOCK_SIZE * 8))
    inter_blocks = i_chunk // _BLOCK_SIZE
    fc2_n_chunks = (n // 2 + 127) // 128
    inter_u32 = fc2_n_chunks * 128 * num_topk
    return _ShapeConfig(
        k_dim=k,
        n=n,
        two_n=two_n,
        weight_E=weight_E,
        num_topk=num_topk,
        k_half=k_half,
        n_half=n_half,
        w1_sf_rows=w1_sf_rows,
        w1_sf_cols=w1_sf_cols,
        w2_sf_rows=w2_sf_rows,
        w2_sf_cols=w2_sf_cols,
        k_blocks=k_blocks,
        k_segments=k_segments,
        k_segments_aligned=k_segments_aligned,
        fc1_chunks=fc1_chunks,
        fc1_chunks_per_block=fc1_chunks_per_block,
        i_chunk=i_chunk,
        rows_per_warp_fc1=rows_per_warp_fc1,
        smem_xh_stride=smem_xh_stride,
        smem_xh_size=smem_xh_size,
        inter_blocks=inter_blocks,
        inter_u32=inter_u32,
        fc2_n_chunks=fc2_n_chunks,
    )


def _remake_shape_config_fc1(cfg: _ShapeConfig, fc1_chunks: int) -> _ShapeConfig:
    i_chunk = cfg.n // fc1_chunks
    rows_per_warp_fc1 = i_chunk // _NUM_WARPS
    smem_xh_stride = cfg.k_segments * (_BLOCK_SIZE // 2) + 1
    smem_xh_size = cfg.k_half + (cfg.k_dim // (_BLOCK_SIZE * 8))
    inter_blocks = i_chunk // _BLOCK_SIZE
    fc2_n_chunks = (cfg.n // 2 + 127) // 128
    inter_u32 = fc2_n_chunks * 128 * cfg.num_topk
    return _ShapeConfig(
        k_dim=cfg.k_dim,
        n=cfg.n,
        two_n=cfg.two_n,
        weight_E=cfg.weight_E,
        num_topk=cfg.num_topk,
        k_half=cfg.k_half,
        n_half=cfg.n_half,
        w1_sf_rows=cfg.w1_sf_rows,
        w1_sf_cols=cfg.w1_sf_cols,
        w2_sf_rows=cfg.w2_sf_rows,
        w2_sf_cols=cfg.w2_sf_cols,
        k_blocks=cfg.k_blocks,
        k_segments=cfg.k_segments,
        k_segments_aligned=cfg.k_segments_aligned,
        fc1_chunks=fc1_chunks,
        fc1_chunks_per_block=cfg.fc1_chunks_per_block,
        i_chunk=i_chunk,
        rows_per_warp_fc1=rows_per_warp_fc1,
        smem_xh_stride=smem_xh_stride,
        smem_xh_size=smem_xh_size,
        inter_blocks=inter_blocks,
        inter_u32=inter_u32,
        fc2_n_chunks=fc2_n_chunks,
    )
