# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""cp.async-pipelined MXFP8 GEMM whose four compute warps hold split-K partials (warp split-K).

The MXFP8 sibling of ``dense_bf16_gemm_warp_splitk``: same CTA structure (weight
loader warps, activation loader warps, four compute warps that each own one K
quarter of every tile and reduce through a shared-memory mailbox), but the
operands are E4M3 and the E8M0 block scales (one per 32 K elements, the 128x4
swizzled layout of ``mxfp8_quantize``) are applied on the FP32 accumulators.

sm_100/sm_103 have no FP8 ``mma.sync``: ptxas lowers ``m16n8k32.e4m3`` to FP16
unpacks (``F2FP``), two ``HMMA.16816`` and one ``FADD`` per accumulator to add C.
The kernel does the same unpack (``cvt.rn.f16x2.e4m3x2``, exact) and issues the two
FP16 ``m16n8k16`` itself, which drops those FADDs.

Scales are not part of the MMA. One K block of 32 covers exactly one scale block,
so each K block is computed into a zeroed accumulator and folded into the running
sum as ``acc += partial * scale_w[row] * scale_x[token]``.

Data movement reuses the BF16 kernel's helpers by viewing the E4M3 tensors as
16-bit elements: a ``k32`` FP8 slice and a ``k16`` BF16 slice are both 32 bytes,
so the swizzled shared-memory layout and the ``ldmatrix`` addressing are shared.
"""

from __future__ import annotations

import dataclasses
import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
import torch as _torch
from cutlass import const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack

from .dense_bf16_gemm_warp_splitk import (
    _A_LOADER_THREADS,
    _A_LOADER_WARPS,
    _COMPUTE_WARPS,
    _CTA_RESERVED_SMEM,
    _MAILBOX_GROUP_VALUES,
    _MAILBOX_PADDED_GROUP_VALUES,
    _MAX_K_TILES,
    _MAX_STAGES,
    _SM_SMEM_BYTES,
    _SMEM_CAPACITY,
    _SUPPORTED_B_LOADER_WARPS,
    _SUPPORTED_OUTPUT_TILES,
    _a_fragment_offset,
    _align_up,
    _b_pair_offset,
    _issue_weight_tile,
    _l2_policy_evict_first,
    _ldmatrix_x4,
    _make_smem_layout,
    _partial_offset,
)

# One K block per MMA step: m16n8k32 in E4M3, run as two FP16 m16n8k16.
_MMA_SHAPE = (16, 8, 32)
# Above 16 tokens every extra token tile re-reads the weight.
_MAX_M = 16
_SUPPORTED_TOKEN_TILES = (8, 16)
# Longest K served: splitting K only inside a CTA cannot add CTAs, so long K is
# left to the cluster split-K kernel.
_MAX_K = 5120
_SF_VEC_SIZE = 32
# One 4-byte scale word covers four K blocks (the inner dimension of the 128x4
# swizzled scale layout), i.e. 128 K elements.
_SF_WORD_K = 4 * _SF_VEC_SIZE
# FP8 elements per K tile. Both are the byte width of the BF16 kernel's 128 / 256
# element tiles, so the shared loader/ldmatrix addressing carries over.
_SUPPORTED_K_TILES = (256, 512)
_OUT_DTYPES = (_torch.bfloat16, _torch.float16)
# Bytes of A+B tiles a CTA keeps in flight (stages * (output_tile + token_tile) *
# k_tile): enough to cover DRAM latency without an over-deep ring.
_TARGET_RING_BYTES = 48 * 1024
_MIN_RING_BYTES = 32 * 1024
_MAX_RING_BYTES = 128 * 1024


@dsl_user_op
def _e8m0_byte_to_f32(words, byte_idx: int, *, loc=None, ip=None):
    """FP32 value of the E8M0 scale in byte ``byte_idx`` (compile-time) of ``words``.

    ``2**(e - 127)`` is the FP32 whose exponent field is ``e``: one ``prmt`` with an
    immediate selector extracts the byte, one ``shl`` places it (a ``bfe`` with a
    runtime position lowers to ~5 SASS instructions). The encodings ``0``
    (2**-127, subnormal in FP32) and ``255`` (NaN) are not special-cased:
    ``mxfp8_quantize`` never emits them for a block with nonzero data.
    """
    selector = 0x4440 | byte_idx  # byte ``byte_idx`` of the word, upper bytes zero
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [cutlass.Int32(words).ir_value(loc=loc, ip=ip)],
            f"{{ .reg .b32 t; prmt.b32 t, $1, 0, {selector}; shl.b32 t, t, 23; "
            "mov.b32 $0, t; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fma_f32(a, b, c, *, loc=None, ip=None):
    """``a * b + c`` as one ``fma.rn.f32`` rather than relying on contraction."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _unpack_e4m3x4(reg, *, loc=None, ip=None):
    """The four E4M3 values of one fragment register as two FP16x2 registers.

    ``lo`` holds bytes 0-1 and ``hi`` bytes 2-3. The conversion is exact: every
    E4M3 value is representable in FP16.
    """
    pair = llvm.inline_asm(
        llvm.StructType.get_literal([cutlass.Int32.mlir_type] * 2),
        [cutlass.Int32(reg).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 l, h; mov.b32 {l, h}, $2; "
        "cvt.rn.f16x2.e4m3x2 $0, l; cvt.rn.f16x2.e4m3x2 $1, h; }",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Int32(llvm.extractvalue(cutlass.Int32.mlir_type, pair, [i]))
        for i in range(2)
    )


@dsl_user_op
def _mma_k32_from_f16(a, b, *, loc=None, ip=None):
    """One E4M3 ``m16n8k32`` from unpacked fragments, into a zero accumulator.

    ``a`` is ``_unpack_e4m3x4`` of the four A registers, ``b`` of the two B
    registers. Register ``r`` of the E4M3 fragment holds K ``4t..4t+3`` (``r`` = 0, 1;
    ``16 + 4t..`` for 2, 3), so ``lo``/``hi`` fill the FP16 slots ``2t, 2t+1`` and
    ``8 + 2t, 8 + 2t + 1`` of one ``m16n8k16``. A and B use the same K permutation,
    so the dot product is unchanged. Returns the four FP32 accumulator values.
    """
    operands = [
        *(a[0][0], a[1][0], a[0][1], a[1][1]),
        *(b[0][0], b[0][1]),
        *(a[2][0], a[3][0], a[2][1], a[3][1]),
        *(b[1][0], b[1][1]),
    ]
    zero4 = "{0f00000000, 0f00000000, 0f00000000, 0f00000000}"
    result = llvm.inline_asm(
        llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 4),
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in operands],
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        f"{{$0, $1, $2, $3}}, {{$4, $5, $6, $7}}, {{$8, $9}}, {zero4};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{$0, $1, $2, $3}, {$10, $11, $12, $13}, {$14, $15}, {$0, $1, $2, $3};",
        # Early-clobber outputs: they are written before $10-$15 are read.
        "=&f,=&f,=&f,=&f," + ",".join(["r"] * 12),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(cutlass.Float32.mlir_type, result, [i]))
        for i in range(4)
    )


@dataclasses.dataclass(frozen=True)
class MxFp8WarpSplitKTactic:
    output_tile: int
    token_tile: int
    k_tile: int  # FP8 elements per K tile
    stages: int
    b_loader_warps: int


def _sf_words(tactic_rows: int, k_tile: int) -> int:
    """4-byte scale words one stage holds for ``tactic_rows`` tile rows."""
    return tactic_rows * (k_tile // _SF_WORD_K)


def _smem_bytes(tactic: MxFp8WarpSplitKTactic) -> int:
    cursor = tactic.output_tile * tactic.k_tile * tactic.stages
    cursor = _align_up(cursor, 1024)
    cursor += tactic.token_tile * tactic.k_tile * tactic.stages
    cursor = _align_up(cursor, 16)
    cursor += (
        _sf_words(tactic.output_tile + tactic.token_tile, tactic.k_tile)
        * 4
        * tactic.stages
    )
    cursor = _align_up(cursor, 16)
    partial_stride = (
        tactic.output_tile * tactic.token_tile // _MAILBOX_GROUP_VALUES
    ) * _MAILBOX_PADDED_GROUP_VALUES
    cursor += _COMPUTE_WARPS * partial_stride * 4
    cursor = _align_up(cursor, 8)
    cursor += tactic.stages * 8
    cursor = _align_up(cursor, 8)
    return cursor + tactic.stages * 8


def _k_tile_count_supported(k_tile: int, k_tile_count: int) -> bool:
    # Same SASS constraint as the BF16 kernel's 256-element tile (512 FP8 bytes):
    # the two-tile schedule can generate an invalid hinted LDGSTS.
    return k_tile != 512 or k_tile_count != 2


def _min_stages(k: int, k_tile: int) -> int:
    return 1 if k // k_tile <= 1 else 2


def _legal_stages(
    output_tile: int, token_tile: int, k_tile: int, k: int, b_loader_warps: int
) -> tuple[int, ...]:
    k_tile_count = k // k_tile
    if k_tile_count > _MAX_K_TILES or not _k_tile_count_supported(k_tile, k_tile_count):
        return ()
    return tuple(
        stages
        for stages in range(_min_stages(k, k_tile), min(_MAX_STAGES, k_tile_count) + 1)
        if _smem_bytes(
            MxFp8WarpSplitKTactic(
                output_tile, token_tile, k_tile, stages, b_loader_warps
            )
        )
        <= _SMEM_CAPACITY
    )


def _ring_bytes(output_tile: int, token_tile: int, k_tile: int, stages: int) -> int:
    return stages * (output_tile + token_tile) * k_tile


def _pruned_stages(
    legal: tuple[int, ...], output_tile: int, token_tile: int, k_tile: int
) -> tuple[int, ...]:
    """Autotune ring depths: powers of two (> 2) holding 32-128 KB in flight.

    Every stage count is a separate compilation, and shallow or very deep rings
    are measurably slower, so they only add compile time and room for tuner
    noise. The deepest legal ring is kept when K is too short to fill the window.
    ``default_tactic`` is always listed by ``autotune_tactics``.
    """
    if not legal:
        return ()
    keep = {
        stages
        for stages in legal
        if stages > 2
        and stages & (stages - 1) == 0
        and _MIN_RING_BYTES
        <= _ring_bytes(output_tile, token_tile, k_tile, stages)
        <= _MAX_RING_BYTES
    }
    return tuple(sorted(keep or {legal[-1]}))


def _k_supported(k: int) -> bool:
    return 0 < k <= _MAX_K and any(
        k % k_tile == 0 and k // k_tile <= _MAX_K_TILES for k_tile in _SUPPORTED_K_TILES
    )


def validate_tactic(tactic: MxFp8WarpSplitKTactic, m: int, n: int, k: int) -> None:
    if tactic.output_tile not in _SUPPORTED_OUTPUT_TILES:
        raise ValueError(f"unsupported output_tile={tactic.output_tile}")
    if tactic.token_tile not in _SUPPORTED_TOKEN_TILES:
        raise ValueError(f"unsupported token_tile={tactic.token_tile}")
    if tactic.k_tile not in _SUPPORTED_K_TILES:
        raise ValueError(f"unsupported k_tile={tactic.k_tile}")
    if tactic.b_loader_warps not in _SUPPORTED_B_LOADER_WARPS:
        raise ValueError(f"unsupported b_loader_warps={tactic.b_loader_warps}")
    if not 1 <= m <= _MAX_M or n <= 0 or n % tactic.output_tile:
        raise ValueError(f"unsupported shape {(m, n, k)}")
    if k <= 0 or k % tactic.k_tile:
        raise ValueError(f"K={k} must be divisible by k_tile={tactic.k_tile}")
    k_tile_count = k // tactic.k_tile
    if k_tile_count > _MAX_K_TILES:
        raise ValueError(
            f"K={k} spans more than {_MAX_K_TILES} tiles of {tactic.k_tile}"
        )
    if not _k_tile_count_supported(tactic.k_tile, k_tile_count):
        raise ValueError(f"unsupported k_tile={tactic.k_tile} for K={k}")
    if not (
        _min_stages(k, tactic.k_tile) <= tactic.stages <= min(_MAX_STAGES, k_tile_count)
    ):
        raise ValueError(f"invalid stages={tactic.stages}")
    required_smem = _smem_bytes(tactic)
    if required_smem > _SMEM_CAPACITY:
        raise ValueError(f"tactic requires {required_smem} bytes of shared memory")


def _swizzled_scale_len(rows: int, k: int) -> int:
    return -(-rows // 128) * 128 * (-(-(k // _SF_VEC_SIZE) // 4) * 4)


def validate_inputs(
    a: _torch.Tensor,
    b: _torch.Tensor,
    a_descale: _torch.Tensor,
    b_descale: _torch.Tensor,
    out: _torch.Tensor,
) -> tuple[int, int, int]:
    """Check one ``mm_mxfp8`` problem; return ``(m, n, k)``.

    ``a`` is ``(M, K)`` row-major E4M3, ``b`` is ``(K, N)`` column-major E4M3,
    both scales are 1D uint8 in the 128x4 swizzled layout.
    """
    tensors = (a, b, a_descale, b_descale, out)
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("a, b, scales, and out must be torch tensors")
    if any(tensor.ndim != 2 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 2D tensors")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all tensors must be CUDA tensors on the same device")
    if a.dtype != _torch.float8_e4m3fn or b.dtype != _torch.float8_e4m3fn:
        raise ValueError("a and b must have float8_e4m3fn dtype")
    if out.dtype not in _OUT_DTYPES:
        raise ValueError("out must have bfloat16 or float16 dtype")
    if a_descale.dtype != _torch.uint8 or b_descale.dtype != _torch.uint8:
        raise ValueError("block scales must be uint8 (E8M0)")
    if a_descale.ndim != 1 or b_descale.ndim != 1:
        raise ValueError("block scales must be 1D swizzled tensors")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"M must be in [1, {_MAX_M}], got {m}")
    if n <= 0 or n % 16:
        raise ValueError(f"N={n} must be a positive multiple of 16")
    if not _k_supported(k):
        raise ValueError(f"K={k} must be a multiple of 256 and at most {_MAX_K}")
    if a_descale.numel() < _swizzled_scale_len(m, k):
        raise ValueError("a_descale is shorter than the swizzled scale layout")
    if b_descale.numel() < _swizzled_scale_len(n, k):
        raise ValueError("b_descale is shorter than the swizzled scale layout")

    if a.stride() != (k, 1):
        raise ValueError(f"a must be packed row-major, got stride {a.stride()}")
    if b.stride() != (1, k):
        raise ValueError(f"b must be packed column-major, got stride {b.stride()}")
    if out.stride() != (n, 1):
        raise ValueError(f"out must be packed row-major, got stride {out.stride()}")
    if not a_descale.is_contiguous() or not b_descale.is_contiguous():
        raise ValueError("block scales must be contiguous")
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)) or any(
        tensor.data_ptr() % 4 for tensor in (a_descale, b_descale)
    ):
        raise ValueError("a, b, out must be 32-byte and scales 4-byte aligned")
    return m, n, k


def default_tactic(m: int, n: int, k: int) -> MxFp8WarpSplitKTactic:
    k_tile = 256 if k <= 1024 or k % 512 else 512
    k_tile_count = k // k_tile
    sm_count = _torch.cuda.get_device_properties(
        _torch.cuda.current_device()
    ).multi_processor_count
    output_tiles = tuple(tile for tile in _SUPPORTED_OUTPUT_TILES if n % tile == 0)
    if not output_tiles:
        raise ValueError(f"N={n} must be divisible by a supported output tile")
    # Minimize scheduled wave work, preferring the wider tile on ties.
    output_tile = min(
        output_tiles,
        key=lambda tile: (((n // tile + sm_count - 1) // sm_count) * tile, -tile),
    )
    # Splitting M over two half-size token tiles doubles the CTA count at the cost
    # of a second pass over each weight tile; that only pays while the doubled grid
    # still fits in one wave.
    token_tile = 8 if m <= 8 else 16
    if token_tile > 8 and 2 * (n // output_tile) <= sm_count:
        token_tile //= 2
    b_loader_warps = 2 if token_tile >= 16 or k_tile_count >= 8 else 1
    legal_stages = _legal_stages(output_tile, token_tile, k_tile, k, b_loader_warps)
    if not legal_stages:
        raise ValueError(f"no legal tactic for shape {(m, n, k)}")
    cta_count = (n // output_tile) * ((m + token_tile - 1) // token_tile)
    target_residency = 2 if cta_count > sm_count else 1
    resident_stages = tuple(
        stages
        for stages in legal_stages
        if target_residency
        * (
            _smem_bytes(
                MxFp8WarpSplitKTactic(
                    output_tile, token_tile, k_tile, stages, b_loader_warps
                )
            )
            + _CTA_RESERVED_SMEM
        )
        <= _SM_SMEM_BYTES
    )
    # The ring closest to _TARGET_RING_BYTES (deeper on ties).
    stages = min(
        resident_stages or legal_stages,
        key=lambda stages: (
            abs(
                _ring_bytes(output_tile, token_tile, k_tile, stages)
                - _TARGET_RING_BYTES
            ),
            -stages,
        ),
    )
    tactic = MxFp8WarpSplitKTactic(
        output_tile, token_tile, k_tile, stages, b_loader_warps
    )
    validate_tactic(tactic, m, n, k)
    return tactic


def autotune_tactics(m: int, n: int, k: int) -> list[MxFp8WarpSplitKTactic]:
    if not 1 <= m <= _MAX_M or n <= 0 or n % 16 or not _k_supported(k):
        return []
    tactics: list[MxFp8WarpSplitKTactic] = []
    for output_tile in _SUPPORTED_OUTPUT_TILES:
        if n % output_tile or (output_tile == 32 and k > 3072):
            continue
        for token_tile in _SUPPORTED_TOKEN_TILES:
            for k_tile in _SUPPORTED_K_TILES:
                if k % k_tile:
                    continue
                # Prune same-grid token padding for multi-tile K.
                # Wider tiles can win when K fits one tile.
                if k // k_tile > 1 and any(
                    smaller < token_tile
                    and (m + smaller - 1) // smaller
                    == (m + token_tile - 1) // token_tile
                    for smaller in _SUPPORTED_TOKEN_TILES
                ):
                    continue
                for b_loader_warps in _SUPPORTED_B_LOADER_WARPS:
                    tactics.extend(
                        MxFp8WarpSplitKTactic(
                            output_tile, token_tile, k_tile, stages, b_loader_warps
                        )
                        for stages in _pruned_stages(
                            _legal_stages(
                                output_tile, token_tile, k_tile, k, b_loader_warps
                            ),
                            output_tile,
                            token_tile,
                            k_tile,
                        )
                    )
    default = default_tactic(m, n, k)
    return [default, *(tactic for tactic in tactics if tactic != default)]


class CpAsyncMxFp8WarpSplitKKernel:
    def __init__(self, tactic: MxFp8WarpSplitKTactic, use_pdl: bool) -> None:
        self.output_tile = tactic.output_tile
        self.token_tile = tactic.token_tile
        self.k_tile = tactic.k_tile  # FP8 elements
        # The kernel addresses data as 16-bit elements (two FP8 each).
        self.k_tile16 = tactic.k_tile // 2
        self.sf_groups = tactic.k_tile // _SF_WORD_K  # scale words per row per tile
        self.stages = tactic.stages
        self.b_loader_warps = tactic.b_loader_warps
        self.compute_warp_base = _A_LOADER_WARPS + self.b_loader_warps
        self.threads = (self.compute_warp_base + _COMPUTE_WARPS) * 32
        self.load_threads = self.compute_warp_base * 32
        self.use_pdl = use_pdl
        self.smem_bytes = _smem_bytes(tactic)

    @cute.experimental.jit
    def __call__(
        self,
        weight,
        activation,
        weight_sf,
        activation_sf,
        output,
        stream: _cuda.CUstream,
    ):
        sA_layout = _make_smem_layout(
            weight.element_type, self.output_tile, self.k_tile16, self.stages
        )
        sB_layout = _make_smem_layout(
            activation.element_type, self.token_tile, self.k_tile16, self.stages
        )
        self.kernel(
            weight, activation, weight_sf, activation_sf, output, sA_layout, sB_layout
        ).launch(
            grid=(
                cute.ceil_div(weight.shape[0], self.output_tile),
                cute.ceil_div(activation.shape[0], self.token_tile),
                1,
            ),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,  # (N, K/2) 16-bit view of the E4M3 weight
        activation: cute.Tensor,  # (M, K/2) 16-bit view of the E4M3 activation
        weight_sf: cute.Tensor,  # 1D uint8, 128x4 swizzled E8M0 scales of the weight
        activation_sf: cute.Tensor,  # 1D uint8, same layout for the activation
        output: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, token_tile_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        reuse_stages = cute.size(weight, mode=[1]) > self.stages * self.k_tile16

        sA = cute_ext.allocate(
            weight.element_type, cute.AddressSpace.smem, sA_layout, alignment=1024
        )
        sB = cute_ext.allocate(
            activation.element_type, cute.AddressSpace.smem, sB_layout, alignment=1024
        )
        a_sf_words = self.output_tile * self.sf_groups
        b_sf_words = self.token_tile * self.sf_groups
        sSFA = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages * a_sf_words),
            alignment=16,
        )
        sSFB = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages * b_sf_words),
            alignment=16,
        )
        values_per_lane = self.output_tile * self.token_tile // cute.arch.WARP_SIZE
        partial_stride = (
            self.output_tile * self.token_tile // _MAILBOX_GROUP_VALUES
        ) * _MAILBOX_PADDED_GROUP_VALUES
        partials = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_COMPUTE_WARPS * partial_stride),
            alignment=16,
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator

        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile16
        # 4-byte scale words per row of the swizzled layout (K / 128).
        sf_k_words = cute.size(weight, mode=[1]) * 2 // _SF_WORD_K

        if warp_idx < _A_LOADER_WARPS:
            # Tile 0 has no mbarrier dependency: issue it before the init handshake
            # so its DRAM latency overlaps init and the CTA barrier.
            _issue_weight_tile(
                sA,
                weight,
                tidx,
                output_tile_idx,
                0,
                0,
                cutlass.Int64(0),
                self.output_tile,
                self.k_tile16,
                False,
            )
            for sf_iter in cutlass.range_constexpr(-(-a_sf_words // _A_LOADER_THREADS)):
                chunk = tidx + sf_iter * _A_LOADER_THREADS
                if chunk < a_sf_words:
                    row = output_tile_idx * self.output_tile + chunk // self.sf_groups
                    word = chunk % self.sf_groups
                    prims.cp_async_shared_global(
                        sSFA.iterator.raw_ptr() + chunk,
                        weight_sf.iterator.raw_ptr()
                        + _sf_byte_offset(row, word, sf_k_words),
                        4,
                        "ca",
                    )
        if warp_idx == self.compute_warp_base:
            if lane_idx < self.stages:
                cute.arch.mbarrier_init(bar_full + lane_idx, self.load_threads)
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_init(bar_empty + lane_idx, _COMPUTE_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx < _A_LOADER_WARPS:
            # Weights stream through L2 evict_first so the activation rows every
            # CTA re-reads stay resident. Only tile 0's arrive may follow the init.
            weight_policy = _l2_policy_evict_first()
            empty_phase = cutlass.Int32(1)
            prims.cp_async_mbarrier_arrive(bar_full.llvm_ptr, noinc=True)
            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                _issue_weight_tile(
                    sA,
                    weight,
                    tidx,
                    output_tile_idx,
                    k_tile,
                    stage,
                    weight_policy,
                    self.output_tile,
                    self.k_tile16,
                    True,
                )
                for sf_iter in cutlass.range_constexpr(
                    -(-a_sf_words // _A_LOADER_THREADS)
                ):
                    chunk = tidx + sf_iter * _A_LOADER_THREADS
                    if chunk < a_sf_words:
                        row = (
                            output_tile_idx * self.output_tile + chunk // self.sf_groups
                        )
                        word = k_tile * self.sf_groups + chunk % self.sf_groups
                        prims.cp_async_shared_global(
                            sSFA.iterator.raw_ptr() + stage * a_sf_words + chunk,
                            weight_sf.iterator.raw_ptr()
                            + _sf_byte_offset(row, word, sf_k_words),
                            4,
                            "ca",
                        )
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx < self.compute_warp_base:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            b_loader_threads = self.b_loader_warps * 32
            local_tid = tidx - _A_LOADER_WARPS * 32
            copy_count = self.token_tile * self.k_tile16 // (8 * b_loader_threads)
            empty_phase = cutlass.Int32(1)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                for copy_idx in cutlass.range_constexpr(copy_count):
                    linear = local_tid * 8 + copy_idx * b_loader_threads * 8
                    row = linear // self.k_tile16
                    col = linear % self.k_tile16
                    global_row = token_tile_idx * self.token_tile + row
                    # Out-of-range token rows are discarded by the output predicate.
                    if global_row < cute.size(activation, mode=[0]):
                        dst = (
                            sB.iterator.raw_ptr()
                            + stage * self.token_tile * self.k_tile16
                            + row * self.k_tile16
                            + (col ^ ((row % 8) * 8))
                        )
                        src = activation.iterator.raw_ptr() + activation.layout(
                            (global_row, k_tile * self.k_tile16 + col)
                        )
                        prims.cp_async_shared_global(dst, src, 16, "cg")
                for sf_iter in cutlass.range_constexpr(
                    -(-b_sf_words // b_loader_threads)
                ):
                    chunk = local_tid + sf_iter * b_loader_threads
                    if chunk < b_sf_words:
                        global_row = (
                            token_tile_idx * self.token_tile + chunk // self.sf_groups
                        )
                        if global_row < cute.size(activation, mode=[0]):
                            word = k_tile * self.sf_groups + chunk % self.sf_groups
                            prims.cp_async_shared_global(
                                sSFB.iterator.raw_ptr() + stage * b_sf_words + chunk,
                                activation_sf.iterator.raw_ptr()
                                + _sf_byte_offset(global_row, word, sf_k_words),
                                4,
                                "ca",
                            )
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
            if const_expr(self.use_pdl):
                # This grid-level hint is idempotent across B-loader warps.
                cute.arch.griddepcontrol_launch_dependents()
        else:
            compute_warp = warp_idx - self.compute_warp_base
            acc = cute.make_rmem_tensor(
                cute.make_layout(values_per_lane), cutlass.Float32
            )
            acc.fill(0.0)
            k_phases = self.k_tile // (_COMPUTE_WARPS * _MMA_SHAPE[2])
            m_iters = self.output_tile // _MMA_SHAPE[0]
            n_iters = self.token_tile // _MMA_SHAPE[1]
            a_offsets = tuple(
                tuple(
                    _a_fragment_offset(
                        lane_idx, compute_warp, m_iter, phase, self.k_tile16
                    )
                    for phase in range(k_phases)
                )
                for m_iter in range(m_iters)
            )
            b_offsets = tuple(
                tuple(
                    _b_pair_offset(
                        lane_idx,
                        compute_warp,
                        n_iter,
                        pair,
                        self.token_tile,
                        self.k_tile16,
                    )
                    for pair in range(k_phases // 2)
                )
                for n_iter in range(n_iters)
            )
            # K block ``kb`` (within the tile) of this warp: its scale lives in
            # byte ``kb % 4`` of word ``kb // 4``. ``k_phases`` (2 or 4) divides 4,
            # so the word is the same for every phase.
            kb_base = compute_warp * k_phases
            sf_word = kb_base // 4
            # Row of the accumulator fragment this lane owns, per half (c0/c1, c2/c3).
            w_row = tuple(
                tuple(lane_idx // 4 + 8 * half + m_iter * 16 for half in range(2))
                for m_iter in range(m_iters)
            )
            x_row = tuple(
                tuple(8 * n_iter + 2 * (lane_idx % 4) + j for j in range(2))
                for n_iter in range(n_iters)
            )
            a_stage_elems = self.output_tile * self.k_tile16
            b_stage_elems = self.token_tile * self.k_tile16
            full_phase = cutlass.Int32(0)
            # The K-tile count is static: unrolling fully folds `stage`, barrier
            # addresses and phases to constants (see the BF16 kernel).
            for k_tile in cutlass.range(k_tile_count, unroll_full=True):
                stage = k_tile % self.stages
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                a_base = stage * a_stage_elems
                b_base = stage * b_stage_elems
                a_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase])
                        for phase in range(k_phases)
                    )
                    for m_iter in range(m_iters)
                )
                b_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair])
                        for pair in range(k_phases // 2)
                    )
                    for n_iter in range(n_iters)
                )
                # This warp's first K block is byte 0 of the word for 512-element tiles
                # and byte 0 or 2 for 256-element ones; shift once per stage so every
                # phase below extracts a compile-time byte.
                if const_expr(k_phases == 4):
                    word_shift = 0
                else:
                    word_shift = (kb_base % 4) * 8
                w_words = tuple(
                    tuple(
                        cutlass.Int32(
                            sSFA[stage * a_sf_words + row * self.sf_groups + sf_word]
                        )
                        >> word_shift
                        for row in w_row[m_iter]
                    )
                    for m_iter in range(m_iters)
                )
                x_words = tuple(
                    tuple(
                        cutlass.Int32(
                            sSFB[stage * b_sf_words + t_row * self.sf_groups + sf_word]
                        )
                        >> word_shift
                        for t_row in x_row[n_iter]
                    )
                    for n_iter in range(n_iters)
                )
                for phase in cutlass.range_constexpr(k_phases):
                    w_scale = tuple(
                        tuple(
                            _e8m0_byte_to_f32(word, phase) for word in w_words[m_iter]
                        )
                        for m_iter in range(m_iters)
                    )
                    x_scale = tuple(
                        tuple(
                            _e8m0_byte_to_f32(word, phase) for word in x_words[n_iter]
                        )
                        for n_iter in range(n_iters)
                    )
                    # Unpack each fragment once per phase: a B fragment serves every
                    # m_iter and an A fragment every n_iter.
                    a_unpacked = tuple(
                        tuple(
                            _unpack_e4m3x4(a_registers[m_iter][phase][i])
                            for i in range(4)
                        )
                        for m_iter in range(m_iters)
                    )
                    b_unpacked = tuple(
                        tuple(
                            _unpack_e4m3x4(
                                b_registers[n_iter][phase // 2][(phase % 2) * 2 + i]
                            )
                            for i in range(2)
                        )
                        for n_iter in range(n_iters)
                    )
                    for n_iter in cutlass.range_constexpr(n_iters):
                        for m_iter in cutlass.range_constexpr(m_iters):
                            acc_base = (m_iter * n_iters + n_iter) * 4
                            block_acc = _mma_k32_from_f16(
                                a_unpacked[m_iter], b_unpacked[n_iter]
                            )
                            for i in cutlass.range_constexpr(4):
                                scale = w_scale[m_iter][i // 2] * x_scale[n_iter][i % 2]
                                acc[acc_base + i] = _fma_f32(
                                    block_acc[i], scale, acc[acc_base + i]
                                )
                if const_expr(reuse_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_empty + stage)
                if stage == self.stages - 1:
                    full_phase = full_phase ^ 1
            for value in cutlass.range_constexpr(values_per_lane):
                atom = value % 4
                fragment = value // 4
                m_iter = fragment // n_iters
                n_iter = fragment % n_iters
                feature = lane_idx // 4 + 8 * (atom // 2) + m_iter * _MMA_SHAPE[0]
                token = 8 * n_iter + 2 * (lane_idx % 4) + atom % 2
                partials[
                    compute_warp * partial_stride
                    + _partial_offset(feature, token, self.token_tile)
                ] = acc[value]
            prims.barrier_cta_sync(1, thread_count=_COMPUTE_WARPS * 32)

        if warp_idx == self.compute_warp_base:
            final_acc = cute.make_rmem_tensor(
                cute.make_layout(values_per_lane), cutlass.Float32
            )
            for value in cutlass.range_constexpr(values_per_lane):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                total = cutlass.Float32(0)
                for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                    total = (
                        total
                        + partials[
                            peer * partial_stride
                            + _partial_offset(feature, token, self.token_tile)
                        ]
                    )
                final_acc[value] = total

            output_base = output_tile_idx * self.output_tile
            token_base = token_tile_idx * self.token_tile
            for value in cutlass.range_constexpr(values_per_lane):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                if token_base + token < cute.size(output, mode=[0]):
                    output[token_base + token, output_base + feature] = final_acc[
                        value
                    ].to(output.element_type)


def _sf_byte_offset(row, word, sf_k_words):
    """Byte offset of the 4-byte scale word ``word`` of ``row`` in the 128x4 swizzle.

    The swizzled layout stores a 128-row x 4-block tile as 512 contiguous bytes:
    row ``r`` of the tile sits at ``(r % 32) * 16 + (r // 32) * 4``, its four K
    blocks in consecutive bytes. Tiles are ordered row-tile major, then K word.
    """
    return (
        (row // 128) * (sf_k_words * 512)
        + word * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
    )


def _from_dlpack(tensor: _torch.Tensor, *, dynamic_m: bool = False):
    tensor = from_dlpack(tensor.detach(), assumed_align=32)
    if dynamic_m:
        # Only M varies; the row stride and N/K extents stay static.
        tensor = tensor.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
    return tensor


def _from_dlpack_1d(tensor: _torch.Tensor, *, dynamic: bool):
    tensor = from_dlpack(tensor.detach(), assumed_align=4)
    if dynamic:
        tensor = tensor.mark_compact_shape_dynamic(
            mode=0, stride_order=(0,), divisibility=1
        )
    return tensor


@functools.cache
def _compile(
    device_index: int,
    out_dtype,
    n: int,
    k: int,
    tactic: MxFp8WarpSplitKTactic,
    use_pdl: bool,
):
    # M is a runtime value: only the tactic, N, K, and dtypes specialize the kernel
    # (the activation rows and its scale length are dynamic below).
    with _torch.cuda.device(device_index):
        weight_sf_len = _swizzled_scale_len(n, k)
        act_sf_len = _swizzled_scale_len(_MAX_M, k)
        return cute_ext.compile(
            CpAsyncMxFp8WarpSplitKKernel(tactic, use_pdl),
            _from_dlpack(_torch.empty((n, k // 2), device="cuda", dtype=_torch.int16)),
            _from_dlpack(
                _torch.empty((_MAX_M, k // 2), device="cuda", dtype=_torch.int16),
                dynamic_m=True,
            ),
            _from_dlpack_1d(
                _torch.empty(weight_sf_len, device="cuda", dtype=_torch.uint8),
                dynamic=False,
            ),
            _from_dlpack_1d(
                _torch.empty(act_sf_len, device="cuda", dtype=_torch.uint8),
                dynamic=True,
            ),
            _from_dlpack(
                _torch.empty((_MAX_M, n), device="cuda", dtype=out_dtype),
                dynamic_m=True,
            ),
            _cuda.CUstream(_torch.cuda.current_stream().cuda_stream),
        )


def run_mxfp8_warp_splitk(
    a, b, a_descale, b_descale, out, pdl: bool, tactic: MxFp8WarpSplitKTactic
):
    m, n, k = validate_inputs(a, b, a_descale, b_descale, out)
    validate_tactic(tactic, m, n, k)
    device_index = a.device.index
    assert device_index is not None
    with _torch.cuda.device(a.device):
        compiled = _compile(device_index, out.dtype, n, k, tactic, pdl)
        # Two E4M3 bytes read as one 16-bit element; the kernel only moves them.
        compiled(
            _from_dlpack(b.T.view(_torch.int16)),
            _from_dlpack(a.view(_torch.int16), dynamic_m=True),
            _from_dlpack_1d(b_descale, dynamic=False),
            _from_dlpack_1d(a_descale, dynamic=True),
            _from_dlpack(out, dynamic_m=True),
            _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream),
        )
    return out


__all__ = [
    "MxFp8WarpSplitKTactic",
    "autotune_tactics",
    "default_tactic",
    "run_mxfp8_warp_splitk",
    "validate_inputs",
    "validate_tactic",
]
