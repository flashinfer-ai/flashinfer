# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""One-launch staging of a batch into the NVLinkOneSided MegaMoE workspace (CuTe DSL).

One kernel writes everything the MegaMoE launch reads from the staging planes:

- **Activations.** One lane owns one 32-value block of a token row. From BF16
  it quantizes the block to MXFP8 with the numerics of
  ``quantize_mxfp8_block32``: the UE8M0 scale is ``2**ceil(log2(amax *
  fl32(1/448)))``, clamped to normal FP32 exponents, with a block that holds a
  NaN or an infinity scaled by ``2**-126``; the values are scaled exactly by
  the power of two and rounded to E4M3 with saturation. Pre-quantized E4M3
  rows and their UE8M0 scales are copied. Either way the sentinel rewrites of
  ``staging.py`` apply: no staged data byte or scale byte is ``0xFF``.
- **Shared expert scales.** With a fused shared expert each staged scale also
  goes to the shared expert's scale plane, at the 32x4x4 atom swizzle of its
  slot row (``BlockScaledSymmBuffer.shared_expert_scale_layout``), for the
  live rows only: the shared expert's outputs past the token count are never
  read.
- **Routing.** Top-k ids are narrowed to the workspace's int32 and the FP32
  routing weights copied; every row from the token count to the capacity gets
  ``-1`` ids, so rows a larger earlier batch left behind dispatch nothing.

The kernel takes raw device addresses; ``stage`` checks the layouts it relies
on and returns False for the others, which keep the Torch staging. Compiled
stagers are cached by their static shape; compilation cannot run during CUDA
graph capture, so a stager must first run eagerly (the layer's warmup does).
"""

from functools import cache
from typing import Dict, Optional, Tuple

import torch

_SF_VEC = 32
# Lanes per CTA. Lane i of the grid stages 32-value block i of the batch and
# routing entry i of the workspace.
_THREADS = 128
# fl32(1 / 448): the factor quantize_mxfp8_block32 applies to the block amax.
_RCP_E4M3_MAX_BITS = 0x3B124925
# BF16 magnitudes at or above this are infinite or NaN.
_BF16_INF_BITS = 0x7F80


def _rewrite_ff(word: str, scratch: Tuple[str, str]) -> list:
    """PTX that clears bit 7 of every byte of ``word`` that equals 0xFF (0xFF -> 0x7F)."""
    z, m = scratch
    return [
        f"not.b32 {z}, {word};",
        # A byte of (z & 0x7F) + 0x7F has bit 7 set iff its low bits are not
        # all zero, and no sum carries into the next byte (at most 0xFE).
        f"and.b32 {m}, {z}, 0x7F7F7F7F;",
        f"add.u32 {m}, {m}, 0x7F7F7F7F;",
        f"or.b32 {m}, {m}, {z};",
        f"or.b32 {m}, {m}, 0x7F7F7F7F;",
        # m: 0x80 in the bytes where z is 0 (the word's byte is 0xFF), else 0.
        f"not.b32 {m}, {m};",
        f"xor.b32 {word}, {word}, {m};",
    ]


def _quantize_asm() -> str:
    """$0..$7: E4M3 words of values 0..31; $8: UE8M0 code; $9..$24: BF16 pairs of the block."""
    words = [f"${9 + i}" for i in range(16)]
    lines = [
        "{",
        ".reg .b32 m, t, z, c, s;",
        ".reg .pred special, low, frac;",
        ".reg .f32 a, p, inv, v<32>;",
        ".reg .b16 q<16>;",
        # Block amax as BF16 magnitude bits; integer order is float order for
        # magnitudes, with every NaN above infinity.
        f"and.b32 m, {words[0]}, 0x7FFF7FFF;",
    ]
    for w in words[1:]:
        lines += [f"and.b32 t, {w}, 0x7FFF7FFF;", "max.u16x2 m, m, t;"]
    lines += [
        "shr.u32 t, m, 16;",
        "and.b32 m, m, 0xFFFF;",
        "max.u32 m, m, t;",
        f"setp.ge.u32 special, m, {_BF16_INF_BITS};",
        # p = amax * fl32(1/448), rounded once; its exponent rounded up is the
        # scale's, with p below the smallest normal (and a non-finite block)
        # clamped to 2**-126 (code 1).
        "shl.b32 t, m, 16;",
        "mov.b32 a, t;",
        f"mul.rn.f32 p, a, 0f{_RCP_E4M3_MAX_BITS:08X};",
        "mov.b32 t, p;",
        "shr.u32 c, t, 23;",
        "and.b32 s, t, 0x7FFFFF;",
        "setp.ne.u32 frac, s, 0;",
        "@frac add.u32 c, c, 1;",
        "setp.lt.u32 low, t, 0x00800000;",
        "or.pred low, low, special;",
        "@low mov.u32 c, 1;",
        "min.u32 c, c, 254;",
        "mov.u32 $8, c;",
        # 2**-e for the scale 2**e (a BF16 amax gives e <= 120, so 2**-e is a
        # normal FP32): values scale exactly unless they leave the FP32 range.
        "sub.u32 s, 254, c;",
        "shl.b32 s, s, 23;",
        "mov.b32 inv, s;",
    ]
    # The multiplies keep subnormals (no .ftz): a block of subnormal BF16
    # values scales by 2**126 into the E4M3 range.
    for i, w in enumerate(words):
        lines += [
            f"shl.b32 t, {w}, 16;",
            f"mov.b32 v{2 * i}, t;",
            f"and.b32 t, {w}, 0xFFFF0000;",
            f"mov.b32 v{2 * i + 1}, t;",
            f"mul.rn.f32 v{2 * i}, v{2 * i}, inv;",
            f"mul.rn.f32 v{2 * i + 1}, v{2 * i + 1}, inv;",
            f"cvt.rn.satfinite.e4m3x2.f32 q{i}, v{2 * i + 1}, v{2 * i};",
        ]
    for o in range(8):
        lines.append(f"mov.b32 ${o}, {{q{2 * o}, q{2 * o + 1}}};")
    # Only a block with a NaN can produce an E4M3 NaN byte.
    lines.append("@!special bra DONE;")
    for o in range(8):
        lines += _rewrite_ff(f"${o}", ("z", "t"))
    lines += ["DONE:", "}"]
    return "\n".join(lines)


def _copy_asm() -> str:
    """$0..$7: staged E4M3 words; $8: staged UE8M0 code; $9..$16: E4M3 words; $17: UE8M0 code."""
    lines = [
        "{",
        ".reg .b32 z, t;",
        ".reg .pred ff;",
    ]
    for o in range(8):
        lines.append(f"mov.b32 ${o}, ${9 + o};")
        lines += _rewrite_ff(f"${o}", ("z", "t"))
    lines += [
        "setp.eq.u32 ff, $17, 0xFF;",
        "selp.u32 $8, 0xFE, $17, ff;",
        "}",
    ]
    return "\n".join(lines)


@cache
def _kernel_class():
    """Build the CuTe DSL stager lazily, keeping CUDA imports out of discovery."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

    def _asm(results: int, operands, text: str, constraints: str, *, loc, ip):
        """Inline PTX with ``results`` 32-bit outputs; one without outputs is a store."""
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * results)
            if results > 1
            else (T.i32() if results else None),
            operands,
            text,
            constraints,
            has_side_effects=results == 0,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        if results == 1:
            return Int32(out)
        return tuple(
            Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip))
            for i in range(results)
        )

    def _ir(values, kind, *, loc, ip):
        return [kind(v).ir_value(loc=loc, ip=ip) for v in values]

    @dsl_user_op
    def load_v8(address: Int64, *, loc=None, ip=None):
        """32 bytes (one sector) of a read-only input, streamed past L1."""
        return _asm(
            8,
            _ir([address], Int64, loc=loc, ip=ip),
            "ld.global.nc.L1::no_allocate.L2::256B.v8.b32 "
            "{$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
            "=r,=r,=r,=r,=r,=r,=r,=r,l",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def load_u8(address: Int64, *, loc=None, ip=None):
        return _asm(
            1,
            _ir([address], Int64, loc=loc, ip=ip),
            "{\n.reg .b16 v;\nld.global.nc.u8 v, [$1];\ncvt.u32.u16 $0, v;\n}",
            "=r,l",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def load_b32(address: Int64, *, loc=None, ip=None):
        return _asm(
            1,
            _ir([address], Int64, loc=loc, ip=ip),
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def store_v8(address: Int64, words, *, loc=None, ip=None):
        _asm(
            0,
            _ir([address], Int64, loc=loc, ip=ip) + _ir(words, Int32, loc=loc, ip=ip),
            "st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
            "l,r,r,r,r,r,r,r,r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def store_u8(address: Int64, value: Int32, *, loc=None, ip=None):
        _asm(
            0,
            _ir([address], Int64, loc=loc, ip=ip) + _ir([value], Int32, loc=loc, ip=ip),
            "{\n.reg .b16 v;\ncvt.u16.u32 v, $1;\nst.global.u8 [$0], v;\n}",
            "l,r",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def store_b32(address: Int64, value: Int32, *, loc=None, ip=None):
        _asm(
            0,
            _ir([address], Int64, loc=loc, ip=ip) + _ir([value], Int32, loc=loc, ip=ip),
            "st.global.b32 [$0], $1;",
            "l,r",
            loc=loc,
            ip=ip,
        )

    quantize_text = _quantize_asm()
    copy_text = _copy_asm()

    @dsl_user_op
    def quantize_block(words, *, loc=None, ip=None):
        """Sixteen BF16 pairs -> (eight staged E4M3 words, staged UE8M0 code)."""
        out = _asm(
            9,
            _ir(words, Int32, loc=loc, ip=ip),
            quantize_text,
            ",".join(["=r"] * 9 + ["r"] * 16),
            loc=loc,
            ip=ip,
        )
        return out[:8], out[8]

    @dsl_user_op
    def copy_block(words, code: Int32, *, loc=None, ip=None):
        """Eight E4M3 words and a UE8M0 code -> their staged values."""
        out = _asm(
            9,
            _ir(list(words) + [code], Int32, loc=loc, ip=ip),
            copy_text,
            ",".join(["=r"] * 9 + ["r"] * 9),
            loc=loc,
            ip=ip,
        )
        return out[:8], out[8]

    class Stager:
        """Stager for one static shape; ``__call__`` launches on ``stream``."""

        def __init__(
            self,
            hidden: int,
            top_k: int,
            quantize: bool,
            wide_ids: bool,
            shared_tiles: Optional[Tuple[int, int]],
        ):
            self.hidden = hidden
            self.top_k = top_k
            self.quantize = quantize
            self.wide_ids = wide_ids
            self.shared_tiles = shared_tiles

        @cute.jit
        def __call__(
            self,
            x: Int64,
            x_stride: Int64,
            sf: Int64,
            sf_stride: Int64,
            ids: Int64,
            weights: Int64,
            x_out: Int64,
            sf_out: Int64,
            sf_out_stride: Int64,
            ids_out: Int64,
            weights_out: Int64,
            shared_sf_out: Int64,
            num_tokens: Int32,
            capacity: Int32,
            grid: Int32,
            stream: cuda.CUstream,
        ):
            """Launch ``grid`` CTAs, enough lanes for every block and routing entry."""
            self.kernel(
                x,
                x_stride,
                sf,
                sf_stride,
                ids,
                weights,
                x_out,
                sf_out,
                sf_out_stride,
                ids_out,
                weights_out,
                shared_sf_out,
                num_tokens,
                capacity,
            ).launch(grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream)

        @cute.kernel
        def kernel(
            self,
            x: Int64,
            x_stride: Int64,
            sf: Int64,
            sf_stride: Int64,
            ids: Int64,
            weights: Int64,
            x_out: Int64,
            sf_out: Int64,
            sf_out_stride: Int64,
            ids_out: Int64,
            weights_out: Int64,
            shared_sf_out: Int64,
            num_tokens: Int32,
            capacity: Int32,
        ):
            """Lane ``i`` stages block ``i`` (row-major) and routing entry ``i``."""
            hidden: cutlass.Constexpr[int] = self.hidden
            top_k: cutlass.Constexpr[int] = self.top_k
            row_blocks: cutlass.Constexpr[int] = hidden // _SF_VEC
            # Bytes of one block's input: BF16 values or E4M3 values.
            in_bytes: cutlass.Constexpr[int] = 64 if self.quantize else 32

            tid, _, _ = cute.arch.thread_idx()
            bid, _, _ = cute.arch.block_idx()
            lane = bid * Int32(_THREADS) + tid

            if lane < num_tokens * Int32(row_blocks):
                row = lane // Int32(row_blocks)
                col = lane % Int32(row_blocks)
                source = x + Int64(row) * x_stride + Int64(col) * Int64(in_bytes)
                if cutlass.const_expr(self.quantize):
                    data, code = quantize_block(
                        load_v8(source) + load_v8(source + Int64(32))
                    )
                else:
                    data, code = copy_block(
                        load_v8(source),
                        load_u8(sf + Int64(row) * sf_stride + Int64(col)),
                    )
                store_v8(
                    x_out + Int64(row) * Int64(hidden) + Int64(col) * Int64(_SF_VEC),
                    data,
                )
                store_u8(sf_out + Int64(row) * sf_out_stride + Int64(col), code)
                if cutlass.const_expr(self.shared_tiles is not None):
                    tile_rows, slot_rows = self.shared_tiles
                    slot = row // Int32(tile_rows) * Int32(slot_rows) + row % Int32(
                        tile_rows
                    )
                    offset = (
                        (
                            (slot // Int32(128)) * Int32(row_blocks // 4)
                            + col // Int32(4)
                        )
                        * Int32(512)
                        + (slot % Int32(32)) * Int32(16)
                        + (slot // Int32(32) % Int32(4)) * Int32(4)
                        + col % Int32(4)
                    )
                    store_u8(shared_sf_out + Int64(offset), code)

            if lane < capacity * Int32(top_k):
                route = Int32(-1)
                if lane < num_tokens * Int32(top_k):
                    if cutlass.const_expr(self.wide_ids):
                        # Narrowing keeps the low word, as a Torch int64 -> int32 copy.
                        route = load_b32(ids + Int64(lane) * Int64(8))
                    else:
                        route = load_b32(ids + Int64(lane) * Int64(4))
                    store_b32(
                        weights_out + Int64(lane) * Int64(4),
                        load_b32(weights + Int64(lane) * Int64(4)),
                    )
                store_b32(ids_out + Int64(lane) * Int64(4), route)

    return Stager


_COMPILED: Dict[tuple, object] = {}


def _compiled(key: tuple):
    """The compiled stager for ``key``; compiles on the first use outside capture."""
    compiled = _COMPILED.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "NVLinkOneSided MegaMoE staging compiles on its first launch, which "
            "cannot run during CUDA graph capture; stage a batch eagerly first "
            "(layer.warmup does)"
        )
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    address, extent, count = cutlass.Int64(0), cutlass.Int64(0), cutlass.Int32(0)
    compiled = cute.compile(
        _kernel_class()(*key),
        *([address, extent] * 2),
        address,
        address,
        address,
        address,
        extent,
        address,
        address,
        address,
        count,
        count,
        count,
        cuda.CUstream(0),
        options="--enable-tvm-ffi",
    )
    _COMPILED[key] = compiled
    return compiled


def _rows_aligned(tensor: torch.Tensor, align: int) -> bool:
    """Unit column stride, and every row starts on ``align`` bytes."""
    return (
        tensor.stride(-1) == 1
        and tensor.data_ptr() % align == 0
        and (tensor.stride(0) * tensor.element_size()) % align == 0
    )


def _extent(tensor: torch.Tensor) -> Tuple[int, int]:
    """Byte range of a 1D or 2D tensor with unit stride in its last dimension."""
    if tensor.ndim == 1:
        size = tensor.shape[0]
    else:
        rows, cols = tensor.shape
        size = (rows - 1) * tensor.stride(0) + cols if rows else 0
    return tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size()


def stage(
    hidden_states: torch.Tensor,
    scales: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    workspace,
    *,
    quantize: bool,
) -> bool:
    """Enqueue the staging of every workspace row on the current stream.

    ``quantize`` takes BF16 ``hidden_states``; otherwise E4M3 ``hidden_states``
    with UE8M0 ``scales``. The live rows get their data, scales and routing;
    the rows from the token count to the capacity get ``-1`` ids and keep
    everything else. Returns False, staging nothing, for layouts the kernel
    does not take: rows not 32-byte aligned or without unit column stride,
    non-contiguous routing or routing planes, or inputs that overlap the
    staging planes (the kernel reads and writes every row at once).
    """
    num_tokens, hidden = hidden_states.shape
    capacity, top_k = workspace.topk_idx.shape
    shared = workspace.config.shared_expert is not None
    lanes = max(num_tokens * (hidden // _SF_VEC), capacity * top_k, 1)
    # The shared expert's scale swizzle takes scale columns in groups of four.
    if (
        hidden != workspace.config.hidden
        or hidden % (4 * _SF_VEC if shared else _SF_VEC)
        or lanes >= 2**31 - _THREADS
    ):
        return False
    if not (
        all(
            t.is_contiguous()
            for t in (
                topk_ids,
                topk_weights,
                workspace.x,
                workspace.topk_idx,
                workspace.topk_weights,
            )
        )
        and workspace.topk_idx.dtype == torch.int32
        and workspace.topk_weights.dtype == torch.float32
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_weights.dtype == torch.float32
        and _rows_aligned(hidden_states, 32)
        and _rows_aligned(workspace.x, 32)
        and workspace.x_sf.stride(1) == 1
        and (quantize or scales.stride(1) == 1)
    ):
        return False
    planes = [workspace.x, workspace.x_sf, workspace.topk_idx, workspace.topk_weights]
    if shared:
        shared_sf, tile_rows, slot_rows = workspace.shared_expert_scale_layout()
        shared_tiles = (int(tile_rows), int(slot_rows))
        shared_address = shared_sf.data_ptr()
        planes.append(shared_sf)
    else:
        shared_tiles, shared_address = None, 0
    inputs = [hidden_states, topk_ids, topk_weights] + ([] if quantize else [scales])
    plane_extents = [_extent(plane) for plane in planes]
    if any(
        max(a[0], b[0]) < min(a[1], b[1])
        for a in map(_extent, inputs)
        for b in plane_extents
    ):
        return False

    key = (hidden, top_k, quantize, topk_ids.dtype == torch.int64, shared_tiles)
    compiled = _compiled(key)
    import cuda.bindings.driver as cuda

    compiled(
        hidden_states.data_ptr(),
        hidden_states.stride(0) * hidden_states.element_size(),
        0 if quantize else scales.data_ptr(),
        0 if quantize else scales.stride(0) * scales.element_size(),
        topk_ids.data_ptr(),
        topk_weights.data_ptr(),
        workspace.x.data_ptr(),
        workspace.x_sf.data_ptr(),
        workspace.x_sf.stride(0) * workspace.x_sf.element_size(),
        workspace.topk_idx.data_ptr(),
        workspace.topk_weights.data_ptr(),
        shared_address,
        num_tokens,
        capacity,
        -(-lanes // _THREADS),
        cuda.CUstream(torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())),
    )
    return True
