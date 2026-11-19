# Copyright (c) 2026 by FlashInfer team.
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

"""Per-token-group FP8 (e4m3) quantization, group_size=128, row-major scales —
CuTe-DSL fast path for sm90.

This kernel implements exactly the default form of
:func:`flashinfer.quantization.per_token_group_quant_8bit`:

  per group of 128 consecutive elements along the last (contiguous) dimension,
  ``scale = max(absmax, eps) / 448``; ``q = clamp(x / scale, -448, 448)`` cast
  to ``float8_e4m3fn``; scales are fp32, row-major ``(*x.shape[:-1], H // 128)``.

Because the input is contiguous and the last dimension is a multiple of 128,
the whole problem is a flat 1-D stream of 128-element groups: group ``g``
covers ``x.view(-1)[128 g : 128 g + 128)`` and writes ``s.view(-1)[g]``. The
kernel is a pure HBM stream (2 B read + 1 B write per element + 4 B per
group): each group is owned by 8 or 16 lanes of a warp, which read the group
as 16 B ``ld.global.nc.v4`` vectors, reduce the bf16 absmax with packed
``max.bf16x2`` and a butterfly shuffle, and write the quantized group back as
8 B stores.

Requires the optional ``nvidia-cutlass-dsl`` runtime (``import cutlass``);
callers must gate on
:func:`flashinfer.cute_dsl.availability.is_cute_dsl_available`. No other
compiler is needed at runtime (no ``tileiras``, no ``nvcc``).

Compilation is JIT on first use, cached in-process per
``(lanes_per_group, threads, device_index)`` variant (two variants per
device; shapes and layouts are dynamic). The launch runs with the input's
device current and is stream-ordered on that device's current CUDA stream
via the TVM-FFI environment stream; nothing else touches process-wide
device state. Under CUDA-graph capture a not-yet-compiled variant raises
instead of JIT-compiling; warm the op up once before capture (the public
``per_token_group_quant_8bit`` dispatcher turns that raise into a stock
``cutile`` fallback with reason ``capturing_before_jit`` for
``backend="auto"`` and propagates it as ``NotImplementedError`` for
``backend="specialized"``).
"""

from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack

from . import _ColdJITDuringCapture


@dsl_user_op
def _ptrtoint(ptr, *, loc=None, ip=None) -> Int64:
    return Int64(llvm.ptrtoint(T.i64(), ptr.llvm_ptr, loc=loc, ip=ip))


def _base_addr(ptr) -> Int64:
    try:
        return _ptrtoint(ptr)
    except Exception:
        return Int64(ptr.toint())


@dsl_user_op
def _ld128(addr: Int64, *, loc=None, ip=None):
    words = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), words, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), words, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), words, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), words, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _absmax8(
    a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(c).ir_value(loc=loc, ip=ip),
                Uint32(d).ir_value(loc=loc, ip=ip),
            ],
            """
            {
            .reg .b32 x0, x1, x2, x3;
            and.b32 x0, $1, 0x7fff7fff;
            and.b32 x1, $2, 0x7fff7fff;
            and.b32 x2, $3, 0x7fff7fff;
            and.b32 x3, $4, 0x7fff7fff;
            max.bf16x2 x0, x0, x1;
            max.bf16x2 x2, x2, x3;
            max.bf16x2 $0, x0, x2;
            }
            """,
            "=r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_bf16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _hmax_bf16x2(a: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            """
            {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            shl.b32 lo, $1, 16;
            and.b32 hi, $1, 0xffff0000;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmax(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rcp(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _quant8(
    a: Uint32, b: Uint32, c: Uint32, d: Uint32, inv: Float32, *, loc=None, ip=None
) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(c).ir_value(loc=loc, ip=ip),
                Uint32(d).ir_value(loc=loc, ip=ip),
                Float32(inv).ir_value(loc=loc, ip=ip),
            ],
            """
            {
            .reg .b32 lo, hi, r0, r1;
            .reg .f32 f0, f1;
            .reg .b16 h0, h1, h2, h3;
            shl.b32 lo, $1, 16; and.b32 hi, $1, 0xffff0000;
            mov.b32 f0, lo; mov.b32 f1, hi;
            mul.f32 f0, f0, $5; mul.f32 f1, f1, $5;
            cvt.rn.satfinite.e4m3x2.f32 h0, f1, f0;
            shl.b32 lo, $2, 16; and.b32 hi, $2, 0xffff0000;
            mov.b32 f0, lo; mov.b32 f1, hi;
            mul.f32 f0, f0, $5; mul.f32 f1, f1, $5;
            cvt.rn.satfinite.e4m3x2.f32 h1, f1, f0;
            shl.b32 lo, $3, 16; and.b32 hi, $3, 0xffff0000;
            mov.b32 f0, lo; mov.b32 f1, hi;
            mul.f32 f0, f0, $5; mul.f32 f1, f1, $5;
            cvt.rn.satfinite.e4m3x2.f32 h2, f1, f0;
            shl.b32 lo, $4, 16; and.b32 hi, $4, 0xffff0000;
            mov.b32 f0, lo; mov.b32 f1, hi;
            mul.f32 f0, f0, $5; mul.f32 f1, f1, $5;
            cvt.rn.satfinite.e4m3x2.f32 h3, f1, f0;
            mov.b32 r0, {h0, h1}; mov.b32 r1, {h2, h3};
            mov.b64 $0, {r0, r1};
            }
            """,
            "=l,r,r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st64(addr: Int64, value: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint64(value).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_scale(addr: Int64, value: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _build(lanes_per_group, threads):
    groups_per_block = threads // lanes_per_group
    vectors_per_lane = 128 // (lanes_per_group * 8)

    @cute.kernel
    def kernel(
        x: cute.Tensor, q: cute.Tensor, s: cute.Tensor, num_groups: Int32, eps: Float32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % lanes_per_group
        group_in_block = tidx // lanes_per_group
        group = bidx * groups_per_block + group_in_block
        valid = group < num_groups
        safe_group = group if valid else num_groups - 1

        x_addr = _base_addr(x.iterator) + Int64(safe_group) * 256 + Int64(lane) * 16
        values = []
        for i in cutlass.range_constexpr(vectors_per_lane):
            values.append(_ld128(x_addr + Int64(i * lanes_per_group * 16)))

        packed_max = _absmax8(values[0][0], values[0][1], values[0][2], values[0][3])
        for i in cutlass.range_constexpr(1, vectors_per_lane):
            v = values[i]
            packed_max = _max_bf16x2(packed_max, _absmax8(v[0], v[1], v[2], v[3]))
        offset = 1
        while offset < lanes_per_group:
            packed_max = _max_bf16x2(
                packed_max, cute.arch.shuffle_sync_bfly(packed_max, offset=offset)
            )
            offset *= 2

        amax = _hmax_bf16x2(packed_max)
        scale = _div_rn(_fmax(amax, eps), Float32(448.0))
        inv_scale = _rcp(scale)

        packed_q = []
        for i in cutlass.range_constexpr(vectors_per_lane):
            v = values[i]
            packed_q.append(_quant8(v[0], v[1], v[2], v[3], inv_scale))

        if valid:
            q_addr = _base_addr(q.iterator) + Int64(group) * 128 + Int64(lane) * 8
            for i in cutlass.range_constexpr(vectors_per_lane):
                _st64(q_addr + Int64(i * lanes_per_group * 8), packed_q[i])
            if lane == 0:
                _st_scale(_base_addr(s.iterator) + Int64(group) * 4, scale)

    @cute.jit
    def launch(
        x: cute.Tensor,
        q: cute.Tensor,
        s: cute.Tensor,
        num_groups: Int32,
        eps: Float32,
        stream,
    ):
        kernel(x, q, s, num_groups, eps).launch(
            grid=[cute.ceil_div(num_groups, groups_per_block), 1, 1],
            block=[threads, 1, 1],
            stream=stream,
        )

    return launch


_CACHE: dict = {}


def _compiled(lanes_per_group, threads, x, q_bytes, s):
    # Keyed by device index as well: the compiled function binds the module
    # for the device it was compiled on, so a second GPU gets its own entry.
    key = (lanes_per_group, threads, x.device.index)
    fn = _CACHE.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise _ColdJITDuringCapture(
                "Warm up per_token_group_quant_8bit (CuTe-DSL sm90 fast path) "
                "before CUDA graph capture: the kernel variant for this problem "
                "size is not compiled yet and JIT compilation is not allowed "
                "under capture. Call the op once eagerly on a representative "
                "shape first."
            )
        # The contract guarantees contiguous row-major tensors, so the
        # leading (stride-1) dimension is always the last one. Passing it
        # explicitly matters when the compile-triggering call has size-1
        # dimensions (e.g. a (1, 128) input, whose (1, 1) scale tensor gives
        # mark_layout_dynamic no stride to deduce the leading dim from).
        x_view = from_dlpack(x, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=x.dim() - 1
        )
        q_view = from_dlpack(q_bytes, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=q_bytes.dim() - 1
        )
        s_view = from_dlpack(s, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=s.dim() - 1
        )
        fn = cute.compile(
            _build(lanes_per_group, threads),
            x_view,
            q_view,
            s_view,
            Int32(0),
            Float32(0.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    return fn


@torch.no_grad()
def per_token_group_quant_fp8_g128_cute_dsl(
    x: torch.Tensor,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` per 128-element group with row-major fp32 scales.

    ``x`` must be a contiguous CUDA bf16 tensor whose last dimension is a
    multiple of 128 and whose data pointer is 16-byte aligned (the kernel
    reads it with 16-byte vectorized loads), on an sm90 device. Returns ``(x_q, x_s)`` with
    ``x_q = float8_e4m3fn`` of ``x.shape`` and ``x_s = float32`` of shape
    ``(*x.shape[:-1], x.shape[-1] // 128)``. Callers are expected to have
    checked the contract via
    :func:`flashinfer.quantization.fp8_quantization.use_per_token_group_quant_specialized`.
    """
    x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // 128,), device=x.device, dtype=torch.float32
    )
    num_groups = x_s.numel()
    lanes_per_group = 8 if num_groups >= 16384 else 16
    threads = 256
    q_bytes = x_q.view(torch.uint8)
    # Make the input's device current around compile and launch so the
    # module and the TVM-FFI environment stream match the tensor's device.
    with torch.cuda.device(x.device):
        _compiled(lanes_per_group, threads, x, q_bytes, x_s)(
            x, q_bytes, x_s, num_groups, float(eps)
        )
    return x_q, x_s
