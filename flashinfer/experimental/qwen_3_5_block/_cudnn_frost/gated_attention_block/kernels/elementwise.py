# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-head elementwise pass over ``[T, H, D]``, with independent strides.

ONE kernel, two uses in the block, separated by a ``const_expr``:

* **stage (5), the sigmoid gate** — ``out = src * sigmoid(gate)``;
* **V compaction** — ``out = src`` with no gate operand at all, which the
  ``has_gate=False`` arm folds down to a strided copy.

The second one exists only because stage (1) is still the UNFORKED FROST GEMM,
which writes a single fused ``[M, N]``. Q and K get de-interleaved for free by
the norm+RoPE pass (it already reads and writes them), but V is untouched
between the projection and the SDPA, so it would reach the SDPA as a
padded-stride view. The historical tensor adapter copied such views; current
SDPA plans bind TMA-compatible strides directly and declare any conversion
scratch explicitly. A hidden copy inside the SDPA adapter is exactly what
AGENTS.md Rule 2 bans, so the block does it as a NAMED, MEASURED stage instead.
**It disappears the moment stage (1) is forked to write four compact buffers**
(``api.py`` § 1); until then it is the honest price of not having forked yet,
and V is 1/16 of Q here so the price is small.

Bandwidth-bound and shaped for it, same as ``qk_norm_rope``: one head row is
``D`` elements and every lane moves 16 bytes, so ``LANES = D // 8`` lanes cover
a row with one ``ld.global.v4`` each — at ``D = 256`` a full warp issuing one
perfectly-coalesced 512 B request. Unlike the norm, there is **no reduction and
no shuffle here**, so nothing serialises between the load and the store: this is
the closest thing in the block to a pure copy, and it should sit closest to the
HBM ceiling.

Traffic: ``2*R + W`` per element with a gate, ``R + W`` without.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, sigmoid
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global_v4, st_global_v4

from .qk_norm_rope import ACCESS_BYTES, ELEMS_PER_ACCESS, fake_rowmajor_dynamic_token_stride, lanes_per_row, vec_chunks

DEFAULT_THREADS_PER_CTA = 128
DEFAULT_ROWS_PER_GROUP = 2
# Bake the head count into the address math. Mirrors `DEFAULT_CONST_HEAD_COUNTS`
# in qk_norm_rope.py, including the reason it is a KNOB and not a hard-code: the
# runtime arm stays traced so the two can be A/B'd, and so one artifact can
# still serve any H if a caller ever wants that. `h` was ALREADY part of the
# compile-cache key, so turning it constant adds no artifact that did not exist.
DEFAULT_CONST_HEAD_COUNT = True
"""Same two occupancy knobs as ``qk_norm_rope``, same reason, and they must be
re-measured per arch — an A100-derived pair was wrong for Rubin there."""

_FAKE_STREAM = None
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with a compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def validate_shape(d: int, threads_per_cta: int) -> None:
    """Raise on any geometry this kernel cannot address."""
    if d % ELEMS_PER_ACCESS != 0:
        raise ValueError(f"d_head must be a multiple of {ELEMS_PER_ACCESS} for 128-bit accesses, got {d}")
    lanes = lanes_per_row(d)
    if 32 % lanes != 0:
        raise ValueError(f"d_head={d} gives {lanes} lanes/row, which must divide a warp")
    if threads_per_cta % lanes != 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a multiple of the {lanes} lanes per row")


@cute.kernel
def frost_elementwise_gate(
    mSrc: cute.Tensor,  # [T, H, D]
    mGate: Optional[cute.Tensor],  # [T, H, D] or None
    mDst: cute.Tensor,  # [T, H, D]
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """``dst = src * sigmoid(gate)``, or ``dst = src`` when ``mGate is None``.

    All three operands carry their OWN token/head strides, which is what lets
    the gate come straight out of the fused projection buffer while ``src`` and
    ``dst`` are compact — no repack, no third buffer.

    ``dst`` may alias ``src``: every lane reads its own elements before it
    writes them and no lane touches another's.
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    has_gate = cutlass.const_expr(mGate is not None)

    # `_h` is the head count the address math actually uses. Under
    # `const_head_count` it is a Python constant, so `row // _h` / `row % _h`
    # strength-reduce (at the block's h_q=32 and h_kv=2, both powers of two, to
    # a shift and a mask). As a runtime Int32 the pair is a SOFTWARE integer
    # divide -- this part has no integer-divide unit -- run `rows_per_group`
    # times by every thread, against a body whose real work is 4 LDG.128,
    # 2 STG.128 and 16 MUFU. That is why it is on by default here.
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)

    # PASS 1: issue every load first. With no reduction in the way this is pure
    # memory-level parallelism -- unlike the norm kernel, nothing here consumes
    # a value before the next load can issue.
    rows = []
    dsts = []
    srcs = []
    gates = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        token = row_r // _h
        head = row_r % _h
        src_addr = mSrc.iterator.toint() + (
            token.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mSrc.stride[1])
        ) * cutlass.Int64(2)
        dst_addr = mDst.iterator.toint() + (
            token.to(cutlass.Int64) * cutlass.Int64(mDst.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mDst.stride[1])
        ) * cutlass.Int64(2)
        row_src = []
        row_gate = []
        for c in cutlass.range_constexpr(chunks):
            off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
            row_src.append([v for pair in [f16x2_to_f32(w, dtype=mSrc.element_type) for w in ld_global_v4(src_addr + off, cutlass.Int32)] for v in pair])
        if cutlass.const_expr(has_gate):
            gate_addr = mGate.iterator.toint() + (
                token.to(cutlass.Int64) * cutlass.Int64(mGate.stride[0]) + head.to(cutlass.Int64) * cutlass.Int64(mGate.stride[1])
            ) * cutlass.Int64(2)
            for c in cutlass.range_constexpr(chunks):
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                row_gate.append([v for pair in [f16x2_to_f32(w, dtype=mGate.element_type) for w in ld_global_v4(gate_addr + off, cutlass.Int32)] for v in pair])
        rows.append(row)
        dsts.append(dst_addr)
        srcs.append(row_src)
        gates.append(row_gate)

    # PASS 2: sigmoid in fp32 (the tanh identity -- one MUFU per pair), multiply,
    # pack, store.
    for r in cutlass.range_constexpr(rows_per_group):
        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(chunks):
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                x = srcs[r][c]
                packed = []
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS // 2):
                    lo, hi = x[2 * i], x[2 * i + 1]
                    if cutlass.const_expr(has_gate):
                        g = gates[r][c]
                        # SCALAR sigmoid and multiply, not the packed f32x2
                        # helpers (`sigmoid2` / `fmul2`): those lower to
                        # `mul.f32x2` / `fma.f32x2`, which ptxas accepts only on
                        # sm_100+. This kernel is bandwidth-bound with ~57x MUFU
                        # headroom at Rubin's HBM rate, so the packed form buys
                        # nothing measurable -- and the scalar form keeps the
                        # kernel runnable (and debuggable) on an A100.
                        lo = lo * sigmoid(g[2 * i])
                        hi = hi * sigmoid(g[2 * i + 1])
                    packed.append(fp32_to_fp16(lo, hi, dtype=mDst.element_type))
                st_global_v4(dsts[r] + off, packed, cutlass.Int32)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def elementwise_gate_launch(
    src: cute.Tensor,
    gate: Optional[cute.Tensor],
    dst: cute.Tensor,
    n_rows: cutlass.Int32,
    h: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_elementwise_gate(src, gate, dst, n_rows, h, h_ct, const_head_count, d, threads_per_cta, rows_per_group, use_pdl).launch(
        grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl
    )


compiled_cache = {}


class ElementwiseRecipe(NamedTuple):
    """Build-time facts of one elementwise launch. Every field is plan-time
    derivable; the token count rides in as a runtime ``Int32`` (Rule 4)."""

    compiled: object
    h: int
    d: int
    rows_per_cta: int
    has_gate: bool
    # The torch dtype the artifact was traced for (it is in the compile-cache
    # key). `run_elementwise_gate` refuses an operand of any other dtype, the
    # way `run_quantize` guards `dtype_in`. Appended with a default so the
    # record grows append-only; `None` (a hand-built recipe) skips that check.
    dtype: object = None


def compile_elementwise_gate(
    *,
    dtype,
    h: int,
    d: int,
    has_gate: bool,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
    use_pdl: bool = False,
) -> ElementwiseRecipe:
    """Build from SHAPES ALONE — no allocation, no launch."""
    global _FAKE_STREAM
    validate_shape(d, threads_per_cta)
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"elementwise gate serves bf16/f16 only, got {dtype}")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (str(dtype), h, d, bool(has_gate), int(threads_per_cta), int(rows_per_group), bool(const_head_count), bool(use_pdl), current_device())
    if key not in compiled_cache:
        # The dict stays the in-process memo; behind it sits the persistent compiled-plan cache (``cudnn.frost.compiled_cache``),
        # keyed by this file's digest + the dict key's plain values: a second process reloads the object instead of re-tracing.
        cache_key = template_key(globals(), {"key": key}, "compile_elementwise_gate")
        tok = cute.sym_int()
        # All three carry their own symbolic token stride: src/dst are compact
        # in the block, gate is a column slice of the fused projection.
        src = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d)
        gate = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) if has_gate else None
        dst = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d)
        compiled_cache[key] = compile_cached(
            elementwise_gate_launch,
            src,
            gate,
            dst,
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int32(h),  # h        )
            cutlass.Int32(0),  # n_blocks )
            int(h),
            bool(const_head_count),
            d,
            int(threads_per_cta),
            int(rows_per_group),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
            cache_key=cache_key,
            symbol="cudnn_block_elementwise_gate",
        )
    return ElementwiseRecipe(
        compiled=compiled_cache[key],
        h=h,
        d=d,
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        has_gate=bool(has_gate),
        dtype=dtype,
    )


def run_elementwise_gate(r: ElementwiseRecipe, src, gate, dst, *, stream) -> None:
    """Launch. ``src``/``gate``/``dst`` are ``[T, H, D]``; ``dst`` may alias ``src``."""
    if r.has_gate and gate is None:
        raise ValueError("this artifact was compiled WITH a gate operand; it must be bound at execute (Rule 1: no silent fallback)")
    if not r.has_gate and gate is not None:
        raise ValueError("this artifact was compiled WITHOUT a gate operand; passing one would silently ignore it (Rule 1)")
    for name, ten in (("src", src), ("gate", gate), ("dst", dst)):
        if ten is not None and r.dtype is not None and ten.dtype != r.dtype:
            # The tvm-ffi boundary also rejects this on cutlass-dsl >= 4.8
            # ("Mismatched Tensor on argument #N ... expected dtype=bfloat16"),
            # but that check is the DSL's, indexed by argument position, and
            # FROST's DSL floor (CUTEDSL_MIN_VERSION, 4.7.0) predates it. Name the operand here, the way
            # `run_quantize` does for its `dtype_in`.
            raise ValueError(f"{name} is {ten.dtype} but this artifact was compiled for {r.dtype}; dtype is fixed per artifact")
        if ten is not None and int(ten.shape[1]) != r.h:
            # Pre-existing hole, not one `const_head_count` introduced: `n_rows`
            # below has ALWAYS been derived from `r.h`, so a mismatched H
            # silently addressed the wrong rows. Under `const_head_count` it
            # would also mis-divide. Fail loudly in both modes.
            raise ValueError(f"{name} has H={int(ten.shape[1])} but this artifact was compiled for H={r.h}; H is fixed per artifact")
    t = int(src.shape[0])
    n_rows = t * r.h
    n_blocks = (n_rows + r.rows_per_cta - 1) // r.rows_per_cta
    r.compiled(src, gate, dst, cutlass.Int32(n_rows), cutlass.Int32(r.h), cutlass.Int32(n_blocks), cuda.CUstream(int(stream)))


def moved_bytes(t: int, h: int, d: int, *, elem_bytes: int = 2, has_gate: bool = True) -> int:
    """HBM traffic of one launch — the denominator for an SOL number.

    Reads src (+ gate) and writes dst. In-place still moves all of it: the write
    is a write whether or not it lands on the read's address.
    """
    return (3 if has_gate else 2) * t * h * d * elem_bytes


frost_elementwise_gate.set_name_prefix("cudnn", remove_cutlass_symbol=True)
