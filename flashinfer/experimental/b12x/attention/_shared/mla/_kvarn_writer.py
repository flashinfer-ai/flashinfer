"""KVarN dual-rate MLA cache-record writer (arXiv:2606.03458), per-token.

Writes the per-token dual-plane record consumed by ``ModelType.KVARN_MLA`` /
``ScaleFormat.KVARN_DUAL``.  One CTA per token; thread 0 runs the whole token
serially (the proven old-writer data flow: two gmem passes over the 512
rotated bf16 values -- min/max, then re-load + quantize + serial LSB-first
bit-pack -- with no value staging in smem, no shuffles, no rmem tensors).
Each record carries inline per-token asymmetric RTN scales for both planes
(the ninfer record scheme -- no VarN, no tile-coupled scales, no whole-page
constraint).  The store path is therefore capture-safe at any batch size and
never re-quantizes older records.

Record layout (per token, latent_dim=512), little-endian::

    [   0, k_code_bytes)          k_codes  (per-token RTN, LSB-first dense)
    [k_code_bytes, +v_code_bytes) v_codes  (per-token RTN, LSB-first dense)
    [+0] k_scale fp16  [+2] k_zp fp16  [+4] v_scale fp16  [+6] v_zp fp16
    [content, record_bytes)          zero pad

Slot->byte mapping (must match the io read path's kvarn branch)::

    block = slot // page_block_size; local = slot % page_block_size
    tile  = local // group;          row   = local % group
    byte  = block*block_stride + tile*tile_stride + row*record_bytes
    tile_stride = group*record_bytes + _TILE_TAIL_BYTES  (reserved region)

Dequant identities (reference = vllm/v1/attention/ops/kvarn_mla.py)::

    k_hat[r,c] = k_codes[r,c] * k_scale[r] + k_zp[r]
    v_hat[r,c] = v_codes[r,c] * v_scale[r] + v_zp[r]
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from threading import RLock

import cutlass
import cutlass.cute as cute
import torch
import cuda.bindings.driver as cuda
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as compile_cute,
    run_compiled,
    tensor_compile_fact,
)
from b12x._lib.program_cache import register_program_cache
from b12x._lib.intrinsics import (
    fmax_f32,
    fmin_f32,
    get_ptr_as_int64,
    pack_f32x2_to_f16x2,
    st_global_u16,
    st_global_u32,
)
from b12x._lib.utils import current_cuda_stream

from .kv_cache import (
    _bf16x2_to_f32x2,
    _ld_global_u32,
    _to_kernel_tensor,
)

_LATENT_DIM = 512
_TILE_TAIL_BYTES = 6400
_DBG_STAGE = int(os.environ.get("KVARN_DBG_STAGE", "99"))


@dsl_user_op
def _kvarn_round_f32(x: Float32, *, loc=None, ip=None) -> Int32:
    """Round-to-nearest-even (torch.round semantics) via ``cvt.rni``."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rni.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _kvarn_rtn_code(x: Float32, mn: Float32, scale: Float32, levels: int) -> Int32:
    """Asymmetric RTN code: round((x - mn) / scale) clamped to [0, levels]."""
    q = (x - mn) / scale
    r = _kvarn_round_f32(q)
    r = fmax_f32(fmin_f32(r, Float32(levels)), Float32(0.0))
    return Int32(r)


@cute.jit
def _kvarn_pack_bit(acc, filled, widx, v, dst_base, bits, code_bytes):
    """LSB-first dense bit-stream assembler; stores a u32 word when 32 bits fill.

    ``acc`` holds the next ``filled`` stream bits (bit 0 = earliest).  Returns
    the updated ``(acc, filled, widx)``.  ``dst_base`` is the plane's gmem base;
    word ``widx`` lands at ``dst_base + widx*4``.  All launched variants have
    ``512 * bits % 32 == 0``, so no residue remains after the last code.
    """
    acc = acc | (Uint64(v) << Uint64(filled))
    filled = filled + Int32(bits)
    if filled >= Int32(32):
        # u16 stores: pool rows are 2 mod 4 on odd rows (602B inline-tail
        # stride), so 4-byte record stores would be misaligned.
        word = (acc & Uint64(0xFFFFFFFF)).to(Uint32)
        st_global_u16(dst_base + Int64(widx) * Int64(4), word & Uint32(0xFFFF))
        st_global_u16(dst_base + Int64(widx) * Int64(4) + Int64(2), word >> Uint32(16))
        widx = widx + Int32(1)
        acc = acc >> Uint64(32)
        filled = filled - Int32(32)
    return acc, filled, widx


def kvarn_record_bytes(k_bits: int, v_bits: int, latent_dim: int = 512) -> int:
    """Record width for a (k_bits, v_bits) KVarN variant.

    Mirrors vllm.v1.attention.ops.kvarn_mla.KvarnMlaLayout.record_bytes:
    dual-plane variants are 8-byte aligned content PLUS the mod-33 rule
    (0 < r % 33 <= 21); single-plane variants (v_bits == 0) skip the bump
    (records end exactly on the C4 boundary, no tile-tail data).
    """
    k_code_bytes = math.ceil(latent_dim * int(k_bits) / 8)
    v_code_bytes = math.ceil(latent_dim * int(v_bits) / 8)
    content = k_code_bytes + v_code_bytes + 8
    r = (content + 7) // 8 * 8
    if int(v_bits) == 0:
        return r
    while r % 33 == 0 or r % 33 > 21:
        r += 8
    return r


def kvarn_page_tail_bytes(latent_dim: int = 512) -> int:
    """Legacy per-page tail width (S_c + rt_scale + rt_zp, fp16 each).

    The per-token record scheme stores every scale inline; this width only
    feeds the reserved-region geometry helpers below.
    """
    return int(latent_dim) * 2 * 3


def kvarn_tile_tail_region(
    latent_dim: int = 512, group: int = 256, record_bytes: int = 536
) -> int:
    """Reserved per-tile tail region: sized so the tile stride is an exact
    multiple of the 8448B C4 page and the region can never overlap the
    neighbor tile's records.  Dual-plane records pad it with slack (their
    mod-33 rule requires a gap); single-plane records (``record % 33 == 0``)
    end exactly on the C4 boundary and carry no tile-tail data, so the
    region is zero and the FP8 index tail starts flush."""
    if int(record_bytes) % 33 == 0:
        return 0
    tile = int(group) * int(record_bytes) + kvarn_page_tail_bytes(latent_dim)
    c4 = 8448
    return (tile + c4 - 1) // c4 * c4 - int(group) * int(record_bytes)


def kvarn_tile_stride_bytes(
    latent_dim: int = 512, group: int = 256, record_bytes: int = 536
) -> int:
    """Byte stride between consecutive group-record tiles in a manager page."""
    return int(group) * int(record_bytes) + kvarn_tile_tail_region(
        latent_dim, group, record_bytes
    )


class ConcatAndCacheKvarnMlaPerTokenKernel:
    """Per-token KVarN dual-rate record writer (inline per-token RTN)."""

    def __init__(self, page_block_size: int, k_bits: int, v_bits: int, group: int):
        self.page_block_size = int(page_block_size)
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        self.group = int(group)
        self.latent_dim = _LATENT_DIM
        self.k_code_bytes = (self.latent_dim * self.k_bits + 7) // 8
        self.v_code_bytes = (self.latent_dim * self.v_bits + 7) // 8
        self.content_bytes = self.k_code_bytes + self.v_code_bytes + 8
        self.record_bytes = kvarn_record_bytes(self.k_bits, self.v_bits)
        # C4-rounded reserved tail region: 6400 for k4v4 (256*536+6400 =
        # 17*8448), but k5v5/k4v2 round differently -- derive, never hardcode.
        self.tile_stride = kvarn_tile_stride_bytes(
            self.latent_dim, self.group, self.record_bytes
        )
        self.k_levels = (1 << self.k_bits) - 1
        self.v_levels = (1 << self.v_bits) - 1

    def __cache_key__(self):
        """Variant identity for the compile disk cache — see
        UnifiedDecodeKernel.__cache_key__ (same warm-cache collision)."""
        return (
            "ConcatAndCacheKvarnMlaPerTokenKernel",
            self.page_block_size,
            self.k_bits,
            self.v_bits,
            self.group,
            self.latent_dim,
            self.k_code_bytes,
            self.v_code_bytes,
            self.record_bytes,
            self.tile_stride,
        )

    @cute.jit
    def __call__(
        self,
        latent: cute.Tensor,
        kv_cache: cute.Tensor,
        slot_mapping: cute.Tensor,
        latent_stride: Int64,
        block_stride: Int64,
        tile_stride: Int64,
        row_stride: Int64,
        slot_capacity: Int64,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            latent,
            kv_cache,
            slot_mapping,
            latent_stride,
            block_stride,
            tile_stride,
            row_stride,
            slot_capacity,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=[32, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        latent: cute.Tensor,
        kv_cache: cute.Tensor,
        slot_mapping: cute.Tensor,
        latent_stride: Int64,
        block_stride: Int64,
        tile_stride: Int64,
        row_stride: Int64,
        slot_capacity: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        token = Int32(token_idx)

        # One thread per CTA runs the whole token serially.
        if tid == Int32(0):
            # Uniform per CTA: the whole CTA exits for padded slots.
            slot = Int64(slot_mapping[token])
            if (slot >= Int64(0)) & (slot < slot_capacity):
                block_size64 = Int64(self.page_block_size)
                group32 = Int32(self.group)
                tile_stride64 = Int64(tile_stride)
                block_idx = slot // block_size64
                local = slot - block_idx * block_size64
                tile = local // Int64(group32)
                row = local - tile * Int64(group32)
                rec = (
                    get_ptr_as_int64(kv_cache, 0)
                    + block_idx * block_stride
                    + tile * tile_stride64
                    + row * row_stride
                )

                # ---- pass 1: token min/max over the 512 rotated values ----
                row_base = get_ptr_as_int64(latent, token.to(Int64) * latent_stride)
                mn = Float32(1e30)
                mx = Float32(-1e30)
                for c in cutlass.range(Int32(_LATENT_DIM // 2), unroll=1):
                    pair = _ld_global_u32(row_base + Int64(c) * Int64(4))
                    f0, f1 = _bf16x2_to_f32x2(pair)
                    mn = fmin_f32(mn, f0)
                    mx = fmax_f32(mx, f0)
                    mn = fmin_f32(mn, f1)
                    mx = fmax_f32(mx, f1)
                k_scale = fmax_f32(mx - mn, Float32(1e-8)) / Float32(self.k_levels)
                if cutlass.const_expr(self.v_bits > 0):
                    v_scale = fmax_f32(mx - mn, Float32(1e-8)) / Float32(self.v_levels)

                # ---- pass 2: re-load (L2-hot), quantize, serial LSB-first pack ----
                k_dst = rec
                v_dst = rec + Int64(self.k_code_bytes)
                k_acc = Uint64(0)
                k_filled = Int32(0)
                k_widx = Int32(0)
                v_acc = Uint64(0)
                v_filled = Int32(0)
                v_widx = Int32(0)
                for c in cutlass.range(Int32(_LATENT_DIM // 2), unroll=1):
                    pair = _ld_global_u32(row_base + Int64(c) * Int64(4))
                    f0, f1 = _bf16x2_to_f32x2(pair)
                    kc0 = _kvarn_rtn_code(f0, mn, k_scale, self.k_levels)
                    kc1 = _kvarn_rtn_code(f1, mn, k_scale, self.k_levels)
                    k_acc, k_filled, k_widx = _kvarn_pack_bit(
                        k_acc,
                        k_filled,
                        k_widx,
                        kc0,
                        k_dst,
                        self.k_bits,
                        self.k_code_bytes,
                    )
                    k_acc, k_filled, k_widx = _kvarn_pack_bit(
                        k_acc,
                        k_filled,
                        k_widx,
                        kc1,
                        k_dst,
                        self.k_bits,
                        self.k_code_bytes,
                    )
                    if cutlass.const_expr(self.v_bits > 0):
                        vc0 = _kvarn_rtn_code(f0, mn, v_scale, self.v_levels)
                        vc1 = _kvarn_rtn_code(f1, mn, v_scale, self.v_levels)
                        v_acc, v_filled, v_widx = _kvarn_pack_bit(
                            v_acc,
                            v_filled,
                            v_widx,
                            vc0,
                            v_dst,
                            self.v_bits,
                            self.v_code_bytes,
                        )
                        v_acc, v_filled, v_widx = _kvarn_pack_bit(
                            v_acc,
                            v_filled,
                            v_widx,
                            vc1,
                            v_dst,
                            self.v_bits,
                            self.v_code_bytes,
                        )

                # ---- inline scale footer: [k_scale, k_zp], [v_scale, v_zp] fp16 ----
                foot = rec + Int64(self.k_code_bytes + self.v_code_bytes)
                w0 = pack_f32x2_to_f16x2(k_scale, mn)
                if cutlass.const_expr(self.v_bits > 0):
                    w1 = pack_f32x2_to_f16x2(v_scale, mn)
                else:
                    # Single-plane: v_scale/v_zp fields reserved as zero.
                    w1 = Uint32(0)
                st_global_u16(foot, w0 & Uint32(0xFFFF))
                st_global_u16(foot + Int64(2), w0 >> Uint32(16))
                st_global_u16(foot + Int64(4), w1 & Uint32(0xFFFF))
                st_global_u16(foot + Int64(6), w1 >> Uint32(16))
                # ---- zero the record pad: [content, record_bytes) ----
                pad_u16 = (self.record_bytes - self.content_bytes) // 2
                for pw in cutlass.range_constexpr(8):
                    if pw < pad_u16:
                        st_global_u16(
                            rec + Int64(self.content_bytes) + Int64(pw) * Int64(2),
                            Uint32(0),
                        )


# ---------------------------------------------------------------------------
# Launch / compile plumbing.
# ---------------------------------------------------------------------------
_KVARN_WRITER_LOCK = RLock()
_KVARN_WRITER_COMPILED: dict[tuple[int, int, int, int, torch.dtype], object] = {}
register_program_cache(_KVARN_WRITER_COMPILED, lock=_KVARN_WRITER_LOCK)


@lru_cache(maxsize=None)
def _build_concat_and_cache_kvarn_mla_per_token_kernel(
    page_block_size: int, k_bits: int, v_bits: int, group: int
) -> ConcatAndCacheKvarnMlaPerTokenKernel:
    return ConcatAndCacheKvarnMlaPerTokenKernel(page_block_size, k_bits, v_bits, group)


def clear_kvarn_mla_kv_cache_kernel_cache() -> None:
    _build_concat_and_cache_kvarn_mla_per_token_kernel.cache_clear()
    with _KVARN_WRITER_LOCK:
        _KVARN_WRITER_COMPILED.clear()


def _kvarn_writer_signature(latent, kv_cache, slot_mapping, k_bits, v_bits, group):
    return (
        int(latent.device.index),
        int(k_bits),
        int(v_bits),
        int(group),
        latent.dtype,
    )


def _kvarn_writer_launch(latent, kv_cache, slot_mapping, k_bits, v_bits, group):
    num_tokens = int(slot_mapping.shape[0])
    page_block_size = int(kv_cache.shape[1]) if kv_cache.ndim >= 2 else group
    if os.environ.get("KVARN_GATHER_DEBUG"):
        print(
            "[kvarn] writer launch: kv_cache",
            tuple(kv_cache.shape),
            "strides",
            tuple(kv_cache.stride()),
            "k_bits",
            k_bits,
            "v_bits",
            v_bits,
            "group",
            group,
            flush=True,
        )
    kernel = _build_concat_and_cache_kvarn_mla_per_token_kernel(
        page_block_size, k_bits, v_bits, group
    )
    latent_arg = _to_kernel_tensor(latent, assumed_align=4, leading_dim=1)
    kv_arg = _to_kernel_tensor(kv_cache, assumed_align=16, leading_dim=2)
    slot_arg = _to_kernel_tensor(
        slot_mapping,
        assumed_align=8 if slot_mapping.dtype == torch.int64 else 4,
        leading_dim=0,
    )
    latent_stride = Int64(latent.stride(0))
    if kv_cache.ndim >= 2:
        block_stride = Int64(kv_cache.stride(0))
        # Records live at the bound row stride (the pool carries an inline
        # page tail per token: record_bytes + page_tail_bytes_per_token);
        # writing at record_bytes drifts every row after the first.
        row_stride_host = int(kv_cache.stride(1))
        row_stride = Int64(row_stride_host)
        # Tiles are LINEAR within a page: tile t starts at row t*group, so
        # its stride is row_stride*group. Deriving from stride(0) breaks on
        # over-strided page views (multi-layer pools whose stride(0) spans
        # more than this layer's page rows).
        tile_stride = Int64(row_stride_host * int(group))
    else:
        # Flat manager-page storage: one page per block view.
        block_stride = Int64(kernel.tile_stride * (page_block_size // group))
        tile_stride = Int64(int(kernel.tile_stride))
        row_stride = Int64(int(kernel.record_bytes))
    # The guard is `0 <= slot < slot_capacity`; the bound is the POOL's
    # token capacity, not the batch token count (runtime slots are pool
    # positions and exceed num_tokens, which silently dropped every write).
    if kv_cache.ndim >= 2:
        slot_capacity = Int64(int(kv_cache.shape[0]) * int(kv_cache.shape[1]))
    else:
        slot_capacity = Int64(num_tokens)
    args = (
        latent_arg,
        kv_arg,
        slot_arg,
        latent_stride,
        block_stride,
        tile_stride,
        row_stride,
        slot_capacity,
        Int32(num_tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_compile_fact("latent", latent, dynamic_dims=(0,), dynamic_strides=(0,)),
        tensor_compile_fact(
            "kv_cache", kv_cache, dynamic_dims=(0,), dynamic_strides=(0,)
        ),
        tensor_compile_fact("slot_mapping", slot_mapping, dynamic_dims=(0,)),
        int(k_bits),
        int(v_bits),
        int(group),
    )
    spec = KernelCompileSpec.from_key(
        "attention.mla.kvarn_mla_kv_cache",
        1,
        cache_key,
        labels=("latent", "kv_cache", "slot_mapping", "k_bits", "v_bits", "group"),
    )
    return kernel, args, spec


def _compile_kvarn_writer(latent, kv_cache, slot_mapping, k_bits, v_bits, group):
    signature = _kvarn_writer_signature(
        latent, kv_cache, slot_mapping, k_bits, v_bits, group
    )
    with _KVARN_WRITER_LOCK:
        compiled = _KVARN_WRITER_COMPILED.get(signature)
    if compiled is None:
        kernel, args, spec = _kvarn_writer_launch(
            latent, kv_cache, slot_mapping, k_bits, v_bits, group
        )
        compiled = compile_cute(kernel, *args, compile_spec=spec)
        if not isinstance(latent, torch._subclasses.fake_tensor.FakeTensor):
            # FakeTensor (preparation-time) compiles are NOT cached for runtime
            # reuse: the fake-traced binary miscompiles this kernel's serial
            # single-thread body. The runtime call recompiles from real tensors.
            with _KVARN_WRITER_LOCK:
                _KVARN_WRITER_COMPILED[signature] = compiled
    return compiled


def concat_and_cache_glm_next_mla_kvarn(
    latent: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_bits: int,
    v_bits: int,
    group: int = 256,
) -> None:
    """Write KVarN dual-rate records for arbitrary token batches.

    ``latent`` is ``[T, 512]`` bf16 (any T >= 0), already Hadamard-rotated.
    ``slot_mapping`` gives the global record slot for each token; slots < 0
    are skipped (padded CUDA-graph slots).  ``kv_cache`` is the paged uint8
    pool whose manager-page stride is ``kv_cache.stride(0)``.
    """
    if int(slot_mapping.shape[0]) == 0:
        return
    if int(latent.shape[0]) != int(slot_mapping.shape[0]):
        raise ValueError("latent and slot_mapping must have the same token count")
    _compile_kvarn_writer(latent, kv_cache, slot_mapping, k_bits, v_bits, group)
    _, args, _ = _kvarn_writer_launch(
        latent, kv_cache, slot_mapping, k_bits, v_bits, group
    )
    with _KVARN_WRITER_LOCK:
        compiled = _KVARN_WRITER_COMPILED.get(
            _kvarn_writer_signature(
                latent, kv_cache, slot_mapping, k_bits, v_bits, group
            )
        )
    run_compiled(compiled, args)


def dequant_kvarn_mla_records(
    records: torch.Tensor,
    k_bits: int,
    v_bits: int,
    latent_dim: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference dequant of per-token KVarN records.

    ``records`` is ``[T, record_bytes]`` uint8.  Returns ``(k_hat, v_hat)``
    ``[T, latent_dim]`` float32, matching
    ``vllm.v1.attention.ops.kvarn_mla.unpack_kvarn_mla_records``.
    """
    k_code_bytes = math.ceil(latent_dim * int(k_bits) / 8)
    v_code_bytes = math.ceil(latent_dim * int(v_bits) / 8)
    rec_f16 = records.view(torch.float16)
    k_codes = _unpack_bits_torch(records[:, :k_code_bytes], k_bits, latent_dim)
    k_scale = rec_f16[:, (k_code_bytes + v_code_bytes) // 2].to(torch.float32)
    k_zp = rec_f16[:, (k_code_bytes + v_code_bytes + 2) // 2].to(torch.float32)
    k_hat = k_codes * k_scale.unsqueeze(1) + k_zp.unsqueeze(1)
    if int(v_bits) == 0:
        # Single-plane: the value path reads the shared k_hat codes.
        return k_hat, k_hat
    v_codes = _unpack_bits_torch(
        records[:, k_code_bytes : k_code_bytes + v_code_bytes], v_bits, latent_dim
    )
    v_scale = rec_f16[:, (k_code_bytes + v_code_bytes + 4) // 2].to(torch.float32)
    v_zp = rec_f16[:, (k_code_bytes + v_code_bytes + 6) // 2].to(torch.float32)
    v_hat = v_codes * v_scale.unsqueeze(1) + v_zp.unsqueeze(1)
    return k_hat, v_hat


def _unpack_bits_torch(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    """Inverse of the kernel's LSB-first dense bit packing (torch reference)."""
    orig_shape = packed.shape
    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
    rows = flat.shape[0]
    device = packed.device
    bit_index = torch.arange(cols, device=device) * bits
    codes = torch.zeros(rows, cols, dtype=torch.int32, device=device)
    for b in range(bits):
        pos = bit_index + b
        byte_idx = (pos >> 3).unsqueeze(0).expand(rows, -1)
        bit_in_byte = pos & 7
        byte_vals = flat.gather(1, byte_idx.clamp_max(packed.shape[-1] - 1))
        codes |= ((byte_vals >> bit_in_byte) & 1) << b
    return codes.reshape(orig_shape[:-1] + (cols,))
