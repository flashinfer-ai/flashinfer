"""Paged K/V cache append for the layer families ``paged_decode`` serves.

Each token's new K and V rows go to their page slot (vLLM slot mapping:
``page * page_size + offset``); negative slots are padded CUDA-graph rows and
are skipped.  Rows move as 16-byte vectors: BF16 caches copy them, FP8 caches
store ``e4m3(x / scale)`` (FP32 division, round to nearest, saturating) for 16
values per store.  All page, slot and head offsets are Int64.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import Int64

from b12x._lib.intrinsics.conversion import cvt_f32x4_to_e4m3x4
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_v4_u32,
    u32_as_f32,
)

KV_WRITE_THREADS = 128


class PagedKVWriteKernel:
    """One CTA per token; each thread moves 16-byte vectors of K then V rows."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim_k: int,
        head_dim_v: int,
        page_size: int,
        kv_fp8: bool,
    ):
        self.page_size = int(page_size)
        self.kv_fp8 = bool(kv_fp8)
        # Source (BF16) elements per 16-byte destination store.
        self.vec = 16 if self.kv_fp8 else 8
        self.cache_elem_bytes = 1 if self.kv_fp8 else 2
        if head_dim_k % self.vec or head_dim_v % self.vec:
            raise ValueError("head dims must be multiples of the 16-byte vector")
        self.k_vecs = head_dim_k // self.vec
        self.v_vecs = head_dim_v // self.vec
        self.vecs_per_head = self.k_vecs + self.v_vecs
        self.total_vecs = int(num_kv_heads) * self.vecs_per_head
        self.iters = -(-self.total_vecs // KV_WRITE_THREADS)

    @cute.jit
    def __call__(
        self,
        key: cute.Tensor,  # [T, Hkv, Dk] bf16, unit inner stride
        value: cute.Tensor,  # [T, Hkv, Dv] bf16, unit inner stride
        k_cache: cute.Tensor,  # [pages, page_size, Hkv, Dk] bf16 or u8 (e4m3)
        v_cache: cute.Tensor,  # [pages, page_size, Hkv, Dv] bf16 or u8 (e4m3)
        slot_mapping: cute.Tensor,  # [T] int64
        k_scale: cute.Tensor,  # [1] f32 (FP8 only; read but unused for BF16)
        v_scale: cute.Tensor,  # [1] f32
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale
        ).launch(
            grid=(num_tokens, 1, 1),
            block=[KV_WRITE_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        key: cute.Tensor,
        value: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        slot_mapping: cute.Tensor,
        k_scale: cute.Tensor,
        v_scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        token = Int64(token_idx)
        # Decode appends a handful of rows per layer, so the kernel is one
        # memory round trip plus its stores.  Everything that does not depend
        # on the slot (the slot itself, the FP8 scales and this thread's
        # source rows) is loaded before the slot is examined.
        slot = Int64(slot_mapping[token_idx])
        k_s = Float32(1.0)
        v_s = Float32(1.0)
        if const_expr(self.kv_fp8):
            k_s = Float32(k_scale[0])
            v_s = Float32(v_scale[0])
        k_src = get_ptr_as_int64(key, 0) + token * Int64(key.stride[0]) * Int64(2)
        v_src = get_ptr_as_int64(value, 0) + token * Int64(value.stride[0]) * Int64(2)
        heads = []
        is_key = []
        cols = []
        rows = []
        for it in cutlass.range_constexpr(self.iters):
            # Idle lanes of the last pass reload the final vector and never store.
            v = cutlass.min(
                Int32(tidx) + Int32(it * KV_WRITE_THREADS), Int32(self.total_vecs - 1)
            )
            head = v // Int32(self.vecs_per_head)
            j = v - head * Int32(self.vecs_per_head)
            k_part = j < Int32(self.k_vecs)
            col = Int64(cutlass.select_(k_part, j, j - Int32(self.k_vecs))) * Int64(
                self.vec
            )
            src = cutlass.select_(
                k_part,
                k_src + (Int64(head) * Int64(key.stride[1]) + col) * Int64(2),
                v_src + (Int64(head) * Int64(value.stride[1]) + col) * Int64(2),
            )
            heads.append(head)
            is_key.append(k_part)
            cols.append(col)
            rows.append(self._load(src))
        capacity = Int64(k_cache.shape[0]) * Int64(self.page_size)
        if (slot >= Int64(0)) & (slot < capacity):
            page = slot // Int64(self.page_size)
            offset = slot - page * Int64(self.page_size)
            eb = Int64(self.cache_elem_bytes)
            k_row = (
                get_ptr_as_int64(k_cache, 0)
                + (page * Int64(k_cache.stride[0]) + offset * Int64(k_cache.stride[1]))
                * eb
            )
            v_row = (
                get_ptr_as_int64(v_cache, 0)
                + (page * Int64(v_cache.stride[0]) + offset * Int64(v_cache.stride[1]))
                * eb
            )
            for i in cutlass.range_constexpr(self.iters):
                if Int32(tidx) + Int32(i * KV_WRITE_THREADS) < Int32(self.total_vecs):
                    head64 = Int64(heads[i])
                    dst = cutlass.select_(
                        is_key[i],
                        k_row + (head64 * Int64(k_cache.stride[2]) + cols[i]) * eb,
                        v_row + (head64 * Int64(v_cache.stride[2]) + cols[i]) * eb,
                    )
                    self._store(dst, rows[i], cutlass.select_(is_key[i], k_s, v_s))

    def _load(self, src: Int64):
        """This thread's source vector: 8 BF16 values, or 16 for an FP8 store."""
        a0, a1, a2, a3 = ld_global_v4_u32(src)
        if const_expr(self.kv_fp8):
            b0, b1, b2, b3 = ld_global_v4_u32(src + Int64(16))
            return a0, a1, a2, a3, b0, b1, b2, b3
        return a0, a1, a2, a3

    def _store(self, dst: Int64, row, scale: Float32):
        if const_expr(self.kv_fp8):
            a0, a1, a2, a3, b0, b1, b2, b3 = row
            st_global_v4_u32(
                dst,
                self._e4m3x4(a0, a1, scale),
                self._e4m3x4(a2, a3, scale),
                self._e4m3x4(b0, b1, scale),
                self._e4m3x4(b2, b3, scale),
            )
        else:
            a0, a1, a2, a3 = row
            st_global_v4_u32(dst, a0, a1, a2, a3)

    @cute.jit
    def _e4m3x4(self, lo: Uint32, hi: Uint32, scale: Float32) -> Uint32:
        """Four BF16 values (two bf16x2 words, low half first) to packed E4M3."""
        top = Uint32(0xFFFF0000)
        return cvt_f32x4_to_e4m3x4(
            u32_as_f32(lo << Uint32(16)) / scale,
            u32_as_f32(lo & top) / scale,
            u32_as_f32(hi << Uint32(16)) / scale,
            u32_as_f32(hi & top) / scale,
        )


__all__ = ["KV_WRITE_THREADS", "PagedKVWriteKernel"]
