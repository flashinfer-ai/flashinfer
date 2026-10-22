# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: BSD-3-Clause
#
# Packed conversion/store sequences adapted from hca_v41.py and
# hca_v41_primitives.py.
"""Selected mixed-cache tiles for the D512 Prims-TS Keep-AB graph.

SWA bytes and reconstructed page64/group16 FP4 share E4M3 cache units. The
FP4 conversion reuses exact BF16 registers before E4M3 RN/SATFINITE packing,
without a request amax, division or descale. Thread-written D128 stages stay
resident through delayed PV; no full converted KV cache is materialized.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch import nvvm_wrappers as nvw
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import SignalingThreads
from cutlass.experimental.task_scheduling.resources import PipelineConfig, StageInfo

from ...attention.prims_ts.kernels.mla_decode.helpers.ops import pack_float4_to_fp8_e4m3
from ...attention.prims_ts.kernels.mla_decode.helpers.stage import MlaStage
from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
    validate_split_softmax_scale_config,
)
from ...attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.resources.smem_resources import (
    SmemKvResource,
)
from . import hca_v41_primitives as hw
from ._sparse_views import bind_combined_sparse_views


@dataclass(frozen=True)
class Fp4KvProducer:
    native_fp4: bool

    def compile_signature(self):
        return ("dsv41_page64_fp4_group16_primary", self.native_fp4)

    def pipeline_config(self, cfg, load_group, umma_group, cta_layout_vmnk):
        if (
            cfg.kernel_variant != "keeps_mma_ab"
            or cfg.qkv_dtype != "bf16"
            or cfg.rope_dim != 0
            or cfg.head_dim_qk != 512
            or cfg.head_dim_per_stage_kv != 128
            or cfg.tile_size_kv != 64
            or cfg.load_num_warps != 4
            or not cfg.sparse_reuse_kv
            or not cfg.sparse_direct
        ):
            raise ValueError(
                "packed V4.1 KV requires BF16 Keep-AB/BK64 retained stages"
            )
        # Every writing thread arrives after its async-proxy fence. Native TMA
        # completion bytes would never arrive for these ordinary SMEM stores.
        return PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.kv_stages,
            producer_group=load_group,
            consumer_group=umma_group,
            cta_layout_vmnk=cta_layout_vmnk,
            producer_signaling_threads=SignalingThreads.All,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
        )

    def make_resource(self, pools, **kwargs):
        return Fp4SmemKvResource(
            fp4_cache=pools[0],
            native_fp4=self.native_fp4,
            **kwargs,
        )


@dataclass(kw_only=True)
class Fp4SmemKvResource(SmemKvResource):
    fp4_cache: object = None
    native_fp4: cutlass.Constexpr[bool] = True

    @cute.jit
    def _store_fragment(self, fragment, stage_base, row, dim):
        # A D128 stage is two consecutive [BK,64] BF16 planes. SW128 XORs
        # bits 3:5 of the element offset with row bits 0:2. This is the native
        # descriptor layout, not Frost's full-D512 SMEM staging layout.
        element = (
            (dim // 64) * (self.cfg.tile_size_kv * 64) + row * 64 + (dim & 63)
        ) ^ ((row & 7) * 8)
        dst = cute.make_ptr(
            cutlass.BFloat16,
            stage_base.data_ptr().toint(cutlass.Int32) + element * 2,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        hw.store_bf16_fragment(fragment, dst)

    @cute.jit
    def _store_fp4_chunk(self, chunk, scale_word, stage_base, row, dim):
        if cutlass.const_expr(self.native_fp4):
            packed = hw.fp8x2_to_bf16_word(scale_word).to(cutlass.Uint32)
            scale0 = ((packed & 65535) * 65537).to(cutlass.Int32)
            scale1 = ((packed >> 16) * 65537).to(cutlass.Int32)
            for word in cutlass.range_constexpr(4):
                scale = scale0 if word < 2 else scale1
                values = hw.fp4x8_to_bf16_words(chunk[word])
                bits = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
                # E2M1 times finite E4M3 is exact in BF16 (at most six
                # significant bits); avoid FP32 expansion on CUDA >= 13.2.
                for pair in cutlass.range_constexpr(4):
                    bits[pair] = prims.mul_bf16x2(values[pair], scale)
                fragment = cute.make_tensor(
                    cute.recast_ptr(bits.iterator, dtype=cutlass.BFloat16),
                    cute.make_layout(8),
                )
                self._store_fragment(fragment, stage_base, row, dim + word * 8)
        else:
            scale_bytes = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Uint8)
            scale_bytes[0] = (scale_word & 255).to(cutlass.Uint8)
            scale_bytes[1] = ((scale_word >> 8) & 255).to(cutlass.Uint8)
            scales = cute.make_tensor(
                cute.recast_ptr(scale_bytes.iterator, dtype=cutlass.Float8E4M3FN),
                cute.make_layout(2),
            )
            for word in cutlass.range_constexpr(4):
                packed = cute.make_tensor(chunk.iterator + word, cute.make_layout(1))
                fp4 = cute.make_tensor(
                    cute.recast_ptr(packed.iterator, dtype=cutlass.Float4E2M1FN),
                    cute.make_layout(8),
                ).load()
                fp16 = nvw.cvt_f4e2m1x8_to_f16x8(fp4)
                values = cute.TensorSSA(fp16, (8,), cutlass.Float16).to(cutlass.Float32)
                scale = scales[word // 2].to(cutlass.Float32)
                fragment = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
                fragment.store((values * scale).to(cutlass.BFloat16))
                self._store_fragment(fragment, stage_base, row, dim + word * 8)

    @cute.jit
    def _load_fp4_stage(
        self,
        pool,
        stage_base,
        cached_page_ids,
        *,
        stage_idx: cutlass.Constexpr[int],
        page_id_slot: cutlass.Constexpr[int],
    ):
        lane = cute.arch.thread_idx()[0] % 32
        warp = cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
        chunk_elements = 32
        chunks_per_row = 128 // chunk_elements
        rows_per_group = 32 // chunks_per_row
        rows_per_warp = self.cfg.tile_size_kv // self.cfg.load_num_warps
        row_bytes, scale_bytes = 256, 32
        capacity = cutlass.Int64(cute.size(pool)) // (row_bytes + scale_bytes)
        base = page_id_slot * self.cfg.sparse_cache_words
        for group in cutlass.range_constexpr(rows_per_warp // rows_per_group):
            local_row = group * rows_per_group + lane // chunks_per_row
            row = warp * rows_per_warp + local_row
            chunk_idx = lane % chunks_per_row
            raw = cutlass.Int32(0x7FFFFFFF)
            # Page staging caches row quads in this warp. All writing lanes
            # recover their row from that cache; no index reread per D.
            for entry in cutlass.range_constexpr(4):
                value = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_page_ids[base + entry],
                    offset=local_row // 4,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                if local_row % 4 == entry:
                    raw = cutlass.Int32(value)
            slot = raw & cutlass.Int32(0x7FFFFFFF)
            chunk = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
            for word in cutlass.range_constexpr(4):
                chunk[word] = 0
            scale_word = cutlass.Int32(0)
            if slot != cutlass.Int32(0x7FFFFFFF) and cutlass.Int64(slot) < capacity:
                # Page data and scale planes are separate. Widen before every
                # byte-stride multiplication, including pools above 2 GiB.
                page_base = cutlass.Int64(slot // 64) * (64 * (row_bytes + scale_bytes))
                token = cutlass.Int64(slot & 63)
                column = stage_idx * 128 + chunk_idx * chunk_elements
                data_offset = page_base + token * row_bytes + cutlass.Int64(column // 2)
                scale_offset = (
                    page_base
                    + 64 * row_bytes
                    + token * scale_bytes
                    + cutlass.Int64(column // 16)
                )
                data_ptr = cute.make_ptr(
                    cutlass.Int32,
                    pool.iterator.toint() + data_offset,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                chunk.store(hw.load_cache_vector(data_ptr))
                scale_ptr = cute.make_ptr(
                    cutlass.Uint16,
                    pool.iterator.toint() + scale_offset,
                    cute.AddressSpace.gmem,
                    assumed_align=2,
                )
                scale_word = hw.load_cache_scale(scale_ptr, cutlass.Uint16)
            dim = chunk_idx * chunk_elements
            self._store_fp4_chunk(chunk, scale_word, stage_base, row, dim)

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        del inst_id, is_v, section
        stage_base = self._stage_base(stage_info)
        self._load_fp4_stage(
            self.fp4_cache,
            stage_base,
            cached_page_ids,
            stage_idx=stage_idx,
            page_id_slot=page_id_slot,
        )
        # Publication is the task graph's AsyncThread producer commit. Each
        # writer fences its own stores; all 128 arrivals precede UMMA's wait.
        cute.arch.fence_view_async_shared()


@dataclass(frozen=True)
class Fp4Fp8KvProducer(Fp4KvProducer):
    def compile_signature(self):
        return ("dsv41_page64_fp4_group16_to_e4m3_units_v1", self.native_fp4)

    def pipeline_config(self, cfg, load_group, umma_group, cta_layout_vmnk):
        if (
            cfg.kernel_variant != "keeps_mma_ab"
            or cfg.qkv_dtype != "e4m3"
            or cfg.rope_dim != 0
            or cfg.head_dim_qk != 512
            or cfg.head_dim_per_stage_kv != 128
            or cfg.tile_size_q != 64
            or cfg.tile_size_kv != 128
            or cfg.kv_stages != 8
            or cfg.load_num_warps != 4
            or not cfg.sparse_reuse_kv
            or not cfg.sparse_direct
            or not cfg.sparse_static_scales
        ):
            raise ValueError(
                "V4.1 FP4-to-E4M3 KV requires Keep-AB/M64/BK128/KV8 retained stages"
            )
        # The inherited loader fences every writer before the same all-writer
        # AsyncThread commit as the BF16 resource; these are not TMA stores.
        return PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.kv_stages,
            producer_group=load_group,
            consumer_group=umma_group,
            cta_layout_vmnk=cta_layout_vmnk,
            producer_signaling_threads=SignalingThreads.All,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
        )

    def make_resource(self, pools, *, runtime_args=None, **kwargs):
        if runtime_args is not None:
            raise ValueError(
                "FP4-to-E4M3 KV uses common cache units, not request scales"
            )
        return Fp4Fp8SmemKvResource(
            fp4_cache=pools[0],
            native_fp4=self.native_fp4,
            **kwargs,
        )


@dataclass(kw_only=True)
class Fp4Fp8SmemKvResource(Fp4SmemKvResource):
    @cute.jit
    def _store_fragment(self, fragment, stage_base, row, dim):
        # Scaled E2M1 is exact in inherited BF16 registers. Both sources are
        # already in common cache units: apply E4M3 RN/SATFINITE directly.
        values = fragment.load().to(cutlass.Float32)
        packed = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.rmem)
        for word in cutlass.range_constexpr(2):
            column = word * 4
            packed[word] = pack_float4_to_fp8_e4m3(
                values[column],
                values[column + 1],
                values[column + 2],
                values[column + 3],
            )
        # Native FP8 D128 descriptors use one SW128 byte plane. Eight values
        # occupy eight bytes, unlike the BF16 base's split-plane 16-byte store.
        offset = (row * 128 + dim) ^ ((row & 7) * 16)
        dst = cutlass.inttoptr(
            stage_base.data_ptr().toint(cutlass.Int32) + offset,
            mem_space=3,
            dtype=cutlass.Int32,
        )
        dst.store(packed.load(0, 2), alignment=8)


@dataclass(frozen=True)
class MixedFp8KvProducer(Fp4Fp8KvProducer):
    def compile_signature(self):
        return (
            "dsv41_shared_pertensor_joint_swa_fp4_e4m3_inline_metadata_v2",
            self.native_fp4,
        )

    @cute.jit
    def bind_sparse_views(
        self, indices, seq_lens, topk_lens, pools, capacity, request=None
    ):
        return bind_combined_sparse_views(
            indices, seq_lens, topk_lens, pools, capacity, request
        )

    def pipeline_config(self, cfg, load_group, umma_group, cta_layout_vmnk):
        if not cfg.sparse_split_softmax_scale:
            raise ValueError("mixed V4.1 KV requires ordered runtime BMM1 softmax")
        validate_split_softmax_scale_config(cfg)
        return super().pipeline_config(cfg, load_group, umma_group, cta_layout_vmnk)

    def make_resource(self, pools, *, runtime_args=None, **kwargs):
        if runtime_args is not None:
            raise ValueError("mixed V4.1 KV does not accept request scales")
        return MixedFp8SmemKvResource(
            fp4_cache=pools[0],
            swa_cache=pools[1],
            native_fp4=self.native_fp4,
            **kwargs,
        )


@dataclass(kw_only=True)
class MixedFp8SmemKvResource(Fp4Fp8SmemKvResource):
    swa_cache: cute.Tensor = None

    @cute.jit
    def _load_swa_stage(
        self,
        stage_base,
        cached_page_ids,
        *,
        stage_idx: cutlass.Constexpr[int],
        page_id_slot: cutlass.Constexpr[int],
    ):
        lane = cute.arch.thread_idx()[0] % 32
        warp = cute.arch.thread_idx()[0] // 32 - self.cfg.load_warp_idx
        capacity = cutlass.Int64(cute.size(self.swa_cache)) // 512
        base = page_id_slot * self.cfg.sparse_cache_words
        # Eight lanes copy each row's D128 slice in disjoint aligned 16B
        # vectors. Smaller stores respect SW128's separate 16B half-chunks;
        # an unpermuted 32B store would silently exchange their contents.
        for group in cutlass.range_constexpr(8):
            local_row = group * 4 + lane // 8
            row = warp * 32 + local_row
            dim = (lane % 8) * 16
            raw = cutlass.Int32(0x7FFFFFFF)
            for entry in cutlass.range_constexpr(4):
                value = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_page_ids[base + entry],
                    offset=local_row // 4,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                if local_row % 4 == entry:
                    raw = cutlass.Int32(value)
            slot = raw & cutlass.Int32(0x7FFFFFFF)
            words = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for word in cutlass.range_constexpr(4):
                words[word] = cutlass.Int32(0)
            if slot != cutlass.Int32(0x7FFFFFFF) and cutlass.Int64(slot) < capacity:
                offset = cutlass.Int64(slot) * 512 + cutlass.Int64(
                    stage_idx * 128 + dim
                )
                src = cute.make_ptr(
                    cutlass.Int32,
                    self.swa_cache.iterator.toint() + offset,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                loaded = hw.load_cache_vector(src)
                for word in cutlass.range_constexpr(4):
                    words[word] = loaded[word]
            offset = (row * 128 + dim) ^ ((row & 7) * 16)
            dst = cutlass.inttoptr(
                stage_base.data_ptr().toint(cutlass.Int32) + offset,
                mem_space=3,
                dtype=cutlass.Int32,
            )
            dst.store(words.load(0, 4), alignment=16)

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        if cutlass.const_expr(inst_id != 0 or is_v != 0):
            raise ValueError("mixed V4.1 KV retains K through PV on one stream")
        stage_base = self._stage_base(stage_info)
        # The fixed 128-slot primary span guarantees exactly one SWA HEAD.
        # Keep-AB's single KV pipe loads all following main tiles in LOOP;
        # counts cannot shorten that span, even for a completely masked SWA.
        if cutlass.const_expr(section == MlaStage.Head):
            self._load_swa_stage(
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )
        elif cutlass.const_expr(section == MlaStage.Loop):
            self._load_fp4_stage(
                self.fp4_cache,
                stage_base,
                cached_page_ids,
                stage_idx=stage_idx,
                page_id_slot=page_id_slot,
            )
        else:
            raise ValueError("unsupported mixed V4.1 KV loading phase")
        # Both sources use thread stores and the same all-128-writer commit.
        # This deliberately avoids mixing TMA completion with thread arrivals;
        # retained K is released by the native graph only after delayed PV.
        cute.arch.fence_view_async_shared()
