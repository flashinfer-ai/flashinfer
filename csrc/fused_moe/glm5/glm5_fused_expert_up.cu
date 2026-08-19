/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// DeepSeek-V3 fused expert-up FP8-MMA kernel.
//
// For each decode token (M <= 4), this kernel consumes router logits and bf16
// hidden states, applies the DeepSeek-V3 no-aux top-k routing rule, quantizes
// activations per 128 columns, and computes shared/routed gate-up projections
// with FP8 MMA. The output is the routed top-k metadata and fp16 expert slots
// consumed by dsv3_fused_expert_down.
//
// The deployed path uses packed weights. The raw-weight launcher remains for
// diagnostics and shape comparison, but Python inference calls the packed op.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <list>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "topk_reduce.cuh"
#include "tvm_ffi_utils.h"

namespace cg = cooperative_groups;

namespace
{

// ---- Constants ----
// Model-architecture constants (unchanged across TP):
constexpr int kNumExperts = 256;
constexpr int kSharedExpert = 256;
constexpr int kTopK = 8;
// kSlotsPerToken is DeepSeek-V3-architecture-locked: 1 shared expert + 8 routed top-k.
// This does NOT depend on TP, despite the legacy name suggesting otherwise.
constexpr int kSlotsPerToken = kSharedExpert / kNumExperts + kTopK; // 1 + 8 = 9
constexpr int kThreadsPerCta = 384;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kThreadsPerCta / kWarpSize;
constexpr int kMaxNumExpertsUnit = 128;
constexpr int kNumExpertWarps = (kNumExperts - 1) / kMaxNumExpertsUnit + 1;
static_assert(kNumExpertWarps == 2);
constexpr int kMaxNumTopGroups = 4;
constexpr int kNumInterTopK = kNumExpertWarps * kTopK;

constexpr int kHidden = 6144;
constexpr int kCtaOutRows = 64;
constexpr int kMaxCudaDevicesForSmemAttr = 64;

// TP-dependent constants. The kernel is templated on kInterPerTpParam; the
// host-side dispatcher selects {256, 512} based on input tensor shape.
//   TP=8: kInterPerTp = 2048/8 = 256 -> kSubRowsPerExpert=4, kCtasPerToken=36
//   TP=4: kInterPerTp = 2048/4 = 512 -> kSubRowsPerExpert=8, kCtasPerToken=72
constexpr int kInterPerTp_TP8 = 256;
constexpr int kInterPerTp_TP4 = 512;

// Helper constexpr functions (used by both host and __global__ template kernels).
__host__ __device__ constexpr int sub_rows_per_expert(int kInterPerTp)
{
    return kInterPerTp / kCtaOutRows;
}

__host__ __device__ constexpr int ctas_per_token(int kInterPerTp)
{
    return kSlotsPerToken * sub_rows_per_expert(kInterPerTp);
}

__host__ __device__ constexpr int weight_scale_m_blocks(int kInterPerTp)
{
    return kInterPerTp / 128;
}

// Invariant: kSlotsPerToken = 1 + kTopK assumes exactly one shared expert.
// The expert_slot==0 branch later (`my_expert = kSharedExpert`) hard-codes
// this; if a future model has 2+ shared experts, that branch and the
// kSlotsPerToken formula above need to be revisited together.
static_assert(kSharedExpert == kNumExperts,
    "kSlotsPerToken derivation assumes exactly one shared expert "
    "(kSharedExpert == kNumExperts).");

// Invariant: each FP8 weight scale m-block covers 128 rows; each CTA covers
// kCtaOutRows=64 rows; so exactly 2 CTAs share one scale m-block. The raw
// scale lookup computes m_block_idx = sr / (kSubRowsPerExpert /
// kWeightScaleMBlocks) and relies on that ratio being 2 at every supported TP.
static_assert(sub_rows_per_expert(kInterPerTp_TP8) / weight_scale_m_blocks(kInterPerTp_TP8) == 2,
    "TP=8: kCtaOutRows must be half the 128-row FP8 scale m-block "
    "(2 CTAs per scale m-block).");
static_assert(sub_rows_per_expert(kInterPerTp_TP4) / weight_scale_m_blocks(kInterPerTp_TP4) == 2,
    "TP=4: kCtaOutRows must be half the 128-row FP8 scale m-block "
    "(2 CTAs per scale m-block).");

// K-axis constants are functions of kHidden only, not TP.
constexpr int kKTile = 768;
constexpr int kNumKIter = kHidden / kKTile; // 8
constexpr int kKSubsPerIter = kKTile / 32;  // 24

// w_scale tensor original shape: [E, kWeightScaleMBlocks, 48]. kWeightScaleMBlocks
// depends on TP (kInterPerTp / 128). kWeightScaleKBlocks is a function of kHidden only.
constexpr int kWeightScaleKBlocks = 48; // original 128-col K-blocks (kHidden / 128)
constexpr int kWeightScaleKBlocksPerKIter = kWeightScaleKBlocks / kNumKIter; // 6

constexpr int kWorkerWarpBase = 4;
constexpr int kNumWorkers = 8;
constexpr int kRowsPerWorker = 8;

constexpr int kTileBytes = kCtaOutRows * kKTile; // 49152 (48 KiB)

constexpr int kStages = 1;
constexpr int kPackedStagesSingle = 1;
// The packed path can afford a second weight stage because this kernel runs one CTA per SM.
constexpr int kPackedStagesDouble = 2;
constexpr int kPackedStagesDefault = kPackedStagesDouble;
constexpr int kPackedLoadCpAsync = 0;
constexpr int kPackedLoadTma = 1;

constexpr float kInvalidScore = -INFINITY;
constexpr float kFp8Max = 448.f;
constexpr float kInvFp8Max = 1.f / kFp8Max;

constexpr int kActBytes = kHidden * 2;                               // 12288
constexpr int kActCpAsyncs = kActBytes / 16;                         // 768
constexpr int kActCpAsyncsPerThread = kActCpAsyncs / kThreadsPerCta; // 2

// K-major slab offset constants. The packed tile stacks 6 sub-slabs along Z
// (k_sixth axis) so each box load stays within the TMA SWIZZLE_NONE 256-byte
// inner-dim cap.
constexpr int kLaneBytes = 16;                                                    // 16 B per lane
constexpr int kKsubBytes = kWarpSize * kLaneBytes;                                // 512 B per K-sub
constexpr int kKSubsPerThird = 4;                                                 // 4 k_subs per k_sixth
constexpr int kMtileSubslabBytes = kKSubsPerThird * kKsubBytes;                   // 2048 B per m_tile within a k_sixth
constexpr int kSubslabBytes = kCtaOutRows / 16 * kMtileSubslabBytes;              // 4 * 2048 = 8192 B per k_sixth
constexpr int kCombinedMtilesPerCta = kCtaOutRows / kRowsPerWorker;
constexpr int kCombinedSubslabBytes = kCombinedMtilesPerCta * kMtileSubslabBytes; // 8 * 2048 = 16384 B
constexpr int kCombinedTileBytes = kWeightScaleKBlocksPerKIter * kCombinedSubslabBytes; // 98304 (96 KiB)
constexpr int kPackedTmaInnerBytes = 128;
constexpr int kCombinedPackedTmaRows = kCombinedSubslabBytes / kPackedTmaInnerBytes;
constexpr int kPackedTmaSubslabs = kTileBytes / kSubslabBytes;
static_assert(kSubslabBytes % kPackedTmaInnerBytes == 0);
static_assert(kCombinedSubslabBytes % kPackedTmaInnerBytes == 0);
static_assert(kPackedTmaSubslabs == kWeightScaleKBlocksPerKIter);
static_assert(kCombinedTileBytes == 2 * kTileBytes);
static_assert(kCombinedMtilesPerCta == kNumWorkers);

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// -------------------------------------------------------------------------
// `ld.shared.v2.b32` = LDS.64 (8 bytes / lane).
__device__ __forceinline__ void lds64_b32x2(uint32_t& r0, uint32_t& r1, __nv_fp8_e4m3 const* smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(const_cast<__nv_fp8_e4m3*>(smem_ptr));
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// mbarrier + cp.async.bulk.tensor.3d wrappers for the packed-weight TMA path.
__device__ __forceinline__ uint32_t cvt_smem_addr(void const* smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, int arrive_count)
{
    uint32_t const addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(arrive_count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t bytes)
{
    uint32_t const addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar)
{
    uint32_t const addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(addr));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase)
{
    uint32_t const addr = cvt_smem_addr(mbar);
    asm volatile(
        "{\n"
        " .reg .pred P;\n"
        " WAIT_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "  @P bra DONE_%=;\n"
        "  bra WAIT_%=;\n"
        " DONE_%=:\n"
        "}\n" ::"r"(addr),
        "r"(phase));
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d(
    void* smem_dst, CUtensorMap const* tmap, int32_t coord_x, int32_t coord_y, int32_t coord_z, uint64_t* mbar)
{
    uint32_t const smem_addr = cvt_smem_addr(smem_dst);
    uint32_t const mbar_addr = cvt_smem_addr(mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(smem_addr),
        "l"(tmap), "r"(coord_x), "r"(coord_y), "r"(coord_z), "r"(mbar_addr));
}

__device__ __forceinline__ void fence_proxy_async_shared()
{
    asm volatile("fence.proxy.async.shared::cta;\n" :::);
}

// -------------------------------------------------------------------------
// Combined packed-tile consumer addressing: 6 stacked sub-slabs (k_sixth dim
// outer, then 8 worker m_tiles, then k_sub_in_sixth, then lane). Each worker
// m_tile stores 8 gate rows followed by the corresponding 8 up rows.
// Variable names `k_third` / `kKSubsPerThird` are historical; the constants
// express 6 sub-slabs of 128 K-cols each.
// -------------------------------------------------------------------------
__device__ __forceinline__ int fused_expert_up_combined_lane_offset(int m_tile, int k_sub, int lane)
{
    int const k_third = k_sub / kKSubsPerThird;        // 0..5
    int const k_sub_in_third = k_sub % kKSubsPerThird; // 0..3
    return k_third * kCombinedSubslabBytes + m_tile * kMtileSubslabBytes + k_sub_in_third * kKsubBytes
        + lane * kLaneBytes;
}

// Compute MMA for ONE K-iter (kKTile = 768 cols => 24 m16n8k32 MMAs per fragment).
//
// The 16-row MMA tile is physically packed as 8 gate rows followed by 8 up
// rows for the same output-row stripe. The top half and bottom half therefore
// use different weight dequant scales, while sharing the same activation scale.
//
__device__ __forceinline__ void compute_mma_kiter_fused_expert_up(__nv_fp8_e4m3 const* __restrict__ smem_tile,
    __nv_fp8_e4m3 const* __restrict__ smem_act_fp8, int k_iter, int my_m, int lane,
    float const (&gate_block_scale)[kWeightScaleKBlocksPerKIter],
    float const (&up_block_scale)[kWeightScaleKBlocksPerKIter], float (&d_out)[4])
{
#pragma unroll
    for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
    {
        float c_frag[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks_in_kb = 0; ks_in_kb < kKSubsPerThird; ++ks_in_kb)
        {
            int const k_sub = kb * kKSubsPerThird + ks_in_kb;
            // Per-lane base - natural K-major layout, 16 contiguous bytes/lane.
            __nv_fp8_e4m3 const* lane_base = smem_tile + fused_expert_up_combined_lane_offset(my_m, k_sub, lane);

            // 2 x LDS.64 fetch the 4 b32 A-frag chunks. ptxas typically merges
            // these into a single LDS.128 in the SASS.
            uint32_t a_frag[4];
            lds64_b32x2(a_frag[0], a_frag[1], lane_base);     // chunks 0+1
            lds64_b32x2(a_frag[2], a_frag[3], lane_base + 8); // chunks 2+3

            // B-fragment (activations). Lanes 0..3 hold the K-contiguous 16
            // bytes/half-K-sub; other lanes hold zeros. Unchanged from v34.
            uint32_t b_frag[2];
            if (lane < 4)
            {
                int const k_base = k_iter * kKTile + k_sub * 32 + (lane & 3) * 4;
                uint32_t b_lo_pair[2];
                uint32_t b_hi_pair[2];
                lds64_b32x2(b_lo_pair[0], b_lo_pair[1], smem_act_fp8 + k_base);
                lds64_b32x2(b_hi_pair[0], b_hi_pair[1], smem_act_fp8 + k_base + 16);
                b_frag[0] = b_lo_pair[0];
                b_frag[1] = b_hi_pair[0];
            }
            else
            {
                b_frag[0] = 0;
                b_frag[1] = 0;
            }

            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(c_frag[0]), "+f"(c_frag[1]), "+f"(c_frag[2]), "+f"(c_frag[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
        }

#pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            d_out[i] += c_frag[i] * gate_block_scale[kb];
            d_out[i + 2] += c_frag[i + 2] * up_block_scale[kb];
        }
    }
}

template <int kInterPerTpParam, bool kGate>
__device__ __forceinline__ __nv_fp8_e4m3 const* raw_weight_ptr(__nv_fp8_e4m3 const* __restrict__ shared_gate_up_weight,
    __nv_fp8_e4m3 const* __restrict__ routed_w3_w1_weight, int expert, int row, int col)
{
    constexpr int kInterPerTp = kInterPerTpParam;
    if (expert == kSharedExpert)
    {
        // Shared expert is stored as one [gate, up] matrix.
        int const row_offset = (kGate ? 0 : kInterPerTp) + row;
        return shared_gate_up_weight + static_cast<int64_t>(row_offset) * kHidden + col;
    }

    // Routed experts are stored as one [up, gate] matrix per expert.
    int const row_offset = (kGate ? kInterPerTp : 0) + row;
    return routed_w3_w1_weight + (static_cast<int64_t>(expert) * (2 * kInterPerTp) + row_offset) * kHidden + col;
}

template <int kInterPerTpParam, bool kGate>
__device__ __forceinline__ float const* raw_scale_ptr(float const* __restrict__ shared_gate_up_scale,
    float const* __restrict__ routed_w3_w1_scale, int expert, int sub_row, int k_iter)
{
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTpParam);
    constexpr int kWeightScaleMBlocks = weight_scale_m_blocks(kInterPerTpParam);
    int const m_block_idx = sub_row / (kSubRowsPerExpert / kWeightScaleMBlocks);
    if (expert == kSharedExpert)
    {
        // Shared expert scales follow [gate, up] order.
        int const m_offset = (kGate ? 0 : kWeightScaleMBlocks) + m_block_idx;
        return shared_gate_up_scale + static_cast<int64_t>(m_offset) * kWeightScaleKBlocks
            + k_iter * kWeightScaleKBlocksPerKIter;
    }

    // Routed expert scales follow [up, gate] order.
    int const m_offset = (kGate ? kWeightScaleMBlocks : 0) + m_block_idx;
    return routed_w3_w1_scale
        + (static_cast<int64_t>(expert) * (2 * kWeightScaleMBlocks) + m_offset) * kWeightScaleKBlocks
        + k_iter * kWeightScaleKBlocksPerKIter;
}

template <int kInterPerTpParam>
__device__ __forceinline__ int packed_weight_tile_idx(int expert, int sub_row, int k_iter)
{
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTpParam);
    return (expert * kSubRowsPerExpert + sub_row) * kNumKIter + k_iter;
}

template <int kInterPerTpParam>
__device__ __forceinline__ __nv_fp8_e4m3 const* packed_weight_tile_ptr(
    __nv_fp8_e4m3 const* __restrict__ expert_gate_up_weight, int expert, int sub_row, int k_iter)
{
    int const tile_idx = packed_weight_tile_idx<kInterPerTpParam>(expert, sub_row, k_iter);
    return expert_gate_up_weight + static_cast<int64_t>(tile_idx) * kCombinedTileBytes;
}

template <int kInterPerTpParam>
__device__ __forceinline__ void load_packed_weight_tile_fused_expert_up(__nv_fp8_e4m3* __restrict__ smem_tile,
    __nv_fp8_e4m3 const* __restrict__ expert_gate_up_weight, int expert, int sub_row, int k_iter, int tidx)
{
    constexpr int kCopyBytes = 16;
    static_assert(kCombinedTileBytes % kCopyBytes == 0);

    char const* const src = reinterpret_cast<char const*>(
        packed_weight_tile_ptr<kInterPerTpParam>(expert_gate_up_weight, expert, sub_row, k_iter));
    char* const dst = reinterpret_cast<char*>(smem_tile);

#pragma unroll 1
    for (int byte_off = tidx * kCopyBytes; byte_off < kCombinedTileBytes; byte_off += kThreadsPerCta * kCopyBytes)
    {
        unsigned const dst_smem = __cvta_generic_to_shared(dst + byte_off);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem), "l"(src + byte_off));
    }
}

template <int kInterPerTpParam>
__device__ __forceinline__ void issue_packed_weight_tma_fused_expert_up(__nv_fp8_e4m3* __restrict__ smem_tile,
    CUtensorMap const* expert_gate_up_tma, int expert, int sub_row, int k_iter, uint64_t* mbar)
{
    int const tile_idx = packed_weight_tile_idx<kInterPerTpParam>(expert, sub_row, k_iter);
    int const coord_z = tile_idx * kPackedTmaSubslabs;
    mbarrier_arrive_expect_tx(mbar, kCombinedTileBytes);
    cp_async_bulk_tensor_3d(smem_tile, expert_gate_up_tma, 0, 0, coord_z, mbar);
}

template <int kInterPerTpParam>
__device__ __forceinline__ void load_raw_weight_tile_fused_expert_up(__nv_fp8_e4m3* __restrict__ smem_tile,
    __nv_fp8_e4m3 const* __restrict__ shared_gate_up_weight, __nv_fp8_e4m3 const* __restrict__ routed_w3_w1_weight,
    int expert, int sub_row, int k_iter, int tidx)
{
    // Fill the combined 96 KiB K-major slab layout directly from the existing
    // row-major model weights. Each worker tile stores 8 gate rows followed by
    // the corresponding 8 up rows.
#pragma unroll 1
    for (int tri = tidx; tri < 6144; tri += kThreadsPerCta)
    {
        int const m_tile = tri / (kKSubsPerIter * kWarpSize);
        int const tri_in_mtile = tri % (kKSubsPerIter * kWarpSize);
        int const k_sub = tri_in_mtile / kWarpSize;
        int const lane = tri_in_mtile & 31;

        int const k_third = k_sub / kKSubsPerThird;
        int const k_sub_in_third = k_sub % kKSubsPerThird;

        int const expert_row = m_tile * kRowsPerWorker + (lane >> 2);
        int const col_lo_in_block = k_sub_in_third * 32 + ((lane & 3) << 2);
        int const col_hi_in_block = col_lo_in_block + 16;

        int const gm = sub_row * kCtaOutRows + expert_row;
        int const gk_lo = k_iter * kKTile + k_third * 128 + col_lo_in_block;
        int const gk_hi = k_iter * kKTile + k_third * 128 + col_hi_in_block;

        uint32_t const a = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, true>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm, gk_lo));
        uint32_t const b = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, false>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm, gk_lo));
        uint32_t const c = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, true>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm, gk_hi));
        uint32_t const d = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, false>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm, gk_hi));

        uint8_t* dst = reinterpret_cast<uint8_t*>(smem_tile) + k_third * kCombinedSubslabBytes
            + m_tile * kMtileSubslabBytes + k_sub_in_third * kKsubBytes + lane * kLaneBytes;
        *reinterpret_cast<uint32_t*>(dst) = a;
        *reinterpret_cast<uint32_t*>(dst + 4) = b;
        *reinterpret_cast<uint32_t*>(dst + 8) = c;
        *reinterpret_cast<uint32_t*>(dst + 12) = d;
    }
}

template <int kQuantWarps>
__device__ __forceinline__ void quant_act_blocks_fused_expert_up(cg::thread_block_tile<kWarpSize> warp,
    __nv_bfloat16 const* __restrict__ smem_act_bf16, __nv_fp8_e4m3* __restrict__ smem_act_fp8,
    float* __restrict__ smem_act_block_scales, int q_warp, int lane)
{
    constexpr int kQuantRounds = (kWeightScaleKBlocks + kQuantWarps - 1) / kQuantWarps;
    constexpr int kElemsPerLane128Block = 128 / kWarpSize;

#pragma unroll
    for (int r = 0; r < kQuantRounds; ++r)
    {
        int const kb_global = q_warp + r * kQuantWarps;
        if (kb_global >= kWeightScaleKBlocks)
        {
            break;
        }

        int const block_off = kb_global * 128; // bf16 element offset

        // Each lane reads 4 contiguous bf16 elements (8 B vector load).
        __nv_bfloat16 const* lane_src = smem_act_bf16 + block_off + lane * kElemsPerLane128Block;
        __nv_bfloat162 v01;
        __nv_bfloat162 v23;
        v01 = *reinterpret_cast<__nv_bfloat162 const*>(lane_src);
        v23 = *reinterpret_cast<__nv_bfloat162 const*>(lane_src + 2);
        float const f0 = __bfloat162float(__low2bfloat16(v01));
        float const f1 = __bfloat162float(__high2bfloat16(v01));
        float const f2 = __bfloat162float(__low2bfloat16(v23));
        float const f3 = __bfloat162float(__high2bfloat16(v23));

        float const lane_max = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
        float amax = cg::reduce(warp, lane_max, cg::greater<float>{});
        amax = fmaxf(amax, 1e-10f);
        float quant_scale = 0.f; // bf16 * quant_scale -> fp8

        if (lane == 0)
        {
            quant_scale = kFp8Max / amax;
            smem_act_block_scales[kb_global] = amax * kInvFp8Max;
        }
        quant_scale = __shfl_sync(0xffffffff, quant_scale, 0);

        // Quantize and write 4 fp8 elements per lane.
        __nv_fp8_e4m3* lane_dst = smem_act_fp8 + block_off + lane * kElemsPerLane128Block;
        float const q0 = fmaxf(-kFp8Max, fminf(kFp8Max, f0 * quant_scale));
        float const q1 = fmaxf(-kFp8Max, fminf(kFp8Max, f1 * quant_scale));
        float const q2 = fmaxf(-kFp8Max, fminf(kFp8Max, f2 * quant_scale));
        float const q3 = fmaxf(-kFp8Max, fminf(kFp8Max, f3 * quant_scale));
        __nv_fp8_e4m3 const fp8_0 = __nv_fp8_e4m3(q0);
        __nv_fp8_e4m3 const fp8_1 = __nv_fp8_e4m3(q1);
        __nv_fp8_e4m3 const fp8_2 = __nv_fp8_e4m3(q2);
        __nv_fp8_e4m3 const fp8_3 = __nv_fp8_e4m3(q3);
        // Pack 4 fp8 = 4 bytes = uint32 store.
        uint32_t const packed = (static_cast<uint32_t>(static_cast<uint8_t>(fp8_0.__x)) << 0)
            | (static_cast<uint32_t>(static_cast<uint8_t>(fp8_1.__x)) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(fp8_2.__x)) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(fp8_3.__x)) << 24);
        *reinterpret_cast<uint32_t*>(lane_dst) = packed;
    }
}

// -------------------------------------------------------------------------
// DeepSeek-V3 fused expert-up kernel.
//
// Defaults to one CTA per SM. Override via -DDSV3_FUSED_EXPERT_UP_LB_BLOCKS_PER_SM=N when
// experimenting with occupancy/register tradeoffs.
// -------------------------------------------------------------------------

#ifndef DSV3_FUSED_EXPERT_UP_LB_BLOCKS_PER_SM
#define DSV3_FUSED_EXPERT_UP_LB_BLOCKS_PER_SM 1
#endif

template <int kInterPerTpParam, bool kUsePackedWeights, int kPackedWeightStagesParam = kPackedStagesDefault,
    int kPackedWeightLoadModeParam = kPackedLoadCpAsync>
__global__ __launch_bounds__(384, DSV3_FUSED_EXPERT_UP_LB_BLOCKS_PER_SM) void dsv3_fused_expert_up_kernel(
    __grid_constant__ const CUtensorMap shared_gate_up_tma, __grid_constant__ const CUtensorMap routed_w3_w1_tma,
    float const* __restrict__ scores, __nv_bfloat16 const* __restrict__ hidden_in,
    __nv_bfloat16 const* __restrict__ bias, __nv_fp8_e4m3 const* __restrict__ shared_gate_up_weight,
    float const* __restrict__ shared_gate_up_scale, __nv_fp8_e4m3 const* __restrict__ routed_w3_w1_weight,
    float const* __restrict__ routed_w3_w1_scale, float* __restrict__ topk_weights, int32_t* __restrict__ topk_indices,
    // hidden_out is fp16 because dsv3_fused_expert_down consumes fp16 slots.
    __half* __restrict__ hidden_out, int64_t num_tokens, float routed_scaling_factor)
{

    constexpr int kInterPerTp = kInterPerTpParam;
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTp);
    static_assert(kPackedWeightStagesParam == kPackedStagesSingle || kPackedWeightStagesParam == kPackedStagesDouble);
    static_assert(kPackedWeightLoadModeParam == kPackedLoadCpAsync || kPackedWeightLoadModeParam == kPackedLoadTma);
    constexpr int kWeightStages = kUsePackedWeights ? kPackedWeightStagesParam : kStages;
    constexpr bool kDoubleBufferedWeights = kUsePackedWeights && (kWeightStages == kPackedStagesDouble);
    constexpr bool kUsePackedTmaWeights = kUsePackedWeights && (kPackedWeightLoadModeParam == kPackedLoadTma);
    constexpr bool kUsePackedTmaFullEmptyPipeline = kUsePackedTmaWeights && kDoubleBufferedWeights;

    int const token = blockIdx.x;
    int const cta_y = blockIdx.y;
    int const expert_slot = cta_y / kSubRowsPerExpert;
    int const sub_row = cta_y % kSubRowsPerExpert;
    int const row_stripe_start = sub_row * kCtaOutRows;

    int const tidx = threadIdx.x;
    int const lane = tidx & (kWarpSize - 1);
    int const warp_idx = __shfl_sync(0xffffffff, tidx / kWarpSize, 0);

    extern __shared__ __align__(128) unsigned char smem_buf[];

    __nv_fp8_e4m3* const smem_weight_tiles = reinterpret_cast<__nv_fp8_e4m3*>(smem_buf);

    __nv_bfloat16* const smem_act_bf16
        = reinterpret_cast<__nv_bfloat16*>(smem_weight_tiles + kWeightStages * kCombinedTileBytes);
    // In-kernel per-128-col act quant writes fp8 to a SEPARATE buffer
    // (not aliased over bf16) so multiple warps can read bf16 / write fp8
    // in parallel without aliasing races. Costs 6 KiB.
    __nv_fp8_e4m3* const smem_act_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(smem_act_bf16 + kHidden);

    auto align_up_128 = [](uintptr_t p) -> uintptr_t { return (p + 127u) & ~uintptr_t(127); };
    uintptr_t rs_base = align_up_128(reinterpret_cast<uintptr_t>(smem_act_fp8 + kHidden));

    float* const smem_score_sigmoid = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kNumExperts;
    rs_base = align_up_128(rs_base);

    float* const smem_score_bias = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kNumExperts;
    rs_base = align_up_128(rs_base);

    float* const smem_inter_scores = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kNumInterTopK;
    rs_base = align_up_128(rs_base);

    int32_t* const smem_inter_experts = reinterpret_cast<int32_t*>(rs_base);
    rs_base += sizeof(int32_t) * kNumInterTopK;
    rs_base = align_up_128(rs_base);

    int32_t* const smem_topk_i = reinterpret_cast<int32_t*>(rs_base);
    rs_base += sizeof(int32_t) * kTopK;
    rs_base = align_up_128(rs_base);

    // Per-128-col activation dequant scales (TRTLLM 1x128 quant scheme).
    // 48 fp32 = one scale per 128-col K-block over kHidden=6144.
    // Computed in-kernel during Phase 1 (overlapped with top-K) and consumed
    // in the K-loop fold (replacing the old single per-tensor scalar).
    float* const smem_act_block_scales = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kWeightScaleKBlocks;
    rs_base = align_up_128(rs_base);

    float* const smem_gate_weight_scales = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kWeightScaleKBlocks;
    rs_base = align_up_128(rs_base);

    float* const smem_up_weight_scales = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kWeightScaleKBlocks;
    rs_base = align_up_128(rs_base);

    rs_base = (rs_base + 15u) & ~uintptr_t(15);
    uint64_t* const smem_tma_full = reinterpret_cast<uint64_t*>(rs_base);
    uint64_t* const smem_tma_empty = smem_tma_full + kWeightStages;
    if constexpr (kUsePackedTmaWeights)
    {
        rs_base += sizeof(uint64_t) * 2 * kWeightStages;
    }

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<kWarpSize>(block);

    // ===== Phase 0: activation cp.async prefetch =====
    {
        __nv_bfloat16 const* x_ptr = hidden_in + static_cast<int64_t>(token) * kHidden;
#pragma unroll
        for (int ii = 0; ii < kActCpAsyncsPerThread; ++ii)
        {
            int const byte_off = (ii * kThreadsPerCta + tidx) * 16;
            __nv_bfloat16 const* src
                = reinterpret_cast<__nv_bfloat16 const*>(reinterpret_cast<char const*>(x_ptr) + byte_off);
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(smem_act_bf16) + byte_off);
            unsigned const dst_smem = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem), "l"(src));
        }
        asm volatile("cp.async.commit_group;\n" :::);
    }
    // ===== Phase 1: noaux_tc score prep - every thread =====
    // Independent of activation, so keep it overlapped with the activation
    // cp.async flight issued above.
    {
        int const expert = tidx;
        bool const valid = expert < kNumExperts;
        if (valid)
        {
            float const bias_val = __bfloat162float(bias[expert]);
            float const score = scores[static_cast<int64_t>(token) * kNumExperts + expert];
            float const score_sigmoid = sigmoid_accurate(score);
            smem_score_sigmoid[expert] = score_sigmoid;
            smem_score_bias[expert] = score_sigmoid + bias_val;
        }
    }
    if constexpr (kUsePackedTmaWeights)
    {
        if constexpr (kUsePackedTmaFullEmptyPipeline)
        {
            if (tidx == 0)
            {
#pragma unroll
                for (int s = 0; s < kWeightStages; ++s)
                {
                    mbarrier_init(&smem_tma_full[s], 1);
                    mbarrier_init(&smem_tma_empty[s], kNumWorkers);
                }
            }
        }
        else
        {
            if (tidx < kWeightStages)
            {
                mbarrier_init(&smem_tma_full[tidx], 1);
            }
        }
        if (tidx == 0)
        {
            fence_proxy_async_shared();
        }
    }
    // Wait for activation cp.async completion before quant warps read
    // smem_act_bf16. The CTA sync also publishes the score-prep writes and any
    // TMA mbarrier initialization.
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    // ===== Phase 1+2 FUSED: top-K (warps 0..kNumExpertWarps-1)
    //                       || per-128-col activation FP8 quant (remaining warps)
    // -------------------------------------------------------------------
    float top_scores[kTopK];
    int32_t top_experts[kTopK];
    constexpr int kQuantWarps = kNumWarps - kNumExpertWarps; // 10

    if (warp_idx < kNumExpertWarps)
    {
        // ----- Top-K stage 1 -----
        int const offset = warp_idx * kWarpSize * kMaxNumTopGroups;
        float in_value[kMaxNumTopGroups];
        int32_t in_idx[kMaxNumTopGroups];
#pragma unroll
        for (int ii = 0; ii < kMaxNumTopGroups; ++ii)
        {
            int const e = ii * kWarpSize + lane;
            in_idx[ii] = offset + e;
            in_value[ii] = (offset + e) < kNumExperts ? smem_score_bias[offset + e] : kInvalidScore;
        }
        mega_topk::reduceTopK<kTopK, float, kMaxNumTopGroups>(
            warp, top_scores, top_experts, in_value, in_idx, kInvalidScore, kTopK);
        if (lane < kTopK)
        {
            smem_inter_scores[warp_idx * kTopK + lane] = top_scores[lane];
            smem_inter_experts[warp_idx * kTopK + lane] = top_experts[lane];
        }

        // Only the two top-K warps need smem_inter_* published before warp 0
        // can run stage 2; quant warps continue independently.
        asm volatile("bar.sync 1, 64;\n" ::: "memory");

        // ----- Top-K stage 2 (warp 0 only). -----
        if (warp_idx == 0)
        {
            float cand_val = (lane < kNumInterTopK) ? smem_inter_scores[lane] : kInvalidScore;
            int32_t cand_idx = (lane < kNumInterTopK) ? smem_inter_experts[lane] : (kNumExperts - 1);
            mega_topk::reduceTopK<kTopK, float>(
                warp, top_scores, top_experts, cand_val, cand_idx, kInvalidScore, kTopK);

            int32_t const expert_idx = (lane < kTopK) ? top_experts[lane] : (kNumExperts - 1);
            float const score_norm = (lane < kTopK) ? smem_score_sigmoid[expert_idx] : 0.f;
            // Match noAuxTcKernels.cu: the warp reduction itself is fp32, then
            // the double scaling factor and double literal promote the final
            // division expression before the OutputT cast.
            float const red_norm = cg::reduce(warp, score_norm, cg::plus<float>{});
            double const final_score_d
                = (double) score_norm * (double) routed_scaling_factor / ((double) red_norm + 1e-20);
            float const final_score = static_cast<float>(final_score_d);

            if (lane < kTopK)
            {
                smem_topk_i[lane] = expert_idx;
                if (blockIdx.y == 0)
                {
                    int64_t out_off = static_cast<int64_t>(token) * kTopK + lane;
                    topk_weights[out_off] = final_score;
                    topk_indices[out_off] = expert_idx;
                }
            }

            if constexpr (kUsePackedTmaFullEmptyPipeline)
            {
                int const early_expert_lane = (expert_slot == 0) ? 0 : (expert_slot - 1);
                int const early_expert_idx = __shfl_sync(0xffffffff, expert_idx, early_expert_lane);
                int const early_packed_expert = (expert_slot == 0) ? 0 : (early_expert_idx + 1);
                if (lane == 0)
                {
#pragma unroll
                    for (int s = 0; s < kWeightStages; ++s)
                    {
                        issue_packed_weight_tma_fused_expert_up<kInterPerTpParam>(
                            smem_weight_tiles + s * kCombinedTileBytes, &shared_gate_up_tma, early_packed_expert,
                            sub_row, s, smem_tma_full + s);
                    }
                }
            }
        }
    }
    else
    {
        quant_act_blocks_fused_expert_up<kQuantWarps>(
            warp, smem_act_bf16, smem_act_fp8, smem_act_block_scales, warp_idx - kNumExpertWarps, lane);
    }
    __syncthreads();

    int const my_expert = (expert_slot == 0) ? kSharedExpert : smem_topk_i[expert_slot - 1];
    int const packed_expert = (expert_slot == 0) ? 0 : (my_expert + 1);

    bool const is_worker = (warp_idx >= kWorkerWarpBase && warp_idx < (kWorkerWarpBase + kNumWorkers));
    int const my_m = is_worker ? (warp_idx - kWorkerWarpBase) : 0;

    float d_pair[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t tma_phase[kPackedStagesDouble] = {0u, 0u};

    if constexpr (kDoubleBufferedWeights)
    {
        // Prime stages before entering the loop; later iterations preload into empty stages.
        if constexpr (!kUsePackedTmaWeights)
        {
            load_packed_weight_tile_fused_expert_up<kInterPerTpParam>(
                smem_weight_tiles, shared_gate_up_weight, packed_expert, sub_row, 0, tidx);
            asm volatile("cp.async.commit_group;\n" :::);
            asm volatile("cp.async.wait_all;\n" :::);
        }
    }

    if (tidx < kWeightScaleKBlocks)
    {
        float const* gate_weight_scale_base = nullptr;
        float const* up_weight_scale_base = nullptr;
        if constexpr (kUsePackedWeights)
        {
            constexpr int kWeightScaleMBlocks = weight_scale_m_blocks(kInterPerTpParam);
            int const m_block_idx = sub_row / (kSubRowsPerExpert / kWeightScaleMBlocks);
            gate_weight_scale_base = shared_gate_up_scale
                + (static_cast<int64_t>(packed_expert) * 2 * kWeightScaleMBlocks + m_block_idx) * kWeightScaleKBlocks;
            up_weight_scale_base = shared_gate_up_scale
                + (static_cast<int64_t>(packed_expert) * 2 * kWeightScaleMBlocks + kWeightScaleMBlocks + m_block_idx)
                    * kWeightScaleKBlocks;
        }
        else
        {
            gate_weight_scale_base = raw_scale_ptr<kInterPerTpParam, true>(
                shared_gate_up_scale, routed_w3_w1_scale, my_expert, sub_row, 0);
            up_weight_scale_base = raw_scale_ptr<kInterPerTpParam, false>(
                shared_gate_up_scale, routed_w3_w1_scale, my_expert, sub_row, 0);
        }
        smem_gate_weight_scales[tidx] = __ldg(gate_weight_scale_base + tidx);
        smem_up_weight_scales[tidx] = __ldg(up_weight_scale_base + tidx);
    }
    __syncthreads();

    // ---- fused expert up K-loop - per-K-BLOCK scaling (6 scales per K-iter, applied
    //      block-wise inside the inner MMA loop). ----
    for (int k = 0; k < kNumKIter; ++k)
    {
        int constexpr kRawStage = 0;
        int const current_stage = kDoubleBufferedWeights ? (k & 1) : kRawStage;
        int const current_phase = kUsePackedTmaFullEmptyPipeline ? ((k / kWeightStages) & 1) : 0;

        if constexpr (kDoubleBufferedWeights)
        {
            if constexpr (kUsePackedTmaFullEmptyPipeline)
            {
                int const k_load = k + kWeightStages;
                if (k_load < kNumKIter && tidx == 0)
                {
                    int const load_stage = k_load % kWeightStages;
                    int const load_phase = (k_load / kWeightStages) & 1;
                    uint32_t const wait_phase = static_cast<uint32_t>(load_phase ^ 1);
                    mbarrier_wait_parity(smem_tma_empty + load_stage, wait_phase);

                    issue_packed_weight_tma_fused_expert_up<kInterPerTpParam>(
                        smem_weight_tiles + load_stage * kCombinedTileBytes, &shared_gate_up_tma, packed_expert,
                        sub_row, k_load, smem_tma_full + load_stage);
                }
            }
            else if (k + 1 < kNumKIter)
            {
                int const preload_stage = (k + 1) & 1;
                // Keep this prefetch ahead of MMA so copy latency overlaps with the current K tile.
                if constexpr (kUsePackedTmaWeights)
                {
                    if (tidx == 0)
                    {
                        issue_packed_weight_tma_fused_expert_up<kInterPerTpParam>(
                            smem_weight_tiles + preload_stage * kCombinedTileBytes, &shared_gate_up_tma, packed_expert,
                            sub_row, k + 1, smem_tma_full + preload_stage);
                    }
                }
                else
                {
                    load_packed_weight_tile_fused_expert_up<kInterPerTpParam>(
                        smem_weight_tiles + preload_stage * kCombinedTileBytes, shared_gate_up_weight, packed_expert,
                        sub_row, k + 1, tidx);
                    asm volatile("cp.async.commit_group;\n" :::);
                }
            }
        }

        if constexpr (kUsePackedWeights && !kDoubleBufferedWeights)
        {
            if constexpr (kUsePackedTmaWeights)
            {
                if (tidx == 0)
                {
                    issue_packed_weight_tma_fused_expert_up<kInterPerTpParam>(
                        smem_weight_tiles, &shared_gate_up_tma, packed_expert, sub_row, k, smem_tma_full);
                }
                if (tidx == 0)
                {
                    mbarrier_wait_parity(smem_tma_full, tma_phase[0]);
                    tma_phase[0] ^= 1u;
                }
            }
            else
            {
                load_packed_weight_tile_fused_expert_up<kInterPerTpParam>(
                    smem_weight_tiles, shared_gate_up_weight, packed_expert, sub_row, k, tidx);
                asm volatile("cp.async.commit_group;\n" :::);
                asm volatile("cp.async.wait_all;\n" :::);
            }
            __syncthreads();
        }
        else if constexpr (!kUsePackedWeights)
        {
            load_raw_weight_tile_fused_expert_up<kInterPerTpParam>(
                smem_weight_tiles, shared_gate_up_weight, routed_w3_w1_weight, my_expert, sub_row, k, tidx);
            __syncthreads();
        }

        if (is_worker)
        {
            float gate_block_scales[kWeightScaleKBlocksPerKIter];
            float up_block_scales[kWeightScaleKBlocksPerKIter];
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
            {
                int const scale_idx = k * kWeightScaleKBlocksPerKIter + kb;
                gate_block_scales[kb] = smem_gate_weight_scales[scale_idx] * smem_act_block_scales[scale_idx];
                up_block_scales[kb] = smem_up_weight_scales[scale_idx] * smem_act_block_scales[scale_idx];
            }
            if constexpr (kUsePackedTmaFullEmptyPipeline)
            {
                mbarrier_wait_parity(smem_tma_full + current_stage, static_cast<uint32_t>(current_phase));
            }
            compute_mma_kiter_fused_expert_up(smem_weight_tiles + current_stage * kCombinedTileBytes, smem_act_fp8, k,
                my_m, lane, gate_block_scales, up_block_scales, d_pair);
            if constexpr (kUsePackedTmaFullEmptyPipeline)
            {
                if (lane == 0)
                {
                    mbarrier_arrive(smem_tma_empty + current_stage);
                }
            }
        }
        if constexpr (kUsePackedTmaFullEmptyPipeline)
        {
            // Stage readiness and ownership are handled by full/empty barriers.
        }
        else if constexpr (kDoubleBufferedWeights)
        {
            if (k + 1 < kNumKIter)
            {
                int const preload_stage = (k + 1) & 1;
                if constexpr (kUsePackedTmaWeights)
                {
                    if (tidx == 0)
                    {
                        mbarrier_wait_parity(smem_tma_full + preload_stage, tma_phase[preload_stage]);
                        tma_phase[preload_stage] ^= 1u;
                    }
                }
                else
                {
                    asm volatile("cp.async.wait_all;\n" :::);
                }
            }
            __syncthreads();
        }
        else
        {
            __syncthreads();
        }
    }

    // ===== Phase 5: SiLU*x writer =====
    if (is_worker && (lane & 3) == 0)
    {
        int const local_row = lane >> 2;
        int const global_row = row_stripe_start + my_m * kRowsPerWorker + local_row;
        float const g = d_pair[0];
        float const u = d_pair[2];
        float const silu_g = g * sigmoid_accurate(g);
        float const h = silu_g * u;
        int64_t const out_off = static_cast<int64_t>(token) * kSlotsPerToken * kInterPerTp
            + static_cast<int64_t>(expert_slot) * kInterPerTp + global_row;
        // dsv3_fused_expert_down consumes this handoff as fp16.
        hidden_out[out_off] = __float2half(h);
    }
}

// Dynamic smem sizing.
template <bool kUsePackedWeights, int kPackedWeightStagesParam = kPackedStagesDefault,
    int kPackedWeightLoadModeParam = kPackedLoadCpAsync>
static inline size_t fused_expert_up_smem_bytes()
{
    auto align_up_128 = [](size_t p) -> size_t { return (p + 127u) & ~size_t(127); };
    static_assert(kPackedWeightLoadModeParam == kPackedLoadCpAsync || kPackedWeightLoadModeParam == kPackedLoadTma);
    constexpr int kWeightStages = kUsePackedWeights ? kPackedWeightStagesParam : kStages;
    constexpr bool kUsePackedTmaWeights = kUsePackedWeights && (kPackedWeightLoadModeParam == kPackedLoadTma);
    size_t bytes = 0;
    bytes += static_cast<size_t>(kWeightStages) * kCombinedTileBytes;
    bytes += static_cast<size_t>(kHidden) * sizeof(__nv_bfloat16);
    // Separate fp8 act buffer (no longer aliased over bf16). +6 KiB.
    bytes += static_cast<size_t>(kHidden) * sizeof(__nv_fp8_e4m3);
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumExperts;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumExperts;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumInterTopK;
    bytes = align_up_128(bytes);
    bytes += sizeof(int32_t) * kNumInterTopK;
    bytes = align_up_128(bytes);
    bytes += sizeof(int32_t) * kTopK;
    bytes = align_up_128(bytes);
    // smem_act_block_scales[48]: per-128-col activation dequant scales.
    bytes += sizeof(float) * kWeightScaleKBlocks;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kWeightScaleKBlocks;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kWeightScaleKBlocks;
    bytes = align_up_128(bytes);
    bytes = (bytes + 15u) & ~size_t(15);
    if constexpr (kUsePackedTmaWeights)
    {
        bytes += sizeof(uint64_t) * 2 * kWeightStages;
        bytes = (bytes + 15u) & ~size_t(15);
    }
    return bytes;
}

static CUtensorMap make_packed_fused_expert_up_tmap(
    void* base_ptr, int num_tiles, CUresult* out_err)
{
    CUtensorMap map = {};
    cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(kPackedTmaInnerBytes),
        static_cast<cuuint64_t>(kCombinedPackedTmaRows),
        static_cast<cuuint64_t>(num_tiles * kPackedTmaSubslabs),
    };
    cuuint64_t global_stride[2] = {
        static_cast<cuuint64_t>(kPackedTmaInnerBytes),
        static_cast<cuuint64_t>(kCombinedSubslabBytes),
    };
    cuuint32_t box_dim[3] = {
        static_cast<cuuint32_t>(kPackedTmaInnerBytes),
        static_cast<cuuint32_t>(kCombinedPackedTmaRows),
        static_cast<cuuint32_t>(kPackedTmaSubslabs),
    };
    cuuint32_t elem_stride[3] = {1u, 1u, 1u};

    *out_err = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /*rank=*/3, base_ptr, global_dim, global_stride, box_dim, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

constexpr size_t kPackedFusedExpertUpTmaDescCacheCap = 256;

struct PackedFusedExpertUpTmaDescKey
{
    void const* base;
    int numTiles;
    int deviceId;

    bool operator==(PackedFusedExpertUpTmaDescKey const& o) const noexcept
    {
        return base == o.base && numTiles == o.numTiles && deviceId == o.deviceId;
    }
};

struct PackedFusedExpertUpTmaDescKeyHash
{
    size_t operator()(PackedFusedExpertUpTmaDescKey const& k) const noexcept
    {
        size_t h = reinterpret_cast<uintptr_t>(k.base);
        h = h * 1099511628211ull + static_cast<size_t>(k.numTiles);
        h = h * 1099511628211ull + static_cast<size_t>(k.deviceId);
        return h;
    }
};

struct PackedFusedExpertUpTmaDescCache
{
    using ListIt =
        std::list<std::pair<PackedFusedExpertUpTmaDescKey, CUtensorMap>>::iterator;
    std::list<std::pair<PackedFusedExpertUpTmaDescKey, CUtensorMap>> order;
    std::unordered_map<PackedFusedExpertUpTmaDescKey, ListIt,
        PackedFusedExpertUpTmaDescKeyHash>
        index;
};

static CUtensorMap get_cached_packed_fused_expert_up_tmap(
    void* base_ptr, int num_tiles, int device_id, CUresult* out_err)
{
    static thread_local PackedFusedExpertUpTmaDescCache cache;
    PackedFusedExpertUpTmaDescKey const key{base_ptr, num_tiles, device_id};
    auto it = cache.index.find(key);
    if (it != cache.index.end())
    {
        cache.order.splice(cache.order.begin(), cache.order, it->second);
        *out_err = CUDA_SUCCESS;
        return it->second->second;
    }

    CUtensorMap const map =
        make_packed_fused_expert_up_tmap(base_ptr, num_tiles, out_err);
    if (*out_err != CUDA_SUCCESS)
    {
        return map;
    }
    if (cache.order.size() >= kPackedFusedExpertUpTmaDescCacheCap)
    {
        cache.index.erase(cache.order.back().first);
        cache.order.pop_back();
    }
    cache.order.emplace_front(key, map);
    cache.index.emplace(key, cache.order.begin());
    return map;
}

template <int kInterPerTpParam, int kPackedWeightStagesParam, int kPackedWeightLoadModeParam>
static void glm5_fused_expert_up_impl(TensorView scores, TensorView hidden_in, TensorView bias,
    TensorView expert_gate_up_weight, TensorView expert_gate_up_scale, TensorView topk_weights,
    TensorView topk_indices, TensorView hidden_out, double routed_scaling_factor)
{
    constexpr int kInterPerTp = kInterPerTpParam;
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTp);
    constexpr int kCtasPerToken = ctas_per_token(kInterPerTp);
    constexpr int kWeightScaleMBlocks = weight_scale_m_blocks(kInterPerTp);
    constexpr int kWeightStages = kPackedWeightStagesParam;
    constexpr bool kUsePackedTmaWeights = kPackedWeightLoadModeParam == kPackedLoadTma;

    int const M = static_cast<int>(scores.size(0));
    ffi::CUDADeviceGuard device_guard(scores.device().device_id);
    cudaStream_t stream = get_stream(scores.device());

    CUtensorMap expert_gate_up_tma = {};
    if constexpr (kUsePackedTmaWeights)
    {
        CUresult tma_err = CUDA_SUCCESS;
        int const num_tiles = (kNumExperts + 1) * kSubRowsPerExpert * kNumKIter;
        expert_gate_up_tma = get_cached_packed_fused_expert_up_tmap(
            expert_gate_up_weight.data_ptr(), num_tiles, scores.device().device_id, &tma_err);
        TVM_FFI_ICHECK(tma_err == CUDA_SUCCESS)
            << "cuTensorMapEncodeTiled for packed GLM5 gate/up weights failed: CUresult="
            << static_cast<int>(tma_err);
    }

    dim3 grid(static_cast<unsigned>(M), kCtasPerToken, 1);
    dim3 block(kThreadsPerCta, 1, 1);
    size_t const smem_bytes =
        fused_expert_up_smem_bytes<true, kPackedWeightStagesParam, kPackedWeightLoadModeParam>();

    int const device_id = scores.device().device_id;
    TVM_FFI_ICHECK(device_id >= 0 && device_id < kMaxCudaDevicesForSmemAttr)
        << "unsupported CUDA device id " << device_id;
    static std::once_flag smem_attribute_once[kMaxCudaDevicesForSmemAttr];
    std::call_once(smem_attribute_once[device_id], [&]()
    {
        cudaError_t err = cudaFuncSetAttribute(
            dsv3_fused_expert_up_kernel<kInterPerTpParam, true, kPackedWeightStagesParam,
                kPackedWeightLoadModeParam>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
        TVM_FFI_ICHECK(err == cudaSuccess)
            << "cudaFuncSetAttribute for GLM5 fused expert-up failed: "
            << cudaGetErrorString(err);
    });

    dsv3_fused_expert_up_kernel<kInterPerTpParam, true, kPackedWeightStagesParam,
        kPackedWeightLoadModeParam><<<grid, block, smem_bytes, stream>>>(
        expert_gate_up_tma, CUtensorMap{},
        static_cast<float const*>(scores.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(hidden_in.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3 const*>(expert_gate_up_weight.data_ptr()),
        static_cast<float const*>(expert_gate_up_scale.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3 const*>(expert_gate_up_weight.data_ptr()),
        static_cast<float const*>(expert_gate_up_scale.data_ptr()),
        static_cast<float*>(topk_weights.data_ptr()),
        static_cast<int32_t*>(topk_indices.data_ptr()),
        reinterpret_cast<__half*>(hidden_out.data_ptr()), M,
        static_cast<float>(routed_scaling_factor));

    cudaError_t launch_err = cudaPeekAtLastError();
    TVM_FFI_ICHECK(launch_err == cudaSuccess)
        << "GLM5 fused expert-up launch failed: " << cudaGetErrorString(launch_err);
}

} // anonymous namespace

namespace flashinfer::glm5
{

void Glm5FusedExpertUp(TensorView scores, TensorView hidden_in, TensorView bias,
    TensorView expert_gate_up_weight, TensorView expert_gate_up_scale, TensorView topk_weights,
    TensorView topk_indices, TensorView hidden_out, double routed_scaling_factor,
    int64_t packed_weight_stages, bool use_tma)
{
    CHECK_INPUT_AND_TYPE(scores, dl_float32);
    CHECK_INPUT_AND_TYPE(hidden_in, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(bias, dl_bfloat16);
    CHECK_INPUT_AND_TYPE(expert_gate_up_weight, dl_float8_e4m3fn);
    CHECK_INPUT_AND_TYPE(expert_gate_up_scale, dl_float32);
    CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
    CHECK_INPUT_AND_TYPE(topk_indices, dl_int32);
    CHECK_INPUT_AND_TYPE(hidden_out, dl_float16);

    CHECK_DEVICE(hidden_in, scores);
    CHECK_DEVICE(bias, scores);
    CHECK_DEVICE(expert_gate_up_weight, scores);
    CHECK_DEVICE(expert_gate_up_scale, scores);
    CHECK_DEVICE(topk_weights, scores);
    CHECK_DEVICE(topk_indices, scores);
    CHECK_DEVICE(hidden_out, scores);

    CHECK_DIM(2, scores);
    CHECK_DIM(2, hidden_in);
    CHECK_DIM(1, bias);
    CHECK_DIM(4, expert_gate_up_weight);
    CHECK_DIM(3, expert_gate_up_scale);
    CHECK_DIM(2, topk_weights);
    CHECK_DIM(2, topk_indices);
    CHECK_DIM(3, hidden_out);

    int64_t const M = scores.size(0);
    TVM_FFI_ICHECK(M >= 1 && M <= 4)
        << "GLM5 low-latency MoE supports 1 <= num_tokens <= 4, got " << M;
    TVM_FFI_ICHECK(scores.size(1) == kNumExperts)
        << "scores must have shape [M, 256]";
    TVM_FFI_ICHECK(hidden_in.size(0) == M && hidden_in.size(1) == kHidden)
        << "hidden_in must have shape [M, 6144]";
    TVM_FFI_ICHECK(bias.size(0) == kNumExperts)
        << "bias must have shape [256]";
    TVM_FFI_ICHECK(expert_gate_up_weight.size(0) == kNumExperts + 1 &&
        expert_gate_up_weight.size(2) == kNumKIter &&
        expert_gate_up_weight.size(3) == kCombinedTileBytes)
        << "packed gate/up weight must have shape [257, I/64, 8, 98304]";
    int64_t const inter_per_tp = expert_gate_up_weight.size(1) * kCtaOutRows;
    TVM_FFI_ICHECK(inter_per_tp == kInterPerTp_TP8 || inter_per_tp == kInterPerTp_TP4)
        << "local intermediate size must be 256 (TP8) or 512 (TP4), got "
        << inter_per_tp;
    TVM_FFI_ICHECK(expert_gate_up_scale.size(0) == kNumExperts + 1 &&
        expert_gate_up_scale.size(1) == 2 * (inter_per_tp / 128) &&
        expert_gate_up_scale.size(2) == kWeightScaleKBlocks)
        << "gate/up scale must have shape [257, 2 * I/128, 48]";
    TVM_FFI_ICHECK(topk_weights.size(0) == M && topk_weights.size(1) == kTopK)
        << "topk_weights must have shape [M, 8]";
    TVM_FFI_ICHECK(topk_indices.size(0) == M && topk_indices.size(1) == kTopK)
        << "topk_indices must have shape [M, 8]";
    TVM_FFI_ICHECK(hidden_out.size(0) == M && hidden_out.size(1) == kSlotsPerToken &&
        hidden_out.size(2) == inter_per_tp)
        << "hidden_out must have shape [M, 9, I]";
    TVM_FFI_ICHECK(std::isfinite(routed_scaling_factor) && routed_scaling_factor > 0.0)
        << "routed_scaling_factor must be positive and finite";
    TVM_FFI_ICHECK(packed_weight_stages == kPackedStagesSingle ||
        packed_weight_stages == kPackedStagesDouble)
        << "packed_weight_stages must be 1 or 2";

    if (inter_per_tp == kInterPerTp_TP8)
    {
        if (packed_weight_stages == kPackedStagesSingle)
        {
            if (use_tma)
                return glm5_fused_expert_up_impl<kInterPerTp_TP8, kPackedStagesSingle,
                    kPackedLoadTma>(scores, hidden_in, bias, expert_gate_up_weight,
                    expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
                    routed_scaling_factor);
            return glm5_fused_expert_up_impl<kInterPerTp_TP8, kPackedStagesSingle,
                kPackedLoadCpAsync>(scores, hidden_in, bias, expert_gate_up_weight,
                expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
                routed_scaling_factor);
        }
        if (use_tma)
            return glm5_fused_expert_up_impl<kInterPerTp_TP8, kPackedStagesDouble,
                kPackedLoadTma>(scores, hidden_in, bias, expert_gate_up_weight,
                expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
                routed_scaling_factor);
        return glm5_fused_expert_up_impl<kInterPerTp_TP8, kPackedStagesDouble,
            kPackedLoadCpAsync>(scores, hidden_in, bias, expert_gate_up_weight,
            expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
            routed_scaling_factor);
    }

    if (packed_weight_stages == kPackedStagesSingle)
    {
        if (use_tma)
            return glm5_fused_expert_up_impl<kInterPerTp_TP4, kPackedStagesSingle,
                kPackedLoadTma>(scores, hidden_in, bias, expert_gate_up_weight,
                expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
                routed_scaling_factor);
        return glm5_fused_expert_up_impl<kInterPerTp_TP4, kPackedStagesSingle,
            kPackedLoadCpAsync>(scores, hidden_in, bias, expert_gate_up_weight,
            expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
            routed_scaling_factor);
    }
    if (use_tma)
        return glm5_fused_expert_up_impl<kInterPerTp_TP4, kPackedStagesDouble,
            kPackedLoadTma>(scores, hidden_in, bias, expert_gate_up_weight,
            expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
            routed_scaling_factor);
    return glm5_fused_expert_up_impl<kInterPerTp_TP4, kPackedStagesDouble,
        kPackedLoadCpAsync>(scores, hidden_in, bias, expert_gate_up_weight,
        expert_gate_up_scale, topk_weights, topk_indices, hidden_out,
        routed_scaling_factor);
}

} // namespace flashinfer::glm5

TVM_FFI_DLL_EXPORT_TYPED_FUNC(glm5_fused_expert_up, flashinfer::glm5::Glm5FusedExpertUp);
