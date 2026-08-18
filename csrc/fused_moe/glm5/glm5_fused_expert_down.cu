/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// DeepSeek-V3 fused expert-down kernel.
//
// This kernel consumes fp16 expert slots from dsv3_fused_expert_up and computes
// shared_down(slot 0) + sum_i topk_weight_i * routed_down(slot i + 1) into
// the caller-provided bf16 output tensor. It supports M in [1, 4] and the
// DeepSeek-V3 TP layouts with K_local=512 (TP=4) or K_local=256 (TP=8).
//
// The deployed path uses packed down weights. The raw-weight path remains for
// diagnostics and uses TMA staging from the original row-major tensors.

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <list>
#include <mutex>
#include <unordered_map>

#include "tvm_ffi_utils.h"

namespace
{

constexpr int kHiddenSize = 6144;
constexpr int kTopKPlusShared = 9;
constexpr int kRoutedSlots = 8;
constexpr int kSharedExpertIdx = 256;
constexpr int kNumExpertsTotal = 257;

constexpr int kThreadsPerCta = 384;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kThreadsPerCta / kWarpSize;                // 12
constexpr int kNumCtas = 148;
constexpr int kRowsPerCta = (kHiddenSize + kNumCtas - 1) / kNumCtas; // 42

constexpr int kBlockK = 128;
constexpr int kBlockN = 128;

constexpr int kMmaM = 16;
constexpr int kMmaK = 16;
constexpr int kKItersPerBlock = kBlockK / kMmaK;                   // 8
constexpr int kRowTilesPerCta = (kRowsPerCta + kMmaM - 1) / kMmaM; // 3
constexpr int kPackedRowTiles = kNumCtas * kRowTilesPerCta;        // 444
// Smem buffers are sized for the maximum decode-token count this kernel supports.
constexpr int kMaxM = 4;

constexpr int kSpecFp8Stages = 4; // always 4
constexpr int kPackedFp8Stages = 2;

// fp8 TMA staging: 16 rows × 128 fp8 bytes = 2048 bytes per stage,
// 2 stages per warp.
constexpr int kWtRows = kMmaM;
constexpr int kWtKChunk = kBlockK;
constexpr int kFp8BytesPerStage = kWtRows * kWtKChunk; // 2048

// Per-K-block grouping for pre-folded scale. K_local=512 (TP=4) -> 4
// K-blocks -> 1 K-group with kKBlocksPerGroup=4.
// K_local=256 (TP=8) -> 2 K-blocks -> 1 K-group with kKBlocksPerGroup=2.
// Inside the templated kernel this is bound to (kKLocal / 128) (since the
// design always yields a single K-group per (N-block, expert)).
// The host binding enforces divisibility.

constexpr int kMaxRoutedPairs = kMaxM * kRoutedSlots; // 32
constexpr int kMaxBuckets = kMaxRoutedPairs;          // 32

constexpr int kHiddenKChunk = kBlockK;                // = 128
constexpr int kArElemsPerAccess = 8;
constexpr int kMaxArRanks = 16;

struct Dsv3LamportComm
{
    __device__ __forceinline__ Dsv3LamportComm(void** workspace, int nranks, int rank)
    {
        counter_ptr = &reinterpret_cast<int*>(workspace[nranks * 3])[0];
        flag_ptr = &reinterpret_cast<int*>(workspace[nranks * 3])[2];
        clear_ptr = &reinterpret_cast<int64_t*>(workspace[nranks * 3 + 1])[0];
        flag_value = *flag_ptr;
        int64_t const comm_size = reinterpret_cast<int64_t*>(workspace[nranks * 3 + 1])[1];
        clear_size = *clear_ptr;
        int const data_offset = flag_value % 3;
        int const clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < nranks && r < kMaxArRanks; ++r)
        {
            data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * nranks + r]) + data_offset * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * nranks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int64_t new_clear_size)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x)
            {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int* counter_ptr;
    int* flag_ptr;
    int64_t* clear_ptr;
    uint8_t* data_bufs[kMaxArRanks];
    uint8_t* clear_buf;
    int64_t clear_size;
    int flag_value;
};

__device__ __forceinline__ bool isNegZero(float v)
{
    return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool isNegZero(float4 v)
{
    return isNegZero(v.x) || isNegZero(v.y) || isNegZero(v.z) || isNegZero(v.w);
}

__device__ __forceinline__ float4 getNegZero()
{
    float4 vec;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
    }
    return vec;
}

__device__ __forceinline__ float4 ldGlobalVolatileFloat4(float4* addr)
{
    float4 val;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(addr));
    return val;
}

__device__ __forceinline__ void sanitizeArSentinel(float4& val)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (isNegZero(reinterpret_cast<float*>(&val)[i]))
        {
            reinterpret_cast<float*>(&val)[i] = 0.0f;
        }
    }
}

__device__ __forceinline__ float4 addBf16x8(float4 const& a, float4 const& b)
{
    union PackedBf16x8
    {
        float4 packed;
        __nv_bfloat162 unpacked[4];
    };

    PackedBf16x8 a_vec{a};
    PackedBf16x8 b_vec{b};
    PackedBf16x8 c_vec;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        c_vec.unpacked[i] = a_vec.unpacked[i] + b_vec.unpacked[i];
    }
    return c_vec.packed;
}

__device__ __forceinline__ float sumSquaresBf16x8(float4 const& v)
{
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kArElemsPerAccess; ++i)
    {
        float const x = __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(&v)[i]);
        sum += x * x;
    }
    return sum;
}

__device__ __forceinline__ float4 rmsNormBf16x8(float4 const& residual, float4 const& weight, float inv_rms)
{
    float4 out;
#pragma unroll
    for (int i = 0; i < kArElemsPerAccess; ++i)
    {
        float const x = __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(&residual)[i]);
        float const w = __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(&weight)[i]);
        reinterpret_cast<__nv_bfloat16*>(&out)[i] = __float2bfloat16(x * inv_rms * w);
    }
    return out;
}

__device__ __forceinline__ float warpReduceSum(float value)
{
    unsigned const mask = 0xffffffffu;
#pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

// -----------------------------------------------------------------------------
// fp8 (e4m3) -> fp16 conversion (single instr — was the lever-killer
// for bf16 on sm_103a, but fp16 path is supported).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint16_t fp8_pair)
{
    uint32_t out;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(out) : "h"(fp8_pair));
    return out;
}

__device__ __forceinline__ void prefetchGlobalL2(void const* ptr)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(ptr));
#else
    (void) ptr;
#endif
}

__device__ __forceinline__ uint4 loadGlobalUint4L2(uint4 const* ptr)
{
    uint4 value;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("ld.global.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr));
#else
    value = *ptr;
#endif
    return value;
}

__device__ __forceinline__ void cp_async_16b(void* smem_dst, void const* global_src)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    uint32_t const dst_smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem), "l"(global_src));
#else
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<uint4 const*>(global_src);
#endif
}

__device__ __forceinline__ void load_packed_w_down_tile(
    uint8_t* __restrict__ smem_tile, __nv_fp8_e4m3 const* __restrict__ packed_tile, int lane)
{
    constexpr int kCopyBytes = 16;
    uint8_t const* const src = reinterpret_cast<uint8_t const*>(packed_tile);
    for (int byte_off = lane * kCopyBytes; byte_off < kFp8BytesPerStage; byte_off += kWarpSize * kCopyBytes)
    {
        cp_async_16b(smem_tile + byte_off, src + byte_off);
    }
}

// -----------------------------------------------------------------------------
// HMMA.16816 f16xf16 -> fp32.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_f16(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, float& c0, float& c1, float& c2, float& c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// -----------------------------------------------------------------------------
// ldmatrix.sync.aligned.x2.b16.
//
// 2 8x8 b16 matrices. Lanes 0..7 supply rows of matrix 0; lanes 8..15
// supply rows of matrix 1; lanes 16..31 are ignored for addressing
// (but the address must still be a legal smem address).
//
// Output per lane (m16n8k16 B-frag, K-major B operand):
//   r0[T] = bf16x2 at matrix_0[row T/4, cols 2*(T%4)..+1]
//   r1[T] = bf16x2 at matrix_1[row T/4, cols 2*(T%4)..+1]
// -----------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x2_b16(uint32_t smem_addr, uint32_t& r0, uint32_t& r1)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

// -----------------------------------------------------------------------------
// mbarrier + cp.async.bulk.tensor.3d wrappers (TMA).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cvt_smem_addr(void const* smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, int arrive_count)
{
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(arrive_count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t bytes)
{
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase)
{
    uint32_t addr = cvt_smem_addr(mbar);
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
    uint32_t smem_addr = cvt_smem_addr(smem_dst);
    uint32_t mbar_addr = cvt_smem_addr(mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(smem_addr),
        "l"(tmap), "r"(coord_x), "r"(coord_y), "r"(coord_z), "r"(mbar_addr));
}

__device__ __forceinline__ void fence_proxy_async_shared()
{
    asm volatile("fence.proxy.async.shared::cta;\n" :::);
}

// -----------------------------------------------------------------------------
// Direct fp8 -> A-fragment loader. It fetches the exact fp8 bytes the
// m16n8k16 A-fragment requires and converts directly to fp16x2 registers.
//
// m16n8k16 A-fragment layout per lane:
//   a_frag[0] = bf16x2 at (row T/4,     cols 2*(T%4)..2*(T%4)+1)  -> 2 bf16
//   a_frag[1] = bf16x2 at (row T/4 + 8, cols 2*(T%4)..2*(T%4)+1)
//   a_frag[2] = bf16x2 at (row T/4,     cols 2*(T%4)+8..2*(T%4)+9)
//   a_frag[3] = bf16x2 at (row T/4 + 8, cols 2*(T%4)+8..2*(T%4)+9)
//
// Per-lane bytes needed per K-iteration:
//   p0_u16 (row_lo, byte col_lo..col_lo+1)
//   p2_u16 (row_lo, byte col_lo+8..col_lo+9)
//   p1_u16 (row_hi, byte col_lo..col_lo+1)
//   p3_u16 (row_hi, byte col_lo+8..col_lo+9)
//
// The kernel loads 2 LDS.U16 per row. Address fold:
//   base + row*kWtKChunk + ki_phys*kMmaK + col_lo
//   base + row*kWtKChunk + ki_phys*kMmaK + col_lo + 8
// -----------------------------------------------------------------------------
__device__ __forceinline__ void cvt_fp8_to_afrag_direct(
    uint32_t fp8_stage_smem, int ki, int lane, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3)
{
    int const row_lo = (lane >> 2);     // 0..7
    int const row_hi = row_lo + 8;      // 8..15
    int const col_lo = (lane & 3) << 1; // 0,2,4,6

    // TMA writes fp8 weight tiles with CU_TENSOR_MAP_SWIZZLE_128B.
    // For logical (row, chunk=ki), physical chunk = ki XOR (row & 7).
    int const ki_phys_lo = ki ^ (row_lo & 7);
    int const ki_phys_hi = ki ^ (row_hi & 7); // == ki_phys_lo

    // Use LDS.U16 with col_lo baked into the address.
    const uint32_t row_lo_base = fp8_stage_smem + (uint32_t) (row_lo * (int) kWtKChunk + ki_phys_lo * (int) kMmaK);
    const uint32_t row_hi_base = fp8_stage_smem + (uint32_t) (row_hi * (int) kWtKChunk + ki_phys_hi * (int) kMmaK);
    const uint32_t addr_p0 = row_lo_base + (uint32_t) col_lo;
    const uint32_t addr_p2 = row_lo_base + (uint32_t) col_lo + 8u;
    const uint32_t addr_p1 = row_hi_base + (uint32_t) col_lo;
    const uint32_t addr_p3 = row_hi_base + (uint32_t) col_lo + 8u;

    uint16_t p0_u16, p1_u16, p2_u16, p3_u16;
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p0_u16) : "r"(addr_p0));
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p2_u16) : "r"(addr_p2));
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p1_u16) : "r"(addr_p1));
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p3_u16) : "r"(addr_p3));

    a0 = fp8x2_to_f16x2(p0_u16);
    a1 = fp8x2_to_f16x2(p1_u16);
    a2 = fp8x2_to_f16x2(p2_u16);
    a3 = fp8x2_to_f16x2(p3_u16);
}

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------
template <int kKLocal, bool kUsePackedWeights, bool kEnableArResidual, bool kEnableRmsNorm>
__global__ __launch_bounds__(kThreadsPerCta, 1) void dsv3_fused_expert_down_kernel(
    __grid_constant__ const CUtensorMap routed_w_down_map, __grid_constant__ const CUtensorMap shared_w_down_map,
    __nv_fp8_e4m3 const* __restrict__ routed_w_down_packed, __nv_fp8_e4m3 const* __restrict__ shared_w_down_packed,
    // Read hidden_in via plain LDG (generic memory proxy) instead of TMA.
    // dsv3_fused_expert_up writes through normal global stores, so same-stream
    // kernel ordering is sufficient for this handoff without proxy fences.
    // The upstream kernel writes fp16 and the downstream MMA consumes fp16.
    __half const* __restrict__ hidden_in_raw, int32_t const* __restrict__ indices, float const* __restrict__ scores,
    float const* __restrict__ routed_w_down_scale, // [256, 48, k_blocks]
    float const* __restrict__ shared_w_down_scale, // [48, k_blocks]
    __nv_bfloat16* __restrict__ output, __nv_bfloat16 const* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16 const* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ hidden_out, float* __restrict__ rms_sums, void** workspace, int rank, int nranks, int M,
    float rms_norm_eps)
{
    // kMaxM sizes smem buffers; M bounds active-token loops.
    constexpr int kFp8Stages = kUsePackedWeights ? kPackedFp8Stages : kSpecFp8Stages;
    constexpr int K_local = kKLocal;            // 512 (TP=4) or 256 (TP=8)
    constexpr int kKBlocks = kKLocal / kBlockK; // 4 or 2

    int const cta_id = blockIdx.x;
    int const tid = threadIdx.x;
    int const warp_id = tid >> 5;
    int const lane = tid & 31;

    int const row_lo = cta_id * kRowsPerCta;
    bool const has_output_rows = row_lo < kHiddenSize;
    if (!has_output_rows && !kEnableArResidual)
    {
        return;
    }
    int const row_hi = min(row_lo + kRowsPerCta, kHiddenSize);
    int const rows_here = has_output_rows ? row_hi - row_lo : 0;

    constexpr int k_blocks = kKBlocks; // 4 (TP=4) or 2 (TP=8)

    extern __shared__ unsigned char smem_raw[];
    // Hidden slots are staged as fp16 because dsv3_fused_expert_up emits fp16.
    __half* smem_hidden = reinterpret_cast<__half*>(smem_raw);
    int const hidden_elems = M * kTopKPlusShared * K_local;
    const size_t hidden_bytes = sizeof(__half) * (size_t) hidden_elems;

    // Cooperatively warm L2 for the small up->down handoff before the per-CTA
    // shared-memory staging below. The buffer is at most 36 KiB for M <= 4.
    if (tid == 0)
    {
        constexpr int kPrefetchBytes = 128;
        int const hidden_prefetch_lines = static_cast<int>((hidden_bytes + kPrefetchBytes - 1) / kPrefetchBytes);
        char const* hidden_prefetch_base = reinterpret_cast<char const*>(hidden_in_raw);
        for (int line = cta_id; line < hidden_prefetch_lines; line += kNumCtas)
        {
            prefetchGlobalL2(hidden_prefetch_base + static_cast<size_t>(line) * kPrefetchBytes);
        }
        if (cta_id == 0)
        {
            prefetchGlobalL2(indices);
            prefetchGlobalL2(scores);
        }
    }

    int32_t* smem_expert_ids = reinterpret_cast<int32_t*>(smem_raw + hidden_bytes);
    float* smem_weights = reinterpret_cast<float*>(smem_expert_ids + M * kTopKPlusShared);

    const size_t partial_base
        = hidden_bytes + sizeof(int32_t) * (size_t) M * kTopKPlusShared + sizeof(float) * (size_t) M * kTopKPlusShared;
    float* smem_part = reinterpret_cast<float*>(smem_raw + partial_base);

    int const part_elems = kTopKPlusShared * kRowTilesPerCta * kMmaM * kMaxM;
    const size_t partial_bytes = sizeof(float) * (size_t) part_elems;

    // ---- Bucketing tables. ----
    size_t bucket_count_base = partial_base + partial_bytes;
    bucket_count_base = (bucket_count_base + 15) & ~size_t(15);
    int32_t* smem_bucket_count = reinterpret_cast<int32_t*>(smem_raw + bucket_count_base);

    size_t bucket_pairs_base = bucket_count_base + sizeof(int32_t) * (size_t) kNumExpertsTotal;
    bucket_pairs_base = (bucket_pairs_base + 15) & ~size_t(15);
    uint8_t* smem_bucket_pairs = reinterpret_cast<uint8_t*>(smem_raw + bucket_pairs_base);

    size_t unique_eid_base = bucket_pairs_base + (size_t) kNumExpertsTotal * kMaxM;
    unique_eid_base = (unique_eid_base + 15) & ~size_t(15);
    int16_t* smem_unique_eid = reinterpret_cast<int16_t*>(smem_raw + unique_eid_base);

    size_t num_unique_base = unique_eid_base + sizeof(int16_t) * (size_t) kMaxBuckets;
    num_unique_base = (num_unique_base + 15) & ~size_t(15);
    int32_t* smem_num_unique = reinterpret_cast<int32_t*>(smem_raw + num_unique_base);

    // ---- TMA mbarrier ring (one per (warp, stage) for W_down). 8B aligned. ----
    size_t mbar_base = num_unique_base + sizeof(int32_t) * 4;
    mbar_base = (mbar_base + 15) & ~size_t(15);
    uint64_t* smem_mbar = reinterpret_cast<uint64_t*>(smem_raw + mbar_base);
    constexpr int kWdMbarCount = kUsePackedWeights ? 0 : kNumWarps * kFp8Stages;
    constexpr int kMbarCount = kWdMbarCount;

    // ---- fp8 TMA W-tile staging.
    //
    // TMA SWIZZLE_128B requires 1024-byte alignment so the chunk-XOR pattern
    // is independent of M-dependent preceding smem allocations.
    size_t fp8_base = mbar_base + sizeof(uint64_t) * kMbarCount;
    fp8_base = (fp8_base + 1023) & ~size_t(1023);
    uint8_t* smem_fp8_stages = smem_raw + fp8_base;

    // lane>=16 fallback for ldmatrix.x2 B-frag uses warp_fp8_addr as a valid
    // smem address.

    // ---- Init mbarriers for the W_down ring. ----
    if constexpr (!kUsePackedWeights)
    {
        if (tid < kMbarCount)
        {
            mbarrier_init(&smem_mbar[tid], 1);
        }
        if (tid == 0)
        {
            fence_proxy_async_shared();
        }
    }
    __syncthreads();

    // ---- Stage hidden_in into SMEM through the generic memory proxy. ----
    {
        constexpr int kHalfElemsPerVec = sizeof(uint4) / sizeof(__half);
        constexpr int kHiddenVecsPerKChunk = kHiddenKChunk / kHalfElemsPerVec;
        constexpr int kKLocalVecs = K_local / kHalfElemsPerVec;
        static_assert(kHiddenKChunk % kHalfElemsPerVec == 0);
        static_assert(K_local % kHalfElemsPerVec == 0);
        int const chunk_elems = M * kTopKPlusShared * kHiddenKChunk;
        int const chunk_vecs = chunk_elems / kHalfElemsPerVec;
        uint4 const* hidden_src = reinterpret_cast<uint4 const*>(hidden_in_raw);
        uint4* hidden_dst = reinterpret_cast<uint4*>(smem_hidden);
#pragma unroll
        for (int chunk = 0; chunk < k_blocks; ++chunk)
        {
            uint4* dst_chunk = hidden_dst + (size_t) chunk * chunk_vecs;
            int const src_chunk_vec = chunk * kHiddenVecsPerKChunk;
            for (int chunk_vec = tid; chunk_vec < chunk_vecs; chunk_vec += kThreadsPerCta)
            {
                int const slot_vec = chunk_vec / kHiddenVecsPerKChunk;
                int const k_vec = chunk_vec - slot_vec * kHiddenVecsPerKChunk;
                int const m = slot_vec / kTopKPlusShared;
                int const slot = slot_vec - m * kTopKPlusShared;
                int const src_vec = (m * kTopKPlusShared + slot) * kKLocalVecs + src_chunk_vec + k_vec;

                dst_chunk[chunk_vec] = loadGlobalUint4L2(hidden_src + src_vec);
            }
        }
    }

    // ---- Build expert-id and weight tables [M, 9] in SMEM. ----
    {
        int const total_slots = M * kTopKPlusShared;
        for (int i = tid; i < total_slots; i += kThreadsPerCta)
        {
            int const m = i / kTopKPlusShared;
            int const s = i - m * kTopKPlusShared;
            int32_t eid;
            float w;
            if (s == 0)
            {
                eid = kSharedExpertIdx;
                w = 1.0f;
            }
            else
            {
                eid = indices[m * kRoutedSlots + (s - 1)];
                w = scores[m * kRoutedSlots + (s - 1)];
            }
            smem_expert_ids[i] = eid;
            smem_weights[i] = w;
        }
    }

    // ---- Initialise partial accumulator. ----
    for (int i = tid; i < part_elems; i += kThreadsPerCta)
    {
        smem_part[i] = 0.0f;
    }

    // ---- Init routed-dedup bucket tables. ----
    for (int i = tid; i < kNumExpertsTotal; i += kThreadsPerCta)
    {
        smem_bucket_count[i] = 0;
    }
    if (tid == 0)
    {
        *smem_num_unique = 0;
    }

    // Mbarriers are already initialised before weight TMA staging.
    __syncthreads();

    // ---- Bucket routed pairs by expert_id. ----
    {
        int const total_routed = M * kRoutedSlots;
        for (int i = tid; i < total_routed; i += kThreadsPerCta)
        {
            int const m = i / kRoutedSlots;
            int const s = i - m * kRoutedSlots;
            int const e_id = indices[m * kRoutedSlots + s];
            if (e_id < 0 || e_id >= kNumExpertsTotal)
                continue;

            uint8_t packed = static_cast<uint8_t>((m << 4) | (s + 1));

            int old_count = atomicAdd(&smem_bucket_count[e_id], 1);
            if (old_count < kMaxM)
            {
                smem_bucket_pairs[e_id * kMaxM + old_count] = packed;
            }
            if (old_count == 0)
            {
                int slot_idx = atomicAdd(smem_num_unique, 1);
                if (slot_idx < kMaxBuckets)
                {
                    smem_unique_eid[slot_idx] = static_cast<int16_t>(e_id);
                }
            }
        }
    }
    __syncthreads();

    // The previous __syncthreads waits for the LDG-based hidden staging,
    // table initialization, and bucket construction before any reader runs.

    int const num_unique = *smem_num_unique;

    // Per-warp mbarrier base index and fp8 stage base.
    int const warp_mbar_base = warp_id * kFp8Stages;
    uint8_t* warp_fp8_base = smem_fp8_stages + warp_id * (kFp8Stages * kFp8BytesPerStage);
    uint32_t warp_fp8_addr = cvt_smem_addr(warp_fp8_base);
    const uint32_t warp_mini_addr = warp_fp8_addr;

    // smem_hidden layout = [K_chunks, M, 9, 128] row-major (innermost = 128
    // fp16 elements). Per-(kb, m, slot) row offset in bytes is:
    //   kb * (M * 9 * 128 * 2)
    // + m  * (9 * 128 * 2)
    // + s  * (128 * 2)
    // The fp16 row stride for a single (kb, m, slot) is 128 elements *
    // 2 bytes = 256 bytes — much larger than a single ldmatrix row
    // (16 bytes = 8 fp16). ldmatrix.x2 only uses per-lane addresses
    // (not strides), so the wide fp16 row works fine as the row source.
    const uint32_t smem_hidden_addr = cvt_smem_addr(smem_hidden);
    // kb_stride depends on runtime M. m_stride and slot_stride remain compile-time.
    const uint32_t kb_stride_bytes = (uint32_t) (M * kTopKPlusShared * kHiddenKChunk * 2);  // M*9*128*2
    constexpr uint32_t m_stride_bytes_h = (uint32_t) (kTopKPlusShared * kHiddenKChunk * 2); // 9*128*2 = 2304
    constexpr uint32_t slot_stride_bytes_h = (uint32_t) (kHiddenKChunk * 2);                // 128*2 = 256

    // Weight issue helper. Raw weights use TMA maps; packed weights are already
    // in the exact 16x128 swizzled tile layout consumed by cvt_fp8_to_afrag_direct().
    auto issue_weight_load = [&](int e_id, int row_base, int row_tile_idx, int kb, int stage_idx)
    {
        if constexpr (kUsePackedWeights)
        {
            uint8_t* stage_smem = warp_fp8_base + stage_idx * kFp8BytesPerStage;
            __nv_fp8_e4m3 const* packed_tile = nullptr;
            if (e_id == kSharedExpertIdx)
            {
                packed_tile = shared_w_down_packed
                    + ((size_t) row_tile_idx * (size_t) k_blocks + (size_t) kb) * (size_t) kFp8BytesPerStage;
            }
            else
            {
                packed_tile = routed_w_down_packed
                    + (((size_t) e_id * (size_t) kPackedRowTiles + (size_t) row_tile_idx) * (size_t) k_blocks
                          + (size_t) kb)
                        * (size_t) kFp8BytesPerStage;
            }
            load_packed_w_down_tile(stage_smem, packed_tile, lane);
            asm volatile("cp.async.commit_group;\n" :::);
        }
        else
        {
            if (lane == 0)
            {
                int mbar_idx = warp_mbar_base + stage_idx;
                uint8_t* stage_smem = warp_fp8_base + stage_idx * kFp8BytesPerStage;
                int const k_off = kb * kBlockK;
                mbarrier_arrive_expect_tx(&smem_mbar[mbar_idx], kFp8BytesPerStage);
                if (e_id == kSharedExpertIdx)
                {
                    cp_async_bulk_tensor_3d(stage_smem, &shared_w_down_map,
                        /*x=*/k_off, /*y=*/row_base, /*z=*/0, &smem_mbar[mbar_idx]);
                }
                else
                {
                    cp_async_bulk_tensor_3d(stage_smem, &routed_w_down_map,
                        /*x=*/k_off, /*y=*/row_base, /*z=*/e_id, &smem_mbar[mbar_idx]);
                }
            }
        }
    };

    // Per-warp mbarrier phase tracking (toggled at each wait).
    // kFp8Stages now spans {1, 2, 3, 4} — size for max-of-template.
    constexpr int kMaxStages = 4;
    uint32_t mbar_phase[kMaxStages] = {0u, 0u, 0u, 0u};

    auto wait_stage = [&](int stage_idx)
    {
        int mbar_idx = warp_mbar_base + stage_idx;
        mbarrier_wait_parity(&smem_mbar[mbar_idx], mbar_phase[stage_idx]);
        mbar_phase[stage_idx] ^= 1u;
    };

    // ---- Phase A: shared-expert path. ----
    {
        int const shared_work = kRowTilesPerCta;
        for (int w = warp_id; w < shared_work; w += kNumWarps)
        {
            int const tile = w;
            int const row_base_in_cta = tile * kMmaM;
            int const row_base = row_lo + row_base_in_cta;
            int const row_tile_idx = cta_id * kRowTilesPerCta + tile;

            int const rows_active = min(kMmaM, row_hi - row_base);
            if (rows_active <= 0)
                continue;

            int const e_id = kSharedExpertIdx;
            // M <= 4 means a single M tile with m_base=0.
            constexpr int kNTilesM = 1;
#pragma unroll
            for (int nt = 0; nt < kNTilesM; ++nt)
            {
                constexpr int m_base = 0;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                if constexpr (kUsePackedWeights)
                {
                    issue_weight_load(e_id, row_base, row_tile_idx, 0, 0);
                    asm volatile("cp.async.wait_all;\n" :::);
                    __syncwarp();
                }
                else if constexpr (kFp8Stages > 1)
                {
                    int const prologue_n = kFp8Stages < k_blocks ? kFp8Stages : k_blocks;
#pragma unroll
                    for (int s = 0; s < kMaxStages; ++s)
                    {
                        if (s < prologue_n)
                        {
                            issue_weight_load(e_id, row_base, row_tile_idx, s, s);
                        }
                    }
                }

                // Fold the per-K-block scale inside the K-loop. Per-lane
                // row->n-block mapping is invariant over kb, so hoist it.
                int const row0_lane_pkb = row_base + (lane >> 2);
                int const row8_lane_pkb = row_base + (lane >> 2) + 8;
                int const nb0_pkb = row0_lane_pkb / kBlockN;
                int const nb8_pkb = row8_lane_pkb / kBlockN;
                // Shared scale shape: [48, k_blocks]. Hoist the per-n-block
                // base; per-kb scales live at consecutive memory.
                float const* const s0_base_pkb = shared_w_down_scale + (size_t) nb0_pkb * (size_t) k_blocks;
                float const* const s8_base_pkb
                    = (nb8_pkb == nb0_pkb) ? s0_base_pkb : (shared_w_down_scale + (size_t) nb8_pkb * (size_t) k_blocks);

#pragma unroll
                for (int kb = 0; kb < k_blocks; ++kb)
                {
                    int const stage = kUsePackedWeights ? (kb & 1) : ((kFp8Stages == 1) ? 0 : (kb % kFp8Stages));

                    if constexpr (kUsePackedWeights)
                    {
                        if (kb + 1 < k_blocks)
                        {
                            issue_weight_load(e_id, row_base, row_tile_idx, kb + 1, (kb + 1) & 1);
                        }
                    }
                    else if constexpr (kFp8Stages == 1)
                    {
                        // 1-stage: issue current then wait.
                        issue_weight_load(e_id, row_base, row_tile_idx, kb, 0);
                    }
                    if constexpr (!kUsePackedWeights)
                    {
                        wait_stage(stage);
                        __syncwarp();
                    }

                    uint32_t fp8_stage_ptr = warp_fp8_addr + (uint32_t) (stage * kFp8BytesPerStage);

                    float c_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
                    for (int ki = 0; ki < kKItersPerBlock; ++ki)
                    {
                        // Direct fp8 -> A-frag, no intermediate smem mini-buffer.
                        uint32_t a_frag[4];
                        cvt_fp8_to_afrag_direct(fp8_stage_ptr, ki, lane, a_frag[0], a_frag[1], a_frag[2], a_frag[3]);

                        // B-frag via ldmatrix.x2.b16 from smem layout [K_chunks, M, 9, 128].
                        // Lanes 0..7 supply mat0 rows (k=0..7 of ki);
                        // lanes 8..15 supply mat1 rows (k=8..15);
                        // lanes 16..31 use warp_mini_addr as a safe
                        // smem address (ignored by ldmatrix.x2).
                        uint32_t b_frag[2];
                        {
                            int const idx_in_row = lane & 7;                               // 0..7 (per-N row)
                            int const mat_id_b = (lane >> 3) & 1;                          // 0 or 1
                            int const m_for_lane = m_base + idx_in_row;
                            int const m_clamped = (m_for_lane < M) ? m_for_lane : (M - 1); // safe in-range
                            int const k_in_off = ki * kMmaK + mat_id_b * 8;
                            const uint32_t row_off = (uint32_t) kb * kb_stride_bytes
                                + (uint32_t) m_clamped * m_stride_bytes_h + 0u * slot_stride_bytes_h // shared slot=0
                                + (uint32_t) k_in_off * 2u;
                            uint32_t b_addr = (lane < 16) ? (smem_hidden_addr + row_off) : warp_mini_addr;
                            ldmatrix_x2_b16(b_addr, b_frag[0], b_frag[1]);
                        }

                        mma_m16n8k16_f16(a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag[0], b_frag[1], c_block[0],
                            c_block[1], c_block[2], c_block[3]);

                        // No intermediate mini-buffer, so no syncwarp is needed here.
                    }

                    // Steady-state: pre-issue (kb + kFp8Stages) into the
                    // just-consumed stage slot.
                    if constexpr (kUsePackedWeights)
                    {
                        if (kb + 1 < k_blocks)
                        {
                            asm volatile("cp.async.wait_all;\n" :::);
                            __syncwarp();
                        }
                    }
                    else if constexpr (kFp8Stages > 1)
                    {
                        int const next_kb = kb + kFp8Stages;
                        if (next_kb < k_blocks)
                        {
                            issue_weight_load(e_id, row_base, row_tile_idx, next_kb, stage);
                        }
                    }

                    // Per-K-block FFMA fold. Each K-block uses its raw source
                    // scale; fp32 accumulation preserves intermediate precision.
                    float const s0_kb = s0_base_pkb[kb];
                    float const s8_kb = (nb8_pkb == nb0_pkb) ? s0_kb : s8_base_pkb[kb];
                    c[0] += c_block[0] * s0_kb;
                    c[1] += c_block[1] * s0_kb;
                    c[2] += c_block[2] * s8_kb;
                    c[3] += c_block[3] * s8_kb;
                }

                {
                    int const row0 = (lane >> 2);
                    int const row1 = row0 + 8;
                    int const col0_local = (lane & 3) * 2;
                    int const col1_local = col0_local + 1;
                    int const col0 = m_base + col0_local;
                    int const col1 = m_base + col1_local;
                    int const slot_off = 0 * kRowTilesPerCta * kMmaM * kMaxM + tile * kMmaM * kMaxM;
                    if (col0 < M && row0 < rows_active)
                        smem_part[slot_off + row0 * kMaxM + col0] = c[0];
                    if (col1 < M && row0 < rows_active)
                        smem_part[slot_off + row0 * kMaxM + col1] = c[1];
                    if (col0 < M && row1 < rows_active)
                        smem_part[slot_off + row1 * kMaxM + col0] = c[2];
                    if (col1 < M && row1 < rows_active)
                        smem_part[slot_off + row1 * kMaxM + col1] = c[3];
                }
            }
        }
    }

    // ---- Phase B: routed-expert dedup path. ----
    {
        int const total_outer = kRowTilesPerCta * num_unique;
        for (int w_outer = warp_id; w_outer < total_outer; w_outer += kNumWarps)
        {
            int const tile = w_outer / num_unique;
            int const b_idx = w_outer - tile * num_unique;
            int const row_base_in_cta = tile * kMmaM;
            int const row_base = row_lo + row_base_in_cta;
            int const row_tile_idx = cta_id * kRowTilesPerCta + tile;
            int const rows_active = min(kMmaM, row_hi - row_base);
            if (rows_active <= 0)
                continue;

            int const e_id = smem_unique_eid[b_idx];
            int const count = static_cast<int>(smem_bucket_count[e_id]);
            int const n_groups_routed = (count + 7) >> 3; // 1 or 2

            for (int g = 0; g < n_groups_routed; ++g)
            {
                int const group_start = g * 8;
                int const group_count = min(8, count - group_start);

                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                // Pre-compute per-lane base offset in smem_hidden for
                // ldmatrix.x2.b16 B-frag loads.
                // Lanes 0..7 address mat0 rows (one pair each); lanes 8..15
                // address mat1 rows (same pair as lane-8); lanes 16..31
                // are unused — fall back to warp_mini_addr.
                //
                // Layout = [K_chunks, M, 9, 128]: the kb stride is added
                // later (per K-iter); here we collect M*9 base only.
                uint32_t b_row_base_bytes = 0u;
                bool lane_has_pair = false;
                {
                    int const pair_idx_in_group = lane & 7;
                    if (lane < 16 && pair_idx_in_group < group_count)
                    {
                        uint8_t packed = smem_bucket_pairs[e_id * kMaxM + group_start + pair_idx_in_group];
                        int const mm = (packed >> 4) & 0xF;
                        int const ss = packed & 0xF;
                        b_row_base_bytes = (uint32_t) mm * m_stride_bytes_h + (uint32_t) ss * slot_stride_bytes_h;
                        lane_has_pair = true;
                    }
                }
                if constexpr (kUsePackedWeights)
                {
                    issue_weight_load(e_id, row_base, row_tile_idx, 0, 0);
                    asm volatile("cp.async.wait_all;\n" :::);
                    __syncwarp();
                }
                else if constexpr (kFp8Stages > 1)
                {
                    int const prologue_n = kFp8Stages < k_blocks ? kFp8Stages : k_blocks;
#pragma unroll
                    for (int s = 0; s < kMaxStages; ++s)
                    {
                        if (s < prologue_n)
                        {
                            issue_weight_load(e_id, row_base, row_tile_idx, s, s);
                        }
                    }
                }

                // Fold routed per-K-block scales inside the K-loop.
                int const row0_lane_pkbR = row_base + (lane >> 2);
                int const row8_lane_pkbR = row_base + (lane >> 2) + 8;
                int const nb0_pkbR = row0_lane_pkbR / kBlockN;
                int const nb8_pkbR = row8_lane_pkbR / kBlockN;
                float const* const s0_base_pkbR
                    = routed_w_down_scale + ((size_t) e_id * (kHiddenSize / kBlockN) + nb0_pkbR) * (size_t) k_blocks;
                float const* const s8_base_pkbR = (nb8_pkbR == nb0_pkbR)
                    ? s0_base_pkbR
                    : (routed_w_down_scale + ((size_t) e_id * (kHiddenSize / kBlockN) + nb8_pkbR) * (size_t) k_blocks);

#pragma unroll
                for (int kb = 0; kb < k_blocks; ++kb)
                {
                    int const stage = kUsePackedWeights ? (kb & 1) : ((kFp8Stages == 1) ? 0 : (kb % kFp8Stages));

                    if constexpr (kUsePackedWeights)
                    {
                        if (kb + 1 < k_blocks)
                        {
                            issue_weight_load(e_id, row_base, row_tile_idx, kb + 1, (kb + 1) & 1);
                        }
                    }
                    else if constexpr (kFp8Stages == 1)
                    {
                        issue_weight_load(e_id, row_base, row_tile_idx, kb, 0);
                    }
                    if constexpr (!kUsePackedWeights)
                    {
                        wait_stage(stage);
                        __syncwarp();
                    }

                    uint32_t fp8_stage_ptr = warp_fp8_addr + (uint32_t) (stage * kFp8BytesPerStage);

                    float c_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
                    for (int ki = 0; ki < kKItersPerBlock; ++ki)
                    {
                        // Direct fp8 -> A-frag, no intermediate smem mini-buffer.
                        uint32_t a_frag[4];
                        cvt_fp8_to_afrag_direct(fp8_stage_ptr, ki, lane, a_frag[0], a_frag[1], a_frag[2], a_frag[3]);

                        // B-frag via ldmatrix.x2.b16 from smem [K_chunks, M, 9, 128].
                        uint32_t b_frag[2];
                        {
                            int const mat_id_b = (lane >> 3) & 1;
                            int const k_in_off = ki * kMmaK + mat_id_b * 8;
                            const uint32_t k_off_bytes = (uint32_t) kb * kb_stride_bytes + (uint32_t) k_in_off * 2u;
                            uint32_t b_addr
                                = lane_has_pair ? (smem_hidden_addr + b_row_base_bytes + k_off_bytes) : warp_mini_addr;
                            ldmatrix_x2_b16(b_addr, b_frag[0], b_frag[1]);
                        }

                        mma_m16n8k16_f16(a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag[0], b_frag[1], c_block[0],
                            c_block[1], c_block[2], c_block[3]);
                        // No intermediate mini-buffer, so no syncwarp is needed here.
                    }

                    // Steady-state: pre-issue (kb + kFp8Stages) into the
                    // freed stage slot.
                    if constexpr (kUsePackedWeights)
                    {
                        if (kb + 1 < k_blocks)
                        {
                            asm volatile("cp.async.wait_all;\n" :::);
                            __syncwarp();
                        }
                    }
                    else if constexpr (kFp8Stages > 1)
                    {
                        int const next_kb = kb + kFp8Stages;
                        if (next_kb < k_blocks)
                        {
                            issue_weight_load(e_id, row_base, row_tile_idx, next_kb, stage);
                        }
                    }

                    // Per-K-block FFMA fold.
                    float const s0_kbR = s0_base_pkbR[kb];
                    float const s8_kbR = (nb8_pkbR == nb0_pkbR) ? s0_kbR : s8_base_pkbR[kb];
                    c[0] += c_block[0] * s0_kbR;
                    c[1] += c_block[1] * s0_kbR;
                    c[2] += c_block[2] * s8_kbR;
                    c[3] += c_block[3] * s8_kbR;
                }

                {
                    int const row0 = (lane >> 2);
                    int const row1 = row0 + 8;
                    int const p0_local = (lane & 3) * 2;
                    int const p1_local = p0_local + 1;
                    int const p0 = group_start + p0_local;
                    int const p1 = group_start + p1_local;

                    auto write_cell = [&](int p, int row_in_tile, float val)
                    {
                        if (p < count && row_in_tile < rows_active)
                        {
                            uint8_t packed = smem_bucket_pairs[e_id * kMaxM + p];
                            const int m = (packed >> 4) & 0xF;
                            const int s = packed & 0xF;
                            const int off
                                = s * kRowTilesPerCta * kMmaM * kMaxM + tile * kMmaM * kMaxM + row_in_tile * kMaxM + m;
                            smem_part[off] = val;
                        }
                    };
                    write_cell(p0, row0, c[0]);
                    write_cell(p1, row0, c[1]);
                    write_cell(p0, row1, c[2]);
                    write_cell(p1, row1, c[3]);
                }
            }
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------
    // Final local reduction. The original branch kernel added residual and
    // then all-reduced across TP peers here. This local-development variant
    // stores only this rank's weighted shared+routed down projection so the
    // existing post-MoE allreduce path can handle cross-rank reduction.
    // -------------------------------------------------------------------
    int const rows_pairs = rows_here >> 1;
    int const tail_rows = rows_here - (rows_pairs << 1);

    int const pair_cells = rows_pairs * M;

    // Each thread owns at most one pair-cell (pair_cells <= 336 < 384).
    bool const has_pair_cell = (tid < pair_cells);

    if (has_pair_cell)
    {
        int const cell = tid;
        int const pair_idx_in_cta = cell / M;
        int const m = cell - pair_idx_in_cta * M;
        int const row_in_cta0 = pair_idx_in_cta << 1;
        int const row_in_cta1 = row_in_cta0 + 1;
        int const row0 = row_lo + row_in_cta0;
        int const row1 = row0 + 1;

        int const tile0 = row_in_cta0 / kMmaM;
        int const rit0 = row_in_cta0 - tile0 * kMmaM;
        int const tile1 = row_in_cta1 / kMmaM;
        int const rit1 = row_in_cta1 - tile1 * kMmaM;

        float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
        for (int s = 0; s < kTopKPlusShared; ++s)
        {
            float const w = smem_weights[m * kTopKPlusShared + s];
            int const off0 = s * kRowTilesPerCta * kMmaM * kMaxM + tile0 * kMmaM * kMaxM + rit0 * kMaxM + m;
            int const off1 = s * kRowTilesPerCta * kMmaM * kMaxM + tile1 * kMmaM * kMaxM + rit1 * kMaxM + m;
            acc0 += smem_part[off0] * w;
            acc1 += smem_part[off1] * w;
        }
        output[(size_t) m * kHiddenSize + row0] = __float2bfloat16(acc0);
        output[(size_t) m * kHiddenSize + row1] = __float2bfloat16(acc1);
    }

    // Tail-row owner (at most one per thread since M ≤ kMaxM=16 < 384).
    bool const has_tail_cell = (tail_rows > 0) && (tid < M);

    if (has_tail_cell)
    {
        int const row_in_cta = rows_here - 1;
        int const row = row_lo + row_in_cta;
        int const tile = row_in_cta / kMmaM;
        int const rit = row_in_cta - tile * kMmaM;
        int const m = tid;

        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < kTopKPlusShared; ++s)
        {
            int const off = s * kRowTilesPerCta * kMmaM * kMaxM + tile * kMmaM * kMaxM + rit * kMaxM + m;
            acc += smem_part[off] * smem_weights[m * kTopKPlusShared + s];
        }
        output[(size_t) m * kHiddenSize + row] = __float2bfloat16(acc);
    }

    if constexpr (kEnableArResidual)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        namespace cg = cooperative_groups;
        int const total_access = (M * kHiddenSize) / kArElemsPerAccess;
        int const linear_tid = blockIdx.x * blockDim.x + threadIdx.x;
        float4 const clear_vec = getNegZero();

        float4* output_vec = reinterpret_cast<float4*>(output);
        float4 const* residual_vec = reinterpret_cast<float4 const*>(residual);
        float4* residual_out_vec = reinterpret_cast<float4*>(residual_out);
        float4 const* norm_weight_vec = reinterpret_cast<float4 const*>(norm_weight);
        float4* hidden_out_vec = reinterpret_cast<float4*>(hidden_out);

        if constexpr (kEnableRmsNorm)
        {
            if (linear_tid < M)
            {
                rms_sums[linear_tid] = 0.0f;
            }
        }

        cg::grid_group grid = cg::this_grid();
        grid.sync();

        Dsv3LamportComm comm(workspace, nranks, rank);
        int const clear_access = static_cast<int>(comm.clear_size / kArElemsPerAccess);
        bool const has_ar_idx = linear_tid < total_access;
        float4 residual_sum = {};

        if (has_ar_idx)
        {
            float4 val = output_vec[linear_tid];
            sanitizeArSentinel(val);
            for (int r = 0; r < nranks; ++r)
            {
                reinterpret_cast<float4*>(comm.data_bufs[r])[rank * total_access + linear_tid] = val;
            }
        }
        if (linear_tid < clear_access)
        {
            reinterpret_cast<float4*>(comm.clear_buf)[linear_tid] = clear_vec;
        }

        if (has_ar_idx)
        {
            bool done = false;
            float4 sum_val = {};
            int const chunk_access = (total_access + nranks - 1) / nranks;
            int const chunk_owner = min(linear_tid / chunk_access, nranks - 1);
            while (!done)
            {
                done = true;
                sum_val = {};
                for (int step = 0; step < nranks; ++step)
                {
                    int const r = (chunk_owner + 1 + step) % nranks;
                    float4 const peer_val = ldGlobalVolatileFloat4(
                        &reinterpret_cast<float4*>(comm.data_bufs[rank])[r * total_access + linear_tid]);
                    done &= !isNegZero(peer_val);
                    sum_val = (step == 0) ? peer_val : addBf16x8(sum_val, peer_val);
                }
            }
            residual_sum = addBf16x8(sum_val, residual_vec[linear_tid]);
            residual_out_vec[linear_tid] = residual_sum;
        }

        if constexpr (kEnableRmsNorm)
        {
            float local_square_sum = has_ar_idx ? sumSquaresBf16x8(residual_sum) : 0.0f;
            local_square_sum = warpReduceSum(local_square_sum);
            float* smem_warp_sums = reinterpret_cast<float*>(smem_raw);
            if (lane == 0)
            {
                smem_warp_sums[warp_id] = local_square_sum;
            }
            __syncthreads();
            if (warp_id == 0)
            {
                float block_square_sum = (lane < kNumWarps) ? smem_warp_sums[lane] : 0.0f;
                block_square_sum = warpReduceSum(block_square_sum);
                if (lane == 0 && blockIdx.x * blockDim.x < total_access)
                {
                    int const token = (blockIdx.x * blockDim.x * kArElemsPerAccess) / kHiddenSize;
                    atomicAdd(&rms_sums[token], block_square_sum);
                }
            }
            grid.sync();
        }
        comm.update(static_cast<int64_t>(M) * static_cast<int64_t>(kHiddenSize) * static_cast<int64_t>(nranks));

        if constexpr (kEnableRmsNorm)
        {
            int const hidden_access_per_token = kHiddenSize / kArElemsPerAccess;
            if (has_ar_idx)
            {
                int const token = linear_tid / hidden_access_per_token;
                int const hidden_access = linear_tid - token * hidden_access_per_token;
                float const inv_rms = rsqrtf(rms_sums[token] / static_cast<float>(kHiddenSize) + rms_norm_eps);
                hidden_out_vec[linear_tid] = rmsNormBf16x8(residual_sum, norm_weight_vec[hidden_access], inv_rms);
            }
        }
#else
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            asm("trap;");
        }
#endif
    }
}

// -----------------------------------------------------------------------------
// CUtensorMap build helper.
// -----------------------------------------------------------------------------
// W_down has shape [E, N=6144, K=K_local], dtype = fp8_e4m3 = 1 byte.
//   * dim 0 (x = K) — element stride 1 byte
//   * dim 1 (y = N rows)
//   * dim 2 (z = E experts)
// Box dim per tile = [kBlockK, kMmaM, 1] = [128, 16, 1].
// SWIZZLE_128B mode uses 128-byte rows and 16 rows, giving 2 swizzle
// periods of 8 rows each. For each row r, the 8 16-byte chunks are permuted
// by chunk_phys = chunk_logical XOR (r & 7).
static CUtensorMap make_w_down_tmap(void* base_ptr, int num_experts, int K_local, CUresult* out_err)
{
    CUtensorMap map = {};
    cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(K_local),
        static_cast<cuuint64_t>(kHiddenSize),
        static_cast<cuuint64_t>(num_experts),
    };
    cuuint64_t global_stride[2] = {
        static_cast<cuuint64_t>(K_local),
        static_cast<cuuint64_t>(K_local) * static_cast<cuuint64_t>(kHiddenSize),
    };
    cuuint32_t box_dim[3] = {
        static_cast<cuuint32_t>(kBlockK),
        static_cast<cuuint32_t>(kMmaM),
        1u,
    };
    cuuint32_t elem_stride[3] = {1u, 1u, 1u};

    *out_err = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /*rank=*/3, base_ptr, global_dim, global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

constexpr size_t kWDownTmaDescCacheCap = 256;
constexpr int kMaxCudaDevicesForSmemAttr = 16;

struct WDownTmaDescKey
{
    void const* base;
    int numExperts;
    int kLocal;
    int deviceId;

    bool operator==(WDownTmaDescKey const& o) const noexcept
    {
        return base == o.base && numExperts == o.numExperts && kLocal == o.kLocal && deviceId == o.deviceId;
    }
};

struct WDownTmaDescKeyHash
{
    size_t operator()(WDownTmaDescKey const& k) const noexcept
    {
        size_t h = reinterpret_cast<uintptr_t>(k.base);
        h = h * 1099511628211ull + static_cast<size_t>(k.numExperts);
        h = h * 1099511628211ull + static_cast<size_t>(k.kLocal);
        h = h * 1099511628211ull + static_cast<size_t>(k.deviceId);
        return h;
    }
};

struct WDownTmaDescCache
{
    using ListIt = std::list<std::pair<WDownTmaDescKey, CUtensorMap>>::iterator;
    std::list<std::pair<WDownTmaDescKey, CUtensorMap>> order;
    std::unordered_map<WDownTmaDescKey, ListIt, WDownTmaDescKeyHash> index;
};

static CUtensorMap get_cached_w_down_tmap(
    void* base_ptr, int num_experts, int K_local, int device_id, CUresult* out_err)
{
    static thread_local WDownTmaDescCache cache;
    WDownTmaDescKey const key{base_ptr, num_experts, K_local, device_id};
    auto it = cache.index.find(key);
    if (it != cache.index.end())
    {
        cache.order.splice(cache.order.begin(), cache.order, it->second);
        *out_err = CUDA_SUCCESS;
        return it->second->second;
    }

    CUtensorMap const map = make_w_down_tmap(base_ptr, num_experts, K_local, out_err);
    if (*out_err != CUDA_SUCCESS)
    {
        return map;
    }
    if (cache.order.size() >= kWDownTmaDescCacheCap)
    {
        cache.index.erase(cache.order.back().first);
        cache.order.pop_back();
    }
    cache.order.emplace_front(key, map);
    cache.index.emplace(key, cache.order.begin());
    return map;
}

// -----------------------------------------------------------------------------
} // anonymous namespace

namespace flashinfer::glm5
{

void Glm5FusedExpertDown(TensorView hidden_in, TensorView indices, TensorView scores,
    TensorView routed_w_down, TensorView routed_w_down_scale, TensorView shared_w_down,
    TensorView shared_w_down_scale, TensorView output)
{
    CHECK_INPUT_AND_TYPE(hidden_in, dl_float16);
    CHECK_INPUT_AND_TYPE(indices, dl_int32);
    CHECK_INPUT_AND_TYPE(scores, dl_float32);
    CHECK_INPUT_AND_TYPE(routed_w_down, dl_float8_e4m3fn);
    CHECK_INPUT_AND_TYPE(routed_w_down_scale, dl_float32);
    CHECK_INPUT_AND_TYPE(shared_w_down, dl_float8_e4m3fn);
    CHECK_INPUT_AND_TYPE(shared_w_down_scale, dl_float32);
    CHECK_INPUT_AND_TYPE(output, dl_bfloat16);

    CHECK_DEVICE(indices, hidden_in);
    CHECK_DEVICE(scores, hidden_in);
    CHECK_DEVICE(routed_w_down, hidden_in);
    CHECK_DEVICE(routed_w_down_scale, hidden_in);
    CHECK_DEVICE(shared_w_down, hidden_in);
    CHECK_DEVICE(shared_w_down_scale, hidden_in);
    CHECK_DEVICE(output, hidden_in);

    CHECK_DIM(3, hidden_in);
    CHECK_DIM(2, indices);
    CHECK_DIM(2, scores);
    CHECK_DIM(3, routed_w_down_scale);
    CHECK_DIM(2, shared_w_down_scale);
    CHECK_DIM(2, output);

    int const M = static_cast<int>(hidden_in.size(0));
    int const K_local = static_cast<int>(hidden_in.size(2));
    bool const use_packed_weights = routed_w_down.ndim() == 4 || shared_w_down.ndim() == 3;

    TVM_FFI_ICHECK(M >= 1 && M <= kMaxM)
        << "GLM5 fused expert-down supports 1 <= num_tokens <= " << kMaxM
        << ", got " << M;
    TVM_FFI_ICHECK(hidden_in.size(1) == kTopKPlusShared)
        << "hidden_in must have shape [M, 9, I]";
    TVM_FFI_ICHECK(indices.size(0) == M && indices.size(1) == kRoutedSlots)
        << "indices must have shape [M, 8]";
    TVM_FFI_ICHECK(scores.size(0) == M && scores.size(1) == kRoutedSlots)
        << "scores must have shape [M, 8]";
    TVM_FFI_ICHECK(K_local == 256 || K_local == 512)
        << "local intermediate size must be 256 (TP8) or 512 (TP4), got "
        << K_local;
    TVM_FFI_ICHECK(output.size(0) == M && output.size(1) == kHiddenSize)
        << "output must have shape [M, 6144]";

    int const k_blocks = K_local / kBlockK;
    if (use_packed_weights)
    {
        TVM_FFI_ICHECK(routed_w_down.ndim() == 4 &&
            routed_w_down.size(0) == kSharedExpertIdx &&
            routed_w_down.size(1) == kPackedRowTiles &&
            routed_w_down.size(2) == k_blocks &&
            routed_w_down.size(3) == kFp8BytesPerStage)
            << "packed routed_w_down must have shape [256, 444, I/128, 2048]";
        TVM_FFI_ICHECK(shared_w_down.ndim() == 3 &&
            shared_w_down.size(0) == kPackedRowTiles &&
            shared_w_down.size(1) == k_blocks &&
            shared_w_down.size(2) == kFp8BytesPerStage)
            << "packed shared_w_down must have shape [444, I/128, 2048]";
    }
    else
    {
        TVM_FFI_ICHECK(routed_w_down.ndim() == 3 &&
            routed_w_down.size(0) == kSharedExpertIdx &&
            routed_w_down.size(1) == kHiddenSize &&
            routed_w_down.size(2) == K_local)
            << "routed_w_down must have shape [256, 6144, I]";
        TVM_FFI_ICHECK(shared_w_down.ndim() == 2 &&
            shared_w_down.size(0) == kHiddenSize &&
            shared_w_down.size(1) == K_local)
            << "shared_w_down must have shape [6144, I]";
    }
    TVM_FFI_ICHECK(routed_w_down_scale.size(0) == kSharedExpertIdx &&
        routed_w_down_scale.size(1) == kHiddenSize / kBlockN &&
        routed_w_down_scale.size(2) == k_blocks)
        << "routed_w_down_scale must have shape [256, 48, I/128]";
    TVM_FFI_ICHECK(shared_w_down_scale.size(0) == kHiddenSize / kBlockN &&
        shared_w_down_scale.size(1) == k_blocks)
        << "shared_w_down_scale must have shape [48, I/128]";

    ffi::CUDADeviceGuard device_guard(hidden_in.device().device_id);
    cudaStream_t stream = get_stream(hidden_in.device());
    int const device_id = hidden_in.device().device_id;

    CUtensorMap routed_w_down_map = {};
    CUtensorMap shared_w_down_map = {};
    if (!use_packed_weights)
    {
        CUresult routed_tma_err = CUDA_SUCCESS;
        routed_w_down_map = get_cached_w_down_tmap(
            routed_w_down.data_ptr(), kSharedExpertIdx, K_local, device_id, &routed_tma_err);
        TVM_FFI_ICHECK(routed_tma_err == CUDA_SUCCESS)
            << "cuTensorMapEncodeTiled for routed down weights failed: CUresult="
            << static_cast<int>(routed_tma_err);
        CUresult shared_tma_err = CUDA_SUCCESS;
        shared_w_down_map = get_cached_w_down_tmap(
            shared_w_down.data_ptr(), 1, K_local, device_id, &shared_tma_err);
        TVM_FFI_ICHECK(shared_tma_err == CUDA_SUCCESS)
            << "cuTensorMapEncodeTiled for shared down weights failed: CUresult="
            << static_cast<int>(shared_tma_err);
    }

    int const chosen_stages = use_packed_weights ? kPackedFp8Stages : kSpecFp8Stages;
    auto compute_smem = [&](int stages, int m_for_smem, bool packed_weights) -> size_t
    {
        size_t hidden_bytes =
            sizeof(__half) * static_cast<size_t>(m_for_smem) * kTopKPlusShared * K_local;
        size_t tables_bytes =
            sizeof(int32_t) * static_cast<size_t>(m_for_smem) * kTopKPlusShared +
            sizeof(float) * static_cast<size_t>(m_for_smem) * kTopKPlusShared;
        size_t partial_bytes =
            sizeof(float) * static_cast<size_t>(kTopKPlusShared) * kRowTilesPerCta *
            kMmaM * kMaxM;

        size_t bucket_count_base = hidden_bytes + tables_bytes + partial_bytes;
        bucket_count_base = (bucket_count_base + 15) & ~size_t(15);
        size_t bucket_count_bytes = sizeof(int32_t) * static_cast<size_t>(kNumExpertsTotal);

        size_t bucket_pairs_base = bucket_count_base + bucket_count_bytes;
        bucket_pairs_base = (bucket_pairs_base + 15) & ~size_t(15);
        size_t bucket_pairs_bytes = static_cast<size_t>(kNumExpertsTotal) * kMaxM;

        size_t unique_eid_base = bucket_pairs_base + bucket_pairs_bytes;
        unique_eid_base = (unique_eid_base + 15) & ~size_t(15);
        size_t unique_eid_bytes = sizeof(int16_t) * static_cast<size_t>(kMaxBuckets);

        size_t num_unique_base = unique_eid_base + unique_eid_bytes;
        num_unique_base = (num_unique_base + 15) & ~size_t(15);
        size_t num_unique_bytes = sizeof(int32_t) * 4;

        size_t mbarrier_base = num_unique_base + num_unique_bytes;
        mbarrier_base = (mbarrier_base + 15) & ~size_t(15);
        size_t mbarrier_bytes =
            packed_weights ? 0 : sizeof(uint64_t) * static_cast<size_t>(kNumWarps * stages);
        size_t fp8_base = mbarrier_base + mbarrier_bytes;
        fp8_base = (fp8_base + 1023) & ~size_t(1023);
        size_t fp8_bytes =
            static_cast<size_t>(kNumWarps) * stages * kFp8BytesPerStage;
        return fp8_base + fp8_bytes;
    };

    size_t const smem_bytes = compute_smem(chosen_stages, M, use_packed_weights);
    size_t const max_smem_bytes =
        compute_smem(chosen_stages, kMaxM, use_packed_weights);
    constexpr size_t kSmemCapBytes = 232448;
    TVM_FFI_ICHECK(max_smem_bytes <= kSmemCapBytes)
        << "GLM5 fused expert-down shared-memory footprint " << max_smem_bytes
        << " exceeds the B200 cap " << kSmemCapBytes;

    using KernelFn = void (*)(const CUtensorMap, const CUtensorMap,
        __nv_fp8_e4m3 const*, __nv_fp8_e4m3 const*, __half const*,
        int32_t const*, float const*, float const*, float const*,
        __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16*,
        __nv_bfloat16 const*, __nv_bfloat16*, float*, void**, int, int, int, float);

    KernelFn kernel = nullptr;
    if (K_local == 512)
        kernel = use_packed_weights
            ? &dsv3_fused_expert_down_kernel<512, true, false, false>
            : &dsv3_fused_expert_down_kernel<512, false, false, false>;
    else
        kernel = use_packed_weights
            ? &dsv3_fused_expert_down_kernel<256, true, false, false>
            : &dsv3_fused_expert_down_kernel<256, false, false, false>;

    TVM_FFI_ICHECK(device_id >= 0 && device_id < kMaxCudaDevicesForSmemAttr)
        << "unsupported CUDA device id " << device_id;
    static std::once_flag smem_512[kMaxCudaDevicesForSmemAttr];
    static std::once_flag smem_256[kMaxCudaDevicesForSmemAttr];
    static std::once_flag smem_512_packed[kMaxCudaDevicesForSmemAttr];
    static std::once_flag smem_256_packed[kMaxCudaDevicesForSmemAttr];
    std::once_flag& smem_flag = K_local == 512
        ? (use_packed_weights ? smem_512_packed[device_id] : smem_512[device_id])
        : (use_packed_weights ? smem_256_packed[device_id] : smem_256[device_id]);
    std::call_once(smem_flag, [&]()
    {
        if (max_smem_bytes > 48 * 1024)
        {
            cudaError_t err = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(max_smem_bytes));
            TVM_FFI_ICHECK(err == cudaSuccess)
                << "cudaFuncSetAttribute for GLM5 fused expert-down failed: "
                << cudaGetErrorString(err);
        }
    });

    __nv_fp8_e4m3 const* routed_packed_ptr = use_packed_weights
        ? reinterpret_cast<__nv_fp8_e4m3 const*>(routed_w_down.data_ptr()) : nullptr;
    __nv_fp8_e4m3 const* shared_packed_ptr = use_packed_weights
        ? reinterpret_cast<__nv_fp8_e4m3 const*>(shared_w_down.data_ptr()) : nullptr;
    __half const* hidden_ptr = reinterpret_cast<__half const*>(hidden_in.data_ptr());
    int32_t const* indices_ptr = static_cast<int32_t const*>(indices.data_ptr());
    float const* scores_ptr = static_cast<float const*>(scores.data_ptr());
    float const* routed_scale_ptr =
        static_cast<float const*>(routed_w_down_scale.data_ptr());
    float const* shared_scale_ptr =
        static_cast<float const*>(shared_w_down_scale.data_ptr());
    __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    __nv_bfloat16 const* null_bf16_const = nullptr;
    __nv_bfloat16* null_bf16 = nullptr;
    float* null_float = nullptr;
    void** null_workspace = nullptr;
    int rank = 0;
    int nranks = 1;
    int m_arg = M;
    float rms_norm_eps = 0.0f;

    void* args[] = {
        &routed_w_down_map, &shared_w_down_map, &routed_packed_ptr,
        &shared_packed_ptr, &hidden_ptr, &indices_ptr, &scores_ptr,
        &routed_scale_ptr, &shared_scale_ptr, &output_ptr,
        &null_bf16_const, &null_bf16, &null_bf16_const, &null_bf16,
        &null_float, &null_workspace, &rank, &nranks, &m_arg, &rms_norm_eps,
    };

    dim3 grid(kNumCtas, 1, 1);
    dim3 block(kThreadsPerCta, 1, 1);
    cudaError_t launch_err = cudaLaunchKernel(
        reinterpret_cast<void const*>(kernel), grid, block, args, smem_bytes, stream);
    TVM_FFI_ICHECK(launch_err == cudaSuccess)
        << "GLM5 fused expert-down launch failed: " << cudaGetErrorString(launch_err);
    launch_err = cudaPeekAtLastError();
    TVM_FFI_ICHECK(launch_err == cudaSuccess)
        << "GLM5 fused expert-down reported a CUDA error: "
        << cudaGetErrorString(launch_err);
}

} // namespace flashinfer::glm5

TVM_FFI_DLL_EXPORT_TYPED_FUNC(glm5_fused_expert_down, flashinfer::glm5::Glm5FusedExpertDown);
