/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAP_RING_STAGES 2
#define SMEM_SMEM_M_OFF 1024
#define SMEM_SMEM_M_STAGE_BYTES 32768
#define SMEM_SMEM_M_STRIDE 32768
#define SMEM_SMEM_CARRY_OFF 66560
#define SMEM_SMEM_CARRY_STAGE_BYTES 8192
#define SMEM_SMEM_CARRY_STRIDE 8192
#define SMEM_TOTAL 74752
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_prefill_d5ad43b36f4231ea0531(float* __restrict__ split_state, __nv_bfloat16* __restrict__ map_state_bf16, float* __restrict__ carry, __nv_bfloat16* __restrict__ carry_hi, __nv_bfloat16* __restrict__ carry_lo, float* __restrict__ final_state, int write_final_state, int num_heads, int* __restrict__ part_cu_seqlens, int* __restrict__ final_indices, long long final_slot_stride, __nv_bfloat16* __restrict__ final_state_bf16, float* __restrict__ rows, long long* __restrict__ row_starts, int write_rows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define map_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_m = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_m_addr = smem + 1024;
    float* smem_carry = reinterpret_cast<float*>(smem_raw + 66560);
    const int smem_carry_addr = smem + 66560;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'map_ring' ---
            // map_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int sequence = bid / 11 / num_heads;
    int part_begin = part_cu_seqlens[sequence];
    int part_stop = part_cu_seqlens[sequence + 1];
    if (num_bids > num_heads * 11 && num_bids <= num_heads * 44) {
        int seq_count = num_bids / (num_heads * 11);
        int place = sequence;
        int bounds[5];
        bounds[0] = 0;
        if (seq_count >= 0) {
            bounds[0] = part_cu_seqlens[0];
        }
        bounds[1] = 0;
        if (seq_count >= 1) {
            bounds[1] = part_cu_seqlens[1];
        }
        bounds[2] = 0;
        if (seq_count >= 2) {
            bounds[2] = part_cu_seqlens[2];
        }
        bounds[3] = 0;
        if (seq_count >= 3) {
            bounds[3] = part_cu_seqlens[3];
        }
        bounds[4] = 0;
        if (seq_count >= 4) {
            bounds[4] = part_cu_seqlens[4];
        }
        int windows[4];
        windows[0] = -1;
        if (seq_count > 0) {
            windows[0] = bounds[1] - bounds[0];
        }
        windows[1] = -1;
        if (seq_count > 1) {
            windows[1] = bounds[2] - bounds[1];
        }
        windows[2] = -1;
        if (seq_count > 2) {
            windows[2] = bounds[3] - bounds[2];
        }
        windows[3] = -1;
        if (seq_count > 3) {
            windows[3] = bounds[4] - bounds[3];
        }
        int wanted = place / 2;
        if (place % 2 == 1) {
            wanted = seq_count - 1 - place / 2;
        }
        int rank = 0;
        if (windows[1] > windows[0]) {
            rank = rank + 1;
        }
        if (windows[2] > windows[0]) {
            rank = rank + 1;
        }
        if (windows[3] > windows[0]) {
            rank = rank + 1;
        }
        if (rank == wanted) {
            sequence = 0;
            part_begin = bounds[0];
            part_stop = bounds[1];
        }
        int rank_0 = 0;
        if (windows[0] >= windows[1]) {
            rank_0 = rank_0 + 1;
        }
        if (windows[2] > windows[1]) {
            rank_0 = rank_0 + 1;
        }
        if (windows[3] > windows[1]) {
            rank_0 = rank_0 + 1;
        }
        if (rank_0 == wanted) {
            sequence = 1;
            part_begin = bounds[1];
            part_stop = bounds[2];
        }
        int rank_1 = 0;
        if (windows[0] >= windows[2]) {
            rank_1 = rank_1 + 1;
        }
        if (windows[1] >= windows[2]) {
            rank_1 = rank_1 + 1;
        }
        if (windows[3] > windows[2]) {
            rank_1 = rank_1 + 1;
        }
        if (rank_1 == wanted) {
            sequence = 2;
            part_begin = bounds[2];
            part_stop = bounds[3];
        }
        int rank_2 = 0;
        if (windows[0] >= windows[3]) {
            rank_2 = rank_2 + 1;
        }
        if (windows[1] >= windows[3]) {
            rank_2 = rank_2 + 1;
        }
        if (windows[2] >= windows[3]) {
            rank_2 = rank_2 + 1;
        }
        if (rank_2 == wanted) {
            sequence = 3;
            part_begin = bounds[3];
            part_stop = bounds[4];
        }
    }
    int task = bid / 11 % num_heads;
    int scan_stop = part_stop;
    int group = warp / 2;
    int row0 = bid % 11 * 12 + group * 6;
    int col = warp % 2 * 64 + lane * 2;
    unsigned int tile = smem_carry_addr + (unsigned int)(group * 4096);
    int row_offset[6];
    int _min_0 = ((row0) < (127) ? (row0) : (127));
    row_offset[0] = _min_0 * 128 + col;
    int _min_1 = ((row0 + 1) < (127) ? (row0 + 1) : (127));
    row_offset[1] = _min_1 * 128 + col;
    int _min_2 = ((row0 + 2) < (127) ? (row0 + 2) : (127));
    row_offset[2] = _min_2 * 128 + col;
    int _min_3 = ((row0 + 3) < (127) ? (row0 + 3) : (127));
    row_offset[3] = _min_3 * 128 + col;
    int _min_4 = ((row0 + 4) < (127) ? (row0 + 4) : (127));
    row_offset[4] = _min_4 * 128 + col;
    int _min_5 = ((row0 + 5) < (127) ? (row0 + 5) : (127));
    row_offset[5] = _min_5 * 128 + col;
    float hi_reg[2];
    float lo_reg[2];
    if (part_begin > 0) {
        float zero[12];
        zero[0] = 0.0f;
        zero[1] = 0.0f;
        zero[2] = 0.0f;
        zero[3] = 0.0f;
        zero[4] = 0.0f;
        zero[5] = 0.0f;
        zero[6] = 0.0f;
        zero[7] = 0.0f;
        zero[8] = 0.0f;
        zero[9] = 0.0f;
        zero[10] = 0.0f;
        zero[11] = 0.0f;
        if (row0 < 128) {
            hi_reg[0] = (float)(__nv_bfloat16)zero[0];
            lo_reg[0] = zero[0] - hi_reg[0];
            hi_reg[1] = (float)(__nv_bfloat16)zero[1];
            lo_reg[1] = zero[1] - hi_reg[1];
            unsigned int packed[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                packed[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_hi + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[0]))[0] = reinterpret_cast<unsigned*>(packed)[0];
            unsigned int packed_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_lo + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[0]))[0] = reinterpret_cast<unsigned*>(packed_1)[0];
        }
        if (row0 + 1 < 128) {
            hi_reg[0] = (float)(__nv_bfloat16)zero[2];
            lo_reg[0] = zero[2] - hi_reg[0];
            hi_reg[1] = (float)(__nv_bfloat16)zero[3];
            lo_reg[1] = zero[3] - hi_reg[1];
            unsigned int packed_2[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_hi + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[1]))[0] = reinterpret_cast<unsigned*>(packed_2)[0];
            unsigned int packed_1_1[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                packed_1_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_lo + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[1]))[0] = reinterpret_cast<unsigned*>(packed_1_1)[0];
        }
        if (row0 + 2 < 128) {
            hi_reg[0] = (float)(__nv_bfloat16)zero[4];
            lo_reg[0] = zero[4] - hi_reg[0];
            hi_reg[1] = (float)(__nv_bfloat16)zero[5];
            lo_reg[1] = zero[5] - hi_reg[1];
            unsigned int packed_3[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_hi + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[2]))[0] = reinterpret_cast<unsigned*>(packed_3)[0];
            unsigned int packed_1_2[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                packed_1_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_lo + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[2]))[0] = reinterpret_cast<unsigned*>(packed_1_2)[0];
        }
        if (row0 + 3 < 128) {
            hi_reg[0] = (float)(__nv_bfloat16)zero[6];
            lo_reg[0] = zero[6] - hi_reg[0];
            hi_reg[1] = (float)(__nv_bfloat16)zero[7];
            lo_reg[1] = zero[7] - hi_reg[1];
            unsigned int packed_4[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_hi + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[3]))[0] = reinterpret_cast<unsigned*>(packed_4)[0];
            unsigned int packed_1_3[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                packed_1_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_lo + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[3]))[0] = reinterpret_cast<unsigned*>(packed_1_3)[0];
        }
        if (row0 + 4 < 128) {
            hi_reg[0] = (float)(__nv_bfloat16)zero[8];
            lo_reg[0] = zero[8] - hi_reg[0];
            hi_reg[1] = (float)(__nv_bfloat16)zero[9];
            lo_reg[1] = zero[9] - hi_reg[1];
            unsigned int packed_5[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_hi + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[4]))[0] = reinterpret_cast<unsigned*>(packed_5)[0];
            unsigned int packed_1_4[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                packed_1_4[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_lo + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[4]))[0] = reinterpret_cast<unsigned*>(packed_1_4)[0];
        }
        if (row0 + 5 < 128) {
            hi_reg[0] = (float)(__nv_bfloat16)zero[10];
            lo_reg[0] = zero[10] - hi_reg[0];
            hi_reg[1] = (float)(__nv_bfloat16)zero[11];
            lo_reg[1] = zero[11] - hi_reg[1];
            unsigned int packed_6[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                packed_6[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_hi + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[5]))[0] = reinterpret_cast<unsigned*>(packed_6)[0];
            unsigned int packed_1_5[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                packed_1_5[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<unsigned*>(carry_lo + (((part_begin - 1) * num_heads + task) * 16384 + row_offset[5]))[0] = reinterpret_cast<unsigned*>(packed_1_5)[0];
        }
    }
    unsigned int _phase_map_full = 0;
    if (part_stop - part_begin > 1) {
        unsigned int map_slot = 0;
        if (scan_stop - part_begin > 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(map_full_addr + (map_slot) * 8, 32768);
                    cp_async_bulk_gmem2smem(smem_m_addr + map_slot * 32768, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(map_state_bf16) + ((unsigned long long)((part_begin * num_heads + task) * 16384) * (unsigned long long)2)), 32768, map_full_addr + (map_slot) * 8);
                }
            }
        }
        float carry_reg[12];
        float next_state[12];
        {
            float2 _v2_6 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[0]);
            carry_reg[0] = _v2_6.x;
            carry_reg[0 + 1] = _v2_6.y;
        }
        {
            float2 _v2_7 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[1]);
            carry_reg[2] = _v2_7.x;
            carry_reg[2 + 1] = _v2_7.y;
        }
        {
            float2 _v2_8 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[2]);
            carry_reg[4] = _v2_8.x;
            carry_reg[4 + 1] = _v2_8.y;
        }
        {
            float2 _v2_9 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[3]);
            carry_reg[6] = _v2_9.x;
            carry_reg[6 + 1] = _v2_9.y;
        }
        {
            float2 _v2_10 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[4]);
            carry_reg[8] = _v2_10.x;
            carry_reg[8 + 1] = _v2_10.y;
        }
        {
            float2 _v2_11 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[5]);
            carry_reg[10] = _v2_11.x;
            carry_reg[10 + 1] = _v2_11.y;
        }
        {
            float2 _v2_12 = *reinterpret_cast<const float2*>(split_state + ((part_begin + 1) * num_heads + task) * 16384 + row_offset[0]);
            next_state[0] = _v2_12.x;
            next_state[0 + 1] = _v2_12.y;
        }
        {
            float2 _v2_13 = *reinterpret_cast<const float2*>(split_state + ((part_begin + 1) * num_heads + task) * 16384 + row_offset[1]);
            next_state[2] = _v2_13.x;
            next_state[2 + 1] = _v2_13.y;
        }
        {
            float2 _v2_14 = *reinterpret_cast<const float2*>(split_state + ((part_begin + 1) * num_heads + task) * 16384 + row_offset[2]);
            next_state[4] = _v2_14.x;
            next_state[4 + 1] = _v2_14.y;
        }
        {
            float2 _v2_15 = *reinterpret_cast<const float2*>(split_state + ((part_begin + 1) * num_heads + task) * 16384 + row_offset[3]);
            next_state[6] = _v2_15.x;
            next_state[6 + 1] = _v2_15.y;
        }
        {
            float2 _v2_16 = *reinterpret_cast<const float2*>(split_state + ((part_begin + 1) * num_heads + task) * 16384 + row_offset[4]);
            next_state[8] = _v2_16.x;
            next_state[8 + 1] = _v2_16.y;
        }
        {
            float2 _v2_17 = *reinterpret_cast<const float2*>(split_state + ((part_begin + 1) * num_heads + task) * 16384 + row_offset[5]);
            next_state[10] = _v2_17.x;
            next_state[10 + 1] = _v2_17.y;
        }
        #pragma unroll 1
        for (int part = part_begin + 1; part < scan_stop; part++) {
            if (row0 < 128) {
                hi_reg[0] = (float)(__nv_bfloat16)carry_reg[0];
                lo_reg[0] = carry_reg[0] - hi_reg[0];
                hi_reg[1] = (float)(__nv_bfloat16)carry_reg[1];
                lo_reg[1] = carry_reg[1] - hi_reg[1];
                unsigned int packed_7[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                    packed_7[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_hi + (((part - 1) * num_heads + task) * 16384 + row_offset[0]))[0] = reinterpret_cast<unsigned*>(packed_7)[0];
                unsigned int packed_1_6[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                    packed_1_6[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_lo + (((part - 1) * num_heads + task) * 16384 + row_offset[0]))[0] = reinterpret_cast<unsigned*>(packed_1_6)[0];
            }
            if (row0 + 1 < 128) {
                hi_reg[0] = (float)(__nv_bfloat16)carry_reg[2];
                lo_reg[0] = carry_reg[2] - hi_reg[0];
                hi_reg[1] = (float)(__nv_bfloat16)carry_reg[3];
                lo_reg[1] = carry_reg[3] - hi_reg[1];
                unsigned int packed_8[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                    packed_8[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_hi + (((part - 1) * num_heads + task) * 16384 + row_offset[1]))[0] = reinterpret_cast<unsigned*>(packed_8)[0];
                unsigned int packed_1_7[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                    packed_1_7[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_lo + (((part - 1) * num_heads + task) * 16384 + row_offset[1]))[0] = reinterpret_cast<unsigned*>(packed_1_7)[0];
            }
            if (row0 + 2 < 128) {
                hi_reg[0] = (float)(__nv_bfloat16)carry_reg[4];
                lo_reg[0] = carry_reg[4] - hi_reg[0];
                hi_reg[1] = (float)(__nv_bfloat16)carry_reg[5];
                lo_reg[1] = carry_reg[5] - hi_reg[1];
                unsigned int packed_9[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                    packed_9[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_hi + (((part - 1) * num_heads + task) * 16384 + row_offset[2]))[0] = reinterpret_cast<unsigned*>(packed_9)[0];
                unsigned int packed_1_8[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                    packed_1_8[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_lo + (((part - 1) * num_heads + task) * 16384 + row_offset[2]))[0] = reinterpret_cast<unsigned*>(packed_1_8)[0];
            }
            if (row0 + 3 < 128) {
                hi_reg[0] = (float)(__nv_bfloat16)carry_reg[6];
                lo_reg[0] = carry_reg[6] - hi_reg[0];
                hi_reg[1] = (float)(__nv_bfloat16)carry_reg[7];
                lo_reg[1] = carry_reg[7] - hi_reg[1];
                unsigned int packed_10[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                    packed_10[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_hi + (((part - 1) * num_heads + task) * 16384 + row_offset[3]))[0] = reinterpret_cast<unsigned*>(packed_10)[0];
                unsigned int packed_1_9[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                    packed_1_9[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_lo + (((part - 1) * num_heads + task) * 16384 + row_offset[3]))[0] = reinterpret_cast<unsigned*>(packed_1_9)[0];
            }
            if (row0 + 4 < 128) {
                hi_reg[0] = (float)(__nv_bfloat16)carry_reg[8];
                lo_reg[0] = carry_reg[8] - hi_reg[0];
                hi_reg[1] = (float)(__nv_bfloat16)carry_reg[9];
                lo_reg[1] = carry_reg[9] - hi_reg[1];
                unsigned int packed_11[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                    packed_11[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_hi + (((part - 1) * num_heads + task) * 16384 + row_offset[4]))[0] = reinterpret_cast<unsigned*>(packed_11)[0];
                unsigned int packed_1_10[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                    packed_1_10[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_lo + (((part - 1) * num_heads + task) * 16384 + row_offset[4]))[0] = reinterpret_cast<unsigned*>(packed_1_10)[0];
            }
            if (row0 + 5 < 128) {
                hi_reg[0] = (float)(__nv_bfloat16)carry_reg[10];
                lo_reg[0] = carry_reg[10] - hi_reg[0];
                hi_reg[1] = (float)(__nv_bfloat16)carry_reg[11];
                lo_reg[1] = carry_reg[11] - hi_reg[1];
                unsigned int packed_12[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi_reg[_lp*2 + 0], hi_reg[_lp*2+1 + 0]));
                    packed_12[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_hi + (((part - 1) * num_heads + task) * 16384 + row_offset[5]))[0] = reinterpret_cast<unsigned*>(packed_12)[0];
                unsigned int packed_1_11[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_reg[_lp*2 + 0], lo_reg[_lp*2+1 + 0]));
                    packed_1_11[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(carry_lo + (((part - 1) * num_heads + task) * 16384 + row_offset[5]))[0] = reinterpret_cast<unsigned*>(packed_1_11)[0];
            }
            if (write_rows != 0) {
                long long row_start = row_starts[part];
                if (row0 < 128) {
                    long long row_index = (row_start * (long long)num_heads + (long long)task) * 16384 + (long long)row_offset[0];
                    {
                        float2 _v2 = make_float2(carry_reg[0 + 0], carry_reg[0 + 1]);
                        *reinterpret_cast<float2*>(rows + row_index) = _v2;
                    }
                }
                if (row0 + 1 < 128) {
                    long long row_index_1 = (row_start * (long long)num_heads + (long long)task) * 16384 + (long long)row_offset[1];
                    {
                        float2 _v2 = make_float2(carry_reg[2 + 0], carry_reg[2 + 1]);
                        *reinterpret_cast<float2*>(rows + row_index_1) = _v2;
                    }
                }
                if (row0 + 2 < 128) {
                    long long row_index_2 = (row_start * (long long)num_heads + (long long)task) * 16384 + (long long)row_offset[2];
                    {
                        float2 _v2 = make_float2(carry_reg[4 + 0], carry_reg[4 + 1]);
                        *reinterpret_cast<float2*>(rows + row_index_2) = _v2;
                    }
                }
                if (row0 + 3 < 128) {
                    long long row_index_3 = (row_start * (long long)num_heads + (long long)task) * 16384 + (long long)row_offset[3];
                    {
                        float2 _v2 = make_float2(carry_reg[6 + 0], carry_reg[6 + 1]);
                        *reinterpret_cast<float2*>(rows + row_index_3) = _v2;
                    }
                }
                if (row0 + 4 < 128) {
                    long long row_index_4 = (row_start * (long long)num_heads + (long long)task) * 16384 + (long long)row_offset[4];
                    {
                        float2 _v2 = make_float2(carry_reg[8 + 0], carry_reg[8 + 1]);
                        *reinterpret_cast<float2*>(rows + row_index_4) = _v2;
                    }
                }
                if (row0 + 5 < 128) {
                    long long row_index_5 = (row_start * (long long)num_heads + (long long)task) * 16384 + (long long)row_offset[5];
                    {
                        float2 _v2 = make_float2(carry_reg[10 + 0], carry_reg[10 + 1]);
                        *reinterpret_cast<float2*>(rows + row_index_5) = _v2;
                    }
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(tile + (unsigned int)((col * 2 >> 3 << 7) + ((col * 2 & 7 ^ col * 2 >> 3 & 7) << 4))), "r"(__float_as_uint(carry_reg[0])), "r"(__float_as_uint(carry_reg[2])), "r"(__float_as_uint(carry_reg[4])), "r"(__float_as_uint(carry_reg[6])) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(tile + (unsigned int)((col * 2 + 1 >> 3 << 7) + ((col * 2 + 1 & 7 ^ col * 2 + 1 >> 3 & 7) << 4))), "r"(__float_as_uint(carry_reg[8])), "r"(__float_as_uint(carry_reg[10])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(tile + (unsigned int)(((col + 1) * 2 >> 3 << 7) + (((col + 1) * 2 & 7 ^ (col + 1) * 2 >> 3 & 7) << 4))), "r"(__float_as_uint(carry_reg[1])), "r"(__float_as_uint(carry_reg[3])), "r"(__float_as_uint(carry_reg[5])), "r"(__float_as_uint(carry_reg[7])) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(tile + (unsigned int)(((col + 1) * 2 + 1 >> 3 << 7) + (((col + 1) * 2 + 1 & 7 ^ (col + 1) * 2 + 1 >> 3 & 7) << 4))), "r"(__float_as_uint(carry_reg[9])), "r"(__float_as_uint(carry_reg[11])) : "memory");
            float2 accum[6];
            accum[0] = make_float2(next_state[0], next_state[1]);
            accum[1] = make_float2(next_state[2], next_state[3]);
            accum[2] = make_float2(next_state[4], next_state[5]);
            accum[3] = make_float2(next_state[6], next_state[7]);
            accum[4] = make_float2(next_state[8], next_state[9]);
            accum[5] = make_float2(next_state[10], next_state[11]);
            if (scan_stop > part + 1) {
                {
                    float2 _v2_18 = *reinterpret_cast<const float2*>(split_state + ((part + 1) * num_heads + task) * 16384 + row_offset[0]);
                    next_state[0] = _v2_18.x;
                    next_state[0 + 1] = _v2_18.y;
                }
                {
                    float2 _v2_19 = *reinterpret_cast<const float2*>(split_state + ((part + 1) * num_heads + task) * 16384 + row_offset[1]);
                    next_state[2] = _v2_19.x;
                    next_state[2 + 1] = _v2_19.y;
                }
                {
                    float2 _v2_20 = *reinterpret_cast<const float2*>(split_state + ((part + 1) * num_heads + task) * 16384 + row_offset[2]);
                    next_state[4] = _v2_20.x;
                    next_state[4 + 1] = _v2_20.y;
                }
                {
                    float2 _v2_21 = *reinterpret_cast<const float2*>(split_state + ((part + 1) * num_heads + task) * 16384 + row_offset[3]);
                    next_state[6] = _v2_21.x;
                    next_state[6 + 1] = _v2_21.y;
                }
                {
                    float2 _v2_22 = *reinterpret_cast<const float2*>(split_state + ((part + 1) * num_heads + task) * 16384 + row_offset[4]);
                    next_state[8] = _v2_22.x;
                    next_state[8 + 1] = _v2_22.y;
                }
                {
                    float2 _v2_23 = *reinterpret_cast<const float2*>(split_state + ((part + 1) * num_heads + task) * 16384 + row_offset[5]);
                    next_state[10] = _v2_23.x;
                    next_state[10 + 1] = _v2_23.y;
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        unsigned int fill_slot = ((map_slot + 1 < 2) ? map_slot + 1 : (unsigned int)0);
                        mbarrier_arrive_expect_tx(map_full_addr + (fill_slot) * 8, 32768);
                        cp_async_bulk_gmem2smem(smem_m_addr + fill_slot * 32768, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(map_state_bf16) + ((unsigned long long)((part * num_heads + task) * 16384) * (unsigned long long)2)), 32768, map_full_addr + (fill_slot) * 8);
                    }
                }
            }
            __syncthreads();
            mbarrier_wait(map_full_addr + (map_slot) * 8, _phase_map_full);
            unsigned int panel_base = smem_m_addr + map_slot * 32768;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int inner = 0; inner < 128; inner++) {
                unsigned int packed_m[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_m[0])) : "r"(panel_base + (unsigned int)((inner * 128 + col) * 2)));
                float _bf16x2_add_f32_0[2];
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                    "}\n"
                    : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(packed_m[0]), "f"(0.0f), "f"(0.0f));
                float2 map_pair = make_float2(_bf16x2_add_f32_0[0], _bf16x2_add_f32_0[1]);
                float unit_words[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&unit_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&unit_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&unit_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&unit_words[(0) + 3]))
                    : "r"(tile + (unsigned int)((inner * 2 >> 3 << 7) + ((inner * 2 & 7 ^ inner * 2 >> 3 & 7) << 4))));
                float unit_words_0[4];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&unit_words_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&unit_words_0[(0) + 1]))
                    : "r"(tile + (unsigned int)((inner * 2 + 1 >> 3 << 7) + ((inner * 2 + 1 & 7 ^ inner * 2 + 1 >> 3 & 7) << 4))));
                accum[0] = fma_f32x2_rn_ftz(make_float2(unit_words[0], unit_words[0]), map_pair, accum[0]);
                accum[1] = fma_f32x2_rn_ftz(make_float2(unit_words[1], unit_words[1]), map_pair, accum[1]);
                accum[2] = fma_f32x2_rn_ftz(make_float2(unit_words[2], unit_words[2]), map_pair, accum[2]);
                accum[3] = fma_f32x2_rn_ftz(make_float2(unit_words[3], unit_words[3]), map_pair, accum[3]);
                accum[4] = fma_f32x2_rn_ftz(make_float2(unit_words_0[0], unit_words_0[0]), map_pair, accum[4]);
                accum[5] = fma_f32x2_rn_ftz(make_float2(unit_words_0[1], unit_words_0[1]), map_pair, accum[5]);
            }
            __syncthreads();
            map_slot += 1;
            if (map_slot == 2) { map_slot = 0; _phase_map_full ^= 1; }
            carry_reg[0] = accum[0].x;
            carry_reg[1] = accum[0].y;
            carry_reg[2] = accum[1].x;
            carry_reg[3] = accum[1].y;
            carry_reg[4] = accum[2].x;
            carry_reg[5] = accum[2].y;
            carry_reg[6] = accum[3].x;
            carry_reg[7] = accum[3].y;
            carry_reg[8] = accum[4].x;
            carry_reg[9] = accum[4].y;
            carry_reg[10] = accum[5].x;
            carry_reg[11] = accum[5].y;
        }
        if (write_final_state != 0) {
            int slot = final_indices[sequence];
            if (row0 < 128) {
                long long final_index = (long long)slot * final_slot_stride + (long long)task * 16384 + (long long)row_offset[0];
                if (write_final_state == 1) {
                    {
                        float2 _v2 = make_float2(carry_reg[0 + 0], carry_reg[0 + 1]);
                        *reinterpret_cast<float2*>(final_state + final_index) = _v2;
                    }
                } else {
                    float rounded[2];
                    rounded[0] = (float)(__nv_bfloat16)carry_reg[0];
                    rounded[1] = (float)(__nv_bfloat16)carry_reg[1];
                    unsigned int packed_13[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded[_lp*2 + 0], rounded[_lp*2+1 + 0]));
                        packed_13[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(final_state_bf16 + (final_index))[0] = reinterpret_cast<unsigned*>(packed_13)[0];
                }
            }
            if (row0 + 1 < 128) {
                long long final_index_1 = (long long)slot * final_slot_stride + (long long)task * 16384 + (long long)row_offset[1];
                if (write_final_state == 1) {
                    {
                        float2 _v2 = make_float2(carry_reg[2 + 0], carry_reg[2 + 1]);
                        *reinterpret_cast<float2*>(final_state + final_index_1) = _v2;
                    }
                } else {
                    float rounded_1[2];
                    rounded_1[0] = (float)(__nv_bfloat16)carry_reg[2];
                    rounded_1[1] = (float)(__nv_bfloat16)carry_reg[3];
                    unsigned int packed_14[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_1[_lp*2 + 0], rounded_1[_lp*2+1 + 0]));
                        packed_14[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_1))[0] = reinterpret_cast<unsigned*>(packed_14)[0];
                }
            }
            if (row0 + 2 < 128) {
                long long final_index_2 = (long long)slot * final_slot_stride + (long long)task * 16384 + (long long)row_offset[2];
                if (write_final_state == 1) {
                    {
                        float2 _v2 = make_float2(carry_reg[4 + 0], carry_reg[4 + 1]);
                        *reinterpret_cast<float2*>(final_state + final_index_2) = _v2;
                    }
                } else {
                    float rounded_2[2];
                    rounded_2[0] = (float)(__nv_bfloat16)carry_reg[4];
                    rounded_2[1] = (float)(__nv_bfloat16)carry_reg[5];
                    unsigned int packed_15[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_2[_lp*2 + 0], rounded_2[_lp*2+1 + 0]));
                        packed_15[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_2))[0] = reinterpret_cast<unsigned*>(packed_15)[0];
                }
            }
            if (row0 + 3 < 128) {
                long long final_index_3 = (long long)slot * final_slot_stride + (long long)task * 16384 + (long long)row_offset[3];
                if (write_final_state == 1) {
                    {
                        float2 _v2 = make_float2(carry_reg[6 + 0], carry_reg[6 + 1]);
                        *reinterpret_cast<float2*>(final_state + final_index_3) = _v2;
                    }
                } else {
                    float rounded_3[2];
                    rounded_3[0] = (float)(__nv_bfloat16)carry_reg[6];
                    rounded_3[1] = (float)(__nv_bfloat16)carry_reg[7];
                    unsigned int packed_16[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_3[_lp*2 + 0], rounded_3[_lp*2+1 + 0]));
                        packed_16[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_3))[0] = reinterpret_cast<unsigned*>(packed_16)[0];
                }
            }
            if (row0 + 4 < 128) {
                long long final_index_4 = (long long)slot * final_slot_stride + (long long)task * 16384 + (long long)row_offset[4];
                if (write_final_state == 1) {
                    {
                        float2 _v2 = make_float2(carry_reg[8 + 0], carry_reg[8 + 1]);
                        *reinterpret_cast<float2*>(final_state + final_index_4) = _v2;
                    }
                } else {
                    float rounded_4[2];
                    rounded_4[0] = (float)(__nv_bfloat16)carry_reg[8];
                    rounded_4[1] = (float)(__nv_bfloat16)carry_reg[9];
                    unsigned int packed_17[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_4[_lp*2 + 0], rounded_4[_lp*2+1 + 0]));
                        packed_17[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_4))[0] = reinterpret_cast<unsigned*>(packed_17)[0];
                }
            }
            if (row0 + 5 < 128) {
                long long final_index_5 = (long long)slot * final_slot_stride + (long long)task * 16384 + (long long)row_offset[5];
                if (write_final_state == 1) {
                    {
                        float2 _v2 = make_float2(carry_reg[10 + 0], carry_reg[10 + 1]);
                        *reinterpret_cast<float2*>(final_state + final_index_5) = _v2;
                    }
                } else {
                    float rounded_5[2];
                    rounded_5[0] = (float)(__nv_bfloat16)carry_reg[10];
                    rounded_5[1] = (float)(__nv_bfloat16)carry_reg[11];
                    unsigned int packed_18[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_5[_lp*2 + 0], rounded_5[_lp*2+1 + 0]));
                        packed_18[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_5))[0] = reinterpret_cast<unsigned*>(packed_18)[0];
                }
            }
        }
    } else if (write_final_state != 0) {
        float single_state[12];
        {
            float2 _v2_24 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[0]);
            single_state[0] = _v2_24.x;
            single_state[0 + 1] = _v2_24.y;
        }
        {
            float2 _v2_25 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[1]);
            single_state[2] = _v2_25.x;
            single_state[2 + 1] = _v2_25.y;
        }
        {
            float2 _v2_26 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[2]);
            single_state[4] = _v2_26.x;
            single_state[4 + 1] = _v2_26.y;
        }
        {
            float2 _v2_27 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[3]);
            single_state[6] = _v2_27.x;
            single_state[6 + 1] = _v2_27.y;
        }
        {
            float2 _v2_28 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[4]);
            single_state[8] = _v2_28.x;
            single_state[8 + 1] = _v2_28.y;
        }
        {
            float2 _v2_29 = *reinterpret_cast<const float2*>(split_state + (part_begin * num_heads + task) * 16384 + row_offset[5]);
            single_state[10] = _v2_29.x;
            single_state[10 + 1] = _v2_29.y;
        }
        int slot_1 = final_indices[sequence];
        if (row0 < 128) {
            long long final_index_6 = (long long)slot_1 * final_slot_stride + (long long)task * 16384 + (long long)row_offset[0];
            if (write_final_state == 1) {
                {
                    float2 _v2 = make_float2(single_state[0 + 0], single_state[0 + 1]);
                    *reinterpret_cast<float2*>(final_state + final_index_6) = _v2;
                }
            } else {
                float rounded_6[2];
                rounded_6[0] = (float)(__nv_bfloat16)single_state[0];
                rounded_6[1] = (float)(__nv_bfloat16)single_state[1];
                unsigned int packed_19[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_6[_lp*2 + 0], rounded_6[_lp*2+1 + 0]));
                    packed_19[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_6))[0] = reinterpret_cast<unsigned*>(packed_19)[0];
            }
        }
        if (row0 + 1 < 128) {
            long long final_index_7 = (long long)slot_1 * final_slot_stride + (long long)task * 16384 + (long long)row_offset[1];
            if (write_final_state == 1) {
                {
                    float2 _v2 = make_float2(single_state[2 + 0], single_state[2 + 1]);
                    *reinterpret_cast<float2*>(final_state + final_index_7) = _v2;
                }
            } else {
                float rounded_7[2];
                rounded_7[0] = (float)(__nv_bfloat16)single_state[2];
                rounded_7[1] = (float)(__nv_bfloat16)single_state[3];
                unsigned int packed_20[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_7[_lp*2 + 0], rounded_7[_lp*2+1 + 0]));
                    packed_20[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_7))[0] = reinterpret_cast<unsigned*>(packed_20)[0];
            }
        }
        if (row0 + 2 < 128) {
            long long final_index_8 = (long long)slot_1 * final_slot_stride + (long long)task * 16384 + (long long)row_offset[2];
            if (write_final_state == 1) {
                {
                    float2 _v2 = make_float2(single_state[4 + 0], single_state[4 + 1]);
                    *reinterpret_cast<float2*>(final_state + final_index_8) = _v2;
                }
            } else {
                float rounded_8[2];
                rounded_8[0] = (float)(__nv_bfloat16)single_state[4];
                rounded_8[1] = (float)(__nv_bfloat16)single_state[5];
                unsigned int packed_21[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_8[_lp*2 + 0], rounded_8[_lp*2+1 + 0]));
                    packed_21[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_8))[0] = reinterpret_cast<unsigned*>(packed_21)[0];
            }
        }
        if (row0 + 3 < 128) {
            long long final_index_9 = (long long)slot_1 * final_slot_stride + (long long)task * 16384 + (long long)row_offset[3];
            if (write_final_state == 1) {
                {
                    float2 _v2 = make_float2(single_state[6 + 0], single_state[6 + 1]);
                    *reinterpret_cast<float2*>(final_state + final_index_9) = _v2;
                }
            } else {
                float rounded_9[2];
                rounded_9[0] = (float)(__nv_bfloat16)single_state[6];
                rounded_9[1] = (float)(__nv_bfloat16)single_state[7];
                unsigned int packed_22[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_9[_lp*2 + 0], rounded_9[_lp*2+1 + 0]));
                    packed_22[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_9))[0] = reinterpret_cast<unsigned*>(packed_22)[0];
            }
        }
        if (row0 + 4 < 128) {
            long long final_index_10 = (long long)slot_1 * final_slot_stride + (long long)task * 16384 + (long long)row_offset[4];
            if (write_final_state == 1) {
                {
                    float2 _v2 = make_float2(single_state[8 + 0], single_state[8 + 1]);
                    *reinterpret_cast<float2*>(final_state + final_index_10) = _v2;
                }
            } else {
                float rounded_10[2];
                rounded_10[0] = (float)(__nv_bfloat16)single_state[8];
                rounded_10[1] = (float)(__nv_bfloat16)single_state[9];
                unsigned int packed_23[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_10[_lp*2 + 0], rounded_10[_lp*2+1 + 0]));
                    packed_23[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_10))[0] = reinterpret_cast<unsigned*>(packed_23)[0];
            }
        }
        if (row0 + 5 < 128) {
            long long final_index_11 = (long long)slot_1 * final_slot_stride + (long long)task * 16384 + (long long)row_offset[5];
            if (write_final_state == 1) {
                {
                    float2 _v2 = make_float2(single_state[10 + 0], single_state[10 + 1]);
                    *reinterpret_cast<float2*>(final_state + final_index_11) = _v2;
                }
            } else {
                float rounded_11[2];
                rounded_11[0] = (float)(__nv_bfloat16)single_state[10];
                rounded_11[1] = (float)(__nv_bfloat16)single_state[11];
                unsigned int packed_24[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(rounded_11[_lp*2 + 0], rounded_11[_lp*2+1 + 0]));
                    packed_24[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(final_state_bf16 + (final_index_11))[0] = reinterpret_cast<unsigned*>(packed_24)[0];
            }
        }
    }
    __threadfence();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Cleanup
    __syncthreads();
}

} // extern "C"
