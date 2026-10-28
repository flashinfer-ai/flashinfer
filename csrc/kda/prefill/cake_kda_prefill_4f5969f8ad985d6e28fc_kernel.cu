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
#define SMEM_SMEM_CARRY_OFF 1024
#define SMEM_SMEM_CARRY_STAGE_BYTES 2048
#define SMEM_SMEM_CARRY_STRIDE 2048
#define SMEM_TOTAL 66560
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 3

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
kernel_cake_kda_prefill_4f5969f8ad985d6e28fc(float* __restrict__ split_state, __nv_bfloat16* __restrict__ map_state_bf16, float* __restrict__ carry, __nv_bfloat16* __restrict__ carry_hi, __nv_bfloat16* __restrict__ carry_lo, float* __restrict__ final_state, int write_final_state, int num_heads, int* __restrict__ part_cu_seqlens, int* __restrict__ final_indices, long long final_slot_stride, __nv_bfloat16* __restrict__ final_state_bf16, float* __restrict__ rows, long long* __restrict__ row_starts, int write_rows)
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
    float* smem_carry = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_carry_addr = smem + 1024;
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
    int sequence = bid / 32 / num_heads;
    int task = bid / 32 % num_heads;
    int part_begin = part_cu_seqlens[sequence];
    int part_stop = part_cu_seqlens[sequence + 1];
    int scan_stop = part_stop;
    int row_band = bid % 32;
    int row = row_band * 4 + warp;
    int col = lane * 4;
    float hi_reg[4];
    float lo_reg[4];
    if (part_begin > 0) {
        float zero[4];
        zero[0] = 0.0f;
        zero[1] = 0.0f;
        zero[2] = 0.0f;
        zero[3] = 0.0f;
        {
            float4 _v4 = make_float4(zero[0 + 0], zero[0 + 1], zero[0 + 2], zero[0 + 3]);
            *reinterpret_cast<float4*>(carry + ((part_begin - 1) * num_heads + task) * 16384 + row * 128 + col) = _v4;
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
        float carry_reg[4];
        float next_state[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(split_state + (part_begin * num_heads + task) * 16384 + row * 128 + col);
            carry_reg[0 + 0] = _v4.x;
            carry_reg[0 + 1] = _v4.y;
            carry_reg[0 + 2] = _v4.z;
            carry_reg[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(split_state + ((part_begin + 1) * num_heads + task) * 16384 + row * 128 + col);
            next_state[0 + 0] = _v4.x;
            next_state[0 + 1] = _v4.y;
            next_state[0 + 2] = _v4.z;
            next_state[0 + 3] = _v4.w;
        }
        __syncthreads();
        float2 accum01 = make_float2(0.0f, 0.0f);
        float2 accum23 = make_float2(0.0f, 0.0f);
        #pragma unroll 1
        for (int part = part_begin + 1; part < scan_stop; part++) {
            {
                float4 _v4 = make_float4(carry_reg[0 + 0], carry_reg[0 + 1], carry_reg[0 + 2], carry_reg[0 + 3]);
                *reinterpret_cast<float4*>(carry + ((part - 1) * num_heads + task) * 16384 + row * 128 + col) = _v4;
            }
            if (write_rows != 0) {
                long long row_start = row_starts[part];
                long long row_offset = (row_start * (long long)num_heads + (long long)task) * 16384 + (long long)(row * 128 + col);
                {
                    float4 _v4 = make_float4(carry_reg[0 + 0], carry_reg[0 + 1], carry_reg[0 + 2], carry_reg[0 + 3]);
                    *reinterpret_cast<float4*>(rows + row_offset) = _v4;
                }
            }
            float accum[4];
            for (int out_col = 0; out_col < 4; out_col++) {
                accum[out_col] = next_state[out_col];
            }
            if (scan_stop > part + 1) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(split_state + ((part + 1) * num_heads + task) * 16384 + row * 128 + col);
                    next_state[0 + 0] = _v4.x;
                    next_state[0 + 1] = _v4.y;
                    next_state[0 + 2] = _v4.z;
                    next_state[0 + 3] = _v4.w;
                }
            }
            if (scan_stop > part + 1) {
                if (warp == 0) {
                    if (elect_sync()) {
                        unsigned int fill_slot = ((map_slot + 1 < 2) ? map_slot + 1 : (unsigned int)0);
                        mbarrier_arrive_expect_tx(map_full_addr + (fill_slot) * 8, 32768);
                        cp_async_bulk_gmem2smem(smem_m_addr + fill_slot * 32768, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(map_state_bf16) + ((unsigned long long)((part * num_heads + task) * 16384) * (unsigned long long)2)), 32768, map_full_addr + (fill_slot) * 8);
                    }
                }
            }
            mbarrier_wait(map_full_addr + (map_slot) * 8, _phase_map_full);
            unsigned int panel_base = smem_m_addr + map_slot * 32768;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int inner = 0; inner < 128; inner++) {
                float a = __shfl_sync(0xFFFFFFFF, carry_reg[inner % 4], inner / 4);
                unsigned int packed_m[2];
                int m_addr = panel_base + (unsigned int)((inner * 128 + col) * 2);
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_m[0])) : "r"(m_addr));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_m[1])) : "r"(m_addr + 4));
                float _bf16x2_add_f32_0[2];
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                    "}\n"
                    : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(packed_m[0]), "f"(0.0f), "f"(0.0f));
                float _bf16x2_add_f32_1[2];
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                    "}\n"
                    : "=&f"(_bf16x2_add_f32_1[0]), "=&f"(_bf16x2_add_f32_1[1]) : "r"(packed_m[1]), "f"(0.0f), "f"(0.0f));
                accum[0] = __fmaf_rn(a, _bf16x2_add_f32_0[0], accum[0]);
                accum[1] = __fmaf_rn(a, _bf16x2_add_f32_0[1], accum[1]);
                accum[2] = __fmaf_rn(a, _bf16x2_add_f32_1[0], accum[2]);
                accum[3] = __fmaf_rn(a, _bf16x2_add_f32_1[1], accum[3]);
            }
            __syncthreads();
            map_slot += 1;
            if (map_slot == 2) { map_slot = 0; _phase_map_full ^= 1; }
            for (int out_col_1 = 0; out_col_1 < 4; out_col_1++) {
                carry_reg[out_col_1] = accum[out_col_1];
            }
        }
        if (write_final_state != 0) {
            int slot = final_indices[sequence];
            long long final_offset = (long long)slot * final_slot_stride + (long long)task * 16384 + (long long)(row * 128 + col);
            if (write_final_state == 1) {
                {
                    float4 _v4 = make_float4(carry_reg[0 + 0], carry_reg[0 + 1], carry_reg[0 + 2], carry_reg[0 + 3]);
                    *reinterpret_cast<float4*>(final_state + final_offset) = _v4;
                }
            } else {
                float rounded[4];
                for (int value_col = 0; value_col < 4; value_col++) {
                    rounded[value_col] = (float)(__nv_bfloat16)carry_reg[value_col];
                }
                for (int half = 0; half < 2; half++) {
                    unsigned int packed[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((rounded + half * 2)[_lp*2 + 0], (rounded + half * 2)[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(final_state_bf16 + (final_offset + (long long)(half * 2)))[0] = reinterpret_cast<unsigned*>(packed)[0];
                }
            }
        }
    } else if (write_final_state != 0) {
        float single_state[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(split_state + (part_begin * num_heads + task) * 16384 + row * 128 + col);
            single_state[0 + 0] = _v4.x;
            single_state[0 + 1] = _v4.y;
            single_state[0 + 2] = _v4.z;
            single_state[0 + 3] = _v4.w;
        }
        int slot_1 = final_indices[sequence];
        long long final_offset_1 = (long long)slot_1 * final_slot_stride + (long long)task * 16384 + (long long)(row * 128 + col);
        if (write_final_state == 1) {
            {
                float4 _v4 = make_float4(single_state[0 + 0], single_state[0 + 1], single_state[0 + 2], single_state[0 + 3]);
                *reinterpret_cast<float4*>(final_state + final_offset_1) = _v4;
            }
        } else {
            float rounded_1[4];
            for (int value_col_1 = 0; value_col_1 < 4; value_col_1++) {
                rounded_1[value_col_1] = (float)(__nv_bfloat16)single_state[value_col_1];
            }
            for (int half_1 = 0; half_1 < 2; half_1++) {
                unsigned int packed_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((rounded_1 + half_1 * 2)[_lp*2 + 0], (rounded_1 + half_1 * 2)[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(final_state_bf16 + (final_offset_1 + (long long)(half_1 * 2)))[0] = reinterpret_cast<unsigned*>(packed_1)[0];
            }
        }
    }
    __threadfence();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Cleanup
    __syncthreads();
}

} // extern "C"
