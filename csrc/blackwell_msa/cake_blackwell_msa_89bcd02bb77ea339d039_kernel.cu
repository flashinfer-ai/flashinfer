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
#define NUM_MAIN_STAGES 1
#define SMEM_PARTIAL_STAGE_OFF 0
#define SMEM_PARTIAL_STAGE_STAGE_BYTES 32768
#define SMEM_PARTIAL_STAGE_STRIDE 32768
#define SMEM_TOTAL 32768
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_blackwell_msa_89bcd02bb77ea339d039(__nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, int return_softmax_lse, int return_temperature_lse)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* partial_stage = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int partial_stage_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int row_group = tid / 8;
    int lane_in_row = lane % 8;
    int source_lane_base = lane / 8 * 8;
    int row = blockIdx.x * 32 + row_group;
    int total_rows_out = total_q * num_q_heads;
    int row_valid = ((row < total_rows_out) ? 1 : 0);
    int split_count = 0;
    if (row_valid != 0) {
        int q_abs = row / num_q_heads;
        int q_head = row - q_abs * num_q_heads;
        int kv_head = q_head / qhead_per_kv;
        split_count = split_counts[q_abs * num_kv_heads + kv_head];
        if (split_count > topk) {
            split_count = topk;
        }
        if (split_count > 16) {
            split_count = 16;
        }
        if (split_count < 0) {
            split_count = 0;
        }
    }
    int split0 = lane_in_row;
    int split1 = lane_in_row + 8;
    float lse0 = -CUDART_INF_F;
    float lse1 = -CUDART_INF_F;
    if (row_valid != 0 && split0 < split_count) {
        lse0 = partial_lse[(long long)split0 * (long long)total_rows_out + (long long)row];
    }
    if (row_valid != 0 && split1 < split_count) {
        lse1 = partial_lse[(long long)split1 * (long long)total_rows_out + (long long)row];
    }
    float lse_max;
    asm("max.f32 %0, %1, %2;" : "=f"(lse_max) : "f"(lse0), "f"(lse1));
    float peer_max = __shfl_xor_sync(0xFFFFFFFF, lse_max, 4);
    if (peer_max > lse_max) {
        lse_max = peer_max;
    }
    float peer_max_0 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 2);
    if (peer_max_0 > lse_max) {
        lse_max = peer_max_0;
    }
    float peer_max_1 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 1);
    if (peer_max_1 > lse_max) {
        lse_max = peer_max_1;
    }
    float safe_lse_max = ((lse_max == -CUDART_INF_F) ? 0.0f : lse_max);
    float weight0 = 0.0f;
    float weight1 = 0.0f;
    if (split0 < split_count) {
        float _exp2_0 = approx_exp2((lse0 - safe_lse_max) * 1.442695f);
        weight0 = _exp2_0;
        if (lse0 == -CUDART_INF_F) {
            weight0 = 0.0f;
        }
    }
    if (split1 < split_count) {
        float _exp2_1 = approx_exp2((lse1 - safe_lse_max) * 1.442695f);
        weight1 = _exp2_1;
        if (lse1 == -CUDART_INF_F) {
            weight1 = 0.0f;
        }
    }
    float lse_sum = weight0 + weight1;
    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 4);
    lse_sum += _shfl_xor_sync_2;
    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 2);
    lse_sum += _shfl_xor_sync_3;
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 1);
    lse_sum += _shfl_xor_sync_4;
    float _rcp_5 = approx_rcp(lse_sum);
    float inv_lse_sum = ((lse_sum > 0.0f && lse_sum == lse_sum) ? _rcp_5 : 0.0f);
    weight0 *= inv_lse_sum;
    weight1 *= inv_lse_sum;
    if (row_valid != 0 && lane_in_row == 0) {
        float _log2_6;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(lse_sum));
        float combined_lse = ((lse_sum > 0.0f) ? __fmaf_rn(_log2_6, 0.6931472f, safe_lse_max) : -CUDART_INF_F);
        if (return_softmax_lse != 0) {
            lse[row] = combined_lse;
        }
        if (return_temperature_lse != 0) {
            temperature_lse[row] = combined_lse;
        }
    }
    if (row_valid != 0) {
        int col_segment = lane_in_row * 16;
        float accum[16];
        #pragma unroll
        for (int elem = 0; elem < 16; elem++) {
            accum[elem] = 0.0f;
        }
        for (int preload_split = 0; preload_split < 4; preload_split++) {
            for (int half = 0; half < 2; half++) {
                asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                    :: "r"(partial_stage_addr + (unsigned int)(preload_split * 8192) + (unsigned int)(tid * 32) + (unsigned int)(half * 16)), "l"(partial_o + (((long long)preload_split * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment + (long long)(half * 8))), "r"((split_count > preload_split) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (int split = 0; split < 16; split++) {
            float source_weight = ((split < 8) ? weight0 : weight1);
            float split_weight = __shfl_sync(0xFFFFFFFF, source_weight, source_lane_base + split % 8);
            asm volatile("cp.async.wait_group 3;");
            unsigned int packed_values[8];
            int stage_value_addr = partial_stage_addr + (unsigned int)(split % 4 * 8192) + (unsigned int)(tid * 32);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&packed_values[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(0) + 3]))
                : "r"(stage_value_addr));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&packed_values[4])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(4) + 3]))
                : "r"(stage_value_addr + 16));
            float _unpack_bf16_0[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_values[_pair]) << 16);
                (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_values[_pair]) & 0xffff0000u);
            }
            if (split_count > split) {
                #pragma unroll
                for (int elem_1 = 0; elem_1 < 16; elem_1++) {
                    accum[elem_1] = __fmaf_rn(_unpack_bf16_0[elem_1], split_weight, accum[elem_1]);
                }
            }
            int next_split = split + 4;
            int safe_next_split = ((next_split < 16) ? next_split : 0);
            for (int half_1 = 0; half_1 < 2; half_1++) {
                asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                    :: "r"(partial_stage_addr + (unsigned int)(split % 4 * 8192) + (unsigned int)(tid * 32) + (unsigned int)(half_1 * 16)), "l"(partial_o + (((long long)safe_next_split * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment + (long long)(half_1 * 8))), "r"((next_split < 16 && next_split < split_count) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.wait_group 0;");
        {
            __nv_bfloat162 _pk[8];
            _pk[0] = __floats2bfloat162_rn(accum[0 + 0], accum[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(accum[0 + 2], accum[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(accum[0 + 4], accum[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(accum[0 + 6], accum[0 + 7]);
            _pk[4] = __floats2bfloat162_rn(accum[0 + 8], accum[0 + 9]);
            _pk[5] = __floats2bfloat162_rn(accum[0 + 10], accum[0 + 11]);
            _pk[6] = __floats2bfloat162_rn(accum[0 + 12], accum[0 + 13]);
            _pk[7] = __floats2bfloat162_rn(accum[0 + 14], accum[0 + 15]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row * 128 + (long long)col_segment)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + ((long long)row * 128 + (long long)col_segment)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
        }
    }
}

} // extern "C"
