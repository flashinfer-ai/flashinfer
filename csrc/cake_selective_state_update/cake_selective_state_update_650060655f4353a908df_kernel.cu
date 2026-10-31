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
#define THREADS 64
#define LAUNCH_MIN_BLOCKS 16

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

__global__ __launch_bounds__(64, LAUNCH_MIN_BLOCKS) void
kernel_cake_selective_state_update_650060655f4353a908df(__nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ x, unsigned long long dt_addr, unsigned long long a_addr, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, unsigned long long d_addr, __nv_bfloat16* __restrict__ z, unsigned long long dt_bias_addr, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, long long* __restrict__ dst_state_batch_indices, int nheads, int ngroups, unsigned long long state_stride_slot, long long x_batch_stride, long long b_batch_stride, long long c_batch_stride, long long out_batch_stride, long long dt_batch_stride, long long dt_head_stride, long long a_head_stride, long long d_head_stride, long long dt_bias_head_stride, int dt_softplus, int has_z, int disable_state_update, long long pad_slot_id)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int dim_tile = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int group = (int)((unsigned int)(head * ngroups) / (unsigned int)nheads);
    int rows_per_tile = 16;
    int dim_base = dim_tile * rows_per_tile;
    long long batch_i64 = (long long)batch;
    long long source_slot = 0;
    long long destination_slot = 0;
    long long _vec_load_0[1];
    {
        asm("ld.global.nc.s64 %0, [%1];" : "=l"(_vec_load_0[0]) : "l"((const void*)(reinterpret_cast<const long long*>(state_batch_indices) + (batch_i64))));
    }
    long long _vec_load_1[1];
    {
        asm("ld.global.nc.s64 %0, [%1];" : "=l"(_vec_load_1[0]) : "l"((const void*)(reinterpret_cast<const long long*>(dst_state_batch_indices) + (batch_i64))));
    }
    source_slot = _vec_load_0[0];
    destination_slot = _vec_load_1[0];
    float a_early = reinterpret_cast<float*>(a_addr)[(long long)head * a_head_stride];
    float dt_raw_early = 0.0f;
    float dt_bias_early = 0.0f;
    float d_early = 0.0f;
    dt_raw_early = reinterpret_cast<float*>(dt_addr)[(long long)batch * dt_batch_stride + (long long)head * dt_head_stride];
    dt_bias_early = reinterpret_cast<float*>(dt_bias_addr)[(long long)head * dt_bias_head_stride];
    d_early = reinterpret_cast<float*>(d_addr)[(long long)head * d_head_stride];
    int row_subgroup = lane / 8;
    int row_member = lane % 8;
    int state_col = row_member * 16;
    long long group_i64 = (long long)group;
    long long b_base = batch_i64 * b_batch_stride + group_i64 * 128;
    long long c_base = batch_i64 * c_batch_stride + group_i64 * 128;
    unsigned int b_carriers[8];
    unsigned int c_carriers[8];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(B + b_base + (long long)state_col);
        uint4* _vdst_0 = reinterpret_cast<uint4*>(&b_carriers[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
            _vdst_0[_blk] = _vptr_0[_blk];
        }
    }
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(C + c_base + (long long)state_col);
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&c_carriers[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    float row_values[16];
    unsigned int row_carriers[8];
    float x_lanes[1];
    float z_lanes[1];
    int live_source = 0;
    if (source_slot != pad_slot_id) {
        live_source = 1;
    }
    int live_store = 0;
    if (disable_state_update == 0) {
        if (source_slot != pad_slot_id) {
            if (destination_slot != pad_slot_id) {
                live_store = 1;
            }
        }
    }
    unsigned long long source_base = (unsigned long long)source_slot * state_stride_slot;
    unsigned long long destination_base = (unsigned long long)destination_slot * state_stride_slot;
    int trip_rows = 8;
    int first_base = dim_base + warp * 4;
    #pragma unroll
    for (int element = 0; element < 16; element++) {
        row_values[element] = 0.0f;
    }
    #pragma unroll
    for (int element_1 = 0; element_1 < 8; element_1++) {
        row_carriers[element_1] = 0;
    }
    #pragma unroll
    for (int r = 0; r < 1; r++) {
        x_lanes[r] = 0.0f;
        z_lanes[r] = 0.0f;
    }
    if (row_member == 0) {
        #pragma unroll
        for (int r_1 = 0; r_1 < 1; r_1++) {
            int dim_index_r = first_base + r_1 * 8 + row_subgroup;
            long long x_index_r = batch_i64 * x_batch_stride + (long long)(head * 64 + dim_index_r);
            x_lanes[r_1] = (float)x[x_index_r];
            if (has_z != 0) {
                z_lanes[r_1] = (float)z[x_index_r];
            }
        }
    }
    if (live_source != 0) {
        #pragma unroll
        for (int r_2 = 0; r_2 < 1; r_2++) {
            int dim_index_r_1 = first_base + r_2 * 8 + row_subgroup;
            unsigned long long row_offset_r = (unsigned long long)((head * 64 + dim_index_r_1) * 128 + state_col);
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(row_carriers[r_2 * 8 + 0]), "=r"(row_carriers[r_2 * 8 + 1]), "=r"(row_carriers[r_2 * 8 + 2]), "=r"(row_carriers[r_2 * 8 + 3]), "=r"(row_carriers[r_2 * 8 + 4]), "=r"(row_carriers[r_2 * 8 + 5]), "=r"(row_carriers[r_2 * 8 + 6]), "=r"(row_carriers[r_2 * 8 + 7]) : "l"((const void*)((const char*)(state + source_base + row_offset_r) + 0)) : "memory");
            }
        }
    }
    float dt_lane = 0.0f;
    float decay_lane = 0.0f;
    float d_lane = 0.0f;
    float dt_early = dt_raw_early + dt_bias_early;
    if (dt_softplus != 0) {
        if (dt_early <= 20.0f) {
            float _exp2_4 = approx_exp2(dt_early * 1.442695f);
            float _log2_3;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(1.0f + _exp2_4));
            dt_early = _log2_3 * 0.6931472f;
        }
    }
    dt_lane = dt_early;
    float _exp2_5 = approx_exp2(a_early * dt_early * 1.442695f);
    decay_lane = _exp2_5;
    d_lane = d_early;
    float dt_value = dt_lane;
    float decay = decay_lane;
    float d_value = d_lane;
    #pragma unroll 1
    for (int trip = 0; trip < 2; trip++) {
        int trip_base = first_base + trip * trip_rows;
        if (trip != 0) {
            #pragma unroll
            for (int element_2 = 0; element_2 < 16; element_2++) {
                row_values[element_2] = 0.0f;
            }
            #pragma unroll
            for (int element_3 = 0; element_3 < 8; element_3++) {
                row_carriers[element_3] = 0;
            }
            #pragma unroll
            for (int r_3 = 0; r_3 < 1; r_3++) {
                x_lanes[r_3] = 0.0f;
                z_lanes[r_3] = 0.0f;
            }
            if (row_member == 0) {
                #pragma unroll
                for (int r_4 = 0; r_4 < 1; r_4++) {
                    int dim_index_r_2 = trip_base + r_4 * 8 + row_subgroup;
                    long long x_index_r_1 = batch_i64 * x_batch_stride + (long long)(head * 64 + dim_index_r_2);
                    x_lanes[r_4] = (float)x[x_index_r_1];
                    if (has_z != 0) {
                        z_lanes[r_4] = (float)z[x_index_r_1];
                    }
                }
            }
            if (live_source != 0) {
                #pragma unroll
                for (int r_5 = 0; r_5 < 1; r_5++) {
                    int dim_index_r_3 = trip_base + r_5 * 8 + row_subgroup;
                    unsigned long long row_offset_r_1 = (unsigned long long)((head * 64 + dim_index_r_3) * 128 + state_col);
                    {
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(row_carriers[r_5 * 8 + 0]), "=r"(row_carriers[r_5 * 8 + 1]), "=r"(row_carriers[r_5 * 8 + 2]), "=r"(row_carriers[r_5 * 8 + 3]), "=r"(row_carriers[r_5 * 8 + 4]), "=r"(row_carriers[r_5 * 8 + 5]), "=r"(row_carriers[r_5 * 8 + 6]), "=r"(row_carriers[r_5 * 8 + 7]) : "l"((const void*)((const char*)(state + source_base + row_offset_r_1) + 0)) : "memory");
                    }
                }
            }
        }
        #pragma unroll
        for (int r_6 = 0; r_6 < 1; r_6++) {
            int dim_index_r_4 = trip_base + r_6 * 8 + row_subgroup;
            float x_value = __shfl_sync(0xFFFFFFFF, x_lanes[r_6], row_subgroup * 8);
            float partial = 0.0f;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                float b_half[8];
                float c_half[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&b_half[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((b_carriers + half * 4)[_pair]) << 16);
                    (&b_half[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((b_carriers + half * 4)[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&c_half[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((c_carriers + half * 4)[_pair]) << 16);
                    (&c_half[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((c_carriers + half * 4)[_pair]) & 0xffff0000u);
                }
                float s_half[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&s_half[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((row_carriers + r_6 * 8 + half * 4)[_pair]) << 16);
                    (&s_half[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((row_carriers + r_6 * 8 + half * 4)[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int element_4 = 0; element_4 < 8; element_4++) {
                    row_values[r_6 * 16 + half * 8 + element_4] = s_half[element_4];
                }
                #pragma unroll
                for (int element_5 = 0; element_5 < 8; element_5++) {
                    float d_b = b_half[element_5] * dt_value;
                    float decayed_state = row_values[r_6 * 16 + half * 8 + element_5] * decay;
                    float new_state = decayed_state + d_b * x_value;
                    row_values[r_6 * 16 + half * 8 + element_5] = new_state;
                    partial += new_state * c_half[element_5];
                }
            }
            if (live_store != 0) {
                unsigned long long row_offset_r_2 = (unsigned long long)((head * 64 + dim_index_r_4) * 128 + state_col);
                unsigned int _packed_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 0], (row_values + r_6 * 16)[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 2], (row_values + r_6 * 16)[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 4], (row_values + r_6 * 16)[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 6], (row_values + r_6 * 16)[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 8], (row_values + r_6 * 16)[_lp*2+1 + 8]));
                    _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 10], (row_values + r_6 * 16)[_lp*2+1 + 10]));
                    _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 12], (row_values + r_6 * 16)[_lp*2+1 + 12]));
                    _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((row_values + r_6 * 16)[_lp*2 + 14], (row_values + r_6 * 16)[_lp*2+1 + 14]));
                    _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(_packed_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(state + (destination_base + row_offset_r_2))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
                }
            }
            float row_sum = partial;
            float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 4);
            row_sum += _shfl_xor_sync_8;
            float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
            row_sum += _shfl_xor_sync_9;
            float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);
            row_sum += _shfl_xor_sync_10;
            if (row_member == 0) {
                float result = row_sum + d_value * x_lanes[r_6];
                if (has_z != 0) {
                    float z_lane = z_lanes[r_6];
                    float sigmoid_z = 0.0f;
                    float _exp2_12 = approx_exp2((-z_lane) * 1.442695f);
                    float _rcp_11 = approx_rcp(1.0f + _exp2_12);
                    sigmoid_z = _rcp_11;
                    result *= z_lane * sigmoid_z;
                }
                long long out_index_r = batch_i64 * out_batch_stride + (long long)(head * 64 + dim_index_r_4);
                output[out_index_r] = result;
            }
        }
    }
}

} // extern "C"
