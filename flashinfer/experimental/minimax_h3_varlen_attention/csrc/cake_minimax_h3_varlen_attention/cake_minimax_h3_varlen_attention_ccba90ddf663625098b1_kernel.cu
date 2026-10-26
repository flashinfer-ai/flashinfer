/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_minimax_h3_varlen_attention_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_AMAX_SMEM_OFF 0
#define SMEM_AMAX_SMEM_STAGE_BYTES 32
#define SMEM_AMAX_SMEM_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_minimax_h3_varlen_attention_ccba90ddf663625098b1(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ q_fp4, uint8_t* __restrict__ k_fp4, uint8_t* __restrict__ q_scale, uint8_t* __restrict__ k_scale, uint8_t* __restrict__ v_fp8, float* __restrict__ v_amax, float* __restrict__ v_amax_partial, int* __restrict__ block_token, int* __restrict__ block_valid, int heads, int PB)
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
    float* amax_smem = reinterpret_cast<float*>(smem_raw + SMEM_AMAX_SMEM_OFF);
    const int amax_smem_addr = smem + SMEM_AMAX_SMEM_OFF;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int bid_0 = bid;
    int tile = bid_0 / 4;
    int sub = bid_0 - tile * 4;
    int head = tile / PB;
    int pblock = tile - head * PB;
    int first_token = block_token[pblock] + sub * 32;
    int valid_rows = block_valid[pblock] - sub * 32;
    float q_values[16];
    float k_values[16];
    float v_values[16];
    #pragma unroll
    for (int iteration = 0; iteration < 1; iteration++) {
        int vector = tid + iteration * 256;
        int row = vector / 8;
        int group = vector - row * 8;
        int token = first_token + row;
        long long input_offset = ((long long)token * (long long)heads + (long long)head) * 128 + (long long)(group * 16);
        if (row < valid_rows) {
            {
                const void* _v8p_0 = (const void*)(q + (input_offset));
                uint32_t _v8_0_0[8];
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
                (&q_values[iteration * 16 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) << 16);
                (&q_values[iteration * 16 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) & 0xffff0000u);
                (&q_values[iteration * 16 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) << 16);
                (&q_values[iteration * 16 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) & 0xffff0000u);
                (&q_values[iteration * 16 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) << 16);
                (&q_values[iteration * 16 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) & 0xffff0000u);
                (&q_values[iteration * 16 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) << 16);
                (&q_values[iteration * 16 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) & 0xffff0000u);
                (&q_values[iteration * 16 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) << 16);
                (&q_values[iteration * 16 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) & 0xffff0000u);
                (&q_values[iteration * 16 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) << 16);
                (&q_values[iteration * 16 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) & 0xffff0000u);
                (&q_values[iteration * 16 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) << 16);
                (&q_values[iteration * 16 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) & 0xffff0000u);
                (&q_values[iteration * 16 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) << 16);
                (&q_values[iteration * 16 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) & 0xffff0000u);
            }
            {
                const void* _v8p_1 = (const void*)(k + (input_offset));
                uint32_t _v8_1_0[8];
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const char*)_v8p_1 + 0) : "memory");
                (&k_values[iteration * 16 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) << 16);
                (&k_values[iteration * 16 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[0]) & 0xffff0000u);
                (&k_values[iteration * 16 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) << 16);
                (&k_values[iteration * 16 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[1]) & 0xffff0000u);
                (&k_values[iteration * 16 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) << 16);
                (&k_values[iteration * 16 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[2]) & 0xffff0000u);
                (&k_values[iteration * 16 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) << 16);
                (&k_values[iteration * 16 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[3]) & 0xffff0000u);
                (&k_values[iteration * 16 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) << 16);
                (&k_values[iteration * 16 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[4]) & 0xffff0000u);
                (&k_values[iteration * 16 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) << 16);
                (&k_values[iteration * 16 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[5]) & 0xffff0000u);
                (&k_values[iteration * 16 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) << 16);
                (&k_values[iteration * 16 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[6]) & 0xffff0000u);
                (&k_values[iteration * 16 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) << 16);
                (&k_values[iteration * 16 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_1_0[7]) & 0xffff0000u);
            }
            {
                const void* _v8p_2 = (const void*)(v + (input_offset));
                uint32_t _v8_2_0[8];
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
                (&v_values[iteration * 16 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) << 16);
                (&v_values[iteration * 16 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) & 0xffff0000u);
                (&v_values[iteration * 16 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) << 16);
                (&v_values[iteration * 16 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) & 0xffff0000u);
                (&v_values[iteration * 16 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) << 16);
                (&v_values[iteration * 16 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) & 0xffff0000u);
                (&v_values[iteration * 16 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) << 16);
                (&v_values[iteration * 16 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) & 0xffff0000u);
                (&v_values[iteration * 16 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) << 16);
                (&v_values[iteration * 16 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) & 0xffff0000u);
                (&v_values[iteration * 16 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) << 16);
                (&v_values[iteration * 16 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) & 0xffff0000u);
                (&v_values[iteration * 16 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) << 16);
                (&v_values[iteration * 16 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) & 0xffff0000u);
                (&v_values[iteration * 16 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) << 16);
                (&v_values[iteration * 16 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) & 0xffff0000u);
            }
        } else {
            #pragma unroll
            for (int element = 0; element < 16; element++) {
                q_values[iteration * 16 + element] = 0.0f;
                k_values[iteration * 16 + element] = 0.0f;
                v_values[iteration * 16 + element] = 0.0f;
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float amax_lo = v_amax_partial[tid];
    float amax_hi = v_amax_partial[tid + 256];
    float _max_0 = max_noftz(amax_lo, amax_hi);
    float amax_local = _max_0;
    float _warp_reduce_0 = amax_local;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    amax_local = _warp_reduce_0;
    float _cross_warp_reduce_0;
    if (lane == 0) { amax_smem[warp] = amax_local; }
    __syncthreads();
    if (warp == 0) {
        float _br = (lane < 8) ? amax_smem[lane] : -CUDART_INF_F;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _br = fmaxf(_br, __shfl_xor_sync(0xFFFFFFFF, _br, offset));
        if (lane == 0) { amax_smem[0] = _br; }
    }
    __syncthreads();
    _cross_warp_reduce_0 = amax_smem[0];
    float amax = _cross_warp_reduce_0;
    float _max_1 = max_noftz(amax, 1e-12f);
    float _rcp_0 = approx_rcp(_max_1);
    float v_inverse_scale = _rcp_0 * 448.0f;
    if (bid_0 == 0) {
        if (tid == 0) {
            *(reinterpret_cast<float*>(v_amax) + (0)) = amax;
        }
    }
    #pragma unroll
    for (int iteration_1 = 0; iteration_1 < 1; iteration_1++) {
        int vector_1 = tid + iteration_1 * 256;
        int row_1 = vector_1 / 8;
        int group_1 = vector_1 - row_1 * 8;
        int row128 = sub * 32 + row_1;
        float q_values_max = (q_values + iteration_1 * 16)[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            q_values_max = max_noftz(q_values_max, (q_values + iteration_1 * 16)[_lr]);
        }
        float value_max = q_values_max;
        float q_values_min = (q_values + iteration_1 * 16)[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            q_values_min = fminf(q_values_min, (q_values + iteration_1 * 16)[_lr]);
        }
        float value_min = q_values_min;
        float _max_2 = max_noftz(value_max, -value_min);
        float amax_0 = _max_2;
        float _max_3 = max_noftz(amax_0 * 0.16666666666666666f, 0.001953125f);
        float raw_scale = _max_3;
        float _fp8_rt_0;
        uint16_t _e4m3x2_3;
        uint32_t _f16x2_3;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_3) : "f"(0.0f), "f"(raw_scale));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3) : "h"(_e4m3x2_3));
        uint16_t _fp8_h0_3 = (uint16_t)(_f16x2_3 & 0xFFFFu);
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_3));
        float rounded_scale = _fp8_rt_0;
        float _rcp_1 = approx_rcp(rounded_scale);
        float inverse_scale = _rcp_1;
        float normalized[16];
        #pragma unroll
        for (int element_1 = 0; element_1 < 16; element_1++) {
            normalized[element_1] = q_values[iteration_1 * 16 + element_1] * inverse_scale;
        }
        unsigned int packed[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(normalized[0]), "f"(normalized[1]), "f"(normalized[2]), "f"(normalized[3]), "f"(normalized[4]), "f"(normalized[5]), "f"(normalized[6]), "f"(normalized[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(normalized[8]), "f"(normalized[9]), "f"(normalized[10]), "f"(normalized[11]), "f"(normalized[12]), "f"(normalized[13]), "f"(normalized[14]), "f"(normalized[15]));
        float k_values_max = (k_values + iteration_1 * 16)[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            k_values_max = max_noftz(k_values_max, (k_values + iteration_1 * 16)[_lr]);
        }
        float value_max_1 = k_values_max;
        float k_values_min = (k_values + iteration_1 * 16)[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            k_values_min = fminf(k_values_min, (k_values + iteration_1 * 16)[_lr]);
        }
        float value_min_2 = k_values_min;
        float _max_4 = max_noftz(value_max_1, -value_min_2);
        float amax_3 = _max_4;
        float _max_5 = max_noftz(amax_3 * 0.16666666666666666f, 0.001953125f);
        float raw_scale_4 = _max_5;
        float _fp8_rt_1;
        uint16_t _e4m3x2_4;
        uint32_t _f16x2_4;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_4) : "f"(0.0f), "f"(raw_scale_4));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4) : "h"(_e4m3x2_4));
        uint16_t _fp8_h0_4 = (uint16_t)(_f16x2_4 & 0xFFFFu);
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_4));
        float rounded_scale_5 = _fp8_rt_1;
        float _rcp_2 = approx_rcp(rounded_scale_5);
        float inverse_scale_6 = _rcp_2;
        float normalized_7[16];
        #pragma unroll
        for (int element_2 = 0; element_2 < 16; element_2++) {
            normalized_7[element_2] = k_values[iteration_1 * 16 + element_2] * inverse_scale_6;
        }
        unsigned int packed_8[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_8[0]) : "f"(normalized_7[0]), "f"(normalized_7[1]), "f"(normalized_7[2]), "f"(normalized_7[3]), "f"(normalized_7[4]), "f"(normalized_7[5]), "f"(normalized_7[6]), "f"(normalized_7[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_8[1]) : "f"(normalized_7[8]), "f"(normalized_7[9]), "f"(normalized_7[10]), "f"(normalized_7[11]), "f"(normalized_7[12]), "f"(normalized_7[13]), "f"(normalized_7[14]), "f"(normalized_7[15]));
        long long output_offset = ((long long)tile * 128 + (long long)row128) * 64 + (long long)(group_1 * 8);
        {
            int2 _iv2 = make_int2(packed[0 + 0], packed[0 + 1]);
            *reinterpret_cast<int2*>(q_fp4 + output_offset + 0) = _iv2;
        }
        {
            int2 _iv2 = make_int2(packed_8[0 + 0], packed_8[0 + 1]);
            *reinterpret_cast<int2*>(k_fp4 + output_offset + 0) = _iv2;
        }
        int row_outer = row128 / 32;
        int row_inner = row128 - row_outer * 32;
        int row_quad = row_inner / 8;
        int row_lane = row_inner - row_quad * 8;
        int group_pair = group_1 / 4;
        int group_lane = group_1 - group_pair * 4;
        int scale_offset = (((row_quad * 2 + group_pair) * 8 + row_lane) * 4 + row_outer) * 4 + group_lane;
        long long scale_tile_offset = (long long)tile * 1024;
        float q_sf4[4];
        float k_sf4[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float _shfl_0 = __shfl_sync(4294967295, raw_scale, j, 4);
            q_sf4[j] = _shfl_0;
            float _shfl_1 = __shfl_sync(4294967295, raw_scale_4, j, 4);
            k_sf4[j] = _shfl_1;
        }
        unsigned int q_sf_word[1];
        unsigned int k_sf_word[1];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(q_sf4[0]), "f"(q_sf4[1]),
                                   "f"(q_sf4[2]), "f"(q_sf4[3]));
            q_sf_word[0] = _packed;
        }
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(k_sf4[0]), "f"(k_sf4[1]),
                                   "f"(k_sf4[2]), "f"(k_sf4[3]));
            k_sf_word[0] = _packed;
        }
        if (group_lane == 0) {
            *(reinterpret_cast<unsigned int*>(q_scale + (scale_tile_offset + (long long)scale_offset)) + (0)) = q_sf_word[0];
            *(reinterpret_cast<unsigned int*>(k_scale + (scale_tile_offset + (long long)scale_offset)) + (0)) = k_sf_word[0];
        }
        long long v_output_offset = ((long long)tile * 128 + (long long)row128) * 128 + (long long)(group_1 * 16);
        {
            const float2 _prescale2_5 = {v_inverse_scale, v_inverse_scale};
            #if __CUDA_ARCH__ >= 1000
            #pragma unroll
            for (int _ps = 0; _ps < 8; _ps++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(&(v_values + iteration_1 * 16)[0])[_ps], _prescale2_5);
            #else
            #pragma unroll
            for (int _ps = 0; _ps < 16; _ps++)
                (v_values + iteration_1 * 16)[0 + _ps] *= v_inverse_scale;
            #endif
            unsigned int _fp8_pk[4];
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[0]) : "f"((v_values + iteration_1 * 16)[0 + 0]), "f"((v_values + iteration_1 * 16)[0 + 1]), "f"((v_values + iteration_1 * 16)[0 + 2]), "f"((v_values + iteration_1 * 16)[0 + 3]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[1]) : "f"((v_values + iteration_1 * 16)[0 + 4]), "f"((v_values + iteration_1 * 16)[0 + 5]), "f"((v_values + iteration_1 * 16)[0 + 6]), "f"((v_values + iteration_1 * 16)[0 + 7]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[2]) : "f"((v_values + iteration_1 * 16)[0 + 8]), "f"((v_values + iteration_1 * 16)[0 + 9]), "f"((v_values + iteration_1 * 16)[0 + 10]), "f"((v_values + iteration_1 * 16)[0 + 11]));
            asm("{\n\t"
                ".reg .b16 _lo, _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}\n"
                : "=r"(_fp8_pk[3]) : "f"((v_values + iteration_1 * 16)[0 + 12]), "f"((v_values + iteration_1 * 16)[0 + 13]), "f"((v_values + iteration_1 * 16)[0 + 14]), "f"((v_values + iteration_1 * 16)[0 + 15]));
            *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(v_fp8 + v_output_offset) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
        }
    }
}

} // extern "C"
