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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define SMEM_V_SMEM_BF16_OFF 0
#define SMEM_V_SMEM_BF16_STAGE_BYTES 8704
#define SMEM_V_SMEM_BF16_STRIDE 8704
#define SMEM_TOTAL 8704
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_minimax_h3_varlen_attention_e70c217406e9eafae840(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ q_fp4, uint8_t* __restrict__ k_fp4, uint8_t* __restrict__ q_scale, uint8_t* __restrict__ k_scale, uint8_t* __restrict__ v_fp4_t, uint8_t* __restrict__ v_scale_lo, uint8_t* __restrict__ v_scale_hi, int* __restrict__ block_token, int* __restrict__ block_valid, int heads, int PB)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* v_smem_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int v_smem_bf16_addr = smem + 0;

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
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 0])[0]), "=f"((&q_values[iteration * 16 + 0])[1])
                    : "r"(_v8_0_0[0]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 2])[0]), "=f"((&q_values[iteration * 16 + 2])[1])
                    : "r"(_v8_0_0[1]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 4])[0]), "=f"((&q_values[iteration * 16 + 4])[1])
                    : "r"(_v8_0_0[2]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 6])[0]), "=f"((&q_values[iteration * 16 + 6])[1])
                    : "r"(_v8_0_0[3]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 8])[0]), "=f"((&q_values[iteration * 16 + 8])[1])
                    : "r"(_v8_0_0[4]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 10])[0]), "=f"((&q_values[iteration * 16 + 10])[1])
                    : "r"(_v8_0_0[5]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 12])[0]), "=f"((&q_values[iteration * 16 + 12])[1])
                    : "r"(_v8_0_0[6]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&q_values[iteration * 16 + 14])[0]), "=f"((&q_values[iteration * 16 + 14])[1])
                    : "r"(_v8_0_0[7]));
            }
            {
                const void* _v8p_1 = (const void*)(k + (input_offset));
                uint32_t _v8_1_0[8];
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const char*)_v8p_1 + 0) : "memory");
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 0])[0]), "=f"((&k_values[iteration * 16 + 0])[1])
                    : "r"(_v8_1_0[0]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 2])[0]), "=f"((&k_values[iteration * 16 + 2])[1])
                    : "r"(_v8_1_0[1]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 4])[0]), "=f"((&k_values[iteration * 16 + 4])[1])
                    : "r"(_v8_1_0[2]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 6])[0]), "=f"((&k_values[iteration * 16 + 6])[1])
                    : "r"(_v8_1_0[3]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 8])[0]), "=f"((&k_values[iteration * 16 + 8])[1])
                    : "r"(_v8_1_0[4]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 10])[0]), "=f"((&k_values[iteration * 16 + 10])[1])
                    : "r"(_v8_1_0[5]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 12])[0]), "=f"((&k_values[iteration * 16 + 12])[1])
                    : "r"(_v8_1_0[6]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&k_values[iteration * 16 + 14])[0]), "=f"((&k_values[iteration * 16 + 14])[1])
                    : "r"(_v8_1_0[7]));
            }
            {
                const void* _v8p_2 = (const void*)(v + (input_offset));
                uint32_t _v8_2_0[8];
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 0])[0]), "=f"((&v_values[iteration * 16 + 0])[1])
                    : "r"(_v8_2_0[0]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 2])[0]), "=f"((&v_values[iteration * 16 + 2])[1])
                    : "r"(_v8_2_0[1]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 4])[0]), "=f"((&v_values[iteration * 16 + 4])[1])
                    : "r"(_v8_2_0[2]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 6])[0]), "=f"((&v_values[iteration * 16 + 6])[1])
                    : "r"(_v8_2_0[3]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 8])[0]), "=f"((&v_values[iteration * 16 + 8])[1])
                    : "r"(_v8_2_0[4]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 10])[0]), "=f"((&v_values[iteration * 16 + 10])[1])
                    : "r"(_v8_2_0[5]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 12])[0]), "=f"((&v_values[iteration * 16 + 12])[1])
                    : "r"(_v8_2_0[6]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&v_values[iteration * 16 + 14])[0]), "=f"((&v_values[iteration * 16 + 14])[1])
                    : "r"(_v8_2_0[7]));
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
        float _max_0 = max_noftz(value_max, -value_min);
        float amax = _max_0;
        float _max_1 = max_noftz(amax * 0.16666666666666666f, 0.001953125f);
        float raw_scale = _max_1;
        float _fp8_rt_0;
        uint16_t _e4m3x2_3;
        uint32_t _f16x2_3;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_3) : "f"(0.0f), "f"(raw_scale));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3) : "h"(_e4m3x2_3));
        uint16_t _fp8_h0_3 = (uint16_t)(_f16x2_3 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_3));
        float rounded_scale = _fp8_rt_0;
        float _rcp_0 = approx_rcp(rounded_scale);
        float inverse_scale = _rcp_0;
        float normalized[16];
        #pragma unroll
        for (int element_1 = 0; element_1 < 16; element_1++) {
            normalized[element_1] = (q_values + iteration_1 * 16)[element_1] * inverse_scale;
        }
        unsigned int packed[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(normalized[0]), "f"(normalized[1]), "f"(normalized[2]), "f"(normalized[3]), "f"(normalized[4]), "f"(normalized[5]), "f"(normalized[6]), "f"(normalized[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(normalized[8]), "f"(normalized[9]), "f"(normalized[10]), "f"(normalized[11]), "f"(normalized[12]), "f"(normalized[13]), "f"(normalized[14]), "f"(normalized[15]));
        float k_values_max = (k_values + iteration_1 * 16)[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            k_values_max = max_noftz(k_values_max, (k_values + iteration_1 * 16)[_lr]);
        }
        float value_max_0 = k_values_max;
        float k_values_min = (k_values + iteration_1 * 16)[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            k_values_min = fminf(k_values_min, (k_values + iteration_1 * 16)[_lr]);
        }
        float value_min_1 = k_values_min;
        float _max_2 = max_noftz(value_max_0, -value_min_1);
        float amax_2 = _max_2;
        float _max_3 = max_noftz(amax_2 * 0.16666666666666666f, 0.001953125f);
        float raw_scale_3 = _max_3;
        float _fp8_rt_1;
        uint16_t _e4m3x2_4;
        uint32_t _f16x2_4;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_4) : "f"(0.0f), "f"(raw_scale_3));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4) : "h"(_e4m3x2_4));
        uint16_t _fp8_h0_4 = (uint16_t)(_f16x2_4 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_4));
        float rounded_scale_4 = _fp8_rt_1;
        float _rcp_1 = approx_rcp(rounded_scale_4);
        float inverse_scale_5 = _rcp_1;
        float normalized_6[16];
        #pragma unroll
        for (int element_2 = 0; element_2 < 16; element_2++) {
            normalized_6[element_2] = (k_values + iteration_1 * 16)[element_2] * inverse_scale_5;
        }
        unsigned int packed_7[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_7[0]) : "f"(normalized_6[0]), "f"(normalized_6[1]), "f"(normalized_6[2]), "f"(normalized_6[3]), "f"(normalized_6[4]), "f"(normalized_6[5]), "f"(normalized_6[6]), "f"(normalized_6[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_7[1]) : "f"(normalized_6[8]), "f"(normalized_6[9]), "f"(normalized_6[10]), "f"(normalized_6[11]), "f"(normalized_6[12]), "f"(normalized_6[13]), "f"(normalized_6[14]), "f"(normalized_6[15]));
        long long output_offset = ((long long)tile * 128 + (long long)row128) * 64 + (long long)(group_1 * 8);
        {
            int2 _iv2 = make_int2(packed[0 + 0], packed[0 + 1]);
            *reinterpret_cast<int2*>(q_fp4 + output_offset + 0) = _iv2;
        }
        {
            int2 _iv2 = make_int2(packed_7[0 + 0], packed_7[0 + 1]);
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
            float _shfl_1 = __shfl_sync(4294967295, raw_scale_3, j, 4);
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
        unsigned int packed_v_bf16[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((v_values + iteration_1 * 16)[_lp*2 + 0], (v_values + iteration_1 * 16)[_lp*2+1 + 0]));
            packed_v_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        int smem_byte = (row_1 * 136 + group_1 * 16) * 2;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(v_smem_bf16_addr + (unsigned int)smem_byte), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[(0) + 3])));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "r"(v_smem_bf16_addr + (unsigned int)smem_byte + 16), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_v_bf16[(4) + 3])));
    }
    __syncthreads();
    #pragma unroll
    for (int iteration_2 = 0; iteration_2 < 1; iteration_2++) {
        int vector_2 = tid + iteration_2 * 256;
        int v_group = vector_2 / 128;
        int dim = vector_2 - v_group * 128;
        float values[16];
        #pragma unroll
        for (int element_3 = 0; element_3 < 16; element_3++) {
            int smem_index = (v_group * 16 + element_3) * 136 + dim;
            values[element_3] = v_smem_bf16[smem_index];
        }
        float values_max = values[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            values_max = max_noftz(values_max, values[_lr]);
        }
        float value_max_1 = values_max;
        float values_min = values[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            values_min = fminf(values_min, values[_lr]);
        }
        float value_min_2 = values_min;
        float _max_4 = max_noftz(value_max_1, -value_min_2);
        float amax_1 = _max_4;
        float _max_5 = max_noftz(amax_1 * 0.16666666666666666f, 0.001953125f);
        float raw_scale_1 = _max_5;
        float _fp8_rt_2;
        uint16_t _e4m3x2_5;
        uint32_t _f16x2_5;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_5) : "f"(0.0f), "f"(raw_scale_1));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5) : "h"(_e4m3x2_5));
        uint16_t _fp8_h0_5 = (uint16_t)(_f16x2_5 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_5));
        float rounded_scale_1 = _fp8_rt_2;
        float _rcp_2 = approx_rcp(rounded_scale_1);
        float inverse_scale_1 = _rcp_2;
        float normalized_1[16];
        #pragma unroll
        for (int element_4 = 0; element_4 < 16; element_4++) {
            normalized_1[element_4] = values[element_4] * inverse_scale_1;
        }
        unsigned int packed_1[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[0]) : "f"(normalized_1[0]), "f"(normalized_1[1]), "f"(normalized_1[2]), "f"(normalized_1[3]), "f"(normalized_1[4]), "f"(normalized_1[5]), "f"(normalized_1[6]), "f"(normalized_1[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[1]) : "f"(normalized_1[8]), "f"(normalized_1[9]), "f"(normalized_1[10]), "f"(normalized_1[11]), "f"(normalized_1[12]), "f"(normalized_1[13]), "f"(normalized_1[14]), "f"(normalized_1[15]));
        int group_2 = sub * 2 + v_group;
        long long output_offset_1 = ((long long)head * 128 + (long long)dim) * ((long long)PB * 64) + (long long)(pblock * 64) + (long long)(group_2 * 8);
        {
            int2 _iv2 = make_int2(packed_1[0 + 0], packed_1[0 + 1]);
            *reinterpret_cast<int2*>(v_fp4_t + output_offset_1 + 0) = _iv2;
        }
        int row_outer_1 = dim / 32;
        int row_inner_1 = dim - row_outer_1 * 32;
        int row_quad_1 = row_inner_1 / 8;
        int row_lane_1 = row_inner_1 - row_quad_1 * 8;
        int group_in_half = group_2 - group_2 / 4 * 4;
        int scale_offset_1 = ((row_quad_1 * 8 + row_lane_1) * 4 + row_outer_1) * 4 + group_in_half;
        long long scale_tile_offset_1 = (long long)tile * 512;
        if (group_2 < 4) {
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(raw_scale_1));
                *(reinterpret_cast<unsigned char*>(v_scale_lo + (scale_tile_offset_1 + (long long)scale_offset_1)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        } else {
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(raw_scale_1));
                *(reinterpret_cast<unsigned char*>(v_scale_hi + (scale_tile_offset_1 + (long long)scale_offset_1)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
        }
    }
}

} // extern "C"
