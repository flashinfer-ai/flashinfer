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
#define SMEM_AMAX_SMEM_OFF 0
#define SMEM_AMAX_SMEM_STAGE_BYTES 32
#define SMEM_AMAX_SMEM_STRIDE 32
#define SMEM_TOTAL 128
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







// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_minimax_h3_varlen_attention_bf109517d2bd582e5452(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ q_fp4, uint8_t* __restrict__ k_fp4, uint8_t* __restrict__ q_scale, uint8_t* __restrict__ k_scale, uint8_t* __restrict__ v_fp8, float* __restrict__ v_amax, float* __restrict__ v_amax_partial, int* __restrict__ block_token, int* __restrict__ block_valid, int heads, int PB)
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
    float* amax_smem = reinterpret_cast<float*>(smem_raw + 0);
    const int amax_smem_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int bid_0 = bid;
    int tile = bid_0 / 4;
    int sub = bid_0 - tile * 4;
    int head = tile / PB;
    int pblock = tile - head * PB;
    int first_token = block_token[pblock] + sub * 32;
    int valid_rows = block_valid[pblock] - sub * 32;
    unsigned int q_words[8];
    unsigned int k_words[8];
    unsigned int v_words[8];
    #pragma unroll
    for (int iteration = 0; iteration < 1; iteration++) {
        int vector = tid + iteration * 256;
        int row = vector / 8;
        int group = vector - row * 8;
        int token = first_token + row;
        long long input_offset = ((long long)token * (long long)heads + (long long)head) * 128 + (long long)(group * 16);
        if (row < valid_rows) {
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(q_words[iteration * 8 + 0]), "=r"(q_words[iteration * 8 + 1]), "=r"(q_words[iteration * 8 + 2]), "=r"(q_words[iteration * 8 + 3]), "=r"(q_words[iteration * 8 + 4]), "=r"(q_words[iteration * 8 + 5]), "=r"(q_words[iteration * 8 + 6]), "=r"(q_words[iteration * 8 + 7]) : "l"((const void*)((const char*)(q + input_offset + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(k_words[iteration * 8 + 0]), "=r"(k_words[iteration * 8 + 1]), "=r"(k_words[iteration * 8 + 2]), "=r"(k_words[iteration * 8 + 3]), "=r"(k_words[iteration * 8 + 4]), "=r"(k_words[iteration * 8 + 5]), "=r"(k_words[iteration * 8 + 6]), "=r"(k_words[iteration * 8 + 7]) : "l"((const void*)((const char*)(k + input_offset + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(v_words[iteration * 8 + 0]), "=r"(v_words[iteration * 8 + 1]), "=r"(v_words[iteration * 8 + 2]), "=r"(v_words[iteration * 8 + 3]), "=r"(v_words[iteration * 8 + 4]), "=r"(v_words[iteration * 8 + 5]), "=r"(v_words[iteration * 8 + 6]), "=r"(v_words[iteration * 8 + 7]) : "l"((const void*)((const char*)(v + input_offset + 0) + 0)) : "memory");
            }
        } else {
            #pragma unroll
            for (int word = 0; word < 8; word++) {
                q_words[iteration * 8 + word] = 0;
                k_words[iteration * 8 + word] = 0;
                v_words[iteration * 8 + word] = 0;
            }
        }
    }
    #pragma unroll
    for (int iteration_1 = 0; iteration_1 < 1; iteration_1++) {
        int vector_1 = tid + iteration_1 * 256;
        int row_1 = vector_1 / 8;
        int group_1 = vector_1 - row_1 * 8;
        int row128 = sub * 32 + row_1;
        float _unpack_bf16_0[16];
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[_pair]) << 16);
            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[1 + _pair]) << 16);
            (&_unpack_bf16_0[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[1 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[2 + _pair]) << 16);
            (&_unpack_bf16_0[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[2 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[3 + _pair]) << 16);
            (&_unpack_bf16_0[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[3 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[4 + _pair]) << 16);
            (&_unpack_bf16_0[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[4 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[10 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[5 + _pair]) << 16);
            (&_unpack_bf16_0[10 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[5 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[6 + _pair]) << 16);
            (&_unpack_bf16_0[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[6 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_0[14 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[7 + _pair]) << 16);
            (&_unpack_bf16_0[14 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((q_words + iteration_1 * 8)[7 + _pair]) & 0xffff0000u);
        }
        float _unpack_bf16_1[16];
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[_pair]) << 16);
            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[1 + _pair]) << 16);
            (&_unpack_bf16_1[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[1 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[2 + _pair]) << 16);
            (&_unpack_bf16_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[2 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[3 + _pair]) << 16);
            (&_unpack_bf16_1[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[3 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[4 + _pair]) << 16);
            (&_unpack_bf16_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[4 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[10 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[5 + _pair]) << 16);
            (&_unpack_bf16_1[10 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[5 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[6 + _pair]) << 16);
            (&_unpack_bf16_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[6 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_1[14 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[7 + _pair]) << 16);
            (&_unpack_bf16_1[14 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((k_words + iteration_1 * 8)[7 + _pair]) & 0xffff0000u);
        }
        float value_max = _unpack_bf16_0[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            value_max = max_noftz(value_max, _unpack_bf16_0[_lr]);
        }
        float value_min = _unpack_bf16_0[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            value_min = fminf(value_min, _unpack_bf16_0[_lr]);
        }
        float amax = fmaxf(value_max, -value_min);
        float raw_scale = fmaxf(amax * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_4;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_4) : "f"(0.0f), "f"(raw_scale));
        uint8_t _cvt_satfinite_e4m3_f32_3 = (uint8_t)_e4m3x2_4;
        uint32_t _f16x2_decode_5;
        float rounded_scale;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_5) : "h"(_e4m3x2_4));
        uint16_t _f16_decode_5 = (uint16_t)_f16x2_decode_5;
        asm("cvt.f32.f16 %0, %1;" : "=f"(rounded_scale) : "h"(_f16_decode_5));
        float inverse_scale = approx_rcp(rounded_scale);
        float normalized[16];
        #pragma unroll
        for (int element = 0; element < 16; element++) {
            normalized[element] = _unpack_bf16_0[element] * inverse_scale;
        }
        unsigned int packed[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(normalized[0]), "f"(normalized[1]), "f"(normalized[2]), "f"(normalized[3]), "f"(normalized[4]), "f"(normalized[5]), "f"(normalized[6]), "f"(normalized[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(normalized[8]), "f"(normalized[9]), "f"(normalized[10]), "f"(normalized[11]), "f"(normalized[12]), "f"(normalized[13]), "f"(normalized[14]), "f"(normalized[15]));
        float value_max_0 = _unpack_bf16_1[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            value_max_0 = max_noftz(value_max_0, _unpack_bf16_1[_lr]);
        }
        float value_min_1 = _unpack_bf16_1[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            value_min_1 = fminf(value_min_1, _unpack_bf16_1[_lr]);
        }
        float amax_2 = fmaxf(value_max_0, -value_min_1);
        float raw_scale_3 = fmaxf(amax_2 * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_7;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_7) : "f"(0.0f), "f"(raw_scale_3));
        uint8_t _cvt_satfinite_e4m3_f32_6 = (uint8_t)_e4m3x2_7;
        uint32_t _f16x2_decode_8;
        float rounded_scale_4;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_8) : "h"(_e4m3x2_7));
        uint16_t _f16_decode_8 = (uint16_t)_f16x2_decode_8;
        asm("cvt.f32.f16 %0, %1;" : "=f"(rounded_scale_4) : "h"(_f16_decode_8));
        float inverse_scale_5 = approx_rcp(rounded_scale_4);
        float normalized_6[16];
        #pragma unroll
        for (int element_1 = 0; element_1 < 16; element_1++) {
            normalized_6[element_1] = _unpack_bf16_1[element_1] * inverse_scale_5;
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
            float _shfl_sync_9 = __shfl_sync(4294967295, raw_scale, j, 4);
            q_sf4[j] = _shfl_sync_9;
            float _shfl_sync_10 = __shfl_sync(4294967295, raw_scale_3, j, 4);
            k_sf4[j] = _shfl_sync_10;
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
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float amax_pair[2];
    {
        uint32_t _scalar_bits_11;
        asm volatile("ld.volatile.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_11) : "l"((const void*)(v_amax_partial + (tid))) : "memory");
        amax_pair[0] = __uint_as_float(_scalar_bits_11);
    }
    {
        uint32_t _scalar_bits_12;
        asm volatile("ld.volatile.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_12) : "l"((const void*)(v_amax_partial + (tid + 256))) : "memory");
        amax_pair[1] = __uint_as_float(_scalar_bits_12);
    }
    float amax_local = fmaxf(amax_pair[0], amax_pair[1]);
    float _warp_reduce_13 = amax_local;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_13 = max_noftz(_warp_reduce_13, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_13, offset));
    amax_local = _warp_reduce_13;
    if (lane == 0) {
        amax_smem[warp] = amax_local;
    }
    __syncthreads();
    if (warp == 0) {
        float amax_partial = ((lane < 8) ? amax_smem[lane] : -CUDART_INF_F);
        float _warp_reduce_0 = amax_partial;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = fmaxf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        if (lane == 0) {
            amax_smem[0] = _warp_reduce_0;
        }
    }
    __syncthreads();
    float amax_1 = amax_smem[0];
    float _rcp_14 = approx_rcp(fmaxf(amax_1, 1e-12f));
    float v_inverse_scale = _rcp_14 * 448.0f;
    if (bid_0 == 0) {
        if (tid == 0) {
            *(reinterpret_cast<float*>(v_amax) + (0)) = amax_1;
        }
    }
    #pragma unroll
    for (int iteration_2 = 0; iteration_2 < 1; iteration_2++) {
        int vector_2 = tid + iteration_2 * 256;
        int row_2 = vector_2 / 8;
        int group_2 = vector_2 - row_2 * 8;
        int row128_1 = sub * 32 + row_2;
        float _unpack_bf16_2[16];
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[_pair]) << 16);
            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[1 + _pair]) << 16);
            (&_unpack_bf16_2[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[1 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[2 + _pair]) << 16);
            (&_unpack_bf16_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[2 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[3 + _pair]) << 16);
            (&_unpack_bf16_2[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[3 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[4 + _pair]) << 16);
            (&_unpack_bf16_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[4 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[10 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[5 + _pair]) << 16);
            (&_unpack_bf16_2[10 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[5 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[6 + _pair]) << 16);
            (&_unpack_bf16_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[6 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[14 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[7 + _pair]) << 16);
            (&_unpack_bf16_2[14 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_2 * 8)[7 + _pair]) & 0xffff0000u);
        }
        long long v_output_offset = ((long long)tile * 128 + (long long)row128_1) * 128 + (long long)(group_2 * 16);
        float _scaled_0[16];
        const float2 _scale2_15 = {v_inverse_scale, v_inverse_scale};
        #pragma unroll
        for (int _ls = 0; _ls < 8; _ls++)
            reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_unpack_bf16_2)[_ls], _scale2_15);
        unsigned int _packed_0[4];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(_scaled_0[0]), "f"(_scaled_0[1]),
                                   "f"(_scaled_0[2]), "f"(_scaled_0[3]));
            _packed_0[0] = _packed;
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
                : "=r"(_packed) : "f"(_scaled_0[4]), "f"(_scaled_0[5]),
                                   "f"(_scaled_0[6]), "f"(_scaled_0[7]));
            _packed_0[1] = _packed;
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
                : "=r"(_packed) : "f"(_scaled_0[8]), "f"(_scaled_0[9]),
                                   "f"(_scaled_0[10]), "f"(_scaled_0[11]));
            _packed_0[2] = _packed;
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
                : "=r"(_packed) : "f"(_scaled_0[12]), "f"(_scaled_0[13]),
                                   "f"(_scaled_0[14]), "f"(_scaled_0[15]));
            _packed_0[3] = _packed;
        }
        reinterpret_cast<int4*>(v_fp8 + v_output_offset)[0] = reinterpret_cast<int4*>(_packed_0)[0];
    }
}

} // extern "C"
