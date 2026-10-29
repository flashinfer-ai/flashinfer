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
kernel_cake_minimax_h3_varlen_attention_90b49eb7762c49d34e34(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, uint8_t* __restrict__ q_fp4, uint8_t* __restrict__ k_fp4, uint8_t* __restrict__ q_scale, uint8_t* __restrict__ k_scale, uint8_t* __restrict__ v_fp8, float* __restrict__ v_amax, float* __restrict__ v_amax_partial, int* __restrict__ block_token, int* __restrict__ block_valid, int heads, int PB)
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float amax_pair[2];
    {
        uint32_t _scalar_bits_3;
        asm volatile("ld.volatile.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_3) : "l"((const void*)(v_amax_partial + (tid))) : "memory");
        amax_pair[0] = __uint_as_float(_scalar_bits_3);
    }
    {
        uint32_t _scalar_bits_4;
        asm volatile("ld.volatile.global.b32 %0, [%1];"
            : "=r"(_scalar_bits_4) : "l"((const void*)(v_amax_partial + (tid + 256))) : "memory");
        amax_pair[1] = __uint_as_float(_scalar_bits_4);
    }
    float amax_local = fmaxf(amax_pair[0], amax_pair[1]);
    float _warp_reduce_5 = amax_local;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_5 = max_noftz(_warp_reduce_5, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset));
    amax_local = _warp_reduce_5;
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
    float amax = amax_smem[0];
    float _rcp_6 = approx_rcp(fmaxf(amax, 1e-12f));
    float v_inverse_scale = _rcp_6 * 448.0f;
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
        float _unpack_bf16_2[16];
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[_pair]) << 16);
            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[1 + _pair]) << 16);
            (&_unpack_bf16_2[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[1 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[2 + _pair]) << 16);
            (&_unpack_bf16_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[2 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[3 + _pair]) << 16);
            (&_unpack_bf16_2[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[3 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[4 + _pair]) << 16);
            (&_unpack_bf16_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[4 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[10 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[5 + _pair]) << 16);
            (&_unpack_bf16_2[10 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[5 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[6 + _pair]) << 16);
            (&_unpack_bf16_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[6 + _pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 1; _pair++) {
            (&_unpack_bf16_2[14 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[7 + _pair]) << 16);
            (&_unpack_bf16_2[14 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_words + iteration_1 * 8)[7 + _pair]) & 0xffff0000u);
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
        float amax_0 = fmaxf(value_max, -value_min);
        float raw_scale = fmaxf(amax_0 * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_8;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_8) : "f"(0.0f), "f"(raw_scale));
        uint8_t _cvt_satfinite_e4m3_f32_7 = (uint8_t)_e4m3x2_8;
        uint32_t _f16x2_decode_9;
        float rounded_scale;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_9) : "h"(_e4m3x2_8));
        uint16_t _f16_decode_9 = (uint16_t)_f16x2_decode_9;
        asm("cvt.f32.f16 %0, %1;" : "=f"(rounded_scale) : "h"(_f16_decode_9));
        float inverse_scale = approx_rcp(rounded_scale);
        float normalized[16];
        #pragma unroll
        for (int element = 0; element < 16; element++) {
            normalized[element] = _unpack_bf16_0[element] * inverse_scale;
        }
        unsigned int packed[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(normalized[0]), "f"(normalized[1]), "f"(normalized[2]), "f"(normalized[3]), "f"(normalized[4]), "f"(normalized[5]), "f"(normalized[6]), "f"(normalized[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(normalized[8]), "f"(normalized[9]), "f"(normalized[10]), "f"(normalized[11]), "f"(normalized[12]), "f"(normalized[13]), "f"(normalized[14]), "f"(normalized[15]));
        float value_max_1 = _unpack_bf16_1[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            value_max_1 = max_noftz(value_max_1, _unpack_bf16_1[_lr]);
        }
        float value_min_2 = _unpack_bf16_1[0];
        #pragma unroll
        for (int _lr = 1; _lr < 16; _lr++) {
            value_min_2 = fminf(value_min_2, _unpack_bf16_1[_lr]);
        }
        float amax_3 = fmaxf(value_max_1, -value_min_2);
        float raw_scale_4 = fmaxf(amax_3 * 0.16666667f, 0.001953125f);
        uint16_t _e4m3x2_11;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_11) : "f"(0.0f), "f"(raw_scale_4));
        uint8_t _cvt_satfinite_e4m3_f32_10 = (uint8_t)_e4m3x2_11;
        uint32_t _f16x2_decode_12;
        float rounded_scale_5;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_12) : "h"(_e4m3x2_11));
        uint16_t _f16_decode_12 = (uint16_t)_f16x2_decode_12;
        asm("cvt.f32.f16 %0, %1;" : "=f"(rounded_scale_5) : "h"(_f16_decode_12));
        float inverse_scale_6 = approx_rcp(rounded_scale_5);
        float normalized_7[16];
        #pragma unroll
        for (int element_1 = 0; element_1 < 16; element_1++) {
            normalized_7[element_1] = _unpack_bf16_1[element_1] * inverse_scale_6;
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
            float _shfl_sync_13 = __shfl_sync(4294967295, raw_scale, j, 4);
            q_sf4[j] = _shfl_sync_13;
            float _shfl_sync_14 = __shfl_sync(4294967295, raw_scale_4, j, 4);
            k_sf4[j] = _shfl_sync_14;
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
