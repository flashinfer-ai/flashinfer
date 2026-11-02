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
#define SMEM_SCRATCH_OFF 0
#define SMEM_SCRATCH_STAGE_BYTES 128
#define SMEM_SCRATCH_STRIDE 128
#define SMEM_HANDOFF_OFF 128
#define SMEM_HANDOFF_STAGE_BYTES 8
#define SMEM_HANDOFF_STRIDE 8
#define SMEM_TOTAL 256
#define THREADS 320

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_fused_norm_combine_b18e0a01ecd652ae20b9(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ collective_out, unsigned long long* __restrict__ workspace, int rank, int tokens, float epsilon)
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
    float* scratch = reinterpret_cast<float*>(smem_raw + 0);
    const int scratch_addr = smem + 0;
    unsigned long long* handoff = reinterpret_cast<unsigned long long*>(smem_raw + 128);
    const int handoff_addr = smem + 128;

    // ---- Role: producers ----
    if (warp <= 4) {
        { // producers_main
            if (tid < 80) {
                long long row = (long long)bid * 5120 + (long long)(tid * 64);
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(x + row)));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(residual + row)));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(weight + (tid * 64))));
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int lane_1 = tid;
            unsigned int gamma0[8];
            unsigned int gamma1[8];
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(gamma0[0 + 0]), "=r"(gamma0[0 + 1]), "=r"(gamma0[0 + 2]), "=r"(gamma0[0 + 3]), "=r"(gamma0[0 + 4]), "=r"(gamma0[0 + 5]), "=r"(gamma0[0 + 6]), "=r"(gamma0[0 + 7]) : "l"((const void*)((const char*)(weight + (lane_1 * 16) + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(gamma1[0 + 0]), "=r"(gamma1[0 + 1]), "=r"(gamma1[0 + 2]), "=r"(gamma1[0 + 3]), "=r"(gamma1[0 + 4]), "=r"(gamma1[0 + 5]), "=r"(gamma1[0 + 6]), "=r"(gamma1[0 + 7]) : "l"((const void*)((const char*)(weight + (2560 + lane_1 * 16) + 0) + 0)) : "memory");
            }
            unsigned int next_x0[8];
            unsigned int next_r0[8];
            unsigned int next_x1[8];
            unsigned int next_r1[8];
            long long base = (long long)bid * 5120 + (long long)(lane_1 * 16);
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(next_x0[0 + 0]), "=r"(next_x0[0 + 1]), "=r"(next_x0[0 + 2]), "=r"(next_x0[0 + 3]), "=r"(next_x0[0 + 4]), "=r"(next_x0[0 + 5]), "=r"(next_x0[0 + 6]), "=r"(next_x0[0 + 7]) : "l"((const void*)((const char*)(x + base + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(next_r0[0 + 0]), "=r"(next_r0[0 + 1]), "=r"(next_r0[0 + 2]), "=r"(next_r0[0 + 3]), "=r"(next_r0[0 + 4]), "=r"(next_r0[0 + 5]), "=r"(next_r0[0 + 6]), "=r"(next_r0[0 + 7]) : "l"((const void*)((const char*)(residual + base + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(next_x1[0 + 0]), "=r"(next_x1[0 + 1]), "=r"(next_x1[0 + 2]), "=r"(next_x1[0 + 3]), "=r"(next_x1[0 + 4]), "=r"(next_x1[0 + 5]), "=r"(next_x1[0 + 6]), "=r"(next_x1[0 + 7]) : "l"((const void*)((const char*)(x + (base + 2560) + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(next_r1[0 + 0]), "=r"(next_r1[0 + 1]), "=r"(next_r1[0 + 2]), "=r"(next_r1[0 + 3]), "=r"(next_r1[0 + 4]), "=r"(next_r1[0 + 5]), "=r"(next_r1[0 + 6]), "=r"(next_r1[0 + 7]) : "l"((const void*)((const char*)(residual + (base + 2560) + 0) + 0)) : "memory");
            }
            unsigned long long owner_address = workspace[16 + bid % 8];
            int slots = (tokens + 7) / 8;
            long long handed_base = 0;
            #pragma unroll 1
            for (int token = bid; token < tokens; token += num_bids) {
                long long base_0 = (long long)token * 5120 + (long long)(lane_1 * 16);
                unsigned int propagated_words[8];
                uint32_t _bf16x2_add_0;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(next_x0[0]), "r"(next_r0[0]));
                propagated_words[0] = _bf16x2_add_0;
                uint32_t _bf16x2_add_1;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_1) : "r"(next_x0[1]), "r"(next_r0[1]));
                propagated_words[1] = _bf16x2_add_1;
                uint32_t _bf16x2_add_2;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_2) : "r"(next_x0[2]), "r"(next_r0[2]));
                propagated_words[2] = _bf16x2_add_2;
                uint32_t _bf16x2_add_3;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_3) : "r"(next_x0[3]), "r"(next_r0[3]));
                propagated_words[3] = _bf16x2_add_3;
                uint32_t _bf16x2_add_4;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_4) : "r"(next_x0[4]), "r"(next_r0[4]));
                propagated_words[4] = _bf16x2_add_4;
                uint32_t _bf16x2_add_5;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_5) : "r"(next_x0[5]), "r"(next_r0[5]));
                propagated_words[5] = _bf16x2_add_5;
                uint32_t _bf16x2_add_6;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_6) : "r"(next_x0[6]), "r"(next_r0[6]));
                propagated_words[6] = _bf16x2_add_6;
                uint32_t _bf16x2_add_7;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_7) : "r"(next_x0[7]), "r"(next_r0[7]));
                propagated_words[7] = _bf16x2_add_7;
                float _unpack_bf16_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(propagated_words[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(propagated_words[_pair]) & 0xffff0000u);
                }
                float square_sum = 0.0f;
                #pragma unroll
                for (int element = 0; element < 16; element++) {
                    square_sum = __fmaf_rn(_unpack_bf16_0[element], _unpack_bf16_0[element], square_sum);
                }
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(residual_out + base_0), "r"((propagated_words)[0]), "r"((propagated_words)[1]), "r"((propagated_words)[2]), "r"((propagated_words)[3]), "r"((propagated_words)[4]), "r"((propagated_words)[5]), "r"((propagated_words)[6]), "r"((propagated_words)[7]) : "memory");
                unsigned int propagated_words_1[8];
                uint32_t _bf16x2_add_8;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_8) : "r"(next_x1[0]), "r"(next_r1[0]));
                propagated_words_1[0] = _bf16x2_add_8;
                uint32_t _bf16x2_add_9;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_9) : "r"(next_x1[1]), "r"(next_r1[1]));
                propagated_words_1[1] = _bf16x2_add_9;
                uint32_t _bf16x2_add_10;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_10) : "r"(next_x1[2]), "r"(next_r1[2]));
                propagated_words_1[2] = _bf16x2_add_10;
                uint32_t _bf16x2_add_11;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_11) : "r"(next_x1[3]), "r"(next_r1[3]));
                propagated_words_1[3] = _bf16x2_add_11;
                uint32_t _bf16x2_add_12;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_12) : "r"(next_x1[4]), "r"(next_r1[4]));
                propagated_words_1[4] = _bf16x2_add_12;
                uint32_t _bf16x2_add_13;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_13) : "r"(next_x1[5]), "r"(next_r1[5]));
                propagated_words_1[5] = _bf16x2_add_13;
                uint32_t _bf16x2_add_14;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_14) : "r"(next_x1[6]), "r"(next_r1[6]));
                propagated_words_1[6] = _bf16x2_add_14;
                uint32_t _bf16x2_add_15;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_15) : "r"(next_x1[7]), "r"(next_r1[7]));
                propagated_words_1[7] = _bf16x2_add_15;
                float _unpack_bf16_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(propagated_words_1[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(propagated_words_1[_pair]) & 0xffff0000u);
                }
                float square_sum_2 = 0.0f;
                #pragma unroll
                for (int element_1 = 0; element_1 < 16; element_1++) {
                    square_sum_2 = __fmaf_rn(_unpack_bf16_1[element_1], _unpack_bf16_1[element_1], square_sum_2);
                }
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(residual_out + (base_0 + 2560)), "r"((propagated_words_1)[0]), "r"((propagated_words_1)[1]), "r"((propagated_words_1)[2]), "r"((propagated_words_1)[3]), "r"((propagated_words_1)[4]), "r"((propagated_words_1)[5]), "r"((propagated_words_1)[6]), "r"((propagated_words_1)[7]) : "memory");
                int following = token + num_bids;
                if (following < tokens) {
                    long long base_1 = (long long)following * 5120 + (long long)(lane_1 * 16);
                    {
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(next_x0[0 + 0]), "=r"(next_x0[0 + 1]), "=r"(next_x0[0 + 2]), "=r"(next_x0[0 + 3]), "=r"(next_x0[0 + 4]), "=r"(next_x0[0 + 5]), "=r"(next_x0[0 + 6]), "=r"(next_x0[0 + 7]) : "l"((const void*)((const char*)(x + base_1 + 0) + 0)) : "memory");
                    }
                    {
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(next_r0[0 + 0]), "=r"(next_r0[0 + 1]), "=r"(next_r0[0 + 2]), "=r"(next_r0[0 + 3]), "=r"(next_r0[0 + 4]), "=r"(next_r0[0 + 5]), "=r"(next_r0[0 + 6]), "=r"(next_r0[0 + 7]) : "l"((const void*)((const char*)(residual + base_1 + 0) + 0)) : "memory");
                    }
                    {
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(next_x1[0 + 0]), "=r"(next_x1[0 + 1]), "=r"(next_x1[0 + 2]), "=r"(next_x1[0 + 3]), "=r"(next_x1[0 + 4]), "=r"(next_x1[0 + 5]), "=r"(next_x1[0 + 6]), "=r"(next_x1[0 + 7]) : "l"((const void*)((const char*)(x + (base_1 + 2560) + 0) + 0)) : "memory");
                    }
                    {
                        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(next_r1[0 + 0]), "=r"(next_r1[0 + 1]), "=r"(next_r1[0 + 2]), "=r"(next_r1[0 + 3]), "=r"(next_r1[0 + 4]), "=r"(next_r1[0 + 5]), "=r"(next_r1[0 + 6]), "=r"(next_r1[0 + 7]) : "l"((const void*)((const char*)(residual + (base_1 + 2560) + 0) + 0)) : "memory");
                    }
                }
                float _warp_reduce_0 = square_sum;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                float _warp_reduce_1 = square_sum_2;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                if (lane == 0) {
                    scratch[warp] = _warp_reduce_0;
                    scratch[8 + warp] = _warp_reduce_1;
                }
                asm volatile("barrier.sync 2, 160;" ::: "memory");
                if (warp == 0) {
                    float partial0 = ((lane < 5) ? scratch[lane] : 0.0f);
                    float partial1 = ((lane < 5) ? scratch[8 + lane] : 0.0f);
                    float _warp_reduce_2 = partial0;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
                    float _warp_reduce_3 = partial1;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                    if (lane == 0) {
                        scratch[16] = _warp_reduce_2;
                        scratch[17] = _warp_reduce_3;
                    }
                }
                asm volatile("barrier.sync 2, 160;" ::: "memory");
                float result0 = scratch[16];
                float result1 = scratch[17];
                float mean[16];
                #pragma unroll
                for (int element_2 = 0; element_2 < 16; element_2++) {
                    mean[element_2] = 0.0f;
                }
                float _rsqrt_0 = rsqrtf(result0 / 2560.0f + epsilon);
                float _unpack_bf16_2[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(propagated_words[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(propagated_words[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_3[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(gamma0[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(gamma0[_pair]) & 0xffff0000u);
                }
                float normalized_values[16];
                #pragma unroll
                for (int element_3 = 0; element_3 < 16; element_3++) {
                    normalized_values[element_3] = _unpack_bf16_2[element_3] * _rsqrt_0 * _unpack_bf16_3[element_3];
                }
                unsigned int normalized_values_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized_values[_lp*2 + 0], normalized_values[_lp*2+1 + 0]));
                    normalized_values_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(norm_out + base_0), "r"((normalized_values_bf16)[0]), "r"((normalized_values_bf16)[1]), "r"((normalized_values_bf16)[2]), "r"((normalized_values_bf16)[3]), "r"((normalized_values_bf16)[4]), "r"((normalized_values_bf16)[5]), "r"((normalized_values_bf16)[6]), "r"((normalized_values_bf16)[7]) : "memory");
                float _unpack_bf16_4[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(normalized_values_bf16[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(normalized_values_bf16[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int element_4 = 0; element_4 < 16; element_4++) {
                    mean[element_4] = __fmaf_rn(_unpack_bf16_4[element_4], 0.5f, mean[element_4]);
                }
                float _rsqrt_1 = rsqrtf(result1 / 2560.0f + epsilon);
                float _unpack_bf16_5[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(propagated_words_1[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(propagated_words_1[_pair]) & 0xffff0000u);
                }
                float _unpack_bf16_6[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(gamma1[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(gamma1[_pair]) & 0xffff0000u);
                }
                float normalized_values_3[16];
                #pragma unroll
                for (int element_5 = 0; element_5 < 16; element_5++) {
                    normalized_values_3[element_5] = _unpack_bf16_5[element_5] * _rsqrt_1 * _unpack_bf16_6[element_5];
                }
                unsigned int normalized_values_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized_values_3[_lp*2 + 0], normalized_values_3[_lp*2+1 + 0]));
                    normalized_values_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(norm_out + (base_0 + 2560)), "r"((normalized_values_3_bf16)[0]), "r"((normalized_values_3_bf16)[1]), "r"((normalized_values_3_bf16)[2]), "r"((normalized_values_3_bf16)[3]), "r"((normalized_values_3_bf16)[4]), "r"((normalized_values_3_bf16)[5]), "r"((normalized_values_3_bf16)[6]), "r"((normalized_values_3_bf16)[7]) : "memory");
                float _unpack_bf16_7[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(normalized_values_3_bf16[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(normalized_values_3_bf16[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int element_6 = 0; element_6 < 16; element_6++) {
                    mean[element_6] = __fmaf_rn(_unpack_bf16_7[element_6], 0.5f, mean[element_6]);
                }
                unsigned int mean_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mean[_lp*2 + 0], mean[_lp*2+1 + 0]));
                    mean_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int positive_zero = 0;
                unsigned int contribution_words[8];
                uint32_t _bf16x2_add_16;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_16) : "r"(mean_bf16[0]), "r"(positive_zero));
                contribution_words[0] = _bf16x2_add_16;
                uint32_t _bf16x2_add_17;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_17) : "r"(mean_bf16[1]), "r"(positive_zero));
                contribution_words[1] = _bf16x2_add_17;
                uint32_t _bf16x2_add_18;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_18) : "r"(mean_bf16[2]), "r"(positive_zero));
                contribution_words[2] = _bf16x2_add_18;
                uint32_t _bf16x2_add_19;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_19) : "r"(mean_bf16[3]), "r"(positive_zero));
                contribution_words[3] = _bf16x2_add_19;
                uint32_t _bf16x2_add_20;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_20) : "r"(mean_bf16[4]), "r"(positive_zero));
                contribution_words[4] = _bf16x2_add_20;
                uint32_t _bf16x2_add_21;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_21) : "r"(mean_bf16[5]), "r"(positive_zero));
                contribution_words[5] = _bf16x2_add_21;
                uint32_t _bf16x2_add_22;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_22) : "r"(mean_bf16[6]), "r"(positive_zero));
                contribution_words[6] = _bf16x2_add_22;
                uint32_t _bf16x2_add_23;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_23) : "r"(mean_bf16[7]), "r"(positive_zero));
                contribution_words[7] = _bf16x2_add_23;
                if (token == bid) {
                    asm volatile("barrier.sync 1, 320;" ::: "memory");
                    handed_base = (long long)handoff[0];
                }
                __nv_bfloat16* owner_payload = reinterpret_cast<__nv_bfloat16*>(owner_address);
                long long offset = handed_base + ((long long)(rank * slots + token / 8) * 160 + (long long)lane_1) * 16;
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(owner_payload + offset), "r"((contribution_words)[0]), "r"((contribution_words)[1]), "r"((contribution_words)[2]), "r"((contribution_words)[3]), "r"((contribution_words)[4]), "r"((contribution_words)[5]), "r"((contribution_words)[6]), "r"((contribution_words)[7]) : "memory");
                owner_address = workspace[16 + following % 8];
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: consumers ----
    if (warp >= 5 && warp <= 9) {
        { // consumers_main
            if (tid == 160) {
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(workspace + 16)));
                uint64_t _sysv_u32_0;
                asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(_sysv_u32_0) : "l"(workspace + 24) : "memory");
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<int*>(_sysv_u32_0))));
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned long long control_address = workspace[24];
            int* control_0 = reinterpret_cast<int*>(control_address);
            int control_current_word = control_0[2];
            int control_bytes_per_buffer_word = control_0[3];
            int control_clear_word = control_0[4];
            int control_current = control_current_word % 3;
            int control_dirty = (control_current_word + 2) % 3;
            __nv_bfloat16* local_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + rank]);
            long long data_base = (long long)control_current * (long long)control_bytes_per_buffer_word / 2;
            long long clear_base = (long long)control_dirty * (long long)control_bytes_per_buffer_word / 2;
            int total_vectors = tokens * 160;
            if (tid == 160) {
                handoff[0] = (unsigned long long)data_base;
            }
            asm volatile("barrier.arrive 1, 320;" ::: "memory");
            asm volatile("barrier.sync 3, 160;" ::: "memory");
            if (tid == 160) {
                asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(control_0)) + (0))), "r"(static_cast<unsigned int>(1)) : "memory");
            }
            int slots_1 = (tokens + 7) / 8;
            int lane_1_1 = tid - 160;
            unsigned int sentinel[8];
            #pragma unroll
            for (int word = 0; word < 8; word++) {
                sentinel[word] = 2147516416;
            }
            int clear_start = bid * 160 + lane_1_1;
            #pragma unroll 1
            for (int clear_index = clear_start; clear_index < control_clear_word / 32; clear_index += num_bids * 160) {
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(local_payload + (clear_base + (long long)clear_index * 16)), "r"((sentinel)[0]), "r"((sentinel)[1]), "r"((sentinel)[2]), "r"((sentinel)[3]), "r"((sentinel)[4]), "r"((sentinel)[5]), "r"((sentinel)[6]), "r"((sentinel)[7]) : "memory");
            }
            #pragma unroll 1
            for (int token_1 = bid; token_1 < tokens; token_1 += num_bids) {
                if (token_1 % 8 == rank) {
                    int slot = token_1 / 8;
                    long long offset_1 = data_base + ((long long)slot * 160 + (long long)lane_1_1) * 16;
                    long long offset_0 = data_base + ((long long)(slots_1 + slot) * 160 + (long long)lane_1_1) * 16;
                    long long offset_1_1 = data_base + ((long long)(2 * slots_1 + slot) * 160 + (long long)lane_1_1) * 16;
                    long long offset_2 = data_base + ((long long)(3 * slots_1 + slot) * 160 + (long long)lane_1_1) * 16;
                    long long offset_3 = data_base + ((long long)(4 * slots_1 + slot) * 160 + (long long)lane_1_1) * 16;
                    long long offset_4 = data_base + ((long long)(5 * slots_1 + slot) * 160 + (long long)lane_1_1) * 16;
                    long long offset_5 = data_base + ((long long)(6 * slots_1 + slot) * 160 + (long long)lane_1_1) * 16;
                    long long offset_6 = data_base + ((long long)(7 * slots_1 + slot) * 160 + (long long)lane_1_1) * 16;
                    uint32_t _sysv_poll256_group_0[64];
                    uint32_t _sysv_min_0;
                    do {
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[0]), "=r"(_sysv_poll256_group_0[1]), "=r"(_sysv_poll256_group_0[2]), "=r"(_sysv_poll256_group_0[3]), "=r"(_sysv_poll256_group_0[4]), "=r"(_sysv_poll256_group_0[5]), "=r"(_sysv_poll256_group_0[6]), "=r"(_sysv_poll256_group_0[7]) : "l"(local_payload + offset_1) : "memory");
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[8]), "=r"(_sysv_poll256_group_0[9]), "=r"(_sysv_poll256_group_0[10]), "=r"(_sysv_poll256_group_0[11]), "=r"(_sysv_poll256_group_0[12]), "=r"(_sysv_poll256_group_0[13]), "=r"(_sysv_poll256_group_0[14]), "=r"(_sysv_poll256_group_0[15]) : "l"(local_payload + offset_0) : "memory");
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[16]), "=r"(_sysv_poll256_group_0[17]), "=r"(_sysv_poll256_group_0[18]), "=r"(_sysv_poll256_group_0[19]), "=r"(_sysv_poll256_group_0[20]), "=r"(_sysv_poll256_group_0[21]), "=r"(_sysv_poll256_group_0[22]), "=r"(_sysv_poll256_group_0[23]) : "l"(local_payload + offset_1_1) : "memory");
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[24]), "=r"(_sysv_poll256_group_0[25]), "=r"(_sysv_poll256_group_0[26]), "=r"(_sysv_poll256_group_0[27]), "=r"(_sysv_poll256_group_0[28]), "=r"(_sysv_poll256_group_0[29]), "=r"(_sysv_poll256_group_0[30]), "=r"(_sysv_poll256_group_0[31]) : "l"(local_payload + offset_2) : "memory");
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[32]), "=r"(_sysv_poll256_group_0[33]), "=r"(_sysv_poll256_group_0[34]), "=r"(_sysv_poll256_group_0[35]), "=r"(_sysv_poll256_group_0[36]), "=r"(_sysv_poll256_group_0[37]), "=r"(_sysv_poll256_group_0[38]), "=r"(_sysv_poll256_group_0[39]) : "l"(local_payload + offset_3) : "memory");
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[40]), "=r"(_sysv_poll256_group_0[41]), "=r"(_sysv_poll256_group_0[42]), "=r"(_sysv_poll256_group_0[43]), "=r"(_sysv_poll256_group_0[44]), "=r"(_sysv_poll256_group_0[45]), "=r"(_sysv_poll256_group_0[46]), "=r"(_sysv_poll256_group_0[47]) : "l"(local_payload + offset_4) : "memory");
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[48]), "=r"(_sysv_poll256_group_0[49]), "=r"(_sysv_poll256_group_0[50]), "=r"(_sysv_poll256_group_0[51]), "=r"(_sysv_poll256_group_0[52]), "=r"(_sysv_poll256_group_0[53]), "=r"(_sysv_poll256_group_0[54]), "=r"(_sysv_poll256_group_0[55]) : "l"(local_payload + offset_5) : "memory");
                        asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[56]), "=r"(_sysv_poll256_group_0[57]), "=r"(_sysv_poll256_group_0[58]), "=r"(_sysv_poll256_group_0[59]), "=r"(_sysv_poll256_group_0[60]), "=r"(_sysv_poll256_group_0[61]), "=r"(_sysv_poll256_group_0[62]), "=r"(_sysv_poll256_group_0[63]) : "l"(local_payload + offset_6) : "memory");
                        uint32_t _sysv_min_1_0;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_0) : "r"(_sysv_poll256_group_0[0]), "r"(_sysv_poll256_group_0[1]));
                        uint32_t _sysv_min_1_1;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_1) : "r"(_sysv_poll256_group_0[2]), "r"(_sysv_poll256_group_0[3]));
                        uint32_t _sysv_min_1_2;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_2) : "r"(_sysv_min_1_0), "r"(_sysv_min_1_1));
                        uint32_t _sysv_min_1_3;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_3) : "r"(_sysv_poll256_group_0[4]), "r"(_sysv_poll256_group_0[5]));
                        uint32_t _sysv_min_1_4;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_4) : "r"(_sysv_poll256_group_0[6]), "r"(_sysv_poll256_group_0[7]));
                        uint32_t _sysv_min_1_5;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_5) : "r"(_sysv_min_1_3), "r"(_sysv_min_1_4));
                        uint32_t _sysv_min_1_6;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_6) : "r"(_sysv_min_1_2), "r"(_sysv_min_1_5));
                        uint32_t _sysv_min_1_7;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_7) : "r"(_sysv_poll256_group_0[8]), "r"(_sysv_poll256_group_0[9]));
                        uint32_t _sysv_min_1_8;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_8) : "r"(_sysv_poll256_group_0[10]), "r"(_sysv_poll256_group_0[11]));
                        uint32_t _sysv_min_1_9;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_9) : "r"(_sysv_min_1_7), "r"(_sysv_min_1_8));
                        uint32_t _sysv_min_1_10;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_10) : "r"(_sysv_poll256_group_0[12]), "r"(_sysv_poll256_group_0[13]));
                        uint32_t _sysv_min_1_11;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_11) : "r"(_sysv_poll256_group_0[14]), "r"(_sysv_poll256_group_0[15]));
                        uint32_t _sysv_min_1_12;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_12) : "r"(_sysv_min_1_10), "r"(_sysv_min_1_11));
                        uint32_t _sysv_min_1_13;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_13) : "r"(_sysv_min_1_9), "r"(_sysv_min_1_12));
                        uint32_t _sysv_min_1_14;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_14) : "r"(_sysv_min_1_6), "r"(_sysv_min_1_13));
                        uint32_t _sysv_min_1_15;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_15) : "r"(_sysv_poll256_group_0[16]), "r"(_sysv_poll256_group_0[17]));
                        uint32_t _sysv_min_1_16;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_16) : "r"(_sysv_poll256_group_0[18]), "r"(_sysv_poll256_group_0[19]));
                        uint32_t _sysv_min_1_17;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_17) : "r"(_sysv_min_1_15), "r"(_sysv_min_1_16));
                        uint32_t _sysv_min_1_18;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_18) : "r"(_sysv_poll256_group_0[20]), "r"(_sysv_poll256_group_0[21]));
                        uint32_t _sysv_min_1_19;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_19) : "r"(_sysv_poll256_group_0[22]), "r"(_sysv_poll256_group_0[23]));
                        uint32_t _sysv_min_1_20;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_20) : "r"(_sysv_min_1_18), "r"(_sysv_min_1_19));
                        uint32_t _sysv_min_1_21;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_21) : "r"(_sysv_min_1_17), "r"(_sysv_min_1_20));
                        uint32_t _sysv_min_1_22;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_22) : "r"(_sysv_poll256_group_0[24]), "r"(_sysv_poll256_group_0[25]));
                        uint32_t _sysv_min_1_23;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_23) : "r"(_sysv_poll256_group_0[26]), "r"(_sysv_poll256_group_0[27]));
                        uint32_t _sysv_min_1_24;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_24) : "r"(_sysv_min_1_22), "r"(_sysv_min_1_23));
                        uint32_t _sysv_min_1_25;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_25) : "r"(_sysv_poll256_group_0[28]), "r"(_sysv_poll256_group_0[29]));
                        uint32_t _sysv_min_1_26;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_26) : "r"(_sysv_poll256_group_0[30]), "r"(_sysv_poll256_group_0[31]));
                        uint32_t _sysv_min_1_27;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_27) : "r"(_sysv_min_1_25), "r"(_sysv_min_1_26));
                        uint32_t _sysv_min_1_28;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_28) : "r"(_sysv_min_1_24), "r"(_sysv_min_1_27));
                        uint32_t _sysv_min_1_29;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_29) : "r"(_sysv_min_1_21), "r"(_sysv_min_1_28));
                        uint32_t _sysv_min_1_30;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_30) : "r"(_sysv_min_1_14), "r"(_sysv_min_1_29));
                        uint32_t _sysv_min_1_31;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_31) : "r"(_sysv_poll256_group_0[32]), "r"(_sysv_poll256_group_0[33]));
                        uint32_t _sysv_min_1_32;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_32) : "r"(_sysv_poll256_group_0[34]), "r"(_sysv_poll256_group_0[35]));
                        uint32_t _sysv_min_1_33;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_33) : "r"(_sysv_min_1_31), "r"(_sysv_min_1_32));
                        uint32_t _sysv_min_1_34;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_34) : "r"(_sysv_poll256_group_0[36]), "r"(_sysv_poll256_group_0[37]));
                        uint32_t _sysv_min_1_35;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_35) : "r"(_sysv_poll256_group_0[38]), "r"(_sysv_poll256_group_0[39]));
                        uint32_t _sysv_min_1_36;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_36) : "r"(_sysv_min_1_34), "r"(_sysv_min_1_35));
                        uint32_t _sysv_min_1_37;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_37) : "r"(_sysv_min_1_33), "r"(_sysv_min_1_36));
                        uint32_t _sysv_min_1_38;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_38) : "r"(_sysv_poll256_group_0[40]), "r"(_sysv_poll256_group_0[41]));
                        uint32_t _sysv_min_1_39;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_39) : "r"(_sysv_poll256_group_0[42]), "r"(_sysv_poll256_group_0[43]));
                        uint32_t _sysv_min_1_40;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_40) : "r"(_sysv_min_1_38), "r"(_sysv_min_1_39));
                        uint32_t _sysv_min_1_41;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_41) : "r"(_sysv_poll256_group_0[44]), "r"(_sysv_poll256_group_0[45]));
                        uint32_t _sysv_min_1_42;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_42) : "r"(_sysv_poll256_group_0[46]), "r"(_sysv_poll256_group_0[47]));
                        uint32_t _sysv_min_1_43;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_43) : "r"(_sysv_min_1_41), "r"(_sysv_min_1_42));
                        uint32_t _sysv_min_1_44;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_44) : "r"(_sysv_min_1_40), "r"(_sysv_min_1_43));
                        uint32_t _sysv_min_1_45;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_45) : "r"(_sysv_min_1_37), "r"(_sysv_min_1_44));
                        uint32_t _sysv_min_1_46;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_46) : "r"(_sysv_poll256_group_0[48]), "r"(_sysv_poll256_group_0[49]));
                        uint32_t _sysv_min_1_47;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_47) : "r"(_sysv_poll256_group_0[50]), "r"(_sysv_poll256_group_0[51]));
                        uint32_t _sysv_min_1_48;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_48) : "r"(_sysv_min_1_46), "r"(_sysv_min_1_47));
                        uint32_t _sysv_min_1_49;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_49) : "r"(_sysv_poll256_group_0[52]), "r"(_sysv_poll256_group_0[53]));
                        uint32_t _sysv_min_1_50;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_50) : "r"(_sysv_poll256_group_0[54]), "r"(_sysv_poll256_group_0[55]));
                        uint32_t _sysv_min_1_51;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_51) : "r"(_sysv_min_1_49), "r"(_sysv_min_1_50));
                        uint32_t _sysv_min_1_52;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_52) : "r"(_sysv_min_1_48), "r"(_sysv_min_1_51));
                        uint32_t _sysv_min_1_53;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_53) : "r"(_sysv_poll256_group_0[56]), "r"(_sysv_poll256_group_0[57]));
                        uint32_t _sysv_min_1_54;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_54) : "r"(_sysv_poll256_group_0[58]), "r"(_sysv_poll256_group_0[59]));
                        uint32_t _sysv_min_1_55;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_55) : "r"(_sysv_min_1_53), "r"(_sysv_min_1_54));
                        uint32_t _sysv_min_1_56;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_56) : "r"(_sysv_poll256_group_0[60]), "r"(_sysv_poll256_group_0[61]));
                        uint32_t _sysv_min_1_57;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_57) : "r"(_sysv_poll256_group_0[62]), "r"(_sysv_poll256_group_0[63]));
                        uint32_t _sysv_min_1_58;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_58) : "r"(_sysv_min_1_56), "r"(_sysv_min_1_57));
                        uint32_t _sysv_min_1_59;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_59) : "r"(_sysv_min_1_55), "r"(_sysv_min_1_58));
                        uint32_t _sysv_min_1_60;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_60) : "r"(_sysv_min_1_52), "r"(_sysv_min_1_59));
                        uint32_t _sysv_min_1_61;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_61) : "r"(_sysv_min_1_45), "r"(_sysv_min_1_60));
                        uint32_t _sysv_min_1_62;
                        asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_1_62) : "r"(_sysv_min_1_30), "r"(_sysv_min_1_61));
                        _sysv_min_0 = _sysv_min_1_62;
                    } while (
                        ((_sysv_min_0 & 0xffffu) == 0x8000u) || ((_sysv_min_0 >> 16) == 0x8000u)
                    );
                    unsigned int result_words[8];
                    uint32_t _bf16x2_add_24;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_24) : "r"(_sysv_poll256_group_0[0]), "r"(_sysv_poll256_group_0[8]));
                    result_words[0] = _bf16x2_add_24;
                    uint32_t _bf16x2_add_25;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_25) : "r"(_sysv_poll256_group_0[1]), "r"(_sysv_poll256_group_0[9]));
                    result_words[1] = _bf16x2_add_25;
                    uint32_t _bf16x2_add_26;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_26) : "r"(_sysv_poll256_group_0[2]), "r"(_sysv_poll256_group_0[10]));
                    result_words[2] = _bf16x2_add_26;
                    uint32_t _bf16x2_add_27;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_27) : "r"(_sysv_poll256_group_0[3]), "r"(_sysv_poll256_group_0[11]));
                    result_words[3] = _bf16x2_add_27;
                    uint32_t _bf16x2_add_28;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_28) : "r"(_sysv_poll256_group_0[4]), "r"(_sysv_poll256_group_0[12]));
                    result_words[4] = _bf16x2_add_28;
                    uint32_t _bf16x2_add_29;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_29) : "r"(_sysv_poll256_group_0[5]), "r"(_sysv_poll256_group_0[13]));
                    result_words[5] = _bf16x2_add_29;
                    uint32_t _bf16x2_add_30;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_30) : "r"(_sysv_poll256_group_0[6]), "r"(_sysv_poll256_group_0[14]));
                    result_words[6] = _bf16x2_add_30;
                    uint32_t _bf16x2_add_31;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_31) : "r"(_sysv_poll256_group_0[7]), "r"(_sysv_poll256_group_0[15]));
                    result_words[7] = _bf16x2_add_31;
                    uint32_t _bf16x2_add_32;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_32) : "r"(result_words[0]), "r"(_sysv_poll256_group_0[16]));
                    result_words[0] = _bf16x2_add_32;
                    uint32_t _bf16x2_add_33;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_33) : "r"(result_words[1]), "r"(_sysv_poll256_group_0[17]));
                    result_words[1] = _bf16x2_add_33;
                    uint32_t _bf16x2_add_34;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_34) : "r"(result_words[2]), "r"(_sysv_poll256_group_0[18]));
                    result_words[2] = _bf16x2_add_34;
                    uint32_t _bf16x2_add_35;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_35) : "r"(result_words[3]), "r"(_sysv_poll256_group_0[19]));
                    result_words[3] = _bf16x2_add_35;
                    uint32_t _bf16x2_add_36;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_36) : "r"(result_words[4]), "r"(_sysv_poll256_group_0[20]));
                    result_words[4] = _bf16x2_add_36;
                    uint32_t _bf16x2_add_37;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_37) : "r"(result_words[5]), "r"(_sysv_poll256_group_0[21]));
                    result_words[5] = _bf16x2_add_37;
                    uint32_t _bf16x2_add_38;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_38) : "r"(result_words[6]), "r"(_sysv_poll256_group_0[22]));
                    result_words[6] = _bf16x2_add_38;
                    uint32_t _bf16x2_add_39;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_39) : "r"(result_words[7]), "r"(_sysv_poll256_group_0[23]));
                    result_words[7] = _bf16x2_add_39;
                    uint32_t _bf16x2_add_40;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_40) : "r"(result_words[0]), "r"(_sysv_poll256_group_0[24]));
                    result_words[0] = _bf16x2_add_40;
                    uint32_t _bf16x2_add_41;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_41) : "r"(result_words[1]), "r"(_sysv_poll256_group_0[25]));
                    result_words[1] = _bf16x2_add_41;
                    uint32_t _bf16x2_add_42;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_42) : "r"(result_words[2]), "r"(_sysv_poll256_group_0[26]));
                    result_words[2] = _bf16x2_add_42;
                    uint32_t _bf16x2_add_43;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_43) : "r"(result_words[3]), "r"(_sysv_poll256_group_0[27]));
                    result_words[3] = _bf16x2_add_43;
                    uint32_t _bf16x2_add_44;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_44) : "r"(result_words[4]), "r"(_sysv_poll256_group_0[28]));
                    result_words[4] = _bf16x2_add_44;
                    uint32_t _bf16x2_add_45;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_45) : "r"(result_words[5]), "r"(_sysv_poll256_group_0[29]));
                    result_words[5] = _bf16x2_add_45;
                    uint32_t _bf16x2_add_46;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_46) : "r"(result_words[6]), "r"(_sysv_poll256_group_0[30]));
                    result_words[6] = _bf16x2_add_46;
                    uint32_t _bf16x2_add_47;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_47) : "r"(result_words[7]), "r"(_sysv_poll256_group_0[31]));
                    result_words[7] = _bf16x2_add_47;
                    uint32_t _bf16x2_add_48;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_48) : "r"(result_words[0]), "r"(_sysv_poll256_group_0[32]));
                    result_words[0] = _bf16x2_add_48;
                    uint32_t _bf16x2_add_49;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_49) : "r"(result_words[1]), "r"(_sysv_poll256_group_0[33]));
                    result_words[1] = _bf16x2_add_49;
                    uint32_t _bf16x2_add_50;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_50) : "r"(result_words[2]), "r"(_sysv_poll256_group_0[34]));
                    result_words[2] = _bf16x2_add_50;
                    uint32_t _bf16x2_add_51;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_51) : "r"(result_words[3]), "r"(_sysv_poll256_group_0[35]));
                    result_words[3] = _bf16x2_add_51;
                    uint32_t _bf16x2_add_52;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_52) : "r"(result_words[4]), "r"(_sysv_poll256_group_0[36]));
                    result_words[4] = _bf16x2_add_52;
                    uint32_t _bf16x2_add_53;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_53) : "r"(result_words[5]), "r"(_sysv_poll256_group_0[37]));
                    result_words[5] = _bf16x2_add_53;
                    uint32_t _bf16x2_add_54;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_54) : "r"(result_words[6]), "r"(_sysv_poll256_group_0[38]));
                    result_words[6] = _bf16x2_add_54;
                    uint32_t _bf16x2_add_55;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_55) : "r"(result_words[7]), "r"(_sysv_poll256_group_0[39]));
                    result_words[7] = _bf16x2_add_55;
                    uint32_t _bf16x2_add_56;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_56) : "r"(result_words[0]), "r"(_sysv_poll256_group_0[40]));
                    result_words[0] = _bf16x2_add_56;
                    uint32_t _bf16x2_add_57;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_57) : "r"(result_words[1]), "r"(_sysv_poll256_group_0[41]));
                    result_words[1] = _bf16x2_add_57;
                    uint32_t _bf16x2_add_58;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_58) : "r"(result_words[2]), "r"(_sysv_poll256_group_0[42]));
                    result_words[2] = _bf16x2_add_58;
                    uint32_t _bf16x2_add_59;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_59) : "r"(result_words[3]), "r"(_sysv_poll256_group_0[43]));
                    result_words[3] = _bf16x2_add_59;
                    uint32_t _bf16x2_add_60;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_60) : "r"(result_words[4]), "r"(_sysv_poll256_group_0[44]));
                    result_words[4] = _bf16x2_add_60;
                    uint32_t _bf16x2_add_61;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_61) : "r"(result_words[5]), "r"(_sysv_poll256_group_0[45]));
                    result_words[5] = _bf16x2_add_61;
                    uint32_t _bf16x2_add_62;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_62) : "r"(result_words[6]), "r"(_sysv_poll256_group_0[46]));
                    result_words[6] = _bf16x2_add_62;
                    uint32_t _bf16x2_add_63;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_63) : "r"(result_words[7]), "r"(_sysv_poll256_group_0[47]));
                    result_words[7] = _bf16x2_add_63;
                    uint32_t _bf16x2_add_64;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_64) : "r"(result_words[0]), "r"(_sysv_poll256_group_0[48]));
                    result_words[0] = _bf16x2_add_64;
                    uint32_t _bf16x2_add_65;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_65) : "r"(result_words[1]), "r"(_sysv_poll256_group_0[49]));
                    result_words[1] = _bf16x2_add_65;
                    uint32_t _bf16x2_add_66;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_66) : "r"(result_words[2]), "r"(_sysv_poll256_group_0[50]));
                    result_words[2] = _bf16x2_add_66;
                    uint32_t _bf16x2_add_67;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_67) : "r"(result_words[3]), "r"(_sysv_poll256_group_0[51]));
                    result_words[3] = _bf16x2_add_67;
                    uint32_t _bf16x2_add_68;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_68) : "r"(result_words[4]), "r"(_sysv_poll256_group_0[52]));
                    result_words[4] = _bf16x2_add_68;
                    uint32_t _bf16x2_add_69;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_69) : "r"(result_words[5]), "r"(_sysv_poll256_group_0[53]));
                    result_words[5] = _bf16x2_add_69;
                    uint32_t _bf16x2_add_70;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_70) : "r"(result_words[6]), "r"(_sysv_poll256_group_0[54]));
                    result_words[6] = _bf16x2_add_70;
                    uint32_t _bf16x2_add_71;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_71) : "r"(result_words[7]), "r"(_sysv_poll256_group_0[55]));
                    result_words[7] = _bf16x2_add_71;
                    uint32_t _bf16x2_add_72;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_72) : "r"(result_words[0]), "r"(_sysv_poll256_group_0[56]));
                    result_words[0] = _bf16x2_add_72;
                    uint32_t _bf16x2_add_73;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_73) : "r"(result_words[1]), "r"(_sysv_poll256_group_0[57]));
                    result_words[1] = _bf16x2_add_73;
                    uint32_t _bf16x2_add_74;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_74) : "r"(result_words[2]), "r"(_sysv_poll256_group_0[58]));
                    result_words[2] = _bf16x2_add_74;
                    uint32_t _bf16x2_add_75;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_75) : "r"(result_words[3]), "r"(_sysv_poll256_group_0[59]));
                    result_words[3] = _bf16x2_add_75;
                    uint32_t _bf16x2_add_76;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_76) : "r"(result_words[4]), "r"(_sysv_poll256_group_0[60]));
                    result_words[4] = _bf16x2_add_76;
                    uint32_t _bf16x2_add_77;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_77) : "r"(result_words[5]), "r"(_sysv_poll256_group_0[61]));
                    result_words[5] = _bf16x2_add_77;
                    uint32_t _bf16x2_add_78;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_78) : "r"(result_words[6]), "r"(_sysv_poll256_group_0[62]));
                    result_words[6] = _bf16x2_add_78;
                    uint32_t _bf16x2_add_79;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_79) : "r"(result_words[7]), "r"(_sysv_poll256_group_0[63]));
                    result_words[7] = _bf16x2_add_79;
                    long long offset_7 = data_base + ((long long)(8 * slots_1 + token_1) * 160 + (long long)lane_1_1) * 16;
                    #pragma unroll
                    for (int peer = 0; peer < 8; peer++) {
                        __nv_bfloat16* peer_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + peer]);
                        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(peer_payload + offset_7), "r"((result_words)[0]), "r"((result_words)[1]), "r"((result_words)[2]), "r"((result_words)[3]), "r"((result_words)[4]), "r"((result_words)[5]), "r"((result_words)[6]), "r"((result_words)[7]) : "memory");
                    }
                }
            }
            #pragma unroll 1
            for (int token_2 = bid; token_2 < tokens; token_2 += num_bids) {
                long long offset_8 = data_base + ((long long)(8 * slots_1 + token_2) * 160 + (long long)lane_1_1) * 16;
                uint32_t _sysv_poll256_group_1[8];
                uint32_t _sysv_min_2;
                do {
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_1[0]), "=r"(_sysv_poll256_group_1[1]), "=r"(_sysv_poll256_group_1[2]), "=r"(_sysv_poll256_group_1[3]), "=r"(_sysv_poll256_group_1[4]), "=r"(_sysv_poll256_group_1[5]), "=r"(_sysv_poll256_group_1[6]), "=r"(_sysv_poll256_group_1[7]) : "l"(local_payload + offset_8) : "memory");
                    uint32_t _sysv_min_3_0;
                    asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_3_0) : "r"(_sysv_poll256_group_1[0]), "r"(_sysv_poll256_group_1[1]));
                    uint32_t _sysv_min_3_1;
                    asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_3_1) : "r"(_sysv_poll256_group_1[2]), "r"(_sysv_poll256_group_1[3]));
                    uint32_t _sysv_min_3_2;
                    asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_3_2) : "r"(_sysv_min_3_0), "r"(_sysv_min_3_1));
                    uint32_t _sysv_min_3_3;
                    asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_3_3) : "r"(_sysv_poll256_group_1[4]), "r"(_sysv_poll256_group_1[5]));
                    uint32_t _sysv_min_3_4;
                    asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_3_4) : "r"(_sysv_poll256_group_1[6]), "r"(_sysv_poll256_group_1[7]));
                    uint32_t _sysv_min_3_5;
                    asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_3_5) : "r"(_sysv_min_3_3), "r"(_sysv_min_3_4));
                    uint32_t _sysv_min_3_6;
                    asm("min.s16x2 %0, %1, %2;" : "=r"(_sysv_min_3_6) : "r"(_sysv_min_3_2), "r"(_sysv_min_3_5));
                    _sysv_min_2 = _sysv_min_3_6;
                } while (
                    ((_sysv_min_2 & 0xffffu) == 0x8000u) || ((_sysv_min_2 >> 16) == 0x8000u)
                );
                unsigned int final_words[8];
                final_words[0] = _sysv_poll256_group_1[0];
                final_words[1] = _sysv_poll256_group_1[1];
                final_words[2] = _sysv_poll256_group_1[2];
                final_words[3] = _sysv_poll256_group_1[3];
                final_words[4] = _sysv_poll256_group_1[4];
                final_words[5] = _sysv_poll256_group_1[5];
                final_words[6] = _sysv_poll256_group_1[6];
                final_words[7] = _sysv_poll256_group_1[7];
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + (((long long)token_2 * 160 + (long long)lane_1_1) * 16)), "r"((final_words)[0]), "r"((final_words)[1]), "r"((final_words)[2]), "r"((final_words)[3]), "r"((final_words)[4]), "r"((final_words)[5]), "r"((final_words)[6]), "r"((final_words)[7]) : "memory");
            }
            if (bid == 0) {
                if (lane_1_1 == 0) {
                    int clear_bytes = (8 * slots_1 + tokens) * 5120;
                    {
                        volatile int* _sysv_wait_p_4 = reinterpret_cast<volatile int*>(reinterpret_cast<unsigned int*>(control_0)) + (0);
                        while (*_sysv_wait_p_4 != static_cast<int>(num_bids)) {}
                    }
                    control_0[2] = (control_current_word + 1) % 3;
                    control_0[4] = clear_bytes;
                    reinterpret_cast<unsigned int*>(control_0)[0] = 0;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
}

} // extern "C"
