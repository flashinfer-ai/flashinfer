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
#define SMEM_TOTAL 128
#define THREADS 160
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) void
kernel_cake_fused_norm_combine_8d72efaafce77c1ae5f9(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ collective_out, unsigned long long* __restrict__ workspace, int rank, int tokens, float epsilon)
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

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int x0[8];
    unsigned int r0[8];
    unsigned int x1[8];
    unsigned int r1[8];
    unsigned int gamma_words0[8];
    unsigned int gamma_words1[8];
    if (bid < tokens) {
        long long base = (long long)bid * 5120 + (long long)(tid * 16);
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(x0[0 + 0]), "=r"(x0[0 + 1]), "=r"(x0[0 + 2]), "=r"(x0[0 + 3]), "=r"(x0[0 + 4]), "=r"(x0[0 + 5]), "=r"(x0[0 + 6]), "=r"(x0[0 + 7]) : "l"((const void*)((const char*)(x + base + 0) + 0)) : "memory");
        }
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(r0[0 + 0]), "=r"(r0[0 + 1]), "=r"(r0[0 + 2]), "=r"(r0[0 + 3]), "=r"(r0[0 + 4]), "=r"(r0[0 + 5]), "=r"(r0[0 + 6]), "=r"(r0[0 + 7]) : "l"((const void*)((const char*)(residual + base + 0) + 0)) : "memory");
        }
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(x1[0 + 0]), "=r"(x1[0 + 1]), "=r"(x1[0 + 2]), "=r"(x1[0 + 3]), "=r"(x1[0 + 4]), "=r"(x1[0 + 5]), "=r"(x1[0 + 6]), "=r"(x1[0 + 7]) : "l"((const void*)((const char*)(x + (base + 2560) + 0) + 0)) : "memory");
        }
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(r1[0 + 0]), "=r"(r1[0 + 1]), "=r"(r1[0 + 2]), "=r"(r1[0 + 3]), "=r"(r1[0 + 4]), "=r"(r1[0 + 5]), "=r"(r1[0 + 6]), "=r"(r1[0 + 7]) : "l"((const void*)((const char*)(residual + (base + 2560) + 0) + 0)) : "memory");
        }
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(gamma_words0[0 + 0]), "=r"(gamma_words0[0 + 1]), "=r"(gamma_words0[0 + 2]), "=r"(gamma_words0[0 + 3]), "=r"(gamma_words0[0 + 4]), "=r"(gamma_words0[0 + 5]), "=r"(gamma_words0[0 + 6]), "=r"(gamma_words0[0 + 7]) : "l"((const void*)((const char*)(weight + (tid * 16) + 0) + 0)) : "memory");
        }
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(gamma_words1[0 + 0]), "=r"(gamma_words1[0 + 1]), "=r"(gamma_words1[0 + 2]), "=r"(gamma_words1[0 + 3]), "=r"(gamma_words1[0 + 4]), "=r"(gamma_words1[0 + 5]), "=r"(gamma_words1[0 + 6]), "=r"(gamma_words1[0 + 7]) : "l"((const void*)((const char*)(weight + (2560 + tid * 16) + 0) + 0)) : "memory");
        }
    }
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
    __syncthreads();
    if (tid == 0) {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(control_0)) + (0))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
    int slots = (tokens + 7) / 8;
    #pragma unroll 1
    for (int token = bid; token < tokens; token += num_bids) {
        long long base_1 = (long long)token * 5120 + (long long)(tid * 16);
        unsigned int propagated_words[8];
        uint32_t _bf16x2_add_0;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(x0[0]), "r"(r0[0]));
        propagated_words[0] = _bf16x2_add_0;
        uint32_t _bf16x2_add_1;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_1) : "r"(x0[1]), "r"(r0[1]));
        propagated_words[1] = _bf16x2_add_1;
        uint32_t _bf16x2_add_2;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_2) : "r"(x0[2]), "r"(r0[2]));
        propagated_words[2] = _bf16x2_add_2;
        uint32_t _bf16x2_add_3;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_3) : "r"(x0[3]), "r"(r0[3]));
        propagated_words[3] = _bf16x2_add_3;
        uint32_t _bf16x2_add_4;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_4) : "r"(x0[4]), "r"(r0[4]));
        propagated_words[4] = _bf16x2_add_4;
        uint32_t _bf16x2_add_5;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_5) : "r"(x0[5]), "r"(r0[5]));
        propagated_words[5] = _bf16x2_add_5;
        uint32_t _bf16x2_add_6;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_6) : "r"(x0[6]), "r"(r0[6]));
        propagated_words[6] = _bf16x2_add_6;
        uint32_t _bf16x2_add_7;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_7) : "r"(x0[7]), "r"(r0[7]));
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
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(residual_out + base_1), "r"((propagated_words)[0]), "r"((propagated_words)[1]), "r"((propagated_words)[2]), "r"((propagated_words)[3]), "r"((propagated_words)[4]), "r"((propagated_words)[5]), "r"((propagated_words)[6]), "r"((propagated_words)[7]) : "memory");
        unsigned int propagated_words_0[8];
        uint32_t _bf16x2_add_8;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_8) : "r"(x1[0]), "r"(r1[0]));
        propagated_words_0[0] = _bf16x2_add_8;
        uint32_t _bf16x2_add_9;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_9) : "r"(x1[1]), "r"(r1[1]));
        propagated_words_0[1] = _bf16x2_add_9;
        uint32_t _bf16x2_add_10;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_10) : "r"(x1[2]), "r"(r1[2]));
        propagated_words_0[2] = _bf16x2_add_10;
        uint32_t _bf16x2_add_11;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_11) : "r"(x1[3]), "r"(r1[3]));
        propagated_words_0[3] = _bf16x2_add_11;
        uint32_t _bf16x2_add_12;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_12) : "r"(x1[4]), "r"(r1[4]));
        propagated_words_0[4] = _bf16x2_add_12;
        uint32_t _bf16x2_add_13;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_13) : "r"(x1[5]), "r"(r1[5]));
        propagated_words_0[5] = _bf16x2_add_13;
        uint32_t _bf16x2_add_14;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_14) : "r"(x1[6]), "r"(r1[6]));
        propagated_words_0[6] = _bf16x2_add_14;
        uint32_t _bf16x2_add_15;
        asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_15) : "r"(x1[7]), "r"(r1[7]));
        propagated_words_0[7] = _bf16x2_add_15;
        float _unpack_bf16_1[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(propagated_words_0[_pair]) << 16);
            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(propagated_words_0[_pair]) & 0xffff0000u);
        }
        float square_sum_1 = 0.0f;
        #pragma unroll
        for (int element_1 = 0; element_1 < 16; element_1++) {
            square_sum_1 = __fmaf_rn(_unpack_bf16_1[element_1], _unpack_bf16_1[element_1], square_sum_1);
        }
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(residual_out + (base_1 + 2560)), "r"((propagated_words_0)[0]), "r"((propagated_words_0)[1]), "r"((propagated_words_0)[2]), "r"((propagated_words_0)[3]), "r"((propagated_words_0)[4]), "r"((propagated_words_0)[5]), "r"((propagated_words_0)[6]), "r"((propagated_words_0)[7]) : "memory");
        float _warp_reduce_0 = square_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float _warp_reduce_1 = square_sum_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        if (lane == 0) {
            scratch[warp] = _warp_reduce_0;
            scratch[8 + warp] = _warp_reduce_1;
        }
        __syncthreads();
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
        __syncthreads();
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
            (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(gamma_words0[_pair]) << 16);
            (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(gamma_words0[_pair]) & 0xffff0000u);
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
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(norm_out + base_1), "r"((normalized_values_bf16)[0]), "r"((normalized_values_bf16)[1]), "r"((normalized_values_bf16)[2]), "r"((normalized_values_bf16)[3]), "r"((normalized_values_bf16)[4]), "r"((normalized_values_bf16)[5]), "r"((normalized_values_bf16)[6]), "r"((normalized_values_bf16)[7]) : "memory");
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
            (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(propagated_words_0[_pair]) << 16);
            (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(propagated_words_0[_pair]) & 0xffff0000u);
        }
        float _unpack_bf16_6[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(gamma_words1[_pair]) << 16);
            (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(gamma_words1[_pair]) & 0xffff0000u);
        }
        float normalized_values_2[16];
        #pragma unroll
        for (int element_5 = 0; element_5 < 16; element_5++) {
            normalized_values_2[element_5] = _unpack_bf16_5[element_5] * _rsqrt_1 * _unpack_bf16_6[element_5];
        }
        unsigned int normalized_values_2_bf16[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized_values_2[_lp*2 + 0], normalized_values_2[_lp*2+1 + 0]));
            normalized_values_2_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(norm_out + (base_1 + 2560)), "r"((normalized_values_2_bf16)[0]), "r"((normalized_values_2_bf16)[1]), "r"((normalized_values_2_bf16)[2]), "r"((normalized_values_2_bf16)[3]), "r"((normalized_values_2_bf16)[4]), "r"((normalized_values_2_bf16)[5]), "r"((normalized_values_2_bf16)[6]), "r"((normalized_values_2_bf16)[7]) : "memory");
        float _unpack_bf16_7[16];
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(normalized_values_2_bf16[_pair]) << 16);
            (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(normalized_values_2_bf16[_pair]) & 0xffff0000u);
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
        int owner = token % 8;
        __nv_bfloat16* owner_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + owner]);
        long long offset = data_base + ((long long)(rank * slots + token / 8) * 160 + (long long)tid) * 16;
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(owner_payload + offset), "r"((contribution_words)[0]), "r"((contribution_words)[1]), "r"((contribution_words)[2]), "r"((contribution_words)[3]), "r"((contribution_words)[4]), "r"((contribution_words)[5]), "r"((contribution_words)[6]), "r"((contribution_words)[7]) : "memory");
        int previous = token - 3 * num_bids;
        if (previous >= 0) {
            if (previous % 8 == rank) {
                int slot = previous / 8;
                long long offset_0 = data_base + ((long long)slot * 160 + (long long)tid) * 16;
                long long offset_1 = data_base + ((long long)(slots + slot) * 160 + (long long)tid) * 16;
                long long offset_2 = data_base + ((long long)(2 * slots + slot) * 160 + (long long)tid) * 16;
                long long offset_3 = data_base + ((long long)(3 * slots + slot) * 160 + (long long)tid) * 16;
                long long offset_4 = data_base + ((long long)(4 * slots + slot) * 160 + (long long)tid) * 16;
                long long offset_5 = data_base + ((long long)(5 * slots + slot) * 160 + (long long)tid) * 16;
                long long offset_6 = data_base + ((long long)(6 * slots + slot) * 160 + (long long)tid) * 16;
                long long offset_7 = data_base + ((long long)(7 * slots + slot) * 160 + (long long)tid) * 16;
                uint32_t _sysv_poll256_group_0[64];
                do {
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[0]), "=r"(_sysv_poll256_group_0[1]), "=r"(_sysv_poll256_group_0[2]), "=r"(_sysv_poll256_group_0[3]), "=r"(_sysv_poll256_group_0[4]), "=r"(_sysv_poll256_group_0[5]), "=r"(_sysv_poll256_group_0[6]), "=r"(_sysv_poll256_group_0[7]) : "l"(local_payload + offset_0) : "memory");
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[8]), "=r"(_sysv_poll256_group_0[9]), "=r"(_sysv_poll256_group_0[10]), "=r"(_sysv_poll256_group_0[11]), "=r"(_sysv_poll256_group_0[12]), "=r"(_sysv_poll256_group_0[13]), "=r"(_sysv_poll256_group_0[14]), "=r"(_sysv_poll256_group_0[15]) : "l"(local_payload + offset_1) : "memory");
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[16]), "=r"(_sysv_poll256_group_0[17]), "=r"(_sysv_poll256_group_0[18]), "=r"(_sysv_poll256_group_0[19]), "=r"(_sysv_poll256_group_0[20]), "=r"(_sysv_poll256_group_0[21]), "=r"(_sysv_poll256_group_0[22]), "=r"(_sysv_poll256_group_0[23]) : "l"(local_payload + offset_2) : "memory");
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[24]), "=r"(_sysv_poll256_group_0[25]), "=r"(_sysv_poll256_group_0[26]), "=r"(_sysv_poll256_group_0[27]), "=r"(_sysv_poll256_group_0[28]), "=r"(_sysv_poll256_group_0[29]), "=r"(_sysv_poll256_group_0[30]), "=r"(_sysv_poll256_group_0[31]) : "l"(local_payload + offset_3) : "memory");
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[32]), "=r"(_sysv_poll256_group_0[33]), "=r"(_sysv_poll256_group_0[34]), "=r"(_sysv_poll256_group_0[35]), "=r"(_sysv_poll256_group_0[36]), "=r"(_sysv_poll256_group_0[37]), "=r"(_sysv_poll256_group_0[38]), "=r"(_sysv_poll256_group_0[39]) : "l"(local_payload + offset_4) : "memory");
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[40]), "=r"(_sysv_poll256_group_0[41]), "=r"(_sysv_poll256_group_0[42]), "=r"(_sysv_poll256_group_0[43]), "=r"(_sysv_poll256_group_0[44]), "=r"(_sysv_poll256_group_0[45]), "=r"(_sysv_poll256_group_0[46]), "=r"(_sysv_poll256_group_0[47]) : "l"(local_payload + offset_5) : "memory");
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[48]), "=r"(_sysv_poll256_group_0[49]), "=r"(_sysv_poll256_group_0[50]), "=r"(_sysv_poll256_group_0[51]), "=r"(_sysv_poll256_group_0[52]), "=r"(_sysv_poll256_group_0[53]), "=r"(_sysv_poll256_group_0[54]), "=r"(_sysv_poll256_group_0[55]) : "l"(local_payload + offset_6) : "memory");
                    asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_0[56]), "=r"(_sysv_poll256_group_0[57]), "=r"(_sysv_poll256_group_0[58]), "=r"(_sysv_poll256_group_0[59]), "=r"(_sysv_poll256_group_0[60]), "=r"(_sysv_poll256_group_0[61]), "=r"(_sysv_poll256_group_0[62]), "=r"(_sysv_poll256_group_0[63]) : "l"(local_payload + offset_7) : "memory");
                } while (
                    (((_sysv_poll256_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[1] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[3] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[5] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[7] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[9] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[11] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[13] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[15] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[17] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[19] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[21] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[23] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[25] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[27] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[29] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[31] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[33] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[35] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[37] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[39] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[41] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[43] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[45] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[47] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[49] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[51] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[53] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[55] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[57] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[59] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[61] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll256_group_0[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_0[63] >> 16) & 0xffffu) == 0x8000u)
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
                long long offset_8 = data_base + ((long long)(8 * slots + previous) * 160 + (long long)tid) * 16;
                #pragma unroll
                for (int peer = 0; peer < 8; peer++) {
                    __nv_bfloat16* peer_payload = reinterpret_cast<__nv_bfloat16*>(workspace[16 + peer]);
                    asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(peer_payload + offset_8), "r"((result_words)[0]), "r"((result_words)[1]), "r"((result_words)[2]), "r"((result_words)[3]), "r"((result_words)[4]), "r"((result_words)[5]), "r"((result_words)[6]), "r"((result_words)[7]) : "memory");
                }
            }
        }
        int earlier = token - 4 * num_bids;
        if (earlier >= 0) {
            long long offset_0_1 = data_base + ((long long)(8 * slots + earlier) * 160 + (long long)tid) * 16;
            uint32_t _sysv_poll256_group_1[8];
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_1[0]), "=r"(_sysv_poll256_group_1[1]), "=r"(_sysv_poll256_group_1[2]), "=r"(_sysv_poll256_group_1[3]), "=r"(_sysv_poll256_group_1[4]), "=r"(_sysv_poll256_group_1[5]), "=r"(_sysv_poll256_group_1[6]), "=r"(_sysv_poll256_group_1[7]) : "l"(local_payload + offset_0_1) : "memory");
            } while (
                (((_sysv_poll256_group_1[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_1[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[3] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_1[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[5] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_1[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_1[7] >> 16) & 0xffffu) == 0x8000u)
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
            asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + (((long long)earlier * 160 + (long long)tid) * 16)), "r"((final_words)[0]), "r"((final_words)[1]), "r"((final_words)[2]), "r"((final_words)[3]), "r"((final_words)[4]), "r"((final_words)[5]), "r"((final_words)[6]), "r"((final_words)[7]) : "memory");
        }
        int following = token + num_bids;
        if (following < tokens) {
            long long base_0 = (long long)following * 5120 + (long long)(tid * 16);
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(x0[0 + 0]), "=r"(x0[0 + 1]), "=r"(x0[0 + 2]), "=r"(x0[0 + 3]), "=r"(x0[0 + 4]), "=r"(x0[0 + 5]), "=r"(x0[0 + 6]), "=r"(x0[0 + 7]) : "l"((const void*)((const char*)(x + base_0 + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(r0[0 + 0]), "=r"(r0[0 + 1]), "=r"(r0[0 + 2]), "=r"(r0[0 + 3]), "=r"(r0[0 + 4]), "=r"(r0[0 + 5]), "=r"(r0[0 + 6]), "=r"(r0[0 + 7]) : "l"((const void*)((const char*)(residual + base_0 + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(x1[0 + 0]), "=r"(x1[0 + 1]), "=r"(x1[0 + 2]), "=r"(x1[0 + 3]), "=r"(x1[0 + 4]), "=r"(x1[0 + 5]), "=r"(x1[0 + 6]), "=r"(x1[0 + 7]) : "l"((const void*)((const char*)(x + (base_0 + 2560) + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(r1[0 + 0]), "=r"(r1[0 + 1]), "=r"(r1[0 + 2]), "=r"(r1[0 + 3]), "=r"(r1[0 + 4]), "=r"(r1[0 + 5]), "=r"(r1[0 + 6]), "=r"(r1[0 + 7]) : "l"((const void*)((const char*)(residual + (base_0 + 2560) + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(gamma_words0[0 + 0]), "=r"(gamma_words0[0 + 1]), "=r"(gamma_words0[0 + 2]), "=r"(gamma_words0[0 + 3]), "=r"(gamma_words0[0 + 4]), "=r"(gamma_words0[0 + 5]), "=r"(gamma_words0[0 + 6]), "=r"(gamma_words0[0 + 7]) : "l"((const void*)((const char*)(weight + (tid * 16) + 0) + 0)) : "memory");
            }
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(gamma_words1[0 + 0]), "=r"(gamma_words1[0 + 1]), "=r"(gamma_words1[0 + 2]), "=r"(gamma_words1[0 + 3]), "=r"(gamma_words1[0 + 4]), "=r"(gamma_words1[0 + 5]), "=r"(gamma_words1[0 + 6]), "=r"(gamma_words1[0 + 7]) : "l"((const void*)((const char*)(weight + (2560 + tid * 16) + 0) + 0)) : "memory");
            }
        }
    }
    unsigned int sentinel[8];
    #pragma unroll
    for (int word = 0; word < 8; word++) {
        sentinel[word] = 2147516416;
    }
    int clear_start = bid * 160 + tid;
    #pragma unroll 1
    for (int clear_index = clear_start; clear_index < control_clear_word / 32; clear_index += num_bids * 160) {
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(local_payload + (clear_base + (long long)clear_index * 16)), "r"((sentinel)[0]), "r"((sentinel)[1]), "r"((sentinel)[2]), "r"((sentinel)[3]), "r"((sentinel)[4]), "r"((sentinel)[5]), "r"((sentinel)[6]), "r"((sentinel)[7]) : "memory");
    }
    int last = bid + (tokens - 1 - bid) / num_bids * num_bids;
    int pending = last - 2 * num_bids;
    if (pending >= 0) {
        if (pending % 8 == rank) {
            int slot_1 = pending / 8;
            long long offset_9 = data_base + ((long long)slot_1 * 160 + (long long)tid) * 16;
            long long offset_0_2 = data_base + ((long long)(slots + slot_1) * 160 + (long long)tid) * 16;
            long long offset_1_1 = data_base + ((long long)(2 * slots + slot_1) * 160 + (long long)tid) * 16;
            long long offset_2_1 = data_base + ((long long)(3 * slots + slot_1) * 160 + (long long)tid) * 16;
            long long offset_3_1 = data_base + ((long long)(4 * slots + slot_1) * 160 + (long long)tid) * 16;
            long long offset_4_1 = data_base + ((long long)(5 * slots + slot_1) * 160 + (long long)tid) * 16;
            long long offset_5_1 = data_base + ((long long)(6 * slots + slot_1) * 160 + (long long)tid) * 16;
            long long offset_6_1 = data_base + ((long long)(7 * slots + slot_1) * 160 + (long long)tid) * 16;
            uint32_t _sysv_poll256_group_2[64];
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[0]), "=r"(_sysv_poll256_group_2[1]), "=r"(_sysv_poll256_group_2[2]), "=r"(_sysv_poll256_group_2[3]), "=r"(_sysv_poll256_group_2[4]), "=r"(_sysv_poll256_group_2[5]), "=r"(_sysv_poll256_group_2[6]), "=r"(_sysv_poll256_group_2[7]) : "l"(local_payload + offset_9) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[8]), "=r"(_sysv_poll256_group_2[9]), "=r"(_sysv_poll256_group_2[10]), "=r"(_sysv_poll256_group_2[11]), "=r"(_sysv_poll256_group_2[12]), "=r"(_sysv_poll256_group_2[13]), "=r"(_sysv_poll256_group_2[14]), "=r"(_sysv_poll256_group_2[15]) : "l"(local_payload + offset_0_2) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[16]), "=r"(_sysv_poll256_group_2[17]), "=r"(_sysv_poll256_group_2[18]), "=r"(_sysv_poll256_group_2[19]), "=r"(_sysv_poll256_group_2[20]), "=r"(_sysv_poll256_group_2[21]), "=r"(_sysv_poll256_group_2[22]), "=r"(_sysv_poll256_group_2[23]) : "l"(local_payload + offset_1_1) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[24]), "=r"(_sysv_poll256_group_2[25]), "=r"(_sysv_poll256_group_2[26]), "=r"(_sysv_poll256_group_2[27]), "=r"(_sysv_poll256_group_2[28]), "=r"(_sysv_poll256_group_2[29]), "=r"(_sysv_poll256_group_2[30]), "=r"(_sysv_poll256_group_2[31]) : "l"(local_payload + offset_2_1) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[32]), "=r"(_sysv_poll256_group_2[33]), "=r"(_sysv_poll256_group_2[34]), "=r"(_sysv_poll256_group_2[35]), "=r"(_sysv_poll256_group_2[36]), "=r"(_sysv_poll256_group_2[37]), "=r"(_sysv_poll256_group_2[38]), "=r"(_sysv_poll256_group_2[39]) : "l"(local_payload + offset_3_1) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[40]), "=r"(_sysv_poll256_group_2[41]), "=r"(_sysv_poll256_group_2[42]), "=r"(_sysv_poll256_group_2[43]), "=r"(_sysv_poll256_group_2[44]), "=r"(_sysv_poll256_group_2[45]), "=r"(_sysv_poll256_group_2[46]), "=r"(_sysv_poll256_group_2[47]) : "l"(local_payload + offset_4_1) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[48]), "=r"(_sysv_poll256_group_2[49]), "=r"(_sysv_poll256_group_2[50]), "=r"(_sysv_poll256_group_2[51]), "=r"(_sysv_poll256_group_2[52]), "=r"(_sysv_poll256_group_2[53]), "=r"(_sysv_poll256_group_2[54]), "=r"(_sysv_poll256_group_2[55]) : "l"(local_payload + offset_5_1) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_2[56]), "=r"(_sysv_poll256_group_2[57]), "=r"(_sysv_poll256_group_2[58]), "=r"(_sysv_poll256_group_2[59]), "=r"(_sysv_poll256_group_2[60]), "=r"(_sysv_poll256_group_2[61]), "=r"(_sysv_poll256_group_2[62]), "=r"(_sysv_poll256_group_2[63]) : "l"(local_payload + offset_6_1) : "memory");
            } while (
                (((_sysv_poll256_group_2[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[3] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[5] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[7] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[9] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[11] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[13] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[15] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[17] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[19] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[21] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[23] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[25] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[27] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[29] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[31] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[33] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[35] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[37] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[39] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[41] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[43] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[45] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[47] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[49] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[51] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[53] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[55] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[57] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[59] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[61] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_2[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_2[63] >> 16) & 0xffffu) == 0x8000u)
            );
            unsigned int result_words_1[8];
            uint32_t _bf16x2_add_80;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_80) : "r"(_sysv_poll256_group_2[0]), "r"(_sysv_poll256_group_2[8]));
            result_words_1[0] = _bf16x2_add_80;
            uint32_t _bf16x2_add_81;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_81) : "r"(_sysv_poll256_group_2[1]), "r"(_sysv_poll256_group_2[9]));
            result_words_1[1] = _bf16x2_add_81;
            uint32_t _bf16x2_add_82;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_82) : "r"(_sysv_poll256_group_2[2]), "r"(_sysv_poll256_group_2[10]));
            result_words_1[2] = _bf16x2_add_82;
            uint32_t _bf16x2_add_83;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_83) : "r"(_sysv_poll256_group_2[3]), "r"(_sysv_poll256_group_2[11]));
            result_words_1[3] = _bf16x2_add_83;
            uint32_t _bf16x2_add_84;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_84) : "r"(_sysv_poll256_group_2[4]), "r"(_sysv_poll256_group_2[12]));
            result_words_1[4] = _bf16x2_add_84;
            uint32_t _bf16x2_add_85;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_85) : "r"(_sysv_poll256_group_2[5]), "r"(_sysv_poll256_group_2[13]));
            result_words_1[5] = _bf16x2_add_85;
            uint32_t _bf16x2_add_86;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_86) : "r"(_sysv_poll256_group_2[6]), "r"(_sysv_poll256_group_2[14]));
            result_words_1[6] = _bf16x2_add_86;
            uint32_t _bf16x2_add_87;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_87) : "r"(_sysv_poll256_group_2[7]), "r"(_sysv_poll256_group_2[15]));
            result_words_1[7] = _bf16x2_add_87;
            uint32_t _bf16x2_add_88;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_88) : "r"(result_words_1[0]), "r"(_sysv_poll256_group_2[16]));
            result_words_1[0] = _bf16x2_add_88;
            uint32_t _bf16x2_add_89;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_89) : "r"(result_words_1[1]), "r"(_sysv_poll256_group_2[17]));
            result_words_1[1] = _bf16x2_add_89;
            uint32_t _bf16x2_add_90;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_90) : "r"(result_words_1[2]), "r"(_sysv_poll256_group_2[18]));
            result_words_1[2] = _bf16x2_add_90;
            uint32_t _bf16x2_add_91;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_91) : "r"(result_words_1[3]), "r"(_sysv_poll256_group_2[19]));
            result_words_1[3] = _bf16x2_add_91;
            uint32_t _bf16x2_add_92;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_92) : "r"(result_words_1[4]), "r"(_sysv_poll256_group_2[20]));
            result_words_1[4] = _bf16x2_add_92;
            uint32_t _bf16x2_add_93;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_93) : "r"(result_words_1[5]), "r"(_sysv_poll256_group_2[21]));
            result_words_1[5] = _bf16x2_add_93;
            uint32_t _bf16x2_add_94;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_94) : "r"(result_words_1[6]), "r"(_sysv_poll256_group_2[22]));
            result_words_1[6] = _bf16x2_add_94;
            uint32_t _bf16x2_add_95;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_95) : "r"(result_words_1[7]), "r"(_sysv_poll256_group_2[23]));
            result_words_1[7] = _bf16x2_add_95;
            uint32_t _bf16x2_add_96;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_96) : "r"(result_words_1[0]), "r"(_sysv_poll256_group_2[24]));
            result_words_1[0] = _bf16x2_add_96;
            uint32_t _bf16x2_add_97;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_97) : "r"(result_words_1[1]), "r"(_sysv_poll256_group_2[25]));
            result_words_1[1] = _bf16x2_add_97;
            uint32_t _bf16x2_add_98;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_98) : "r"(result_words_1[2]), "r"(_sysv_poll256_group_2[26]));
            result_words_1[2] = _bf16x2_add_98;
            uint32_t _bf16x2_add_99;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_99) : "r"(result_words_1[3]), "r"(_sysv_poll256_group_2[27]));
            result_words_1[3] = _bf16x2_add_99;
            uint32_t _bf16x2_add_100;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_100) : "r"(result_words_1[4]), "r"(_sysv_poll256_group_2[28]));
            result_words_1[4] = _bf16x2_add_100;
            uint32_t _bf16x2_add_101;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_101) : "r"(result_words_1[5]), "r"(_sysv_poll256_group_2[29]));
            result_words_1[5] = _bf16x2_add_101;
            uint32_t _bf16x2_add_102;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_102) : "r"(result_words_1[6]), "r"(_sysv_poll256_group_2[30]));
            result_words_1[6] = _bf16x2_add_102;
            uint32_t _bf16x2_add_103;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_103) : "r"(result_words_1[7]), "r"(_sysv_poll256_group_2[31]));
            result_words_1[7] = _bf16x2_add_103;
            uint32_t _bf16x2_add_104;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_104) : "r"(result_words_1[0]), "r"(_sysv_poll256_group_2[32]));
            result_words_1[0] = _bf16x2_add_104;
            uint32_t _bf16x2_add_105;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_105) : "r"(result_words_1[1]), "r"(_sysv_poll256_group_2[33]));
            result_words_1[1] = _bf16x2_add_105;
            uint32_t _bf16x2_add_106;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_106) : "r"(result_words_1[2]), "r"(_sysv_poll256_group_2[34]));
            result_words_1[2] = _bf16x2_add_106;
            uint32_t _bf16x2_add_107;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_107) : "r"(result_words_1[3]), "r"(_sysv_poll256_group_2[35]));
            result_words_1[3] = _bf16x2_add_107;
            uint32_t _bf16x2_add_108;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_108) : "r"(result_words_1[4]), "r"(_sysv_poll256_group_2[36]));
            result_words_1[4] = _bf16x2_add_108;
            uint32_t _bf16x2_add_109;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_109) : "r"(result_words_1[5]), "r"(_sysv_poll256_group_2[37]));
            result_words_1[5] = _bf16x2_add_109;
            uint32_t _bf16x2_add_110;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_110) : "r"(result_words_1[6]), "r"(_sysv_poll256_group_2[38]));
            result_words_1[6] = _bf16x2_add_110;
            uint32_t _bf16x2_add_111;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_111) : "r"(result_words_1[7]), "r"(_sysv_poll256_group_2[39]));
            result_words_1[7] = _bf16x2_add_111;
            uint32_t _bf16x2_add_112;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_112) : "r"(result_words_1[0]), "r"(_sysv_poll256_group_2[40]));
            result_words_1[0] = _bf16x2_add_112;
            uint32_t _bf16x2_add_113;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_113) : "r"(result_words_1[1]), "r"(_sysv_poll256_group_2[41]));
            result_words_1[1] = _bf16x2_add_113;
            uint32_t _bf16x2_add_114;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_114) : "r"(result_words_1[2]), "r"(_sysv_poll256_group_2[42]));
            result_words_1[2] = _bf16x2_add_114;
            uint32_t _bf16x2_add_115;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_115) : "r"(result_words_1[3]), "r"(_sysv_poll256_group_2[43]));
            result_words_1[3] = _bf16x2_add_115;
            uint32_t _bf16x2_add_116;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_116) : "r"(result_words_1[4]), "r"(_sysv_poll256_group_2[44]));
            result_words_1[4] = _bf16x2_add_116;
            uint32_t _bf16x2_add_117;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_117) : "r"(result_words_1[5]), "r"(_sysv_poll256_group_2[45]));
            result_words_1[5] = _bf16x2_add_117;
            uint32_t _bf16x2_add_118;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_118) : "r"(result_words_1[6]), "r"(_sysv_poll256_group_2[46]));
            result_words_1[6] = _bf16x2_add_118;
            uint32_t _bf16x2_add_119;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_119) : "r"(result_words_1[7]), "r"(_sysv_poll256_group_2[47]));
            result_words_1[7] = _bf16x2_add_119;
            uint32_t _bf16x2_add_120;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_120) : "r"(result_words_1[0]), "r"(_sysv_poll256_group_2[48]));
            result_words_1[0] = _bf16x2_add_120;
            uint32_t _bf16x2_add_121;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_121) : "r"(result_words_1[1]), "r"(_sysv_poll256_group_2[49]));
            result_words_1[1] = _bf16x2_add_121;
            uint32_t _bf16x2_add_122;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_122) : "r"(result_words_1[2]), "r"(_sysv_poll256_group_2[50]));
            result_words_1[2] = _bf16x2_add_122;
            uint32_t _bf16x2_add_123;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_123) : "r"(result_words_1[3]), "r"(_sysv_poll256_group_2[51]));
            result_words_1[3] = _bf16x2_add_123;
            uint32_t _bf16x2_add_124;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_124) : "r"(result_words_1[4]), "r"(_sysv_poll256_group_2[52]));
            result_words_1[4] = _bf16x2_add_124;
            uint32_t _bf16x2_add_125;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_125) : "r"(result_words_1[5]), "r"(_sysv_poll256_group_2[53]));
            result_words_1[5] = _bf16x2_add_125;
            uint32_t _bf16x2_add_126;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_126) : "r"(result_words_1[6]), "r"(_sysv_poll256_group_2[54]));
            result_words_1[6] = _bf16x2_add_126;
            uint32_t _bf16x2_add_127;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_127) : "r"(result_words_1[7]), "r"(_sysv_poll256_group_2[55]));
            result_words_1[7] = _bf16x2_add_127;
            uint32_t _bf16x2_add_128;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_128) : "r"(result_words_1[0]), "r"(_sysv_poll256_group_2[56]));
            result_words_1[0] = _bf16x2_add_128;
            uint32_t _bf16x2_add_129;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_129) : "r"(result_words_1[1]), "r"(_sysv_poll256_group_2[57]));
            result_words_1[1] = _bf16x2_add_129;
            uint32_t _bf16x2_add_130;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_130) : "r"(result_words_1[2]), "r"(_sysv_poll256_group_2[58]));
            result_words_1[2] = _bf16x2_add_130;
            uint32_t _bf16x2_add_131;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_131) : "r"(result_words_1[3]), "r"(_sysv_poll256_group_2[59]));
            result_words_1[3] = _bf16x2_add_131;
            uint32_t _bf16x2_add_132;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_132) : "r"(result_words_1[4]), "r"(_sysv_poll256_group_2[60]));
            result_words_1[4] = _bf16x2_add_132;
            uint32_t _bf16x2_add_133;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_133) : "r"(result_words_1[5]), "r"(_sysv_poll256_group_2[61]));
            result_words_1[5] = _bf16x2_add_133;
            uint32_t _bf16x2_add_134;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_134) : "r"(result_words_1[6]), "r"(_sysv_poll256_group_2[62]));
            result_words_1[6] = _bf16x2_add_134;
            uint32_t _bf16x2_add_135;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_135) : "r"(result_words_1[7]), "r"(_sysv_poll256_group_2[63]));
            result_words_1[7] = _bf16x2_add_135;
            long long offset_7_1 = data_base + ((long long)(8 * slots + pending) * 160 + (long long)tid) * 16;
            #pragma unroll
            for (int peer_1 = 0; peer_1 < 8; peer_1++) {
                __nv_bfloat16* peer_payload_1 = reinterpret_cast<__nv_bfloat16*>(workspace[16 + peer_1]);
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(peer_payload_1 + offset_7_1), "r"((result_words_1)[0]), "r"((result_words_1)[1]), "r"((result_words_1)[2]), "r"((result_words_1)[3]), "r"((result_words_1)[4]), "r"((result_words_1)[5]), "r"((result_words_1)[6]), "r"((result_words_1)[7]) : "memory");
            }
        }
    }
    int pending_1 = last - num_bids;
    if (pending_1 >= 0) {
        if (pending_1 % 8 == rank) {
            int slot_2 = pending_1 / 8;
            long long offset_10 = data_base + ((long long)slot_2 * 160 + (long long)tid) * 16;
            long long offset_0_3 = data_base + ((long long)(slots + slot_2) * 160 + (long long)tid) * 16;
            long long offset_1_2 = data_base + ((long long)(2 * slots + slot_2) * 160 + (long long)tid) * 16;
            long long offset_2_2 = data_base + ((long long)(3 * slots + slot_2) * 160 + (long long)tid) * 16;
            long long offset_3_2 = data_base + ((long long)(4 * slots + slot_2) * 160 + (long long)tid) * 16;
            long long offset_4_2 = data_base + ((long long)(5 * slots + slot_2) * 160 + (long long)tid) * 16;
            long long offset_5_2 = data_base + ((long long)(6 * slots + slot_2) * 160 + (long long)tid) * 16;
            long long offset_6_2 = data_base + ((long long)(7 * slots + slot_2) * 160 + (long long)tid) * 16;
            uint32_t _sysv_poll256_group_3[64];
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[0]), "=r"(_sysv_poll256_group_3[1]), "=r"(_sysv_poll256_group_3[2]), "=r"(_sysv_poll256_group_3[3]), "=r"(_sysv_poll256_group_3[4]), "=r"(_sysv_poll256_group_3[5]), "=r"(_sysv_poll256_group_3[6]), "=r"(_sysv_poll256_group_3[7]) : "l"(local_payload + offset_10) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[8]), "=r"(_sysv_poll256_group_3[9]), "=r"(_sysv_poll256_group_3[10]), "=r"(_sysv_poll256_group_3[11]), "=r"(_sysv_poll256_group_3[12]), "=r"(_sysv_poll256_group_3[13]), "=r"(_sysv_poll256_group_3[14]), "=r"(_sysv_poll256_group_3[15]) : "l"(local_payload + offset_0_3) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[16]), "=r"(_sysv_poll256_group_3[17]), "=r"(_sysv_poll256_group_3[18]), "=r"(_sysv_poll256_group_3[19]), "=r"(_sysv_poll256_group_3[20]), "=r"(_sysv_poll256_group_3[21]), "=r"(_sysv_poll256_group_3[22]), "=r"(_sysv_poll256_group_3[23]) : "l"(local_payload + offset_1_2) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[24]), "=r"(_sysv_poll256_group_3[25]), "=r"(_sysv_poll256_group_3[26]), "=r"(_sysv_poll256_group_3[27]), "=r"(_sysv_poll256_group_3[28]), "=r"(_sysv_poll256_group_3[29]), "=r"(_sysv_poll256_group_3[30]), "=r"(_sysv_poll256_group_3[31]) : "l"(local_payload + offset_2_2) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[32]), "=r"(_sysv_poll256_group_3[33]), "=r"(_sysv_poll256_group_3[34]), "=r"(_sysv_poll256_group_3[35]), "=r"(_sysv_poll256_group_3[36]), "=r"(_sysv_poll256_group_3[37]), "=r"(_sysv_poll256_group_3[38]), "=r"(_sysv_poll256_group_3[39]) : "l"(local_payload + offset_3_2) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[40]), "=r"(_sysv_poll256_group_3[41]), "=r"(_sysv_poll256_group_3[42]), "=r"(_sysv_poll256_group_3[43]), "=r"(_sysv_poll256_group_3[44]), "=r"(_sysv_poll256_group_3[45]), "=r"(_sysv_poll256_group_3[46]), "=r"(_sysv_poll256_group_3[47]) : "l"(local_payload + offset_4_2) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[48]), "=r"(_sysv_poll256_group_3[49]), "=r"(_sysv_poll256_group_3[50]), "=r"(_sysv_poll256_group_3[51]), "=r"(_sysv_poll256_group_3[52]), "=r"(_sysv_poll256_group_3[53]), "=r"(_sysv_poll256_group_3[54]), "=r"(_sysv_poll256_group_3[55]) : "l"(local_payload + offset_5_2) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_3[56]), "=r"(_sysv_poll256_group_3[57]), "=r"(_sysv_poll256_group_3[58]), "=r"(_sysv_poll256_group_3[59]), "=r"(_sysv_poll256_group_3[60]), "=r"(_sysv_poll256_group_3[61]), "=r"(_sysv_poll256_group_3[62]), "=r"(_sysv_poll256_group_3[63]) : "l"(local_payload + offset_6_2) : "memory");
            } while (
                (((_sysv_poll256_group_3[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[3] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[5] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[7] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[9] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[11] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[13] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[15] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[17] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[19] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[21] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[23] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[25] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[27] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[29] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[31] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[33] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[35] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[37] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[39] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[41] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[43] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[45] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[47] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[49] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[51] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[53] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[55] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[57] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[59] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[61] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_3[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_3[63] >> 16) & 0xffffu) == 0x8000u)
            );
            unsigned int result_words_2[8];
            uint32_t _bf16x2_add_136;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_136) : "r"(_sysv_poll256_group_3[0]), "r"(_sysv_poll256_group_3[8]));
            result_words_2[0] = _bf16x2_add_136;
            uint32_t _bf16x2_add_137;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_137) : "r"(_sysv_poll256_group_3[1]), "r"(_sysv_poll256_group_3[9]));
            result_words_2[1] = _bf16x2_add_137;
            uint32_t _bf16x2_add_138;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_138) : "r"(_sysv_poll256_group_3[2]), "r"(_sysv_poll256_group_3[10]));
            result_words_2[2] = _bf16x2_add_138;
            uint32_t _bf16x2_add_139;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_139) : "r"(_sysv_poll256_group_3[3]), "r"(_sysv_poll256_group_3[11]));
            result_words_2[3] = _bf16x2_add_139;
            uint32_t _bf16x2_add_140;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_140) : "r"(_sysv_poll256_group_3[4]), "r"(_sysv_poll256_group_3[12]));
            result_words_2[4] = _bf16x2_add_140;
            uint32_t _bf16x2_add_141;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_141) : "r"(_sysv_poll256_group_3[5]), "r"(_sysv_poll256_group_3[13]));
            result_words_2[5] = _bf16x2_add_141;
            uint32_t _bf16x2_add_142;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_142) : "r"(_sysv_poll256_group_3[6]), "r"(_sysv_poll256_group_3[14]));
            result_words_2[6] = _bf16x2_add_142;
            uint32_t _bf16x2_add_143;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_143) : "r"(_sysv_poll256_group_3[7]), "r"(_sysv_poll256_group_3[15]));
            result_words_2[7] = _bf16x2_add_143;
            uint32_t _bf16x2_add_144;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_144) : "r"(result_words_2[0]), "r"(_sysv_poll256_group_3[16]));
            result_words_2[0] = _bf16x2_add_144;
            uint32_t _bf16x2_add_145;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_145) : "r"(result_words_2[1]), "r"(_sysv_poll256_group_3[17]));
            result_words_2[1] = _bf16x2_add_145;
            uint32_t _bf16x2_add_146;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_146) : "r"(result_words_2[2]), "r"(_sysv_poll256_group_3[18]));
            result_words_2[2] = _bf16x2_add_146;
            uint32_t _bf16x2_add_147;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_147) : "r"(result_words_2[3]), "r"(_sysv_poll256_group_3[19]));
            result_words_2[3] = _bf16x2_add_147;
            uint32_t _bf16x2_add_148;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_148) : "r"(result_words_2[4]), "r"(_sysv_poll256_group_3[20]));
            result_words_2[4] = _bf16x2_add_148;
            uint32_t _bf16x2_add_149;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_149) : "r"(result_words_2[5]), "r"(_sysv_poll256_group_3[21]));
            result_words_2[5] = _bf16x2_add_149;
            uint32_t _bf16x2_add_150;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_150) : "r"(result_words_2[6]), "r"(_sysv_poll256_group_3[22]));
            result_words_2[6] = _bf16x2_add_150;
            uint32_t _bf16x2_add_151;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_151) : "r"(result_words_2[7]), "r"(_sysv_poll256_group_3[23]));
            result_words_2[7] = _bf16x2_add_151;
            uint32_t _bf16x2_add_152;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_152) : "r"(result_words_2[0]), "r"(_sysv_poll256_group_3[24]));
            result_words_2[0] = _bf16x2_add_152;
            uint32_t _bf16x2_add_153;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_153) : "r"(result_words_2[1]), "r"(_sysv_poll256_group_3[25]));
            result_words_2[1] = _bf16x2_add_153;
            uint32_t _bf16x2_add_154;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_154) : "r"(result_words_2[2]), "r"(_sysv_poll256_group_3[26]));
            result_words_2[2] = _bf16x2_add_154;
            uint32_t _bf16x2_add_155;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_155) : "r"(result_words_2[3]), "r"(_sysv_poll256_group_3[27]));
            result_words_2[3] = _bf16x2_add_155;
            uint32_t _bf16x2_add_156;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_156) : "r"(result_words_2[4]), "r"(_sysv_poll256_group_3[28]));
            result_words_2[4] = _bf16x2_add_156;
            uint32_t _bf16x2_add_157;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_157) : "r"(result_words_2[5]), "r"(_sysv_poll256_group_3[29]));
            result_words_2[5] = _bf16x2_add_157;
            uint32_t _bf16x2_add_158;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_158) : "r"(result_words_2[6]), "r"(_sysv_poll256_group_3[30]));
            result_words_2[6] = _bf16x2_add_158;
            uint32_t _bf16x2_add_159;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_159) : "r"(result_words_2[7]), "r"(_sysv_poll256_group_3[31]));
            result_words_2[7] = _bf16x2_add_159;
            uint32_t _bf16x2_add_160;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_160) : "r"(result_words_2[0]), "r"(_sysv_poll256_group_3[32]));
            result_words_2[0] = _bf16x2_add_160;
            uint32_t _bf16x2_add_161;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_161) : "r"(result_words_2[1]), "r"(_sysv_poll256_group_3[33]));
            result_words_2[1] = _bf16x2_add_161;
            uint32_t _bf16x2_add_162;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_162) : "r"(result_words_2[2]), "r"(_sysv_poll256_group_3[34]));
            result_words_2[2] = _bf16x2_add_162;
            uint32_t _bf16x2_add_163;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_163) : "r"(result_words_2[3]), "r"(_sysv_poll256_group_3[35]));
            result_words_2[3] = _bf16x2_add_163;
            uint32_t _bf16x2_add_164;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_164) : "r"(result_words_2[4]), "r"(_sysv_poll256_group_3[36]));
            result_words_2[4] = _bf16x2_add_164;
            uint32_t _bf16x2_add_165;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_165) : "r"(result_words_2[5]), "r"(_sysv_poll256_group_3[37]));
            result_words_2[5] = _bf16x2_add_165;
            uint32_t _bf16x2_add_166;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_166) : "r"(result_words_2[6]), "r"(_sysv_poll256_group_3[38]));
            result_words_2[6] = _bf16x2_add_166;
            uint32_t _bf16x2_add_167;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_167) : "r"(result_words_2[7]), "r"(_sysv_poll256_group_3[39]));
            result_words_2[7] = _bf16x2_add_167;
            uint32_t _bf16x2_add_168;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_168) : "r"(result_words_2[0]), "r"(_sysv_poll256_group_3[40]));
            result_words_2[0] = _bf16x2_add_168;
            uint32_t _bf16x2_add_169;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_169) : "r"(result_words_2[1]), "r"(_sysv_poll256_group_3[41]));
            result_words_2[1] = _bf16x2_add_169;
            uint32_t _bf16x2_add_170;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_170) : "r"(result_words_2[2]), "r"(_sysv_poll256_group_3[42]));
            result_words_2[2] = _bf16x2_add_170;
            uint32_t _bf16x2_add_171;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_171) : "r"(result_words_2[3]), "r"(_sysv_poll256_group_3[43]));
            result_words_2[3] = _bf16x2_add_171;
            uint32_t _bf16x2_add_172;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_172) : "r"(result_words_2[4]), "r"(_sysv_poll256_group_3[44]));
            result_words_2[4] = _bf16x2_add_172;
            uint32_t _bf16x2_add_173;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_173) : "r"(result_words_2[5]), "r"(_sysv_poll256_group_3[45]));
            result_words_2[5] = _bf16x2_add_173;
            uint32_t _bf16x2_add_174;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_174) : "r"(result_words_2[6]), "r"(_sysv_poll256_group_3[46]));
            result_words_2[6] = _bf16x2_add_174;
            uint32_t _bf16x2_add_175;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_175) : "r"(result_words_2[7]), "r"(_sysv_poll256_group_3[47]));
            result_words_2[7] = _bf16x2_add_175;
            uint32_t _bf16x2_add_176;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_176) : "r"(result_words_2[0]), "r"(_sysv_poll256_group_3[48]));
            result_words_2[0] = _bf16x2_add_176;
            uint32_t _bf16x2_add_177;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_177) : "r"(result_words_2[1]), "r"(_sysv_poll256_group_3[49]));
            result_words_2[1] = _bf16x2_add_177;
            uint32_t _bf16x2_add_178;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_178) : "r"(result_words_2[2]), "r"(_sysv_poll256_group_3[50]));
            result_words_2[2] = _bf16x2_add_178;
            uint32_t _bf16x2_add_179;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_179) : "r"(result_words_2[3]), "r"(_sysv_poll256_group_3[51]));
            result_words_2[3] = _bf16x2_add_179;
            uint32_t _bf16x2_add_180;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_180) : "r"(result_words_2[4]), "r"(_sysv_poll256_group_3[52]));
            result_words_2[4] = _bf16x2_add_180;
            uint32_t _bf16x2_add_181;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_181) : "r"(result_words_2[5]), "r"(_sysv_poll256_group_3[53]));
            result_words_2[5] = _bf16x2_add_181;
            uint32_t _bf16x2_add_182;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_182) : "r"(result_words_2[6]), "r"(_sysv_poll256_group_3[54]));
            result_words_2[6] = _bf16x2_add_182;
            uint32_t _bf16x2_add_183;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_183) : "r"(result_words_2[7]), "r"(_sysv_poll256_group_3[55]));
            result_words_2[7] = _bf16x2_add_183;
            uint32_t _bf16x2_add_184;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_184) : "r"(result_words_2[0]), "r"(_sysv_poll256_group_3[56]));
            result_words_2[0] = _bf16x2_add_184;
            uint32_t _bf16x2_add_185;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_185) : "r"(result_words_2[1]), "r"(_sysv_poll256_group_3[57]));
            result_words_2[1] = _bf16x2_add_185;
            uint32_t _bf16x2_add_186;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_186) : "r"(result_words_2[2]), "r"(_sysv_poll256_group_3[58]));
            result_words_2[2] = _bf16x2_add_186;
            uint32_t _bf16x2_add_187;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_187) : "r"(result_words_2[3]), "r"(_sysv_poll256_group_3[59]));
            result_words_2[3] = _bf16x2_add_187;
            uint32_t _bf16x2_add_188;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_188) : "r"(result_words_2[4]), "r"(_sysv_poll256_group_3[60]));
            result_words_2[4] = _bf16x2_add_188;
            uint32_t _bf16x2_add_189;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_189) : "r"(result_words_2[5]), "r"(_sysv_poll256_group_3[61]));
            result_words_2[5] = _bf16x2_add_189;
            uint32_t _bf16x2_add_190;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_190) : "r"(result_words_2[6]), "r"(_sysv_poll256_group_3[62]));
            result_words_2[6] = _bf16x2_add_190;
            uint32_t _bf16x2_add_191;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_191) : "r"(result_words_2[7]), "r"(_sysv_poll256_group_3[63]));
            result_words_2[7] = _bf16x2_add_191;
            long long offset_7_2 = data_base + ((long long)(8 * slots + pending_1) * 160 + (long long)tid) * 16;
            #pragma unroll
            for (int peer_2 = 0; peer_2 < 8; peer_2++) {
                __nv_bfloat16* peer_payload_2 = reinterpret_cast<__nv_bfloat16*>(workspace[16 + peer_2]);
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(peer_payload_2 + offset_7_2), "r"((result_words_2)[0]), "r"((result_words_2)[1]), "r"((result_words_2)[2]), "r"((result_words_2)[3]), "r"((result_words_2)[4]), "r"((result_words_2)[5]), "r"((result_words_2)[6]), "r"((result_words_2)[7]) : "memory");
            }
        }
    }
    int pending_2 = last;
    if (pending_2 >= 0) {
        if (pending_2 % 8 == rank) {
            int slot_3 = pending_2 / 8;
            long long offset_11 = data_base + ((long long)slot_3 * 160 + (long long)tid) * 16;
            long long offset_0_4 = data_base + ((long long)(slots + slot_3) * 160 + (long long)tid) * 16;
            long long offset_1_3 = data_base + ((long long)(2 * slots + slot_3) * 160 + (long long)tid) * 16;
            long long offset_2_3 = data_base + ((long long)(3 * slots + slot_3) * 160 + (long long)tid) * 16;
            long long offset_3_3 = data_base + ((long long)(4 * slots + slot_3) * 160 + (long long)tid) * 16;
            long long offset_4_3 = data_base + ((long long)(5 * slots + slot_3) * 160 + (long long)tid) * 16;
            long long offset_5_3 = data_base + ((long long)(6 * slots + slot_3) * 160 + (long long)tid) * 16;
            long long offset_6_3 = data_base + ((long long)(7 * slots + slot_3) * 160 + (long long)tid) * 16;
            uint32_t _sysv_poll256_group_4[64];
            do {
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[0]), "=r"(_sysv_poll256_group_4[1]), "=r"(_sysv_poll256_group_4[2]), "=r"(_sysv_poll256_group_4[3]), "=r"(_sysv_poll256_group_4[4]), "=r"(_sysv_poll256_group_4[5]), "=r"(_sysv_poll256_group_4[6]), "=r"(_sysv_poll256_group_4[7]) : "l"(local_payload + offset_11) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[8]), "=r"(_sysv_poll256_group_4[9]), "=r"(_sysv_poll256_group_4[10]), "=r"(_sysv_poll256_group_4[11]), "=r"(_sysv_poll256_group_4[12]), "=r"(_sysv_poll256_group_4[13]), "=r"(_sysv_poll256_group_4[14]), "=r"(_sysv_poll256_group_4[15]) : "l"(local_payload + offset_0_4) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[16]), "=r"(_sysv_poll256_group_4[17]), "=r"(_sysv_poll256_group_4[18]), "=r"(_sysv_poll256_group_4[19]), "=r"(_sysv_poll256_group_4[20]), "=r"(_sysv_poll256_group_4[21]), "=r"(_sysv_poll256_group_4[22]), "=r"(_sysv_poll256_group_4[23]) : "l"(local_payload + offset_1_3) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[24]), "=r"(_sysv_poll256_group_4[25]), "=r"(_sysv_poll256_group_4[26]), "=r"(_sysv_poll256_group_4[27]), "=r"(_sysv_poll256_group_4[28]), "=r"(_sysv_poll256_group_4[29]), "=r"(_sysv_poll256_group_4[30]), "=r"(_sysv_poll256_group_4[31]) : "l"(local_payload + offset_2_3) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[32]), "=r"(_sysv_poll256_group_4[33]), "=r"(_sysv_poll256_group_4[34]), "=r"(_sysv_poll256_group_4[35]), "=r"(_sysv_poll256_group_4[36]), "=r"(_sysv_poll256_group_4[37]), "=r"(_sysv_poll256_group_4[38]), "=r"(_sysv_poll256_group_4[39]) : "l"(local_payload + offset_3_3) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[40]), "=r"(_sysv_poll256_group_4[41]), "=r"(_sysv_poll256_group_4[42]), "=r"(_sysv_poll256_group_4[43]), "=r"(_sysv_poll256_group_4[44]), "=r"(_sysv_poll256_group_4[45]), "=r"(_sysv_poll256_group_4[46]), "=r"(_sysv_poll256_group_4[47]) : "l"(local_payload + offset_4_3) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[48]), "=r"(_sysv_poll256_group_4[49]), "=r"(_sysv_poll256_group_4[50]), "=r"(_sysv_poll256_group_4[51]), "=r"(_sysv_poll256_group_4[52]), "=r"(_sysv_poll256_group_4[53]), "=r"(_sysv_poll256_group_4[54]), "=r"(_sysv_poll256_group_4[55]) : "l"(local_payload + offset_5_3) : "memory");
                asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_4[56]), "=r"(_sysv_poll256_group_4[57]), "=r"(_sysv_poll256_group_4[58]), "=r"(_sysv_poll256_group_4[59]), "=r"(_sysv_poll256_group_4[60]), "=r"(_sysv_poll256_group_4[61]), "=r"(_sysv_poll256_group_4[62]), "=r"(_sysv_poll256_group_4[63]) : "l"(local_payload + offset_6_3) : "memory");
            } while (
                (((_sysv_poll256_group_4[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[3] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[5] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[7] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[9] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[11] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[13] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[15] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[17] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[19] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[21] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[23] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[25] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[27] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[29] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[31] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[32] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[32] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[33] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[33] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[34] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[34] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[35] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[35] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[36] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[36] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[37] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[37] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[38] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[38] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[39] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[39] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[40] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[40] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[41] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[41] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[42] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[42] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[43] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[43] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[44] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[44] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[45] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[45] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[46] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[46] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[47] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[47] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[48] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[48] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[49] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[49] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[50] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[50] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[51] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[51] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[52] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[52] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[53] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[53] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[54] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[54] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[55] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[55] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[56] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[56] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[57] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[57] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[58] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[58] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[59] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[59] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[60] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[60] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[61] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[61] >> 16) & 0xffffu) == 0x8000u) ||
                (((_sysv_poll256_group_4[62] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[62] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[63] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_4[63] >> 16) & 0xffffu) == 0x8000u)
            );
            unsigned int result_words_3[8];
            uint32_t _bf16x2_add_192;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_192) : "r"(_sysv_poll256_group_4[0]), "r"(_sysv_poll256_group_4[8]));
            result_words_3[0] = _bf16x2_add_192;
            uint32_t _bf16x2_add_193;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_193) : "r"(_sysv_poll256_group_4[1]), "r"(_sysv_poll256_group_4[9]));
            result_words_3[1] = _bf16x2_add_193;
            uint32_t _bf16x2_add_194;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_194) : "r"(_sysv_poll256_group_4[2]), "r"(_sysv_poll256_group_4[10]));
            result_words_3[2] = _bf16x2_add_194;
            uint32_t _bf16x2_add_195;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_195) : "r"(_sysv_poll256_group_4[3]), "r"(_sysv_poll256_group_4[11]));
            result_words_3[3] = _bf16x2_add_195;
            uint32_t _bf16x2_add_196;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_196) : "r"(_sysv_poll256_group_4[4]), "r"(_sysv_poll256_group_4[12]));
            result_words_3[4] = _bf16x2_add_196;
            uint32_t _bf16x2_add_197;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_197) : "r"(_sysv_poll256_group_4[5]), "r"(_sysv_poll256_group_4[13]));
            result_words_3[5] = _bf16x2_add_197;
            uint32_t _bf16x2_add_198;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_198) : "r"(_sysv_poll256_group_4[6]), "r"(_sysv_poll256_group_4[14]));
            result_words_3[6] = _bf16x2_add_198;
            uint32_t _bf16x2_add_199;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_199) : "r"(_sysv_poll256_group_4[7]), "r"(_sysv_poll256_group_4[15]));
            result_words_3[7] = _bf16x2_add_199;
            uint32_t _bf16x2_add_200;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_200) : "r"(result_words_3[0]), "r"(_sysv_poll256_group_4[16]));
            result_words_3[0] = _bf16x2_add_200;
            uint32_t _bf16x2_add_201;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_201) : "r"(result_words_3[1]), "r"(_sysv_poll256_group_4[17]));
            result_words_3[1] = _bf16x2_add_201;
            uint32_t _bf16x2_add_202;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_202) : "r"(result_words_3[2]), "r"(_sysv_poll256_group_4[18]));
            result_words_3[2] = _bf16x2_add_202;
            uint32_t _bf16x2_add_203;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_203) : "r"(result_words_3[3]), "r"(_sysv_poll256_group_4[19]));
            result_words_3[3] = _bf16x2_add_203;
            uint32_t _bf16x2_add_204;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_204) : "r"(result_words_3[4]), "r"(_sysv_poll256_group_4[20]));
            result_words_3[4] = _bf16x2_add_204;
            uint32_t _bf16x2_add_205;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_205) : "r"(result_words_3[5]), "r"(_sysv_poll256_group_4[21]));
            result_words_3[5] = _bf16x2_add_205;
            uint32_t _bf16x2_add_206;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_206) : "r"(result_words_3[6]), "r"(_sysv_poll256_group_4[22]));
            result_words_3[6] = _bf16x2_add_206;
            uint32_t _bf16x2_add_207;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_207) : "r"(result_words_3[7]), "r"(_sysv_poll256_group_4[23]));
            result_words_3[7] = _bf16x2_add_207;
            uint32_t _bf16x2_add_208;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_208) : "r"(result_words_3[0]), "r"(_sysv_poll256_group_4[24]));
            result_words_3[0] = _bf16x2_add_208;
            uint32_t _bf16x2_add_209;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_209) : "r"(result_words_3[1]), "r"(_sysv_poll256_group_4[25]));
            result_words_3[1] = _bf16x2_add_209;
            uint32_t _bf16x2_add_210;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_210) : "r"(result_words_3[2]), "r"(_sysv_poll256_group_4[26]));
            result_words_3[2] = _bf16x2_add_210;
            uint32_t _bf16x2_add_211;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_211) : "r"(result_words_3[3]), "r"(_sysv_poll256_group_4[27]));
            result_words_3[3] = _bf16x2_add_211;
            uint32_t _bf16x2_add_212;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_212) : "r"(result_words_3[4]), "r"(_sysv_poll256_group_4[28]));
            result_words_3[4] = _bf16x2_add_212;
            uint32_t _bf16x2_add_213;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_213) : "r"(result_words_3[5]), "r"(_sysv_poll256_group_4[29]));
            result_words_3[5] = _bf16x2_add_213;
            uint32_t _bf16x2_add_214;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_214) : "r"(result_words_3[6]), "r"(_sysv_poll256_group_4[30]));
            result_words_3[6] = _bf16x2_add_214;
            uint32_t _bf16x2_add_215;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_215) : "r"(result_words_3[7]), "r"(_sysv_poll256_group_4[31]));
            result_words_3[7] = _bf16x2_add_215;
            uint32_t _bf16x2_add_216;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_216) : "r"(result_words_3[0]), "r"(_sysv_poll256_group_4[32]));
            result_words_3[0] = _bf16x2_add_216;
            uint32_t _bf16x2_add_217;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_217) : "r"(result_words_3[1]), "r"(_sysv_poll256_group_4[33]));
            result_words_3[1] = _bf16x2_add_217;
            uint32_t _bf16x2_add_218;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_218) : "r"(result_words_3[2]), "r"(_sysv_poll256_group_4[34]));
            result_words_3[2] = _bf16x2_add_218;
            uint32_t _bf16x2_add_219;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_219) : "r"(result_words_3[3]), "r"(_sysv_poll256_group_4[35]));
            result_words_3[3] = _bf16x2_add_219;
            uint32_t _bf16x2_add_220;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_220) : "r"(result_words_3[4]), "r"(_sysv_poll256_group_4[36]));
            result_words_3[4] = _bf16x2_add_220;
            uint32_t _bf16x2_add_221;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_221) : "r"(result_words_3[5]), "r"(_sysv_poll256_group_4[37]));
            result_words_3[5] = _bf16x2_add_221;
            uint32_t _bf16x2_add_222;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_222) : "r"(result_words_3[6]), "r"(_sysv_poll256_group_4[38]));
            result_words_3[6] = _bf16x2_add_222;
            uint32_t _bf16x2_add_223;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_223) : "r"(result_words_3[7]), "r"(_sysv_poll256_group_4[39]));
            result_words_3[7] = _bf16x2_add_223;
            uint32_t _bf16x2_add_224;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_224) : "r"(result_words_3[0]), "r"(_sysv_poll256_group_4[40]));
            result_words_3[0] = _bf16x2_add_224;
            uint32_t _bf16x2_add_225;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_225) : "r"(result_words_3[1]), "r"(_sysv_poll256_group_4[41]));
            result_words_3[1] = _bf16x2_add_225;
            uint32_t _bf16x2_add_226;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_226) : "r"(result_words_3[2]), "r"(_sysv_poll256_group_4[42]));
            result_words_3[2] = _bf16x2_add_226;
            uint32_t _bf16x2_add_227;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_227) : "r"(result_words_3[3]), "r"(_sysv_poll256_group_4[43]));
            result_words_3[3] = _bf16x2_add_227;
            uint32_t _bf16x2_add_228;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_228) : "r"(result_words_3[4]), "r"(_sysv_poll256_group_4[44]));
            result_words_3[4] = _bf16x2_add_228;
            uint32_t _bf16x2_add_229;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_229) : "r"(result_words_3[5]), "r"(_sysv_poll256_group_4[45]));
            result_words_3[5] = _bf16x2_add_229;
            uint32_t _bf16x2_add_230;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_230) : "r"(result_words_3[6]), "r"(_sysv_poll256_group_4[46]));
            result_words_3[6] = _bf16x2_add_230;
            uint32_t _bf16x2_add_231;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_231) : "r"(result_words_3[7]), "r"(_sysv_poll256_group_4[47]));
            result_words_3[7] = _bf16x2_add_231;
            uint32_t _bf16x2_add_232;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_232) : "r"(result_words_3[0]), "r"(_sysv_poll256_group_4[48]));
            result_words_3[0] = _bf16x2_add_232;
            uint32_t _bf16x2_add_233;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_233) : "r"(result_words_3[1]), "r"(_sysv_poll256_group_4[49]));
            result_words_3[1] = _bf16x2_add_233;
            uint32_t _bf16x2_add_234;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_234) : "r"(result_words_3[2]), "r"(_sysv_poll256_group_4[50]));
            result_words_3[2] = _bf16x2_add_234;
            uint32_t _bf16x2_add_235;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_235) : "r"(result_words_3[3]), "r"(_sysv_poll256_group_4[51]));
            result_words_3[3] = _bf16x2_add_235;
            uint32_t _bf16x2_add_236;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_236) : "r"(result_words_3[4]), "r"(_sysv_poll256_group_4[52]));
            result_words_3[4] = _bf16x2_add_236;
            uint32_t _bf16x2_add_237;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_237) : "r"(result_words_3[5]), "r"(_sysv_poll256_group_4[53]));
            result_words_3[5] = _bf16x2_add_237;
            uint32_t _bf16x2_add_238;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_238) : "r"(result_words_3[6]), "r"(_sysv_poll256_group_4[54]));
            result_words_3[6] = _bf16x2_add_238;
            uint32_t _bf16x2_add_239;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_239) : "r"(result_words_3[7]), "r"(_sysv_poll256_group_4[55]));
            result_words_3[7] = _bf16x2_add_239;
            uint32_t _bf16x2_add_240;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_240) : "r"(result_words_3[0]), "r"(_sysv_poll256_group_4[56]));
            result_words_3[0] = _bf16x2_add_240;
            uint32_t _bf16x2_add_241;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_241) : "r"(result_words_3[1]), "r"(_sysv_poll256_group_4[57]));
            result_words_3[1] = _bf16x2_add_241;
            uint32_t _bf16x2_add_242;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_242) : "r"(result_words_3[2]), "r"(_sysv_poll256_group_4[58]));
            result_words_3[2] = _bf16x2_add_242;
            uint32_t _bf16x2_add_243;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_243) : "r"(result_words_3[3]), "r"(_sysv_poll256_group_4[59]));
            result_words_3[3] = _bf16x2_add_243;
            uint32_t _bf16x2_add_244;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_244) : "r"(result_words_3[4]), "r"(_sysv_poll256_group_4[60]));
            result_words_3[4] = _bf16x2_add_244;
            uint32_t _bf16x2_add_245;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_245) : "r"(result_words_3[5]), "r"(_sysv_poll256_group_4[61]));
            result_words_3[5] = _bf16x2_add_245;
            uint32_t _bf16x2_add_246;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_246) : "r"(result_words_3[6]), "r"(_sysv_poll256_group_4[62]));
            result_words_3[6] = _bf16x2_add_246;
            uint32_t _bf16x2_add_247;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_247) : "r"(result_words_3[7]), "r"(_sysv_poll256_group_4[63]));
            result_words_3[7] = _bf16x2_add_247;
            long long offset_7_3 = data_base + ((long long)(8 * slots + pending_2) * 160 + (long long)tid) * 16;
            #pragma unroll
            for (int peer_3 = 0; peer_3 < 8; peer_3++) {
                __nv_bfloat16* peer_payload_3 = reinterpret_cast<__nv_bfloat16*>(workspace[16 + peer_3]);
                asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(peer_payload_3 + offset_7_3), "r"((result_words_3)[0]), "r"((result_words_3)[1]), "r"((result_words_3)[2]), "r"((result_words_3)[3]), "r"((result_words_3)[4]), "r"((result_words_3)[5]), "r"((result_words_3)[6]), "r"((result_words_3)[7]) : "memory");
            }
        }
    }
    int unfinished = last - 3 * num_bids;
    if (unfinished >= 0) {
        long long offset_12 = data_base + ((long long)(8 * slots + unfinished) * 160 + (long long)tid) * 16;
        uint32_t _sysv_poll256_group_5[8];
        do {
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_5[0]), "=r"(_sysv_poll256_group_5[1]), "=r"(_sysv_poll256_group_5[2]), "=r"(_sysv_poll256_group_5[3]), "=r"(_sysv_poll256_group_5[4]), "=r"(_sysv_poll256_group_5[5]), "=r"(_sysv_poll256_group_5[6]), "=r"(_sysv_poll256_group_5[7]) : "l"(local_payload + offset_12) : "memory");
        } while (
            (((_sysv_poll256_group_5[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_5[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[3] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_5[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[5] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_5[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_5[7] >> 16) & 0xffffu) == 0x8000u)
        );
        unsigned int final_words_1[8];
        final_words_1[0] = _sysv_poll256_group_5[0];
        final_words_1[1] = _sysv_poll256_group_5[1];
        final_words_1[2] = _sysv_poll256_group_5[2];
        final_words_1[3] = _sysv_poll256_group_5[3];
        final_words_1[4] = _sysv_poll256_group_5[4];
        final_words_1[5] = _sysv_poll256_group_5[5];
        final_words_1[6] = _sysv_poll256_group_5[6];
        final_words_1[7] = _sysv_poll256_group_5[7];
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + (((long long)unfinished * 160 + (long long)tid) * 16)), "r"((final_words_1)[0]), "r"((final_words_1)[1]), "r"((final_words_1)[2]), "r"((final_words_1)[3]), "r"((final_words_1)[4]), "r"((final_words_1)[5]), "r"((final_words_1)[6]), "r"((final_words_1)[7]) : "memory");
    }
    int unfinished_3 = last - 2 * num_bids;
    if (unfinished_3 >= 0) {
        long long offset_13 = data_base + ((long long)(8 * slots + unfinished_3) * 160 + (long long)tid) * 16;
        uint32_t _sysv_poll256_group_6[8];
        do {
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_6[0]), "=r"(_sysv_poll256_group_6[1]), "=r"(_sysv_poll256_group_6[2]), "=r"(_sysv_poll256_group_6[3]), "=r"(_sysv_poll256_group_6[4]), "=r"(_sysv_poll256_group_6[5]), "=r"(_sysv_poll256_group_6[6]), "=r"(_sysv_poll256_group_6[7]) : "l"(local_payload + offset_13) : "memory");
        } while (
            (((_sysv_poll256_group_6[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_6[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[3] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_6[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[5] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_6[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_6[7] >> 16) & 0xffffu) == 0x8000u)
        );
        unsigned int final_words_2[8];
        final_words_2[0] = _sysv_poll256_group_6[0];
        final_words_2[1] = _sysv_poll256_group_6[1];
        final_words_2[2] = _sysv_poll256_group_6[2];
        final_words_2[3] = _sysv_poll256_group_6[3];
        final_words_2[4] = _sysv_poll256_group_6[4];
        final_words_2[5] = _sysv_poll256_group_6[5];
        final_words_2[6] = _sysv_poll256_group_6[6];
        final_words_2[7] = _sysv_poll256_group_6[7];
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + (((long long)unfinished_3 * 160 + (long long)tid) * 16)), "r"((final_words_2)[0]), "r"((final_words_2)[1]), "r"((final_words_2)[2]), "r"((final_words_2)[3]), "r"((final_words_2)[4]), "r"((final_words_2)[5]), "r"((final_words_2)[6]), "r"((final_words_2)[7]) : "memory");
    }
    int unfinished_4 = last - num_bids;
    if (unfinished_4 >= 0) {
        long long offset_14 = data_base + ((long long)(8 * slots + unfinished_4) * 160 + (long long)tid) * 16;
        uint32_t _sysv_poll256_group_7[8];
        do {
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_7[0]), "=r"(_sysv_poll256_group_7[1]), "=r"(_sysv_poll256_group_7[2]), "=r"(_sysv_poll256_group_7[3]), "=r"(_sysv_poll256_group_7[4]), "=r"(_sysv_poll256_group_7[5]), "=r"(_sysv_poll256_group_7[6]), "=r"(_sysv_poll256_group_7[7]) : "l"(local_payload + offset_14) : "memory");
        } while (
            (((_sysv_poll256_group_7[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_7[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[3] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_7[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[5] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_7[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_7[7] >> 16) & 0xffffu) == 0x8000u)
        );
        unsigned int final_words_3[8];
        final_words_3[0] = _sysv_poll256_group_7[0];
        final_words_3[1] = _sysv_poll256_group_7[1];
        final_words_3[2] = _sysv_poll256_group_7[2];
        final_words_3[3] = _sysv_poll256_group_7[3];
        final_words_3[4] = _sysv_poll256_group_7[4];
        final_words_3[5] = _sysv_poll256_group_7[5];
        final_words_3[6] = _sysv_poll256_group_7[6];
        final_words_3[7] = _sysv_poll256_group_7[7];
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + (((long long)unfinished_4 * 160 + (long long)tid) * 16)), "r"((final_words_3)[0]), "r"((final_words_3)[1]), "r"((final_words_3)[2]), "r"((final_words_3)[3]), "r"((final_words_3)[4]), "r"((final_words_3)[5]), "r"((final_words_3)[6]), "r"((final_words_3)[7]) : "memory");
    }
    int unfinished_5 = last;
    if (unfinished_5 >= 0) {
        long long offset_15 = data_base + ((long long)(8 * slots + unfinished_5) * 160 + (long long)tid) * 16;
        uint32_t _sysv_poll256_group_8[8];
        do {
            asm volatile("ld.volatile.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_sysv_poll256_group_8[0]), "=r"(_sysv_poll256_group_8[1]), "=r"(_sysv_poll256_group_8[2]), "=r"(_sysv_poll256_group_8[3]), "=r"(_sysv_poll256_group_8[4]), "=r"(_sysv_poll256_group_8[5]), "=r"(_sysv_poll256_group_8[6]), "=r"(_sysv_poll256_group_8[7]) : "l"(local_payload + offset_15) : "memory");
        } while (
            (((_sysv_poll256_group_8[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_8[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[3] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_8[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[5] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll256_group_8[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll256_group_8[7] >> 16) & 0xffffu) == 0x8000u)
        );
        unsigned int final_words_4[8];
        final_words_4[0] = _sysv_poll256_group_8[0];
        final_words_4[1] = _sysv_poll256_group_8[1];
        final_words_4[2] = _sysv_poll256_group_8[2];
        final_words_4[3] = _sysv_poll256_group_8[3];
        final_words_4[4] = _sysv_poll256_group_8[4];
        final_words_4[5] = _sysv_poll256_group_8[5];
        final_words_4[6] = _sysv_poll256_group_8[6];
        final_words_4[7] = _sysv_poll256_group_8[7];
        asm volatile("st.volatile.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(collective_out + (((long long)unfinished_5 * 160 + (long long)tid) * 16)), "r"((final_words_4)[0]), "r"((final_words_4)[1]), "r"((final_words_4)[2]), "r"((final_words_4)[3]), "r"((final_words_4)[4]), "r"((final_words_4)[5]), "r"((final_words_4)[6]), "r"((final_words_4)[7]) : "memory");
    }
    if (bid == 0) {
        if (tid == 0) {
            int clear_bytes = (8 * slots + tokens) * 5120;
            {
                volatile int* _sysv_wait_p_12 = reinterpret_cast<volatile int*>(reinterpret_cast<unsigned int*>(control_0)) + (0);
                while (*_sysv_wait_p_12 != static_cast<int>(num_bids)) {}
            }
            control_0[2] = (control_current_word + 1) % 3;
            control_0[4] = clear_bytes;
            reinterpret_cast<unsigned int*>(control_0)[0] = 0;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
