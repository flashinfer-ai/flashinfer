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
#define SMEM_SLOTS_OFF 0
#define SMEM_SLOTS_STAGE_BYTES 128
#define SMEM_SLOTS_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 512
#define LAUNCH_BOUNDS_THREADS 512

#include <math_constants.h>


__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_muse_spark_norm_1aafa313e4b01bb7f5b7(const __nv_bfloat16* branch, const float* residual, const float* __restrict__ alpha, const float* __restrict__ beta, const __nv_bfloat16* __restrict__ next_weight, const __nv_bfloat16* __restrict__ post_weight, float* residual_out, __nv_bfloat16* ffn_input, int M, float post_eps, float next_eps, float post_gain_center, float next_gain_center, float alpha_value)
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
    float* slots = reinterpret_cast<float*>(smem_raw + 0);
    const int slots_addr = smem + 0;

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int lane_1 = lane;
    int rank = 0;
    int first = bid;
    int stride = num_bids;
    int col0 = (rank * 512 + tid) * 16;
    long long col64 = (long long)col0;
    float be[16];
    unsigned int nw[8];
    {
        unsigned _ldv8_0_0;
        unsigned _ldv8_0_1;
        unsigned _ldv8_0_2;
        unsigned _ldv8_0_3;
        unsigned _ldv8_0_4;
        unsigned _ldv8_0_5;
        unsigned _ldv8_0_6;
        unsigned _ldv8_0_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(beta + col0 + (0))) : "memory");
        be[0 + 0] = __uint_as_float(_ldv8_0_0);
        be[0 + 1] = __uint_as_float(_ldv8_0_1);
        be[0 + 2] = __uint_as_float(_ldv8_0_2);
        be[0 + 3] = __uint_as_float(_ldv8_0_3);
        be[0 + 4] = __uint_as_float(_ldv8_0_4);
        be[0 + 5] = __uint_as_float(_ldv8_0_5);
        be[0 + 6] = __uint_as_float(_ldv8_0_6);
        be[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    {
        unsigned _ldv8_1_0;
        unsigned _ldv8_1_1;
        unsigned _ldv8_1_2;
        unsigned _ldv8_1_3;
        unsigned _ldv8_1_4;
        unsigned _ldv8_1_5;
        unsigned _ldv8_1_6;
        unsigned _ldv8_1_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(beta + (col0 + 8) + (0))) : "memory");
        be[8 + 0] = __uint_as_float(_ldv8_1_0);
        be[8 + 1] = __uint_as_float(_ldv8_1_1);
        be[8 + 2] = __uint_as_float(_ldv8_1_2);
        be[8 + 3] = __uint_as_float(_ldv8_1_3);
        be[8 + 4] = __uint_as_float(_ldv8_1_4);
        be[8 + 5] = __uint_as_float(_ldv8_1_5);
        be[8 + 6] = __uint_as_float(_ldv8_1_6);
        be[8 + 7] = __uint_as_float(_ldv8_1_7);
    }
    {
        asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(nw[0 + 0]), "=r"(nw[0 + 1]), "=r"(nw[0 + 2]), "=r"(nw[0 + 3]), "=r"(nw[0 + 4]), "=r"(nw[0 + 5]), "=r"(nw[0 + 6]), "=r"(nw[0 + 7]) : "l"((const void*)((const char*)(next_weight + col0 + 0) + 0)) : "memory");
    }
    unsigned int cur_b[8];
    float cur_r[16];
    #pragma unroll 1
    for (int row0 = first; row0 < first + 1; row0++) {
        int row = row0;
        long long rbase = (long long)row * 8192 + col64;
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(cur_b[0 + 0]), "=r"(cur_b[0 + 1]), "=r"(cur_b[0 + 2]), "=r"(cur_b[0 + 3]), "=r"(cur_b[0 + 4]), "=r"(cur_b[0 + 5]), "=r"(cur_b[0 + 6]), "=r"(cur_b[0 + 7]) : "l"((const void*)((const char*)(branch + rbase + 0) + 0)) : "memory");
        }
        {
            unsigned _ldv8_4_0;
            unsigned _ldv8_4_1;
            unsigned _ldv8_4_2;
            unsigned _ldv8_4_3;
            unsigned _ldv8_4_4;
            unsigned _ldv8_4_5;
            unsigned _ldv8_4_6;
            unsigned _ldv8_4_7;
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(residual + rbase + (0))) : "memory");
            cur_r[0 + 0] = __uint_as_float(_ldv8_4_0);
            cur_r[0 + 1] = __uint_as_float(_ldv8_4_1);
            cur_r[0 + 2] = __uint_as_float(_ldv8_4_2);
            cur_r[0 + 3] = __uint_as_float(_ldv8_4_3);
            cur_r[0 + 4] = __uint_as_float(_ldv8_4_4);
            cur_r[0 + 5] = __uint_as_float(_ldv8_4_5);
            cur_r[0 + 6] = __uint_as_float(_ldv8_4_6);
            cur_r[0 + 7] = __uint_as_float(_ldv8_4_7);
        }
        {
            unsigned _ldv8_5_0;
            unsigned _ldv8_5_1;
            unsigned _ldv8_5_2;
            unsigned _ldv8_5_3;
            unsigned _ldv8_5_4;
            unsigned _ldv8_5_5;
            unsigned _ldv8_5_6;
            unsigned _ldv8_5_7;
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3), "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7) : "l"((const void*)(residual + (rbase + 8) + (0))) : "memory");
            cur_r[8 + 0] = __uint_as_float(_ldv8_5_0);
            cur_r[8 + 1] = __uint_as_float(_ldv8_5_1);
            cur_r[8 + 2] = __uint_as_float(_ldv8_5_2);
            cur_r[8 + 3] = __uint_as_float(_ldv8_5_3);
            cur_r[8 + 4] = __uint_as_float(_ldv8_5_4);
            cur_r[8 + 5] = __uint_as_float(_ldv8_5_5);
            cur_r[8 + 6] = __uint_as_float(_ldv8_5_6);
            cur_r[8 + 7] = __uint_as_float(_ldv8_5_7);
        }
        float bb[16];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(cur_b[k]));
            bb[2 * k] = _cvt_f32_0.x;
            bb[2 * k + 1] = _cvt_f32_0.y;
        }
        float pa = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            pa = __fmaf_rn(bb[j], bb[j], pa);
            cur_r[j] = alpha_value * cur_r[j];
            bb[j] = be[j] * bb[j];
        }
        float _warp_reduce_0 = pa;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        if (lane_1 == 0) {
            slots[warp_0] = _warp_reduce_0;
        }
        __syncthreads();
        int idx = ((lane_1) < (15) ? (lane_1) : (15));
        float value = slots[idx];
        if (lane_1 >= 16) {
            value = 0.0f;
        }
        float _warp_reduce_7 = value;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
        float _sqrt_6;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_6) : "f"(__fmaf_rn(_warp_reduce_7, 0.00012207031f, post_eps)));
        float r1 = __frcp_rn(_sqrt_6);
        float p2 = 0.0f;
        #pragma unroll
        for (int j_1 = 0; j_1 < 16; j_1++) {
            cur_r[j_1] = __fmaf_rn(bb[j_1], r1, cur_r[j_1]);
            p2 = __fmaf_rn(cur_r[j_1], cur_r[j_1], p2);
        }
        {
            unsigned _stv8_8_0 = __float_as_uint(cur_r[0 + 0]);
            unsigned _stv8_8_1 = __float_as_uint(cur_r[0 + 1]);
            unsigned _stv8_8_2 = __float_as_uint(cur_r[0 + 2]);
            unsigned _stv8_8_3 = __float_as_uint(cur_r[0 + 3]);
            unsigned _stv8_8_4 = __float_as_uint(cur_r[0 + 4]);
            unsigned _stv8_8_5 = __float_as_uint(cur_r[0 + 5]);
            unsigned _stv8_8_6 = __float_as_uint(cur_r[0 + 6]);
            unsigned _stv8_8_7 = __float_as_uint(cur_r[0 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(residual_out + rbase + (0))), "r"(_stv8_8_0), "r"(_stv8_8_1), "r"(_stv8_8_2), "r"(_stv8_8_3), "r"(_stv8_8_4), "r"(_stv8_8_5), "r"(_stv8_8_6), "r"(_stv8_8_7) : "memory");
        }
        {
            unsigned _stv8_9_0 = __float_as_uint(cur_r[8 + 0]);
            unsigned _stv8_9_1 = __float_as_uint(cur_r[8 + 1]);
            unsigned _stv8_9_2 = __float_as_uint(cur_r[8 + 2]);
            unsigned _stv8_9_3 = __float_as_uint(cur_r[8 + 3]);
            unsigned _stv8_9_4 = __float_as_uint(cur_r[8 + 4]);
            unsigned _stv8_9_5 = __float_as_uint(cur_r[8 + 5]);
            unsigned _stv8_9_6 = __float_as_uint(cur_r[8 + 6]);
            unsigned _stv8_9_7 = __float_as_uint(cur_r[8 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(residual_out + (rbase + 8) + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
        }
        float warp_reduce = p2;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce += __shfl_xor_sync(0xFFFFFFFF, warp_reduce, offset);
        if (lane_1 == 0) {
            slots[16 + warp_0] = warp_reduce;
        }
        __syncthreads();
        int idx_0 = ((lane_1) < (15) ? (lane_1) : (15));
        float value_1 = slots[16 + idx_0];
        if (lane_1 >= 16) {
            value_1 = 0.0f;
        }
        float _warp_reduce_11 = value_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
        float _sqrt_10;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_10) : "f"(__fmaf_rn(_warp_reduce_11, 0.00012207031f, next_eps)));
        float r2 = __frcp_rn(_sqrt_10);
        #pragma unroll
        for (int k_1 = 0; k_1 < 8; k_1++) {
            float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(nw[k_1]));
            bb[2 * k_1] = cur_r[2 * k_1] * r2 * (_cvt_f32_1.x + next_gain_center);
            bb[2 * k_1 + 1] = cur_r[2 * k_1 + 1] * r2 * (_cvt_f32_1.y + next_gain_center);
        }
        unsigned int _packed_0[8];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 0], bb[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 2], bb[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 4], bb[_lp*2+1 + 4]));
            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 6], bb[_lp*2+1 + 6]));
            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 8], bb[_lp*2+1 + 8]));
            _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 10], bb[_lp*2+1 + 10]));
            _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 12], bb[_lp*2+1 + 12]));
            _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 14], bb[_lp*2+1 + 14]));
            _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
        }
        {
            const unsigned* _raw_stv8_12 = reinterpret_cast<const unsigned*>(_packed_0);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(ffn_input + rbase)), "r"(_raw_stv8_12[0]), "r"(_raw_stv8_12[1]), "r"(_raw_stv8_12[2]), "r"(_raw_stv8_12[3]), "r"(_raw_stv8_12[4]), "r"(_raw_stv8_12[5]), "r"(_raw_stv8_12[6]), "r"(_raw_stv8_12[7]) : "memory");
        }
    }
}

} // extern "C"
