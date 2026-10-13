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
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 64
#define SMEM_PARTIALS_STRIDE 64
#define SMEM_TOTAL 128
#define THREADS 256
#define ROWS_PER_CTA 1
#define GROUP_THREADS 256
#define LAUNCH_BOUNDS_THREADS 256

#include <math_constants.h>


__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_rmsnorm_train_a4c187b2830114c1a279(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, long long x_stride, long long u_stride, float eps)
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
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (dependency order) ===
    int warp_0 = warp;
    int group = tid / GROUP_THREADS;
    int gtid = tid - group * GROUP_THREADS;
    float h_f = 6144.0f;
    int it = 0;
    #pragma unroll 1
    for (int row0 = bid * ROWS_PER_CTA; row0 < T; row0 += num_bids * ROWS_PER_CTA) {
        int row = row0 + group;
        long long row64 = (long long)row;
        long long xbase = row64 * x_stride;
        long long obase = row64 * 6144;
        unsigned int x_words[12];
        float sum_sq = 0.0f;
        if (row < T) {
            #pragma unroll
            for (int v = 0; v < 3; v++) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (xbase + (long long)(gtid * 8 + v * (GROUP_THREADS * 8))) + 0);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&x_words[v * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_0[_blk] = _vptr_0[_blk];
                    }
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 3; v_1++) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float _bf16x2_dot_f32_0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_bf16x2_dot_f32_0) : "r"(x_words[v_1 * 4 + k]), "r"(x_words[v_1 * 4 + k]), "f"(sum_sq));
                    sum_sq = _bf16x2_dot_f32_0;
                }
            }
        }
        float _warp_reduce_0 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        sum_sq = _warp_reduce_0;
        int par = (it & 1) * 8;
        if (lane == 0) {
            partials[par + warp_0] = sum_sq;
        }
        __syncthreads();
        float total = partials[par + group * 8];
        #pragma unroll
        for (int k_1 = 1; k_1 < 8; k_1++) {
            total += partials[par + group * 8 + k_1];
        }
        it += 1;
        float _fdiv_rn_0 = __fdiv_rn(total, h_f);
        float mean = _fdiv_rn_0;
        float _sqrt_0;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(mean + eps));
        float _rcp_0 = __frcp_rn(_sqrt_0);
        float rstd = _rcp_0;
        if (row < T) {
            if (gtid == 0) {
                r[row] = rstd;
            }
            #pragma unroll
            for (int v_2 = 0; v_2 < 3; v_2++) {
                unsigned int wwords[4];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (gtid * 8 + v_2 * (GROUP_THREADS * 8)) + 0);
                    uint4* _vdst_1 = reinterpret_cast<uint4*>(&wwords[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)) : "memory");
                    }
                }
                float out[8];
                #pragma unroll
                for (int k_2 = 0; k_2 < 4; k_2++) {
                    float2 _cvt_f32_0 = __bfloat1622float2(__as_bf16x2(x_words[v_2 * 4 + k_2]));
                    float2 _cvt_f32_1 = __bfloat1622float2(__as_bf16x2(wwords[k_2]));
                    out[2 * k_2] = _cvt_f32_0.x * rstd * _cvt_f32_1.x;
                    out[2 * k_2 + 1] = _cvt_f32_0.y * rstd * _cvt_f32_1.y;
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(out[0 + 4], out[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(out[0 + 6], out[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y + (obase + (long long)(gtid * 8 + v_2 * (GROUP_THREADS * 8)))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
    }
}

} // extern "C"
