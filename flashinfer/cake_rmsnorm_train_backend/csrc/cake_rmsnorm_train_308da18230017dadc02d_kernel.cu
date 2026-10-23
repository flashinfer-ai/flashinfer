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
#define SMEM_PARTIALS_STAGE_BYTES 32
#define SMEM_PARTIALS_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 128
#define ROWS_PER_CTA 1
#define GROUP_THREADS 128
#define LAUNCH_BOUNDS_THREADS 128
#define LAUNCH_BOUNDS_MIN_BLOCKS 4
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS, LAUNCH_BOUNDS_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_308da18230017dadc02d(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
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

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int group = tid / GROUP_THREADS;
    int gtid = tid - group * GROUP_THREADS;
    float h_f = 6144.0f;
    int it = 0;
    #pragma unroll 1
    for (int row0 = bid * ROWS_PER_CTA; row0 < T; row0 += num_bids * ROWS_PER_CTA) {
        int row = row0 + group;
        long long row64 = (long long)row;
        long long xbase = row64 * x_stride_p0;
        long long obase = row64 * 6144;
        unsigned int x_words[24];
        unsigned int u_words[24];
        float sum_sq = 0.0f;
        if (row < T) {
            long long ubase = row64 * u_stride_p0;
            #pragma unroll
            for (int v = 0; v < 6; v++) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (xbase + (long long)(gtid * 8 + v * (GROUP_THREADS * 8))) + 0);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&x_words[v * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(u + (ubase + (long long)(gtid * 8 + v * (GROUP_THREADS * 8))) + 0);
                    uint4* _vdst_1 = reinterpret_cast<uint4*>(&u_words[v * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 6; v_1++) {
                float hsum[8];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float2 _f32x2_bf16x2_add_rhs_2 = cake_bf16x2_to_f32x2(__as_bf16x2(u_words[v_1 * 4 + k]));
                    float _bf16x2_add_f32_0[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %3;\n\t"
                        "add.rn.f32.bf16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(x_words[v_1 * 4 + k]), "f"(_f32x2_bf16x2_add_rhs_2.x), "f"(_f32x2_bf16x2_add_rhs_2.y));
                    __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_bf16x2_add_f32_0[0], _bf16x2_add_f32_0[1]));
                    unsigned int hword = __as_u32(_bf16x2_0);
                    x_words[v_1 * 4 + k] = hword;
                    float _dot_3;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                        "}\n"
                        : "=f"(_dot_3) : "r"(hword), "r"(hword), "f"(sum_sq));
                    sum_sq = _dot_3;
                    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(_bf16x2_0);
                    hsum[2 * k] = _cvt_f32_0.x;
                    hsum[2 * k + 1] = _cvt_f32_0.y;
                }
                unsigned int _packed_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hsum[_lp*2 + 0], hsum[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hsum[_lp*2 + 2], hsum[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hsum[_lp*2 + 4], hsum[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hsum[_lp*2 + 6], hsum[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(h_new + (obase + (long long)(gtid * 8 + v_1 * (GROUP_THREADS * 8))))[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
        }
        float _warp_reduce_4 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
        sum_sq = _warp_reduce_4;
        int par = (it & 1) * 4;
        if (lane == 0) {
            partials[par + warp_0] = sum_sq;
        }
        __syncthreads();
        float total = partials[par + group * 4];
        #pragma unroll
        for (int k_1 = 1; k_1 < 4; k_1++) {
            total += partials[par + group * 4 + k_1];
        }
        it += 1;
        float mean = __fdiv_rn(total, h_f);
        float _sqrt_5;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_5) : "f"(mean + eps));
        float rstd = __frcp_rn(_sqrt_5);
        if (row < T) {
            if (gtid == 0) {
                r[row] = rstd;
            }
            #pragma unroll
            for (int v_2 = 0; v_2 < 6; v_2++) {
                unsigned int wwords[4];
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(w + (gtid * 8 + v_2 * (GROUP_THREADS * 8)) + 0);
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&wwords[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)) : "memory");
                    }
                }
                float out[8];
                #pragma unroll
                for (int k_2 = 0; k_2 < 4; k_2++) {
                    float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(x_words[v_2 * 4 + k_2]));
                    float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(wwords[k_2]));
                    out[2 * k_2] = _cvt_f32_1.x * rstd * _cvt_f32_2.x;
                    out[2 * k_2 + 1] = _cvt_f32_1.y * rstd * _cvt_f32_2.y;
                }
                unsigned int _packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(y + (obase + (long long)(gtid * 8 + v_2 * (GROUP_THREADS * 8))))[0] = reinterpret_cast<int4*>(_packed_1)[0];
            }
        }
    }
}

} // extern "C"
