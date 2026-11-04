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
#define SMEM_DPRE_PART_OFF 0
#define SMEM_DPRE_PART_STAGE_BYTES 1024
#define SMEM_DPRE_PART_STRIDE 1024
#define SMEM_FOLD_OFF 1024
#define SMEM_FOLD_STAGE_BYTES 4096
#define SMEM_FOLD_STRIDE 4096
#define SMEM_GSUM_OFF 5120
#define SMEM_GSUM_STAGE_BYTES 128
#define SMEM_GSUM_STRIDE 128
#define SMEM_HIST_S_OFF 6144
#define SMEM_HIST_S_STAGE_BYTES 65152
#define SMEM_HIST_S_STRIDE 65152
#define SMEM_TOTAL 71296
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" __device__ float __nv_expf(float);


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_bwd_token__rt64__h7168(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps, int sinkhorn_iters)
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
    float* dpre_part = reinterpret_cast<float*>(smem_raw + 0);
    const int dpre_part_addr = smem + 0;
    float* fold = reinterpret_cast<float*>(smem_raw + 1024);
    const int fold_addr = smem + 1024;
    float* gsum = reinterpret_cast<float*>(smem_raw + 5120);
    const int gsum_addr = smem + 5120;
    float* hist_s = reinterpret_cast<float*>(smem_raw + 6144);
    const int hist_s_addr = smem + 6144;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int warp_0 = (int)warp;
    int lane_1 = (int)lane;
    int tid_2 = tid;
    int t0 = bid * rows_per_cta;
    int t_end = t0 + rows_per_cta;
    int t1 = ((t_end < T) ? t_end : T);
    int t_last = t1 - 1;
    int n = ((lane_1 < 24) ? lane_1 : 23);
    long long n64 = (long long)n;
    long long lane64 = (long long)lane_1;
    float sc0 = scale[0];
    float sc1 = scale[1];
    float sc2 = scale[2];
    float sc = ((lane_1 < 4) ? sc0 : ((lane_1 < 8) ? sc1 : sc2));
    float bs = base[n64];
    int lane8 = lane_1 & 7;
    int lane_hi = lane_1 >> 3;
    int lc4 = ((lane_1 < 4) ? lane_1 : 3);
    int gcol = ((lane_1 >= 4) ? lane_1 - 4 : 0);
    long long col_off = (long long)tid_2 * 28;
    float acc[2];
    float comp[2];
    #pragma unroll
    for (int e = 0; e < 2; e++) {
        acc[e] = 0.0f;
        comp[e] = 0.0f;
    }
    int n_pairs = sinkhorn_iters - 1;
    bool is_logit = lane_1 >= 8 && lane_1 < 24;
    int hb = warp_0 * 16 + (lane_1 - 8);
    int zb = 16272 - (warp_0 & 1) * 16;
    int ld_base = ((is_logit) ? hb : zb);
    int ld_step = ((is_logit) ? 128 : 0);
    if (warp_0 == 0) {
        hist_s[16256 + lane_1] = 0.0f;
    }
    #pragma unroll 1
    for (int g0 = t0; g0 < t1; g0 += 8) {
        int tok_a_0 = g0;
        if (tok_a_0 < t1) {
            long long ta64_0 = (long long)tok_a_0;
            float du_row[28];
            {
                uint2 _vld_0;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_0.x), "=r"(_vld_0.y) : "l"((const void*)(dU + (ta64_0 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&du_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1.x), "=r"(_vld_1.y) : "l"((const void*)(dU + (ta64_0 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&du_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_2;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_2.x), "=r"(_vld_2.y) : "l"((const void*)(dU + (ta64_0 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&du_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_3;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_3.x), "=r"(_vld_3.y) : "l"((const void*)(dU + (ta64_0 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&du_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_4;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_4.x), "=r"(_vld_4.y) : "l"((const void*)(dU + (ta64_0 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&du_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_5;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_5.x), "=r"(_vld_5.y) : "l"((const void*)(dU + (ta64_0 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&du_row[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_6;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_6.x), "=r"(_vld_6.y) : "l"((const void*)(dU + (ta64_0 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                    (&du_row[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                }
            }
            float dots_p[4];
            float x_row[28];
            {
                uint2 _vld_7;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_7.x), "=r"(_vld_7.y) : "l"((const void*)(X + (ta64_0 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_8;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_8.x), "=r"(_vld_8.y) : "l"((const void*)(X + (ta64_0 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_9;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_9.x), "=r"(_vld_9.y) : "l"((const void*)(X + (ta64_0 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_10;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_10.x), "=r"(_vld_10.y) : "l"((const void*)(X + (ta64_0 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_11;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_11.x), "=r"(_vld_11.y) : "l"((const void*)(X + (ta64_0 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_12;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_12.x), "=r"(_vld_12.y) : "l"((const void*)(X + (ta64_0 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                    (&x_row[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_13;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_13.x), "=r"(_vld_13.y) : "l"((const void*)(X + (ta64_0 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                    (&x_row[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                }
            }
            dots_p[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_14;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[0]));
            dots_p[0] = _fma_14;
            float _fma_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[0]));
            dots_p[0] = _fma_15;
            float _fma_16;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[0]));
            dots_p[0] = _fma_16;
            float _fma_17;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[0]));
            dots_p[0] = _fma_17;
            float _fma_18;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[0]));
            dots_p[0] = _fma_18;
            float _fma_19;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[0]));
            dots_p[0] = _fma_19;
            float _fma_20;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[0]));
            dots_p[0] = _fma_20;
            float _fma_21;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[0]));
            dots_p[0] = _fma_21;
            float _fma_22;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[0]));
            dots_p[0] = _fma_22;
            float _fma_23;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[0]));
            dots_p[0] = _fma_23;
            float _fma_24;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[0]));
            dots_p[0] = _fma_24;
            float _fma_25;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[0]));
            dots_p[0] = _fma_25;
            float _fma_26;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[0]));
            dots_p[0] = _fma_26;
            float _fma_27;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[0]));
            dots_p[0] = _fma_27;
            float _fma_28;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[0]));
            dots_p[0] = _fma_28;
            float _fma_29;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[0]));
            dots_p[0] = _fma_29;
            float _fma_30;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[0]));
            dots_p[0] = _fma_30;
            float _fma_31;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[0]));
            dots_p[0] = _fma_31;
            float _fma_32;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[0]));
            dots_p[0] = _fma_32;
            float _fma_33;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(x_row[20]), "f"(du_row[20]), "f"(dots_p[0]));
            dots_p[0] = _fma_33;
            float _fma_34;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x_row[21]), "f"(du_row[21]), "f"(dots_p[0]));
            dots_p[0] = _fma_34;
            float _fma_35;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x_row[22]), "f"(du_row[22]), "f"(dots_p[0]));
            dots_p[0] = _fma_35;
            float _fma_36;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x_row[23]), "f"(du_row[23]), "f"(dots_p[0]));
            dots_p[0] = _fma_36;
            float _fma_37;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(x_row[24]), "f"(du_row[24]), "f"(dots_p[0]));
            dots_p[0] = _fma_37;
            float _fma_38;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(x_row[25]), "f"(du_row[25]), "f"(dots_p[0]));
            dots_p[0] = _fma_38;
            float _fma_39;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(x_row[26]), "f"(du_row[26]), "f"(dots_p[0]));
            dots_p[0] = _fma_39;
            float _fma_40;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(x_row[27]), "f"(du_row[27]), "f"(dots_p[0]));
            dots_p[0] = _fma_40;
            {
                uint2 _vld_41;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_41.x), "=r"(_vld_41.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_42;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_42.x), "=r"(_vld_42.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_43;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_43.x), "=r"(_vld_43.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_44;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_44.x), "=r"(_vld_44.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_45;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_45.x), "=r"(_vld_45.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_46;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_46.x), "=r"(_vld_46.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_46 = reinterpret_cast<uint32_t*>(&_vld_46);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) << 16);
                    (&x_row[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_47;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_47.x), "=r"(_vld_47.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                    (&x_row[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
                }
            }
            dots_p[1] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[1]));
            dots_p[1] = _fma_48;
            float _fma_49;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[1]));
            dots_p[1] = _fma_49;
            float _fma_50;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[1]));
            dots_p[1] = _fma_50;
            float _fma_51;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[1]));
            dots_p[1] = _fma_51;
            float _fma_52;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[1]));
            dots_p[1] = _fma_52;
            float _fma_53;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[1]));
            dots_p[1] = _fma_53;
            float _fma_54;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[1]));
            dots_p[1] = _fma_54;
            float _fma_55;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[1]));
            dots_p[1] = _fma_55;
            float _fma_56;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[1]));
            dots_p[1] = _fma_56;
            float _fma_57;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[1]));
            dots_p[1] = _fma_57;
            float _fma_58;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[1]));
            dots_p[1] = _fma_58;
            float _fma_59;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[1]));
            dots_p[1] = _fma_59;
            float _fma_60;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[1]));
            dots_p[1] = _fma_60;
            float _fma_61;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[1]));
            dots_p[1] = _fma_61;
            float _fma_62;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[1]));
            dots_p[1] = _fma_62;
            float _fma_63;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[1]));
            dots_p[1] = _fma_63;
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[1]));
            dots_p[1] = _fma_64;
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[1]));
            dots_p[1] = _fma_65;
            float _fma_66;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[1]));
            dots_p[1] = _fma_66;
            float _fma_67;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(x_row[20]), "f"(du_row[20]), "f"(dots_p[1]));
            dots_p[1] = _fma_67;
            float _fma_68;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(x_row[21]), "f"(du_row[21]), "f"(dots_p[1]));
            dots_p[1] = _fma_68;
            float _fma_69;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(x_row[22]), "f"(du_row[22]), "f"(dots_p[1]));
            dots_p[1] = _fma_69;
            float _fma_70;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(x_row[23]), "f"(du_row[23]), "f"(dots_p[1]));
            dots_p[1] = _fma_70;
            float _fma_71;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(x_row[24]), "f"(du_row[24]), "f"(dots_p[1]));
            dots_p[1] = _fma_71;
            float _fma_72;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(x_row[25]), "f"(du_row[25]), "f"(dots_p[1]));
            dots_p[1] = _fma_72;
            float _fma_73;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x_row[26]), "f"(du_row[26]), "f"(dots_p[1]));
            dots_p[1] = _fma_73;
            float _fma_74;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x_row[27]), "f"(du_row[27]), "f"(dots_p[1]));
            dots_p[1] = _fma_74;
            {
                uint2 _vld_75;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_75.x), "=r"(_vld_75.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_75 = reinterpret_cast<uint32_t*>(&_vld_75);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_75[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_76;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_76.x), "=r"(_vld_76.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_76 = reinterpret_cast<uint32_t*>(&_vld_76);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_76[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_77;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_77.x), "=r"(_vld_77.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_77 = reinterpret_cast<uint32_t*>(&_vld_77);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_78;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_78.x), "=r"(_vld_78.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_79;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_79.x), "=r"(_vld_79.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_79 = reinterpret_cast<uint32_t*>(&_vld_79);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_80;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_80.x), "=r"(_vld_80.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                    (&x_row[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_81;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_81.x), "=r"(_vld_81.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_81 = reinterpret_cast<uint32_t*>(&_vld_81);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) << 16);
                    (&x_row[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) & 0xffff0000u);
                }
            }
            dots_p[2] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[2]));
            dots_p[2] = _fma_82;
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[2]));
            dots_p[2] = _fma_83;
            float _fma_84;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[2]));
            dots_p[2] = _fma_84;
            float _fma_85;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[2]));
            dots_p[2] = _fma_85;
            float _fma_86;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[2]));
            dots_p[2] = _fma_86;
            float _fma_87;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[2]));
            dots_p[2] = _fma_87;
            float _fma_88;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[2]));
            dots_p[2] = _fma_88;
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[2]));
            dots_p[2] = _fma_89;
            float _fma_90;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[2]));
            dots_p[2] = _fma_90;
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[2]));
            dots_p[2] = _fma_91;
            float _fma_92;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[2]));
            dots_p[2] = _fma_92;
            float _fma_93;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[2]));
            dots_p[2] = _fma_93;
            float _fma_94;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[2]));
            dots_p[2] = _fma_94;
            float _fma_95;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[2]));
            dots_p[2] = _fma_95;
            float _fma_96;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[2]));
            dots_p[2] = _fma_96;
            float _fma_97;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[2]));
            dots_p[2] = _fma_97;
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[2]));
            dots_p[2] = _fma_98;
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[2]));
            dots_p[2] = _fma_99;
            float _fma_100;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[2]));
            dots_p[2] = _fma_100;
            float _fma_101;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(x_row[20]), "f"(du_row[20]), "f"(dots_p[2]));
            dots_p[2] = _fma_101;
            float _fma_102;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(x_row[21]), "f"(du_row[21]), "f"(dots_p[2]));
            dots_p[2] = _fma_102;
            float _fma_103;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(x_row[22]), "f"(du_row[22]), "f"(dots_p[2]));
            dots_p[2] = _fma_103;
            float _fma_104;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(x_row[23]), "f"(du_row[23]), "f"(dots_p[2]));
            dots_p[2] = _fma_104;
            float _fma_105;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(x_row[24]), "f"(du_row[24]), "f"(dots_p[2]));
            dots_p[2] = _fma_105;
            float _fma_106;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(x_row[25]), "f"(du_row[25]), "f"(dots_p[2]));
            dots_p[2] = _fma_106;
            float _fma_107;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(x_row[26]), "f"(du_row[26]), "f"(dots_p[2]));
            dots_p[2] = _fma_107;
            float _fma_108;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(x_row[27]), "f"(du_row[27]), "f"(dots_p[2]));
            dots_p[2] = _fma_108;
            {
                uint2 _vld_109;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_109.x), "=r"(_vld_109.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_109 = reinterpret_cast<uint32_t*>(&_vld_109);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_110;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_110.x), "=r"(_vld_110.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_110 = reinterpret_cast<uint32_t*>(&_vld_110);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_110[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_110[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_111;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_111.x), "=r"(_vld_111.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_111 = reinterpret_cast<uint32_t*>(&_vld_111);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_112;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_112.x), "=r"(_vld_112.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_112 = reinterpret_cast<uint32_t*>(&_vld_112);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_112[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_112[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_113;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_113.x), "=r"(_vld_113.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_113 = reinterpret_cast<uint32_t*>(&_vld_113);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_114;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_114.x), "=r"(_vld_114.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_114 = reinterpret_cast<uint32_t*>(&_vld_114);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_114[_pair]) << 16);
                    (&x_row[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_114[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_115;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_115.x), "=r"(_vld_115.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_115 = reinterpret_cast<uint32_t*>(&_vld_115);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_115[_pair]) << 16);
                    (&x_row[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_115[_pair]) & 0xffff0000u);
                }
            }
            dots_p[3] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_116;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[3]));
            dots_p[3] = _fma_116;
            float _fma_117;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[3]));
            dots_p[3] = _fma_117;
            float _fma_118;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[3]));
            dots_p[3] = _fma_118;
            float _fma_119;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[3]));
            dots_p[3] = _fma_119;
            float _fma_120;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[3]));
            dots_p[3] = _fma_120;
            float _fma_121;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[3]));
            dots_p[3] = _fma_121;
            float _fma_122;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[3]));
            dots_p[3] = _fma_122;
            float _fma_123;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[3]));
            dots_p[3] = _fma_123;
            float _fma_124;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[3]));
            dots_p[3] = _fma_124;
            float _fma_125;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[3]));
            dots_p[3] = _fma_125;
            float _fma_126;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[3]));
            dots_p[3] = _fma_126;
            float _fma_127;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[3]));
            dots_p[3] = _fma_127;
            float _fma_128;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[3]));
            dots_p[3] = _fma_128;
            float _fma_129;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[3]));
            dots_p[3] = _fma_129;
            float _fma_130;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[3]));
            dots_p[3] = _fma_130;
            float _fma_131;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[3]));
            dots_p[3] = _fma_131;
            float _fma_132;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[3]));
            dots_p[3] = _fma_132;
            float _fma_133;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[3]));
            dots_p[3] = _fma_133;
            float _fma_134;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[3]));
            dots_p[3] = _fma_134;
            float _fma_135;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(x_row[20]), "f"(du_row[20]), "f"(dots_p[3]));
            dots_p[3] = _fma_135;
            float _fma_136;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(x_row[21]), "f"(du_row[21]), "f"(dots_p[3]));
            dots_p[3] = _fma_136;
            float _fma_137;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(x_row[22]), "f"(du_row[22]), "f"(dots_p[3]));
            dots_p[3] = _fma_137;
            float _fma_138;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(x_row[23]), "f"(du_row[23]), "f"(dots_p[3]));
            dots_p[3] = _fma_138;
            float _fma_139;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(x_row[24]), "f"(du_row[24]), "f"(dots_p[3]));
            dots_p[3] = _fma_139;
            float _fma_140;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(x_row[25]), "f"(du_row[25]), "f"(dots_p[3]));
            dots_p[3] = _fma_140;
            float _fma_141;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(x_row[26]), "f"(du_row[26]), "f"(dots_p[3]));
            dots_p[3] = _fma_141;
            float _fma_142;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(x_row[27]), "f"(du_row[27]), "f"(dots_p[3]));
            dots_p[3] = _fma_142;
            int upper = lane_1 & 16;
            float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
            float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
            float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
            float add;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(keep), "f"(_shfl_xor_sync_143));
            float send_0 = ((upper != 0) ? dots_p[1] : dots_p[3]);
            float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
            float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
            float add_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_144));
            int upper_2 = lane_1 & 8;
            float send_3 = ((upper_2 != 0) ? add : add_1);
            float keep_4 = ((upper_2 != 0) ? add_1 : add);
            float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
            float add_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_4), "f"(_shfl_xor_sync_145));
            float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
            float add_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_146));
            float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
            float add_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(add_3), "f"(_shfl_xor_sync_147));
            float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, add_4, 1);
            float add_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_4), "f"(_shfl_xor_sync_148));
            if (lane8 == 0) {
                dpre_part[warp_0 * 4 + lane_hi] = add_5;
            }
        }
        int tok_a_1 = g0 + 1;
        if (tok_a_1 < t1) {
            long long ta64_1 = (long long)tok_a_1;
            float du_row_1[28];
            {
                uint2 _vld_149;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_149.x), "=r"(_vld_149.y) : "l"((const void*)(dU + (ta64_1 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_149 = reinterpret_cast<uint32_t*>(&_vld_149);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_149[_pair]) << 16);
                    (&du_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_149[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_150;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_150.x), "=r"(_vld_150.y) : "l"((const void*)(dU + (ta64_1 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_150 = reinterpret_cast<uint32_t*>(&_vld_150);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_150[_pair]) << 16);
                    (&du_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_150[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_151;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_151.x), "=r"(_vld_151.y) : "l"((const void*)(dU + (ta64_1 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_151 = reinterpret_cast<uint32_t*>(&_vld_151);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_151[_pair]) << 16);
                    (&du_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_151[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_152;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_152.x), "=r"(_vld_152.y) : "l"((const void*)(dU + (ta64_1 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_152 = reinterpret_cast<uint32_t*>(&_vld_152);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_152[_pair]) << 16);
                    (&du_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_152[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_153;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_153.x), "=r"(_vld_153.y) : "l"((const void*)(dU + (ta64_1 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_153 = reinterpret_cast<uint32_t*>(&_vld_153);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_153[_pair]) << 16);
                    (&du_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_153[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_154;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_154.x), "=r"(_vld_154.y) : "l"((const void*)(dU + (ta64_1 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_154 = reinterpret_cast<uint32_t*>(&_vld_154);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_154[_pair]) << 16);
                    (&du_row_1[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_154[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_155;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_155.x), "=r"(_vld_155.y) : "l"((const void*)(dU + (ta64_1 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_155 = reinterpret_cast<uint32_t*>(&_vld_155);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_155[_pair]) << 16);
                    (&du_row_1[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_155[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_1[4];
            float x_row_1[28];
            {
                uint2 _vld_156;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_156.x), "=r"(_vld_156.y) : "l"((const void*)(X + (ta64_1 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_156 = reinterpret_cast<uint32_t*>(&_vld_156);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_156[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_156[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_157;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_157.x), "=r"(_vld_157.y) : "l"((const void*)(X + (ta64_1 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_157 = reinterpret_cast<uint32_t*>(&_vld_157);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_157[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_157[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_158;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_158.x), "=r"(_vld_158.y) : "l"((const void*)(X + (ta64_1 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_158 = reinterpret_cast<uint32_t*>(&_vld_158);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_158[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_158[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_159;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_159.x), "=r"(_vld_159.y) : "l"((const void*)(X + (ta64_1 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_159 = reinterpret_cast<uint32_t*>(&_vld_159);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_159[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_159[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_160;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_160.x), "=r"(_vld_160.y) : "l"((const void*)(X + (ta64_1 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_160 = reinterpret_cast<uint32_t*>(&_vld_160);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_160[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_160[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_161;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_161.x), "=r"(_vld_161.y) : "l"((const void*)(X + (ta64_1 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_161 = reinterpret_cast<uint32_t*>(&_vld_161);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_161[_pair]) << 16);
                    (&x_row_1[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_161[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_162;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_162.x), "=r"(_vld_162.y) : "l"((const void*)(X + (ta64_1 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_162 = reinterpret_cast<uint32_t*>(&_vld_162);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_162[_pair]) << 16);
                    (&x_row_1[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_162[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[0] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_163;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_163;
            float _fma_164;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_164;
            float _fma_165;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_165;
            float _fma_166;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_166;
            float _fma_167;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_167;
            float _fma_168;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_168;
            float _fma_169;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_169;
            float _fma_170;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_170;
            float _fma_171;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_171;
            float _fma_172;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_172;
            float _fma_173;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_173;
            float _fma_174;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_174;
            float _fma_175;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_175;
            float _fma_176;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_176;
            float _fma_177;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_177;
            float _fma_178;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_178;
            float _fma_179;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_179;
            float _fma_180;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_180;
            float _fma_181;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_181;
            float _fma_182;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(x_row_1[20]), "f"(du_row_1[20]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_182;
            float _fma_183;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(x_row_1[21]), "f"(du_row_1[21]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_183;
            float _fma_184;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(x_row_1[22]), "f"(du_row_1[22]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_184;
            float _fma_185;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(x_row_1[23]), "f"(du_row_1[23]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_185;
            float _fma_186;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(x_row_1[24]), "f"(du_row_1[24]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_186;
            float _fma_187;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(x_row_1[25]), "f"(du_row_1[25]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_187;
            float _fma_188;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(x_row_1[26]), "f"(du_row_1[26]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_188;
            float _fma_189;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(x_row_1[27]), "f"(du_row_1[27]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_189;
            {
                uint2 _vld_190;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_190.x), "=r"(_vld_190.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_190 = reinterpret_cast<uint32_t*>(&_vld_190);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_190[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_190[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_191;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_191.x), "=r"(_vld_191.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_191 = reinterpret_cast<uint32_t*>(&_vld_191);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_191[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_191[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_192;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_192.x), "=r"(_vld_192.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_192 = reinterpret_cast<uint32_t*>(&_vld_192);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_192[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_192[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_193;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_193.x), "=r"(_vld_193.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_193 = reinterpret_cast<uint32_t*>(&_vld_193);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_193[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_193[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_194;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_194.x), "=r"(_vld_194.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_194 = reinterpret_cast<uint32_t*>(&_vld_194);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_194[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_194[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_195;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_195.x), "=r"(_vld_195.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_195 = reinterpret_cast<uint32_t*>(&_vld_195);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_195[_pair]) << 16);
                    (&x_row_1[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_195[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_196;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_196.x), "=r"(_vld_196.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_196 = reinterpret_cast<uint32_t*>(&_vld_196);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_196[_pair]) << 16);
                    (&x_row_1[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_196[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[1] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_197;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_197;
            float _fma_198;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_198;
            float _fma_199;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_199;
            float _fma_200;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_200;
            float _fma_201;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_201;
            float _fma_202;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_202;
            float _fma_203;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_203;
            float _fma_204;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_204;
            float _fma_205;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_205;
            float _fma_206;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_206;
            float _fma_207;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_207;
            float _fma_208;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_208;
            float _fma_209;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_209;
            float _fma_210;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_210;
            float _fma_211;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_211;
            float _fma_212;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_212;
            float _fma_213;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_213;
            float _fma_214;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_214;
            float _fma_215;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_215;
            float _fma_216;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(x_row_1[20]), "f"(du_row_1[20]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_216;
            float _fma_217;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(x_row_1[21]), "f"(du_row_1[21]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_217;
            float _fma_218;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(x_row_1[22]), "f"(du_row_1[22]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_218;
            float _fma_219;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(x_row_1[23]), "f"(du_row_1[23]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_219;
            float _fma_220;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(x_row_1[24]), "f"(du_row_1[24]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_220;
            float _fma_221;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(x_row_1[25]), "f"(du_row_1[25]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_221;
            float _fma_222;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(x_row_1[26]), "f"(du_row_1[26]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_222;
            float _fma_223;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(x_row_1[27]), "f"(du_row_1[27]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_223;
            {
                uint2 _vld_224;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_224.x), "=r"(_vld_224.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_224 = reinterpret_cast<uint32_t*>(&_vld_224);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_224[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_224[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_225;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_225.x), "=r"(_vld_225.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_225 = reinterpret_cast<uint32_t*>(&_vld_225);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_225[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_225[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_226;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_226.x), "=r"(_vld_226.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_226 = reinterpret_cast<uint32_t*>(&_vld_226);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_226[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_226[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_227;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_227.x), "=r"(_vld_227.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_227 = reinterpret_cast<uint32_t*>(&_vld_227);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_227[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_227[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_228;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_228.x), "=r"(_vld_228.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_228 = reinterpret_cast<uint32_t*>(&_vld_228);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_228[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_228[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_229;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_229.x), "=r"(_vld_229.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_229 = reinterpret_cast<uint32_t*>(&_vld_229);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_229[_pair]) << 16);
                    (&x_row_1[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_229[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_230;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_230.x), "=r"(_vld_230.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_230 = reinterpret_cast<uint32_t*>(&_vld_230);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_230[_pair]) << 16);
                    (&x_row_1[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_230[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[2] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_231;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_231;
            float _fma_232;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_232;
            float _fma_233;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_233;
            float _fma_234;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_234;
            float _fma_235;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_235;
            float _fma_236;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_236;
            float _fma_237;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_237;
            float _fma_238;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_238;
            float _fma_239;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_239;
            float _fma_240;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_240;
            float _fma_241;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_241;
            float _fma_242;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_242;
            float _fma_243;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_243;
            float _fma_244;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_244;
            float _fma_245;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_245;
            float _fma_246;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_246;
            float _fma_247;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_247;
            float _fma_248;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_248;
            float _fma_249;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_249;
            float _fma_250;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(x_row_1[20]), "f"(du_row_1[20]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_250;
            float _fma_251;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(x_row_1[21]), "f"(du_row_1[21]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_251;
            float _fma_252;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(x_row_1[22]), "f"(du_row_1[22]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_252;
            float _fma_253;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(x_row_1[23]), "f"(du_row_1[23]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_253;
            float _fma_254;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(x_row_1[24]), "f"(du_row_1[24]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_254;
            float _fma_255;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(x_row_1[25]), "f"(du_row_1[25]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_255;
            float _fma_256;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(x_row_1[26]), "f"(du_row_1[26]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_256;
            float _fma_257;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(x_row_1[27]), "f"(du_row_1[27]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_257;
            {
                uint2 _vld_258;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_258.x), "=r"(_vld_258.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_258 = reinterpret_cast<uint32_t*>(&_vld_258);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_258[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_258[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_259;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_259.x), "=r"(_vld_259.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_259 = reinterpret_cast<uint32_t*>(&_vld_259);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_259[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_259[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_260;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_260.x), "=r"(_vld_260.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_260 = reinterpret_cast<uint32_t*>(&_vld_260);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_260[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_260[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_261;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_261.x), "=r"(_vld_261.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_261 = reinterpret_cast<uint32_t*>(&_vld_261);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_261[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_261[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_262;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_262.x), "=r"(_vld_262.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_262 = reinterpret_cast<uint32_t*>(&_vld_262);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_262[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_262[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_263;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_263.x), "=r"(_vld_263.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_263 = reinterpret_cast<uint32_t*>(&_vld_263);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_263[_pair]) << 16);
                    (&x_row_1[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_263[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_264;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_264.x), "=r"(_vld_264.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_264 = reinterpret_cast<uint32_t*>(&_vld_264);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_264[_pair]) << 16);
                    (&x_row_1[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_264[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[3] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_265;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_265;
            float _fma_266;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_266;
            float _fma_267;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_267;
            float _fma_268;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_268;
            float _fma_269;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_269;
            float _fma_270;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_270;
            float _fma_271;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_271;
            float _fma_272;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_272;
            float _fma_273;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_273;
            float _fma_274;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_274;
            float _fma_275;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_275;
            float _fma_276;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_276;
            float _fma_277;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_277;
            float _fma_278;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_278;
            float _fma_279;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_279;
            float _fma_280;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_280;
            float _fma_281;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_281;
            float _fma_282;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_282;
            float _fma_283;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_283;
            float _fma_284;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(x_row_1[20]), "f"(du_row_1[20]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_284;
            float _fma_285;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(x_row_1[21]), "f"(du_row_1[21]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_285;
            float _fma_286;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(x_row_1[22]), "f"(du_row_1[22]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_286;
            float _fma_287;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(x_row_1[23]), "f"(du_row_1[23]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_287;
            float _fma_288;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(x_row_1[24]), "f"(du_row_1[24]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_288;
            float _fma_289;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(x_row_1[25]), "f"(du_row_1[25]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_289;
            float _fma_290;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(x_row_1[26]), "f"(du_row_1[26]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_290;
            float _fma_291;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(x_row_1[27]), "f"(du_row_1[27]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_291;
            int upper_1 = lane_1 & 16;
            float send_1 = ((upper_1 != 0) ? dots_p_1[0] : dots_p_1[2]);
            float keep_2 = ((upper_1 != 0) ? dots_p_1[2] : dots_p_1[0]);
            float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
            float add_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(keep_2), "f"(_shfl_xor_sync_292));
            float send_0_1 = ((upper_1 != 0) ? dots_p_1[1] : dots_p_1[3]);
            float keep_1_1 = ((upper_1 != 0) ? dots_p_1[3] : dots_p_1[1]);
            float _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, send_0_1, 16);
            float add_1_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_1) : "f"(keep_1_1), "f"(_shfl_xor_sync_293));
            int upper_2_1 = lane_1 & 8;
            float send_3_1 = ((upper_2_1 != 0) ? add_6 : add_1_1);
            float keep_4_1 = ((upper_2_1 != 0) ? add_1_1 : add_6);
            float _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, send_3_1, 8);
            float add_2_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_1) : "f"(keep_4_1), "f"(_shfl_xor_sync_294));
            float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, add_2_1, 4);
            float add_3_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_1) : "f"(add_2_1), "f"(_shfl_xor_sync_295));
            float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, add_3_1, 2);
            float add_4_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_1) : "f"(add_3_1), "f"(_shfl_xor_sync_296));
            float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, add_4_1, 1);
            float add_5_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_1) : "f"(add_4_1), "f"(_shfl_xor_sync_297));
            if (lane8 == 0) {
                dpre_part[32 + warp_0 * 4 + lane_hi] = add_5_1;
            }
        }
        int tok_a_2 = g0 + 2;
        if (tok_a_2 < t1) {
            long long ta64_2 = (long long)tok_a_2;
            float du_row_2[28];
            {
                uint2 _vld_298;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_298.x), "=r"(_vld_298.y) : "l"((const void*)(dU + (ta64_2 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_298 = reinterpret_cast<uint32_t*>(&_vld_298);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_298[_pair]) << 16);
                    (&du_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_298[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_299;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_299.x), "=r"(_vld_299.y) : "l"((const void*)(dU + (ta64_2 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_299 = reinterpret_cast<uint32_t*>(&_vld_299);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_299[_pair]) << 16);
                    (&du_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_299[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_300;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_300.x), "=r"(_vld_300.y) : "l"((const void*)(dU + (ta64_2 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_300 = reinterpret_cast<uint32_t*>(&_vld_300);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_300[_pair]) << 16);
                    (&du_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_300[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_301;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_301.x), "=r"(_vld_301.y) : "l"((const void*)(dU + (ta64_2 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_301 = reinterpret_cast<uint32_t*>(&_vld_301);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_301[_pair]) << 16);
                    (&du_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_301[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_302;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_302.x), "=r"(_vld_302.y) : "l"((const void*)(dU + (ta64_2 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_302 = reinterpret_cast<uint32_t*>(&_vld_302);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_302[_pair]) << 16);
                    (&du_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_302[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_303;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_303.x), "=r"(_vld_303.y) : "l"((const void*)(dU + (ta64_2 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_303 = reinterpret_cast<uint32_t*>(&_vld_303);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_303[_pair]) << 16);
                    (&du_row_2[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_303[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_304;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_304.x), "=r"(_vld_304.y) : "l"((const void*)(dU + (ta64_2 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_304 = reinterpret_cast<uint32_t*>(&_vld_304);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_304[_pair]) << 16);
                    (&du_row_2[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_304[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_2[4];
            float x_row_2[28];
            {
                uint2 _vld_305;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_305.x), "=r"(_vld_305.y) : "l"((const void*)(X + (ta64_2 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_305 = reinterpret_cast<uint32_t*>(&_vld_305);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_305[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_305[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_306;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_306.x), "=r"(_vld_306.y) : "l"((const void*)(X + (ta64_2 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_306 = reinterpret_cast<uint32_t*>(&_vld_306);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_306[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_306[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_307;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_307.x), "=r"(_vld_307.y) : "l"((const void*)(X + (ta64_2 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_307 = reinterpret_cast<uint32_t*>(&_vld_307);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_307[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_307[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_308;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_308.x), "=r"(_vld_308.y) : "l"((const void*)(X + (ta64_2 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_308 = reinterpret_cast<uint32_t*>(&_vld_308);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_308[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_308[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_309;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_309.x), "=r"(_vld_309.y) : "l"((const void*)(X + (ta64_2 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_309 = reinterpret_cast<uint32_t*>(&_vld_309);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_309[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_309[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_310;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_310.x), "=r"(_vld_310.y) : "l"((const void*)(X + (ta64_2 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_310 = reinterpret_cast<uint32_t*>(&_vld_310);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_310[_pair]) << 16);
                    (&x_row_2[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_310[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_311;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_311.x), "=r"(_vld_311.y) : "l"((const void*)(X + (ta64_2 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_311 = reinterpret_cast<uint32_t*>(&_vld_311);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_311[_pair]) << 16);
                    (&x_row_2[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_311[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[0] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_312;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_312;
            float _fma_313;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_313;
            float _fma_314;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_314;
            float _fma_315;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_315;
            float _fma_316;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_316;
            float _fma_317;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_317;
            float _fma_318;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_318;
            float _fma_319;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_319;
            float _fma_320;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_320;
            float _fma_321;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_321;
            float _fma_322;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_322;
            float _fma_323;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_323;
            float _fma_324;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_324;
            float _fma_325;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_325;
            float _fma_326;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_326;
            float _fma_327;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_327;
            float _fma_328;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_328;
            float _fma_329;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_329) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_329;
            float _fma_330;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_330;
            float _fma_331;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(x_row_2[20]), "f"(du_row_2[20]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_331;
            float _fma_332;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(x_row_2[21]), "f"(du_row_2[21]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_332;
            float _fma_333;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(x_row_2[22]), "f"(du_row_2[22]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_333;
            float _fma_334;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(x_row_2[23]), "f"(du_row_2[23]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_334;
            float _fma_335;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(x_row_2[24]), "f"(du_row_2[24]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_335;
            float _fma_336;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(x_row_2[25]), "f"(du_row_2[25]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_336;
            float _fma_337;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(x_row_2[26]), "f"(du_row_2[26]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_337;
            float _fma_338;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(x_row_2[27]), "f"(du_row_2[27]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_338;
            {
                uint2 _vld_339;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_339.x), "=r"(_vld_339.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_339 = reinterpret_cast<uint32_t*>(&_vld_339);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_339[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_339[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_340;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_340.x), "=r"(_vld_340.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_340 = reinterpret_cast<uint32_t*>(&_vld_340);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_340[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_340[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_341;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_341.x), "=r"(_vld_341.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_341 = reinterpret_cast<uint32_t*>(&_vld_341);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_341[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_341[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_342;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_342.x), "=r"(_vld_342.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_342 = reinterpret_cast<uint32_t*>(&_vld_342);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_342[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_342[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_343;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_343.x), "=r"(_vld_343.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_343 = reinterpret_cast<uint32_t*>(&_vld_343);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_343[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_343[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_344;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_344.x), "=r"(_vld_344.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_344 = reinterpret_cast<uint32_t*>(&_vld_344);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_344[_pair]) << 16);
                    (&x_row_2[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_344[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_345;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_345.x), "=r"(_vld_345.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_345 = reinterpret_cast<uint32_t*>(&_vld_345);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_345[_pair]) << 16);
                    (&x_row_2[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_345[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[1] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_346;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_346;
            float _fma_347;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_347;
            float _fma_348;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_348;
            float _fma_349;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_349;
            float _fma_350;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_350) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_350;
            float _fma_351;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_351) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_351;
            float _fma_352;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_352) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_352;
            float _fma_353;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_353) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_353;
            float _fma_354;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_354;
            float _fma_355;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_355;
            float _fma_356;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_356;
            float _fma_357;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_357;
            float _fma_358;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_358;
            float _fma_359;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_359;
            float _fma_360;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_360;
            float _fma_361;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_361;
            float _fma_362;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_362;
            float _fma_363;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_363;
            float _fma_364;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_364;
            float _fma_365;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(x_row_2[20]), "f"(du_row_2[20]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_365;
            float _fma_366;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(x_row_2[21]), "f"(du_row_2[21]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_366;
            float _fma_367;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(x_row_2[22]), "f"(du_row_2[22]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_367;
            float _fma_368;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(x_row_2[23]), "f"(du_row_2[23]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_368;
            float _fma_369;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(x_row_2[24]), "f"(du_row_2[24]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_369;
            float _fma_370;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(x_row_2[25]), "f"(du_row_2[25]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_370;
            float _fma_371;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(x_row_2[26]), "f"(du_row_2[26]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_371;
            float _fma_372;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(x_row_2[27]), "f"(du_row_2[27]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_372;
            {
                uint2 _vld_373;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_373.x), "=r"(_vld_373.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_373 = reinterpret_cast<uint32_t*>(&_vld_373);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_373[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_373[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_374;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_374.x), "=r"(_vld_374.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_374 = reinterpret_cast<uint32_t*>(&_vld_374);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_374[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_374[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_375;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_375.x), "=r"(_vld_375.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_375 = reinterpret_cast<uint32_t*>(&_vld_375);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_375[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_375[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_376;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_376.x), "=r"(_vld_376.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_376 = reinterpret_cast<uint32_t*>(&_vld_376);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_376[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_376[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_377;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_377.x), "=r"(_vld_377.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_377 = reinterpret_cast<uint32_t*>(&_vld_377);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_377[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_377[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_378;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_378.x), "=r"(_vld_378.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_378 = reinterpret_cast<uint32_t*>(&_vld_378);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_378[_pair]) << 16);
                    (&x_row_2[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_378[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_379;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_379.x), "=r"(_vld_379.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_379 = reinterpret_cast<uint32_t*>(&_vld_379);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_379[_pair]) << 16);
                    (&x_row_2[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_379[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[2] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_380;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_380;
            float _fma_381;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_381) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_381;
            float _fma_382;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_382) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_382;
            float _fma_383;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_383) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_383;
            float _fma_384;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_384) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_384;
            float _fma_385;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_385;
            float _fma_386;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_386) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_386;
            float _fma_387;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_387) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_387;
            float _fma_388;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_388) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_388;
            float _fma_389;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_389) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_389;
            float _fma_390;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_390) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_390;
            float _fma_391;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_391;
            float _fma_392;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_392;
            float _fma_393;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_393;
            float _fma_394;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_394) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_394;
            float _fma_395;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_395;
            float _fma_396;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_396;
            float _fma_397;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_397;
            float _fma_398;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_398;
            float _fma_399;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_399) : "f"(x_row_2[20]), "f"(du_row_2[20]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_399;
            float _fma_400;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_400) : "f"(x_row_2[21]), "f"(du_row_2[21]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_400;
            float _fma_401;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_401) : "f"(x_row_2[22]), "f"(du_row_2[22]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_401;
            float _fma_402;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(x_row_2[23]), "f"(du_row_2[23]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_402;
            float _fma_403;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(x_row_2[24]), "f"(du_row_2[24]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_403;
            float _fma_404;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(x_row_2[25]), "f"(du_row_2[25]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_404;
            float _fma_405;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(x_row_2[26]), "f"(du_row_2[26]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_405;
            float _fma_406;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(x_row_2[27]), "f"(du_row_2[27]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_406;
            {
                uint2 _vld_407;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_407.x), "=r"(_vld_407.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_407 = reinterpret_cast<uint32_t*>(&_vld_407);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_407[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_407[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_408;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_408.x), "=r"(_vld_408.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_408 = reinterpret_cast<uint32_t*>(&_vld_408);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_408[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_408[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_409;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_409.x), "=r"(_vld_409.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_409 = reinterpret_cast<uint32_t*>(&_vld_409);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_409[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_409[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_410;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_410.x), "=r"(_vld_410.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_410 = reinterpret_cast<uint32_t*>(&_vld_410);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_410[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_410[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_411;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_411.x), "=r"(_vld_411.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_411 = reinterpret_cast<uint32_t*>(&_vld_411);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_411[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_411[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_412;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_412.x), "=r"(_vld_412.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_412 = reinterpret_cast<uint32_t*>(&_vld_412);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_412[_pair]) << 16);
                    (&x_row_2[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_412[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_413;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_413.x), "=r"(_vld_413.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_413 = reinterpret_cast<uint32_t*>(&_vld_413);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_413[_pair]) << 16);
                    (&x_row_2[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_413[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[3] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_414;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_414) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_414;
            float _fma_415;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_415) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_415;
            float _fma_416;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_416) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_416;
            float _fma_417;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_417) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_417;
            float _fma_418;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_418) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_418;
            float _fma_419;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_419) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_419;
            float _fma_420;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_420) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_420;
            float _fma_421;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_421;
            float _fma_422;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_422) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_422;
            float _fma_423;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_423) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_423;
            float _fma_424;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_424) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_424;
            float _fma_425;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_425) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_425;
            float _fma_426;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_426) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_426;
            float _fma_427;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_427) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_427;
            float _fma_428;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_428) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_428;
            float _fma_429;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_429) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_429;
            float _fma_430;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_430) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_430;
            float _fma_431;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_431) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_431;
            float _fma_432;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_432) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_432;
            float _fma_433;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_433) : "f"(x_row_2[20]), "f"(du_row_2[20]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_433;
            float _fma_434;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_434) : "f"(x_row_2[21]), "f"(du_row_2[21]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_434;
            float _fma_435;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_435) : "f"(x_row_2[22]), "f"(du_row_2[22]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_435;
            float _fma_436;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_436) : "f"(x_row_2[23]), "f"(du_row_2[23]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_436;
            float _fma_437;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_437) : "f"(x_row_2[24]), "f"(du_row_2[24]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_437;
            float _fma_438;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(x_row_2[25]), "f"(du_row_2[25]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_438;
            float _fma_439;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(x_row_2[26]), "f"(du_row_2[26]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_439;
            float _fma_440;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(x_row_2[27]), "f"(du_row_2[27]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_440;
            int upper_3 = lane_1 & 16;
            float send_2 = ((upper_3 != 0) ? dots_p_2[0] : dots_p_2[2]);
            float keep_3 = ((upper_3 != 0) ? dots_p_2[2] : dots_p_2[0]);
            float _shfl_xor_sync_441 = __shfl_xor_sync(0xFFFFFFFF, send_2, 16);
            float add_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(keep_3), "f"(_shfl_xor_sync_441));
            float send_0_2 = ((upper_3 != 0) ? dots_p_2[1] : dots_p_2[3]);
            float keep_1_2 = ((upper_3 != 0) ? dots_p_2[3] : dots_p_2[1]);
            float _shfl_xor_sync_442 = __shfl_xor_sync(0xFFFFFFFF, send_0_2, 16);
            float add_1_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_2) : "f"(keep_1_2), "f"(_shfl_xor_sync_442));
            int upper_2_2 = lane_1 & 8;
            float send_3_2 = ((upper_2_2 != 0) ? add_7 : add_1_2);
            float keep_4_2 = ((upper_2_2 != 0) ? add_1_2 : add_7);
            float _shfl_xor_sync_443 = __shfl_xor_sync(0xFFFFFFFF, send_3_2, 8);
            float add_2_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_2) : "f"(keep_4_2), "f"(_shfl_xor_sync_443));
            float _shfl_xor_sync_444 = __shfl_xor_sync(0xFFFFFFFF, add_2_2, 4);
            float add_3_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_2) : "f"(add_2_2), "f"(_shfl_xor_sync_444));
            float _shfl_xor_sync_445 = __shfl_xor_sync(0xFFFFFFFF, add_3_2, 2);
            float add_4_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_2) : "f"(add_3_2), "f"(_shfl_xor_sync_445));
            float _shfl_xor_sync_446 = __shfl_xor_sync(0xFFFFFFFF, add_4_2, 1);
            float add_5_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_2) : "f"(add_4_2), "f"(_shfl_xor_sync_446));
            if (lane8 == 0) {
                dpre_part[64 + warp_0 * 4 + lane_hi] = add_5_2;
            }
        }
        int tok_a_3 = g0 + 3;
        if (tok_a_3 < t1) {
            long long ta64_3 = (long long)tok_a_3;
            float du_row_3[28];
            {
                uint2 _vld_447;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_447.x), "=r"(_vld_447.y) : "l"((const void*)(dU + (ta64_3 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_447 = reinterpret_cast<uint32_t*>(&_vld_447);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_447[_pair]) << 16);
                    (&du_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_447[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_448;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_448.x), "=r"(_vld_448.y) : "l"((const void*)(dU + (ta64_3 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_448 = reinterpret_cast<uint32_t*>(&_vld_448);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_448[_pair]) << 16);
                    (&du_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_448[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_449;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_449.x), "=r"(_vld_449.y) : "l"((const void*)(dU + (ta64_3 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_449 = reinterpret_cast<uint32_t*>(&_vld_449);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_449[_pair]) << 16);
                    (&du_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_449[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_450;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_450.x), "=r"(_vld_450.y) : "l"((const void*)(dU + (ta64_3 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_450 = reinterpret_cast<uint32_t*>(&_vld_450);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_450[_pair]) << 16);
                    (&du_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_450[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_451;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_451.x), "=r"(_vld_451.y) : "l"((const void*)(dU + (ta64_3 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_451 = reinterpret_cast<uint32_t*>(&_vld_451);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_451[_pair]) << 16);
                    (&du_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_451[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_452;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_452.x), "=r"(_vld_452.y) : "l"((const void*)(dU + (ta64_3 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_452 = reinterpret_cast<uint32_t*>(&_vld_452);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_452[_pair]) << 16);
                    (&du_row_3[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_452[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_453;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_453.x), "=r"(_vld_453.y) : "l"((const void*)(dU + (ta64_3 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_453 = reinterpret_cast<uint32_t*>(&_vld_453);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_453[_pair]) << 16);
                    (&du_row_3[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_453[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_3[4];
            float x_row_3[28];
            {
                uint2 _vld_454;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_454.x), "=r"(_vld_454.y) : "l"((const void*)(X + (ta64_3 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_454 = reinterpret_cast<uint32_t*>(&_vld_454);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_454[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_454[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_455;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_455.x), "=r"(_vld_455.y) : "l"((const void*)(X + (ta64_3 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_455 = reinterpret_cast<uint32_t*>(&_vld_455);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_455[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_455[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_456;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_456.x), "=r"(_vld_456.y) : "l"((const void*)(X + (ta64_3 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_456 = reinterpret_cast<uint32_t*>(&_vld_456);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_456[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_456[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_457;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_457.x), "=r"(_vld_457.y) : "l"((const void*)(X + (ta64_3 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_457 = reinterpret_cast<uint32_t*>(&_vld_457);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_457[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_457[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_458;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_458.x), "=r"(_vld_458.y) : "l"((const void*)(X + (ta64_3 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_458 = reinterpret_cast<uint32_t*>(&_vld_458);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_458[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_458[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_459;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_459.x), "=r"(_vld_459.y) : "l"((const void*)(X + (ta64_3 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_459 = reinterpret_cast<uint32_t*>(&_vld_459);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_459[_pair]) << 16);
                    (&x_row_3[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_459[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_460;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_460.x), "=r"(_vld_460.y) : "l"((const void*)(X + (ta64_3 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_460 = reinterpret_cast<uint32_t*>(&_vld_460);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_460[_pair]) << 16);
                    (&x_row_3[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_460[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[0] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_461;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_461) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_461;
            float _fma_462;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_462) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_462;
            float _fma_463;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_463) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_463;
            float _fma_464;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_464) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_464;
            float _fma_465;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_465) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_465;
            float _fma_466;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_466) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_466;
            float _fma_467;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_467) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_467;
            float _fma_468;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_468) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_468;
            float _fma_469;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_469) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_469;
            float _fma_470;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_470) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_470;
            float _fma_471;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_471) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_471;
            float _fma_472;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_472) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_472;
            float _fma_473;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_473;
            float _fma_474;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_474;
            float _fma_475;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_475;
            float _fma_476;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_476;
            float _fma_477;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_477;
            float _fma_478;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_478;
            float _fma_479;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_479;
            float _fma_480;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(x_row_3[20]), "f"(du_row_3[20]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_480;
            float _fma_481;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(x_row_3[21]), "f"(du_row_3[21]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_481;
            float _fma_482;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(x_row_3[22]), "f"(du_row_3[22]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_482;
            float _fma_483;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_483) : "f"(x_row_3[23]), "f"(du_row_3[23]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_483;
            float _fma_484;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_484) : "f"(x_row_3[24]), "f"(du_row_3[24]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_484;
            float _fma_485;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_485) : "f"(x_row_3[25]), "f"(du_row_3[25]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_485;
            float _fma_486;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_486) : "f"(x_row_3[26]), "f"(du_row_3[26]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_486;
            float _fma_487;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_487) : "f"(x_row_3[27]), "f"(du_row_3[27]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_487;
            {
                uint2 _vld_488;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_488.x), "=r"(_vld_488.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_488 = reinterpret_cast<uint32_t*>(&_vld_488);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_488[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_488[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_489;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_489.x), "=r"(_vld_489.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_489 = reinterpret_cast<uint32_t*>(&_vld_489);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_489[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_489[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_490;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_490.x), "=r"(_vld_490.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_490 = reinterpret_cast<uint32_t*>(&_vld_490);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_490[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_490[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_491;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_491.x), "=r"(_vld_491.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_491 = reinterpret_cast<uint32_t*>(&_vld_491);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_491[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_491[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_492;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_492.x), "=r"(_vld_492.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_492 = reinterpret_cast<uint32_t*>(&_vld_492);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_492[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_492[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_493;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_493.x), "=r"(_vld_493.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_493 = reinterpret_cast<uint32_t*>(&_vld_493);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_493[_pair]) << 16);
                    (&x_row_3[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_493[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_494;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_494.x), "=r"(_vld_494.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_494 = reinterpret_cast<uint32_t*>(&_vld_494);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_494[_pair]) << 16);
                    (&x_row_3[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_494[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[1] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_495;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_495) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_495;
            float _fma_496;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_496) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_496;
            float _fma_497;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_497) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_497;
            float _fma_498;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_498) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_498;
            float _fma_499;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_499) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_499;
            float _fma_500;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_500) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_500;
            float _fma_501;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_501) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_501;
            float _fma_502;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_502) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_502;
            float _fma_503;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_503) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_503;
            float _fma_504;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_504) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_504;
            float _fma_505;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_505) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_505;
            float _fma_506;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_506) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_506;
            float _fma_507;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_507) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_507;
            float _fma_508;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_508) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_508;
            float _fma_509;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_509) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_509;
            float _fma_510;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_510) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_510;
            float _fma_511;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_511) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_511;
            float _fma_512;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_512) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_512;
            float _fma_513;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_513) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_513;
            float _fma_514;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_514) : "f"(x_row_3[20]), "f"(du_row_3[20]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_514;
            float _fma_515;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_515) : "f"(x_row_3[21]), "f"(du_row_3[21]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_515;
            float _fma_516;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_516) : "f"(x_row_3[22]), "f"(du_row_3[22]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_516;
            float _fma_517;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_517) : "f"(x_row_3[23]), "f"(du_row_3[23]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_517;
            float _fma_518;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_518) : "f"(x_row_3[24]), "f"(du_row_3[24]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_518;
            float _fma_519;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_519) : "f"(x_row_3[25]), "f"(du_row_3[25]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_519;
            float _fma_520;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_520) : "f"(x_row_3[26]), "f"(du_row_3[26]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_520;
            float _fma_521;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_521) : "f"(x_row_3[27]), "f"(du_row_3[27]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_521;
            {
                uint2 _vld_522;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_522.x), "=r"(_vld_522.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_522 = reinterpret_cast<uint32_t*>(&_vld_522);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_522[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_522[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_523;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_523.x), "=r"(_vld_523.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_523 = reinterpret_cast<uint32_t*>(&_vld_523);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_523[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_523[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_524;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_524.x), "=r"(_vld_524.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_524 = reinterpret_cast<uint32_t*>(&_vld_524);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_524[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_524[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_525;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_525.x), "=r"(_vld_525.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_525 = reinterpret_cast<uint32_t*>(&_vld_525);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_525[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_525[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_526;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_526.x), "=r"(_vld_526.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_526 = reinterpret_cast<uint32_t*>(&_vld_526);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_526[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_526[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_527;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_527.x), "=r"(_vld_527.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_527 = reinterpret_cast<uint32_t*>(&_vld_527);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_527[_pair]) << 16);
                    (&x_row_3[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_527[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_528;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_528.x), "=r"(_vld_528.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_528 = reinterpret_cast<uint32_t*>(&_vld_528);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_528[_pair]) << 16);
                    (&x_row_3[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_528[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[2] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_529;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_529) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_529;
            float _fma_530;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_530) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_530;
            float _fma_531;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_531) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_531;
            float _fma_532;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_532) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_532;
            float _fma_533;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_533) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_533;
            float _fma_534;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_534) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_534;
            float _fma_535;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_535) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_535;
            float _fma_536;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_536) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_536;
            float _fma_537;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_537) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_537;
            float _fma_538;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_538) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_538;
            float _fma_539;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_539) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_539;
            float _fma_540;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_540) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_540;
            float _fma_541;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_541) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_541;
            float _fma_542;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_542) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_542;
            float _fma_543;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_543) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_543;
            float _fma_544;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_544) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_544;
            float _fma_545;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_545) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_545;
            float _fma_546;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_546) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_546;
            float _fma_547;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_547) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_547;
            float _fma_548;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_548) : "f"(x_row_3[20]), "f"(du_row_3[20]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_548;
            float _fma_549;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_549) : "f"(x_row_3[21]), "f"(du_row_3[21]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_549;
            float _fma_550;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_550) : "f"(x_row_3[22]), "f"(du_row_3[22]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_550;
            float _fma_551;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_551) : "f"(x_row_3[23]), "f"(du_row_3[23]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_551;
            float _fma_552;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_552) : "f"(x_row_3[24]), "f"(du_row_3[24]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_552;
            float _fma_553;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_553) : "f"(x_row_3[25]), "f"(du_row_3[25]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_553;
            float _fma_554;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_554) : "f"(x_row_3[26]), "f"(du_row_3[26]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_554;
            float _fma_555;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_555) : "f"(x_row_3[27]), "f"(du_row_3[27]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_555;
            {
                uint2 _vld_556;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_556.x), "=r"(_vld_556.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_556 = reinterpret_cast<uint32_t*>(&_vld_556);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_556[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_556[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_557;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_557.x), "=r"(_vld_557.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_557 = reinterpret_cast<uint32_t*>(&_vld_557);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_557[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_557[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_558;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_558.x), "=r"(_vld_558.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_558 = reinterpret_cast<uint32_t*>(&_vld_558);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_558[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_558[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_559;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_559.x), "=r"(_vld_559.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_559 = reinterpret_cast<uint32_t*>(&_vld_559);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_559[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_559[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_560;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_560.x), "=r"(_vld_560.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_560 = reinterpret_cast<uint32_t*>(&_vld_560);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_560[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_560[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_561;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_561.x), "=r"(_vld_561.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_561 = reinterpret_cast<uint32_t*>(&_vld_561);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_561[_pair]) << 16);
                    (&x_row_3[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_561[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_562;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_562.x), "=r"(_vld_562.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_562 = reinterpret_cast<uint32_t*>(&_vld_562);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_562[_pair]) << 16);
                    (&x_row_3[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_562[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[3] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_563;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_563) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_563;
            float _fma_564;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_564) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_564;
            float _fma_565;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_565) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_565;
            float _fma_566;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_566) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_566;
            float _fma_567;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_567) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_567;
            float _fma_568;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_568) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_568;
            float _fma_569;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_569) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_569;
            float _fma_570;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_570) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_570;
            float _fma_571;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_571) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_571;
            float _fma_572;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_572) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_572;
            float _fma_573;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_573) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_573;
            float _fma_574;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_574) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_574;
            float _fma_575;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_575) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_575;
            float _fma_576;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_576) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_576;
            float _fma_577;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_577) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_577;
            float _fma_578;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_578) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_578;
            float _fma_579;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_579) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_579;
            float _fma_580;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_580) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_580;
            float _fma_581;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_581) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_581;
            float _fma_582;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_582) : "f"(x_row_3[20]), "f"(du_row_3[20]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_582;
            float _fma_583;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_583) : "f"(x_row_3[21]), "f"(du_row_3[21]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_583;
            float _fma_584;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_584) : "f"(x_row_3[22]), "f"(du_row_3[22]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_584;
            float _fma_585;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_585) : "f"(x_row_3[23]), "f"(du_row_3[23]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_585;
            float _fma_586;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_586) : "f"(x_row_3[24]), "f"(du_row_3[24]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_586;
            float _fma_587;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_587) : "f"(x_row_3[25]), "f"(du_row_3[25]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_587;
            float _fma_588;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_588) : "f"(x_row_3[26]), "f"(du_row_3[26]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_588;
            float _fma_589;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_589) : "f"(x_row_3[27]), "f"(du_row_3[27]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_589;
            int upper_4 = lane_1 & 16;
            float send_4 = ((upper_4 != 0) ? dots_p_3[0] : dots_p_3[2]);
            float keep_5 = ((upper_4 != 0) ? dots_p_3[2] : dots_p_3[0]);
            float _shfl_xor_sync_590 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
            float add_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(keep_5), "f"(_shfl_xor_sync_590));
            float send_0_3 = ((upper_4 != 0) ? dots_p_3[1] : dots_p_3[3]);
            float keep_1_3 = ((upper_4 != 0) ? dots_p_3[3] : dots_p_3[1]);
            float _shfl_xor_sync_591 = __shfl_xor_sync(0xFFFFFFFF, send_0_3, 16);
            float add_1_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_3) : "f"(keep_1_3), "f"(_shfl_xor_sync_591));
            int upper_2_3 = lane_1 & 8;
            float send_3_3 = ((upper_2_3 != 0) ? add_8 : add_1_3);
            float keep_4_3 = ((upper_2_3 != 0) ? add_1_3 : add_8);
            float _shfl_xor_sync_592 = __shfl_xor_sync(0xFFFFFFFF, send_3_3, 8);
            float add_2_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_3) : "f"(keep_4_3), "f"(_shfl_xor_sync_592));
            float _shfl_xor_sync_593 = __shfl_xor_sync(0xFFFFFFFF, add_2_3, 4);
            float add_3_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_3) : "f"(add_2_3), "f"(_shfl_xor_sync_593));
            float _shfl_xor_sync_594 = __shfl_xor_sync(0xFFFFFFFF, add_3_3, 2);
            float add_4_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_3) : "f"(add_3_3), "f"(_shfl_xor_sync_594));
            float _shfl_xor_sync_595 = __shfl_xor_sync(0xFFFFFFFF, add_4_3, 1);
            float add_5_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_3) : "f"(add_4_3), "f"(_shfl_xor_sync_595));
            if (lane8 == 0) {
                dpre_part[96 + warp_0 * 4 + lane_hi] = add_5_3;
            }
        }
        int tok_a_4 = g0 + 4;
        if (tok_a_4 < t1) {
            long long ta64_4 = (long long)tok_a_4;
            float du_row_4[28];
            {
                uint2 _vld_596;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_596.x), "=r"(_vld_596.y) : "l"((const void*)(dU + (ta64_4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_596 = reinterpret_cast<uint32_t*>(&_vld_596);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_596[_pair]) << 16);
                    (&du_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_596[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_597;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_597.x), "=r"(_vld_597.y) : "l"((const void*)(dU + (ta64_4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_597 = reinterpret_cast<uint32_t*>(&_vld_597);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_597[_pair]) << 16);
                    (&du_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_597[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_598;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_598.x), "=r"(_vld_598.y) : "l"((const void*)(dU + (ta64_4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_598 = reinterpret_cast<uint32_t*>(&_vld_598);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_598[_pair]) << 16);
                    (&du_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_598[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_599;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_599.x), "=r"(_vld_599.y) : "l"((const void*)(dU + (ta64_4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_599 = reinterpret_cast<uint32_t*>(&_vld_599);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_599[_pair]) << 16);
                    (&du_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_599[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_600;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_600.x), "=r"(_vld_600.y) : "l"((const void*)(dU + (ta64_4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_600 = reinterpret_cast<uint32_t*>(&_vld_600);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_600[_pair]) << 16);
                    (&du_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_600[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_601;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_601.x), "=r"(_vld_601.y) : "l"((const void*)(dU + (ta64_4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_601 = reinterpret_cast<uint32_t*>(&_vld_601);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_601[_pair]) << 16);
                    (&du_row_4[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_601[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_602;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_602.x), "=r"(_vld_602.y) : "l"((const void*)(dU + (ta64_4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_602 = reinterpret_cast<uint32_t*>(&_vld_602);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_602[_pair]) << 16);
                    (&du_row_4[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_602[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_4[4];
            float x_row_4[28];
            {
                uint2 _vld_603;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_603.x), "=r"(_vld_603.y) : "l"((const void*)(X + (ta64_4 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_603 = reinterpret_cast<uint32_t*>(&_vld_603);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_603[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_603[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_604;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_604.x), "=r"(_vld_604.y) : "l"((const void*)(X + (ta64_4 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_604 = reinterpret_cast<uint32_t*>(&_vld_604);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_604[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_604[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_605;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_605.x), "=r"(_vld_605.y) : "l"((const void*)(X + (ta64_4 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_605 = reinterpret_cast<uint32_t*>(&_vld_605);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_605[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_605[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_606;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_606.x), "=r"(_vld_606.y) : "l"((const void*)(X + (ta64_4 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_606 = reinterpret_cast<uint32_t*>(&_vld_606);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_606[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_606[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_607;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_607.x), "=r"(_vld_607.y) : "l"((const void*)(X + (ta64_4 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_607 = reinterpret_cast<uint32_t*>(&_vld_607);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_607[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_607[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_608;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_608.x), "=r"(_vld_608.y) : "l"((const void*)(X + (ta64_4 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_608 = reinterpret_cast<uint32_t*>(&_vld_608);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_608[_pair]) << 16);
                    (&x_row_4[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_608[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_609;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_609.x), "=r"(_vld_609.y) : "l"((const void*)(X + (ta64_4 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_609 = reinterpret_cast<uint32_t*>(&_vld_609);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_609[_pair]) << 16);
                    (&x_row_4[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_609[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[0] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_610;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_610) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_610;
            float _fma_611;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_611) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_611;
            float _fma_612;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_612) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_612;
            float _fma_613;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_613) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_613;
            float _fma_614;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_614) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_614;
            float _fma_615;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_615) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_615;
            float _fma_616;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_616) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_616;
            float _fma_617;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_617) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_617;
            float _fma_618;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_618) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_618;
            float _fma_619;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_619) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_619;
            float _fma_620;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_620) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_620;
            float _fma_621;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_621) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_621;
            float _fma_622;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_622) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_622;
            float _fma_623;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_623) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_623;
            float _fma_624;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_624) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_624;
            float _fma_625;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_625) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_625;
            float _fma_626;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_626) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_626;
            float _fma_627;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_627) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_627;
            float _fma_628;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_628) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_628;
            float _fma_629;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_629) : "f"(x_row_4[20]), "f"(du_row_4[20]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_629;
            float _fma_630;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_630) : "f"(x_row_4[21]), "f"(du_row_4[21]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_630;
            float _fma_631;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_631) : "f"(x_row_4[22]), "f"(du_row_4[22]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_631;
            float _fma_632;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_632) : "f"(x_row_4[23]), "f"(du_row_4[23]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_632;
            float _fma_633;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_633) : "f"(x_row_4[24]), "f"(du_row_4[24]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_633;
            float _fma_634;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_634) : "f"(x_row_4[25]), "f"(du_row_4[25]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_634;
            float _fma_635;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_635) : "f"(x_row_4[26]), "f"(du_row_4[26]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_635;
            float _fma_636;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_636) : "f"(x_row_4[27]), "f"(du_row_4[27]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_636;
            {
                uint2 _vld_637;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_637.x), "=r"(_vld_637.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_637 = reinterpret_cast<uint32_t*>(&_vld_637);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_637[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_637[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_638;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_638.x), "=r"(_vld_638.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_638 = reinterpret_cast<uint32_t*>(&_vld_638);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_638[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_638[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_639;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_639.x), "=r"(_vld_639.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_639 = reinterpret_cast<uint32_t*>(&_vld_639);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_639[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_639[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_640;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_640.x), "=r"(_vld_640.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_640 = reinterpret_cast<uint32_t*>(&_vld_640);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_640[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_640[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_641;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_641.x), "=r"(_vld_641.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_641 = reinterpret_cast<uint32_t*>(&_vld_641);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_641[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_641[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_642;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_642.x), "=r"(_vld_642.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_642 = reinterpret_cast<uint32_t*>(&_vld_642);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_642[_pair]) << 16);
                    (&x_row_4[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_642[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_643;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_643.x), "=r"(_vld_643.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_643 = reinterpret_cast<uint32_t*>(&_vld_643);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_643[_pair]) << 16);
                    (&x_row_4[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_643[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[1] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_644;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_644) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_644;
            float _fma_645;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_645) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_645;
            float _fma_646;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_646) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_646;
            float _fma_647;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_647) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_647;
            float _fma_648;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_648) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_648;
            float _fma_649;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_649) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_649;
            float _fma_650;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_650) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_650;
            float _fma_651;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_651) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_651;
            float _fma_652;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_652) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_652;
            float _fma_653;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_653) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_653;
            float _fma_654;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_654) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_654;
            float _fma_655;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_655) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_655;
            float _fma_656;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_656) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_656;
            float _fma_657;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_657) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_657;
            float _fma_658;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_658) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_658;
            float _fma_659;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_659) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_659;
            float _fma_660;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_660) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_660;
            float _fma_661;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_661) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_661;
            float _fma_662;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_662) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_662;
            float _fma_663;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_663) : "f"(x_row_4[20]), "f"(du_row_4[20]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_663;
            float _fma_664;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_664) : "f"(x_row_4[21]), "f"(du_row_4[21]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_664;
            float _fma_665;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_665) : "f"(x_row_4[22]), "f"(du_row_4[22]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_665;
            float _fma_666;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_666) : "f"(x_row_4[23]), "f"(du_row_4[23]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_666;
            float _fma_667;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_667) : "f"(x_row_4[24]), "f"(du_row_4[24]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_667;
            float _fma_668;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_668) : "f"(x_row_4[25]), "f"(du_row_4[25]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_668;
            float _fma_669;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_669) : "f"(x_row_4[26]), "f"(du_row_4[26]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_669;
            float _fma_670;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_670) : "f"(x_row_4[27]), "f"(du_row_4[27]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_670;
            {
                uint2 _vld_671;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_671.x), "=r"(_vld_671.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_671 = reinterpret_cast<uint32_t*>(&_vld_671);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_671[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_671[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_672;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_672.x), "=r"(_vld_672.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_672 = reinterpret_cast<uint32_t*>(&_vld_672);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_672[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_672[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_673;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_673.x), "=r"(_vld_673.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_673 = reinterpret_cast<uint32_t*>(&_vld_673);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_673[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_673[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_674;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_674.x), "=r"(_vld_674.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_674 = reinterpret_cast<uint32_t*>(&_vld_674);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_674[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_674[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_675;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_675.x), "=r"(_vld_675.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_675 = reinterpret_cast<uint32_t*>(&_vld_675);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_675[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_675[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_676;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_676.x), "=r"(_vld_676.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_676 = reinterpret_cast<uint32_t*>(&_vld_676);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_676[_pair]) << 16);
                    (&x_row_4[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_676[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_677;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_677.x), "=r"(_vld_677.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_677 = reinterpret_cast<uint32_t*>(&_vld_677);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_677[_pair]) << 16);
                    (&x_row_4[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_677[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[2] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_678;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_678) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_678;
            float _fma_679;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_679) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_679;
            float _fma_680;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_680) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_680;
            float _fma_681;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_681) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_681;
            float _fma_682;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_682) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_682;
            float _fma_683;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_683) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_683;
            float _fma_684;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_684) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_684;
            float _fma_685;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_685) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_685;
            float _fma_686;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_686) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_686;
            float _fma_687;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_687) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_687;
            float _fma_688;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_688) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_688;
            float _fma_689;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_689) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_689;
            float _fma_690;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_690) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_690;
            float _fma_691;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_691) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_691;
            float _fma_692;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_692) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_692;
            float _fma_693;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_693) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_693;
            float _fma_694;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_694) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_694;
            float _fma_695;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_695) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_695;
            float _fma_696;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_696) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_696;
            float _fma_697;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_697) : "f"(x_row_4[20]), "f"(du_row_4[20]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_697;
            float _fma_698;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_698) : "f"(x_row_4[21]), "f"(du_row_4[21]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_698;
            float _fma_699;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_699) : "f"(x_row_4[22]), "f"(du_row_4[22]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_699;
            float _fma_700;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_700) : "f"(x_row_4[23]), "f"(du_row_4[23]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_700;
            float _fma_701;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_701) : "f"(x_row_4[24]), "f"(du_row_4[24]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_701;
            float _fma_702;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_702) : "f"(x_row_4[25]), "f"(du_row_4[25]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_702;
            float _fma_703;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_703) : "f"(x_row_4[26]), "f"(du_row_4[26]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_703;
            float _fma_704;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_704) : "f"(x_row_4[27]), "f"(du_row_4[27]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_704;
            {
                uint2 _vld_705;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_705.x), "=r"(_vld_705.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_705 = reinterpret_cast<uint32_t*>(&_vld_705);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_705[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_705[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_706;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_706.x), "=r"(_vld_706.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_706 = reinterpret_cast<uint32_t*>(&_vld_706);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_706[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_706[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_707;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_707.x), "=r"(_vld_707.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_707 = reinterpret_cast<uint32_t*>(&_vld_707);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_707[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_707[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_708;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_708.x), "=r"(_vld_708.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_708 = reinterpret_cast<uint32_t*>(&_vld_708);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_708[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_708[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_709;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_709.x), "=r"(_vld_709.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_709 = reinterpret_cast<uint32_t*>(&_vld_709);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_709[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_709[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_710;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_710.x), "=r"(_vld_710.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_710 = reinterpret_cast<uint32_t*>(&_vld_710);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_710[_pair]) << 16);
                    (&x_row_4[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_710[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_711;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_711.x), "=r"(_vld_711.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_711 = reinterpret_cast<uint32_t*>(&_vld_711);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_711[_pair]) << 16);
                    (&x_row_4[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_711[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[3] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_712;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_712) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_712;
            float _fma_713;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_713) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_713;
            float _fma_714;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_714) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_714;
            float _fma_715;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_715) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_715;
            float _fma_716;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_716) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_716;
            float _fma_717;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_717) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_717;
            float _fma_718;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_718) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_718;
            float _fma_719;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_719) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_719;
            float _fma_720;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_720) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_720;
            float _fma_721;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_721) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_721;
            float _fma_722;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_722) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_722;
            float _fma_723;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_723) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_723;
            float _fma_724;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_724) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_724;
            float _fma_725;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_725) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_725;
            float _fma_726;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_726) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_726;
            float _fma_727;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_727) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_727;
            float _fma_728;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_728) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_728;
            float _fma_729;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_729) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_729;
            float _fma_730;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_730) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_730;
            float _fma_731;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_731) : "f"(x_row_4[20]), "f"(du_row_4[20]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_731;
            float _fma_732;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_732) : "f"(x_row_4[21]), "f"(du_row_4[21]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_732;
            float _fma_733;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_733) : "f"(x_row_4[22]), "f"(du_row_4[22]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_733;
            float _fma_734;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_734) : "f"(x_row_4[23]), "f"(du_row_4[23]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_734;
            float _fma_735;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_735) : "f"(x_row_4[24]), "f"(du_row_4[24]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_735;
            float _fma_736;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_736) : "f"(x_row_4[25]), "f"(du_row_4[25]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_736;
            float _fma_737;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_737) : "f"(x_row_4[26]), "f"(du_row_4[26]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_737;
            float _fma_738;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_738) : "f"(x_row_4[27]), "f"(du_row_4[27]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_738;
            int upper_5 = lane_1 & 16;
            float send_5 = ((upper_5 != 0) ? dots_p_4[0] : dots_p_4[2]);
            float keep_6 = ((upper_5 != 0) ? dots_p_4[2] : dots_p_4[0]);
            float _shfl_xor_sync_739 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
            float add_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(keep_6), "f"(_shfl_xor_sync_739));
            float send_0_4 = ((upper_5 != 0) ? dots_p_4[1] : dots_p_4[3]);
            float keep_1_4 = ((upper_5 != 0) ? dots_p_4[3] : dots_p_4[1]);
            float _shfl_xor_sync_740 = __shfl_xor_sync(0xFFFFFFFF, send_0_4, 16);
            float add_1_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_4) : "f"(keep_1_4), "f"(_shfl_xor_sync_740));
            int upper_2_4 = lane_1 & 8;
            float send_3_4 = ((upper_2_4 != 0) ? add_9 : add_1_4);
            float keep_4_4 = ((upper_2_4 != 0) ? add_1_4 : add_9);
            float _shfl_xor_sync_741 = __shfl_xor_sync(0xFFFFFFFF, send_3_4, 8);
            float add_2_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_4) : "f"(keep_4_4), "f"(_shfl_xor_sync_741));
            float _shfl_xor_sync_742 = __shfl_xor_sync(0xFFFFFFFF, add_2_4, 4);
            float add_3_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_4) : "f"(add_2_4), "f"(_shfl_xor_sync_742));
            float _shfl_xor_sync_743 = __shfl_xor_sync(0xFFFFFFFF, add_3_4, 2);
            float add_4_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_4) : "f"(add_3_4), "f"(_shfl_xor_sync_743));
            float _shfl_xor_sync_744 = __shfl_xor_sync(0xFFFFFFFF, add_4_4, 1);
            float add_5_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_4) : "f"(add_4_4), "f"(_shfl_xor_sync_744));
            if (lane8 == 0) {
                dpre_part[128 + warp_0 * 4 + lane_hi] = add_5_4;
            }
        }
        int tok_a_5 = g0 + 5;
        if (tok_a_5 < t1) {
            long long ta64_5 = (long long)tok_a_5;
            float du_row_5[28];
            {
                uint2 _vld_745;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_745.x), "=r"(_vld_745.y) : "l"((const void*)(dU + (ta64_5 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_745 = reinterpret_cast<uint32_t*>(&_vld_745);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_745[_pair]) << 16);
                    (&du_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_745[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_746;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_746.x), "=r"(_vld_746.y) : "l"((const void*)(dU + (ta64_5 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_746 = reinterpret_cast<uint32_t*>(&_vld_746);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_746[_pair]) << 16);
                    (&du_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_746[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_747;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_747.x), "=r"(_vld_747.y) : "l"((const void*)(dU + (ta64_5 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_747 = reinterpret_cast<uint32_t*>(&_vld_747);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_747[_pair]) << 16);
                    (&du_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_747[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_748;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_748.x), "=r"(_vld_748.y) : "l"((const void*)(dU + (ta64_5 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_748 = reinterpret_cast<uint32_t*>(&_vld_748);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_748[_pair]) << 16);
                    (&du_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_748[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_749;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_749.x), "=r"(_vld_749.y) : "l"((const void*)(dU + (ta64_5 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_749 = reinterpret_cast<uint32_t*>(&_vld_749);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_749[_pair]) << 16);
                    (&du_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_749[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_750;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_750.x), "=r"(_vld_750.y) : "l"((const void*)(dU + (ta64_5 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_750 = reinterpret_cast<uint32_t*>(&_vld_750);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_750[_pair]) << 16);
                    (&du_row_5[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_750[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_751;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_751.x), "=r"(_vld_751.y) : "l"((const void*)(dU + (ta64_5 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_751 = reinterpret_cast<uint32_t*>(&_vld_751);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_751[_pair]) << 16);
                    (&du_row_5[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_751[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_5[4];
            float x_row_5[28];
            {
                uint2 _vld_752;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_752.x), "=r"(_vld_752.y) : "l"((const void*)(X + (ta64_5 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_752 = reinterpret_cast<uint32_t*>(&_vld_752);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_752[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_752[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_753;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_753.x), "=r"(_vld_753.y) : "l"((const void*)(X + (ta64_5 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_753 = reinterpret_cast<uint32_t*>(&_vld_753);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_753[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_753[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_754;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_754.x), "=r"(_vld_754.y) : "l"((const void*)(X + (ta64_5 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_754 = reinterpret_cast<uint32_t*>(&_vld_754);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_754[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_754[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_755;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_755.x), "=r"(_vld_755.y) : "l"((const void*)(X + (ta64_5 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_755 = reinterpret_cast<uint32_t*>(&_vld_755);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_755[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_755[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_756;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_756.x), "=r"(_vld_756.y) : "l"((const void*)(X + (ta64_5 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_756 = reinterpret_cast<uint32_t*>(&_vld_756);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_756[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_756[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_757;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_757.x), "=r"(_vld_757.y) : "l"((const void*)(X + (ta64_5 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_757 = reinterpret_cast<uint32_t*>(&_vld_757);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_757[_pair]) << 16);
                    (&x_row_5[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_757[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_758;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_758.x), "=r"(_vld_758.y) : "l"((const void*)(X + (ta64_5 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_758 = reinterpret_cast<uint32_t*>(&_vld_758);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_758[_pair]) << 16);
                    (&x_row_5[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_758[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[0] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_759;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_759) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_759;
            float _fma_760;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_760) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_760;
            float _fma_761;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_761) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_761;
            float _fma_762;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_762) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_762;
            float _fma_763;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_763) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_763;
            float _fma_764;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_764) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_764;
            float _fma_765;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_765) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_765;
            float _fma_766;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_766) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_766;
            float _fma_767;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_767) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_767;
            float _fma_768;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_768) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_768;
            float _fma_769;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_769) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_769;
            float _fma_770;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_770) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_770;
            float _fma_771;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_771) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_771;
            float _fma_772;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_772) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_772;
            float _fma_773;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_773) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_773;
            float _fma_774;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_774) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_774;
            float _fma_775;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_775) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_775;
            float _fma_776;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_776) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_776;
            float _fma_777;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_777) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_777;
            float _fma_778;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_778) : "f"(x_row_5[20]), "f"(du_row_5[20]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_778;
            float _fma_779;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_779) : "f"(x_row_5[21]), "f"(du_row_5[21]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_779;
            float _fma_780;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_780) : "f"(x_row_5[22]), "f"(du_row_5[22]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_780;
            float _fma_781;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_781) : "f"(x_row_5[23]), "f"(du_row_5[23]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_781;
            float _fma_782;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_782) : "f"(x_row_5[24]), "f"(du_row_5[24]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_782;
            float _fma_783;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_783) : "f"(x_row_5[25]), "f"(du_row_5[25]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_783;
            float _fma_784;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_784) : "f"(x_row_5[26]), "f"(du_row_5[26]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_784;
            float _fma_785;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_785) : "f"(x_row_5[27]), "f"(du_row_5[27]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_785;
            {
                uint2 _vld_786;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_786.x), "=r"(_vld_786.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_786 = reinterpret_cast<uint32_t*>(&_vld_786);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_786[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_786[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_787;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_787.x), "=r"(_vld_787.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_787 = reinterpret_cast<uint32_t*>(&_vld_787);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_787[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_787[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_788;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_788.x), "=r"(_vld_788.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_788 = reinterpret_cast<uint32_t*>(&_vld_788);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_788[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_788[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_789;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_789.x), "=r"(_vld_789.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_789 = reinterpret_cast<uint32_t*>(&_vld_789);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_789[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_789[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_790;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_790.x), "=r"(_vld_790.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_790 = reinterpret_cast<uint32_t*>(&_vld_790);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_790[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_790[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_791;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_791.x), "=r"(_vld_791.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_791 = reinterpret_cast<uint32_t*>(&_vld_791);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_791[_pair]) << 16);
                    (&x_row_5[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_791[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_792;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_792.x), "=r"(_vld_792.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_792 = reinterpret_cast<uint32_t*>(&_vld_792);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_792[_pair]) << 16);
                    (&x_row_5[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_792[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[1] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_793;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_793) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_793;
            float _fma_794;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_794) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_794;
            float _fma_795;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_795) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_795;
            float _fma_796;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_796) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_796;
            float _fma_797;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_797) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_797;
            float _fma_798;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_798) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_798;
            float _fma_799;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_799) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_799;
            float _fma_800;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_800) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_800;
            float _fma_801;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_801) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_801;
            float _fma_802;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_802) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_802;
            float _fma_803;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_803) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_803;
            float _fma_804;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_804) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_804;
            float _fma_805;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_805) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_805;
            float _fma_806;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_806) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_806;
            float _fma_807;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_807) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_807;
            float _fma_808;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_808) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_808;
            float _fma_809;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_809) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_809;
            float _fma_810;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_810) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_810;
            float _fma_811;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_811) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_811;
            float _fma_812;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_812) : "f"(x_row_5[20]), "f"(du_row_5[20]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_812;
            float _fma_813;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_813) : "f"(x_row_5[21]), "f"(du_row_5[21]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_813;
            float _fma_814;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_814) : "f"(x_row_5[22]), "f"(du_row_5[22]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_814;
            float _fma_815;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_815) : "f"(x_row_5[23]), "f"(du_row_5[23]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_815;
            float _fma_816;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_816) : "f"(x_row_5[24]), "f"(du_row_5[24]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_816;
            float _fma_817;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_817) : "f"(x_row_5[25]), "f"(du_row_5[25]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_817;
            float _fma_818;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_818) : "f"(x_row_5[26]), "f"(du_row_5[26]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_818;
            float _fma_819;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_819) : "f"(x_row_5[27]), "f"(du_row_5[27]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_819;
            {
                uint2 _vld_820;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_820.x), "=r"(_vld_820.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_820 = reinterpret_cast<uint32_t*>(&_vld_820);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_820[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_820[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_821;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_821.x), "=r"(_vld_821.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_821 = reinterpret_cast<uint32_t*>(&_vld_821);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_821[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_821[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_822;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_822.x), "=r"(_vld_822.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_822 = reinterpret_cast<uint32_t*>(&_vld_822);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_822[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_822[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_823;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_823.x), "=r"(_vld_823.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_823 = reinterpret_cast<uint32_t*>(&_vld_823);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_823[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_823[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_824;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_824.x), "=r"(_vld_824.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_824 = reinterpret_cast<uint32_t*>(&_vld_824);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_824[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_824[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_825;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_825.x), "=r"(_vld_825.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_825 = reinterpret_cast<uint32_t*>(&_vld_825);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_825[_pair]) << 16);
                    (&x_row_5[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_825[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_826;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_826.x), "=r"(_vld_826.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_826 = reinterpret_cast<uint32_t*>(&_vld_826);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_826[_pair]) << 16);
                    (&x_row_5[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_826[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[2] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_827;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_827) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_827;
            float _fma_828;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_828) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_828;
            float _fma_829;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_829) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_829;
            float _fma_830;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_830) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_830;
            float _fma_831;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_831) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_831;
            float _fma_832;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_832) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_832;
            float _fma_833;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_833) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_833;
            float _fma_834;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_834) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_834;
            float _fma_835;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_835) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_835;
            float _fma_836;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_836) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_836;
            float _fma_837;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_837) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_837;
            float _fma_838;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_838) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_838;
            float _fma_839;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_839) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_839;
            float _fma_840;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_840) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_840;
            float _fma_841;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_841) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_841;
            float _fma_842;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_842) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_842;
            float _fma_843;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_843) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_843;
            float _fma_844;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_844) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_844;
            float _fma_845;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_845) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_845;
            float _fma_846;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_846) : "f"(x_row_5[20]), "f"(du_row_5[20]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_846;
            float _fma_847;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_847) : "f"(x_row_5[21]), "f"(du_row_5[21]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_847;
            float _fma_848;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_848) : "f"(x_row_5[22]), "f"(du_row_5[22]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_848;
            float _fma_849;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_849) : "f"(x_row_5[23]), "f"(du_row_5[23]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_849;
            float _fma_850;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_850) : "f"(x_row_5[24]), "f"(du_row_5[24]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_850;
            float _fma_851;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_851) : "f"(x_row_5[25]), "f"(du_row_5[25]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_851;
            float _fma_852;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_852) : "f"(x_row_5[26]), "f"(du_row_5[26]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_852;
            float _fma_853;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_853) : "f"(x_row_5[27]), "f"(du_row_5[27]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_853;
            {
                uint2 _vld_854;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_854.x), "=r"(_vld_854.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_854 = reinterpret_cast<uint32_t*>(&_vld_854);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_854[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_854[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_855;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_855.x), "=r"(_vld_855.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_855 = reinterpret_cast<uint32_t*>(&_vld_855);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_855[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_855[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_856;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_856.x), "=r"(_vld_856.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_856 = reinterpret_cast<uint32_t*>(&_vld_856);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_856[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_856[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_857;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_857.x), "=r"(_vld_857.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_857 = reinterpret_cast<uint32_t*>(&_vld_857);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_857[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_857[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_858;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_858.x), "=r"(_vld_858.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_858 = reinterpret_cast<uint32_t*>(&_vld_858);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_858[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_858[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_859;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_859.x), "=r"(_vld_859.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_859 = reinterpret_cast<uint32_t*>(&_vld_859);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_859[_pair]) << 16);
                    (&x_row_5[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_859[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_860;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_860.x), "=r"(_vld_860.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_860 = reinterpret_cast<uint32_t*>(&_vld_860);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_860[_pair]) << 16);
                    (&x_row_5[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_860[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[3] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_861;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_861) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_861;
            float _fma_862;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_862) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_862;
            float _fma_863;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_863) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_863;
            float _fma_864;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_864) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_864;
            float _fma_865;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_865) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_865;
            float _fma_866;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_866) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_866;
            float _fma_867;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_867) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_867;
            float _fma_868;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_868) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_868;
            float _fma_869;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_869) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_869;
            float _fma_870;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_870) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_870;
            float _fma_871;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_871) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_871;
            float _fma_872;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_872) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_872;
            float _fma_873;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_873) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_873;
            float _fma_874;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_874) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_874;
            float _fma_875;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_875) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_875;
            float _fma_876;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_876) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_876;
            float _fma_877;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_877) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_877;
            float _fma_878;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_878) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_878;
            float _fma_879;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_879) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_879;
            float _fma_880;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_880) : "f"(x_row_5[20]), "f"(du_row_5[20]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_880;
            float _fma_881;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_881) : "f"(x_row_5[21]), "f"(du_row_5[21]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_881;
            float _fma_882;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_882) : "f"(x_row_5[22]), "f"(du_row_5[22]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_882;
            float _fma_883;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_883) : "f"(x_row_5[23]), "f"(du_row_5[23]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_883;
            float _fma_884;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_884) : "f"(x_row_5[24]), "f"(du_row_5[24]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_884;
            float _fma_885;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_885) : "f"(x_row_5[25]), "f"(du_row_5[25]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_885;
            float _fma_886;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_886) : "f"(x_row_5[26]), "f"(du_row_5[26]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_886;
            float _fma_887;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_887) : "f"(x_row_5[27]), "f"(du_row_5[27]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_887;
            int upper_6 = lane_1 & 16;
            float send_6 = ((upper_6 != 0) ? dots_p_5[0] : dots_p_5[2]);
            float keep_7 = ((upper_6 != 0) ? dots_p_5[2] : dots_p_5[0]);
            float _shfl_xor_sync_888 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
            float add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(keep_7), "f"(_shfl_xor_sync_888));
            float send_0_5 = ((upper_6 != 0) ? dots_p_5[1] : dots_p_5[3]);
            float keep_1_5 = ((upper_6 != 0) ? dots_p_5[3] : dots_p_5[1]);
            float _shfl_xor_sync_889 = __shfl_xor_sync(0xFFFFFFFF, send_0_5, 16);
            float add_1_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_5) : "f"(keep_1_5), "f"(_shfl_xor_sync_889));
            int upper_2_5 = lane_1 & 8;
            float send_3_5 = ((upper_2_5 != 0) ? add_10 : add_1_5);
            float keep_4_5 = ((upper_2_5 != 0) ? add_1_5 : add_10);
            float _shfl_xor_sync_890 = __shfl_xor_sync(0xFFFFFFFF, send_3_5, 8);
            float add_2_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_5) : "f"(keep_4_5), "f"(_shfl_xor_sync_890));
            float _shfl_xor_sync_891 = __shfl_xor_sync(0xFFFFFFFF, add_2_5, 4);
            float add_3_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_5) : "f"(add_2_5), "f"(_shfl_xor_sync_891));
            float _shfl_xor_sync_892 = __shfl_xor_sync(0xFFFFFFFF, add_3_5, 2);
            float add_4_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_5) : "f"(add_3_5), "f"(_shfl_xor_sync_892));
            float _shfl_xor_sync_893 = __shfl_xor_sync(0xFFFFFFFF, add_4_5, 1);
            float add_5_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_5) : "f"(add_4_5), "f"(_shfl_xor_sync_893));
            if (lane8 == 0) {
                dpre_part[160 + warp_0 * 4 + lane_hi] = add_5_5;
            }
        }
        int tok_a_6 = g0 + 6;
        if (tok_a_6 < t1) {
            long long ta64_6 = (long long)tok_a_6;
            float du_row_6[28];
            {
                uint2 _vld_894;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_894.x), "=r"(_vld_894.y) : "l"((const void*)(dU + (ta64_6 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_894 = reinterpret_cast<uint32_t*>(&_vld_894);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_894[_pair]) << 16);
                    (&du_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_894[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_895;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_895.x), "=r"(_vld_895.y) : "l"((const void*)(dU + (ta64_6 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_895 = reinterpret_cast<uint32_t*>(&_vld_895);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_895[_pair]) << 16);
                    (&du_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_895[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_896;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_896.x), "=r"(_vld_896.y) : "l"((const void*)(dU + (ta64_6 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_896 = reinterpret_cast<uint32_t*>(&_vld_896);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_896[_pair]) << 16);
                    (&du_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_896[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_897;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_897.x), "=r"(_vld_897.y) : "l"((const void*)(dU + (ta64_6 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_897 = reinterpret_cast<uint32_t*>(&_vld_897);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_897[_pair]) << 16);
                    (&du_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_897[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_898;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_898.x), "=r"(_vld_898.y) : "l"((const void*)(dU + (ta64_6 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_898 = reinterpret_cast<uint32_t*>(&_vld_898);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_898[_pair]) << 16);
                    (&du_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_898[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_899;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_899.x), "=r"(_vld_899.y) : "l"((const void*)(dU + (ta64_6 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_899 = reinterpret_cast<uint32_t*>(&_vld_899);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_899[_pair]) << 16);
                    (&du_row_6[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_899[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_900;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_900.x), "=r"(_vld_900.y) : "l"((const void*)(dU + (ta64_6 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_900 = reinterpret_cast<uint32_t*>(&_vld_900);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_900[_pair]) << 16);
                    (&du_row_6[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_900[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_6[4];
            float x_row_6[28];
            {
                uint2 _vld_901;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_901.x), "=r"(_vld_901.y) : "l"((const void*)(X + (ta64_6 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_901 = reinterpret_cast<uint32_t*>(&_vld_901);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_901[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_901[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_902;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_902.x), "=r"(_vld_902.y) : "l"((const void*)(X + (ta64_6 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_902 = reinterpret_cast<uint32_t*>(&_vld_902);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_902[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_902[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_903;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_903.x), "=r"(_vld_903.y) : "l"((const void*)(X + (ta64_6 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_903 = reinterpret_cast<uint32_t*>(&_vld_903);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_903[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_903[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_904;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_904.x), "=r"(_vld_904.y) : "l"((const void*)(X + (ta64_6 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_904 = reinterpret_cast<uint32_t*>(&_vld_904);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_904[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_904[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_905;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_905.x), "=r"(_vld_905.y) : "l"((const void*)(X + (ta64_6 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_905 = reinterpret_cast<uint32_t*>(&_vld_905);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_905[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_905[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_906;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_906.x), "=r"(_vld_906.y) : "l"((const void*)(X + (ta64_6 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_906 = reinterpret_cast<uint32_t*>(&_vld_906);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_906[_pair]) << 16);
                    (&x_row_6[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_906[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_907;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_907.x), "=r"(_vld_907.y) : "l"((const void*)(X + (ta64_6 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_907 = reinterpret_cast<uint32_t*>(&_vld_907);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_907[_pair]) << 16);
                    (&x_row_6[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_907[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[0] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_908;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_908) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_908;
            float _fma_909;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_909) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_909;
            float _fma_910;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_910) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_910;
            float _fma_911;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_911) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_911;
            float _fma_912;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_912) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_912;
            float _fma_913;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_913) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_913;
            float _fma_914;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_914) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_914;
            float _fma_915;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_915) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_915;
            float _fma_916;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_916) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_916;
            float _fma_917;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_917) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_917;
            float _fma_918;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_918) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_918;
            float _fma_919;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_919) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_919;
            float _fma_920;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_920) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_920;
            float _fma_921;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_921) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_921;
            float _fma_922;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_922) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_922;
            float _fma_923;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_923) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_923;
            float _fma_924;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_924) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_924;
            float _fma_925;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_925) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_925;
            float _fma_926;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_926) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_926;
            float _fma_927;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_927) : "f"(x_row_6[20]), "f"(du_row_6[20]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_927;
            float _fma_928;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_928) : "f"(x_row_6[21]), "f"(du_row_6[21]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_928;
            float _fma_929;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_929) : "f"(x_row_6[22]), "f"(du_row_6[22]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_929;
            float _fma_930;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_930) : "f"(x_row_6[23]), "f"(du_row_6[23]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_930;
            float _fma_931;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_931) : "f"(x_row_6[24]), "f"(du_row_6[24]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_931;
            float _fma_932;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_932) : "f"(x_row_6[25]), "f"(du_row_6[25]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_932;
            float _fma_933;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_933) : "f"(x_row_6[26]), "f"(du_row_6[26]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_933;
            float _fma_934;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_934) : "f"(x_row_6[27]), "f"(du_row_6[27]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_934;
            {
                uint2 _vld_935;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_935.x), "=r"(_vld_935.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_935 = reinterpret_cast<uint32_t*>(&_vld_935);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_935[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_935[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_936;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_936.x), "=r"(_vld_936.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_936 = reinterpret_cast<uint32_t*>(&_vld_936);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_936[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_936[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_937;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_937.x), "=r"(_vld_937.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_937 = reinterpret_cast<uint32_t*>(&_vld_937);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_937[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_937[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_938;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_938.x), "=r"(_vld_938.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_938 = reinterpret_cast<uint32_t*>(&_vld_938);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_938[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_938[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_939;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_939.x), "=r"(_vld_939.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_939 = reinterpret_cast<uint32_t*>(&_vld_939);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_939[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_939[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_940;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_940.x), "=r"(_vld_940.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_940 = reinterpret_cast<uint32_t*>(&_vld_940);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_940[_pair]) << 16);
                    (&x_row_6[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_940[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_941;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_941.x), "=r"(_vld_941.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_941 = reinterpret_cast<uint32_t*>(&_vld_941);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_941[_pair]) << 16);
                    (&x_row_6[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_941[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[1] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_942;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_942) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_942;
            float _fma_943;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_943) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_943;
            float _fma_944;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_944) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_944;
            float _fma_945;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_945) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_945;
            float _fma_946;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_946) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_946;
            float _fma_947;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_947) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_947;
            float _fma_948;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_948) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_948;
            float _fma_949;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_949) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_949;
            float _fma_950;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_950) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_950;
            float _fma_951;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_951) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_951;
            float _fma_952;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_952) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_952;
            float _fma_953;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_953) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_953;
            float _fma_954;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_954) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_954;
            float _fma_955;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_955) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_955;
            float _fma_956;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_956) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_956;
            float _fma_957;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_957) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_957;
            float _fma_958;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_958) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_958;
            float _fma_959;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_959) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_959;
            float _fma_960;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_960) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_960;
            float _fma_961;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_961) : "f"(x_row_6[20]), "f"(du_row_6[20]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_961;
            float _fma_962;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_962) : "f"(x_row_6[21]), "f"(du_row_6[21]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_962;
            float _fma_963;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_963) : "f"(x_row_6[22]), "f"(du_row_6[22]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_963;
            float _fma_964;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_964) : "f"(x_row_6[23]), "f"(du_row_6[23]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_964;
            float _fma_965;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_965) : "f"(x_row_6[24]), "f"(du_row_6[24]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_965;
            float _fma_966;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_966) : "f"(x_row_6[25]), "f"(du_row_6[25]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_966;
            float _fma_967;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_967) : "f"(x_row_6[26]), "f"(du_row_6[26]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_967;
            float _fma_968;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_968) : "f"(x_row_6[27]), "f"(du_row_6[27]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_968;
            {
                uint2 _vld_969;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_969.x), "=r"(_vld_969.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_969 = reinterpret_cast<uint32_t*>(&_vld_969);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_969[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_969[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_970;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_970.x), "=r"(_vld_970.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_970 = reinterpret_cast<uint32_t*>(&_vld_970);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_970[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_970[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_971;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_971.x), "=r"(_vld_971.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_971 = reinterpret_cast<uint32_t*>(&_vld_971);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_971[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_971[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_972;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_972.x), "=r"(_vld_972.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_972 = reinterpret_cast<uint32_t*>(&_vld_972);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_972[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_972[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_973;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_973.x), "=r"(_vld_973.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_973 = reinterpret_cast<uint32_t*>(&_vld_973);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_973[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_973[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_974;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_974.x), "=r"(_vld_974.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_974 = reinterpret_cast<uint32_t*>(&_vld_974);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_974[_pair]) << 16);
                    (&x_row_6[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_974[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_975;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_975.x), "=r"(_vld_975.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_975 = reinterpret_cast<uint32_t*>(&_vld_975);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_975[_pair]) << 16);
                    (&x_row_6[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_975[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[2] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_976;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_976) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_976;
            float _fma_977;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_977) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_977;
            float _fma_978;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_978) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_978;
            float _fma_979;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_979) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_979;
            float _fma_980;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_980) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_980;
            float _fma_981;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_981) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_981;
            float _fma_982;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_982) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_982;
            float _fma_983;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_983) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_983;
            float _fma_984;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_984) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_984;
            float _fma_985;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_985) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_985;
            float _fma_986;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_986) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_986;
            float _fma_987;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_987) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_987;
            float _fma_988;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_988) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_988;
            float _fma_989;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_989) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_989;
            float _fma_990;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_990) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_990;
            float _fma_991;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_991) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_991;
            float _fma_992;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_992) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_992;
            float _fma_993;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_993) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_993;
            float _fma_994;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_994) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_994;
            float _fma_995;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_995) : "f"(x_row_6[20]), "f"(du_row_6[20]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_995;
            float _fma_996;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_996) : "f"(x_row_6[21]), "f"(du_row_6[21]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_996;
            float _fma_997;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_997) : "f"(x_row_6[22]), "f"(du_row_6[22]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_997;
            float _fma_998;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_998) : "f"(x_row_6[23]), "f"(du_row_6[23]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_998;
            float _fma_999;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_999) : "f"(x_row_6[24]), "f"(du_row_6[24]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_999;
            float _fma_1000;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1000) : "f"(x_row_6[25]), "f"(du_row_6[25]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_1000;
            float _fma_1001;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1001) : "f"(x_row_6[26]), "f"(du_row_6[26]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_1001;
            float _fma_1002;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1002) : "f"(x_row_6[27]), "f"(du_row_6[27]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_1002;
            {
                uint2 _vld_1003;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1003.x), "=r"(_vld_1003.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1003 = reinterpret_cast<uint32_t*>(&_vld_1003);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1003[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1003[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1004;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1004.x), "=r"(_vld_1004.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1004 = reinterpret_cast<uint32_t*>(&_vld_1004);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1004[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1004[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1005;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1005.x), "=r"(_vld_1005.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1005 = reinterpret_cast<uint32_t*>(&_vld_1005);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1005[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1005[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1006;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1006.x), "=r"(_vld_1006.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1006 = reinterpret_cast<uint32_t*>(&_vld_1006);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1006[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1006[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1007;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1007.x), "=r"(_vld_1007.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1007 = reinterpret_cast<uint32_t*>(&_vld_1007);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1007[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1007[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1008;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1008.x), "=r"(_vld_1008.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1008 = reinterpret_cast<uint32_t*>(&_vld_1008);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1008[_pair]) << 16);
                    (&x_row_6[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1008[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1009;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1009.x), "=r"(_vld_1009.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1009 = reinterpret_cast<uint32_t*>(&_vld_1009);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1009[_pair]) << 16);
                    (&x_row_6[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1009[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[3] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_1010;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1010) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1010;
            float _fma_1011;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1011) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1011;
            float _fma_1012;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1012) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1012;
            float _fma_1013;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1013) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1013;
            float _fma_1014;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1014) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1014;
            float _fma_1015;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1015) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1015;
            float _fma_1016;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1016) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1016;
            float _fma_1017;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1017) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1017;
            float _fma_1018;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1018) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1018;
            float _fma_1019;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1019) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1019;
            float _fma_1020;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1020) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1020;
            float _fma_1021;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1021) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1021;
            float _fma_1022;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1022) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1022;
            float _fma_1023;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1023) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1023;
            float _fma_1024;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1024) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1024;
            float _fma_1025;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1025) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1025;
            float _fma_1026;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1026) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1026;
            float _fma_1027;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1027) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1027;
            float _fma_1028;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1028) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1028;
            float _fma_1029;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1029) : "f"(x_row_6[20]), "f"(du_row_6[20]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1029;
            float _fma_1030;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1030) : "f"(x_row_6[21]), "f"(du_row_6[21]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1030;
            float _fma_1031;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1031) : "f"(x_row_6[22]), "f"(du_row_6[22]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1031;
            float _fma_1032;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1032) : "f"(x_row_6[23]), "f"(du_row_6[23]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1032;
            float _fma_1033;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1033) : "f"(x_row_6[24]), "f"(du_row_6[24]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1033;
            float _fma_1034;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1034) : "f"(x_row_6[25]), "f"(du_row_6[25]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1034;
            float _fma_1035;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1035) : "f"(x_row_6[26]), "f"(du_row_6[26]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1035;
            float _fma_1036;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1036) : "f"(x_row_6[27]), "f"(du_row_6[27]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_1036;
            int upper_7 = lane_1 & 16;
            float send_7 = ((upper_7 != 0) ? dots_p_6[0] : dots_p_6[2]);
            float keep_8 = ((upper_7 != 0) ? dots_p_6[2] : dots_p_6[0]);
            float _shfl_xor_sync_1037 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
            float add_11;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(keep_8), "f"(_shfl_xor_sync_1037));
            float send_0_6 = ((upper_7 != 0) ? dots_p_6[1] : dots_p_6[3]);
            float keep_1_6 = ((upper_7 != 0) ? dots_p_6[3] : dots_p_6[1]);
            float _shfl_xor_sync_1038 = __shfl_xor_sync(0xFFFFFFFF, send_0_6, 16);
            float add_1_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_6) : "f"(keep_1_6), "f"(_shfl_xor_sync_1038));
            int upper_2_6 = lane_1 & 8;
            float send_3_6 = ((upper_2_6 != 0) ? add_11 : add_1_6);
            float keep_4_6 = ((upper_2_6 != 0) ? add_1_6 : add_11);
            float _shfl_xor_sync_1039 = __shfl_xor_sync(0xFFFFFFFF, send_3_6, 8);
            float add_2_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_6) : "f"(keep_4_6), "f"(_shfl_xor_sync_1039));
            float _shfl_xor_sync_1040 = __shfl_xor_sync(0xFFFFFFFF, add_2_6, 4);
            float add_3_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_6) : "f"(add_2_6), "f"(_shfl_xor_sync_1040));
            float _shfl_xor_sync_1041 = __shfl_xor_sync(0xFFFFFFFF, add_3_6, 2);
            float add_4_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_6) : "f"(add_3_6), "f"(_shfl_xor_sync_1041));
            float _shfl_xor_sync_1042 = __shfl_xor_sync(0xFFFFFFFF, add_4_6, 1);
            float add_5_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_6) : "f"(add_4_6), "f"(_shfl_xor_sync_1042));
            if (lane8 == 0) {
                dpre_part[192 + warp_0 * 4 + lane_hi] = add_5_6;
            }
        }
        int tok_a_7 = g0 + 7;
        if (tok_a_7 < t1) {
            long long ta64_7 = (long long)tok_a_7;
            float du_row_7[28];
            {
                uint2 _vld_1043;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1043.x), "=r"(_vld_1043.y) : "l"((const void*)(dU + (ta64_7 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1043 = reinterpret_cast<uint32_t*>(&_vld_1043);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1043[_pair]) << 16);
                    (&du_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1043[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1044;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1044.x), "=r"(_vld_1044.y) : "l"((const void*)(dU + (ta64_7 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1044 = reinterpret_cast<uint32_t*>(&_vld_1044);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1044[_pair]) << 16);
                    (&du_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1044[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1045;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1045.x), "=r"(_vld_1045.y) : "l"((const void*)(dU + (ta64_7 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1045 = reinterpret_cast<uint32_t*>(&_vld_1045);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1045[_pair]) << 16);
                    (&du_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1045[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1046;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1046.x), "=r"(_vld_1046.y) : "l"((const void*)(dU + (ta64_7 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1046 = reinterpret_cast<uint32_t*>(&_vld_1046);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1046[_pair]) << 16);
                    (&du_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1046[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1047;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1047.x), "=r"(_vld_1047.y) : "l"((const void*)(dU + (ta64_7 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1047 = reinterpret_cast<uint32_t*>(&_vld_1047);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1047[_pair]) << 16);
                    (&du_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1047[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1048;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1048.x), "=r"(_vld_1048.y) : "l"((const void*)(dU + (ta64_7 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1048 = reinterpret_cast<uint32_t*>(&_vld_1048);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1048[_pair]) << 16);
                    (&du_row_7[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1048[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1049;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1049.x), "=r"(_vld_1049.y) : "l"((const void*)(dU + (ta64_7 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1049 = reinterpret_cast<uint32_t*>(&_vld_1049);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1049[_pair]) << 16);
                    (&du_row_7[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1049[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_7[4];
            float x_row_7[28];
            {
                uint2 _vld_1050;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1050.x), "=r"(_vld_1050.y) : "l"((const void*)(X + (ta64_7 * 4 * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1050 = reinterpret_cast<uint32_t*>(&_vld_1050);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1050[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1050[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1051;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1051.x), "=r"(_vld_1051.y) : "l"((const void*)(X + (ta64_7 * 4 * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1051 = reinterpret_cast<uint32_t*>(&_vld_1051);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1051[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1051[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1052;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1052.x), "=r"(_vld_1052.y) : "l"((const void*)(X + (ta64_7 * 4 * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1052 = reinterpret_cast<uint32_t*>(&_vld_1052);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1052[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1052[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1053;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1053.x), "=r"(_vld_1053.y) : "l"((const void*)(X + (ta64_7 * 4 * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1053 = reinterpret_cast<uint32_t*>(&_vld_1053);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1053[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1053[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1054;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1054.x), "=r"(_vld_1054.y) : "l"((const void*)(X + (ta64_7 * 4 * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1054 = reinterpret_cast<uint32_t*>(&_vld_1054);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1054[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1054[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1055;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1055.x), "=r"(_vld_1055.y) : "l"((const void*)(X + (ta64_7 * 4 * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1055 = reinterpret_cast<uint32_t*>(&_vld_1055);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1055[_pair]) << 16);
                    (&x_row_7[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1055[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1056;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1056.x), "=r"(_vld_1056.y) : "l"((const void*)(X + (ta64_7 * 4 * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1056 = reinterpret_cast<uint32_t*>(&_vld_1056);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1056[_pair]) << 16);
                    (&x_row_7[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1056[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[0] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_1057;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1057) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1057;
            float _fma_1058;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1058) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1058;
            float _fma_1059;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1059) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1059;
            float _fma_1060;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1060) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1060;
            float _fma_1061;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1061) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1061;
            float _fma_1062;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1062) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1062;
            float _fma_1063;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1063) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1063;
            float _fma_1064;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1064) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1064;
            float _fma_1065;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1065) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1065;
            float _fma_1066;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1066) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1066;
            float _fma_1067;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1067) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1067;
            float _fma_1068;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1068) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1068;
            float _fma_1069;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1069) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1069;
            float _fma_1070;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1070) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1070;
            float _fma_1071;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1071) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1071;
            float _fma_1072;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1072) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1072;
            float _fma_1073;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1073) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1073;
            float _fma_1074;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1074) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1074;
            float _fma_1075;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1075) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1075;
            float _fma_1076;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1076) : "f"(x_row_7[20]), "f"(du_row_7[20]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1076;
            float _fma_1077;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1077) : "f"(x_row_7[21]), "f"(du_row_7[21]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1077;
            float _fma_1078;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1078) : "f"(x_row_7[22]), "f"(du_row_7[22]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1078;
            float _fma_1079;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1079) : "f"(x_row_7[23]), "f"(du_row_7[23]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1079;
            float _fma_1080;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1080) : "f"(x_row_7[24]), "f"(du_row_7[24]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1080;
            float _fma_1081;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1081) : "f"(x_row_7[25]), "f"(du_row_7[25]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1081;
            float _fma_1082;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1082) : "f"(x_row_7[26]), "f"(du_row_7[26]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1082;
            float _fma_1083;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1083) : "f"(x_row_7[27]), "f"(du_row_7[27]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_1083;
            {
                uint2 _vld_1084;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1084.x), "=r"(_vld_1084.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1084 = reinterpret_cast<uint32_t*>(&_vld_1084);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1084[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1084[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1085;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1085.x), "=r"(_vld_1085.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1085 = reinterpret_cast<uint32_t*>(&_vld_1085);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1085[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1085[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1086;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1086.x), "=r"(_vld_1086.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1086 = reinterpret_cast<uint32_t*>(&_vld_1086);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1086[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1086[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1087;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1087.x), "=r"(_vld_1087.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1087 = reinterpret_cast<uint32_t*>(&_vld_1087);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1087[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1087[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1088;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1088.x), "=r"(_vld_1088.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1088 = reinterpret_cast<uint32_t*>(&_vld_1088);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1088[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1088[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1089;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1089.x), "=r"(_vld_1089.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1089 = reinterpret_cast<uint32_t*>(&_vld_1089);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1089[_pair]) << 16);
                    (&x_row_7[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1089[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1090;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1090.x), "=r"(_vld_1090.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1090 = reinterpret_cast<uint32_t*>(&_vld_1090);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1090[_pair]) << 16);
                    (&x_row_7[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1090[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[1] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_1091;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1091) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1091;
            float _fma_1092;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1092) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1092;
            float _fma_1093;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1093) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1093;
            float _fma_1094;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1094) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1094;
            float _fma_1095;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1095) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1095;
            float _fma_1096;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1096) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1096;
            float _fma_1097;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1097) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1097;
            float _fma_1098;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1098) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1098;
            float _fma_1099;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1099) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1099;
            float _fma_1100;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1100) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1100;
            float _fma_1101;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1101) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1101;
            float _fma_1102;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1102) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1102;
            float _fma_1103;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1103) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1103;
            float _fma_1104;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1104) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1104;
            float _fma_1105;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1105) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1105;
            float _fma_1106;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1106) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1106;
            float _fma_1107;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1107) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1107;
            float _fma_1108;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1108) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1108;
            float _fma_1109;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1109) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1109;
            float _fma_1110;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1110) : "f"(x_row_7[20]), "f"(du_row_7[20]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1110;
            float _fma_1111;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1111) : "f"(x_row_7[21]), "f"(du_row_7[21]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1111;
            float _fma_1112;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1112) : "f"(x_row_7[22]), "f"(du_row_7[22]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1112;
            float _fma_1113;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1113) : "f"(x_row_7[23]), "f"(du_row_7[23]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1113;
            float _fma_1114;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1114) : "f"(x_row_7[24]), "f"(du_row_7[24]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1114;
            float _fma_1115;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1115) : "f"(x_row_7[25]), "f"(du_row_7[25]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1115;
            float _fma_1116;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1116) : "f"(x_row_7[26]), "f"(du_row_7[26]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1116;
            float _fma_1117;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1117) : "f"(x_row_7[27]), "f"(du_row_7[27]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_1117;
            {
                uint2 _vld_1118;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1118.x), "=r"(_vld_1118.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1118 = reinterpret_cast<uint32_t*>(&_vld_1118);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1118[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1118[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1119;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1119.x), "=r"(_vld_1119.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1119 = reinterpret_cast<uint32_t*>(&_vld_1119);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1119[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1119[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1120;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1120.x), "=r"(_vld_1120.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1120 = reinterpret_cast<uint32_t*>(&_vld_1120);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1120[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1120[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1121;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1121.x), "=r"(_vld_1121.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1121 = reinterpret_cast<uint32_t*>(&_vld_1121);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1121[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1121[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1122;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1122.x), "=r"(_vld_1122.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1122 = reinterpret_cast<uint32_t*>(&_vld_1122);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1122[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1122[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1123;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1123.x), "=r"(_vld_1123.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1123 = reinterpret_cast<uint32_t*>(&_vld_1123);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1123[_pair]) << 16);
                    (&x_row_7[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1123[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1124;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1124.x), "=r"(_vld_1124.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1124 = reinterpret_cast<uint32_t*>(&_vld_1124);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1124[_pair]) << 16);
                    (&x_row_7[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1124[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[2] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_1125;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1125) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1125;
            float _fma_1126;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1126) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1126;
            float _fma_1127;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1127) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1127;
            float _fma_1128;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1128) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1128;
            float _fma_1129;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1129) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1129;
            float _fma_1130;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1130) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1130;
            float _fma_1131;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1131) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1131;
            float _fma_1132;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1132) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1132;
            float _fma_1133;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1133) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1133;
            float _fma_1134;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1134) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1134;
            float _fma_1135;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1135) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1135;
            float _fma_1136;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1136) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1136;
            float _fma_1137;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1137) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1137;
            float _fma_1138;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1138) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1138;
            float _fma_1139;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1139) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1139;
            float _fma_1140;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1140) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1140;
            float _fma_1141;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1141) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1141;
            float _fma_1142;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1142) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1142;
            float _fma_1143;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1143) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1143;
            float _fma_1144;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1144) : "f"(x_row_7[20]), "f"(du_row_7[20]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1144;
            float _fma_1145;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1145) : "f"(x_row_7[21]), "f"(du_row_7[21]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1145;
            float _fma_1146;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1146) : "f"(x_row_7[22]), "f"(du_row_7[22]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1146;
            float _fma_1147;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1147) : "f"(x_row_7[23]), "f"(du_row_7[23]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1147;
            float _fma_1148;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1148) : "f"(x_row_7[24]), "f"(du_row_7[24]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1148;
            float _fma_1149;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1149) : "f"(x_row_7[25]), "f"(du_row_7[25]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1149;
            float _fma_1150;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1150) : "f"(x_row_7[26]), "f"(du_row_7[26]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1150;
            float _fma_1151;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1151) : "f"(x_row_7[27]), "f"(du_row_7[27]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_1151;
            {
                uint2 _vld_1152;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1152.x), "=r"(_vld_1152.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 7168 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1152 = reinterpret_cast<uint32_t*>(&_vld_1152);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1152[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1152[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1153;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1153.x), "=r"(_vld_1153.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 7168 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1153 = reinterpret_cast<uint32_t*>(&_vld_1153);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1153[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1153[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1154;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1154.x), "=r"(_vld_1154.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 7168 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1154 = reinterpret_cast<uint32_t*>(&_vld_1154);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1154[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1154[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1155;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1155.x), "=r"(_vld_1155.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 7168 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1155 = reinterpret_cast<uint32_t*>(&_vld_1155);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1155[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1155[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1156;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1156.x), "=r"(_vld_1156.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 7168 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1156 = reinterpret_cast<uint32_t*>(&_vld_1156);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1156[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1156[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1157;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1157.x), "=r"(_vld_1157.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 7168 + col_off + 20) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1157 = reinterpret_cast<uint32_t*>(&_vld_1157);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[20 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1157[_pair]) << 16);
                    (&x_row_7[20 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1157[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1158;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_1158.x), "=r"(_vld_1158.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 7168 + col_off + 24) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_1158 = reinterpret_cast<uint32_t*>(&_vld_1158);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[24 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1158[_pair]) << 16);
                    (&x_row_7[24 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1158[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[3] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_1159;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1159) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1159;
            float _fma_1160;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1160) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1160;
            float _fma_1161;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1161) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1161;
            float _fma_1162;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1162) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1162;
            float _fma_1163;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1163) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1163;
            float _fma_1164;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1164) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1164;
            float _fma_1165;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1165) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1165;
            float _fma_1166;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1166) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1166;
            float _fma_1167;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1167) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1167;
            float _fma_1168;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1168) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1168;
            float _fma_1169;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1169) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1169;
            float _fma_1170;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1170) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1170;
            float _fma_1171;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1171) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1171;
            float _fma_1172;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1172) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1172;
            float _fma_1173;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1173) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1173;
            float _fma_1174;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1174) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1174;
            float _fma_1175;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1175) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1175;
            float _fma_1176;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1176) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1176;
            float _fma_1177;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1177) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1177;
            float _fma_1178;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1178) : "f"(x_row_7[20]), "f"(du_row_7[20]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1178;
            float _fma_1179;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1179) : "f"(x_row_7[21]), "f"(du_row_7[21]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1179;
            float _fma_1180;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1180) : "f"(x_row_7[22]), "f"(du_row_7[22]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1180;
            float _fma_1181;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1181) : "f"(x_row_7[23]), "f"(du_row_7[23]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1181;
            float _fma_1182;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1182) : "f"(x_row_7[24]), "f"(du_row_7[24]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1182;
            float _fma_1183;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1183) : "f"(x_row_7[25]), "f"(du_row_7[25]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1183;
            float _fma_1184;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1184) : "f"(x_row_7[26]), "f"(du_row_7[26]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1184;
            float _fma_1185;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1185) : "f"(x_row_7[27]), "f"(du_row_7[27]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_1185;
            int upper_8 = lane_1 & 16;
            float send_8 = ((upper_8 != 0) ? dots_p_7[0] : dots_p_7[2]);
            float keep_9 = ((upper_8 != 0) ? dots_p_7[2] : dots_p_7[0]);
            float _shfl_xor_sync_1186 = __shfl_xor_sync(0xFFFFFFFF, send_8, 16);
            float add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(keep_9), "f"(_shfl_xor_sync_1186));
            float send_0_7 = ((upper_8 != 0) ? dots_p_7[1] : dots_p_7[3]);
            float keep_1_7 = ((upper_8 != 0) ? dots_p_7[3] : dots_p_7[1]);
            float _shfl_xor_sync_1187 = __shfl_xor_sync(0xFFFFFFFF, send_0_7, 16);
            float add_1_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_7) : "f"(keep_1_7), "f"(_shfl_xor_sync_1187));
            int upper_2_7 = lane_1 & 8;
            float send_3_7 = ((upper_2_7 != 0) ? add_12 : add_1_7);
            float keep_4_7 = ((upper_2_7 != 0) ? add_1_7 : add_12);
            float _shfl_xor_sync_1188 = __shfl_xor_sync(0xFFFFFFFF, send_3_7, 8);
            float add_2_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_7) : "f"(keep_4_7), "f"(_shfl_xor_sync_1188));
            float _shfl_xor_sync_1189 = __shfl_xor_sync(0xFFFFFFFF, add_2_7, 4);
            float add_3_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_7) : "f"(add_2_7), "f"(_shfl_xor_sync_1189));
            float _shfl_xor_sync_1190 = __shfl_xor_sync(0xFFFFFFFF, add_3_7, 2);
            float add_4_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_7) : "f"(add_3_7), "f"(_shfl_xor_sync_1190));
            float _shfl_xor_sync_1191 = __shfl_xor_sync(0xFFFFFFFF, add_4_7, 1);
            float add_5_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_7) : "f"(add_4_7), "f"(_shfl_xor_sync_1191));
            if (lane8 == 0) {
                dpre_part[224 + warp_0 * 4 + lane_hi] = add_5_7;
            }
        }
        __syncthreads();
        int tok = g0 + warp_0;
        int tokc = ((tok < t1) ? tok : t_last);
        long long tok64 = (long long)tokc;
        float dp[1];
        long long srow = tok64 * 64;
        float r = stats[srow];
        float z_load = stats[srow + 1 + n64];
        float gval = gstats[tok64 * 32 + (long long)gcol];
        float z = ((lane_1 < 24) ? z_load : 0.0f);
        float a_map;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a_map) : "f"(z), "f"(sc), "f"(bs));
        float a = ((lane_1 < 24) ? a_map : 0.0f);
        float gtwo;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(gtwo) : "f"(gval), "f"(2.0f));
        float _sub_1192;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1192) : "f"(0.0f), "f"(fabsf(a)));
        float _exp_0 = __nv_expf(_sub_1192);
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp_0));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_0);
        float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_0);
        float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
        float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(sig), "f"(hc_eps));
        float logit = ((is_logit) ? a : 0.0f);
        float _shfl_xor_sync_1193 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
        float _max_0 = fmaxf(logit, _shfl_xor_sync_1193);
        float _shfl_xor_sync_1195 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
        float _sub_1194;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1194) : "f"(logit), "f"(fmaxf(_max_0, _shfl_xor_sync_1195)));
        float _exp_1 = __nv_expf(_sub_1194);
        float _shfl_xor_sync_1196 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(_exp_1), "f"(_shfl_xor_sync_1196));
        float _shfl_xor_sync_1198 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
        float _add_1197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1197) : "f"(add_13), "f"(_shfl_xor_sync_1198));
        float S = __fdiv_rn(_exp_1, _add_1197);
        float C0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(C0) : "f"(S), "f"(hc_eps));
        float _shfl_xor_sync_1199 = __shfl_xor_sync(0xFFFFFFFF, C0, 4);
        float s1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s1) : "f"(C0), "f"(_shfl_xor_sync_1199));
        float _shfl_xor_sync_1202 = __shfl_xor_sync(0xFFFFFFFF, s1, 24);
        float _add_1201;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1201) : "f"(s1), "f"(_shfl_xor_sync_1202));
        float _add_1200;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1200) : "f"(_add_1201), "f"(hc_eps));
        float c = __fdiv_rn(C0, _add_1200);
        if (is_logit) {
            hist_s[hb] = c;
        }
        #pragma unroll 1
        for (int p = 0; p < n_pairs; p++) {
            float _shfl_xor_sync_1203 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
            float sr;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sr) : "f"(c), "f"(_shfl_xor_sync_1203));
            float _shfl_xor_sync_1207 = __shfl_xor_sync(0xFFFFFFFF, sr, 2);
            float _add_1206;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1206) : "f"(sr), "f"(_shfl_xor_sync_1207));
            float _add_1205;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1205) : "f"(_add_1206), "f"(hc_eps));
            float _fdiv_1204 = __fdiv_rn(c, _add_1205);
            c = _fdiv_1204;
            if (is_logit) {
                hist_s[(2 * p + 1) * 128 + hb] = c;
            }
            float _shfl_xor_sync_1208 = __shfl_xor_sync(0xFFFFFFFF, c, 4);
            float sk;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sk) : "f"(c), "f"(_shfl_xor_sync_1208));
            float _shfl_xor_sync_1212 = __shfl_xor_sync(0xFFFFFFFF, sk, 24);
            float _add_1211;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1211) : "f"(sk), "f"(_shfl_xor_sync_1212));
            float _add_1210;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1210) : "f"(_add_1211), "f"(hc_eps));
            float _fdiv_1209 = __fdiv_rn(c, _add_1210);
            c = _fdiv_1209;
            if (is_logit) {
                hist_s[(2 * p + 2) * 128 + hb] = c;
            }
        }
        float G = ((is_logit) ? gval : 0.0f);
        float cur = c;
        #pragma unroll 1
        for (int q = 0; q < n_pairs; q++) {
            int pb = n_pairs - 1 - q;
            float hp = hist_s[(2 * pb + 1) * ld_step + ld_base];
            float _shfl_xor_sync_1213 = __shfl_xor_sync(0xFFFFFFFF, hp, 4);
            float sp;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sp) : "f"(hp), "f"(_shfl_xor_sync_1213));
            float _shfl_xor_sync_1215 = __shfl_xor_sync(0xFFFFFFFF, sp, 24);
            float _add_1214;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1214) : "f"(sp), "f"(_shfl_xor_sync_1215));
            float vp;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vp) : "f"(_add_1214), "f"(hc_eps));
            float mp;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(mp) : "f"(G), "f"(cur));
            float _shfl_xor_sync_1216 = __shfl_xor_sync(0xFFFFFFFF, mp, 4);
            float ipp;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ipp) : "f"(mp), "f"(_shfl_xor_sync_1216));
            float _shfl_xor_sync_1220 = __shfl_xor_sync(0xFFFFFFFF, ipp, 24);
            float _add_1219;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1219) : "f"(ipp), "f"(_shfl_xor_sync_1220));
            float _sub_1218;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1218) : "f"(G), "f"(_add_1219));
            float _fdiv_1217 = __fdiv_rn(_sub_1218, vp);
            G = _fdiv_1217;
            float hq = hist_s[2 * pb * ld_step + ld_base];
            float _shfl_xor_sync_1221 = __shfl_xor_sync(0xFFFFFFFF, hq, 1);
            float sq;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sq) : "f"(hq), "f"(_shfl_xor_sync_1221));
            float _shfl_xor_sync_1223 = __shfl_xor_sync(0xFFFFFFFF, sq, 2);
            float _add_1222;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1222) : "f"(sq), "f"(_shfl_xor_sync_1223));
            float vq;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vq) : "f"(_add_1222), "f"(hc_eps));
            float mq;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(mq) : "f"(G), "f"(hp));
            float _shfl_xor_sync_1224 = __shfl_xor_sync(0xFFFFFFFF, mq, 1);
            float ipq;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ipq) : "f"(mq), "f"(_shfl_xor_sync_1224));
            float _shfl_xor_sync_1228 = __shfl_xor_sync(0xFFFFFFFF, ipq, 2);
            float _add_1227;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1227) : "f"(ipq), "f"(_shfl_xor_sync_1228));
            float _sub_1226;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1226) : "f"(G), "f"(_add_1227));
            float _fdiv_1225 = __fdiv_rn(_sub_1226, vq);
            G = _fdiv_1225;
            cur = hq;
        }
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(S), "f"(hc_eps));
        float _shfl_xor_sync_1229 = __shfl_xor_sync(0xFFFFFFFF, _add_195, 4);
        float add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(_add_195), "f"(_shfl_xor_sync_1229));
        float _shfl_xor_sync_1231 = __shfl_xor_sync(0xFFFFFFFF, add_0, 24);
        float _add_1230;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1230) : "f"(add_0), "f"(_shfl_xor_sync_1231));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_add_1230), "f"(hc_eps));
        float _mul_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(G), "f"(cur));
        float _shfl_xor_sync_1232 = __shfl_xor_sync(0xFFFFFFFF, _mul_39, 4);
        float add_1_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_8) : "f"(_mul_39), "f"(_shfl_xor_sync_1232));
        float _shfl_xor_sync_1235 = __shfl_xor_sync(0xFFFFFFFF, add_1_8, 24);
        float _add_1234;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1234) : "f"(add_1_8), "f"(_shfl_xor_sync_1235));
        float _sub_1233;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1233) : "f"(G), "f"(_add_1234));
        float g1 = __fdiv_rn(_sub_1233, _add_197);
        float acc_162 = 0.0f;
        float _shfl_xor_sync_1236 = __shfl_xor_sync(0xFFFFFFFF, S, 2);
        float _shfl_xor_sync_1238 = __shfl_xor_sync(0xFFFFFFFF, g1, 2);
        float _sub_1237;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1237) : "f"(g1), "f"(_shfl_xor_sync_1238));
        float _shfl_xor_sync_1240 = __shfl_xor_sync(0xFFFFFFFF, S, 1);
        float _shfl_xor_sync_1242 = __shfl_xor_sync(0xFFFFFFFF, g1, 1);
        float _sub_1241;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1241) : "f"(g1), "f"(_shfl_xor_sync_1242));
        float _fma_1239;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1239) : "f"(_shfl_xor_sync_1240), "f"(_sub_1241), "f"(acc_162));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(_shfl_xor_sync_1236), "f"(_sub_1237), "f"(_fma_1239));
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, S, 3);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, g1, 3);
        float dsig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dsig) : "f"(sig), "f"(sneg));
        float _sub_1244;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1244) : "f"(g1), "f"(_shfl_xor_1));
        float _fma_1243;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1243) : "f"(_shfl_xor_0), "f"(_sub_1244), "f"(_fma_0));
        float da = ((is_logit) ? mul_f32_rn_noftz(S, _fma_1243) : mul_f32_rn_noftz(1.0f, dsig));
        dp[0] = dpre_part[warp_0 * 32 + lc4];
        float _add_1245;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1245) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 4 + lc4]));
        dp[0] = _add_1245;
        float _add_1246;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1246) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 8 + lc4]));
        dp[0] = _add_1246;
        float _add_1247;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1247) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 12 + lc4]));
        dp[0] = _add_1247;
        float _add_1248;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1248) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 16 + lc4]));
        dp[0] = _add_1248;
        float _add_1249;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1249) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 20 + lc4]));
        dp[0] = _add_1249;
        float _add_1250;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1250) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 24 + lc4]));
        dp[0] = _add_1250;
        float _add_1251;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1251) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 28 + lc4]));
        dp[0] = _add_1251;
        float gsig = ((lane_1 < 4) ? dp[0] : gtwo);
        float da_sig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(da_sig) : "f"(gsig), "f"(da));
        float da_late = ((lane_1 >= 8 && lane_1 < 24) ? da : da_sig);
        float da_0 = ((lane_1 < 24) ? da_late : 0.0f);
        float dz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dz) : "f"(da_0), "f"(sc));
        float daz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(daz) : "f"(da_0), "f"(z));
        float dzz;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dzz) : "f"(dz), "f"(z));
        float _warp_reduce_0 = dzz;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float rr;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(rr) : "f"(r), "f"(r));
        float c_t = (float)((double)rr * (double)_warp_reduce_0 * 3.487723214285714e-05);
        float dzr;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dzr) : "f"(dz), "f"(r));
        float cvt_f32 = (float)(__nv_bfloat16)dzr;
        float sub;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(dzr), "f"(cvt_f32));
        float cvt_f32_1 = (float)(__nv_bfloat16)sub;
        float _sub_1252;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1252) : "f"(sub), "f"(cvt_f32_1));
        float cvt_f32_2 = (float)(__nv_bfloat16)_sub_1252;
        float p1 = ((lane_1 < 24) ? cvt_f32 : 0.0f);
        float p2 = ((lane_1 < 24) ? cvt_f32_1 : 0.0f);
        float p3 = ((lane_1 < 24) ? cvt_f32_2 : 0.0f);
        long long drow = tok64 * 96 + lane64;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p1)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 32])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p2)), "r"((unsigned int)(tok < t1)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&dzp[drow + 64])), "h"(__bfloat16_as_ushort((__nv_bfloat16)p3)), "r"((unsigned int)(tok < t1)) : "memory");
        long long trow = tok64 * 24;
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow])), "f"(r), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 1])), "f"(c_t), "r"((unsigned int)(tok < t1 && lane_1 == 0)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + 2 + lane64])), "f"(_add_1), "r"((unsigned int)(tok < t1 && lane_1 < 4)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(c), "r"((unsigned int)(tok < t1 && is_logit)) : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&tokp[trow + lane64 - 2])), "f"(0.0f), "r"((unsigned int)(tok < t1 && (lane_1 >= 24 && lane_1 < 26))) : "memory");
        float _sub_0;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(da_0), "f"(comp[0]));
        float _add_2;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(acc[0]), "f"(_sub_0));
        float _sub_1254;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1254) : "f"(_add_2), "f"(acc[0]));
        float _sub_1253;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1253) : "f"(_sub_1254), "f"(_sub_0));
        comp[0] = ((tok < t1) ? _sub_1253 : comp[0]);
        acc[0] = ((tok < t1) ? _add_2 : acc[0]);
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(daz), "f"(comp[1]));
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(acc[1]), "f"(_sub_1));
        float _sub_1256;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1256) : "f"(_add_3), "f"(acc[1]));
        float _sub_1255;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1255) : "f"(_sub_1256), "f"(_sub_1));
        comp[1] = ((tok < t1) ? _sub_1255 : comp[1]);
        acc[1] = ((tok < t1) ? _add_3 : acc[1]);
        __syncthreads();
    }
    int fbase = (warp_0 * 32 + lane_1) * 4;
    fold[fbase] = acc[0];
    fold[fbase + 1] = comp[0];
    fold[fbase + 2] = acc[1];
    fold[fbase + 3] = comp[1];
    __syncthreads();
    if (warp_0 == 0) {
        long long prow = (long long)bid * 32;
        float o[2];
        float oc[2];
        #pragma unroll
        for (int e_1 = 0; e_1 < 2; e_1++) {
            o[e_1] = 0.0f;
            oc[e_1] = 0.0f;
        }
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            int src = (w * 32 + lane_1) * 4;
            float sub_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(fold[src]), "f"(oc[0]));
            float add_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_14) : "f"(o[0]), "f"(sub_1));
            float _sub_1258;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1258) : "f"(add_14), "f"(o[0]));
            float _sub_1257;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1257) : "f"(_sub_1258), "f"(sub_1));
            oc[0] = _sub_1257;
            o[0] = add_14;
            float _sub_1259;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1259) : "f"(0.0f), "f"(fold[src + 1]));
            float sub_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_0) : "f"(_sub_1259), "f"(oc[0]));
            float add_2_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_8) : "f"(o[0]), "f"(sub_0));
            float _sub_1261;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1261) : "f"(add_2_8), "f"(o[0]));
            float _sub_1260;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1260) : "f"(_sub_1261), "f"(sub_0));
            oc[0] = _sub_1260;
            o[0] = add_2_8;
            float sub_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1_1) : "f"(fold[src + 2]), "f"(oc[1]));
            float add_3_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_8) : "f"(o[1]), "f"(sub_1_1));
            float _sub_1263;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1263) : "f"(add_3_8), "f"(o[1]));
            float _sub_1262;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1262) : "f"(_sub_1263), "f"(sub_1_1));
            oc[1] = _sub_1262;
            o[1] = add_3_8;
            float _sub_1264;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1264) : "f"(0.0f), "f"(fold[src + 3]));
            float sub_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(_sub_1264), "f"(oc[1]));
            float add_4_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_8) : "f"(o[1]), "f"(sub_2));
            float _sub_1266;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1266) : "f"(add_4_8), "f"(o[1]));
            float _sub_1265;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1265) : "f"(_sub_1266), "f"(sub_2));
            oc[1] = _sub_1265;
            o[1] = add_4_8;
        }
        float dbase_n;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(dbase_n) : "f"(o[0]), "f"(oc[0]));
        float dsz_n;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(dsz_n) : "f"(o[1]), "f"(oc[1]));
        gsum[lane_1] = dsz_n;
        __syncwarp();
        if (lane_1 < 24) {
            *(reinterpret_cast<float*>(param_partial + (prow + lane64)) + (0)) = dbase_n;
        }
        if (lane_1 < 3) {
            int glo = ((lane_1 < 2) ? lane_1 * 4 : 8);
            int ghi = ((lane_1 < 2) ? lane_1 * 4 + 4 : 24);
            float ga[1];
            float gc[1];
            ga[0] = 0.0f;
            gc[0] = 0.0f;
            #pragma unroll 1
            for (int mrow = glo; mrow < ghi; mrow++) {
                float sub_3;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(gsum[mrow]), "f"(gc[0]));
                float add_15;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(add_15) : "f"(ga[0]), "f"(sub_3));
                float _sub_1268;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1268) : "f"(add_15), "f"(ga[0]));
                float _sub_1267;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1267) : "f"(_sub_1268), "f"(sub_3));
                gc[0] = _sub_1267;
                ga[0] = add_15;
            }
            float _sub_1269;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1269) : "f"(ga[0]), "f"(gc[0]));
            *(reinterpret_cast<float*>(param_partial + (prow + 24 + lane64)) + (0)) = _sub_1269;
        }
        if (lane_1 >= 27) {
            *(reinterpret_cast<float*>(param_partial + (prow + lane64)) + (0)) = 0.0f;
        }
    }
    if (t0 < T && t_end >= T) {
        float zero = 0.0f;
        int npad = t_pad - T;
        long long t64 = (long long)T;
        int nz = npad * 96;
        int trips_z = (nz + 255) / 256;
        #pragma unroll 1
        for (int k = 0; k < trips_z; k++) {
            int ez = k * 256 + tid_2;
            if (ez < nz) {
                *(reinterpret_cast<__nv_bfloat16*>(dzp + (t64 * 96 + (long long)ez)) + (0)) = (__nv_bfloat16)zero;
            }
        }
        int nt = npad * 24;
        int trips_t = (nt + 255) / 256;
        #pragma unroll 1
        for (int k_1 = 0; k_1 < trips_t; k_1++) {
            int et = k_1 * 256 + tid_2;
            if (et < nt) {
                *(reinterpret_cast<float*>(tokp + (t64 * 24 + (long long)et)) + (0)) = zero;
            }
        }
    }
}

} // extern "C"
