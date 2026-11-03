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
kernel_cake_mhc_train_bwd_token__rt64__h5120(__nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ dU, float* __restrict__ stats, float* __restrict__ gstats, float* __restrict__ scale, float* __restrict__ base, __nv_bfloat16* __restrict__ dzp, float* __restrict__ tokp, float* __restrict__ param_partial, int T, int t_pad, int rows_per_cta, float hc_eps, int sinkhorn_iters)
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
    long long col_off = (long long)tid_2 * 20;
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
            float du_row[20];
            {
                uint2 _vld_0;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_0.x), "=r"(_vld_0.y) : "l"((const void*)(dU + (ta64_0 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
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
                    : "=r"(_vld_1.x), "=r"(_vld_1.y) : "l"((const void*)(dU + (ta64_0 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
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
                    : "=r"(_vld_2.x), "=r"(_vld_2.y) : "l"((const void*)(dU + (ta64_0 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
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
                    : "=r"(_vld_3.x), "=r"(_vld_3.y) : "l"((const void*)(dU + (ta64_0 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
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
                    : "=r"(_vld_4.x), "=r"(_vld_4.y) : "l"((const void*)(dU + (ta64_0 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&du_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
            float dots_p[4];
            float x_row[20];
            {
                uint2 _vld_5;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_5.x), "=r"(_vld_5.y) : "l"((const void*)(X + (ta64_0 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_6;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_6.x), "=r"(_vld_6.y) : "l"((const void*)(X + (ta64_0 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_7;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_7.x), "=r"(_vld_7.y) : "l"((const void*)(X + (ta64_0 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_8;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_8.x), "=r"(_vld_8.y) : "l"((const void*)(X + (ta64_0 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_9;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_9.x), "=r"(_vld_9.y) : "l"((const void*)(X + (ta64_0 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                }
            }
            dots_p[0] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_10;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[0]));
            dots_p[0] = _fma_10;
            float _fma_11;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[0]));
            dots_p[0] = _fma_11;
            float _fma_12;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[0]));
            dots_p[0] = _fma_12;
            float _fma_13;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[0]));
            dots_p[0] = _fma_13;
            float _fma_14;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[0]));
            dots_p[0] = _fma_14;
            float _fma_15;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[0]));
            dots_p[0] = _fma_15;
            float _fma_16;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[0]));
            dots_p[0] = _fma_16;
            float _fma_17;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[0]));
            dots_p[0] = _fma_17;
            float _fma_18;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[0]));
            dots_p[0] = _fma_18;
            float _fma_19;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[0]));
            dots_p[0] = _fma_19;
            float _fma_20;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[0]));
            dots_p[0] = _fma_20;
            float _fma_21;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[0]));
            dots_p[0] = _fma_21;
            float _fma_22;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[0]));
            dots_p[0] = _fma_22;
            float _fma_23;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[0]));
            dots_p[0] = _fma_23;
            float _fma_24;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[0]));
            dots_p[0] = _fma_24;
            float _fma_25;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[0]));
            dots_p[0] = _fma_25;
            float _fma_26;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[0]));
            dots_p[0] = _fma_26;
            float _fma_27;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[0]));
            dots_p[0] = _fma_27;
            float _fma_28;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[0]));
            dots_p[0] = _fma_28;
            {
                uint2 _vld_29;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_29.x), "=r"(_vld_29.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_30;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_30.x), "=r"(_vld_30.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_31;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_31.x), "=r"(_vld_31.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_32;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_32.x), "=r"(_vld_32.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_33;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_33.x), "=r"(_vld_33.y) : "l"((const void*)(X + ((ta64_0 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                }
            }
            dots_p[1] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_34;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[1]));
            dots_p[1] = _fma_34;
            float _fma_35;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[1]));
            dots_p[1] = _fma_35;
            float _fma_36;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[1]));
            dots_p[1] = _fma_36;
            float _fma_37;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[1]));
            dots_p[1] = _fma_37;
            float _fma_38;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[1]));
            dots_p[1] = _fma_38;
            float _fma_39;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[1]));
            dots_p[1] = _fma_39;
            float _fma_40;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[1]));
            dots_p[1] = _fma_40;
            float _fma_41;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[1]));
            dots_p[1] = _fma_41;
            float _fma_42;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[1]));
            dots_p[1] = _fma_42;
            float _fma_43;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[1]));
            dots_p[1] = _fma_43;
            float _fma_44;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[1]));
            dots_p[1] = _fma_44;
            float _fma_45;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[1]));
            dots_p[1] = _fma_45;
            float _fma_46;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[1]));
            dots_p[1] = _fma_46;
            float _fma_47;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[1]));
            dots_p[1] = _fma_47;
            float _fma_48;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[1]));
            dots_p[1] = _fma_48;
            float _fma_49;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[1]));
            dots_p[1] = _fma_49;
            float _fma_50;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[1]));
            dots_p[1] = _fma_50;
            float _fma_51;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[1]));
            dots_p[1] = _fma_51;
            float _fma_52;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[1]));
            dots_p[1] = _fma_52;
            {
                uint2 _vld_53;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_53.x), "=r"(_vld_53.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_53 = reinterpret_cast<uint32_t*>(&_vld_53);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_54;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_54.x), "=r"(_vld_54.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_54 = reinterpret_cast<uint32_t*>(&_vld_54);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_55;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_55.x), "=r"(_vld_55.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_56;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_56.x), "=r"(_vld_56.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_57;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_57.x), "=r"(_vld_57.y) : "l"((const void*)(X + ((ta64_0 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
                }
            }
            dots_p[2] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_58;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[2]));
            dots_p[2] = _fma_58;
            float _fma_59;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[2]));
            dots_p[2] = _fma_59;
            float _fma_60;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[2]));
            dots_p[2] = _fma_60;
            float _fma_61;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[2]));
            dots_p[2] = _fma_61;
            float _fma_62;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[2]));
            dots_p[2] = _fma_62;
            float _fma_63;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[2]));
            dots_p[2] = _fma_63;
            float _fma_64;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[2]));
            dots_p[2] = _fma_64;
            float _fma_65;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[2]));
            dots_p[2] = _fma_65;
            float _fma_66;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[2]));
            dots_p[2] = _fma_66;
            float _fma_67;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[2]));
            dots_p[2] = _fma_67;
            float _fma_68;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[2]));
            dots_p[2] = _fma_68;
            float _fma_69;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[2]));
            dots_p[2] = _fma_69;
            float _fma_70;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[2]));
            dots_p[2] = _fma_70;
            float _fma_71;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[2]));
            dots_p[2] = _fma_71;
            float _fma_72;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[2]));
            dots_p[2] = _fma_72;
            float _fma_73;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[2]));
            dots_p[2] = _fma_73;
            float _fma_74;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[2]));
            dots_p[2] = _fma_74;
            float _fma_75;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[2]));
            dots_p[2] = _fma_75;
            float _fma_76;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[2]));
            dots_p[2] = _fma_76;
            {
                uint2 _vld_77;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_77.x), "=r"(_vld_77.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_77 = reinterpret_cast<uint32_t*>(&_vld_77);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) << 16);
                    (&x_row[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_77[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_78;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_78.x), "=r"(_vld_78.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                    (&x_row[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_79;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_79.x), "=r"(_vld_79.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_79 = reinterpret_cast<uint32_t*>(&_vld_79);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) << 16);
                    (&x_row[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_79[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_80;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_80.x), "=r"(_vld_80.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                    (&x_row[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_81;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_81.x), "=r"(_vld_81.y) : "l"((const void*)(X + ((ta64_0 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_81 = reinterpret_cast<uint32_t*>(&_vld_81);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) << 16);
                    (&x_row[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) & 0xffff0000u);
                }
            }
            dots_p[3] = mul_f32_rn_noftz(x_row[0], du_row[0]);
            float _fma_82;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(x_row[1]), "f"(du_row[1]), "f"(dots_p[3]));
            dots_p[3] = _fma_82;
            float _fma_83;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(x_row[2]), "f"(du_row[2]), "f"(dots_p[3]));
            dots_p[3] = _fma_83;
            float _fma_84;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x_row[3]), "f"(du_row[3]), "f"(dots_p[3]));
            dots_p[3] = _fma_84;
            float _fma_85;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(x_row[4]), "f"(du_row[4]), "f"(dots_p[3]));
            dots_p[3] = _fma_85;
            float _fma_86;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(x_row[5]), "f"(du_row[5]), "f"(dots_p[3]));
            dots_p[3] = _fma_86;
            float _fma_87;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(x_row[6]), "f"(du_row[6]), "f"(dots_p[3]));
            dots_p[3] = _fma_87;
            float _fma_88;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(x_row[7]), "f"(du_row[7]), "f"(dots_p[3]));
            dots_p[3] = _fma_88;
            float _fma_89;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(x_row[8]), "f"(du_row[8]), "f"(dots_p[3]));
            dots_p[3] = _fma_89;
            float _fma_90;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(x_row[9]), "f"(du_row[9]), "f"(dots_p[3]));
            dots_p[3] = _fma_90;
            float _fma_91;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x_row[10]), "f"(du_row[10]), "f"(dots_p[3]));
            dots_p[3] = _fma_91;
            float _fma_92;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x_row[11]), "f"(du_row[11]), "f"(dots_p[3]));
            dots_p[3] = _fma_92;
            float _fma_93;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(x_row[12]), "f"(du_row[12]), "f"(dots_p[3]));
            dots_p[3] = _fma_93;
            float _fma_94;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(x_row[13]), "f"(du_row[13]), "f"(dots_p[3]));
            dots_p[3] = _fma_94;
            float _fma_95;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(x_row[14]), "f"(du_row[14]), "f"(dots_p[3]));
            dots_p[3] = _fma_95;
            float _fma_96;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(x_row[15]), "f"(du_row[15]), "f"(dots_p[3]));
            dots_p[3] = _fma_96;
            float _fma_97;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(x_row[16]), "f"(du_row[16]), "f"(dots_p[3]));
            dots_p[3] = _fma_97;
            float _fma_98;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(x_row[17]), "f"(du_row[17]), "f"(dots_p[3]));
            dots_p[3] = _fma_98;
            float _fma_99;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x_row[18]), "f"(du_row[18]), "f"(dots_p[3]));
            dots_p[3] = _fma_99;
            float _fma_100;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(x_row[19]), "f"(du_row[19]), "f"(dots_p[3]));
            dots_p[3] = _fma_100;
            int upper = lane_1 & 16;
            float send = ((upper != 0) ? dots_p[0] : dots_p[2]);
            float keep = ((upper != 0) ? dots_p[2] : dots_p[0]);
            float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, send, 16);
            float add;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(keep), "f"(_shfl_xor_sync_101));
            float send_0 = ((upper != 0) ? dots_p[1] : dots_p[3]);
            float keep_1 = ((upper != 0) ? dots_p[3] : dots_p[1]);
            float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
            float add_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(keep_1), "f"(_shfl_xor_sync_102));
            int upper_2 = lane_1 & 8;
            float send_3 = ((upper_2 != 0) ? add : add_1);
            float keep_4 = ((upper_2 != 0) ? add_1 : add);
            float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
            float add_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(keep_4), "f"(_shfl_xor_sync_103));
            float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, add_2, 4);
            float add_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(add_2), "f"(_shfl_xor_sync_104));
            float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, add_3, 2);
            float add_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4) : "f"(add_3), "f"(_shfl_xor_sync_105));
            float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, add_4, 1);
            float add_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5) : "f"(add_4), "f"(_shfl_xor_sync_106));
            if (lane8 == 0) {
                dpre_part[warp_0 * 4 + lane_hi] = add_5;
            }
        }
        int tok_a_1 = g0 + 1;
        if (tok_a_1 < t1) {
            long long ta64_1 = (long long)tok_a_1;
            float du_row_1[20];
            {
                uint2 _vld_107;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_107.x), "=r"(_vld_107.y) : "l"((const void*)(dU + (ta64_1 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_107 = reinterpret_cast<uint32_t*>(&_vld_107);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_107[_pair]) << 16);
                    (&du_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_107[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_108;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_108.x), "=r"(_vld_108.y) : "l"((const void*)(dU + (ta64_1 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_108 = reinterpret_cast<uint32_t*>(&_vld_108);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_108[_pair]) << 16);
                    (&du_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_108[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_109;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_109.x), "=r"(_vld_109.y) : "l"((const void*)(dU + (ta64_1 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_109 = reinterpret_cast<uint32_t*>(&_vld_109);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) << 16);
                    (&du_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_109[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_110;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_110.x), "=r"(_vld_110.y) : "l"((const void*)(dU + (ta64_1 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_110 = reinterpret_cast<uint32_t*>(&_vld_110);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_110[_pair]) << 16);
                    (&du_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_110[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_111;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_111.x), "=r"(_vld_111.y) : "l"((const void*)(dU + (ta64_1 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_111 = reinterpret_cast<uint32_t*>(&_vld_111);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) << 16);
                    (&du_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_1[4];
            float x_row_1[20];
            {
                uint2 _vld_112;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_112.x), "=r"(_vld_112.y) : "l"((const void*)(X + (ta64_1 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_112 = reinterpret_cast<uint32_t*>(&_vld_112);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_112[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_112[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_113;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_113.x), "=r"(_vld_113.y) : "l"((const void*)(X + (ta64_1 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_113 = reinterpret_cast<uint32_t*>(&_vld_113);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_114;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_114.x), "=r"(_vld_114.y) : "l"((const void*)(X + (ta64_1 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_114 = reinterpret_cast<uint32_t*>(&_vld_114);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_114[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_114[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_115;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_115.x), "=r"(_vld_115.y) : "l"((const void*)(X + (ta64_1 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_115 = reinterpret_cast<uint32_t*>(&_vld_115);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_115[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_115[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_116;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_116.x), "=r"(_vld_116.y) : "l"((const void*)(X + (ta64_1 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_116 = reinterpret_cast<uint32_t*>(&_vld_116);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_116[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_116[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[0] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_117;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_117;
            float _fma_118;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_118;
            float _fma_119;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_119;
            float _fma_120;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_120;
            float _fma_121;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_121;
            float _fma_122;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_122;
            float _fma_123;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_123;
            float _fma_124;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_124;
            float _fma_125;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_125;
            float _fma_126;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_126;
            float _fma_127;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_127;
            float _fma_128;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_128;
            float _fma_129;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_129;
            float _fma_130;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_130;
            float _fma_131;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_131;
            float _fma_132;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_132;
            float _fma_133;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_133;
            float _fma_134;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_134;
            float _fma_135;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[0]));
            dots_p_1[0] = _fma_135;
            {
                uint2 _vld_136;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_136.x), "=r"(_vld_136.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_136 = reinterpret_cast<uint32_t*>(&_vld_136);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_136[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_136[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_137;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_137.x), "=r"(_vld_137.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_137 = reinterpret_cast<uint32_t*>(&_vld_137);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_137[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_137[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_138;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_138.x), "=r"(_vld_138.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_138 = reinterpret_cast<uint32_t*>(&_vld_138);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_138[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_138[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_139;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_139.x), "=r"(_vld_139.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_139 = reinterpret_cast<uint32_t*>(&_vld_139);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_139[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_139[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_140;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_140.x), "=r"(_vld_140.y) : "l"((const void*)(X + ((ta64_1 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_140 = reinterpret_cast<uint32_t*>(&_vld_140);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_140[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_140[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[1] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_141;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_141;
            float _fma_142;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_142;
            float _fma_143;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_143;
            float _fma_144;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_144;
            float _fma_145;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_145;
            float _fma_146;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_146;
            float _fma_147;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_147;
            float _fma_148;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_148;
            float _fma_149;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_149;
            float _fma_150;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_150;
            float _fma_151;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_151;
            float _fma_152;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_152;
            float _fma_153;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_153;
            float _fma_154;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_154;
            float _fma_155;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_155;
            float _fma_156;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_156;
            float _fma_157;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_157;
            float _fma_158;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_158;
            float _fma_159;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[1]));
            dots_p_1[1] = _fma_159;
            {
                uint2 _vld_160;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_160.x), "=r"(_vld_160.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_160 = reinterpret_cast<uint32_t*>(&_vld_160);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_160[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_160[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_161;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_161.x), "=r"(_vld_161.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_161 = reinterpret_cast<uint32_t*>(&_vld_161);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_161[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_161[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_162;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_162.x), "=r"(_vld_162.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_162 = reinterpret_cast<uint32_t*>(&_vld_162);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_162[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_162[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_163;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_163.x), "=r"(_vld_163.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_163 = reinterpret_cast<uint32_t*>(&_vld_163);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_163[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_163[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_164;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_164.x), "=r"(_vld_164.y) : "l"((const void*)(X + ((ta64_1 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_164 = reinterpret_cast<uint32_t*>(&_vld_164);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_164[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_164[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[2] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_165;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_165;
            float _fma_166;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_166;
            float _fma_167;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_167;
            float _fma_168;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_168;
            float _fma_169;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_169;
            float _fma_170;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_170;
            float _fma_171;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_171;
            float _fma_172;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_172;
            float _fma_173;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_173;
            float _fma_174;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_174;
            float _fma_175;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_175;
            float _fma_176;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_176;
            float _fma_177;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_177;
            float _fma_178;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_178;
            float _fma_179;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_179;
            float _fma_180;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_180;
            float _fma_181;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_181;
            float _fma_182;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_182;
            float _fma_183;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[2]));
            dots_p_1[2] = _fma_183;
            {
                uint2 _vld_184;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_184.x), "=r"(_vld_184.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_184 = reinterpret_cast<uint32_t*>(&_vld_184);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_184[_pair]) << 16);
                    (&x_row_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_184[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_185;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_185.x), "=r"(_vld_185.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_185 = reinterpret_cast<uint32_t*>(&_vld_185);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_185[_pair]) << 16);
                    (&x_row_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_185[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_186;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_186.x), "=r"(_vld_186.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_186 = reinterpret_cast<uint32_t*>(&_vld_186);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_186[_pair]) << 16);
                    (&x_row_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_186[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_187;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_187.x), "=r"(_vld_187.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_187 = reinterpret_cast<uint32_t*>(&_vld_187);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_187[_pair]) << 16);
                    (&x_row_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_187[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_188;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_188.x), "=r"(_vld_188.y) : "l"((const void*)(X + ((ta64_1 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_188 = reinterpret_cast<uint32_t*>(&_vld_188);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_1[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_188[_pair]) << 16);
                    (&x_row_1[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_188[_pair]) & 0xffff0000u);
                }
            }
            dots_p_1[3] = mul_f32_rn_noftz(x_row_1[0], du_row_1[0]);
            float _fma_189;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(x_row_1[1]), "f"(du_row_1[1]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_189;
            float _fma_190;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(x_row_1[2]), "f"(du_row_1[2]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_190;
            float _fma_191;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(x_row_1[3]), "f"(du_row_1[3]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_191;
            float _fma_192;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(x_row_1[4]), "f"(du_row_1[4]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_192;
            float _fma_193;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(x_row_1[5]), "f"(du_row_1[5]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_193;
            float _fma_194;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(x_row_1[6]), "f"(du_row_1[6]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_194;
            float _fma_195;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(x_row_1[7]), "f"(du_row_1[7]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_195;
            float _fma_196;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(x_row_1[8]), "f"(du_row_1[8]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_196;
            float _fma_197;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(x_row_1[9]), "f"(du_row_1[9]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_197;
            float _fma_198;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(x_row_1[10]), "f"(du_row_1[10]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_198;
            float _fma_199;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(x_row_1[11]), "f"(du_row_1[11]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_199;
            float _fma_200;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(x_row_1[12]), "f"(du_row_1[12]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_200;
            float _fma_201;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(x_row_1[13]), "f"(du_row_1[13]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_201;
            float _fma_202;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(x_row_1[14]), "f"(du_row_1[14]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_202;
            float _fma_203;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(x_row_1[15]), "f"(du_row_1[15]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_203;
            float _fma_204;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(x_row_1[16]), "f"(du_row_1[16]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_204;
            float _fma_205;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(x_row_1[17]), "f"(du_row_1[17]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_205;
            float _fma_206;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(x_row_1[18]), "f"(du_row_1[18]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_206;
            float _fma_207;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(x_row_1[19]), "f"(du_row_1[19]), "f"(dots_p_1[3]));
            dots_p_1[3] = _fma_207;
            int upper_1 = lane_1 & 16;
            float send_1 = ((upper_1 != 0) ? dots_p_1[0] : dots_p_1[2]);
            float keep_2 = ((upper_1 != 0) ? dots_p_1[2] : dots_p_1[0]);
            float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
            float add_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_6) : "f"(keep_2), "f"(_shfl_xor_sync_208));
            float send_0_1 = ((upper_1 != 0) ? dots_p_1[1] : dots_p_1[3]);
            float keep_1_1 = ((upper_1 != 0) ? dots_p_1[3] : dots_p_1[1]);
            float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, send_0_1, 16);
            float add_1_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_1) : "f"(keep_1_1), "f"(_shfl_xor_sync_209));
            int upper_2_1 = lane_1 & 8;
            float send_3_1 = ((upper_2_1 != 0) ? add_6 : add_1_1);
            float keep_4_1 = ((upper_2_1 != 0) ? add_1_1 : add_6);
            float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, send_3_1, 8);
            float add_2_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_1) : "f"(keep_4_1), "f"(_shfl_xor_sync_210));
            float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, add_2_1, 4);
            float add_3_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_1) : "f"(add_2_1), "f"(_shfl_xor_sync_211));
            float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, add_3_1, 2);
            float add_4_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_1) : "f"(add_3_1), "f"(_shfl_xor_sync_212));
            float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, add_4_1, 1);
            float add_5_1;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_1) : "f"(add_4_1), "f"(_shfl_xor_sync_213));
            if (lane8 == 0) {
                dpre_part[32 + warp_0 * 4 + lane_hi] = add_5_1;
            }
        }
        int tok_a_2 = g0 + 2;
        if (tok_a_2 < t1) {
            long long ta64_2 = (long long)tok_a_2;
            float du_row_2[20];
            {
                uint2 _vld_214;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_214.x), "=r"(_vld_214.y) : "l"((const void*)(dU + (ta64_2 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_214 = reinterpret_cast<uint32_t*>(&_vld_214);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_214[_pair]) << 16);
                    (&du_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_214[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_215;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_215.x), "=r"(_vld_215.y) : "l"((const void*)(dU + (ta64_2 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_215 = reinterpret_cast<uint32_t*>(&_vld_215);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_215[_pair]) << 16);
                    (&du_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_215[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_216;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_216.x), "=r"(_vld_216.y) : "l"((const void*)(dU + (ta64_2 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_216 = reinterpret_cast<uint32_t*>(&_vld_216);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_216[_pair]) << 16);
                    (&du_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_216[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_217;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_217.x), "=r"(_vld_217.y) : "l"((const void*)(dU + (ta64_2 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_217 = reinterpret_cast<uint32_t*>(&_vld_217);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_217[_pair]) << 16);
                    (&du_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_217[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_218;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_218.x), "=r"(_vld_218.y) : "l"((const void*)(dU + (ta64_2 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_218 = reinterpret_cast<uint32_t*>(&_vld_218);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_218[_pair]) << 16);
                    (&du_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_218[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_2[4];
            float x_row_2[20];
            {
                uint2 _vld_219;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_219.x), "=r"(_vld_219.y) : "l"((const void*)(X + (ta64_2 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_219 = reinterpret_cast<uint32_t*>(&_vld_219);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_219[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_219[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_220;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_220.x), "=r"(_vld_220.y) : "l"((const void*)(X + (ta64_2 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_220 = reinterpret_cast<uint32_t*>(&_vld_220);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_220[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_220[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_221;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_221.x), "=r"(_vld_221.y) : "l"((const void*)(X + (ta64_2 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_221 = reinterpret_cast<uint32_t*>(&_vld_221);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_221[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_221[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_222;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_222.x), "=r"(_vld_222.y) : "l"((const void*)(X + (ta64_2 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_222 = reinterpret_cast<uint32_t*>(&_vld_222);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_222[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_222[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_223;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_223.x), "=r"(_vld_223.y) : "l"((const void*)(X + (ta64_2 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_223 = reinterpret_cast<uint32_t*>(&_vld_223);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_223[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_223[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[0] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_224;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_224;
            float _fma_225;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_225;
            float _fma_226;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_226;
            float _fma_227;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_227;
            float _fma_228;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_228;
            float _fma_229;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_229;
            float _fma_230;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_230;
            float _fma_231;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_231;
            float _fma_232;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_232;
            float _fma_233;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_233;
            float _fma_234;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_234;
            float _fma_235;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_235;
            float _fma_236;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_236;
            float _fma_237;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_237;
            float _fma_238;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_238;
            float _fma_239;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_239;
            float _fma_240;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_240;
            float _fma_241;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_241;
            float _fma_242;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[0]));
            dots_p_2[0] = _fma_242;
            {
                uint2 _vld_243;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_243.x), "=r"(_vld_243.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_243 = reinterpret_cast<uint32_t*>(&_vld_243);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_243[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_243[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_244;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_244.x), "=r"(_vld_244.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_244 = reinterpret_cast<uint32_t*>(&_vld_244);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_244[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_244[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_245;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_245.x), "=r"(_vld_245.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_245 = reinterpret_cast<uint32_t*>(&_vld_245);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_245[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_245[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_246;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_246.x), "=r"(_vld_246.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_246 = reinterpret_cast<uint32_t*>(&_vld_246);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_246[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_246[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_247;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_247.x), "=r"(_vld_247.y) : "l"((const void*)(X + ((ta64_2 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_247 = reinterpret_cast<uint32_t*>(&_vld_247);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_247[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_247[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[1] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_248;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_248;
            float _fma_249;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_249;
            float _fma_250;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_250;
            float _fma_251;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_251;
            float _fma_252;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_252;
            float _fma_253;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_253;
            float _fma_254;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_254;
            float _fma_255;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_255;
            float _fma_256;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_256;
            float _fma_257;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_257;
            float _fma_258;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_258;
            float _fma_259;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_259;
            float _fma_260;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_260;
            float _fma_261;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_261;
            float _fma_262;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_262;
            float _fma_263;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_263;
            float _fma_264;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_264;
            float _fma_265;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_265;
            float _fma_266;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[1]));
            dots_p_2[1] = _fma_266;
            {
                uint2 _vld_267;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_267.x), "=r"(_vld_267.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_267 = reinterpret_cast<uint32_t*>(&_vld_267);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_267[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_267[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_268;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_268.x), "=r"(_vld_268.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_268 = reinterpret_cast<uint32_t*>(&_vld_268);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_268[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_268[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_269;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_269.x), "=r"(_vld_269.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_269 = reinterpret_cast<uint32_t*>(&_vld_269);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_269[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_269[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_270;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_270.x), "=r"(_vld_270.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_270 = reinterpret_cast<uint32_t*>(&_vld_270);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_270[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_270[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_271;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_271.x), "=r"(_vld_271.y) : "l"((const void*)(X + ((ta64_2 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_271 = reinterpret_cast<uint32_t*>(&_vld_271);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_271[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_271[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[2] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_272;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_272;
            float _fma_273;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_273;
            float _fma_274;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_274;
            float _fma_275;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_275;
            float _fma_276;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_276;
            float _fma_277;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_277;
            float _fma_278;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_278;
            float _fma_279;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_279;
            float _fma_280;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_280;
            float _fma_281;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_281;
            float _fma_282;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_282;
            float _fma_283;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_283;
            float _fma_284;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_284;
            float _fma_285;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_285;
            float _fma_286;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_286;
            float _fma_287;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_287;
            float _fma_288;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_288;
            float _fma_289;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_289;
            float _fma_290;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[2]));
            dots_p_2[2] = _fma_290;
            {
                uint2 _vld_291;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_291.x), "=r"(_vld_291.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_291 = reinterpret_cast<uint32_t*>(&_vld_291);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_291[_pair]) << 16);
                    (&x_row_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_291[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_292;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_292.x), "=r"(_vld_292.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_292 = reinterpret_cast<uint32_t*>(&_vld_292);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_292[_pair]) << 16);
                    (&x_row_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_292[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_293;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_293.x), "=r"(_vld_293.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_293 = reinterpret_cast<uint32_t*>(&_vld_293);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_293[_pair]) << 16);
                    (&x_row_2[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_293[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_294;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_294.x), "=r"(_vld_294.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_294 = reinterpret_cast<uint32_t*>(&_vld_294);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_294[_pair]) << 16);
                    (&x_row_2[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_294[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_295;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_295.x), "=r"(_vld_295.y) : "l"((const void*)(X + ((ta64_2 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_295 = reinterpret_cast<uint32_t*>(&_vld_295);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_2[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_295[_pair]) << 16);
                    (&x_row_2[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_295[_pair]) & 0xffff0000u);
                }
            }
            dots_p_2[3] = mul_f32_rn_noftz(x_row_2[0], du_row_2[0]);
            float _fma_296;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(x_row_2[1]), "f"(du_row_2[1]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_296;
            float _fma_297;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(x_row_2[2]), "f"(du_row_2[2]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_297;
            float _fma_298;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(x_row_2[3]), "f"(du_row_2[3]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_298;
            float _fma_299;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(x_row_2[4]), "f"(du_row_2[4]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_299;
            float _fma_300;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(x_row_2[5]), "f"(du_row_2[5]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_300;
            float _fma_301;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(x_row_2[6]), "f"(du_row_2[6]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_301;
            float _fma_302;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(x_row_2[7]), "f"(du_row_2[7]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_302;
            float _fma_303;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(x_row_2[8]), "f"(du_row_2[8]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_303;
            float _fma_304;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(x_row_2[9]), "f"(du_row_2[9]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_304;
            float _fma_305;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(x_row_2[10]), "f"(du_row_2[10]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_305;
            float _fma_306;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(x_row_2[11]), "f"(du_row_2[11]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_306;
            float _fma_307;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(x_row_2[12]), "f"(du_row_2[12]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_307;
            float _fma_308;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(x_row_2[13]), "f"(du_row_2[13]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_308;
            float _fma_309;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(x_row_2[14]), "f"(du_row_2[14]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_309;
            float _fma_310;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(x_row_2[15]), "f"(du_row_2[15]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_310;
            float _fma_311;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(x_row_2[16]), "f"(du_row_2[16]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_311;
            float _fma_312;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(x_row_2[17]), "f"(du_row_2[17]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_312;
            float _fma_313;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(x_row_2[18]), "f"(du_row_2[18]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_313;
            float _fma_314;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(x_row_2[19]), "f"(du_row_2[19]), "f"(dots_p_2[3]));
            dots_p_2[3] = _fma_314;
            int upper_3 = lane_1 & 16;
            float send_2 = ((upper_3 != 0) ? dots_p_2[0] : dots_p_2[2]);
            float keep_3 = ((upper_3 != 0) ? dots_p_2[2] : dots_p_2[0]);
            float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, send_2, 16);
            float add_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_7) : "f"(keep_3), "f"(_shfl_xor_sync_315));
            float send_0_2 = ((upper_3 != 0) ? dots_p_2[1] : dots_p_2[3]);
            float keep_1_2 = ((upper_3 != 0) ? dots_p_2[3] : dots_p_2[1]);
            float _shfl_xor_sync_316 = __shfl_xor_sync(0xFFFFFFFF, send_0_2, 16);
            float add_1_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_2) : "f"(keep_1_2), "f"(_shfl_xor_sync_316));
            int upper_2_2 = lane_1 & 8;
            float send_3_2 = ((upper_2_2 != 0) ? add_7 : add_1_2);
            float keep_4_2 = ((upper_2_2 != 0) ? add_1_2 : add_7);
            float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, send_3_2, 8);
            float add_2_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_2) : "f"(keep_4_2), "f"(_shfl_xor_sync_317));
            float _shfl_xor_sync_318 = __shfl_xor_sync(0xFFFFFFFF, add_2_2, 4);
            float add_3_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_2) : "f"(add_2_2), "f"(_shfl_xor_sync_318));
            float _shfl_xor_sync_319 = __shfl_xor_sync(0xFFFFFFFF, add_3_2, 2);
            float add_4_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_2) : "f"(add_3_2), "f"(_shfl_xor_sync_319));
            float _shfl_xor_sync_320 = __shfl_xor_sync(0xFFFFFFFF, add_4_2, 1);
            float add_5_2;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_2) : "f"(add_4_2), "f"(_shfl_xor_sync_320));
            if (lane8 == 0) {
                dpre_part[64 + warp_0 * 4 + lane_hi] = add_5_2;
            }
        }
        int tok_a_3 = g0 + 3;
        if (tok_a_3 < t1) {
            long long ta64_3 = (long long)tok_a_3;
            float du_row_3[20];
            {
                uint2 _vld_321;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_321.x), "=r"(_vld_321.y) : "l"((const void*)(dU + (ta64_3 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_321 = reinterpret_cast<uint32_t*>(&_vld_321);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_321[_pair]) << 16);
                    (&du_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_321[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_322;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_322.x), "=r"(_vld_322.y) : "l"((const void*)(dU + (ta64_3 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_322 = reinterpret_cast<uint32_t*>(&_vld_322);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_322[_pair]) << 16);
                    (&du_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_322[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_323;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_323.x), "=r"(_vld_323.y) : "l"((const void*)(dU + (ta64_3 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_323 = reinterpret_cast<uint32_t*>(&_vld_323);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_323[_pair]) << 16);
                    (&du_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_323[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_324;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_324.x), "=r"(_vld_324.y) : "l"((const void*)(dU + (ta64_3 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_324 = reinterpret_cast<uint32_t*>(&_vld_324);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_324[_pair]) << 16);
                    (&du_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_324[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_325;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_325.x), "=r"(_vld_325.y) : "l"((const void*)(dU + (ta64_3 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_325 = reinterpret_cast<uint32_t*>(&_vld_325);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_325[_pair]) << 16);
                    (&du_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_325[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_3[4];
            float x_row_3[20];
            {
                uint2 _vld_326;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_326.x), "=r"(_vld_326.y) : "l"((const void*)(X + (ta64_3 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_326 = reinterpret_cast<uint32_t*>(&_vld_326);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_326[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_326[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_327;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_327.x), "=r"(_vld_327.y) : "l"((const void*)(X + (ta64_3 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_327 = reinterpret_cast<uint32_t*>(&_vld_327);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_327[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_327[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_328;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_328.x), "=r"(_vld_328.y) : "l"((const void*)(X + (ta64_3 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_328 = reinterpret_cast<uint32_t*>(&_vld_328);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_328[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_328[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_329;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_329.x), "=r"(_vld_329.y) : "l"((const void*)(X + (ta64_3 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_329 = reinterpret_cast<uint32_t*>(&_vld_329);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_329[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_329[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_330;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_330.x), "=r"(_vld_330.y) : "l"((const void*)(X + (ta64_3 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_330 = reinterpret_cast<uint32_t*>(&_vld_330);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_330[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_330[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[0] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_331;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_331;
            float _fma_332;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_332;
            float _fma_333;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_333;
            float _fma_334;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_334;
            float _fma_335;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_335;
            float _fma_336;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_336;
            float _fma_337;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_337;
            float _fma_338;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_338;
            float _fma_339;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_339;
            float _fma_340;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_340;
            float _fma_341;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_341;
            float _fma_342;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_342;
            float _fma_343;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_343;
            float _fma_344;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_344;
            float _fma_345;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_345;
            float _fma_346;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_346;
            float _fma_347;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_347;
            float _fma_348;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_348;
            float _fma_349;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[0]));
            dots_p_3[0] = _fma_349;
            {
                uint2 _vld_350;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_350.x), "=r"(_vld_350.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_350 = reinterpret_cast<uint32_t*>(&_vld_350);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_350[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_350[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_351;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_351.x), "=r"(_vld_351.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_351 = reinterpret_cast<uint32_t*>(&_vld_351);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_351[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_351[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_352;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_352.x), "=r"(_vld_352.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_352 = reinterpret_cast<uint32_t*>(&_vld_352);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_352[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_352[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_353;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_353.x), "=r"(_vld_353.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_353 = reinterpret_cast<uint32_t*>(&_vld_353);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_353[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_353[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_354;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_354.x), "=r"(_vld_354.y) : "l"((const void*)(X + ((ta64_3 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_354 = reinterpret_cast<uint32_t*>(&_vld_354);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_354[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_354[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[1] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_355;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_355;
            float _fma_356;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_356;
            float _fma_357;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_357;
            float _fma_358;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_358;
            float _fma_359;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_359;
            float _fma_360;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_360;
            float _fma_361;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_361;
            float _fma_362;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_362;
            float _fma_363;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_363;
            float _fma_364;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_364;
            float _fma_365;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_365;
            float _fma_366;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_366;
            float _fma_367;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_367;
            float _fma_368;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_368;
            float _fma_369;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_369;
            float _fma_370;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_370;
            float _fma_371;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_371;
            float _fma_372;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_372;
            float _fma_373;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_373) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[1]));
            dots_p_3[1] = _fma_373;
            {
                uint2 _vld_374;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_374.x), "=r"(_vld_374.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_374 = reinterpret_cast<uint32_t*>(&_vld_374);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_374[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_374[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_375;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_375.x), "=r"(_vld_375.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_375 = reinterpret_cast<uint32_t*>(&_vld_375);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_375[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_375[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_376;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_376.x), "=r"(_vld_376.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_376 = reinterpret_cast<uint32_t*>(&_vld_376);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_376[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_376[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_377;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_377.x), "=r"(_vld_377.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_377 = reinterpret_cast<uint32_t*>(&_vld_377);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_377[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_377[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_378;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_378.x), "=r"(_vld_378.y) : "l"((const void*)(X + ((ta64_3 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_378 = reinterpret_cast<uint32_t*>(&_vld_378);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_378[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_378[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[2] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_379;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_379) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_379;
            float _fma_380;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_380;
            float _fma_381;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_381) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_381;
            float _fma_382;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_382) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_382;
            float _fma_383;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_383) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_383;
            float _fma_384;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_384) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_384;
            float _fma_385;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_385;
            float _fma_386;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_386) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_386;
            float _fma_387;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_387) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_387;
            float _fma_388;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_388) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_388;
            float _fma_389;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_389) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_389;
            float _fma_390;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_390) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_390;
            float _fma_391;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_391;
            float _fma_392;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_392;
            float _fma_393;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_393;
            float _fma_394;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_394) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_394;
            float _fma_395;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_395;
            float _fma_396;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_396;
            float _fma_397;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[2]));
            dots_p_3[2] = _fma_397;
            {
                uint2 _vld_398;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_398.x), "=r"(_vld_398.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_398 = reinterpret_cast<uint32_t*>(&_vld_398);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_398[_pair]) << 16);
                    (&x_row_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_398[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_399;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_399.x), "=r"(_vld_399.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_399 = reinterpret_cast<uint32_t*>(&_vld_399);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_399[_pair]) << 16);
                    (&x_row_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_399[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_400;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_400.x), "=r"(_vld_400.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_400 = reinterpret_cast<uint32_t*>(&_vld_400);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_400[_pair]) << 16);
                    (&x_row_3[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_400[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_401;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_401.x), "=r"(_vld_401.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_401 = reinterpret_cast<uint32_t*>(&_vld_401);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_401[_pair]) << 16);
                    (&x_row_3[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_401[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_402;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_402.x), "=r"(_vld_402.y) : "l"((const void*)(X + ((ta64_3 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_402 = reinterpret_cast<uint32_t*>(&_vld_402);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_3[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_402[_pair]) << 16);
                    (&x_row_3[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_402[_pair]) & 0xffff0000u);
                }
            }
            dots_p_3[3] = mul_f32_rn_noftz(x_row_3[0], du_row_3[0]);
            float _fma_403;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(x_row_3[1]), "f"(du_row_3[1]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_403;
            float _fma_404;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(x_row_3[2]), "f"(du_row_3[2]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_404;
            float _fma_405;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(x_row_3[3]), "f"(du_row_3[3]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_405;
            float _fma_406;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(x_row_3[4]), "f"(du_row_3[4]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_406;
            float _fma_407;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(x_row_3[5]), "f"(du_row_3[5]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_407;
            float _fma_408;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_408) : "f"(x_row_3[6]), "f"(du_row_3[6]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_408;
            float _fma_409;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(x_row_3[7]), "f"(du_row_3[7]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_409;
            float _fma_410;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(x_row_3[8]), "f"(du_row_3[8]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_410;
            float _fma_411;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_411) : "f"(x_row_3[9]), "f"(du_row_3[9]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_411;
            float _fma_412;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_412) : "f"(x_row_3[10]), "f"(du_row_3[10]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_412;
            float _fma_413;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_413) : "f"(x_row_3[11]), "f"(du_row_3[11]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_413;
            float _fma_414;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_414) : "f"(x_row_3[12]), "f"(du_row_3[12]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_414;
            float _fma_415;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_415) : "f"(x_row_3[13]), "f"(du_row_3[13]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_415;
            float _fma_416;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_416) : "f"(x_row_3[14]), "f"(du_row_3[14]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_416;
            float _fma_417;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_417) : "f"(x_row_3[15]), "f"(du_row_3[15]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_417;
            float _fma_418;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_418) : "f"(x_row_3[16]), "f"(du_row_3[16]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_418;
            float _fma_419;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_419) : "f"(x_row_3[17]), "f"(du_row_3[17]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_419;
            float _fma_420;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_420) : "f"(x_row_3[18]), "f"(du_row_3[18]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_420;
            float _fma_421;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(x_row_3[19]), "f"(du_row_3[19]), "f"(dots_p_3[3]));
            dots_p_3[3] = _fma_421;
            int upper_4 = lane_1 & 16;
            float send_4 = ((upper_4 != 0) ? dots_p_3[0] : dots_p_3[2]);
            float keep_5 = ((upper_4 != 0) ? dots_p_3[2] : dots_p_3[0]);
            float _shfl_xor_sync_422 = __shfl_xor_sync(0xFFFFFFFF, send_4, 16);
            float add_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_8) : "f"(keep_5), "f"(_shfl_xor_sync_422));
            float send_0_3 = ((upper_4 != 0) ? dots_p_3[1] : dots_p_3[3]);
            float keep_1_3 = ((upper_4 != 0) ? dots_p_3[3] : dots_p_3[1]);
            float _shfl_xor_sync_423 = __shfl_xor_sync(0xFFFFFFFF, send_0_3, 16);
            float add_1_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_3) : "f"(keep_1_3), "f"(_shfl_xor_sync_423));
            int upper_2_3 = lane_1 & 8;
            float send_3_3 = ((upper_2_3 != 0) ? add_8 : add_1_3);
            float keep_4_3 = ((upper_2_3 != 0) ? add_1_3 : add_8);
            float _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, send_3_3, 8);
            float add_2_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_3) : "f"(keep_4_3), "f"(_shfl_xor_sync_424));
            float _shfl_xor_sync_425 = __shfl_xor_sync(0xFFFFFFFF, add_2_3, 4);
            float add_3_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_3) : "f"(add_2_3), "f"(_shfl_xor_sync_425));
            float _shfl_xor_sync_426 = __shfl_xor_sync(0xFFFFFFFF, add_3_3, 2);
            float add_4_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_3) : "f"(add_3_3), "f"(_shfl_xor_sync_426));
            float _shfl_xor_sync_427 = __shfl_xor_sync(0xFFFFFFFF, add_4_3, 1);
            float add_5_3;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_3) : "f"(add_4_3), "f"(_shfl_xor_sync_427));
            if (lane8 == 0) {
                dpre_part[96 + warp_0 * 4 + lane_hi] = add_5_3;
            }
        }
        int tok_a_4 = g0 + 4;
        if (tok_a_4 < t1) {
            long long ta64_4 = (long long)tok_a_4;
            float du_row_4[20];
            {
                uint2 _vld_428;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_428.x), "=r"(_vld_428.y) : "l"((const void*)(dU + (ta64_4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_428 = reinterpret_cast<uint32_t*>(&_vld_428);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_428[_pair]) << 16);
                    (&du_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_428[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_429;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_429.x), "=r"(_vld_429.y) : "l"((const void*)(dU + (ta64_4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_429 = reinterpret_cast<uint32_t*>(&_vld_429);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_429[_pair]) << 16);
                    (&du_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_429[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_430;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_430.x), "=r"(_vld_430.y) : "l"((const void*)(dU + (ta64_4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_430 = reinterpret_cast<uint32_t*>(&_vld_430);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_430[_pair]) << 16);
                    (&du_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_430[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_431;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_431.x), "=r"(_vld_431.y) : "l"((const void*)(dU + (ta64_4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_431 = reinterpret_cast<uint32_t*>(&_vld_431);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_431[_pair]) << 16);
                    (&du_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_431[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_432;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_432.x), "=r"(_vld_432.y) : "l"((const void*)(dU + (ta64_4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_432 = reinterpret_cast<uint32_t*>(&_vld_432);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_432[_pair]) << 16);
                    (&du_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_432[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_4[4];
            float x_row_4[20];
            {
                uint2 _vld_433;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_433.x), "=r"(_vld_433.y) : "l"((const void*)(X + (ta64_4 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_433 = reinterpret_cast<uint32_t*>(&_vld_433);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_433[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_433[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_434;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_434.x), "=r"(_vld_434.y) : "l"((const void*)(X + (ta64_4 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_434 = reinterpret_cast<uint32_t*>(&_vld_434);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_434[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_434[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_435;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_435.x), "=r"(_vld_435.y) : "l"((const void*)(X + (ta64_4 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_435 = reinterpret_cast<uint32_t*>(&_vld_435);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_435[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_435[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_436;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_436.x), "=r"(_vld_436.y) : "l"((const void*)(X + (ta64_4 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_436 = reinterpret_cast<uint32_t*>(&_vld_436);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_436[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_436[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_437;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_437.x), "=r"(_vld_437.y) : "l"((const void*)(X + (ta64_4 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_437 = reinterpret_cast<uint32_t*>(&_vld_437);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_437[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_437[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[0] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_438;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_438;
            float _fma_439;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_439;
            float _fma_440;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_440;
            float _fma_441;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_441;
            float _fma_442;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_442) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_442;
            float _fma_443;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_443;
            float _fma_444;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_444) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_444;
            float _fma_445;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_445;
            float _fma_446;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_446) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_446;
            float _fma_447;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_447;
            float _fma_448;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_448) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_448;
            float _fma_449;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_449) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_449;
            float _fma_450;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_450) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_450;
            float _fma_451;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_451) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_451;
            float _fma_452;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_452) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_452;
            float _fma_453;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_453) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_453;
            float _fma_454;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_454) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_454;
            float _fma_455;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_455) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_455;
            float _fma_456;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_456) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[0]));
            dots_p_4[0] = _fma_456;
            {
                uint2 _vld_457;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_457.x), "=r"(_vld_457.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_457 = reinterpret_cast<uint32_t*>(&_vld_457);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_457[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_457[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_458;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_458.x), "=r"(_vld_458.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_458 = reinterpret_cast<uint32_t*>(&_vld_458);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_458[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_458[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_459;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_459.x), "=r"(_vld_459.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_459 = reinterpret_cast<uint32_t*>(&_vld_459);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_459[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_459[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_460;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_460.x), "=r"(_vld_460.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_460 = reinterpret_cast<uint32_t*>(&_vld_460);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_460[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_460[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_461;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_461.x), "=r"(_vld_461.y) : "l"((const void*)(X + ((ta64_4 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_461 = reinterpret_cast<uint32_t*>(&_vld_461);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_461[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_461[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[1] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_462;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_462) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_462;
            float _fma_463;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_463) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_463;
            float _fma_464;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_464) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_464;
            float _fma_465;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_465) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_465;
            float _fma_466;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_466) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_466;
            float _fma_467;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_467) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_467;
            float _fma_468;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_468) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_468;
            float _fma_469;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_469) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_469;
            float _fma_470;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_470) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_470;
            float _fma_471;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_471) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_471;
            float _fma_472;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_472) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_472;
            float _fma_473;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_473;
            float _fma_474;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_474;
            float _fma_475;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_475;
            float _fma_476;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_476;
            float _fma_477;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_477;
            float _fma_478;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_478;
            float _fma_479;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_479;
            float _fma_480;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[1]));
            dots_p_4[1] = _fma_480;
            {
                uint2 _vld_481;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_481.x), "=r"(_vld_481.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_481 = reinterpret_cast<uint32_t*>(&_vld_481);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_481[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_481[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_482;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_482.x), "=r"(_vld_482.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_482 = reinterpret_cast<uint32_t*>(&_vld_482);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_482[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_482[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_483;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_483.x), "=r"(_vld_483.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_483 = reinterpret_cast<uint32_t*>(&_vld_483);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_483[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_483[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_484;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_484.x), "=r"(_vld_484.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_484 = reinterpret_cast<uint32_t*>(&_vld_484);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_484[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_484[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_485;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_485.x), "=r"(_vld_485.y) : "l"((const void*)(X + ((ta64_4 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_485 = reinterpret_cast<uint32_t*>(&_vld_485);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_485[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_485[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[2] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_486;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_486) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_486;
            float _fma_487;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_487) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_487;
            float _fma_488;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_488) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_488;
            float _fma_489;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_489) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_489;
            float _fma_490;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_490) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_490;
            float _fma_491;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_491) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_491;
            float _fma_492;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_492) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_492;
            float _fma_493;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_493) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_493;
            float _fma_494;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_494) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_494;
            float _fma_495;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_495) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_495;
            float _fma_496;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_496) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_496;
            float _fma_497;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_497) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_497;
            float _fma_498;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_498) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_498;
            float _fma_499;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_499) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_499;
            float _fma_500;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_500) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_500;
            float _fma_501;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_501) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_501;
            float _fma_502;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_502) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_502;
            float _fma_503;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_503) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_503;
            float _fma_504;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_504) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[2]));
            dots_p_4[2] = _fma_504;
            {
                uint2 _vld_505;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_505.x), "=r"(_vld_505.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_505 = reinterpret_cast<uint32_t*>(&_vld_505);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_505[_pair]) << 16);
                    (&x_row_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_505[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_506;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_506.x), "=r"(_vld_506.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_506 = reinterpret_cast<uint32_t*>(&_vld_506);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_506[_pair]) << 16);
                    (&x_row_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_506[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_507;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_507.x), "=r"(_vld_507.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_507 = reinterpret_cast<uint32_t*>(&_vld_507);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_507[_pair]) << 16);
                    (&x_row_4[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_507[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_508;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_508.x), "=r"(_vld_508.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_508 = reinterpret_cast<uint32_t*>(&_vld_508);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_508[_pair]) << 16);
                    (&x_row_4[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_508[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_509;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_509.x), "=r"(_vld_509.y) : "l"((const void*)(X + ((ta64_4 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_509 = reinterpret_cast<uint32_t*>(&_vld_509);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_4[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_509[_pair]) << 16);
                    (&x_row_4[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_509[_pair]) & 0xffff0000u);
                }
            }
            dots_p_4[3] = mul_f32_rn_noftz(x_row_4[0], du_row_4[0]);
            float _fma_510;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_510) : "f"(x_row_4[1]), "f"(du_row_4[1]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_510;
            float _fma_511;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_511) : "f"(x_row_4[2]), "f"(du_row_4[2]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_511;
            float _fma_512;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_512) : "f"(x_row_4[3]), "f"(du_row_4[3]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_512;
            float _fma_513;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_513) : "f"(x_row_4[4]), "f"(du_row_4[4]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_513;
            float _fma_514;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_514) : "f"(x_row_4[5]), "f"(du_row_4[5]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_514;
            float _fma_515;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_515) : "f"(x_row_4[6]), "f"(du_row_4[6]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_515;
            float _fma_516;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_516) : "f"(x_row_4[7]), "f"(du_row_4[7]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_516;
            float _fma_517;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_517) : "f"(x_row_4[8]), "f"(du_row_4[8]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_517;
            float _fma_518;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_518) : "f"(x_row_4[9]), "f"(du_row_4[9]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_518;
            float _fma_519;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_519) : "f"(x_row_4[10]), "f"(du_row_4[10]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_519;
            float _fma_520;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_520) : "f"(x_row_4[11]), "f"(du_row_4[11]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_520;
            float _fma_521;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_521) : "f"(x_row_4[12]), "f"(du_row_4[12]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_521;
            float _fma_522;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_522) : "f"(x_row_4[13]), "f"(du_row_4[13]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_522;
            float _fma_523;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_523) : "f"(x_row_4[14]), "f"(du_row_4[14]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_523;
            float _fma_524;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_524) : "f"(x_row_4[15]), "f"(du_row_4[15]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_524;
            float _fma_525;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_525) : "f"(x_row_4[16]), "f"(du_row_4[16]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_525;
            float _fma_526;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_526) : "f"(x_row_4[17]), "f"(du_row_4[17]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_526;
            float _fma_527;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_527) : "f"(x_row_4[18]), "f"(du_row_4[18]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_527;
            float _fma_528;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_528) : "f"(x_row_4[19]), "f"(du_row_4[19]), "f"(dots_p_4[3]));
            dots_p_4[3] = _fma_528;
            int upper_5 = lane_1 & 16;
            float send_5 = ((upper_5 != 0) ? dots_p_4[0] : dots_p_4[2]);
            float keep_6 = ((upper_5 != 0) ? dots_p_4[2] : dots_p_4[0]);
            float _shfl_xor_sync_529 = __shfl_xor_sync(0xFFFFFFFF, send_5, 16);
            float add_9;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_9) : "f"(keep_6), "f"(_shfl_xor_sync_529));
            float send_0_4 = ((upper_5 != 0) ? dots_p_4[1] : dots_p_4[3]);
            float keep_1_4 = ((upper_5 != 0) ? dots_p_4[3] : dots_p_4[1]);
            float _shfl_xor_sync_530 = __shfl_xor_sync(0xFFFFFFFF, send_0_4, 16);
            float add_1_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_4) : "f"(keep_1_4), "f"(_shfl_xor_sync_530));
            int upper_2_4 = lane_1 & 8;
            float send_3_4 = ((upper_2_4 != 0) ? add_9 : add_1_4);
            float keep_4_4 = ((upper_2_4 != 0) ? add_1_4 : add_9);
            float _shfl_xor_sync_531 = __shfl_xor_sync(0xFFFFFFFF, send_3_4, 8);
            float add_2_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_4) : "f"(keep_4_4), "f"(_shfl_xor_sync_531));
            float _shfl_xor_sync_532 = __shfl_xor_sync(0xFFFFFFFF, add_2_4, 4);
            float add_3_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_4) : "f"(add_2_4), "f"(_shfl_xor_sync_532));
            float _shfl_xor_sync_533 = __shfl_xor_sync(0xFFFFFFFF, add_3_4, 2);
            float add_4_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_4) : "f"(add_3_4), "f"(_shfl_xor_sync_533));
            float _shfl_xor_sync_534 = __shfl_xor_sync(0xFFFFFFFF, add_4_4, 1);
            float add_5_4;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_4) : "f"(add_4_4), "f"(_shfl_xor_sync_534));
            if (lane8 == 0) {
                dpre_part[128 + warp_0 * 4 + lane_hi] = add_5_4;
            }
        }
        int tok_a_5 = g0 + 5;
        if (tok_a_5 < t1) {
            long long ta64_5 = (long long)tok_a_5;
            float du_row_5[20];
            {
                uint2 _vld_535;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_535.x), "=r"(_vld_535.y) : "l"((const void*)(dU + (ta64_5 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_535 = reinterpret_cast<uint32_t*>(&_vld_535);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_535[_pair]) << 16);
                    (&du_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_535[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_536;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_536.x), "=r"(_vld_536.y) : "l"((const void*)(dU + (ta64_5 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_536 = reinterpret_cast<uint32_t*>(&_vld_536);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_536[_pair]) << 16);
                    (&du_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_536[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_537;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_537.x), "=r"(_vld_537.y) : "l"((const void*)(dU + (ta64_5 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_537 = reinterpret_cast<uint32_t*>(&_vld_537);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_537[_pair]) << 16);
                    (&du_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_537[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_538;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_538.x), "=r"(_vld_538.y) : "l"((const void*)(dU + (ta64_5 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_538 = reinterpret_cast<uint32_t*>(&_vld_538);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_538[_pair]) << 16);
                    (&du_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_538[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_539;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_539.x), "=r"(_vld_539.y) : "l"((const void*)(dU + (ta64_5 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_539 = reinterpret_cast<uint32_t*>(&_vld_539);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_539[_pair]) << 16);
                    (&du_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_539[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_5[4];
            float x_row_5[20];
            {
                uint2 _vld_540;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_540.x), "=r"(_vld_540.y) : "l"((const void*)(X + (ta64_5 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_540 = reinterpret_cast<uint32_t*>(&_vld_540);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_540[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_540[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_541;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_541.x), "=r"(_vld_541.y) : "l"((const void*)(X + (ta64_5 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_541 = reinterpret_cast<uint32_t*>(&_vld_541);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_541[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_541[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_542;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_542.x), "=r"(_vld_542.y) : "l"((const void*)(X + (ta64_5 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_542 = reinterpret_cast<uint32_t*>(&_vld_542);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_542[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_542[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_543;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_543.x), "=r"(_vld_543.y) : "l"((const void*)(X + (ta64_5 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_543 = reinterpret_cast<uint32_t*>(&_vld_543);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_543[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_543[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_544;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_544.x), "=r"(_vld_544.y) : "l"((const void*)(X + (ta64_5 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_544 = reinterpret_cast<uint32_t*>(&_vld_544);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_544[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_544[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[0] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_545;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_545) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_545;
            float _fma_546;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_546) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_546;
            float _fma_547;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_547) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_547;
            float _fma_548;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_548) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_548;
            float _fma_549;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_549) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_549;
            float _fma_550;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_550) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_550;
            float _fma_551;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_551) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_551;
            float _fma_552;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_552) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_552;
            float _fma_553;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_553) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_553;
            float _fma_554;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_554) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_554;
            float _fma_555;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_555) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_555;
            float _fma_556;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_556) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_556;
            float _fma_557;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_557) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_557;
            float _fma_558;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_558) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_558;
            float _fma_559;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_559) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_559;
            float _fma_560;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_560) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_560;
            float _fma_561;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_561) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_561;
            float _fma_562;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_562) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_562;
            float _fma_563;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_563) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[0]));
            dots_p_5[0] = _fma_563;
            {
                uint2 _vld_564;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_564.x), "=r"(_vld_564.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_564 = reinterpret_cast<uint32_t*>(&_vld_564);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_564[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_564[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_565;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_565.x), "=r"(_vld_565.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_565 = reinterpret_cast<uint32_t*>(&_vld_565);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_565[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_565[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_566;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_566.x), "=r"(_vld_566.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_566 = reinterpret_cast<uint32_t*>(&_vld_566);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_566[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_566[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_567;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_567.x), "=r"(_vld_567.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_567 = reinterpret_cast<uint32_t*>(&_vld_567);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_567[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_567[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_568;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_568.x), "=r"(_vld_568.y) : "l"((const void*)(X + ((ta64_5 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_568 = reinterpret_cast<uint32_t*>(&_vld_568);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_568[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_568[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[1] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_569;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_569) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_569;
            float _fma_570;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_570) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_570;
            float _fma_571;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_571) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_571;
            float _fma_572;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_572) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_572;
            float _fma_573;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_573) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_573;
            float _fma_574;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_574) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_574;
            float _fma_575;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_575) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_575;
            float _fma_576;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_576) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_576;
            float _fma_577;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_577) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_577;
            float _fma_578;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_578) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_578;
            float _fma_579;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_579) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_579;
            float _fma_580;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_580) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_580;
            float _fma_581;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_581) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_581;
            float _fma_582;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_582) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_582;
            float _fma_583;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_583) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_583;
            float _fma_584;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_584) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_584;
            float _fma_585;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_585) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_585;
            float _fma_586;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_586) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_586;
            float _fma_587;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_587) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[1]));
            dots_p_5[1] = _fma_587;
            {
                uint2 _vld_588;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_588.x), "=r"(_vld_588.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_588 = reinterpret_cast<uint32_t*>(&_vld_588);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_588[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_588[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_589;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_589.x), "=r"(_vld_589.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_589 = reinterpret_cast<uint32_t*>(&_vld_589);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_589[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_589[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_590;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_590.x), "=r"(_vld_590.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_590 = reinterpret_cast<uint32_t*>(&_vld_590);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_590[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_590[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_591;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_591.x), "=r"(_vld_591.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_591 = reinterpret_cast<uint32_t*>(&_vld_591);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_591[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_591[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_592;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_592.x), "=r"(_vld_592.y) : "l"((const void*)(X + ((ta64_5 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_592 = reinterpret_cast<uint32_t*>(&_vld_592);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_592[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_592[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[2] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_593;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_593) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_593;
            float _fma_594;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_594) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_594;
            float _fma_595;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_595) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_595;
            float _fma_596;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_596) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_596;
            float _fma_597;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_597) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_597;
            float _fma_598;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_598) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_598;
            float _fma_599;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_599) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_599;
            float _fma_600;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_600) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_600;
            float _fma_601;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_601) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_601;
            float _fma_602;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_602) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_602;
            float _fma_603;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_603) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_603;
            float _fma_604;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_604) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_604;
            float _fma_605;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_605) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_605;
            float _fma_606;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_606) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_606;
            float _fma_607;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_607) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_607;
            float _fma_608;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_608) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_608;
            float _fma_609;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_609) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_609;
            float _fma_610;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_610) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_610;
            float _fma_611;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_611) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[2]));
            dots_p_5[2] = _fma_611;
            {
                uint2 _vld_612;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_612.x), "=r"(_vld_612.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_612 = reinterpret_cast<uint32_t*>(&_vld_612);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_612[_pair]) << 16);
                    (&x_row_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_612[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_613;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_613.x), "=r"(_vld_613.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_613 = reinterpret_cast<uint32_t*>(&_vld_613);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_613[_pair]) << 16);
                    (&x_row_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_613[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_614;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_614.x), "=r"(_vld_614.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_614 = reinterpret_cast<uint32_t*>(&_vld_614);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_614[_pair]) << 16);
                    (&x_row_5[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_614[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_615;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_615.x), "=r"(_vld_615.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_615 = reinterpret_cast<uint32_t*>(&_vld_615);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_615[_pair]) << 16);
                    (&x_row_5[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_615[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_616;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_616.x), "=r"(_vld_616.y) : "l"((const void*)(X + ((ta64_5 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_616 = reinterpret_cast<uint32_t*>(&_vld_616);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_5[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_616[_pair]) << 16);
                    (&x_row_5[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_616[_pair]) & 0xffff0000u);
                }
            }
            dots_p_5[3] = mul_f32_rn_noftz(x_row_5[0], du_row_5[0]);
            float _fma_617;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_617) : "f"(x_row_5[1]), "f"(du_row_5[1]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_617;
            float _fma_618;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_618) : "f"(x_row_5[2]), "f"(du_row_5[2]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_618;
            float _fma_619;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_619) : "f"(x_row_5[3]), "f"(du_row_5[3]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_619;
            float _fma_620;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_620) : "f"(x_row_5[4]), "f"(du_row_5[4]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_620;
            float _fma_621;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_621) : "f"(x_row_5[5]), "f"(du_row_5[5]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_621;
            float _fma_622;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_622) : "f"(x_row_5[6]), "f"(du_row_5[6]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_622;
            float _fma_623;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_623) : "f"(x_row_5[7]), "f"(du_row_5[7]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_623;
            float _fma_624;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_624) : "f"(x_row_5[8]), "f"(du_row_5[8]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_624;
            float _fma_625;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_625) : "f"(x_row_5[9]), "f"(du_row_5[9]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_625;
            float _fma_626;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_626) : "f"(x_row_5[10]), "f"(du_row_5[10]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_626;
            float _fma_627;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_627) : "f"(x_row_5[11]), "f"(du_row_5[11]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_627;
            float _fma_628;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_628) : "f"(x_row_5[12]), "f"(du_row_5[12]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_628;
            float _fma_629;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_629) : "f"(x_row_5[13]), "f"(du_row_5[13]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_629;
            float _fma_630;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_630) : "f"(x_row_5[14]), "f"(du_row_5[14]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_630;
            float _fma_631;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_631) : "f"(x_row_5[15]), "f"(du_row_5[15]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_631;
            float _fma_632;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_632) : "f"(x_row_5[16]), "f"(du_row_5[16]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_632;
            float _fma_633;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_633) : "f"(x_row_5[17]), "f"(du_row_5[17]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_633;
            float _fma_634;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_634) : "f"(x_row_5[18]), "f"(du_row_5[18]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_634;
            float _fma_635;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_635) : "f"(x_row_5[19]), "f"(du_row_5[19]), "f"(dots_p_5[3]));
            dots_p_5[3] = _fma_635;
            int upper_6 = lane_1 & 16;
            float send_6 = ((upper_6 != 0) ? dots_p_5[0] : dots_p_5[2]);
            float keep_7 = ((upper_6 != 0) ? dots_p_5[2] : dots_p_5[0]);
            float _shfl_xor_sync_636 = __shfl_xor_sync(0xFFFFFFFF, send_6, 16);
            float add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_10) : "f"(keep_7), "f"(_shfl_xor_sync_636));
            float send_0_5 = ((upper_6 != 0) ? dots_p_5[1] : dots_p_5[3]);
            float keep_1_5 = ((upper_6 != 0) ? dots_p_5[3] : dots_p_5[1]);
            float _shfl_xor_sync_637 = __shfl_xor_sync(0xFFFFFFFF, send_0_5, 16);
            float add_1_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_5) : "f"(keep_1_5), "f"(_shfl_xor_sync_637));
            int upper_2_5 = lane_1 & 8;
            float send_3_5 = ((upper_2_5 != 0) ? add_10 : add_1_5);
            float keep_4_5 = ((upper_2_5 != 0) ? add_1_5 : add_10);
            float _shfl_xor_sync_638 = __shfl_xor_sync(0xFFFFFFFF, send_3_5, 8);
            float add_2_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_5) : "f"(keep_4_5), "f"(_shfl_xor_sync_638));
            float _shfl_xor_sync_639 = __shfl_xor_sync(0xFFFFFFFF, add_2_5, 4);
            float add_3_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_5) : "f"(add_2_5), "f"(_shfl_xor_sync_639));
            float _shfl_xor_sync_640 = __shfl_xor_sync(0xFFFFFFFF, add_3_5, 2);
            float add_4_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_5) : "f"(add_3_5), "f"(_shfl_xor_sync_640));
            float _shfl_xor_sync_641 = __shfl_xor_sync(0xFFFFFFFF, add_4_5, 1);
            float add_5_5;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_5) : "f"(add_4_5), "f"(_shfl_xor_sync_641));
            if (lane8 == 0) {
                dpre_part[160 + warp_0 * 4 + lane_hi] = add_5_5;
            }
        }
        int tok_a_6 = g0 + 6;
        if (tok_a_6 < t1) {
            long long ta64_6 = (long long)tok_a_6;
            float du_row_6[20];
            {
                uint2 _vld_642;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_642.x), "=r"(_vld_642.y) : "l"((const void*)(dU + (ta64_6 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_642 = reinterpret_cast<uint32_t*>(&_vld_642);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_642[_pair]) << 16);
                    (&du_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_642[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_643;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_643.x), "=r"(_vld_643.y) : "l"((const void*)(dU + (ta64_6 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_643 = reinterpret_cast<uint32_t*>(&_vld_643);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_643[_pair]) << 16);
                    (&du_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_643[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_644;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_644.x), "=r"(_vld_644.y) : "l"((const void*)(dU + (ta64_6 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_644 = reinterpret_cast<uint32_t*>(&_vld_644);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_644[_pair]) << 16);
                    (&du_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_644[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_645;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_645.x), "=r"(_vld_645.y) : "l"((const void*)(dU + (ta64_6 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_645 = reinterpret_cast<uint32_t*>(&_vld_645);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_645[_pair]) << 16);
                    (&du_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_645[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_646;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_646.x), "=r"(_vld_646.y) : "l"((const void*)(dU + (ta64_6 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_646 = reinterpret_cast<uint32_t*>(&_vld_646);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_646[_pair]) << 16);
                    (&du_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_646[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_6[4];
            float x_row_6[20];
            {
                uint2 _vld_647;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_647.x), "=r"(_vld_647.y) : "l"((const void*)(X + (ta64_6 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_647 = reinterpret_cast<uint32_t*>(&_vld_647);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_647[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_647[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_648;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_648.x), "=r"(_vld_648.y) : "l"((const void*)(X + (ta64_6 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_648 = reinterpret_cast<uint32_t*>(&_vld_648);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_648[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_648[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_649;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_649.x), "=r"(_vld_649.y) : "l"((const void*)(X + (ta64_6 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_649 = reinterpret_cast<uint32_t*>(&_vld_649);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_649[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_649[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_650;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_650.x), "=r"(_vld_650.y) : "l"((const void*)(X + (ta64_6 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_650 = reinterpret_cast<uint32_t*>(&_vld_650);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_650[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_650[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_651;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_651.x), "=r"(_vld_651.y) : "l"((const void*)(X + (ta64_6 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_651 = reinterpret_cast<uint32_t*>(&_vld_651);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_651[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_651[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[0] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_652;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_652) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_652;
            float _fma_653;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_653) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_653;
            float _fma_654;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_654) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_654;
            float _fma_655;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_655) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_655;
            float _fma_656;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_656) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_656;
            float _fma_657;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_657) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_657;
            float _fma_658;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_658) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_658;
            float _fma_659;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_659) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_659;
            float _fma_660;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_660) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_660;
            float _fma_661;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_661) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_661;
            float _fma_662;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_662) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_662;
            float _fma_663;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_663) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_663;
            float _fma_664;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_664) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_664;
            float _fma_665;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_665) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_665;
            float _fma_666;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_666) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_666;
            float _fma_667;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_667) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_667;
            float _fma_668;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_668) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_668;
            float _fma_669;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_669) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_669;
            float _fma_670;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_670) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[0]));
            dots_p_6[0] = _fma_670;
            {
                uint2 _vld_671;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_671.x), "=r"(_vld_671.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_671 = reinterpret_cast<uint32_t*>(&_vld_671);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_671[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_671[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_672;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_672.x), "=r"(_vld_672.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_672 = reinterpret_cast<uint32_t*>(&_vld_672);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_672[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_672[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_673;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_673.x), "=r"(_vld_673.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_673 = reinterpret_cast<uint32_t*>(&_vld_673);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_673[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_673[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_674;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_674.x), "=r"(_vld_674.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_674 = reinterpret_cast<uint32_t*>(&_vld_674);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_674[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_674[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_675;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_675.x), "=r"(_vld_675.y) : "l"((const void*)(X + ((ta64_6 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_675 = reinterpret_cast<uint32_t*>(&_vld_675);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_675[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_675[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[1] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_676;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_676) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_676;
            float _fma_677;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_677) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_677;
            float _fma_678;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_678) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_678;
            float _fma_679;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_679) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_679;
            float _fma_680;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_680) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_680;
            float _fma_681;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_681) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_681;
            float _fma_682;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_682) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_682;
            float _fma_683;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_683) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_683;
            float _fma_684;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_684) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_684;
            float _fma_685;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_685) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_685;
            float _fma_686;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_686) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_686;
            float _fma_687;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_687) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_687;
            float _fma_688;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_688) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_688;
            float _fma_689;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_689) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_689;
            float _fma_690;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_690) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_690;
            float _fma_691;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_691) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_691;
            float _fma_692;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_692) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_692;
            float _fma_693;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_693) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_693;
            float _fma_694;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_694) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[1]));
            dots_p_6[1] = _fma_694;
            {
                uint2 _vld_695;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_695.x), "=r"(_vld_695.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_695 = reinterpret_cast<uint32_t*>(&_vld_695);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_695[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_695[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_696;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_696.x), "=r"(_vld_696.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_696 = reinterpret_cast<uint32_t*>(&_vld_696);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_696[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_696[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_697;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_697.x), "=r"(_vld_697.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_697 = reinterpret_cast<uint32_t*>(&_vld_697);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_697[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_697[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_698;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_698.x), "=r"(_vld_698.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_698 = reinterpret_cast<uint32_t*>(&_vld_698);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_698[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_698[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_699;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_699.x), "=r"(_vld_699.y) : "l"((const void*)(X + ((ta64_6 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_699 = reinterpret_cast<uint32_t*>(&_vld_699);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_699[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_699[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[2] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_700;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_700) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_700;
            float _fma_701;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_701) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_701;
            float _fma_702;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_702) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_702;
            float _fma_703;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_703) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_703;
            float _fma_704;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_704) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_704;
            float _fma_705;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_705) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_705;
            float _fma_706;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_706) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_706;
            float _fma_707;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_707) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_707;
            float _fma_708;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_708) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_708;
            float _fma_709;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_709) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_709;
            float _fma_710;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_710) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_710;
            float _fma_711;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_711) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_711;
            float _fma_712;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_712) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_712;
            float _fma_713;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_713) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_713;
            float _fma_714;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_714) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_714;
            float _fma_715;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_715) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_715;
            float _fma_716;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_716) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_716;
            float _fma_717;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_717) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_717;
            float _fma_718;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_718) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[2]));
            dots_p_6[2] = _fma_718;
            {
                uint2 _vld_719;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_719.x), "=r"(_vld_719.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_719 = reinterpret_cast<uint32_t*>(&_vld_719);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_719[_pair]) << 16);
                    (&x_row_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_719[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_720;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_720.x), "=r"(_vld_720.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_720 = reinterpret_cast<uint32_t*>(&_vld_720);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_720[_pair]) << 16);
                    (&x_row_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_720[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_721;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_721.x), "=r"(_vld_721.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_721 = reinterpret_cast<uint32_t*>(&_vld_721);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_721[_pair]) << 16);
                    (&x_row_6[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_721[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_722;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_722.x), "=r"(_vld_722.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_722 = reinterpret_cast<uint32_t*>(&_vld_722);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_722[_pair]) << 16);
                    (&x_row_6[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_722[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_723;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_723.x), "=r"(_vld_723.y) : "l"((const void*)(X + ((ta64_6 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_723 = reinterpret_cast<uint32_t*>(&_vld_723);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_6[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_723[_pair]) << 16);
                    (&x_row_6[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_723[_pair]) & 0xffff0000u);
                }
            }
            dots_p_6[3] = mul_f32_rn_noftz(x_row_6[0], du_row_6[0]);
            float _fma_724;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_724) : "f"(x_row_6[1]), "f"(du_row_6[1]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_724;
            float _fma_725;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_725) : "f"(x_row_6[2]), "f"(du_row_6[2]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_725;
            float _fma_726;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_726) : "f"(x_row_6[3]), "f"(du_row_6[3]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_726;
            float _fma_727;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_727) : "f"(x_row_6[4]), "f"(du_row_6[4]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_727;
            float _fma_728;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_728) : "f"(x_row_6[5]), "f"(du_row_6[5]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_728;
            float _fma_729;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_729) : "f"(x_row_6[6]), "f"(du_row_6[6]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_729;
            float _fma_730;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_730) : "f"(x_row_6[7]), "f"(du_row_6[7]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_730;
            float _fma_731;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_731) : "f"(x_row_6[8]), "f"(du_row_6[8]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_731;
            float _fma_732;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_732) : "f"(x_row_6[9]), "f"(du_row_6[9]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_732;
            float _fma_733;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_733) : "f"(x_row_6[10]), "f"(du_row_6[10]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_733;
            float _fma_734;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_734) : "f"(x_row_6[11]), "f"(du_row_6[11]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_734;
            float _fma_735;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_735) : "f"(x_row_6[12]), "f"(du_row_6[12]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_735;
            float _fma_736;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_736) : "f"(x_row_6[13]), "f"(du_row_6[13]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_736;
            float _fma_737;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_737) : "f"(x_row_6[14]), "f"(du_row_6[14]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_737;
            float _fma_738;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_738) : "f"(x_row_6[15]), "f"(du_row_6[15]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_738;
            float _fma_739;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_739) : "f"(x_row_6[16]), "f"(du_row_6[16]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_739;
            float _fma_740;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_740) : "f"(x_row_6[17]), "f"(du_row_6[17]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_740;
            float _fma_741;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_741) : "f"(x_row_6[18]), "f"(du_row_6[18]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_741;
            float _fma_742;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_742) : "f"(x_row_6[19]), "f"(du_row_6[19]), "f"(dots_p_6[3]));
            dots_p_6[3] = _fma_742;
            int upper_7 = lane_1 & 16;
            float send_7 = ((upper_7 != 0) ? dots_p_6[0] : dots_p_6[2]);
            float keep_8 = ((upper_7 != 0) ? dots_p_6[2] : dots_p_6[0]);
            float _shfl_xor_sync_743 = __shfl_xor_sync(0xFFFFFFFF, send_7, 16);
            float add_11;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_11) : "f"(keep_8), "f"(_shfl_xor_sync_743));
            float send_0_6 = ((upper_7 != 0) ? dots_p_6[1] : dots_p_6[3]);
            float keep_1_6 = ((upper_7 != 0) ? dots_p_6[3] : dots_p_6[1]);
            float _shfl_xor_sync_744 = __shfl_xor_sync(0xFFFFFFFF, send_0_6, 16);
            float add_1_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_6) : "f"(keep_1_6), "f"(_shfl_xor_sync_744));
            int upper_2_6 = lane_1 & 8;
            float send_3_6 = ((upper_2_6 != 0) ? add_11 : add_1_6);
            float keep_4_6 = ((upper_2_6 != 0) ? add_1_6 : add_11);
            float _shfl_xor_sync_745 = __shfl_xor_sync(0xFFFFFFFF, send_3_6, 8);
            float add_2_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_6) : "f"(keep_4_6), "f"(_shfl_xor_sync_745));
            float _shfl_xor_sync_746 = __shfl_xor_sync(0xFFFFFFFF, add_2_6, 4);
            float add_3_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_6) : "f"(add_2_6), "f"(_shfl_xor_sync_746));
            float _shfl_xor_sync_747 = __shfl_xor_sync(0xFFFFFFFF, add_3_6, 2);
            float add_4_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_6) : "f"(add_3_6), "f"(_shfl_xor_sync_747));
            float _shfl_xor_sync_748 = __shfl_xor_sync(0xFFFFFFFF, add_4_6, 1);
            float add_5_6;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_6) : "f"(add_4_6), "f"(_shfl_xor_sync_748));
            if (lane8 == 0) {
                dpre_part[192 + warp_0 * 4 + lane_hi] = add_5_6;
            }
        }
        int tok_a_7 = g0 + 7;
        if (tok_a_7 < t1) {
            long long ta64_7 = (long long)tok_a_7;
            float du_row_7[20];
            {
                uint2 _vld_749;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_749.x), "=r"(_vld_749.y) : "l"((const void*)(dU + (ta64_7 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_749 = reinterpret_cast<uint32_t*>(&_vld_749);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_749[_pair]) << 16);
                    (&du_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_749[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_750;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_750.x), "=r"(_vld_750.y) : "l"((const void*)(dU + (ta64_7 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_750 = reinterpret_cast<uint32_t*>(&_vld_750);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_750[_pair]) << 16);
                    (&du_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_750[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_751;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_751.x), "=r"(_vld_751.y) : "l"((const void*)(dU + (ta64_7 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_751 = reinterpret_cast<uint32_t*>(&_vld_751);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_751[_pair]) << 16);
                    (&du_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_751[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_752;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_752.x), "=r"(_vld_752.y) : "l"((const void*)(dU + (ta64_7 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_752 = reinterpret_cast<uint32_t*>(&_vld_752);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_752[_pair]) << 16);
                    (&du_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_752[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_753;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_753.x), "=r"(_vld_753.y) : "l"((const void*)(dU + (ta64_7 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_753 = reinterpret_cast<uint32_t*>(&_vld_753);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&du_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_753[_pair]) << 16);
                    (&du_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_753[_pair]) & 0xffff0000u);
                }
            }
            float dots_p_7[4];
            float x_row_7[20];
            {
                uint2 _vld_754;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_754.x), "=r"(_vld_754.y) : "l"((const void*)(X + (ta64_7 * 4 * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_754 = reinterpret_cast<uint32_t*>(&_vld_754);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_754[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_754[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_755;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_755.x), "=r"(_vld_755.y) : "l"((const void*)(X + (ta64_7 * 4 * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_755 = reinterpret_cast<uint32_t*>(&_vld_755);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_755[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_755[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_756;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_756.x), "=r"(_vld_756.y) : "l"((const void*)(X + (ta64_7 * 4 * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_756 = reinterpret_cast<uint32_t*>(&_vld_756);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_756[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_756[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_757;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_757.x), "=r"(_vld_757.y) : "l"((const void*)(X + (ta64_7 * 4 * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_757 = reinterpret_cast<uint32_t*>(&_vld_757);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_757[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_757[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_758;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_758.x), "=r"(_vld_758.y) : "l"((const void*)(X + (ta64_7 * 4 * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_758 = reinterpret_cast<uint32_t*>(&_vld_758);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_758[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_758[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[0] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_759;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_759) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_759;
            float _fma_760;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_760) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_760;
            float _fma_761;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_761) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_761;
            float _fma_762;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_762) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_762;
            float _fma_763;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_763) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_763;
            float _fma_764;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_764) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_764;
            float _fma_765;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_765) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_765;
            float _fma_766;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_766) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_766;
            float _fma_767;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_767) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_767;
            float _fma_768;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_768) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_768;
            float _fma_769;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_769) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_769;
            float _fma_770;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_770) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_770;
            float _fma_771;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_771) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_771;
            float _fma_772;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_772) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_772;
            float _fma_773;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_773) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_773;
            float _fma_774;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_774) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_774;
            float _fma_775;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_775) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_775;
            float _fma_776;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_776) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_776;
            float _fma_777;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_777) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[0]));
            dots_p_7[0] = _fma_777;
            {
                uint2 _vld_778;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_778.x), "=r"(_vld_778.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_778 = reinterpret_cast<uint32_t*>(&_vld_778);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_778[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_778[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_779;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_779.x), "=r"(_vld_779.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_779 = reinterpret_cast<uint32_t*>(&_vld_779);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_779[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_779[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_780;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_780.x), "=r"(_vld_780.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_780 = reinterpret_cast<uint32_t*>(&_vld_780);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_780[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_780[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_781;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_781.x), "=r"(_vld_781.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_781 = reinterpret_cast<uint32_t*>(&_vld_781);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_781[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_781[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_782;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_782.x), "=r"(_vld_782.y) : "l"((const void*)(X + ((ta64_7 * 4 + 1) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_782 = reinterpret_cast<uint32_t*>(&_vld_782);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_782[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_782[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[1] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_783;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_783) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_783;
            float _fma_784;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_784) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_784;
            float _fma_785;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_785) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_785;
            float _fma_786;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_786) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_786;
            float _fma_787;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_787) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_787;
            float _fma_788;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_788) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_788;
            float _fma_789;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_789) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_789;
            float _fma_790;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_790) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_790;
            float _fma_791;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_791) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_791;
            float _fma_792;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_792) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_792;
            float _fma_793;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_793) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_793;
            float _fma_794;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_794) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_794;
            float _fma_795;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_795) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_795;
            float _fma_796;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_796) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_796;
            float _fma_797;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_797) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_797;
            float _fma_798;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_798) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_798;
            float _fma_799;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_799) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_799;
            float _fma_800;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_800) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_800;
            float _fma_801;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_801) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[1]));
            dots_p_7[1] = _fma_801;
            {
                uint2 _vld_802;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_802.x), "=r"(_vld_802.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_802 = reinterpret_cast<uint32_t*>(&_vld_802);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_802[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_802[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_803;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_803.x), "=r"(_vld_803.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_803 = reinterpret_cast<uint32_t*>(&_vld_803);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_803[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_803[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_804;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_804.x), "=r"(_vld_804.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_804 = reinterpret_cast<uint32_t*>(&_vld_804);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_804[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_804[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_805;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_805.x), "=r"(_vld_805.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_805 = reinterpret_cast<uint32_t*>(&_vld_805);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_805[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_805[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_806;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_806.x), "=r"(_vld_806.y) : "l"((const void*)(X + ((ta64_7 * 4 + 2) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_806 = reinterpret_cast<uint32_t*>(&_vld_806);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_806[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_806[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[2] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_807;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_807) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_807;
            float _fma_808;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_808) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_808;
            float _fma_809;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_809) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_809;
            float _fma_810;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_810) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_810;
            float _fma_811;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_811) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_811;
            float _fma_812;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_812) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_812;
            float _fma_813;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_813) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_813;
            float _fma_814;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_814) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_814;
            float _fma_815;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_815) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_815;
            float _fma_816;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_816) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_816;
            float _fma_817;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_817) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_817;
            float _fma_818;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_818) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_818;
            float _fma_819;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_819) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_819;
            float _fma_820;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_820) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_820;
            float _fma_821;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_821) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_821;
            float _fma_822;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_822) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_822;
            float _fma_823;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_823) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_823;
            float _fma_824;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_824) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_824;
            float _fma_825;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_825) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[2]));
            dots_p_7[2] = _fma_825;
            {
                uint2 _vld_826;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_826.x), "=r"(_vld_826.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 5120 + col_off) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_826 = reinterpret_cast<uint32_t*>(&_vld_826);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_826[_pair]) << 16);
                    (&x_row_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_826[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_827;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_827.x), "=r"(_vld_827.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 5120 + col_off + 4) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_827 = reinterpret_cast<uint32_t*>(&_vld_827);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_827[_pair]) << 16);
                    (&x_row_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_827[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_828;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_828.x), "=r"(_vld_828.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 5120 + col_off + 8) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_828 = reinterpret_cast<uint32_t*>(&_vld_828);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_828[_pair]) << 16);
                    (&x_row_7[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_828[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_829;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_829.x), "=r"(_vld_829.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 5120 + col_off + 12) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_829 = reinterpret_cast<uint32_t*>(&_vld_829);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_829[_pair]) << 16);
                    (&x_row_7[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_829[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_830;
                asm volatile("ld.global.L2::cache_hint.v2.b32 {%0, %1}, [%2], %3;"
                    : "=r"(_vld_830.x), "=r"(_vld_830.y) : "l"((const void*)(X + ((ta64_7 * 4 + 3) * 5120 + col_off + 16) + 0)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_830 = reinterpret_cast<uint32_t*>(&_vld_830);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&x_row_7[16 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_830[_pair]) << 16);
                    (&x_row_7[16 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_830[_pair]) & 0xffff0000u);
                }
            }
            dots_p_7[3] = mul_f32_rn_noftz(x_row_7[0], du_row_7[0]);
            float _fma_831;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_831) : "f"(x_row_7[1]), "f"(du_row_7[1]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_831;
            float _fma_832;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_832) : "f"(x_row_7[2]), "f"(du_row_7[2]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_832;
            float _fma_833;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_833) : "f"(x_row_7[3]), "f"(du_row_7[3]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_833;
            float _fma_834;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_834) : "f"(x_row_7[4]), "f"(du_row_7[4]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_834;
            float _fma_835;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_835) : "f"(x_row_7[5]), "f"(du_row_7[5]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_835;
            float _fma_836;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_836) : "f"(x_row_7[6]), "f"(du_row_7[6]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_836;
            float _fma_837;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_837) : "f"(x_row_7[7]), "f"(du_row_7[7]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_837;
            float _fma_838;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_838) : "f"(x_row_7[8]), "f"(du_row_7[8]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_838;
            float _fma_839;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_839) : "f"(x_row_7[9]), "f"(du_row_7[9]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_839;
            float _fma_840;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_840) : "f"(x_row_7[10]), "f"(du_row_7[10]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_840;
            float _fma_841;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_841) : "f"(x_row_7[11]), "f"(du_row_7[11]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_841;
            float _fma_842;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_842) : "f"(x_row_7[12]), "f"(du_row_7[12]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_842;
            float _fma_843;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_843) : "f"(x_row_7[13]), "f"(du_row_7[13]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_843;
            float _fma_844;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_844) : "f"(x_row_7[14]), "f"(du_row_7[14]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_844;
            float _fma_845;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_845) : "f"(x_row_7[15]), "f"(du_row_7[15]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_845;
            float _fma_846;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_846) : "f"(x_row_7[16]), "f"(du_row_7[16]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_846;
            float _fma_847;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_847) : "f"(x_row_7[17]), "f"(du_row_7[17]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_847;
            float _fma_848;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_848) : "f"(x_row_7[18]), "f"(du_row_7[18]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_848;
            float _fma_849;
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_849) : "f"(x_row_7[19]), "f"(du_row_7[19]), "f"(dots_p_7[3]));
            dots_p_7[3] = _fma_849;
            int upper_8 = lane_1 & 16;
            float send_8 = ((upper_8 != 0) ? dots_p_7[0] : dots_p_7[2]);
            float keep_9 = ((upper_8 != 0) ? dots_p_7[2] : dots_p_7[0]);
            float _shfl_xor_sync_850 = __shfl_xor_sync(0xFFFFFFFF, send_8, 16);
            float add_12;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_12) : "f"(keep_9), "f"(_shfl_xor_sync_850));
            float send_0_7 = ((upper_8 != 0) ? dots_p_7[1] : dots_p_7[3]);
            float keep_1_7 = ((upper_8 != 0) ? dots_p_7[3] : dots_p_7[1]);
            float _shfl_xor_sync_851 = __shfl_xor_sync(0xFFFFFFFF, send_0_7, 16);
            float add_1_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_7) : "f"(keep_1_7), "f"(_shfl_xor_sync_851));
            int upper_2_7 = lane_1 & 8;
            float send_3_7 = ((upper_2_7 != 0) ? add_12 : add_1_7);
            float keep_4_7 = ((upper_2_7 != 0) ? add_1_7 : add_12);
            float _shfl_xor_sync_852 = __shfl_xor_sync(0xFFFFFFFF, send_3_7, 8);
            float add_2_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_7) : "f"(keep_4_7), "f"(_shfl_xor_sync_852));
            float _shfl_xor_sync_853 = __shfl_xor_sync(0xFFFFFFFF, add_2_7, 4);
            float add_3_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_7) : "f"(add_2_7), "f"(_shfl_xor_sync_853));
            float _shfl_xor_sync_854 = __shfl_xor_sync(0xFFFFFFFF, add_3_7, 2);
            float add_4_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_7) : "f"(add_3_7), "f"(_shfl_xor_sync_854));
            float _shfl_xor_sync_855 = __shfl_xor_sync(0xFFFFFFFF, add_4_7, 1);
            float add_5_7;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_5_7) : "f"(add_4_7), "f"(_shfl_xor_sync_855));
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
        float _sub_856;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_856) : "f"(0.0f), "f"(fabsf(a)));
        float _exp_0 = __nv_expf(_sub_856);
        float _add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp_0));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _add_0);
        float _fdiv_rn_1 = __fdiv_rn(_exp_0, _add_0);
        float sig = ((a >= 0.0f) ? _fdiv_rn_0 : _fdiv_rn_1);
        float sneg = ((a >= 0.0f) ? _fdiv_rn_1 : _fdiv_rn_0);
        float _add_1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(sig), "f"(hc_eps));
        float logit = ((is_logit) ? a : 0.0f);
        float _shfl_xor_sync_857 = __shfl_xor_sync(0xFFFFFFFF, logit, 1);
        float _max_0 = fmaxf(logit, _shfl_xor_sync_857);
        float _shfl_xor_sync_859 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 2);
        float _sub_858;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_858) : "f"(logit), "f"(fmaxf(_max_0, _shfl_xor_sync_859)));
        float _exp_1 = __nv_expf(_sub_858);
        float _shfl_xor_sync_860 = __shfl_xor_sync(0xFFFFFFFF, _exp_1, 1);
        float add_13;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_13) : "f"(_exp_1), "f"(_shfl_xor_sync_860));
        float _shfl_xor_sync_862 = __shfl_xor_sync(0xFFFFFFFF, add_13, 2);
        float _add_861;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_861) : "f"(add_13), "f"(_shfl_xor_sync_862));
        float S = __fdiv_rn(_exp_1, _add_861);
        float C0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(C0) : "f"(S), "f"(hc_eps));
        float _shfl_xor_sync_863 = __shfl_xor_sync(0xFFFFFFFF, C0, 4);
        float s1;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(s1) : "f"(C0), "f"(_shfl_xor_sync_863));
        float _shfl_xor_sync_866 = __shfl_xor_sync(0xFFFFFFFF, s1, 24);
        float _add_865;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_865) : "f"(s1), "f"(_shfl_xor_sync_866));
        float _add_864;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_864) : "f"(_add_865), "f"(hc_eps));
        float c = __fdiv_rn(C0, _add_864);
        if (is_logit) {
            hist_s[hb] = c;
        }
        #pragma unroll 1
        for (int p = 0; p < n_pairs; p++) {
            float _shfl_xor_sync_867 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
            float sr;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sr) : "f"(c), "f"(_shfl_xor_sync_867));
            float _shfl_xor_sync_871 = __shfl_xor_sync(0xFFFFFFFF, sr, 2);
            float _add_870;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_870) : "f"(sr), "f"(_shfl_xor_sync_871));
            float _add_869;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_869) : "f"(_add_870), "f"(hc_eps));
            float _fdiv_868 = __fdiv_rn(c, _add_869);
            c = _fdiv_868;
            if (is_logit) {
                hist_s[(2 * p + 1) * 128 + hb] = c;
            }
            float _shfl_xor_sync_872 = __shfl_xor_sync(0xFFFFFFFF, c, 4);
            float sk;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sk) : "f"(c), "f"(_shfl_xor_sync_872));
            float _shfl_xor_sync_876 = __shfl_xor_sync(0xFFFFFFFF, sk, 24);
            float _add_875;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_875) : "f"(sk), "f"(_shfl_xor_sync_876));
            float _add_874;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_874) : "f"(_add_875), "f"(hc_eps));
            float _fdiv_873 = __fdiv_rn(c, _add_874);
            c = _fdiv_873;
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
            float _shfl_xor_sync_877 = __shfl_xor_sync(0xFFFFFFFF, hp, 4);
            float sp;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sp) : "f"(hp), "f"(_shfl_xor_sync_877));
            float _shfl_xor_sync_879 = __shfl_xor_sync(0xFFFFFFFF, sp, 24);
            float _add_878;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_878) : "f"(sp), "f"(_shfl_xor_sync_879));
            float vp;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vp) : "f"(_add_878), "f"(hc_eps));
            float mp;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(mp) : "f"(G), "f"(cur));
            float _shfl_xor_sync_880 = __shfl_xor_sync(0xFFFFFFFF, mp, 4);
            float ipp;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ipp) : "f"(mp), "f"(_shfl_xor_sync_880));
            float _shfl_xor_sync_884 = __shfl_xor_sync(0xFFFFFFFF, ipp, 24);
            float _add_883;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_883) : "f"(ipp), "f"(_shfl_xor_sync_884));
            float _sub_882;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_882) : "f"(G), "f"(_add_883));
            float _fdiv_881 = __fdiv_rn(_sub_882, vp);
            G = _fdiv_881;
            float hq = hist_s[2 * pb * ld_step + ld_base];
            float _shfl_xor_sync_885 = __shfl_xor_sync(0xFFFFFFFF, hq, 1);
            float sq;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(sq) : "f"(hq), "f"(_shfl_xor_sync_885));
            float _shfl_xor_sync_887 = __shfl_xor_sync(0xFFFFFFFF, sq, 2);
            float _add_886;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_886) : "f"(sq), "f"(_shfl_xor_sync_887));
            float vq;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(vq) : "f"(_add_886), "f"(hc_eps));
            float mq;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(mq) : "f"(G), "f"(hp));
            float _shfl_xor_sync_888 = __shfl_xor_sync(0xFFFFFFFF, mq, 1);
            float ipq;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(ipq) : "f"(mq), "f"(_shfl_xor_sync_888));
            float _shfl_xor_sync_892 = __shfl_xor_sync(0xFFFFFFFF, ipq, 2);
            float _add_891;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_891) : "f"(ipq), "f"(_shfl_xor_sync_892));
            float _sub_890;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_890) : "f"(G), "f"(_add_891));
            float _fdiv_889 = __fdiv_rn(_sub_890, vq);
            G = _fdiv_889;
            cur = hq;
        }
        float _add_195;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_195) : "f"(S), "f"(hc_eps));
        float _shfl_xor_sync_893 = __shfl_xor_sync(0xFFFFFFFF, _add_195, 4);
        float add_0;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_0) : "f"(_add_195), "f"(_shfl_xor_sync_893));
        float _shfl_xor_sync_895 = __shfl_xor_sync(0xFFFFFFFF, add_0, 24);
        float _add_894;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_894) : "f"(add_0), "f"(_shfl_xor_sync_895));
        float _add_197;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(_add_894), "f"(hc_eps));
        float _mul_39;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(G), "f"(cur));
        float _shfl_xor_sync_896 = __shfl_xor_sync(0xFFFFFFFF, _mul_39, 4);
        float add_1_8;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1_8) : "f"(_mul_39), "f"(_shfl_xor_sync_896));
        float _shfl_xor_sync_899 = __shfl_xor_sync(0xFFFFFFFF, add_1_8, 24);
        float _add_898;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_898) : "f"(add_1_8), "f"(_shfl_xor_sync_899));
        float _sub_897;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_897) : "f"(G), "f"(_add_898));
        float g1 = __fdiv_rn(_sub_897, _add_197);
        float acc_162 = 0.0f;
        float _shfl_xor_sync_900 = __shfl_xor_sync(0xFFFFFFFF, S, 2);
        float _shfl_xor_sync_902 = __shfl_xor_sync(0xFFFFFFFF, g1, 2);
        float _sub_901;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_901) : "f"(g1), "f"(_shfl_xor_sync_902));
        float _shfl_xor_sync_904 = __shfl_xor_sync(0xFFFFFFFF, S, 1);
        float _shfl_xor_sync_906 = __shfl_xor_sync(0xFFFFFFFF, g1, 1);
        float _sub_905;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_905) : "f"(g1), "f"(_shfl_xor_sync_906));
        float _fma_903;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_903) : "f"(_shfl_xor_sync_904), "f"(_sub_905), "f"(acc_162));
        float _fma_0;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(_shfl_xor_sync_900), "f"(_sub_901), "f"(_fma_903));
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, S, 3);
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, g1, 3);
        float dsig;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dsig) : "f"(sig), "f"(sneg));
        float _sub_908;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_908) : "f"(g1), "f"(_shfl_xor_1));
        float _fma_907;
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_907) : "f"(_shfl_xor_0), "f"(_sub_908), "f"(_fma_0));
        float da = ((is_logit) ? mul_f32_rn_noftz(S, _fma_907) : mul_f32_rn_noftz(1.0f, dsig));
        dp[0] = dpre_part[warp_0 * 32 + lc4];
        float _add_909;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_909) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 4 + lc4]));
        dp[0] = _add_909;
        float _add_910;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_910) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 8 + lc4]));
        dp[0] = _add_910;
        float _add_911;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_911) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 12 + lc4]));
        dp[0] = _add_911;
        float _add_912;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_912) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 16 + lc4]));
        dp[0] = _add_912;
        float _add_913;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_913) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 20 + lc4]));
        dp[0] = _add_913;
        float _add_914;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_914) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 24 + lc4]));
        dp[0] = _add_914;
        float _add_915;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_915) : "f"(dp[0]), "f"(dpre_part[warp_0 * 32 + 28 + lc4]));
        dp[0] = _add_915;
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
        float c_t = (float)((double)rr * (double)_warp_reduce_0 * 4.8828125e-05);
        float dzr;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(dzr) : "f"(dz), "f"(r));
        float cvt_f32 = (float)(__nv_bfloat16)dzr;
        float sub;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(dzr), "f"(cvt_f32));
        float cvt_f32_1 = (float)(__nv_bfloat16)sub;
        float _sub_916;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_916) : "f"(sub), "f"(cvt_f32_1));
        float cvt_f32_2 = (float)(__nv_bfloat16)_sub_916;
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
        float _sub_918;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_918) : "f"(_add_2), "f"(acc[0]));
        float _sub_917;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_917) : "f"(_sub_918), "f"(_sub_0));
        comp[0] = ((tok < t1) ? _sub_917 : comp[0]);
        acc[0] = ((tok < t1) ? _add_2 : acc[0]);
        float _sub_1;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(daz), "f"(comp[1]));
        float _add_3;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(acc[1]), "f"(_sub_1));
        float _sub_920;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_920) : "f"(_add_3), "f"(acc[1]));
        float _sub_919;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_919) : "f"(_sub_920), "f"(_sub_1));
        comp[1] = ((tok < t1) ? _sub_919 : comp[1]);
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
            float _sub_922;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_922) : "f"(add_14), "f"(o[0]));
            float _sub_921;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_921) : "f"(_sub_922), "f"(sub_1));
            oc[0] = _sub_921;
            o[0] = add_14;
            float _sub_923;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_923) : "f"(0.0f), "f"(fold[src + 1]));
            float sub_0;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_0) : "f"(_sub_923), "f"(oc[0]));
            float add_2_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2_8) : "f"(o[0]), "f"(sub_0));
            float _sub_925;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_925) : "f"(add_2_8), "f"(o[0]));
            float _sub_924;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_924) : "f"(_sub_925), "f"(sub_0));
            oc[0] = _sub_924;
            o[0] = add_2_8;
            float sub_1_1;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1_1) : "f"(fold[src + 2]), "f"(oc[1]));
            float add_3_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3_8) : "f"(o[1]), "f"(sub_1_1));
            float _sub_927;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_927) : "f"(add_3_8), "f"(o[1]));
            float _sub_926;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_926) : "f"(_sub_927), "f"(sub_1_1));
            oc[1] = _sub_926;
            o[1] = add_3_8;
            float _sub_928;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_928) : "f"(0.0f), "f"(fold[src + 3]));
            float sub_2;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(_sub_928), "f"(oc[1]));
            float add_4_8;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_4_8) : "f"(o[1]), "f"(sub_2));
            float _sub_930;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_930) : "f"(add_4_8), "f"(o[1]));
            float _sub_929;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_929) : "f"(_sub_930), "f"(sub_2));
            oc[1] = _sub_929;
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
                float _sub_932;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_932) : "f"(add_15), "f"(ga[0]));
                float _sub_931;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_931) : "f"(_sub_932), "f"(sub_3));
                gc[0] = _sub_931;
                ga[0] = add_15;
            }
            float _sub_933;
            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_933) : "f"(ga[0]), "f"(gc[0]));
            *(reinterpret_cast<float*>(param_partial + (prow + 24 + lane64)) + (0)) = _sub_933;
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
