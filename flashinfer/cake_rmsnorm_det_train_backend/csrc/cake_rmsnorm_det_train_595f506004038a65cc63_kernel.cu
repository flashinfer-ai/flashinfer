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
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 16
#define SMEM_PARTIALS_STRIDE 16
#define SMEM_W_SM_OFF 128
#define SMEM_W_SM_STAGE_BYTES 8192
#define SMEM_W_SM_STRIDE 8192
#define SMEM_TOTAL 8320
#define THREADS 64
#define ROWS_PER_CTA 1
#define GROUP_THREADS 64
#define LAUNCH_BOUNDS_THREADS 64

#include <math_constants.h>


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

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_rmsnorm_det_train_595f506004038a65cc63(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;
    float* w_sm = reinterpret_cast<float*>(smem_raw + 128);
    const int w_sm_addr = smem + 128;

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int group = tid / GROUP_THREADS;
    int gtid = tid - group * GROUP_THREADS;
    float h_f = 2048.0f;
    int it = 0;
    int stride = num_bids * ROWS_PER_CTA;
    unsigned int cur[16];
    unsigned int nxt[16];
    int first = bid * ROWS_PER_CTA + group;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        cur[i] = 0;
    }
    if (first < T) {
        long long fbase = (long long)first * x_stride_p0;
        #pragma unroll
        for (int v = 0; v < 2; v++) {
            {
                asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(cur[v * 8 + 0]), "=r"(cur[v * 8 + 1]), "=r"(cur[v * 8 + 2]), "=r"(cur[v * 8 + 3]), "=r"(cur[v * 8 + 4]), "=r"(cur[v * 8 + 5]), "=r"(cur[v * 8 + 6]), "=r"(cur[v * 8 + 7]) : "l"((const void*)((const char*)(x + (fbase + (long long)(gtid * 16 + v * (GROUP_THREADS * 16))) + 0) + 0)) : "memory");
            }
        }
    }
    int staged = 0;
    if (num_bids < T) {
        staged = 1;
    }
    if (staged == 1) {
        if (group == 0) {
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                float wst[16];
                {
                    unsigned _ldv8_1_0;
                    unsigned _ldv8_1_1;
                    unsigned _ldv8_1_2;
                    unsigned _ldv8_1_3;
                    unsigned _ldv8_1_4;
                    unsigned _ldv8_1_5;
                    unsigned _ldv8_1_6;
                    unsigned _ldv8_1_7;
                    asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + (gtid * 16 + v_1 * (GROUP_THREADS * 16)) + (0))) : "memory");
                    wst[0 + 0] = __uint_as_float(_ldv8_1_0);
                    wst[0 + 1] = __uint_as_float(_ldv8_1_1);
                    wst[0 + 2] = __uint_as_float(_ldv8_1_2);
                    wst[0 + 3] = __uint_as_float(_ldv8_1_3);
                    wst[0 + 4] = __uint_as_float(_ldv8_1_4);
                    wst[0 + 5] = __uint_as_float(_ldv8_1_5);
                    wst[0 + 6] = __uint_as_float(_ldv8_1_6);
                    wst[0 + 7] = __uint_as_float(_ldv8_1_7);
                }
                {
                    unsigned _ldv8_2_0;
                    unsigned _ldv8_2_1;
                    unsigned _ldv8_2_2;
                    unsigned _ldv8_2_3;
                    unsigned _ldv8_2_4;
                    unsigned _ldv8_2_5;
                    unsigned _ldv8_2_6;
                    unsigned _ldv8_2_7;
                    asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(w + (gtid * 16 + v_1 * (GROUP_THREADS * 16) + 8) + (0))) : "memory");
                    wst[8 + 0] = __uint_as_float(_ldv8_2_0);
                    wst[8 + 1] = __uint_as_float(_ldv8_2_1);
                    wst[8 + 2] = __uint_as_float(_ldv8_2_2);
                    wst[8 + 3] = __uint_as_float(_ldv8_2_3);
                    wst[8 + 4] = __uint_as_float(_ldv8_2_4);
                    wst[8 + 5] = __uint_as_float(_ldv8_2_5);
                    wst[8 + 6] = __uint_as_float(_ldv8_2_6);
                    wst[8 + 7] = __uint_as_float(_ldv8_2_7);
                }
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(w_sm_addr + (unsigned int)(((v_1 * 4 + q) * GROUP_THREADS + gtid) * 16)), "f"(wst[4 * q]), "f"(wst[4 * q + 1]), "f"(wst[4 * q + 2]), "f"(wst[4 * q + 3]) : "memory");
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll 1
    for (int row0 = bid * ROWS_PER_CTA; row0 < T; row0 += num_bids * ROWS_PER_CTA) {
        int row = row0 + group;
        int nrow = row + stride;
        if (nrow < T) {
            long long nbase = (long long)nrow * x_stride_p0;
            #pragma unroll
            for (int v_2 = 0; v_2 < 2; v_2++) {
                {
                    asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(nxt[v_2 * 8 + 0]), "=r"(nxt[v_2 * 8 + 1]), "=r"(nxt[v_2 * 8 + 2]), "=r"(nxt[v_2 * 8 + 3]), "=r"(nxt[v_2 * 8 + 4]), "=r"(nxt[v_2 * 8 + 5]), "=r"(nxt[v_2 * 8 + 6]), "=r"(nxt[v_2 * 8 + 7]) : "l"((const void*)((const char*)(x + (nbase + (long long)(gtid * 16 + v_2 * (GROUP_THREADS * 16))) + 0) + 0)) : "memory");
                }
            }
        }
        float sum_sq = 0.0f;
        #pragma unroll
        for (int i_1 = 0; i_1 < 16; i_1++) {
            float _dot_4;
            asm volatile(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_4) : "r"(cur[i_1]), "r"(cur[i_1]), "f"(sum_sq));
            sum_sq = _dot_4;
        }
        float _warp_reduce_5 = sum_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
        sum_sq = _warp_reduce_5;
        int par = (it & 1) * 2;
        if (lane == 0) {
            partials[par + warp_0] = sum_sq;
        }
        __syncthreads();
        float total = partials[par + group * 2];
        #pragma unroll
        for (int k = 1; k < 2; k++) {
            total += partials[par + group * 2 + k];
        }
        it += 1;
        float mean = total * 0.00048828125f;
        float _sqrt_6;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_6) : "f"(mean + eps));
        float rstd = __frcp_rn(_sqrt_6);
        if (row < T) {
            if (gtid == 0) {
                r[row] = rstd;
            }
            long long obase = (long long)row * 2048;
            #pragma unroll
            for (int v_3 = 0; v_3 < 2; v_3++) {
                float wf[16];
                if (staged == 1) {
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 4; q_1++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wf[4 * q_1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(4 * q_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(4 * q_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wf[(4 * q_1) + 3]))
                            : "r"(w_sm_addr + (unsigned int)(((v_3 * 4 + q_1) * GROUP_THREADS + gtid) * 16)));
                    }
                } else {
                    {
                        unsigned _ldv8_7_0;
                        unsigned _ldv8_7_1;
                        unsigned _ldv8_7_2;
                        unsigned _ldv8_7_3;
                        unsigned _ldv8_7_4;
                        unsigned _ldv8_7_5;
                        unsigned _ldv8_7_6;
                        unsigned _ldv8_7_7;
                        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_7_0), "=r"(_ldv8_7_1), "=r"(_ldv8_7_2), "=r"(_ldv8_7_3), "=r"(_ldv8_7_4), "=r"(_ldv8_7_5), "=r"(_ldv8_7_6), "=r"(_ldv8_7_7) : "l"((const void*)(w + (gtid * 16 + v_3 * (GROUP_THREADS * 16)) + (0))) : "memory");
                        wf[0 + 0] = __uint_as_float(_ldv8_7_0);
                        wf[0 + 1] = __uint_as_float(_ldv8_7_1);
                        wf[0 + 2] = __uint_as_float(_ldv8_7_2);
                        wf[0 + 3] = __uint_as_float(_ldv8_7_3);
                        wf[0 + 4] = __uint_as_float(_ldv8_7_4);
                        wf[0 + 5] = __uint_as_float(_ldv8_7_5);
                        wf[0 + 6] = __uint_as_float(_ldv8_7_6);
                        wf[0 + 7] = __uint_as_float(_ldv8_7_7);
                    }
                    {
                        unsigned _ldv8_8_0;
                        unsigned _ldv8_8_1;
                        unsigned _ldv8_8_2;
                        unsigned _ldv8_8_3;
                        unsigned _ldv8_8_4;
                        unsigned _ldv8_8_5;
                        unsigned _ldv8_8_6;
                        unsigned _ldv8_8_7;
                        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_8_0), "=r"(_ldv8_8_1), "=r"(_ldv8_8_2), "=r"(_ldv8_8_3), "=r"(_ldv8_8_4), "=r"(_ldv8_8_5), "=r"(_ldv8_8_6), "=r"(_ldv8_8_7) : "l"((const void*)(w + (gtid * 16 + v_3 * (GROUP_THREADS * 16) + 8) + (0))) : "memory");
                        wf[8 + 0] = __uint_as_float(_ldv8_8_0);
                        wf[8 + 1] = __uint_as_float(_ldv8_8_1);
                        wf[8 + 2] = __uint_as_float(_ldv8_8_2);
                        wf[8 + 3] = __uint_as_float(_ldv8_8_3);
                        wf[8 + 4] = __uint_as_float(_ldv8_8_4);
                        wf[8 + 5] = __uint_as_float(_ldv8_8_5);
                        wf[8 + 6] = __uint_as_float(_ldv8_8_6);
                        wf[8 + 7] = __uint_as_float(_ldv8_8_7);
                    }
                }
                float out[16];
                #pragma unroll
                for (int k_1 = 0; k_1 < 8; k_1++) {
                    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(cur[v_3 * 8 + k_1]));
                    out[2 * k_1] = _cvt_f32_0.x * rstd * wf[2 * k_1];
                    out[2 * k_1 + 1] = _cvt_f32_0.y * rstd * wf[2 * k_1 + 1];
                }
                unsigned int _packed_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 8], out[_lp*2+1 + 8]));
                    _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 10], out[_lp*2+1 + 10]));
                    _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 12], out[_lp*2+1 + 12]));
                    _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 14], out[_lp*2+1 + 14]));
                    _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(_packed_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(y + (obase + (long long)(gtid * 16 + v_3 * (GROUP_THREADS * 16))))), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
                }
            }
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 16; i_2++) {
            cur[i_2] = nxt[i_2];
        }
    }
}

} // extern "C"
