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
#define SMEM_PARTIALS_STAGE_BYTES 32
#define SMEM_PARTIALS_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 64
#define ROWS_PER_CTA 2
#define GROUP_THREADS 64
#define LAUNCH_BOUNDS_THREADS 64

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_rmsnorm_det_train_a72c6ece1900f748eaa2(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, float* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
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

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int group = tid / GROUP_THREADS;
    int gtid = tid - group * GROUP_THREADS;
    float h_f = 2048.0f;
    int it = 0;
    #pragma unroll 1
    for (int row0 = bid * ROWS_PER_CTA; row0 < T; row0 += num_bids * ROWS_PER_CTA) {
        float x_cache[64];
        float sums[2];
        float zero = 0.0f;
        #pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            sums[rr] = zero;
        }
        #pragma unroll
        for (int rr_1 = 0; rr_1 < 2; rr_1++) {
            int rowa = row0 + group * 2 + rr_1;
            if (rowa < T) {
                long long row64a = (long long)rowa;
                long long xbasea = row64a * x_stride_p0;
                #pragma unroll
                for (int v = 0; v < 2; v++) {
                    float _vec_load_0[16];
                    {
                        const void* _v8p_0 = (const void*)(x + (xbasea + (long long)(gtid * 16 + v * (GROUP_THREADS * 16))) + (0));
                        uint32_t _v8_0_0[8];
                        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                            : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const void*)((const char*)_v8p_0 + 0)), "l"(0x12F0000000000000ULL) : "memory");
                        (&_vec_load_0[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) << 16);
                        (&_vec_load_0[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) & 0xffff0000u);
                        (&_vec_load_0[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) << 16);
                        (&_vec_load_0[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) & 0xffff0000u);
                        (&_vec_load_0[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) << 16);
                        (&_vec_load_0[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) & 0xffff0000u);
                        (&_vec_load_0[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) << 16);
                        (&_vec_load_0[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) & 0xffff0000u);
                        (&_vec_load_0[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) << 16);
                        (&_vec_load_0[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) & 0xffff0000u);
                        (&_vec_load_0[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) << 16);
                        (&_vec_load_0[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) & 0xffff0000u);
                        (&_vec_load_0[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) << 16);
                        (&_vec_load_0[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) & 0xffff0000u);
                        (&_vec_load_0[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) << 16);
                        (&_vec_load_0[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        float value = _vec_load_0[j];
                        x_cache[rr_1 * 32 + v * 16 + j] = value;
                    }
                }
            }
        }
        #pragma unroll
        for (int rr_2 = 0; rr_2 < 2; rr_2++) {
            int rowb = row0 + group * 2 + rr_2;
            if (rowb < T) {
                float acc = 0.0f;
                #pragma unroll
                for (int v_1 = 0; v_1 < 2; v_1++) {
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 16; j_1++) {
                        acc = __fmaf_rn(x_cache[rr_2 * 32 + v_1 * 16 + j_1], x_cache[rr_2 * 32 + v_1 * 16 + j_1], acc);
                    }
                }
                sums[rr_2] = acc;
            }
        }
        int par = (it & 1) * 4;
        #pragma unroll
        for (int rr_3 = 0; rr_3 < 2; rr_3++) {
            float red = sums[rr_3];
            float _warp_reduce_1 = red;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            red = _warp_reduce_1;
            if (lane == 0) {
                partials[par + warp_0 * 2 + rr_3] = red;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int rr_4 = 0; rr_4 < 2; rr_4++) {
            float tot = partials[par + group * 2 * 2 + rr_4];
            #pragma unroll
            for (int k = 1; k < 2; k++) {
                tot += partials[par + (group * 2 + k) * 2 + rr_4];
            }
            sums[rr_4] = tot;
        }
        it += 1;
        #pragma unroll
        for (int rr_5 = 0; rr_5 < 2; rr_5++) {
            float total_r = sums[rr_5];
            float mean_r = total_r * 0.00048828125f;
            float _sqrt_2;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_2) : "f"(mean_r + eps));
            float rstd_r = __frcp_rn(_sqrt_2);
            int rowc = row0 + group * 2 + rr_5;
            if (rowc < T) {
                long long row64c = (long long)rowc;
                long long obasec = row64c * 2048;
                if (gtid == 0) {
                    r[rowc] = rstd_r;
                }
                #pragma unroll
                for (int v_2 = 0; v_2 < 2; v_2++) {
                    float _vec_load_1[16];
                    {
                        unsigned _ldv8_3_0;
                        unsigned _ldv8_3_1;
                        unsigned _ldv8_3_2;
                        unsigned _ldv8_3_3;
                        unsigned _ldv8_3_4;
                        unsigned _ldv8_3_5;
                        unsigned _ldv8_3_6;
                        unsigned _ldv8_3_7;
                        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(w + (gtid * 16 + v_2 * (GROUP_THREADS * 16)) + (0))) : "memory");
                        _vec_load_1[0 + 0] = __uint_as_float(_ldv8_3_0);
                        _vec_load_1[0 + 1] = __uint_as_float(_ldv8_3_1);
                        _vec_load_1[0 + 2] = __uint_as_float(_ldv8_3_2);
                        _vec_load_1[0 + 3] = __uint_as_float(_ldv8_3_3);
                        _vec_load_1[0 + 4] = __uint_as_float(_ldv8_3_4);
                        _vec_load_1[0 + 5] = __uint_as_float(_ldv8_3_5);
                        _vec_load_1[0 + 6] = __uint_as_float(_ldv8_3_6);
                        _vec_load_1[0 + 7] = __uint_as_float(_ldv8_3_7);
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
                        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(w + (gtid * 16 + v_2 * (GROUP_THREADS * 16) + 8) + (0))) : "memory");
                        _vec_load_1[8 + 0] = __uint_as_float(_ldv8_4_0);
                        _vec_load_1[8 + 1] = __uint_as_float(_ldv8_4_1);
                        _vec_load_1[8 + 2] = __uint_as_float(_ldv8_4_2);
                        _vec_load_1[8 + 3] = __uint_as_float(_ldv8_4_3);
                        _vec_load_1[8 + 4] = __uint_as_float(_ldv8_4_4);
                        _vec_load_1[8 + 5] = __uint_as_float(_ldv8_4_5);
                        _vec_load_1[8 + 6] = __uint_as_float(_ldv8_4_6);
                        _vec_load_1[8 + 7] = __uint_as_float(_ldv8_4_7);
                    }
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 16; j_2++) {
                        x_cache[rr_5 * 32 + v_2 * 16 + j_2] = x_cache[rr_5 * 32 + v_2 * 16 + j_2] * rstd_r * _vec_load_1[j_2];
                    }
                    unsigned int packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 0], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 2], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 2]));
                        packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 4], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 4]));
                        packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 6], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 6]));
                        packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 8], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 8]));
                        packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 10], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 10]));
                        packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 12], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 12]));
                        packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((x_cache + rr_5 * 32 + v_2 * 16)[_lp*2 + 14], (x_cache + rr_5 * 32 + v_2 * 16)[_lp*2+1 + 14]));
                        packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(y + (obasec + (long long)(gtid * 16 + v_2 * (GROUP_THREADS * 16))))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                    }
                }
            }
        }
    }
}

} // extern "C"
