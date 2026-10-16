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
#define SMEM_PARTIALS_STAGE_BYTES 16
#define SMEM_PARTIALS_STRIDE 16
#define SMEM_TOTAL 128
#define THREADS 64
#define ROWS_PER_CTA 2
#define GROUP_THREADS 64
#define LAUNCH_BOUNDS_THREADS 64

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) void
kernel_cake_rmsnorm_train_d44010e821a6c74587a6(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, long long x_stride, long long u_stride, float eps)
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
                long long xbasea = row64a * x_stride;
                #pragma unroll
                for (int v = 0; v < 2; v++) {
                    float _vec_load_0[16];
                    {
                        const void* _v8p_0 = (const void*)(x + (xbasea + (long long)(gtid * 16 + v * (GROUP_THREADS * 16))) + (0));
                        uint32_t _v8_0_0[8];
                        asm volatile("ld.global.L2::cache_hint.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
                            : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const void*)((const char*)_v8p_0 + 0)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 0])[0]), "=f"((&_vec_load_0[0 + 0])[1])
                            : "r"(_v8_0_0[0]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 2])[0]), "=f"((&_vec_load_0[0 + 2])[1])
                            : "r"(_v8_0_0[1]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 4])[0]), "=f"((&_vec_load_0[0 + 4])[1])
                            : "r"(_v8_0_0[2]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 6])[0]), "=f"((&_vec_load_0[0 + 6])[1])
                            : "r"(_v8_0_0[3]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 8])[0]), "=f"((&_vec_load_0[0 + 8])[1])
                            : "r"(_v8_0_0[4]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 10])[0]), "=f"((&_vec_load_0[0 + 10])[1])
                            : "r"(_v8_0_0[5]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 12])[0]), "=f"((&_vec_load_0[0 + 12])[1])
                            : "r"(_v8_0_0[6]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + 14])[0]), "=f"((&_vec_load_0[0 + 14])[1])
                            : "r"(_v8_0_0[7]));
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
                        acc += x_cache[rr_2 * 32 + v_1 * 16 + j_1] * x_cache[rr_2 * 32 + v_1 * 16 + j_1];
                    }
                }
                sums[rr_2] = acc;
            }
        }
        int par = (it & 1) * 4;
        #pragma unroll
        for (int rr_3 = 0; rr_3 < 2; rr_3++) {
            float red = sums[rr_3];
            float _warp_reduce_0 = red;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            red = _warp_reduce_0;
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
            float _sqrt_0;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(mean_r + eps));
            float _rcp_0 = __frcp_rn(_sqrt_0);
            float rstd_r = _rcp_0;
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
                        const void* _v8p_1 = (const void*)(w + (gtid * 16 + v_2 * (GROUP_THREADS * 16)) + (0));
                        uint32_t _v8_1_0[8];
                        asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_v8_1_0[0]), "=r"(_v8_1_0[1]), "=r"(_v8_1_0[2]), "=r"(_v8_1_0[3]), "=r"(_v8_1_0[4]), "=r"(_v8_1_0[5]), "=r"(_v8_1_0[6]), "=r"(_v8_1_0[7]) : "l"((const void*)((const char*)_v8p_1 + 0)) : "memory");
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 0])[0]), "=f"((&_vec_load_1[0 + 0])[1])
                            : "r"(_v8_1_0[0]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 2])[0]), "=f"((&_vec_load_1[0 + 2])[1])
                            : "r"(_v8_1_0[1]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 4])[0]), "=f"((&_vec_load_1[0 + 4])[1])
                            : "r"(_v8_1_0[2]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 6])[0]), "=f"((&_vec_load_1[0 + 6])[1])
                            : "r"(_v8_1_0[3]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 8])[0]), "=f"((&_vec_load_1[0 + 8])[1])
                            : "r"(_v8_1_0[4]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 10])[0]), "=f"((&_vec_load_1[0 + 10])[1])
                            : "r"(_v8_1_0[5]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 12])[0]), "=f"((&_vec_load_1[0 + 12])[1])
                            : "r"(_v8_1_0[6]));
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + 14])[0]), "=f"((&_vec_load_1[0 + 14])[1])
                            : "r"(_v8_1_0[7]));
                    }
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 16; j_2++) {
                        x_cache[rr_5 * 32 + v_2 * 16 + j_2] = x_cache[rr_5 * 32 + v_2 * 16 + j_2] * rstd_r * _vec_load_1[j_2];
                    }
                    {
                        {
                            __nv_bfloat162 _pk0 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 0], x_cache[rr_5 * 32 + v_2 * 16 + 1]);
                            unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                            __nv_bfloat162 _pk1 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 2], x_cache[rr_5 * 32 + v_2 * 16 + 3]);
                            unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                            __nv_bfloat162 _pk2 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 4], x_cache[rr_5 * 32 + v_2 * 16 + 5]);
                            unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                            __nv_bfloat162 _pk3 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 6], x_cache[rr_5 * 32 + v_2 * 16 + 7]);
                            unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                            __nv_bfloat162 _pk4 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 8], x_cache[rr_5 * 32 + v_2 * 16 + 9]);
                            unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                            __nv_bfloat162 _pk5 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 10], x_cache[rr_5 * 32 + v_2 * 16 + 11]);
                            unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                            __nv_bfloat162 _pk6 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 12], x_cache[rr_5 * 32 + v_2 * 16 + 13]);
                            unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                            __nv_bfloat162 _pk7 = __floats2bfloat162_rn(x_cache[rr_5 * 32 + v_2 * 16 + 14], x_cache[rr_5 * 32 + v_2 * 16 + 15]);
                            unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(&((__nv_bfloat16*)(y + (obasec + (long long)(gtid * 16 + v_2 * (GROUP_THREADS * 16)))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
