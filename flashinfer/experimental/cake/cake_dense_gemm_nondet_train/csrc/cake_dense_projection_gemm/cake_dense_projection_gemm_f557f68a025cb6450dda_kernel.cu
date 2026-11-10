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

#include <math_constants.h>


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}


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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_dense_projection_gemm_f557f68a025cb6450dda(float* __restrict__ A, float* __restrict__ B, float* __restrict__ out, int M, int N, int K, int lda, int ldb, int ldo)
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

    // Prologue
    float* smem_1 = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int slice_id = tid / 32;
    int st = tid % 32;
    int tx = st % 8;
    int ty = st / 8;
    int n0 = blockIdx.x * 64;
    int m0 = blockIdx.y * 32;
    unsigned int ring = smem_addr + (unsigned int)(slice_id * 15360);
    unsigned int turned = ring + 13312;
    float2 acc[32];
    #pragma unroll
    for (int e = 0; e < 32; e++) {
        acc[e] = make_float2(0.0f, 0.0f);
    }
    int mine = ((K + 16 - 1) / 16 - slice_id + 4 - 1) / 4;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (mine > 0) {
        int k0 = slice_id * 16;
        unsigned int fill = ring + 0;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int bv = c * 32 + st;
            int bk = k0 + bv / 16;
            int _min_0 = ((bk) < (K - 1) ? (bk) : (K - 1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(bv * 16)), "l"(B + ((long long)_min_0 * (long long)ldb + (long long)(n0 + bv % 16 * 4))), "r"((bk < K) ? 16 : 0));
        }
        #pragma unroll
        for (int c_1 = 0; c_1 < 4; c_1++) {
            int av = c_1 * 32 + st;
            int am = m0 + av / 4;
            int ak = k0 + av % 4 * 4;
            int _min_1 = ((am) < (M - 1) ? (am) : (M - 1));
            int _min_2 = ((ak) < (K - 4) ? (ak) : (K - 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(4096 + (av / 4 * 20 + av % 4 * 4) * 4)), "l"(A + ((long long)_min_1 * (long long)lda + (long long)_min_2)), "r"((am < M && ak < K) ? 16 : 0));
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int i = 0; i < mine; i++) {
        if (i + 1 < mine) {
            int k0_1 = (slice_id + (i + 1) * 4) * 16;
            unsigned int fill_1 = ring + (unsigned int)((i + 1) % 2 * 6656);
            #pragma unroll
            for (int c_2 = 0; c_2 < 8; c_2++) {
                int bv_1 = c_2 * 32 + st;
                int bk_1 = k0_1 + bv_1 / 16;
                int _min_3 = ((bk_1) < (K - 1) ? (bk_1) : (K - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(bv_1 * 16)), "l"(B + ((long long)_min_3 * (long long)ldb + (long long)(n0 + bv_1 % 16 * 4))), "r"((bk_1 < K) ? 16 : 0));
            }
            #pragma unroll
            for (int c_3 = 0; c_3 < 4; c_3++) {
                int av_1 = c_3 * 32 + st;
                int am_1 = m0 + av_1 / 4;
                int ak_1 = k0_1 + av_1 % 4 * 4;
                int _min_4 = ((am_1) < (M - 1) ? (am_1) : (M - 1));
                int _min_5 = ((ak_1) < (K - 4) ? (ak_1) : (K - 4));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(4096 + (av_1 / 4 * 20 + av_1 % 4 * 4) * 4)), "l"(A + ((long long)_min_4 * (long long)lda + (long long)_min_5)), "r"((am_1 < M && ak_1 < K) ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        asm volatile("barrier.sync %0, 32;" :: "r"(slice_id + 1) : "memory");
        unsigned int cur = ring + (unsigned int)(i % 2 * 6656);
        #pragma unroll
        for (int c_4 = 0; c_4 < 4; c_4++) {
            int tv = c_4 * 32 + st;
            int trow = tv % 32;
            int tq = tv / 32;
            float quad[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 3]))
                : "r"(cur + (unsigned int)(4096 + (trow * 20 + tq * 4) * 4)));
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(turned + (unsigned int)(((tq * 4 + e_1) * 32 + trow) * 4)), "r"(__as_u32(quad[e_1])) : "memory");
            }
        }
        asm volatile("barrier.sync %0, 32;" :: "r"(slice_id + 1) : "memory");
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float am_2[8];
            float bn[8];
            unsigned int a_row = turned + (unsigned int)(kk * 128);
            unsigned int b_row = cur + (unsigned int)(kk * 256);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&am_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&am_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&am_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&am_2[(0) + 3]))
                : "r"(a_row + (unsigned int)(ty * 16)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&am_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&am_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&am_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&am_2[(4) + 3]))
                : "r"(a_row + (unsigned int)((16 + ty * 4) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bn[0])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 3]))
                : "r"(b_row + (unsigned int)(tx * 16)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bn[4])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 3]))
                : "r"(b_row + (unsigned int)((32 + tx * 4) * 4)));
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                float2 a2 = make_float2(am_2[r], am_2[r]);
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    acc[r * 4 + p] = fma_f32x2_rn_noftz(a2, make_float2(bn[2 * p], bn[2 * p + 1]), acc[r * 4 + p]);
                }
            }
        }
        asm volatile("barrier.sync %0, 32;" :: "r"(slice_id + 1) : "memory");
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned int park = smem_addr + (unsigned int)(warp * 8192 + lane * 16);
    #pragma unroll
    for (int r_1 = 0; r_1 < 8; r_1++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(park + (unsigned int)((r_1 * 2 + h) * (32 * 16))), "r"(__as_u32(acc[r_1 * 4 + h * 2].x)), "r"(__as_u32(acc[r_1 * 4 + h * 2].y)), "r"(__as_u32(acc[r_1 * 4 + h * 2 + 1].x)), "r"(__as_u32(acc[r_1 * 4 + h * 2 + 1].y)) : "memory");
        }
    }
    __syncthreads();
    #pragma unroll
    for (int c_5 = 0; c_5 < 4; c_5++) {
        int v = c_5 * 128 + tid;
        int m = v / 16;
        int nc = v % 16;
        int pst = m % 16 / 4 * 8 + nc % 8;
        int prow = m / 16 * 4 + m % 4;
        int chunk = (pst / 32 * 16 + prow * 2 + nc / 8) * 32 + pst % 32;
        float total[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&total[0])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 3]))
            : "r"(smem_addr + (unsigned int)(chunk * 16)));
        #pragma unroll
        for (int q = 1; q < 4; q++) {
            float other[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[0])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 3]))
                : "r"(smem_addr + (unsigned int)(q * 8192 + chunk * 16)));
            #pragma unroll
            for (int e_2 = 0; e_2 < 4; e_2++) {
                total[e_2] = total[e_2] + other[e_2];
            }
        }
        int _min_6 = ((m0 + m) < (M - 1) ? (m0 + m) : (M - 1));
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(out + ((long long)_min_6 * (long long)ldo + (long long)(n0 + nc * 4))), "f"(total[(0) + 0]), "f"(total[(0) + 1]), "f"(total[(0) + 2]), "f"(total[(0) + 3]), "r"((unsigned int)(m0 + m < M)) : "memory");
    }
}

} // extern "C"
