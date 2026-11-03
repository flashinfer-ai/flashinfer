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

__global__ __launch_bounds__(256) void
kernel_cake_dense_projection_gemm_23cd6c7b07c9130bef18(float* __restrict__ A, float* __restrict__ B, float* __restrict__ out, int M, int N, int K, int lda, int ldb, int ldo)
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
    int slice_id = tid / 128;
    int st = tid % 128;
    int tx = st % 8;
    int ty = st / 8;
    int m0 = blockIdx.x * 64;
    int n0 = blockIdx.y * 128;
    unsigned int ring = smem_addr + (unsigned int)(slice_id * 24576);
    float acc[64];
    #pragma unroll
    for (int e = 0; e < 64; e++) {
        acc[e] = 0.0f;
    }
    int mine = ((K + 16 - 1) / 16 - slice_id + 2 - 1) / 2;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (mine > 0) {
        int k0 = slice_id * 16;
        unsigned int fill = ring + 0;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int bv = c * 128 + st;
            int bk = k0 + bv / 32;
            int _min_0 = ((bk) < (K - 1) ? (bk) : (K - 1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(bv * 16)), "l"(B + ((long long)_min_0 * (long long)ldb + (long long)(n0 + bv % 32 * 4))), "r"((bk < K) ? 16 : 0));
        }
        #pragma unroll
        for (int c_1 = 0; c_1 < 2; c_1++) {
            int av = c_1 * 128 + st;
            int ak = k0 + av / 16;
            int _min_1 = ((ak) < (K - 1) ? (ak) : (K - 1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(8192 + av * 16)), "l"(A + ((long long)_min_1 * (long long)lda + (long long)(m0 + av % 16 * 4))), "r"((ak < K) ? 16 : 0));
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int i = 0; i < mine; i++) {
        if (i + 1 < mine) {
            int k0_1 = (slice_id + (i + 1) * 2) * 16;
            unsigned int fill_1 = ring + (unsigned int)((i + 1) % 2 * 12288);
            #pragma unroll
            for (int c_2 = 0; c_2 < 4; c_2++) {
                int bv_1 = c_2 * 128 + st;
                int bk_1 = k0_1 + bv_1 / 32;
                int _min_2 = ((bk_1) < (K - 1) ? (bk_1) : (K - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(bv_1 * 16)), "l"(B + ((long long)_min_2 * (long long)ldb + (long long)(n0 + bv_1 % 32 * 4))), "r"((bk_1 < K) ? 16 : 0));
            }
            #pragma unroll
            for (int c_3 = 0; c_3 < 2; c_3++) {
                int av_1 = c_3 * 128 + st;
                int ak_1 = k0_1 + av_1 / 16;
                int _min_3 = ((ak_1) < (K - 1) ? (ak_1) : (K - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(8192 + av_1 * 16)), "l"(A + ((long long)_min_3 * (long long)lda + (long long)(m0 + av_1 % 16 * 4))), "r"((ak_1 < K) ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        asm volatile("barrier.sync %0, 128;" :: "r"(slice_id + 1) : "memory");
        unsigned int cur = ring + (unsigned int)(i % 2 * 12288);
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float bn[8];
            float am[8];
            unsigned int b_row = cur + (unsigned int)(kk * 512);
            unsigned int a_row = cur + (unsigned int)(8192 + kk * 256);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bn[0])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 3]))
                : "r"(b_row + (unsigned int)(ty * 16)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bn[4])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 3]))
                : "r"(b_row + (unsigned int)((64 + ty * 4) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&am[0])), "=r"(*reinterpret_cast<uint32_t*>(&am[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&am[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&am[(0) + 3]))
                : "r"(a_row + (unsigned int)(tx * 16)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&am[4])), "=r"(*reinterpret_cast<uint32_t*>(&am[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&am[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&am[(4) + 3]))
                : "r"(a_row + (unsigned int)((32 + tx * 4) * 4)));
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                #pragma unroll
                for (int c_4 = 0; c_4 < 8; c_4++) {
                    acc[r * 8 + c_4] = __fmaf_rn(bn[r], am[c_4], acc[r * 8 + c_4]);
                }
            }
        }
        asm volatile("barrier.sync %0, 128;" :: "r"(slice_id + 1) : "memory");
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned int park = smem_addr + (unsigned int)(warp * 8192 + lane * 16);
    #pragma unroll
    for (int r_1 = 0; r_1 < 8; r_1++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(park + (unsigned int)((r_1 * 2 + h) * (32 * 16))), "r"(__as_u32(acc[r_1 * 8 + h * 4])), "r"(__as_u32(acc[r_1 * 8 + h * 4 + 1])), "r"(__as_u32(acc[r_1 * 8 + h * 4 + 2])), "r"(__as_u32(acc[r_1 * 8 + h * 4 + 3])) : "memory");
        }
    }
    __syncthreads();
    #pragma unroll
    for (int c_5 = 0; c_5 < 8; c_5++) {
        int v = c_5 * 256 + tid;
        int n = v / 16;
        int mc = v % 16;
        int pst = n % 64 / 4 * 8 + mc % 8;
        int prow = n / 64 * 4 + n % 4;
        int chunk = (pst / 32 * 16 + prow * 2 + mc / 8) * 32 + pst % 32;
        float total[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&total[0])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 3]))
            : "r"(smem_addr + (unsigned int)(chunk * 16)));
        #pragma unroll
        for (int q = 1; q < 2; q++) {
            float other[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&other[0])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other[(0) + 3]))
                : "r"(smem_addr + (unsigned int)(q * 32768 + chunk * 16)));
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                total[e_1] = total[e_1] + other[e_1];
            }
        }
        {
            float4 _v4 = make_float4(total[0 + 0], total[0 + 1], total[0 + 2], total[0 + 3]);
            *reinterpret_cast<float4*>((out + ((long long)(n0 + v / 16) * (long long)ldo + (long long)(m0 + v % 16 * 4))) + 0) = _v4;
        }
    }
}

} // extern "C"
