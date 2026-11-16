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

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


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

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
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
kernel_cake_dense_projection_gemm_5aabaa7971ddc277e20f(float* __restrict__ A, float* __restrict__ B, float* __restrict__ out, int M, int N, int K, int lda, int ldb, int ldo)
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
    int slice_id = tid / 64;
    int st = tid % 64;
    int tx = st % 8;
    int ty = st / 8;
    unsigned int b_lo = (unsigned int)(ty * 16);
    unsigned int b_hi = (unsigned int)((32 + ty * 4) * 4);
    unsigned int a_lo = (unsigned int)(tx * 16);
    unsigned int a_hi = (unsigned int)((32 + tx * 4) * 4);
    int m0 = blockIdx.x * 64;
    int n0 = blockIdx.y * 64;
    unsigned int ring = smem_addr + (unsigned int)(slice_id * 8192);
    float2 acc[32];
    float2 comp[32];
    #pragma unroll
    for (int e = 0; e < 32; e++) {
        acc[e] = make_float2(0.0f, 0.0f);
        comp[e] = make_float2(0.0f, 0.0f);
    }
    int mine = ((K + 8 - 1) / 8 - slice_id + 2 - 1) / 2;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (mine > 0) {
        int k0 = slice_id * 8;
        unsigned int fill = ring + 0;
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            int bv = c * 64 + st;
            int bk = k0 + bv / 16;
            int _min_0 = ((bk) < (K - 1) ? (bk) : (K - 1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(bv * 16)), "l"(B + ((long long)_min_0 * (long long)ldb + (long long)(n0 + bv % 16 * 4))), "r"((bk < K) ? 16 : 0));
        }
        #pragma unroll
        for (int c_1 = 0; c_1 < 2; c_1++) {
            int av = c_1 * 64 + st;
            int ak = k0 + av / 16;
            int _min_1 = ((ak) < (K - 1) ? (ak) : (K - 1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(2048 + av * 16)), "l"(A + ((long long)_min_1 * (long long)lda + (long long)(m0 + av % 16 * 4))), "r"((ak < K) ? 16 : 0));
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int i = 0; i < mine; i++) {
        if (i + 1 < mine) {
            int k0_1 = (slice_id + (i + 1) * 2) * 8;
            unsigned int fill_1 = ring + (unsigned int)((i + 1) % 2 * 4096);
            #pragma unroll
            for (int c_2 = 0; c_2 < 2; c_2++) {
                int bv_1 = c_2 * 64 + st;
                int bk_1 = k0_1 + bv_1 / 16;
                int _min_2 = ((bk_1) < (K - 1) ? (bk_1) : (K - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(bv_1 * 16)), "l"(B + ((long long)_min_2 * (long long)ldb + (long long)(n0 + bv_1 % 16 * 4))), "r"((bk_1 < K) ? 16 : 0));
            }
            #pragma unroll
            for (int c_3 = 0; c_3 < 2; c_3++) {
                int av_1 = c_3 * 64 + st;
                int ak_1 = k0_1 + av_1 / 16;
                int _min_3 = ((ak_1) < (K - 1) ? (ak_1) : (K - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(2048 + av_1 * 16)), "l"(A + ((long long)_min_3 * (long long)lda + (long long)(m0 + av_1 % 16 * 4))), "r"((ak_1 < K) ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        asm volatile("barrier.sync %0, 64;" :: "r"(slice_id + 1) : "memory");
        unsigned int cur = ring + (unsigned int)(i % 2 * 4096);
        int rows = K - (slice_id + i * 2) * 8;
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            if (rows > kk) {
                float bn[8];
                float am[8];
                unsigned int b_row = cur + (unsigned int)(kk * 256);
                unsigned int a_row = cur + (unsigned int)(2048 + kk * 256);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&bn[0])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(0) + 3]))
                    : "r"(b_row + b_lo));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&bn[4])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bn[(4) + 3]))
                    : "r"(b_row + b_hi));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&am[0])), "=r"(*reinterpret_cast<uint32_t*>(&am[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&am[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&am[(0) + 3]))
                    : "r"(a_row + a_lo));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&am[4])), "=r"(*reinterpret_cast<uint32_t*>(&am[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&am[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&am[(4) + 3]))
                    : "r"(a_row + a_hi));
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float2 b2 = make_float2(bn[r], bn[r]);
                    #pragma unroll
                    for (int p = 0; p < 4; p++) {
                        float nz = __uint_as_float((unsigned int)K >> 31 | 2147483648u);
                        float2 ab = make_float2(am[2 * p], am[2 * p + 1]);
                        float2 pr = fma_f32x2_rn_noftz(b2, ab, make_float2(nz, nz));
                        float2 pe = fma_sub_f32x2_rn_noftz(b2, ab, pr);
                        float2 sp = add_f32x2_noftz(acc[r * 4 + p], pr);
                        float2 sv = sub_f32x2_noftz(sp, acc[r * 4 + p]);
                        float2 sq = add_f32x2_noftz(sub_f32x2_noftz(acc[r * 4 + p], sub_f32x2_noftz(sp, sv)), sub_f32x2_noftz(pr, sv));
                        comp[r * 4 + p] = sub_f32x2_noftz(comp[r * 4 + p], add_f32x2_noftz(sq, pe));
                        acc[r * 4 + p] = sp;
                    }
                }
            }
        }
        asm volatile("barrier.sync %0, 64;" :: "r"(slice_id + 1) : "memory");
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned int park = smem_addr + (unsigned int)(warp * 8192 + lane * 16);
    #pragma unroll
    for (int r_1 = 0; r_1 < 8; r_1++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(park + (unsigned int)((r_1 * 2 + h) * (32 * 16))), "r"(__as_u32(acc[r_1 * 4 + h * 2].x)), "r"(__as_u32(acc[r_1 * 4 + h * 2].y)), "r"(__as_u32(acc[r_1 * 4 + h * 2 + 1].x)), "r"(__as_u32(acc[r_1 * 4 + h * 2 + 1].y)) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(park + (unsigned int)(32768 + (r_1 * 2 + h) * (32 * 16))), "r"(__as_u32(comp[r_1 * 4 + h * 2].x)), "r"(__as_u32(comp[r_1 * 4 + h * 2].y)), "r"(__as_u32(comp[r_1 * 4 + h * 2 + 1].x)), "r"(__as_u32(comp[r_1 * 4 + h * 2 + 1].y)) : "memory");
        }
    }
    __syncthreads();
    #pragma unroll
    for (int c_4 = 0; c_4 < 8; c_4++) {
        int v = c_4 * 128 + tid;
        int n = v / 16;
        int mc = v % 16;
        int pst = n % 32 / 4 * 8 + mc % 8;
        int prow = n / 32 * 4 + n % 4;
        int chunk = (pst / 32 * 16 + prow * 2 + mc / 8) * 32 + pst % 32;
        float total[4];
        float err[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&total[0])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&total[(0) + 3]))
            : "r"(smem_addr + (unsigned int)(chunk * 16)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&err[0])), "=r"(*reinterpret_cast<uint32_t*>(&err[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&err[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&err[(0) + 3]))
            : "r"(smem_addr + (unsigned int)(32768 + chunk * 16)));
        #pragma unroll
        for (int e_1 = 0; e_1 < 4; e_1++) {
            float _neg_4;
            asm("neg.f32 %0, %1;" : "=f"(_neg_4) : "f"(err[e_1]));
            err[e_1] = _neg_4;
        }
        #pragma unroll
        for (int q = 1; q < 2; q++) {
            float part[4];
            float pcomp[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
                : "r"(smem_addr + (unsigned int)(q * 16384 + chunk * 16)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&pcomp[0])), "=r"(*reinterpret_cast<uint32_t*>(&pcomp[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pcomp[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pcomp[(0) + 3]))
                : "r"(smem_addr + (unsigned int)(32768 + q * 16384 + chunk * 16)));
            #pragma unroll
            for (int e_2 = 0; e_2 < 4; e_2++) {
                float x;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(x) : "f"(total[e_2]), "f"(part[e_2]));
                float z;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(total[e_2]));
                float _sub_6;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(x), "f"(z));
                float _sub_5;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(total[e_2]), "f"(_sub_6));
                float _sub_7;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(part[e_2]), "f"(z));
                float two;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(two) : "f"(_sub_5), "f"(_sub_7));
                float _sub_9;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(two), "f"(pcomp[e_2]));
                float _add_8;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(err[e_2]), "f"(_sub_9));
                err[e_2] = _add_8;
                total[e_2] = x;
            }
        }
        #pragma unroll
        for (int e_3 = 0; e_3 < 4; e_3++) {
            float _add_10;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(total[e_3]), "f"(err[e_3]));
            total[e_3] = _add_10;
        }
        {
            float4 _v4 = make_float4(total[0 + 0], total[0 + 1], total[0 + 2], total[0 + 3]);
            *reinterpret_cast<float4*>((out + ((long long)(n0 + v / 16) * (long long)ldo + (long long)(m0 + v % 16 * 4))) + 0) = _v4;
        }
    }
}

} // extern "C"
