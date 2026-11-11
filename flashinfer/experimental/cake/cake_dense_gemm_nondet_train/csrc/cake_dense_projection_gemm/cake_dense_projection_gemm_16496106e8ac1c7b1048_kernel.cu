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

extern "C" {

__global__ __launch_bounds__(256) __cluster_dims__(8,1,1) void
kernel_cake_dense_projection_gemm_16496106e8ac1c7b1048(float* __restrict__ X, float* __restrict__ W, float* __restrict__ out, int T, int N, int K, int ldx, int ldw, int ldo)
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
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* smem_1 = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int rank = blockIdx.x % 8;
    int e0 = blockIdx.x / 8 * 8;
    int t0 = blockIdx.y * 4;
    int seg = K / 8;
    int k0 = rank * seg;
    int chunks = seg / 128;
    int j = lane % 32;
    int ge = tid / 32 % 8;
    int gt = tid / 32 / 8;
    float acc[4];
    float comp[4];
    #pragma unroll
    for (int o = 0; o < 4; o++) {
        acc[o] = 0.0f;
    }
    #pragma unroll
    for (int o_1 = 0; o_1 < 4; o_1++) {
        comp[o_1] = 0.0f;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (0 < chunks) {
        unsigned int fill = smem_addr + 0;
        int col0 = k0;
        #pragma unroll
        for (int i = 0; i < 1; i++) {
            int wv_ = i * 256 + tid;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(fill + (unsigned int)((wv_ / 32 * 144 + wv_ % 32 * 4) * 4)), "l"(W + ((long long)(e0 + wv_ / 32) * (long long)ldw + (long long)(col0 + wv_ % 32 * 4))));
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 1; i_1++) {
            int xv_ = i_1 * 256 + tid;
            if (xv_ < 128) {
                int tok = t0 + xv_ / 32;
                int _min_0 = ((tok) < (T - 1) ? (tok) : (T - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill + (unsigned int)(((8 + xv_ / 32) * 144 + xv_ % 32 * 4) * 4)), "l"(X + ((long long)_min_0 * (long long)ldx + (long long)(col0 + xv_ % 32 * 4))), "r"((tok < T) ? 16 : 0));
            }
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int c = 0; c < chunks; c++) {
        if (c + 2 - 1 < chunks) {
            unsigned int fill_1 = smem_addr + (unsigned int)((c + 2 - 1) % 2 * 6912);
            int col0_1 = k0 + (c + 2 - 1) * 128;
            #pragma unroll
            for (int i_2 = 0; i_2 < 1; i_2++) {
                int wv__1 = i_2 * 256 + tid;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(fill_1 + (unsigned int)((wv__1 / 32 * 144 + wv__1 % 32 * 4) * 4)), "l"(W + ((long long)(e0 + wv__1 / 32) * (long long)ldw + (long long)(col0_1 + wv__1 % 32 * 4))));
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 1; i_3++) {
                int xv__1 = i_3 * 256 + tid;
                if (xv__1 < 128) {
                    int tok_1 = t0 + xv__1 / 32;
                    int _min_1 = ((tok_1) < (T - 1) ? (tok_1) : (T - 1));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(fill_1 + (unsigned int)(((8 + xv__1 / 32) * 144 + xv__1 % 32 * 4) * 4)), "l"(X + ((long long)_min_1 * (long long)ldx + (long long)(col0_1 + xv__1 % 32 * 4))), "r"((tok_1 < T) ? 16 : 0));
                }
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        unsigned int cur = smem_addr + (unsigned int)(c % 2 * 6912);
        float wv[1];
        float xv[4];
        float yv[4];
        #pragma unroll
        for (int g4 = 0; g4 < 1; g4++) {
            #pragma unroll
            for (int o_2 = 0; o_2 < 4; o_2++) {
                yv[o_2] = -comp[o_2];
            }
            int col_0 = j + 32 * (g4 * 4);
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])) : "r"(cur + (unsigned int)((ge * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])) : "r"(cur + (unsigned int)(((8 + gt) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[1])) : "r"(cur + (unsigned int)(((8 + gt + 1) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[2])) : "r"(cur + (unsigned int)(((8 + gt + 2) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[3])) : "r"(cur + (unsigned int)(((8 + gt + 3) * 144 + col_0) * 4)));
            yv[0] = __fmaf_rn(xv[0], wv[0], yv[0]);
            yv[1] = __fmaf_rn(xv[1], wv[0], yv[1]);
            yv[2] = __fmaf_rn(xv[2], wv[0], yv[2]);
            yv[3] = __fmaf_rn(xv[3], wv[0], yv[3]);
            int col_1 = j + 32 * (g4 * 4 + 1);
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])) : "r"(cur + (unsigned int)((ge * 144 + col_1) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])) : "r"(cur + (unsigned int)(((8 + gt) * 144 + col_1) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[1])) : "r"(cur + (unsigned int)(((8 + gt + 1) * 144 + col_1) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[2])) : "r"(cur + (unsigned int)(((8 + gt + 2) * 144 + col_1) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[3])) : "r"(cur + (unsigned int)(((8 + gt + 3) * 144 + col_1) * 4)));
            yv[0] = __fmaf_rn(xv[0], wv[0], yv[0]);
            yv[1] = __fmaf_rn(xv[1], wv[0], yv[1]);
            yv[2] = __fmaf_rn(xv[2], wv[0], yv[2]);
            yv[3] = __fmaf_rn(xv[3], wv[0], yv[3]);
            int col_2 = j + 32 * (g4 * 4 + 2);
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])) : "r"(cur + (unsigned int)((ge * 144 + col_2) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])) : "r"(cur + (unsigned int)(((8 + gt) * 144 + col_2) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[1])) : "r"(cur + (unsigned int)(((8 + gt + 1) * 144 + col_2) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[2])) : "r"(cur + (unsigned int)(((8 + gt + 2) * 144 + col_2) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[3])) : "r"(cur + (unsigned int)(((8 + gt + 3) * 144 + col_2) * 4)));
            yv[0] = __fmaf_rn(xv[0], wv[0], yv[0]);
            yv[1] = __fmaf_rn(xv[1], wv[0], yv[1]);
            yv[2] = __fmaf_rn(xv[2], wv[0], yv[2]);
            yv[3] = __fmaf_rn(xv[3], wv[0], yv[3]);
            int col_3 = j + 32 * (g4 * 4 + 3);
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])) : "r"(cur + (unsigned int)((ge * 144 + col_3) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])) : "r"(cur + (unsigned int)(((8 + gt) * 144 + col_3) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[1])) : "r"(cur + (unsigned int)(((8 + gt + 1) * 144 + col_3) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[2])) : "r"(cur + (unsigned int)(((8 + gt + 2) * 144 + col_3) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[3])) : "r"(cur + (unsigned int)(((8 + gt + 3) * 144 + col_3) * 4)));
            yv[0] = __fmaf_rn(xv[0], wv[0], yv[0]);
            yv[1] = __fmaf_rn(xv[1], wv[0], yv[1]);
            yv[2] = __fmaf_rn(xv[2], wv[0], yv[2]);
            yv[3] = __fmaf_rn(xv[3], wv[0], yv[3]);
            #pragma unroll
            for (int o_3 = 0; o_3 < 4; o_3++) {
                float tsum = acc[o_3] + yv[o_3];
                comp[o_3] = tsum - acc[o_3] - yv[o_3];
                acc[o_3] = tsum;
            }
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");
    #pragma unroll
    for (int o_4 = 0; o_4 < 4; o_4++) {
        comp[o_4] = -comp[o_4];
    }
    #pragma unroll
    for (int o_0 = 0; o_0 < 4; o_0++) {
        float peer_s_0 = __shfl_xor_sync(0xFFFFFFFF, acc[o_0], 1);
        float peer_l_0 = __shfl_xor_sync(0xFFFFFFFF, comp[o_0], 1);
        float s2_0 = acc[o_0] + peer_s_0;
        float bb_0 = s2_0 - acc[o_0];
        float q2_0 = acc[o_0] - (s2_0 - bb_0) + (peer_s_0 - bb_0);
        comp[o_0] = comp[o_0] + peer_l_0 + q2_0;
        acc[o_0] = s2_0;
    }
    #pragma unroll
    for (int o_1_1 = 0; o_1_1 < 4; o_1_1++) {
        float peer_s_1 = __shfl_xor_sync(0xFFFFFFFF, acc[o_1_1], 2);
        float peer_l_1 = __shfl_xor_sync(0xFFFFFFFF, comp[o_1_1], 2);
        float s2_1 = acc[o_1_1] + peer_s_1;
        float bb_1 = s2_1 - acc[o_1_1];
        float q2_1 = acc[o_1_1] - (s2_1 - bb_1) + (peer_s_1 - bb_1);
        comp[o_1_1] = comp[o_1_1] + peer_l_1 + q2_1;
        acc[o_1_1] = s2_1;
    }
    #pragma unroll
    for (int o_2_1 = 0; o_2_1 < 4; o_2_1++) {
        float peer_s_2 = __shfl_xor_sync(0xFFFFFFFF, acc[o_2_1], 4);
        float peer_l_2 = __shfl_xor_sync(0xFFFFFFFF, comp[o_2_1], 4);
        float s2_2 = acc[o_2_1] + peer_s_2;
        float bb_2 = s2_2 - acc[o_2_1];
        float q2_2 = acc[o_2_1] - (s2_2 - bb_2) + (peer_s_2 - bb_2);
        comp[o_2_1] = comp[o_2_1] + peer_l_2 + q2_2;
        acc[o_2_1] = s2_2;
    }
    #pragma unroll
    for (int o_3_1 = 0; o_3_1 < 4; o_3_1++) {
        float peer_s_3 = __shfl_xor_sync(0xFFFFFFFF, acc[o_3_1], 8);
        float peer_l_3 = __shfl_xor_sync(0xFFFFFFFF, comp[o_3_1], 8);
        float s2_3 = acc[o_3_1] + peer_s_3;
        float bb_3 = s2_3 - acc[o_3_1];
        float q2_3 = acc[o_3_1] - (s2_3 - bb_3) + (peer_s_3 - bb_3);
        comp[o_3_1] = comp[o_3_1] + peer_l_3 + q2_3;
        acc[o_3_1] = s2_3;
    }
    #pragma unroll
    for (int o_4_1 = 0; o_4_1 < 4; o_4_1++) {
        float peer_s_4 = __shfl_xor_sync(0xFFFFFFFF, acc[o_4_1], 16);
        float peer_l_4 = __shfl_xor_sync(0xFFFFFFFF, comp[o_4_1], 16);
        float s2_4 = acc[o_4_1] + peer_s_4;
        float bb_4 = s2_4 - acc[o_4_1];
        float q2_4 = acc[o_4_1] - (s2_4 - bb_4) + (peer_s_4 - bb_4);
        comp[o_4_1] = comp[o_4_1] + peer_l_4 + q2_4;
        acc[o_4_1] = s2_4;
    }
    if (j == 0) {
        int red_o_0 = ge * 4 + gt;
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(smem_addr + (unsigned int)(red_o_0 * 8)), "f"(acc[0]), "f"(comp[0]) : "memory");
    }
    if (j == 1) {
        int red_o_1 = ge * 4 + gt + 1;
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(smem_addr + (unsigned int)(red_o_1 * 8)), "f"(acc[1]), "f"(comp[1]) : "memory");
    }
    if (j == 2) {
        int red_o_2 = ge * 4 + gt + 2;
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(smem_addr + (unsigned int)(red_o_2 * 8)), "f"(acc[2]), "f"(comp[2]) : "memory");
    }
    if (j == 3) {
        int red_o_3 = ge * 4 + gt + 3;
        asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(smem_addr + (unsigned int)(red_o_3 * 8)), "f"(acc[3]), "f"(comp[3]) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    float peer[16];
    float res[1];
    int idx_0 = tid;
    if (idx_0 < 4) {
        int el_0 = rank;
        int tl_0 = idx_0;
        unsigned int red_byte_0 = (unsigned int)((el_0 * 4 + tl_0) * 2 * 4);
        uint32_t _cluster_mapa_2;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_2) : "r"(smem_addr + red_byte_0), "r"(0));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(0) + 1]))
            : "r"(_cluster_mapa_2));
        uint32_t _cluster_mapa_3;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_3) : "r"(smem_addr + red_byte_0), "r"(1));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[2])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(2) + 1]))
            : "r"(_cluster_mapa_3));
        uint32_t _cluster_mapa_4;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_4) : "r"(smem_addr + red_byte_0), "r"(2));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[4])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(4) + 1]))
            : "r"(_cluster_mapa_4));
        uint32_t _cluster_mapa_5;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_5) : "r"(smem_addr + red_byte_0), "r"(3));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[6])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(6) + 1]))
            : "r"(_cluster_mapa_5));
        uint32_t _cluster_mapa_6;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_6) : "r"(smem_addr + red_byte_0), "r"(4));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[8])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(8) + 1]))
            : "r"(_cluster_mapa_6));
        uint32_t _cluster_mapa_7;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_7) : "r"(smem_addr + red_byte_0), "r"(5));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[10])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(10) + 1]))
            : "r"(_cluster_mapa_7));
        uint32_t _cluster_mapa_8;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_8) : "r"(smem_addr + red_byte_0), "r"(6));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[12])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(12) + 1]))
            : "r"(_cluster_mapa_8));
        uint32_t _cluster_mapa_9;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_9) : "r"(smem_addr + red_byte_0), "r"(7));
        asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&peer[14])), "=r"(*reinterpret_cast<uint32_t*>(&peer[(14) + 1]))
            : "r"(_cluster_mapa_9));
        float tot_0 = peer[0];
        float err_0 = peer[1];
        float s3_0 = tot_0 + peer[2];
        float b3_0 = s3_0 - tot_0;
        float q3_0 = tot_0 - (s3_0 - b3_0) + (peer[2] - b3_0);
        err_0 = err_0 + (q3_0 + peer[3]);
        tot_0 = s3_0;
        float s3_1 = tot_0 + peer[4];
        float b3_1 = s3_1 - tot_0;
        float q3_1 = tot_0 - (s3_1 - b3_1) + (peer[4] - b3_1);
        err_0 = err_0 + (q3_1 + peer[5]);
        tot_0 = s3_1;
        float s3_2 = tot_0 + peer[6];
        float b3_2 = s3_2 - tot_0;
        float q3_2 = tot_0 - (s3_2 - b3_2) + (peer[6] - b3_2);
        err_0 = err_0 + (q3_2 + peer[7]);
        tot_0 = s3_2;
        float s3_3 = tot_0 + peer[8];
        float b3_3 = s3_3 - tot_0;
        float q3_3 = tot_0 - (s3_3 - b3_3) + (peer[8] - b3_3);
        err_0 = err_0 + (q3_3 + peer[9]);
        tot_0 = s3_3;
        float s3_4 = tot_0 + peer[10];
        float b3_4 = s3_4 - tot_0;
        float q3_4 = tot_0 - (s3_4 - b3_4) + (peer[10] - b3_4);
        err_0 = err_0 + (q3_4 + peer[11]);
        tot_0 = s3_4;
        float s3_5 = tot_0 + peer[12];
        float b3_5 = s3_5 - tot_0;
        float q3_5 = tot_0 - (s3_5 - b3_5) + (peer[12] - b3_5);
        err_0 = err_0 + (q3_5 + peer[13]);
        tot_0 = s3_5;
        float s3_6 = tot_0 + peer[14];
        float b3_6 = s3_6 - tot_0;
        float q3_6 = tot_0 - (s3_6 - b3_6) + (peer[14] - b3_6);
        err_0 = err_0 + (q3_6 + peer[15]);
        tot_0 = s3_6;
        res[0] = tot_0 + err_0;
        if (t0 + tl_0 < T) {
            *(reinterpret_cast<float*>(out + ((long long)(t0 + tl_0) * (long long)ldo + (long long)(e0 + el_0))) + (0)) = res[0];
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

} // extern "C"
