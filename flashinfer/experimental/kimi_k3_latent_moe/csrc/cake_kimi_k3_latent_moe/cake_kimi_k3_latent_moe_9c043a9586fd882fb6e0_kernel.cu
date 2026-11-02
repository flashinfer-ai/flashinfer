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
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 8
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_L3_BUF_OFF 214016
#define SMEM_L3_BUF_STAGE_BYTES 512
#define SMEM_L3_BUF_STRIDE 512
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 4096
#define SMEM_LAND_STRIDE 4096
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 6144
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 137216
#define SMEM_SMEM_B_STAGE_BYTES 1024
#define SMEM_SMEM_B_STRIDE 1024
#define SMEM_SMEM_BN_OFF 145408
#define SMEM_SMEM_BN_STAGE_BYTES 1024
#define SMEM_SMEM_BN_STRIDE 1024
#define SMEM_SMEM_V5_OFF 149504
#define SMEM_SMEM_V5_STAGE_BYTES 57344
#define SMEM_SMEM_V5_STRIDE 57344
#define SMEM_SMEM_V6_OFF 149504
#define SMEM_SMEM_V6_STAGE_BYTES 57344
#define SMEM_SMEM_V6_STRIDE 57344
#define SMEM_SMEM_V7_OFF 206848
#define SMEM_SMEM_V7_STAGE_BYTES 7168
#define SMEM_SMEM_V7_STRIDE 7168
#define SMEM_SMEM_V8_OFF 206848
#define SMEM_SMEM_V8_STAGE_BYTES 7168
#define SMEM_SMEM_V8_STRIDE 7168
#define SMEM_SMEM_AG_OFF 6144
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 14336
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 214528

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
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

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_9c043a9586fd882fb6e0(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define a_empty_addr (mbar_base + 64)
    #define acc_full_addr (mbar_base + 128)
    #define acc_empty_addr (mbar_base + 136)
    #define red_bar_addr (mbar_base + 144)
    #define issue_bar_addr (mbar_base + 152)
    #define rows_bar_addr (mbar_base + 160)
    #define bn_bar_addr (mbar_base + 168)
    #define rows_full_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    unsigned int* l3_buf = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int l3_buf_addr = smem + 214016;
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    unsigned int* free_word = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int free_word_addr = smem + 1536;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_a_addr = smem + 6144;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_b_addr = smem + 137216;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 145408);
    const int smem_bn_addr = smem + 145408;
    __nv_bfloat16* smem_v5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 149504);
    const int smem_v5_addr = smem + 149504;
    unsigned int* smem_v6 = reinterpret_cast<unsigned int*>(smem_raw + 149504);
    const int smem_v6_addr = smem + 149504;
    __nv_bfloat16* smem_v7 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int smem_v7_addr = smem + 206848;
    unsigned int* smem_v8 = reinterpret_cast<unsigned int*>(smem_raw + 206848);
    const int smem_v8_addr = smem + 206848;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_ag_addr = smem + 6144;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 14336);
    const int smem_au_addr = smem + 14336;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 23 barriers)
    // Mbarriers at smem_raw[0..184)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // a_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // rows_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 144, 4096);
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184);
    if (warp == 0) {
        int _tmem_hold = smem + 184;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int g_e = bid;
            int epi_warp = warp % 4;
            int tid_1 = epi_warp * 32 + lane;
            int tl_e = g_e * 16;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int rank_e = 0;
            rank_e = g_e % 2;
            int c_lo = k1_off + ((rank_e == 0) ? 0 : 4);
            int c_cnt = ((rank_e == 0) ? 4 : 3);
            int col_b = lane % 8 * 16;
            unsigned long long partial_stride = (unsigned long long)num_tokens * 3584;
            mbarrier_wait(rows_full_addr, 0);
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(issue_bar_addr);
                }
            }
            unsigned int l1_nw[16] = {0};
            #pragma unroll
            for (int q0 = 0; q0 < 4; q0++) {
                int l1_c0 = epi_warp * 4 + q0;
                if (l1_c0 < 14) {
                    unsigned int smem_regs[4];
                    #pragma unroll
                    for (int _lr = 0; _lr < 4; _lr++) {
                        smem_regs[_lr] = smem_v8[(lane + l1_c0 * 32) * 4 + _lr];
                    }
                    #pragma unroll
                    for (int k0 = 0; k0 < 4; k0++) {
                        l1_nw[q0 * 4 + k0] = smem_regs[k0];
                    }
                }
            }
            unsigned int l1_zero = 0;
            #pragma unroll 1
            for (int l1_ra = 0; l1_ra < num_tokens; l1_ra += 2) {
                float l1_acc[64];
                #pragma unroll
                for (int rp1 = 0; rp1 < 2; rp1++) {
                    int l1_r1 = l1_ra + rp1;
                    if (l1_r1 < num_tokens) {
                        #pragma unroll
                        for (int q1 = 0; q1 < 4; q1++) {
                            int l1_c1 = epi_warp * 4 + q1;
                            if (l1_c1 < 14) {
                                unsigned int smem_regs_1[4];
                                #pragma unroll
                                for (int _lr_1 = 0; _lr_1 < 4; _lr_1++) {
                                    smem_regs_1[_lr_1] = smem_v6[l1_r1 * 1792 + (lane + l1_c1 * 32) * 4 + _lr_1];
                                }
                                #pragma unroll
                                for (int k1 = 0; k1 < 4; k1++) {
                                    float l1_v[2];
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 lo, hi;\n\t"
                                        "mov.b32 {lo, hi}, %2;\n\t"
                                        "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                        "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                        "}\n"
                                        : "=&f"(l1_v[0]), "=&f"(l1_v[1]) : "r"(smem_regs_1[k1]), "f"(0.0f), "f"(0.0f));
                                    l1_acc[(rp1 * 4 + q1) * 8 + 2 * k1] = l1_v[0];
                                    l1_acc[(rp1 * 4 + q1) * 8 + 2 * k1 + 1] = l1_v[1];
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int l1_p = 1; l1_p < num_partials; l1_p++) {
                            int l1_row = l1_p * num_tokens + l1_r1;
                            #pragma unroll
                            for (int q2 = 0; q2 < 4; q2++) {
                                int l1_c2 = epi_warp * 4 + q2;
                                if (l1_c2 < 14) {
                                    unsigned int smem_regs_2[4];
                                    #pragma unroll
                                    for (int _lr_2 = 0; _lr_2 < 4; _lr_2++) {
                                        smem_regs_2[_lr_2] = smem_v6[l1_row * 1792 + (lane + l1_c2 * 32) * 4 + _lr_2];
                                    }
                                    #pragma unroll
                                    for (int k2 = 0; k2 < 4; k2++) {
                                        unsigned int l1_wd = smem_regs_2[k2];
                                        l1_acc[(rp1 * 4 + q2) * 8 + 2 * k2] = l1_acc[(rp1 * 4 + q2) * 8 + 2 * k2] + __uint_as_float(l1_wd << 16);
                                        l1_acc[(rp1 * 4 + q2) * 8 + 2 * k2 + 1] = l1_acc[(rp1 * 4 + q2) * 8 + 2 * k2 + 1] + __uint_as_float(l1_wd >> 16 << 16);
                                    }
                                }
                            }
                        }
                    }
                }
                if (epi_warp == 0) {
                    #pragma unroll
                    for (int rp5_0 = 0; rp5_0 < 2; rp5_0++) {
                        float l1_s_0 = 0.0f;
                        #pragma unroll
                        for (int j5_0 = 0; j5_0 < 8; j5_0++) {
                            float l1_sv_0 = (float)(__nv_bfloat16)l1_acc[rp5_0 * 4 * 8 + j5_0];
                            l1_s_0 = __fmaf_rn(l1_sv_0, l1_sv_0, l1_s_0);
                        }
                        #pragma unroll
                        for (int j5_1 = 0; j5_1 < 8; j5_1++) {
                            float l1_sv_1 = (float)(__nv_bfloat16)l1_acc[(rp5_0 * 4 + 1) * 8 + j5_1];
                            l1_s_0 = __fmaf_rn(l1_sv_1, l1_sv_1, l1_s_0);
                        }
                        #pragma unroll
                        for (int j5_2 = 0; j5_2 < 8; j5_2++) {
                            float l1_sv_2 = (float)(__nv_bfloat16)l1_acc[(rp5_0 * 4 + 2) * 8 + j5_2];
                            l1_s_0 = __fmaf_rn(l1_sv_2, l1_sv_2, l1_s_0);
                        }
                        #pragma unroll
                        for (int j5_3 = 0; j5_3 < 8; j5_3++) {
                            float l1_sv_3 = (float)(__nv_bfloat16)l1_acc[(rp5_0 * 4 + 3) * 8 + j5_3];
                            l1_s_0 = __fmaf_rn(l1_sv_3, l1_sv_3, l1_s_0);
                        }
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(l3_buf_addr + (unsigned int)((rp5_0 * 32 + lane) * 4)), "r"(__as_u32(l1_s_0)) : "memory");
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (epi_warp == 1) {
                    #pragma unroll
                    for (int rp5_1 = 0; rp5_1 < 2; rp5_1++) {
                        float l1_s_1 = 0.0f;
                        unsigned int l1_h_0[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&l1_h_0[0])) : "r"(l3_buf_addr + (unsigned int)((rp5_1 * 32 + lane) * 4)));
                        l1_s_1 = __uint_as_float(l1_h_0[0]);
                        #pragma unroll
                        for (int j5_4 = 0; j5_4 < 8; j5_4++) {
                            float l1_sv_4 = (float)(__nv_bfloat16)l1_acc[rp5_1 * 4 * 8 + j5_4];
                            l1_s_1 = __fmaf_rn(l1_sv_4, l1_sv_4, l1_s_1);
                        }
                        #pragma unroll
                        for (int j5_5 = 0; j5_5 < 8; j5_5++) {
                            float l1_sv_5 = (float)(__nv_bfloat16)l1_acc[(rp5_1 * 4 + 1) * 8 + j5_5];
                            l1_s_1 = __fmaf_rn(l1_sv_5, l1_sv_5, l1_s_1);
                        }
                        #pragma unroll
                        for (int j5_6 = 0; j5_6 < 8; j5_6++) {
                            float l1_sv_6 = (float)(__nv_bfloat16)l1_acc[(rp5_1 * 4 + 2) * 8 + j5_6];
                            l1_s_1 = __fmaf_rn(l1_sv_6, l1_sv_6, l1_s_1);
                        }
                        #pragma unroll
                        for (int j5_7 = 0; j5_7 < 8; j5_7++) {
                            float l1_sv_7 = (float)(__nv_bfloat16)l1_acc[(rp5_1 * 4 + 3) * 8 + j5_7];
                            l1_s_1 = __fmaf_rn(l1_sv_7, l1_sv_7, l1_s_1);
                        }
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(l3_buf_addr + (unsigned int)((rp5_1 * 32 + lane) * 4)), "r"(__as_u32(l1_s_1)) : "memory");
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (epi_warp == 2) {
                    #pragma unroll
                    for (int rp5_2 = 0; rp5_2 < 2; rp5_2++) {
                        float l1_s_2 = 0.0f;
                        unsigned int l1_h_1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&l1_h_1[0])) : "r"(l3_buf_addr + (unsigned int)((rp5_2 * 32 + lane) * 4)));
                        l1_s_2 = __uint_as_float(l1_h_1[0]);
                        #pragma unroll
                        for (int j5_8 = 0; j5_8 < 8; j5_8++) {
                            float l1_sv_8 = (float)(__nv_bfloat16)l1_acc[rp5_2 * 4 * 8 + j5_8];
                            l1_s_2 = __fmaf_rn(l1_sv_8, l1_sv_8, l1_s_2);
                        }
                        #pragma unroll
                        for (int j5_9 = 0; j5_9 < 8; j5_9++) {
                            float l1_sv_9 = (float)(__nv_bfloat16)l1_acc[(rp5_2 * 4 + 1) * 8 + j5_9];
                            l1_s_2 = __fmaf_rn(l1_sv_9, l1_sv_9, l1_s_2);
                        }
                        #pragma unroll
                        for (int j5_10 = 0; j5_10 < 8; j5_10++) {
                            float l1_sv_10 = (float)(__nv_bfloat16)l1_acc[(rp5_2 * 4 + 2) * 8 + j5_10];
                            l1_s_2 = __fmaf_rn(l1_sv_10, l1_sv_10, l1_s_2);
                        }
                        #pragma unroll
                        for (int j5_11 = 0; j5_11 < 8; j5_11++) {
                            float l1_sv_11 = (float)(__nv_bfloat16)l1_acc[(rp5_2 * 4 + 3) * 8 + j5_11];
                            l1_s_2 = __fmaf_rn(l1_sv_11, l1_sv_11, l1_s_2);
                        }
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(l3_buf_addr + (unsigned int)((rp5_2 * 32 + lane) * 4)), "r"(__as_u32(l1_s_2)) : "memory");
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (epi_warp == 3) {
                    #pragma unroll
                    for (int rp5_3 = 0; rp5_3 < 2; rp5_3++) {
                        float l1_s_3 = 0.0f;
                        unsigned int l1_h_2[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&l1_h_2[0])) : "r"(l3_buf_addr + (unsigned int)((rp5_3 * 32 + lane) * 4)));
                        l1_s_3 = __uint_as_float(l1_h_2[0]);
                        #pragma unroll
                        for (int j5_12 = 0; j5_12 < 8; j5_12++) {
                            float l1_sv_12 = (float)(__nv_bfloat16)l1_acc[rp5_3 * 4 * 8 + j5_12];
                            l1_s_3 = __fmaf_rn(l1_sv_12, l1_sv_12, l1_s_3);
                        }
                        #pragma unroll
                        for (int j5_13 = 0; j5_13 < 8; j5_13++) {
                            float l1_sv_13 = (float)(__nv_bfloat16)l1_acc[(rp5_3 * 4 + 1) * 8 + j5_13];
                            l1_s_3 = __fmaf_rn(l1_sv_13, l1_sv_13, l1_s_3);
                        }
                        float l1_tot_0 = l1_s_3;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            l1_tot_0 += __shfl_xor_sync(0xFFFFFFFF, l1_tot_0, offset);
                        float l1_rs_0 = rsqrtf(l1_tot_0 / 3584.0f + eps);
                        if (lane == 0) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(l3_buf_addr + (unsigned int)((64 + rp5_3) * 4)), "r"(__as_u32(l1_rs_0)) : "memory");
                        }
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                #pragma unroll
                for (int rp6 = 0; rp6 < 2; rp6++) {
                    int l1_r6 = l1_ra + rp6;
                    if (l1_r6 < num_tokens) {
                        unsigned int l1_rr[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&l1_rr[0])) : "r"(l3_buf_addr + (unsigned int)((64 + rp6) * 4)));
                        float l1_rstd = __uint_as_float(l1_rr[0]);
                        int l1_own = (l1_r6 - g_e) % 112;
                        unsigned long long l1_nb = (unsigned long long)l1_r6 * 3584;
                        #pragma unroll
                        for (int q6 = 0; q6 < 4; q6++) {
                            int l1_c6 = epi_warp * 4 + q6;
                            if (l1_c6 < 14) {
                                int l1_k6 = (lane + l1_c6 * 32) * 8;
                                float l1_nv[8];
                                #pragma unroll
                                for (int jj6 = 0; jj6 < 4; jj6++) {
                                    float l1_wf[2];
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 lo, hi;\n\t"
                                        "mov.b32 {lo, hi}, %2;\n\t"
                                        "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                        "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                        "}\n"
                                        : "=&f"(l1_wf[0]), "=&f"(l1_wf[1]) : "r"(l1_nw[q6 * 4 + jj6]), "f"(0.0f), "f"(0.0f));
                                    #pragma unroll
                                    for (int h6 = 0; h6 < 2; h6++) {
                                        float l1_s2 = (float)(__nv_bfloat16)l1_acc[(rp6 * 4 + q6) * 8 + jj6 * 2 + h6];
                                        float l1_nm = (float)(__nv_bfloat16)(l1_s2 * l1_rstd);
                                        l1_nv[jj6 * 2 + h6] = (float)(__nv_bfloat16)(l1_wf[h6] * l1_nm);
                                    }
                                }
                                if (l1_own == 0) {
                                    unsigned int l1_pk[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l1_nv[_lp*2 + 0], l1_nv[_lp*2+1 + 0]));
                                        l1_pk[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l1_nv[_lp*2 + 2], l1_nv[_lp*2+1 + 2]));
                                        l1_pk[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l1_nv[_lp*2 + 4], l1_nv[_lp*2+1 + 4]));
                                        l1_pk[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l1_nv[_lp*2 + 6], l1_nv[_lp*2+1 + 6]));
                                        l1_pk[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(y_out + (l1_nb + (unsigned long long)l1_k6))[0] = reinterpret_cast<int4*>(l1_pk)[0];
                                }
                                int l1_cr = l1_c6 * 4 + lane / 8 - c_lo;
                                if (l1_cr >= 0) {
                                    if (l1_cr < c_cnt) {
                                        unsigned int l1_b4[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(l1_nv[_lp*2 + 0], l1_nv[_lp*2+1 + 0]));
                                            l1_b4[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(l1_cr * 1024) + (unsigned int)(l1_r6 * 128 + col_b ^ (l1_r6 * 128 + col_b >> 7 & 7) << 4))), "r"(l1_b4[0]), "r"(l1_b4[1]), "r"(l1_b4[2]), "r"(l1_b4[3]) : "memory");
                                    }
                                }
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
            }
            #pragma unroll 1
            for (int l1_rz = num_tokens; l1_rz < 8; l1_rz++) {
                #pragma unroll
                for (int q7 = 0; q7 < 4; q7++) {
                    int l1_c7 = epi_warp * 4 + q7;
                    if (l1_c7 < 14) {
                        int l1_cz = l1_c7 * 4 + lane / 8 - c_lo;
                        if (l1_cz >= 0) {
                            if (l1_cz < c_cnt) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(l1_cz * 1024) + (unsigned int)(l1_rz * 128 + col_b ^ (l1_rz * 128 + col_b >> 7 & 7) << 4))), "r"(l1_zero), "r"(l1_zero), "r"(l1_zero), "r"(l1_zero) : "memory");
                            }
                        }
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(bn_bar_addr);
                }
            }
            int tile = g_e;
            int slot = 0;
            tile = g_e / 2;
            slot = g_e % 2;
            int cls = ((tile < 0) ? 0 : ((tile < 56) ? 1 : 2));
            int row_r = tile * 128;
            int row_l = tile * 128;
            int row_s = (tile - 56) * 64;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            float red[8];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[8];
            tmem_ld_x8(&_tmem_load_0[0], lane_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            red[0] = _tmem_load_0[0];
            red[1] = _tmem_load_0[1];
            red[2] = _tmem_load_0[2];
            red[3] = _tmem_load_0[3];
            red[4] = _tmem_load_0[4];
            red[5] = _tmem_load_0[5];
            red[6] = _tmem_load_0[6];
            red[7] = _tmem_load_0[7];
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr);
                }
            }
            if (slot == 1) {
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(0));
                uint32_t _mbar_peer_0;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_0) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mbar_peer_0) : "memory");
                uint32_t _mbar_peer_1;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_1) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mbar_peer_1) : "memory");
                fin = 0;
            } else {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(red_bar_addr);
                    }
                }
                mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
                red[0] = red[0] + land[tid_1 * 4];
                red[1] = red[1] + land[tid_1 * 4 + 1];
                red[2] = red[2] + land[tid_1 * 4 + 2];
                red[3] = red[3] + land[tid_1 * 4 + 3];
                red[4] = red[4] + land[(128 + tid_1) * 4];
                red[5] = red[5] + land[(128 + tid_1) * 4 + 1];
                red[6] = red[6] + land[(128 + tid_1) * 4 + 2];
                red[7] = red[7] + land[(128 + tid_1) * 4 + 3];
            }
            if (fin == 1) {
                if (num_tokens > 0) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + tid_1)) + (0)) = (__nv_bfloat16)red[0];
                }
                if (num_tokens > 1) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + 7168 + tid_1)) + (0)) = (__nv_bfloat16)red[1];
                }
                if (num_tokens > 2) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + 14336 + tid_1)) + (0)) = (__nv_bfloat16)red[2];
                }
                if (num_tokens > 3) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + 21504 + tid_1)) + (0)) = (__nv_bfloat16)red[3];
                }
                if (num_tokens > 4) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + 28672 + tid_1)) + (0)) = (__nv_bfloat16)red[4];
                }
                if (num_tokens > 5) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + 35840 + tid_1)) + (0)) = (__nv_bfloat16)red[5];
                }
                if (num_tokens > 6) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + 43008 + tid_1)) + (0)) = (__nv_bfloat16)red[6];
                }
                if (num_tokens > 7) {
                    *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + 50176 + tid_1)) + (0)) = (__nv_bfloat16)red[7];
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int g_l = bid;
            int tl_l = g_l * 16;
            int tile_l = g_l;
            int rank_l = 0;
            tile_l = g_l / 2;
            rank_l = g_l % 2;
            int d_lo_l = ((rank_l == 0) ? 0 : 6);
            int d_cnt_l = ((rank_l == 0) ? 6 : 6);
            int u_lo_l = ((rank_l == 0) ? 0 : 4);
            int u_cnt_l = ((rank_l == 0) ? 4 : 3);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int pro = 0;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_R))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_L))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_S))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_2))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_1))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_2))) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(rows_full_addr, (num_tokens * num_partials + 1) * 7168);
                cp_async_bulk_gmem2smem(smem_v7_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(norm_w) + ((unsigned long long)0 * (unsigned long long)2)), 7168, rows_full_addr);
                #pragma unroll 1
                for (int rr = 0; rr < num_tokens * num_partials; rr++) {
                    cp_async_bulk_gmem2smem(smem_v5_addr + (unsigned int)(rr * 3584 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(routed) + ((unsigned long long)((unsigned long long)rr * 3584) * (unsigned long long)2)), 7168, rows_full_addr);
                }
            }
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                int norm_ok = 0;
                #pragma unroll 1
                for (int s = 0; s < u_count_l; s++) {
                    mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                    int tile_1 = tile_l;
                    int m = ((d_cnt_l > s) ? d_lo_l + s : 12 + u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 0) ? 0 : ((tile_1 < 56) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = tile_1 * 128;
                    int row_s_1 = (tile_1 - 56) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    if (m < 12) {
                        int kb2 = m;
                        tma_3d_gmem2smem(smem_b_addr + stage * 1024, (&B_2), 0, 0, kb2, a_full_addr + (stage) * 8);
                    }
                    if (need_a == 1) {
                        if (m < 12) {
                            int kc2 = m;
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_2), 0, row_l_1, kc2, a_full_addr + (stage) * 8);
                        } else {
                            int kc = k1_off + (m - 12);
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_L), 0, row_l_1, kc, a_full_addr + (stage) * 8);
                        }
                    }
                    int txb = ((m < 12) ? 17408 : 16384);
                    mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, txb);
                    stage += 1;
                    if (stage == 8) { stage = 0; _phase_a_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int g_m = bid;
            int tl_m = g_m * 16;
            int tile_2 = g_m;
            int rank_m = 0;
            tile_2 = g_m / 2;
            rank_m = g_m % 2;
            int u_count_m = ((rank_m == 0) ? 10 : 9);
            int d_cnt_m = ((rank_m == 0) ? 6 : 6);
            unsigned int stage_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                int cls_2 = ((tile_2 < 0) ? 0 : ((tile_2 < 56) ? 1 : 2));
                mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                #pragma unroll 1
                for (int m_1 = 0; m_1 < u_count_m; m_1++) {
                    mbarrier_wait(a_full_addr + (stage_1) * 8, _phase_a_full);
                    if (m_1 == d_cnt_m) {
                        mbarrier_wait(bn_bar_addr, 0);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    int init_sub = init_flag;
                    if (d_cnt_m > m_1) {
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 64;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)) : "memory");
                    } else {
                        unsigned int ub = (unsigned int)(m_1 - d_cnt_m);
                        int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _smem_desc_lo_3 = (((smem_bn_addr) >> 4) & 0x3FFF) + (ub) * 64;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)) : "memory");
                    }
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 8) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
