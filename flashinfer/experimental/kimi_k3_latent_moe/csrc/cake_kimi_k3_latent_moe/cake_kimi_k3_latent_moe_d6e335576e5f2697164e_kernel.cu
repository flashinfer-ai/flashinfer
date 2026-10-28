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
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 32768
#define SMEM_LAND_STRIDE 32768
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 2048
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 116736
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_BN_OFF 116736
#define SMEM_SMEM_BN_STAGE_BYTES 16384
#define SMEM_SMEM_BN_STRIDE 16384
#define SMEM_SMEM_AG_OFF 2048
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 10240
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 231424

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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_d6e335576e5f2697164e(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define a_empty_addr (mbar_base + 56)
    #define acc_full_addr (mbar_base + 112)
    #define acc_empty_addr (mbar_base + 120)
    #define red_bar_addr (mbar_base + 128)
    #define issue_bar_addr (mbar_base + 136)
    #define rows_bar_addr (mbar_base + 144)
    #define bn_bar_addr (mbar_base + 152)
    #define free_bar_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    unsigned int* free_word = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int free_word_addr = smem + 1536;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_a_addr = smem + 2048;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 116736);
    const int smem_b_addr = smem + 116736;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 116736);
    const int smem_bn_addr = smem + 116736;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_ag_addr = smem + 2048;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 10240);
    const int smem_au_addr = smem + 10240;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 7 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // a_empty: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // free_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 128, 32768);
            mbarrier_expect_tx(smem + 160, 8);
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 0) {
        int _tmem_hold = smem + 168;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
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
            int nrow = g_e + epi_warp * 112;
            if (nrow < num_tokens) {
                unsigned long long nrow_base = (unsigned long long)nrow * 3584;
                unsigned long long partial_stride = (unsigned long long)num_tokens * 3584;
                unsigned int nwp[56];
                unsigned int xp[56];
                float nacc[112];
                #pragma unroll
                for (int i = 0; i < 14; i++) {
                    int kk0 = (lane + i * 32) * 8;
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(routed + (nrow_base + (unsigned long long)kk0) + 0);
                        uint4* _vdst_0 = reinterpret_cast<uint4*>(&xp[i * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_0[_blk] = _vptr_0[_blk];
                        }
                    }
                }
                #pragma unroll
                for (int i_1 = 0; i_1 < 14; i_1++) {
                    int kw = (lane + i_1 * 32) * 8;
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(norm_w + kw + 0);
                        uint4* _vdst_1 = reinterpret_cast<uint4*>(&nwp[i_1 * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_1[_blk] = _vptr_1[_blk];
                        }
                    }
                }
                #pragma unroll
                for (int i_2 = 0; i_2 < 56; i_2++) {
                    float _bf16x2_add_f32_0[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %3;\n\t"
                        "add.rn.f32.bf16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(xp[i_2]), "f"(0.0f), "f"(0.0f));
                    nacc[2 * i_2] = _bf16x2_add_f32_0[0];
                    nacc[2 * i_2 + 1] = _bf16x2_add_f32_0[1];
                }
                #pragma unroll 1
                for (int p = 1; p < num_partials; p++) {
                    unsigned long long src_base = (unsigned long long)p * partial_stride + nrow_base;
                    #pragma unroll
                    for (int i_3 = 0; i_3 < 14; i_3++) {
                        int kk = (lane + i_3 * 32) * 8;
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(routed + (src_base + (unsigned long long)kk) + 0);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            nacc[i_3 * 8 + j] = nacc[i_3 * 8 + j] + _vec_load_0[j];
                        }
                    }
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(issue_bar_addr);
                    }
                }
                unsigned int latp[56];
                #pragma unroll
                for (int i_4 = 0; i_4 < 14; i_4++) {
                    float lat8[8];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        lat8[j_1] = nacc[i_4 * 8 + j_1];
                    }
                    unsigned int lat8_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lat8[_lp*2 + 0], lat8[_lp*2+1 + 0]));
                        lat8_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        latp[i_4 * 4 + k] = lat8_bf16[k];
                    }
                }
                float sum_sq = 0.0f;
                #pragma unroll
                for (int i_5 = 0; i_5 < 56; i_5++) {
                    float _bf16x2_add_f32_1[2];
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %3;\n\t"
                        "add.rn.f32.bf16 %1, hi, %4;\n\t"
                        "}\n"
                        : "=&f"(_bf16x2_add_f32_1[0]), "=&f"(_bf16x2_add_f32_1[1]) : "r"(latp[i_5]), "f"(0.0f), "f"(0.0f));
                    sum_sq = __fmaf_rn(_bf16x2_add_f32_1[0], _bf16x2_add_f32_1[0], sum_sq);
                    sum_sq = __fmaf_rn(_bf16x2_add_f32_1[1], _bf16x2_add_f32_1[1], sum_sq);
                }
                float total = sum_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    total += __shfl_xor_sync(0xFFFFFFFF, total, offset);
                float rstd = rsqrtf(total / 3584.0f + eps);
                #pragma unroll
                for (int i_6 = 0; i_6 < 14; i_6++) {
                    int kk2 = (lane + i_6 * 32) * 8;
                    float y8[8];
                    #pragma unroll
                    for (int k_1 = 0; k_1 < 4; k_1++) {
                        float _bf16x2_add_f32_2[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_2[0]), "=&f"(_bf16x2_add_f32_2[1]) : "r"(latp[i_6 * 4 + k_1]), "f"(0.0f), "f"(0.0f));
                        y8[2 * k_1] = _bf16x2_add_f32_2[0] * rstd;
                        y8[2 * k_1 + 1] = _bf16x2_add_f32_2[1] * rstd;
                    }
                    unsigned int y8_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y8[_lp*2 + 0], y8[_lp*2+1 + 0]));
                        y8_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int ow[4];
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 4; k_2++) {
                        uint32_t _bf16x2_mul_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(nwp[i_6 * 4 + k_2]), "r"(y8_bf16[k_2]));
                        ow[k_2] = _bf16x2_mul_0;
                    }
                    reinterpret_cast<int4*>(y_out + (nrow_base + (unsigned long long)kk2))[0] = reinterpret_cast<int4*>(ow)[0];
                }
                asm volatile("fence.release.gpu;" ::: "memory");
            } else if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(issue_bar_addr);
                }
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("fence.release.gpu;" ::: "memory");
                    uint32_t _atomic_inc_old_0;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_inc_old_0) : "l"(&counters[0]), "r"(static_cast<uint32_t>(111)) : "memory");
                    mbarrier_arrive(rows_bar_addr);
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
            float red[128];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned long long zero_free = 0;
            if (slot == 0) {
                if (warp == 0) {
                    if (elect_sync()) {
                        uint32_t _cluster_mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_3) : "r"(free_word_addr), "r"(1));
                        uint32_t _mbar_peer_4;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_4) : "r"(free_bar_addr), "r"(1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_3), "l"(zero_free), "r"(_mbar_peer_4) : "memory");
                    }
                }
            }
            if (slot == 1) {
                if (warp == 0) {
                    if (elect_sync()) {
                        uint32_t _cluster_mapa_5;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_5) : "r"(free_word_addr), "r"(0));
                        uint32_t _mbar_peer_6;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_6) : "r"(free_bar_addr), "r"(0));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_5), "l"(zero_free), "r"(_mbar_peer_6) : "memory");
                    }
                }
            }
            float _tmem_load_0[128];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                : "r"(lane_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                : "r"(lane_addr + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                : "r"(lane_addr + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                : "r"(lane_addr + 96));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            red[0] = _tmem_load_0[0];
            red[1] = _tmem_load_0[1];
            red[2] = _tmem_load_0[2];
            red[3] = _tmem_load_0[3];
            red[4] = _tmem_load_0[4];
            red[5] = _tmem_load_0[5];
            red[6] = _tmem_load_0[6];
            red[7] = _tmem_load_0[7];
            red[8] = _tmem_load_0[8];
            red[9] = _tmem_load_0[9];
            red[10] = _tmem_load_0[10];
            red[11] = _tmem_load_0[11];
            red[12] = _tmem_load_0[12];
            red[13] = _tmem_load_0[13];
            red[14] = _tmem_load_0[14];
            red[15] = _tmem_load_0[15];
            red[16] = _tmem_load_0[16];
            red[17] = _tmem_load_0[17];
            red[18] = _tmem_load_0[18];
            red[19] = _tmem_load_0[19];
            red[20] = _tmem_load_0[20];
            red[21] = _tmem_load_0[21];
            red[22] = _tmem_load_0[22];
            red[23] = _tmem_load_0[23];
            red[24] = _tmem_load_0[24];
            red[25] = _tmem_load_0[25];
            red[26] = _tmem_load_0[26];
            red[27] = _tmem_load_0[27];
            red[28] = _tmem_load_0[28];
            red[29] = _tmem_load_0[29];
            red[30] = _tmem_load_0[30];
            red[31] = _tmem_load_0[31];
            red[32] = _tmem_load_0[32];
            red[33] = _tmem_load_0[33];
            red[34] = _tmem_load_0[34];
            red[35] = _tmem_load_0[35];
            red[36] = _tmem_load_0[36];
            red[37] = _tmem_load_0[37];
            red[38] = _tmem_load_0[38];
            red[39] = _tmem_load_0[39];
            red[40] = _tmem_load_0[40];
            red[41] = _tmem_load_0[41];
            red[42] = _tmem_load_0[42];
            red[43] = _tmem_load_0[43];
            red[44] = _tmem_load_0[44];
            red[45] = _tmem_load_0[45];
            red[46] = _tmem_load_0[46];
            red[47] = _tmem_load_0[47];
            red[48] = _tmem_load_0[48];
            red[49] = _tmem_load_0[49];
            red[50] = _tmem_load_0[50];
            red[51] = _tmem_load_0[51];
            red[52] = _tmem_load_0[52];
            red[53] = _tmem_load_0[53];
            red[54] = _tmem_load_0[54];
            red[55] = _tmem_load_0[55];
            red[56] = _tmem_load_0[56];
            red[57] = _tmem_load_0[57];
            red[58] = _tmem_load_0[58];
            red[59] = _tmem_load_0[59];
            red[60] = _tmem_load_0[60];
            red[61] = _tmem_load_0[61];
            red[62] = _tmem_load_0[62];
            red[63] = _tmem_load_0[63];
            red[64] = _tmem_load_0[64];
            red[65] = _tmem_load_0[65];
            red[66] = _tmem_load_0[66];
            red[67] = _tmem_load_0[67];
            red[68] = _tmem_load_0[68];
            red[69] = _tmem_load_0[69];
            red[70] = _tmem_load_0[70];
            red[71] = _tmem_load_0[71];
            red[72] = _tmem_load_0[72];
            red[73] = _tmem_load_0[73];
            red[74] = _tmem_load_0[74];
            red[75] = _tmem_load_0[75];
            red[76] = _tmem_load_0[76];
            red[77] = _tmem_load_0[77];
            red[78] = _tmem_load_0[78];
            red[79] = _tmem_load_0[79];
            red[80] = _tmem_load_0[80];
            red[81] = _tmem_load_0[81];
            red[82] = _tmem_load_0[82];
            red[83] = _tmem_load_0[83];
            red[84] = _tmem_load_0[84];
            red[85] = _tmem_load_0[85];
            red[86] = _tmem_load_0[86];
            red[87] = _tmem_load_0[87];
            red[88] = _tmem_load_0[88];
            red[89] = _tmem_load_0[89];
            red[90] = _tmem_load_0[90];
            red[91] = _tmem_load_0[91];
            red[92] = _tmem_load_0[92];
            red[93] = _tmem_load_0[93];
            red[94] = _tmem_load_0[94];
            red[95] = _tmem_load_0[95];
            red[96] = _tmem_load_0[96];
            red[97] = _tmem_load_0[97];
            red[98] = _tmem_load_0[98];
            red[99] = _tmem_load_0[99];
            red[100] = _tmem_load_0[100];
            red[101] = _tmem_load_0[101];
            red[102] = _tmem_load_0[102];
            red[103] = _tmem_load_0[103];
            red[104] = _tmem_load_0[104];
            red[105] = _tmem_load_0[105];
            red[106] = _tmem_load_0[106];
            red[107] = _tmem_load_0[107];
            red[108] = _tmem_load_0[108];
            red[109] = _tmem_load_0[109];
            red[110] = _tmem_load_0[110];
            red[111] = _tmem_load_0[111];
            red[112] = _tmem_load_0[112];
            red[113] = _tmem_load_0[113];
            red[114] = _tmem_load_0[114];
            red[115] = _tmem_load_0[115];
            red[116] = _tmem_load_0[116];
            red[117] = _tmem_load_0[117];
            red[118] = _tmem_load_0[118];
            red[119] = _tmem_load_0[119];
            red[120] = _tmem_load_0[120];
            red[121] = _tmem_load_0[121];
            red[122] = _tmem_load_0[122];
            red[123] = _tmem_load_0[123];
            red[124] = _tmem_load_0[124];
            red[125] = _tmem_load_0[125];
            red[126] = _tmem_load_0[126];
            red[127] = _tmem_load_0[127];
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr);
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(free_bar_addr);
                }
            }
            mbarrier_wait_cluster_hint(free_bar_addr, 0, 10000000);
            if (slot == 0) {
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(1));
                uint32_t _mbar_peer_7;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_7) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0), "r"(__float_as_uint(red[64])), "r"(__float_as_uint(red[65])), "r"(__float_as_uint(red[66])), "r"(__float_as_uint(red[67])), "r"(_mbar_peer_7) : "memory");
                uint32_t _mbar_peer_8;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_8) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(red[68])), "r"(__float_as_uint(red[69])), "r"(__float_as_uint(red[70])), "r"(__float_as_uint(red[71])), "r"(_mbar_peer_8) : "memory");
                uint32_t _mbar_peer_9;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_9) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(red[72])), "r"(__float_as_uint(red[73])), "r"(__float_as_uint(red[74])), "r"(__float_as_uint(red[75])), "r"(_mbar_peer_9) : "memory");
                uint32_t _mbar_peer_10;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_10) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(red[76])), "r"(__float_as_uint(red[77])), "r"(__float_as_uint(red[78])), "r"(__float_as_uint(red[79])), "r"(_mbar_peer_10) : "memory");
                uint32_t _mbar_peer_11;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_11) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 8192), "r"(__float_as_uint(red[80])), "r"(__float_as_uint(red[81])), "r"(__float_as_uint(red[82])), "r"(__float_as_uint(red[83])), "r"(_mbar_peer_11) : "memory");
                uint32_t _mbar_peer_12;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_12) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 10240), "r"(__float_as_uint(red[84])), "r"(__float_as_uint(red[85])), "r"(__float_as_uint(red[86])), "r"(__float_as_uint(red[87])), "r"(_mbar_peer_12) : "memory");
                uint32_t _mbar_peer_13;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_13) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 12288), "r"(__float_as_uint(red[88])), "r"(__float_as_uint(red[89])), "r"(__float_as_uint(red[90])), "r"(__float_as_uint(red[91])), "r"(_mbar_peer_13) : "memory");
                uint32_t _mbar_peer_14;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_14) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 14336), "r"(__float_as_uint(red[92])), "r"(__float_as_uint(red[93])), "r"(__float_as_uint(red[94])), "r"(__float_as_uint(red[95])), "r"(_mbar_peer_14) : "memory");
                uint32_t _mbar_peer_15;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_15) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 16384), "r"(__float_as_uint(red[96])), "r"(__float_as_uint(red[97])), "r"(__float_as_uint(red[98])), "r"(__float_as_uint(red[99])), "r"(_mbar_peer_15) : "memory");
                uint32_t _mbar_peer_16;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_16) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 18432), "r"(__float_as_uint(red[100])), "r"(__float_as_uint(red[101])), "r"(__float_as_uint(red[102])), "r"(__float_as_uint(red[103])), "r"(_mbar_peer_16) : "memory");
                uint32_t _mbar_peer_17;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_17) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 20480), "r"(__float_as_uint(red[104])), "r"(__float_as_uint(red[105])), "r"(__float_as_uint(red[106])), "r"(__float_as_uint(red[107])), "r"(_mbar_peer_17) : "memory");
                uint32_t _mbar_peer_18;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_18) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 22528), "r"(__float_as_uint(red[108])), "r"(__float_as_uint(red[109])), "r"(__float_as_uint(red[110])), "r"(__float_as_uint(red[111])), "r"(_mbar_peer_18) : "memory");
                uint32_t _mbar_peer_19;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_19) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 24576), "r"(__float_as_uint(red[112])), "r"(__float_as_uint(red[113])), "r"(__float_as_uint(red[114])), "r"(__float_as_uint(red[115])), "r"(_mbar_peer_19) : "memory");
                uint32_t _mbar_peer_20;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_20) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 26624), "r"(__float_as_uint(red[116])), "r"(__float_as_uint(red[117])), "r"(__float_as_uint(red[118])), "r"(__float_as_uint(red[119])), "r"(_mbar_peer_20) : "memory");
                uint32_t _mbar_peer_21;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_21) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 28672), "r"(__float_as_uint(red[120])), "r"(__float_as_uint(red[121])), "r"(__float_as_uint(red[122])), "r"(__float_as_uint(red[123])), "r"(_mbar_peer_21) : "memory");
                uint32_t _mbar_peer_22;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_22) : "r"(red_bar_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 30720), "r"(__float_as_uint(red[124])), "r"(__float_as_uint(red[125])), "r"(__float_as_uint(red[126])), "r"(__float_as_uint(red[127])), "r"(_mbar_peer_22) : "memory");
            }
            if (slot == 1) {
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(0));
                uint32_t _mbar_peer_23;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_23) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mbar_peer_23) : "memory");
                uint32_t _mbar_peer_24;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_24) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mbar_peer_24) : "memory");
                uint32_t _mbar_peer_25;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_25) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(red[8])), "r"(__float_as_uint(red[9])), "r"(__float_as_uint(red[10])), "r"(__float_as_uint(red[11])), "r"(_mbar_peer_25) : "memory");
                uint32_t _mbar_peer_26;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_26) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(red[12])), "r"(__float_as_uint(red[13])), "r"(__float_as_uint(red[14])), "r"(__float_as_uint(red[15])), "r"(_mbar_peer_26) : "memory");
                uint32_t _mbar_peer_27;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_27) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 8192), "r"(__float_as_uint(red[16])), "r"(__float_as_uint(red[17])), "r"(__float_as_uint(red[18])), "r"(__float_as_uint(red[19])), "r"(_mbar_peer_27) : "memory");
                uint32_t _mbar_peer_28;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_28) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 10240), "r"(__float_as_uint(red[20])), "r"(__float_as_uint(red[21])), "r"(__float_as_uint(red[22])), "r"(__float_as_uint(red[23])), "r"(_mbar_peer_28) : "memory");
                uint32_t _mbar_peer_29;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_29) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 12288), "r"(__float_as_uint(red[24])), "r"(__float_as_uint(red[25])), "r"(__float_as_uint(red[26])), "r"(__float_as_uint(red[27])), "r"(_mbar_peer_29) : "memory");
                uint32_t _mbar_peer_30;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_30) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 14336), "r"(__float_as_uint(red[28])), "r"(__float_as_uint(red[29])), "r"(__float_as_uint(red[30])), "r"(__float_as_uint(red[31])), "r"(_mbar_peer_30) : "memory");
                uint32_t _mbar_peer_31;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_31) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 16384), "r"(__float_as_uint(red[32])), "r"(__float_as_uint(red[33])), "r"(__float_as_uint(red[34])), "r"(__float_as_uint(red[35])), "r"(_mbar_peer_31) : "memory");
                uint32_t _mbar_peer_32;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_32) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 18432), "r"(__float_as_uint(red[36])), "r"(__float_as_uint(red[37])), "r"(__float_as_uint(red[38])), "r"(__float_as_uint(red[39])), "r"(_mbar_peer_32) : "memory");
                uint32_t _mbar_peer_33;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_33) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 20480), "r"(__float_as_uint(red[40])), "r"(__float_as_uint(red[41])), "r"(__float_as_uint(red[42])), "r"(__float_as_uint(red[43])), "r"(_mbar_peer_33) : "memory");
                uint32_t _mbar_peer_34;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_34) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 22528), "r"(__float_as_uint(red[44])), "r"(__float_as_uint(red[45])), "r"(__float_as_uint(red[46])), "r"(__float_as_uint(red[47])), "r"(_mbar_peer_34) : "memory");
                uint32_t _mbar_peer_35;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_35) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 24576), "r"(__float_as_uint(red[48])), "r"(__float_as_uint(red[49])), "r"(__float_as_uint(red[50])), "r"(__float_as_uint(red[51])), "r"(_mbar_peer_35) : "memory");
                uint32_t _mbar_peer_36;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_36) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 26624), "r"(__float_as_uint(red[52])), "r"(__float_as_uint(red[53])), "r"(__float_as_uint(red[54])), "r"(__float_as_uint(red[55])), "r"(_mbar_peer_36) : "memory");
                uint32_t _mbar_peer_37;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_37) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 28672), "r"(__float_as_uint(red[56])), "r"(__float_as_uint(red[57])), "r"(__float_as_uint(red[58])), "r"(__float_as_uint(red[59])), "r"(_mbar_peer_37) : "memory");
                uint32_t _mbar_peer_38;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_38) : "r"(red_bar_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 30720), "r"(__float_as_uint(red[60])), "r"(__float_as_uint(red[61])), "r"(__float_as_uint(red[62])), "r"(__float_as_uint(red[63])), "r"(_mbar_peer_38) : "memory");
            }
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(red_bar_addr);
                }
            }
            mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
            if (slot == 0) {
                red[0] = red[0] + land[tid_1 * 4];
                red[1] = red[1] + land[tid_1 * 4 + 1];
                red[2] = red[2] + land[tid_1 * 4 + 2];
                red[3] = red[3] + land[tid_1 * 4 + 3];
                red[4] = red[4] + land[(128 + tid_1) * 4];
                red[5] = red[5] + land[(128 + tid_1) * 4 + 1];
                red[6] = red[6] + land[(128 + tid_1) * 4 + 2];
                red[7] = red[7] + land[(128 + tid_1) * 4 + 3];
                red[8] = red[8] + land[(256 + tid_1) * 4];
                red[9] = red[9] + land[(256 + tid_1) * 4 + 1];
                red[10] = red[10] + land[(256 + tid_1) * 4 + 2];
                red[11] = red[11] + land[(256 + tid_1) * 4 + 3];
                red[12] = red[12] + land[(384 + tid_1) * 4];
                red[13] = red[13] + land[(384 + tid_1) * 4 + 1];
                red[14] = red[14] + land[(384 + tid_1) * 4 + 2];
                red[15] = red[15] + land[(384 + tid_1) * 4 + 3];
                red[16] = red[16] + land[(512 + tid_1) * 4];
                red[17] = red[17] + land[(512 + tid_1) * 4 + 1];
                red[18] = red[18] + land[(512 + tid_1) * 4 + 2];
                red[19] = red[19] + land[(512 + tid_1) * 4 + 3];
                red[20] = red[20] + land[(640 + tid_1) * 4];
                red[21] = red[21] + land[(640 + tid_1) * 4 + 1];
                red[22] = red[22] + land[(640 + tid_1) * 4 + 2];
                red[23] = red[23] + land[(640 + tid_1) * 4 + 3];
                red[24] = red[24] + land[(768 + tid_1) * 4];
                red[25] = red[25] + land[(768 + tid_1) * 4 + 1];
                red[26] = red[26] + land[(768 + tid_1) * 4 + 2];
                red[27] = red[27] + land[(768 + tid_1) * 4 + 3];
                red[28] = red[28] + land[(896 + tid_1) * 4];
                red[29] = red[29] + land[(896 + tid_1) * 4 + 1];
                red[30] = red[30] + land[(896 + tid_1) * 4 + 2];
                red[31] = red[31] + land[(896 + tid_1) * 4 + 3];
                red[32] = red[32] + land[(1024 + tid_1) * 4];
                red[33] = red[33] + land[(1024 + tid_1) * 4 + 1];
                red[34] = red[34] + land[(1024 + tid_1) * 4 + 2];
                red[35] = red[35] + land[(1024 + tid_1) * 4 + 3];
                red[36] = red[36] + land[(1152 + tid_1) * 4];
                red[37] = red[37] + land[(1152 + tid_1) * 4 + 1];
                red[38] = red[38] + land[(1152 + tid_1) * 4 + 2];
                red[39] = red[39] + land[(1152 + tid_1) * 4 + 3];
                red[40] = red[40] + land[(1280 + tid_1) * 4];
                red[41] = red[41] + land[(1280 + tid_1) * 4 + 1];
                red[42] = red[42] + land[(1280 + tid_1) * 4 + 2];
                red[43] = red[43] + land[(1280 + tid_1) * 4 + 3];
                red[44] = red[44] + land[(1408 + tid_1) * 4];
                red[45] = red[45] + land[(1408 + tid_1) * 4 + 1];
                red[46] = red[46] + land[(1408 + tid_1) * 4 + 2];
                red[47] = red[47] + land[(1408 + tid_1) * 4 + 3];
                red[48] = red[48] + land[(1536 + tid_1) * 4];
                red[49] = red[49] + land[(1536 + tid_1) * 4 + 1];
                red[50] = red[50] + land[(1536 + tid_1) * 4 + 2];
                red[51] = red[51] + land[(1536 + tid_1) * 4 + 3];
                red[52] = red[52] + land[(1664 + tid_1) * 4];
                red[53] = red[53] + land[(1664 + tid_1) * 4 + 1];
                red[54] = red[54] + land[(1664 + tid_1) * 4 + 2];
                red[55] = red[55] + land[(1664 + tid_1) * 4 + 3];
                red[56] = red[56] + land[(1792 + tid_1) * 4];
                red[57] = red[57] + land[(1792 + tid_1) * 4 + 1];
                red[58] = red[58] + land[(1792 + tid_1) * 4 + 2];
                red[59] = red[59] + land[(1792 + tid_1) * 4 + 3];
                red[60] = red[60] + land[(1920 + tid_1) * 4];
                red[61] = red[61] + land[(1920 + tid_1) * 4 + 1];
                red[62] = red[62] + land[(1920 + tid_1) * 4 + 2];
                red[63] = red[63] + land[(1920 + tid_1) * 4 + 3];
            }
            if (slot == 1) {
                red[64] = red[64] + land[tid_1 * 4];
                red[65] = red[65] + land[tid_1 * 4 + 1];
                red[66] = red[66] + land[tid_1 * 4 + 2];
                red[67] = red[67] + land[tid_1 * 4 + 3];
                red[68] = red[68] + land[(128 + tid_1) * 4];
                red[69] = red[69] + land[(128 + tid_1) * 4 + 1];
                red[70] = red[70] + land[(128 + tid_1) * 4 + 2];
                red[71] = red[71] + land[(128 + tid_1) * 4 + 3];
                red[72] = red[72] + land[(256 + tid_1) * 4];
                red[73] = red[73] + land[(256 + tid_1) * 4 + 1];
                red[74] = red[74] + land[(256 + tid_1) * 4 + 2];
                red[75] = red[75] + land[(256 + tid_1) * 4 + 3];
                red[76] = red[76] + land[(384 + tid_1) * 4];
                red[77] = red[77] + land[(384 + tid_1) * 4 + 1];
                red[78] = red[78] + land[(384 + tid_1) * 4 + 2];
                red[79] = red[79] + land[(384 + tid_1) * 4 + 3];
                red[80] = red[80] + land[(512 + tid_1) * 4];
                red[81] = red[81] + land[(512 + tid_1) * 4 + 1];
                red[82] = red[82] + land[(512 + tid_1) * 4 + 2];
                red[83] = red[83] + land[(512 + tid_1) * 4 + 3];
                red[84] = red[84] + land[(640 + tid_1) * 4];
                red[85] = red[85] + land[(640 + tid_1) * 4 + 1];
                red[86] = red[86] + land[(640 + tid_1) * 4 + 2];
                red[87] = red[87] + land[(640 + tid_1) * 4 + 3];
                red[88] = red[88] + land[(768 + tid_1) * 4];
                red[89] = red[89] + land[(768 + tid_1) * 4 + 1];
                red[90] = red[90] + land[(768 + tid_1) * 4 + 2];
                red[91] = red[91] + land[(768 + tid_1) * 4 + 3];
                red[92] = red[92] + land[(896 + tid_1) * 4];
                red[93] = red[93] + land[(896 + tid_1) * 4 + 1];
                red[94] = red[94] + land[(896 + tid_1) * 4 + 2];
                red[95] = red[95] + land[(896 + tid_1) * 4 + 3];
                red[96] = red[96] + land[(1024 + tid_1) * 4];
                red[97] = red[97] + land[(1024 + tid_1) * 4 + 1];
                red[98] = red[98] + land[(1024 + tid_1) * 4 + 2];
                red[99] = red[99] + land[(1024 + tid_1) * 4 + 3];
                red[100] = red[100] + land[(1152 + tid_1) * 4];
                red[101] = red[101] + land[(1152 + tid_1) * 4 + 1];
                red[102] = red[102] + land[(1152 + tid_1) * 4 + 2];
                red[103] = red[103] + land[(1152 + tid_1) * 4 + 3];
                red[104] = red[104] + land[(1280 + tid_1) * 4];
                red[105] = red[105] + land[(1280 + tid_1) * 4 + 1];
                red[106] = red[106] + land[(1280 + tid_1) * 4 + 2];
                red[107] = red[107] + land[(1280 + tid_1) * 4 + 3];
                red[108] = red[108] + land[(1408 + tid_1) * 4];
                red[109] = red[109] + land[(1408 + tid_1) * 4 + 1];
                red[110] = red[110] + land[(1408 + tid_1) * 4 + 2];
                red[111] = red[111] + land[(1408 + tid_1) * 4 + 3];
                red[112] = red[112] + land[(1536 + tid_1) * 4];
                red[113] = red[113] + land[(1536 + tid_1) * 4 + 1];
                red[114] = red[114] + land[(1536 + tid_1) * 4 + 2];
                red[115] = red[115] + land[(1536 + tid_1) * 4 + 3];
                red[116] = red[116] + land[(1664 + tid_1) * 4];
                red[117] = red[117] + land[(1664 + tid_1) * 4 + 1];
                red[118] = red[118] + land[(1664 + tid_1) * 4 + 2];
                red[119] = red[119] + land[(1664 + tid_1) * 4 + 3];
                red[120] = red[120] + land[(1792 + tid_1) * 4];
                red[121] = red[121] + land[(1792 + tid_1) * 4 + 1];
                red[122] = red[122] + land[(1792 + tid_1) * 4 + 2];
                red[123] = red[123] + land[(1792 + tid_1) * 4 + 3];
                red[124] = red[124] + land[(1920 + tid_1) * 4];
                red[125] = red[125] + land[(1920 + tid_1) * 4 + 1];
                red[126] = red[126] + land[(1920 + tid_1) * 4 + 2];
                red[127] = red[127] + land[(1920 + tid_1) * 4 + 3];
            }
            if (fin == 1) {
                if (slot == 0) {
                    if (num_tokens > 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + tid_1)) + (0)) = (__nv_bfloat16)red[0];
                    }
                    if (num_tokens > 1) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (7168 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[1];
                    }
                    if (num_tokens > 2) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (14336 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[2];
                    }
                    if (num_tokens > 3) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (21504 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[3];
                    }
                    if (num_tokens > 4) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (28672 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[4];
                    }
                    if (num_tokens > 5) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (35840 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[5];
                    }
                    if (num_tokens > 6) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (43008 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[6];
                    }
                    if (num_tokens > 7) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (50176 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[7];
                    }
                    if (num_tokens > 8) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (57344 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[8];
                    }
                    if (num_tokens > 9) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (64512 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[9];
                    }
                    if (num_tokens > 10) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (71680 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[10];
                    }
                    if (num_tokens > 11) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (78848 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[11];
                    }
                    if (num_tokens > 12) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (86016 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[12];
                    }
                    if (num_tokens > 13) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (93184 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[13];
                    }
                    if (num_tokens > 14) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (100352 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[14];
                    }
                    if (num_tokens > 15) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (107520 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[15];
                    }
                    if (num_tokens > 16) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (114688 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[16];
                    }
                    if (num_tokens > 17) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (121856 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[17];
                    }
                    if (num_tokens > 18) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (129024 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[18];
                    }
                    if (num_tokens > 19) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (136192 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[19];
                    }
                    if (num_tokens > 20) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (143360 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[20];
                    }
                    if (num_tokens > 21) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (150528 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[21];
                    }
                    if (num_tokens > 22) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (157696 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[22];
                    }
                    if (num_tokens > 23) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (164864 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[23];
                    }
                    if (num_tokens > 24) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (172032 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[24];
                    }
                    if (num_tokens > 25) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (179200 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[25];
                    }
                    if (num_tokens > 26) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (186368 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[26];
                    }
                    if (num_tokens > 27) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (193536 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[27];
                    }
                    if (num_tokens > 28) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (200704 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[28];
                    }
                    if (num_tokens > 29) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (207872 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[29];
                    }
                    if (num_tokens > 30) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (215040 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[30];
                    }
                    if (num_tokens > 31) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (222208 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[31];
                    }
                    if (num_tokens > 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (229376 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[32];
                    }
                    if (num_tokens > 33) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (236544 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[33];
                    }
                    if (num_tokens > 34) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (243712 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[34];
                    }
                    if (num_tokens > 35) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (250880 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[35];
                    }
                    if (num_tokens > 36) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (258048 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[36];
                    }
                    if (num_tokens > 37) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (265216 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[37];
                    }
                    if (num_tokens > 38) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (272384 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[38];
                    }
                    if (num_tokens > 39) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (279552 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[39];
                    }
                    if (num_tokens > 40) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (286720 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[40];
                    }
                    if (num_tokens > 41) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (293888 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[41];
                    }
                    if (num_tokens > 42) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (301056 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[42];
                    }
                    if (num_tokens > 43) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (308224 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[43];
                    }
                    if (num_tokens > 44) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (315392 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[44];
                    }
                    if (num_tokens > 45) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (322560 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[45];
                    }
                    if (num_tokens > 46) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (329728 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[46];
                    }
                    if (num_tokens > 47) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (336896 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[47];
                    }
                    if (num_tokens > 48) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (344064 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[48];
                    }
                    if (num_tokens > 49) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (351232 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[49];
                    }
                    if (num_tokens > 50) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (358400 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[50];
                    }
                    if (num_tokens > 51) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (365568 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[51];
                    }
                    if (num_tokens > 52) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (372736 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[52];
                    }
                    if (num_tokens > 53) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (379904 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[53];
                    }
                    if (num_tokens > 54) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (387072 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[54];
                    }
                    if (num_tokens > 55) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (394240 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[55];
                    }
                    if (num_tokens > 56) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (401408 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[56];
                    }
                    if (num_tokens > 57) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (408576 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[57];
                    }
                    if (num_tokens > 58) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (415744 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[58];
                    }
                    if (num_tokens > 59) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (422912 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[59];
                    }
                    if (num_tokens > 60) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (430080 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[60];
                    }
                    if (num_tokens > 61) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (437248 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[61];
                    }
                    if (num_tokens > 62) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (444416 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[62];
                    }
                    if (num_tokens > 63) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (451584 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[63];
                    }
                }
                if (slot == 1) {
                    if (num_tokens > 64) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (458752 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[64];
                    }
                    if (num_tokens > 65) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (465920 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[65];
                    }
                    if (num_tokens > 66) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (473088 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[66];
                    }
                    if (num_tokens > 67) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (480256 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[67];
                    }
                    if (num_tokens > 68) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (487424 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[68];
                    }
                    if (num_tokens > 69) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (494592 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[69];
                    }
                    if (num_tokens > 70) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (501760 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[70];
                    }
                    if (num_tokens > 71) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (508928 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[71];
                    }
                    if (num_tokens > 72) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (516096 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[72];
                    }
                    if (num_tokens > 73) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (523264 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[73];
                    }
                    if (num_tokens > 74) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (530432 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[74];
                    }
                    if (num_tokens > 75) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (537600 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[75];
                    }
                    if (num_tokens > 76) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (544768 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[76];
                    }
                    if (num_tokens > 77) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (551936 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[77];
                    }
                    if (num_tokens > 78) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (559104 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[78];
                    }
                    if (num_tokens > 79) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (566272 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[79];
                    }
                    if (num_tokens > 80) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (573440 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[80];
                    }
                    if (num_tokens > 81) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (580608 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[81];
                    }
                    if (num_tokens > 82) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (587776 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[82];
                    }
                    if (num_tokens > 83) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (594944 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[83];
                    }
                    if (num_tokens > 84) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (602112 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[84];
                    }
                    if (num_tokens > 85) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (609280 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[85];
                    }
                    if (num_tokens > 86) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (616448 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[86];
                    }
                    if (num_tokens > 87) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (623616 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[87];
                    }
                    if (num_tokens > 88) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (630784 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[88];
                    }
                    if (num_tokens > 89) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (637952 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[89];
                    }
                    if (num_tokens > 90) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (645120 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[90];
                    }
                    if (num_tokens > 91) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (652288 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[91];
                    }
                    if (num_tokens > 92) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (659456 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[92];
                    }
                    if (num_tokens > 93) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (666624 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[93];
                    }
                    if (num_tokens > 94) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (673792 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[94];
                    }
                    if (num_tokens > 95) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (680960 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[95];
                    }
                    if (num_tokens > 96) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (688128 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[96];
                    }
                    if (num_tokens > 97) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (695296 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[97];
                    }
                    if (num_tokens > 98) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (702464 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[98];
                    }
                    if (num_tokens > 99) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (709632 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[99];
                    }
                    if (num_tokens > 100) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (716800 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[100];
                    }
                    if (num_tokens > 101) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (723968 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[101];
                    }
                    if (num_tokens > 102) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (731136 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[102];
                    }
                    if (num_tokens > 103) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (738304 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[103];
                    }
                    if (num_tokens > 104) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (745472 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[104];
                    }
                    if (num_tokens > 105) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (752640 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[105];
                    }
                    if (num_tokens > 106) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (759808 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[106];
                    }
                    if (num_tokens > 107) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (766976 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[107];
                    }
                    if (num_tokens > 108) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (774144 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[108];
                    }
                    if (num_tokens > 109) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (781312 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[109];
                    }
                    if (num_tokens > 110) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (788480 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[110];
                    }
                    if (num_tokens > 111) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (795648 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[111];
                    }
                    if (num_tokens > 112) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (802816 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[112];
                    }
                    if (num_tokens > 113) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (809984 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[113];
                    }
                    if (num_tokens > 114) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (817152 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[114];
                    }
                    if (num_tokens > 115) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (824320 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[115];
                    }
                    if (num_tokens > 116) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (831488 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[116];
                    }
                    if (num_tokens > 117) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (838656 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[117];
                    }
                    if (num_tokens > 118) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (845824 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[118];
                    }
                    if (num_tokens > 119) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (852992 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[119];
                    }
                    if (num_tokens > 120) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (860160 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[120];
                    }
                    if (num_tokens > 121) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (867328 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[121];
                    }
                    if (num_tokens > 122) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (874496 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[122];
                    }
                    if (num_tokens > 123) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (881664 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[123];
                    }
                    if (num_tokens > 124) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (888832 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[124];
                    }
                    if (num_tokens > 125) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (896000 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[125];
                    }
                    if (num_tokens > 126) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (903168 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[126];
                    }
                    if (num_tokens > 127) {
                        *(reinterpret_cast<__nv_bfloat16*>(out_l + (910336 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[127];
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
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
            int d_lo_l = ((rank_l == 0) ? 0 : 48);
            int d_cnt_l = ((rank_l == 0) ? 48 : 48);
            int u_lo_l = ((rank_l == 0) ? 0 : 28);
            int u_cnt_l = ((rank_l == 0) ? 28 : 28);
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
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                int norm_ok = 0;
                #pragma unroll 1
                for (int s = 0; s < u_count_l; s++) {
                    mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                    int tile_1 = tile_l;
                    int m = ((d_cnt_l > s) ? d_lo_l + s : 96 + u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 0) ? 0 : ((tile_1 < 56) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = tile_1 * 128;
                    int row_s_1 = (tile_1 - 56) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    if (m < 96) {
                        int kb2 = m;
                        tma_3d_gmem2smem(smem_b_addr + stage * 16384, (&B_2), 0, 0, kb2, a_full_addr + (stage) * 8);
                    } else {
                        if (norm_ok == 0) {
                            mbarrier_wait(rows_bar_addr, 0);
                            {
                                unsigned int _poll_v_0;
                                asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<unsigned int*>(counters)) : "memory");
                                #pragma unroll 1
                                for (int _poll_trip_0 = 0; _poll_trip_0 < 4194304; _poll_trip_0++) {
                                    if (_poll_v_0 == static_cast<unsigned int>(0)) {
                                        break;
                                    }
                                    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(_poll_v_0) : "l"(reinterpret_cast<unsigned int*>(counters)) : "memory");
                                }
                                if (_poll_v_0 != static_cast<unsigned int>(0)) {
                                    asm volatile("trap;" ::: "memory");
                                }
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                            asm volatile("fence.proxy.async;" ::: "memory");
                            norm_ok = 1;
                        }
                        int kb = k1_off + (m - 96);
                        tma_3d_gmem2smem(smem_b_addr + stage * 16384, (&B_1), 0, 0, kb, a_full_addr + (stage) * 8);
                    }
                    if (need_a == 1) {
                        if (m < 96) {
                            int kc2 = m;
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 16384), "l"((&A_2)), "r"(0), "r"(row_l_1), "r"(kc2),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            int kc = k1_off + (m - 96);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 16384), "l"((&A_L)), "r"(0), "r"(row_l_1), "r"(kc),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 32768);
                    stage += 1;
                    if (stage == 7) { stage = 0; _phase_a_empty ^= 1; }
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
            int u_count_m = ((rank_m == 0) ? 76 : 76);
            int d_cnt_m = ((rank_m == 0) ? 48 : 48);
            unsigned int stage_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                int cls_2 = ((tile_2 < 0) ? 0 : ((tile_2 < 56) ? 1 : 2));
                mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                #pragma unroll 1
                for (int m_1 = 0; m_1 < u_count_m; m_1++) {
                    mbarrier_wait(a_full_addr + (stage_1) * 8, _phase_a_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    int init_sub = init_flag;
                    int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 7) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
