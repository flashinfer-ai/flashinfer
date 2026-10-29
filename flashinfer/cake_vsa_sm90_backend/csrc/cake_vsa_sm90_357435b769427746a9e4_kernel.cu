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

template <typename T, int N>
struct CakeParamArray {
    T v[N];
    __device__ __forceinline__ const T& operator[](int i) const { return v[i]; }
};

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_1_STAGES 1
#define SMEM_MSTATS_OFF 214016
#define SMEM_MSTATS_STAGE_BYTES 1024
#define SMEM_MSTATS_STRIDE 1024
#define SMEM_PART_LO_OFF 99328
#define SMEM_PART_LO_STAGE_BYTES 16384
#define SMEM_PART_LO_STRIDE 16384
#define SMEM_PART_HI_OFF 197632
#define SMEM_PART_HI_STAGE_BYTES 16384
#define SMEM_PART_HI_STRIDE 16384
#define SMEM_PART_HI0_OFF 82944
#define SMEM_PART_HI0_STAGE_BYTES 16384
#define SMEM_PART_HI0_STRIDE 16384
#define SMEM_FLAG_OFF 215040
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_STATS_SMEM_OFF 17408
#define SMEM_STATS_SMEM_STAGE_BYTES 16384
#define SMEM_STATS_SMEM_STRIDE 16384
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 115712
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 215168
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 1

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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(8,1,1) void
kernel_cake_vsa_sm90_357435b769427746a9e4(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, float* __restrict__ Wo, float* __restrict__ Ws, unsigned int* __restrict__ Wc, int g_shift)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define k_full0_addr (mbar_base + 8)
    #define k_full1_addr (mbar_base + 16)
    #define k_full2_addr (mbar_base + 24)
    #define k_full3_addr (mbar_base + 32)
    #define k_full4_addr (mbar_base + 40)
    #define k_full5_addr (mbar_base + 48)
    #define v_full0_addr (mbar_base + 56)
    #define v_full1_addr (mbar_base + 64)
    #define v_full2_addr (mbar_base + 72)
    #define v_full3_addr (mbar_base + 80)
    #define v_full4_addr (mbar_base + 88)
    #define v_full5_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* mstats = reinterpret_cast<float*>(smem_raw + 214016);
    const int mstats_addr = smem + 214016;
    float* part_lo = reinterpret_cast<float*>(smem_raw + 99328);
    const int part_lo_addr = smem + 99328;
    float* part_hi = reinterpret_cast<float*>(smem_raw + 197632);
    const int part_hi_addr = smem + 197632;
    float* part_hi0 = reinterpret_cast<float*>(smem_raw + 82944);
    const int part_hi0_addr = smem + 82944;
    int* flag = reinterpret_cast<int*>(smem_raw + 215040);
    const int flag_addr = smem + 215040;
    float* stats_smem = reinterpret_cast<float*>(smem_raw + 17408);
    const int stats_smem_addr = smem + 17408;
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    __nv_bfloat16* vt_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int vt_smem_addr = smem + 115712;

    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // k_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // k_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // k_full3: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // k_full4: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // k_full5: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // v_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // v_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // v_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // v_full3: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // v_full4: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // v_full5: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // === Task calls (declaration order) ===
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vt))) : "memory"); }
    int item = bid;
    int mb = seqlen_q / 64;
    int plan_base = (item - cta_rank) / 8 * 49;
    int meta = plan[plan_base];
    int cnt = meta;
    int tile = (item - cta_rank) / 8;
    int blk_base = plan_base + 1;
    int sc_split = cta_rank;
    int sc_first = item - sc_split;
    int sc_nsplit = (meta + 5) / 6;
    int sl_base = meta / sc_nsplit;
    int sl_extra = meta - sl_base * sc_nsplit;
    int sl_start = sc_split * sl_base + ((sc_split < sl_extra) ? sc_split : sl_extra);
    cnt = sl_base + ((sc_split < sl_extra) ? 1 : 0);
    if (sc_split >= sc_nsplit) {
        cnt = 0;
        sl_start = 0;
    }
    blk_base = blk_base + sl_start;
    int head = tile / mb;
    int qb = tile - head * mb;
    int q_row = head * seqlen_q + qb * 64;
    int kv_base = (head >> g_shift) * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            int blk_pre[6];
            blk_pre[0] = plan[blk_base];
            blk_pre[1] = plan[blk_base + 1];
            blk_pre[2] = plan[blk_base + 2];
            blk_pre[3] = plan[blk_base + 3];
            blk_pre[4] = plan[blk_base + 4];
            blk_pre[5] = plan[blk_base + 5];
            mbarrier_arrive_expect_tx(q_full_addr, 16384);
            tma_3d_gmem2smem(q_smem_addr, (&Q), 0, q_row, 0, q_full_addr);
            if (cnt > 0) {
                int blk_row_1_0 = kv_base + blk_pre[0] * 64;
                mbarrier_arrive_expect_tx(k_full0_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr, (&K), 0, blk_row_1_0, 0, k_full0_addr);
                mbarrier_arrive_expect_tx(v_full0_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr, (&Vt), 0, 0, blk_row_1_0 / 8, 0, v_full0_addr);
            }
            if (cnt > 1) {
                int blk_row_1_1 = kv_base + blk_pre[1] * 64;
                mbarrier_arrive_expect_tx(k_full1_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 16384, (&K), 0, blk_row_1_1, 0, k_full1_addr);
                mbarrier_arrive_expect_tx(v_full1_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 16384, (&Vt), 0, 0, blk_row_1_1 / 8, 0, v_full1_addr);
            }
            if (cnt > 2) {
                int blk_row_1_2 = kv_base + blk_pre[2] * 64;
                mbarrier_arrive_expect_tx(k_full2_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 32768, (&K), 0, blk_row_1_2, 0, k_full2_addr);
                mbarrier_arrive_expect_tx(v_full2_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 32768, (&Vt), 0, 0, blk_row_1_2 / 8, 0, v_full2_addr);
            }
            if (cnt > 3) {
                int blk_row_1_3 = kv_base + blk_pre[3] * 64;
                mbarrier_arrive_expect_tx(k_full3_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 49152, (&K), 0, blk_row_1_3, 0, k_full3_addr);
                mbarrier_arrive_expect_tx(v_full3_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 49152, (&Vt), 0, 0, blk_row_1_3 / 8, 0, v_full3_addr);
            }
            if (cnt > 4) {
                int blk_row_1_4 = kv_base + blk_pre[4] * 64;
                mbarrier_arrive_expect_tx(k_full4_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 65536, (&K), 0, blk_row_1_4, 0, k_full4_addr);
                mbarrier_arrive_expect_tx(v_full4_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 65536, (&Vt), 0, 0, blk_row_1_4 / 8, 0, v_full4_addr);
            }
            if (cnt > 5) {
                int blk_row_1_5 = kv_base + blk_pre[5] * 64;
                mbarrier_arrive_expect_tx(k_full5_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr + 81920, (&K), 0, blk_row_1_5, 0, k_full5_addr);
                mbarrier_arrive_expect_tx(v_full5_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr + 81920, (&Vt), 0, 0, blk_row_1_5 / 8, 0, v_full5_addr);
            }
        }
    }
    int warp_raw = warp;
    int warp_u = __shfl_sync(0xFFFFFFFF, warp_raw, 0);
    int wg = warp_u / 4;
    int warp_in_wg = warp_u - wg * 4;
    int tid_wg = warp_in_wg * 32 + lane;
    int quad = warp_in_wg * 8 + lane / 4;
    int lim_even = 0;
    int lim_odd = 0;
    if (wg == 0) {
        lim_even = cnt;
    } else {
        lim_odd = cnt;
    }
    int m0_local = warp_in_wg * 16 + lane / 4;
    int m1_local = m0_local + 8;
    float d_o[64];
    float d_qk[32];
    unsigned int p_bf16[16];
    float row_max0 = -CUDART_INF_F;
    float row_max1 = -CUDART_INF_F;
    float row_sum0 = 0.0f;
    float row_sum1 = 0.0f;
    if (cnt <= wg) {
        d_o[0] = 0.0f;
        d_o[1] = 0.0f;
        d_o[2] = 0.0f;
        d_o[3] = 0.0f;
        d_o[4] = 0.0f;
        d_o[5] = 0.0f;
        d_o[6] = 0.0f;
        d_o[7] = 0.0f;
        d_o[8] = 0.0f;
        d_o[9] = 0.0f;
        d_o[10] = 0.0f;
        d_o[11] = 0.0f;
        d_o[12] = 0.0f;
        d_o[13] = 0.0f;
        d_o[14] = 0.0f;
        d_o[15] = 0.0f;
        d_o[16] = 0.0f;
        d_o[17] = 0.0f;
        d_o[18] = 0.0f;
        d_o[19] = 0.0f;
        d_o[20] = 0.0f;
        d_o[21] = 0.0f;
        d_o[22] = 0.0f;
        d_o[23] = 0.0f;
        d_o[24] = 0.0f;
        d_o[25] = 0.0f;
        d_o[26] = 0.0f;
        d_o[27] = 0.0f;
        d_o[28] = 0.0f;
        d_o[29] = 0.0f;
        d_o[30] = 0.0f;
        d_o[31] = 0.0f;
        d_o[32] = 0.0f;
        d_o[33] = 0.0f;
        d_o[34] = 0.0f;
        d_o[35] = 0.0f;
        d_o[36] = 0.0f;
        d_o[37] = 0.0f;
        d_o[38] = 0.0f;
        d_o[39] = 0.0f;
        d_o[40] = 0.0f;
        d_o[41] = 0.0f;
        d_o[42] = 0.0f;
        d_o[43] = 0.0f;
        d_o[44] = 0.0f;
        d_o[45] = 0.0f;
        d_o[46] = 0.0f;
        d_o[47] = 0.0f;
        d_o[48] = 0.0f;
        d_o[49] = 0.0f;
        d_o[50] = 0.0f;
        d_o[51] = 0.0f;
        d_o[52] = 0.0f;
        d_o[53] = 0.0f;
        d_o[54] = 0.0f;
        d_o[55] = 0.0f;
        d_o[56] = 0.0f;
        d_o[57] = 0.0f;
        d_o[58] = 0.0f;
        d_o[59] = 0.0f;
        d_o[60] = 0.0f;
        d_o[61] = 0.0f;
        d_o[62] = 0.0f;
        d_o[63] = 0.0f;
    }
    unsigned int _phase_q_full_0 = 0;
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;
    uint64_t _wgmma_desc_0 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
    uint64_t _wgmma_desc_1 = (((uint64_t)(((k_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
    uint64_t _wgmma_desc_2 = (((uint64_t)(((q_smem_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    uint64_t _wgmma_desc_3 = (((uint64_t)(((k_smem_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
    uint64_t _wgmma_desc_4 = (((uint64_t)(((vt_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    if (lim_even > 0) {
        mbarrier_wait(k_full0_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_b_0)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_0 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_0 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_0 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1), "l"(_wgmma_b_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 2), "l"(_wgmma_b_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 4), "l"(_wgmma_b_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 6), "l"(_wgmma_b_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0 = -CUDART_INF_F;
        float new_max1 = -CUDART_INF_F;
        new_max0 = fmaxf(new_max0, d_qk[0]);
        new_max0 = fmaxf(new_max0, d_qk[1]);
        new_max0 = fmaxf(new_max0, d_qk[4]);
        new_max0 = fmaxf(new_max0, d_qk[5]);
        new_max0 = fmaxf(new_max0, d_qk[8]);
        new_max0 = fmaxf(new_max0, d_qk[9]);
        new_max0 = fmaxf(new_max0, d_qk[12]);
        new_max0 = fmaxf(new_max0, d_qk[13]);
        new_max0 = fmaxf(new_max0, d_qk[16]);
        new_max0 = fmaxf(new_max0, d_qk[17]);
        new_max0 = fmaxf(new_max0, d_qk[20]);
        new_max0 = fmaxf(new_max0, d_qk[21]);
        new_max0 = fmaxf(new_max0, d_qk[24]);
        new_max0 = fmaxf(new_max0, d_qk[25]);
        new_max0 = fmaxf(new_max0, d_qk[28]);
        new_max0 = fmaxf(new_max0, d_qk[29]);
        new_max1 = fmaxf(new_max1, d_qk[2]);
        new_max1 = fmaxf(new_max1, d_qk[3]);
        new_max1 = fmaxf(new_max1, d_qk[6]);
        new_max1 = fmaxf(new_max1, d_qk[7]);
        new_max1 = fmaxf(new_max1, d_qk[10]);
        new_max1 = fmaxf(new_max1, d_qk[11]);
        new_max1 = fmaxf(new_max1, d_qk[14]);
        new_max1 = fmaxf(new_max1, d_qk[15]);
        new_max1 = fmaxf(new_max1, d_qk[18]);
        new_max1 = fmaxf(new_max1, d_qk[19]);
        new_max1 = fmaxf(new_max1, d_qk[22]);
        new_max1 = fmaxf(new_max1, d_qk[23]);
        new_max1 = fmaxf(new_max1, d_qk[26]);
        new_max1 = fmaxf(new_max1, d_qk[27]);
        new_max1 = fmaxf(new_max1, d_qk[30]);
        new_max1 = fmaxf(new_max1, d_qk[31]);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, new_max0, 2);
        new_max0 = fmaxf(new_max0, _shfl_xor_sync_5);
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, new_max0, 1);
        new_max0 = fmaxf(new_max0, _shfl_xor_sync_6);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, new_max1, 2);
        new_max1 = fmaxf(new_max1, _shfl_xor_sync_7);
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, new_max1, 1);
        new_max1 = fmaxf(new_max1, _shfl_xor_sync_8);
        row_max0 = new_max0;
        row_max1 = new_max1;
        float exp2_0 = approx_exp2(d_qk[0] - row_max0);
        float exp2_1_0 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = exp2_0;
        d_qk[1] = exp2_1_0;
        row_sum0 += exp2_0 + exp2_1_0;
        float exp2_1 = approx_exp2(d_qk[2] - row_max1);
        float exp2_1_1 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = exp2_1;
        d_qk[3] = exp2_1_1;
        row_sum1 += exp2_1 + exp2_1_1;
        float exp2_2 = approx_exp2(d_qk[4] - row_max0);
        float exp2_1_2 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = exp2_2;
        d_qk[5] = exp2_1_2;
        row_sum0 += exp2_2 + exp2_1_2;
        float exp2_3 = approx_exp2(d_qk[6] - row_max1);
        float exp2_1_3 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = exp2_3;
        d_qk[7] = exp2_1_3;
        row_sum1 += exp2_3 + exp2_1_3;
        float exp2_4 = approx_exp2(d_qk[8] - row_max0);
        float exp2_1_4 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = exp2_4;
        d_qk[9] = exp2_1_4;
        row_sum0 += exp2_4 + exp2_1_4;
        float exp2_5 = approx_exp2(d_qk[10] - row_max1);
        float exp2_1_5 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = exp2_5;
        d_qk[11] = exp2_1_5;
        row_sum1 += exp2_5 + exp2_1_5;
        float exp2_6 = approx_exp2(d_qk[12] - row_max0);
        float exp2_1_6 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = exp2_6;
        d_qk[13] = exp2_1_6;
        row_sum0 += exp2_6 + exp2_1_6;
        float exp2_7 = approx_exp2(d_qk[14] - row_max1);
        float exp2_1_7 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = exp2_7;
        d_qk[15] = exp2_1_7;
        row_sum1 += exp2_7 + exp2_1_7;
        __nv_bfloat162 _bit_cast_src_9 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_9)[0];
        __nv_bfloat162 _bit_cast_src_10 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_10)[0];
        __nv_bfloat162 _bit_cast_src_11 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_11)[0];
        __nv_bfloat162 _bit_cast_src_12 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_12)[0];
        __nv_bfloat162 _bit_cast_src_13 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_13)[0];
        __nv_bfloat162 _bit_cast_src_14 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_14)[0];
        __nv_bfloat162 _bit_cast_src_15 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_15)[0];
        __nv_bfloat162 _bit_cast_src_16 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_16)[0];
        mbarrier_wait(v_full0_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_2 + 128)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_8 = approx_exp2(d_qk[16] - row_max0);
        float exp2_1_8 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = exp2_8;
        d_qk[17] = exp2_1_8;
        row_sum0 += exp2_8 + exp2_1_8;
        float exp2_9 = approx_exp2(d_qk[18] - row_max1);
        float exp2_1_9 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = exp2_9;
        d_qk[19] = exp2_1_9;
        row_sum1 += exp2_9 + exp2_1_9;
        float exp2_10 = approx_exp2(d_qk[20] - row_max0);
        float exp2_1_10 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = exp2_10;
        d_qk[21] = exp2_1_10;
        row_sum0 += exp2_10 + exp2_1_10;
        float exp2_11 = approx_exp2(d_qk[22] - row_max1);
        float exp2_1_11 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = exp2_11;
        d_qk[23] = exp2_1_11;
        row_sum1 += exp2_11 + exp2_1_11;
        float exp2_12 = approx_exp2(d_qk[24] - row_max0);
        float exp2_1_12 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = exp2_12;
        d_qk[25] = exp2_1_12;
        row_sum0 += exp2_12 + exp2_1_12;
        float exp2_13 = approx_exp2(d_qk[26] - row_max1);
        float exp2_1_13 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = exp2_13;
        d_qk[27] = exp2_1_13;
        row_sum1 += exp2_13 + exp2_1_13;
        float exp2_14 = approx_exp2(d_qk[28] - row_max0);
        float exp2_1_14 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = exp2_14;
        d_qk[29] = exp2_1_14;
        row_sum0 += exp2_14 + exp2_1_14;
        float exp2_15 = approx_exp2(d_qk[30] - row_max1);
        float exp2_1_15 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = exp2_15;
        d_qk[31] = exp2_1_15;
        row_sum1 += exp2_15 + exp2_1_15;
        __nv_bfloat162 _bit_cast_src_17 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_17)[0];
        __nv_bfloat162 _bit_cast_src_18 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_18)[0];
        __nv_bfloat162 _bit_cast_src_19 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_19)[0];
        __nv_bfloat162 _bit_cast_src_20 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_20)[0];
        __nv_bfloat162 _bit_cast_src_21 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_21)[0];
        __nv_bfloat162 _bit_cast_src_22 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_22)[0];
        __nv_bfloat162 _bit_cast_src_23 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_23)[0];
        __nv_bfloat162 _bit_cast_src_24 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_24)[0];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_2 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_2 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_25 = (((uint64_t)(((k_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_25 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_25);
    uint64_t _wgmma_desc_26 = (((uint64_t)(((k_smem_addr + 16384 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk1_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_26 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_26);
    uint64_t _wgmma_desc_27 = (((uint64_t)(((vt_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bv_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_27 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_27);
    if (lim_odd > 1) {
        mbarrier_wait(k_full1_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_bk0_0)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_bk0_0 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_bk0_0 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_bk0_0 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1), "l"(_wgmma_bk1_0)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 2), "l"(_wgmma_bk1_0 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 4), "l"(_wgmma_bk1_0 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 6), "l"(_wgmma_bk1_0 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_0 = -CUDART_INF_F;
        float new_max1_0 = -CUDART_INF_F;
        new_max0_0 = fmaxf(new_max0_0, d_qk[0]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[1]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[4]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[5]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[8]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[9]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[12]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[13]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[16]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[17]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[20]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[21]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[24]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[25]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[28]);
        new_max0_0 = fmaxf(new_max0_0, d_qk[29]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[2]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[3]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[6]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[7]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[10]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[11]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[14]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[15]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[18]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[19]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[22]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[23]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[26]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[27]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[30]);
        new_max1_0 = fmaxf(new_max1_0, d_qk[31]);
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, new_max0_0, 2);
        new_max0_0 = fmaxf(new_max0_0, _shfl_xor_sync_28);
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, new_max0_0, 1);
        new_max0_0 = fmaxf(new_max0_0, _shfl_xor_sync_29);
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, new_max1_0, 2);
        new_max1_0 = fmaxf(new_max1_0, _shfl_xor_sync_30);
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, new_max1_0, 1);
        new_max1_0 = fmaxf(new_max1_0, _shfl_xor_sync_31);
        float merged_max0_0 = fmaxf(row_max0, new_max0_0);
        float merged_max1_0 = fmaxf(row_max1, new_max1_0);
        float _exp2_lo_0 = approx_exp2(row_max0 - merged_max0_0);
        float _exp2_hi_0 = approx_exp2(row_max1 - merged_max1_0);
        d_o[0] = d_o[0] * _exp2_lo_0;
        d_o[1] = d_o[1] * _exp2_lo_0;
        d_o[4] = d_o[4] * _exp2_lo_0;
        d_o[5] = d_o[5] * _exp2_lo_0;
        d_o[8] = d_o[8] * _exp2_lo_0;
        d_o[9] = d_o[9] * _exp2_lo_0;
        d_o[12] = d_o[12] * _exp2_lo_0;
        d_o[13] = d_o[13] * _exp2_lo_0;
        d_o[16] = d_o[16] * _exp2_lo_0;
        d_o[17] = d_o[17] * _exp2_lo_0;
        d_o[20] = d_o[20] * _exp2_lo_0;
        d_o[21] = d_o[21] * _exp2_lo_0;
        d_o[24] = d_o[24] * _exp2_lo_0;
        d_o[25] = d_o[25] * _exp2_lo_0;
        d_o[28] = d_o[28] * _exp2_lo_0;
        d_o[29] = d_o[29] * _exp2_lo_0;
        d_o[32] = d_o[32] * _exp2_lo_0;
        d_o[33] = d_o[33] * _exp2_lo_0;
        d_o[36] = d_o[36] * _exp2_lo_0;
        d_o[37] = d_o[37] * _exp2_lo_0;
        d_o[40] = d_o[40] * _exp2_lo_0;
        d_o[41] = d_o[41] * _exp2_lo_0;
        d_o[44] = d_o[44] * _exp2_lo_0;
        d_o[45] = d_o[45] * _exp2_lo_0;
        d_o[48] = d_o[48] * _exp2_lo_0;
        d_o[49] = d_o[49] * _exp2_lo_0;
        d_o[52] = d_o[52] * _exp2_lo_0;
        d_o[53] = d_o[53] * _exp2_lo_0;
        d_o[56] = d_o[56] * _exp2_lo_0;
        d_o[57] = d_o[57] * _exp2_lo_0;
        d_o[60] = d_o[60] * _exp2_lo_0;
        d_o[61] = d_o[61] * _exp2_lo_0;
        d_o[2] = d_o[2] * _exp2_hi_0;
        d_o[3] = d_o[3] * _exp2_hi_0;
        d_o[6] = d_o[6] * _exp2_hi_0;
        d_o[7] = d_o[7] * _exp2_hi_0;
        d_o[10] = d_o[10] * _exp2_hi_0;
        d_o[11] = d_o[11] * _exp2_hi_0;
        d_o[14] = d_o[14] * _exp2_hi_0;
        d_o[15] = d_o[15] * _exp2_hi_0;
        d_o[18] = d_o[18] * _exp2_hi_0;
        d_o[19] = d_o[19] * _exp2_hi_0;
        d_o[22] = d_o[22] * _exp2_hi_0;
        d_o[23] = d_o[23] * _exp2_hi_0;
        d_o[26] = d_o[26] * _exp2_hi_0;
        d_o[27] = d_o[27] * _exp2_hi_0;
        d_o[30] = d_o[30] * _exp2_hi_0;
        d_o[31] = d_o[31] * _exp2_hi_0;
        d_o[34] = d_o[34] * _exp2_hi_0;
        d_o[35] = d_o[35] * _exp2_hi_0;
        d_o[38] = d_o[38] * _exp2_hi_0;
        d_o[39] = d_o[39] * _exp2_hi_0;
        d_o[42] = d_o[42] * _exp2_hi_0;
        d_o[43] = d_o[43] * _exp2_hi_0;
        d_o[46] = d_o[46] * _exp2_hi_0;
        d_o[47] = d_o[47] * _exp2_hi_0;
        d_o[50] = d_o[50] * _exp2_hi_0;
        d_o[51] = d_o[51] * _exp2_hi_0;
        d_o[54] = d_o[54] * _exp2_hi_0;
        d_o[55] = d_o[55] * _exp2_hi_0;
        d_o[58] = d_o[58] * _exp2_hi_0;
        d_o[59] = d_o[59] * _exp2_hi_0;
        d_o[62] = d_o[62] * _exp2_hi_0;
        d_o[63] = d_o[63] * _exp2_hi_0;
        row_sum0 = row_sum0 * _exp2_lo_0;
        row_sum1 = row_sum1 * _exp2_hi_0;
        row_max0 = merged_max0_0;
        row_max1 = merged_max1_0;
        float exp2_16 = approx_exp2(d_qk[0] - row_max0);
        float exp2_1_16 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = exp2_16;
        d_qk[1] = exp2_1_16;
        row_sum0 += exp2_16 + exp2_1_16;
        float exp2_17 = approx_exp2(d_qk[2] - row_max1);
        float exp2_1_17 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = exp2_17;
        d_qk[3] = exp2_1_17;
        row_sum1 += exp2_17 + exp2_1_17;
        float exp2_18 = approx_exp2(d_qk[4] - row_max0);
        float exp2_1_18 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = exp2_18;
        d_qk[5] = exp2_1_18;
        row_sum0 += exp2_18 + exp2_1_18;
        float exp2_19 = approx_exp2(d_qk[6] - row_max1);
        float exp2_1_19 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = exp2_19;
        d_qk[7] = exp2_1_19;
        row_sum1 += exp2_19 + exp2_1_19;
        float exp2_20 = approx_exp2(d_qk[8] - row_max0);
        float exp2_1_20 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = exp2_20;
        d_qk[9] = exp2_1_20;
        row_sum0 += exp2_20 + exp2_1_20;
        float exp2_21 = approx_exp2(d_qk[10] - row_max1);
        float exp2_1_21 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = exp2_21;
        d_qk[11] = exp2_1_21;
        row_sum1 += exp2_21 + exp2_1_21;
        float exp2_22 = approx_exp2(d_qk[12] - row_max0);
        float exp2_1_22 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = exp2_22;
        d_qk[13] = exp2_1_22;
        row_sum0 += exp2_22 + exp2_1_22;
        float exp2_23 = approx_exp2(d_qk[14] - row_max1);
        float exp2_1_23 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = exp2_23;
        d_qk[15] = exp2_1_23;
        row_sum1 += exp2_23 + exp2_1_23;
        __nv_bfloat162 _bit_cast_src_32 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_32)[0];
        __nv_bfloat162 _bit_cast_src_33 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_33)[0];
        __nv_bfloat162 _bit_cast_src_34 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_34)[0];
        __nv_bfloat162 _bit_cast_src_35 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_35)[0];
        __nv_bfloat162 _bit_cast_src_36 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_36)[0];
        __nv_bfloat162 _bit_cast_src_37 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_37)[0];
        __nv_bfloat162 _bit_cast_src_38 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_38)[0];
        __nv_bfloat162 _bit_cast_src_39 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_39)[0];
        mbarrier_wait(v_full1_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_bv_0)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_bv_0 + 128)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_24 = approx_exp2(d_qk[16] - row_max0);
        float exp2_1_24 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = exp2_24;
        d_qk[17] = exp2_1_24;
        row_sum0 += exp2_24 + exp2_1_24;
        float exp2_25 = approx_exp2(d_qk[18] - row_max1);
        float exp2_1_25 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = exp2_25;
        d_qk[19] = exp2_1_25;
        row_sum1 += exp2_25 + exp2_1_25;
        float exp2_26 = approx_exp2(d_qk[20] - row_max0);
        float exp2_1_26 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = exp2_26;
        d_qk[21] = exp2_1_26;
        row_sum0 += exp2_26 + exp2_1_26;
        float exp2_27 = approx_exp2(d_qk[22] - row_max1);
        float exp2_1_27 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = exp2_27;
        d_qk[23] = exp2_1_27;
        row_sum1 += exp2_27 + exp2_1_27;
        float exp2_28 = approx_exp2(d_qk[24] - row_max0);
        float exp2_1_28 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = exp2_28;
        d_qk[25] = exp2_1_28;
        row_sum0 += exp2_28 + exp2_1_28;
        float exp2_29 = approx_exp2(d_qk[26] - row_max1);
        float exp2_1_29 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = exp2_29;
        d_qk[27] = exp2_1_29;
        row_sum1 += exp2_29 + exp2_1_29;
        float exp2_30 = approx_exp2(d_qk[28] - row_max0);
        float exp2_1_30 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = exp2_30;
        d_qk[29] = exp2_1_30;
        row_sum0 += exp2_30 + exp2_1_30;
        float exp2_31 = approx_exp2(d_qk[30] - row_max1);
        float exp2_1_31 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = exp2_31;
        d_qk[31] = exp2_1_31;
        row_sum1 += exp2_31 + exp2_1_31;
        __nv_bfloat162 _bit_cast_src_40 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_40)[0];
        __nv_bfloat162 _bit_cast_src_41 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_41)[0];
        __nv_bfloat162 _bit_cast_src_42 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_42)[0];
        __nv_bfloat162 _bit_cast_src_43 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
        __nv_bfloat162 _bit_cast_src_44 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
        __nv_bfloat162 _bit_cast_src_45 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
        __nv_bfloat162 _bit_cast_src_46 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_46)[0];
        __nv_bfloat162 _bit_cast_src_47 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_47)[0];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_bv_0 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_bv_0 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_48 = (((uint64_t)(((k_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_48 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_48);
    uint64_t _wgmma_desc_49 = (((uint64_t)(((k_smem_addr + 32768 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk1_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_49 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_49);
    uint64_t _wgmma_desc_50 = (((uint64_t)(((vt_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bv_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_50 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_50);
    if (lim_even > 2) {
        mbarrier_wait(k_full2_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_bk0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_bk0_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_bk0_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_bk0_1 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1), "l"(_wgmma_bk1_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 2), "l"(_wgmma_bk1_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 4), "l"(_wgmma_bk1_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 6), "l"(_wgmma_bk1_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_1 = -CUDART_INF_F;
        float new_max1_1 = -CUDART_INF_F;
        new_max0_1 = fmaxf(new_max0_1, d_qk[0]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[1]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[4]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[5]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[8]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[9]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[12]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[13]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[16]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[17]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[20]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[21]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[24]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[25]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[28]);
        new_max0_1 = fmaxf(new_max0_1, d_qk[29]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[2]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[3]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[6]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[7]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[10]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[11]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[14]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[15]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[18]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[19]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[22]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[23]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[26]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[27]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[30]);
        new_max1_1 = fmaxf(new_max1_1, d_qk[31]);
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, new_max0_1, 2);
        new_max0_1 = fmaxf(new_max0_1, _shfl_xor_sync_51);
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, new_max0_1, 1);
        new_max0_1 = fmaxf(new_max0_1, _shfl_xor_sync_52);
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, new_max1_1, 2);
        new_max1_1 = fmaxf(new_max1_1, _shfl_xor_sync_53);
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, new_max1_1, 1);
        new_max1_1 = fmaxf(new_max1_1, _shfl_xor_sync_54);
        float merged_max0_1 = fmaxf(row_max0, new_max0_1);
        float merged_max1_1 = fmaxf(row_max1, new_max1_1);
        float _exp2_lo_1 = approx_exp2(row_max0 - merged_max0_1);
        float _exp2_hi_1 = approx_exp2(row_max1 - merged_max1_1);
        d_o[0] = d_o[0] * _exp2_lo_1;
        d_o[1] = d_o[1] * _exp2_lo_1;
        d_o[4] = d_o[4] * _exp2_lo_1;
        d_o[5] = d_o[5] * _exp2_lo_1;
        d_o[8] = d_o[8] * _exp2_lo_1;
        d_o[9] = d_o[9] * _exp2_lo_1;
        d_o[12] = d_o[12] * _exp2_lo_1;
        d_o[13] = d_o[13] * _exp2_lo_1;
        d_o[16] = d_o[16] * _exp2_lo_1;
        d_o[17] = d_o[17] * _exp2_lo_1;
        d_o[20] = d_o[20] * _exp2_lo_1;
        d_o[21] = d_o[21] * _exp2_lo_1;
        d_o[24] = d_o[24] * _exp2_lo_1;
        d_o[25] = d_o[25] * _exp2_lo_1;
        d_o[28] = d_o[28] * _exp2_lo_1;
        d_o[29] = d_o[29] * _exp2_lo_1;
        d_o[32] = d_o[32] * _exp2_lo_1;
        d_o[33] = d_o[33] * _exp2_lo_1;
        d_o[36] = d_o[36] * _exp2_lo_1;
        d_o[37] = d_o[37] * _exp2_lo_1;
        d_o[40] = d_o[40] * _exp2_lo_1;
        d_o[41] = d_o[41] * _exp2_lo_1;
        d_o[44] = d_o[44] * _exp2_lo_1;
        d_o[45] = d_o[45] * _exp2_lo_1;
        d_o[48] = d_o[48] * _exp2_lo_1;
        d_o[49] = d_o[49] * _exp2_lo_1;
        d_o[52] = d_o[52] * _exp2_lo_1;
        d_o[53] = d_o[53] * _exp2_lo_1;
        d_o[56] = d_o[56] * _exp2_lo_1;
        d_o[57] = d_o[57] * _exp2_lo_1;
        d_o[60] = d_o[60] * _exp2_lo_1;
        d_o[61] = d_o[61] * _exp2_lo_1;
        d_o[2] = d_o[2] * _exp2_hi_1;
        d_o[3] = d_o[3] * _exp2_hi_1;
        d_o[6] = d_o[6] * _exp2_hi_1;
        d_o[7] = d_o[7] * _exp2_hi_1;
        d_o[10] = d_o[10] * _exp2_hi_1;
        d_o[11] = d_o[11] * _exp2_hi_1;
        d_o[14] = d_o[14] * _exp2_hi_1;
        d_o[15] = d_o[15] * _exp2_hi_1;
        d_o[18] = d_o[18] * _exp2_hi_1;
        d_o[19] = d_o[19] * _exp2_hi_1;
        d_o[22] = d_o[22] * _exp2_hi_1;
        d_o[23] = d_o[23] * _exp2_hi_1;
        d_o[26] = d_o[26] * _exp2_hi_1;
        d_o[27] = d_o[27] * _exp2_hi_1;
        d_o[30] = d_o[30] * _exp2_hi_1;
        d_o[31] = d_o[31] * _exp2_hi_1;
        d_o[34] = d_o[34] * _exp2_hi_1;
        d_o[35] = d_o[35] * _exp2_hi_1;
        d_o[38] = d_o[38] * _exp2_hi_1;
        d_o[39] = d_o[39] * _exp2_hi_1;
        d_o[42] = d_o[42] * _exp2_hi_1;
        d_o[43] = d_o[43] * _exp2_hi_1;
        d_o[46] = d_o[46] * _exp2_hi_1;
        d_o[47] = d_o[47] * _exp2_hi_1;
        d_o[50] = d_o[50] * _exp2_hi_1;
        d_o[51] = d_o[51] * _exp2_hi_1;
        d_o[54] = d_o[54] * _exp2_hi_1;
        d_o[55] = d_o[55] * _exp2_hi_1;
        d_o[58] = d_o[58] * _exp2_hi_1;
        d_o[59] = d_o[59] * _exp2_hi_1;
        d_o[62] = d_o[62] * _exp2_hi_1;
        d_o[63] = d_o[63] * _exp2_hi_1;
        row_sum0 = row_sum0 * _exp2_lo_1;
        row_sum1 = row_sum1 * _exp2_hi_1;
        row_max0 = merged_max0_1;
        row_max1 = merged_max1_1;
        float exp2_32 = approx_exp2(d_qk[0] - row_max0);
        float exp2_1_32 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = exp2_32;
        d_qk[1] = exp2_1_32;
        row_sum0 += exp2_32 + exp2_1_32;
        float exp2_33 = approx_exp2(d_qk[2] - row_max1);
        float exp2_1_33 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = exp2_33;
        d_qk[3] = exp2_1_33;
        row_sum1 += exp2_33 + exp2_1_33;
        float exp2_34 = approx_exp2(d_qk[4] - row_max0);
        float exp2_1_34 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = exp2_34;
        d_qk[5] = exp2_1_34;
        row_sum0 += exp2_34 + exp2_1_34;
        float exp2_35 = approx_exp2(d_qk[6] - row_max1);
        float exp2_1_35 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = exp2_35;
        d_qk[7] = exp2_1_35;
        row_sum1 += exp2_35 + exp2_1_35;
        float exp2_36 = approx_exp2(d_qk[8] - row_max0);
        float exp2_1_36 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = exp2_36;
        d_qk[9] = exp2_1_36;
        row_sum0 += exp2_36 + exp2_1_36;
        float exp2_37 = approx_exp2(d_qk[10] - row_max1);
        float exp2_1_37 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = exp2_37;
        d_qk[11] = exp2_1_37;
        row_sum1 += exp2_37 + exp2_1_37;
        float exp2_38 = approx_exp2(d_qk[12] - row_max0);
        float exp2_1_38 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = exp2_38;
        d_qk[13] = exp2_1_38;
        row_sum0 += exp2_38 + exp2_1_38;
        float exp2_39 = approx_exp2(d_qk[14] - row_max1);
        float exp2_1_39 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = exp2_39;
        d_qk[15] = exp2_1_39;
        row_sum1 += exp2_39 + exp2_1_39;
        __nv_bfloat162 _bit_cast_src_55 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_55)[0];
        __nv_bfloat162 _bit_cast_src_56 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_56)[0];
        __nv_bfloat162 _bit_cast_src_57 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_57)[0];
        __nv_bfloat162 _bit_cast_src_58 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_58)[0];
        __nv_bfloat162 _bit_cast_src_59 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_59)[0];
        __nv_bfloat162 _bit_cast_src_60 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_60)[0];
        __nv_bfloat162 _bit_cast_src_61 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_61)[0];
        __nv_bfloat162 _bit_cast_src_62 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_62)[0];
        mbarrier_wait(v_full2_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_bv_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_bv_1 + 128)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_40 = approx_exp2(d_qk[16] - row_max0);
        float exp2_1_40 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = exp2_40;
        d_qk[17] = exp2_1_40;
        row_sum0 += exp2_40 + exp2_1_40;
        float exp2_41 = approx_exp2(d_qk[18] - row_max1);
        float exp2_1_41 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = exp2_41;
        d_qk[19] = exp2_1_41;
        row_sum1 += exp2_41 + exp2_1_41;
        float exp2_42 = approx_exp2(d_qk[20] - row_max0);
        float exp2_1_42 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = exp2_42;
        d_qk[21] = exp2_1_42;
        row_sum0 += exp2_42 + exp2_1_42;
        float exp2_43 = approx_exp2(d_qk[22] - row_max1);
        float exp2_1_43 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = exp2_43;
        d_qk[23] = exp2_1_43;
        row_sum1 += exp2_43 + exp2_1_43;
        float exp2_44 = approx_exp2(d_qk[24] - row_max0);
        float exp2_1_44 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = exp2_44;
        d_qk[25] = exp2_1_44;
        row_sum0 += exp2_44 + exp2_1_44;
        float exp2_45 = approx_exp2(d_qk[26] - row_max1);
        float exp2_1_45 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = exp2_45;
        d_qk[27] = exp2_1_45;
        row_sum1 += exp2_45 + exp2_1_45;
        float exp2_46 = approx_exp2(d_qk[28] - row_max0);
        float exp2_1_46 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = exp2_46;
        d_qk[29] = exp2_1_46;
        row_sum0 += exp2_46 + exp2_1_46;
        float exp2_47 = approx_exp2(d_qk[30] - row_max1);
        float exp2_1_47 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = exp2_47;
        d_qk[31] = exp2_1_47;
        row_sum1 += exp2_47 + exp2_1_47;
        __nv_bfloat162 _bit_cast_src_63 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_63)[0];
        __nv_bfloat162 _bit_cast_src_64 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_64)[0];
        __nv_bfloat162 _bit_cast_src_65 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_65)[0];
        __nv_bfloat162 _bit_cast_src_66 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_66)[0];
        __nv_bfloat162 _bit_cast_src_67 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_67)[0];
        __nv_bfloat162 _bit_cast_src_68 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_68)[0];
        __nv_bfloat162 _bit_cast_src_69 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_69)[0];
        __nv_bfloat162 _bit_cast_src_70 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_70)[0];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_bv_1 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_bv_1 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_71 = (((uint64_t)(((k_smem_addr + 49152)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_71 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_71);
    uint64_t _wgmma_desc_72 = (((uint64_t)(((k_smem_addr + 49152 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk1_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_72 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_72);
    uint64_t _wgmma_desc_73 = (((uint64_t)(((vt_smem_addr + 49152)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bv_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_73 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_73);
    if (lim_odd > 3) {
        mbarrier_wait(k_full3_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_bk0_2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_bk0_2 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_bk0_2 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_bk0_2 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1), "l"(_wgmma_bk1_2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 2), "l"(_wgmma_bk1_2 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 4), "l"(_wgmma_bk1_2 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 6), "l"(_wgmma_bk1_2 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_2 = -CUDART_INF_F;
        float new_max1_2 = -CUDART_INF_F;
        new_max0_2 = fmaxf(new_max0_2, d_qk[0]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[1]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[4]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[5]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[8]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[9]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[12]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[13]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[16]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[17]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[20]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[21]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[24]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[25]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[28]);
        new_max0_2 = fmaxf(new_max0_2, d_qk[29]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[2]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[3]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[6]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[7]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[10]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[11]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[14]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[15]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[18]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[19]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[22]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[23]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[26]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[27]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[30]);
        new_max1_2 = fmaxf(new_max1_2, d_qk[31]);
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, new_max0_2, 2);
        new_max0_2 = fmaxf(new_max0_2, _shfl_xor_sync_74);
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, new_max0_2, 1);
        new_max0_2 = fmaxf(new_max0_2, _shfl_xor_sync_75);
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, new_max1_2, 2);
        new_max1_2 = fmaxf(new_max1_2, _shfl_xor_sync_76);
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, new_max1_2, 1);
        new_max1_2 = fmaxf(new_max1_2, _shfl_xor_sync_77);
        float merged_max0_2 = fmaxf(row_max0, new_max0_2);
        float merged_max1_2 = fmaxf(row_max1, new_max1_2);
        float _exp2_lo_2 = approx_exp2(row_max0 - merged_max0_2);
        float _exp2_hi_2 = approx_exp2(row_max1 - merged_max1_2);
        d_o[0] = d_o[0] * _exp2_lo_2;
        d_o[1] = d_o[1] * _exp2_lo_2;
        d_o[4] = d_o[4] * _exp2_lo_2;
        d_o[5] = d_o[5] * _exp2_lo_2;
        d_o[8] = d_o[8] * _exp2_lo_2;
        d_o[9] = d_o[9] * _exp2_lo_2;
        d_o[12] = d_o[12] * _exp2_lo_2;
        d_o[13] = d_o[13] * _exp2_lo_2;
        d_o[16] = d_o[16] * _exp2_lo_2;
        d_o[17] = d_o[17] * _exp2_lo_2;
        d_o[20] = d_o[20] * _exp2_lo_2;
        d_o[21] = d_o[21] * _exp2_lo_2;
        d_o[24] = d_o[24] * _exp2_lo_2;
        d_o[25] = d_o[25] * _exp2_lo_2;
        d_o[28] = d_o[28] * _exp2_lo_2;
        d_o[29] = d_o[29] * _exp2_lo_2;
        d_o[32] = d_o[32] * _exp2_lo_2;
        d_o[33] = d_o[33] * _exp2_lo_2;
        d_o[36] = d_o[36] * _exp2_lo_2;
        d_o[37] = d_o[37] * _exp2_lo_2;
        d_o[40] = d_o[40] * _exp2_lo_2;
        d_o[41] = d_o[41] * _exp2_lo_2;
        d_o[44] = d_o[44] * _exp2_lo_2;
        d_o[45] = d_o[45] * _exp2_lo_2;
        d_o[48] = d_o[48] * _exp2_lo_2;
        d_o[49] = d_o[49] * _exp2_lo_2;
        d_o[52] = d_o[52] * _exp2_lo_2;
        d_o[53] = d_o[53] * _exp2_lo_2;
        d_o[56] = d_o[56] * _exp2_lo_2;
        d_o[57] = d_o[57] * _exp2_lo_2;
        d_o[60] = d_o[60] * _exp2_lo_2;
        d_o[61] = d_o[61] * _exp2_lo_2;
        d_o[2] = d_o[2] * _exp2_hi_2;
        d_o[3] = d_o[3] * _exp2_hi_2;
        d_o[6] = d_o[6] * _exp2_hi_2;
        d_o[7] = d_o[7] * _exp2_hi_2;
        d_o[10] = d_o[10] * _exp2_hi_2;
        d_o[11] = d_o[11] * _exp2_hi_2;
        d_o[14] = d_o[14] * _exp2_hi_2;
        d_o[15] = d_o[15] * _exp2_hi_2;
        d_o[18] = d_o[18] * _exp2_hi_2;
        d_o[19] = d_o[19] * _exp2_hi_2;
        d_o[22] = d_o[22] * _exp2_hi_2;
        d_o[23] = d_o[23] * _exp2_hi_2;
        d_o[26] = d_o[26] * _exp2_hi_2;
        d_o[27] = d_o[27] * _exp2_hi_2;
        d_o[30] = d_o[30] * _exp2_hi_2;
        d_o[31] = d_o[31] * _exp2_hi_2;
        d_o[34] = d_o[34] * _exp2_hi_2;
        d_o[35] = d_o[35] * _exp2_hi_2;
        d_o[38] = d_o[38] * _exp2_hi_2;
        d_o[39] = d_o[39] * _exp2_hi_2;
        d_o[42] = d_o[42] * _exp2_hi_2;
        d_o[43] = d_o[43] * _exp2_hi_2;
        d_o[46] = d_o[46] * _exp2_hi_2;
        d_o[47] = d_o[47] * _exp2_hi_2;
        d_o[50] = d_o[50] * _exp2_hi_2;
        d_o[51] = d_o[51] * _exp2_hi_2;
        d_o[54] = d_o[54] * _exp2_hi_2;
        d_o[55] = d_o[55] * _exp2_hi_2;
        d_o[58] = d_o[58] * _exp2_hi_2;
        d_o[59] = d_o[59] * _exp2_hi_2;
        d_o[62] = d_o[62] * _exp2_hi_2;
        d_o[63] = d_o[63] * _exp2_hi_2;
        row_sum0 = row_sum0 * _exp2_lo_2;
        row_sum1 = row_sum1 * _exp2_hi_2;
        row_max0 = merged_max0_2;
        row_max1 = merged_max1_2;
        float exp2_48 = approx_exp2(d_qk[0] - row_max0);
        float exp2_1_48 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = exp2_48;
        d_qk[1] = exp2_1_48;
        row_sum0 += exp2_48 + exp2_1_48;
        float exp2_49 = approx_exp2(d_qk[2] - row_max1);
        float exp2_1_49 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = exp2_49;
        d_qk[3] = exp2_1_49;
        row_sum1 += exp2_49 + exp2_1_49;
        float exp2_50 = approx_exp2(d_qk[4] - row_max0);
        float exp2_1_50 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = exp2_50;
        d_qk[5] = exp2_1_50;
        row_sum0 += exp2_50 + exp2_1_50;
        float exp2_51 = approx_exp2(d_qk[6] - row_max1);
        float exp2_1_51 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = exp2_51;
        d_qk[7] = exp2_1_51;
        row_sum1 += exp2_51 + exp2_1_51;
        float exp2_52 = approx_exp2(d_qk[8] - row_max0);
        float exp2_1_52 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = exp2_52;
        d_qk[9] = exp2_1_52;
        row_sum0 += exp2_52 + exp2_1_52;
        float exp2_53 = approx_exp2(d_qk[10] - row_max1);
        float exp2_1_53 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = exp2_53;
        d_qk[11] = exp2_1_53;
        row_sum1 += exp2_53 + exp2_1_53;
        float exp2_54 = approx_exp2(d_qk[12] - row_max0);
        float exp2_1_54 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = exp2_54;
        d_qk[13] = exp2_1_54;
        row_sum0 += exp2_54 + exp2_1_54;
        float exp2_55 = approx_exp2(d_qk[14] - row_max1);
        float exp2_1_55 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = exp2_55;
        d_qk[15] = exp2_1_55;
        row_sum1 += exp2_55 + exp2_1_55;
        __nv_bfloat162 _bit_cast_src_78 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_78)[0];
        __nv_bfloat162 _bit_cast_src_79 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_79)[0];
        __nv_bfloat162 _bit_cast_src_80 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_80)[0];
        __nv_bfloat162 _bit_cast_src_81 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_81)[0];
        __nv_bfloat162 _bit_cast_src_82 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_82)[0];
        __nv_bfloat162 _bit_cast_src_83 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_83)[0];
        __nv_bfloat162 _bit_cast_src_84 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_84)[0];
        __nv_bfloat162 _bit_cast_src_85 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_85)[0];
        mbarrier_wait(v_full3_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_bv_2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_bv_2 + 128)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_56 = approx_exp2(d_qk[16] - row_max0);
        float exp2_1_56 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = exp2_56;
        d_qk[17] = exp2_1_56;
        row_sum0 += exp2_56 + exp2_1_56;
        float exp2_57 = approx_exp2(d_qk[18] - row_max1);
        float exp2_1_57 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = exp2_57;
        d_qk[19] = exp2_1_57;
        row_sum1 += exp2_57 + exp2_1_57;
        float exp2_58 = approx_exp2(d_qk[20] - row_max0);
        float exp2_1_58 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = exp2_58;
        d_qk[21] = exp2_1_58;
        row_sum0 += exp2_58 + exp2_1_58;
        float exp2_59 = approx_exp2(d_qk[22] - row_max1);
        float exp2_1_59 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = exp2_59;
        d_qk[23] = exp2_1_59;
        row_sum1 += exp2_59 + exp2_1_59;
        float exp2_60 = approx_exp2(d_qk[24] - row_max0);
        float exp2_1_60 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = exp2_60;
        d_qk[25] = exp2_1_60;
        row_sum0 += exp2_60 + exp2_1_60;
        float exp2_61 = approx_exp2(d_qk[26] - row_max1);
        float exp2_1_61 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = exp2_61;
        d_qk[27] = exp2_1_61;
        row_sum1 += exp2_61 + exp2_1_61;
        float exp2_62 = approx_exp2(d_qk[28] - row_max0);
        float exp2_1_62 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = exp2_62;
        d_qk[29] = exp2_1_62;
        row_sum0 += exp2_62 + exp2_1_62;
        float exp2_63 = approx_exp2(d_qk[30] - row_max1);
        float exp2_1_63 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = exp2_63;
        d_qk[31] = exp2_1_63;
        row_sum1 += exp2_63 + exp2_1_63;
        __nv_bfloat162 _bit_cast_src_86 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_86)[0];
        __nv_bfloat162 _bit_cast_src_87 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_87)[0];
        __nv_bfloat162 _bit_cast_src_88 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_88)[0];
        __nv_bfloat162 _bit_cast_src_89 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_89)[0];
        __nv_bfloat162 _bit_cast_src_90 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_90)[0];
        __nv_bfloat162 _bit_cast_src_91 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_91)[0];
        __nv_bfloat162 _bit_cast_src_92 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_92)[0];
        __nv_bfloat162 _bit_cast_src_93 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_93)[0];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_bv_2 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_bv_2 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_94 = (((uint64_t)(((k_smem_addr + 65536)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_94 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_94);
    uint64_t _wgmma_desc_95 = (((uint64_t)(((k_smem_addr + 65536 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk1_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_95 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_95);
    uint64_t _wgmma_desc_96 = (((uint64_t)(((vt_smem_addr + 65536)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bv_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_96 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_96);
    if (lim_even > 4) {
        mbarrier_wait(k_full4_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_bk0_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_bk0_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_bk0_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_bk0_3 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1), "l"(_wgmma_bk1_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 2), "l"(_wgmma_bk1_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 4), "l"(_wgmma_bk1_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 6), "l"(_wgmma_bk1_3 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_3 = -CUDART_INF_F;
        float new_max1_3 = -CUDART_INF_F;
        new_max0_3 = fmaxf(new_max0_3, d_qk[0]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[1]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[4]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[5]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[8]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[9]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[12]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[13]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[16]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[17]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[20]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[21]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[24]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[25]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[28]);
        new_max0_3 = fmaxf(new_max0_3, d_qk[29]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[2]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[3]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[6]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[7]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[10]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[11]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[14]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[15]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[18]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[19]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[22]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[23]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[26]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[27]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[30]);
        new_max1_3 = fmaxf(new_max1_3, d_qk[31]);
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, new_max0_3, 2);
        new_max0_3 = fmaxf(new_max0_3, _shfl_xor_sync_97);
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, new_max0_3, 1);
        new_max0_3 = fmaxf(new_max0_3, _shfl_xor_sync_98);
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, new_max1_3, 2);
        new_max1_3 = fmaxf(new_max1_3, _shfl_xor_sync_99);
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, new_max1_3, 1);
        new_max1_3 = fmaxf(new_max1_3, _shfl_xor_sync_100);
        float merged_max0_3 = fmaxf(row_max0, new_max0_3);
        float merged_max1_3 = fmaxf(row_max1, new_max1_3);
        float _exp2_lo_3 = approx_exp2(row_max0 - merged_max0_3);
        float _exp2_hi_3 = approx_exp2(row_max1 - merged_max1_3);
        d_o[0] = d_o[0] * _exp2_lo_3;
        d_o[1] = d_o[1] * _exp2_lo_3;
        d_o[4] = d_o[4] * _exp2_lo_3;
        d_o[5] = d_o[5] * _exp2_lo_3;
        d_o[8] = d_o[8] * _exp2_lo_3;
        d_o[9] = d_o[9] * _exp2_lo_3;
        d_o[12] = d_o[12] * _exp2_lo_3;
        d_o[13] = d_o[13] * _exp2_lo_3;
        d_o[16] = d_o[16] * _exp2_lo_3;
        d_o[17] = d_o[17] * _exp2_lo_3;
        d_o[20] = d_o[20] * _exp2_lo_3;
        d_o[21] = d_o[21] * _exp2_lo_3;
        d_o[24] = d_o[24] * _exp2_lo_3;
        d_o[25] = d_o[25] * _exp2_lo_3;
        d_o[28] = d_o[28] * _exp2_lo_3;
        d_o[29] = d_o[29] * _exp2_lo_3;
        d_o[32] = d_o[32] * _exp2_lo_3;
        d_o[33] = d_o[33] * _exp2_lo_3;
        d_o[36] = d_o[36] * _exp2_lo_3;
        d_o[37] = d_o[37] * _exp2_lo_3;
        d_o[40] = d_o[40] * _exp2_lo_3;
        d_o[41] = d_o[41] * _exp2_lo_3;
        d_o[44] = d_o[44] * _exp2_lo_3;
        d_o[45] = d_o[45] * _exp2_lo_3;
        d_o[48] = d_o[48] * _exp2_lo_3;
        d_o[49] = d_o[49] * _exp2_lo_3;
        d_o[52] = d_o[52] * _exp2_lo_3;
        d_o[53] = d_o[53] * _exp2_lo_3;
        d_o[56] = d_o[56] * _exp2_lo_3;
        d_o[57] = d_o[57] * _exp2_lo_3;
        d_o[60] = d_o[60] * _exp2_lo_3;
        d_o[61] = d_o[61] * _exp2_lo_3;
        d_o[2] = d_o[2] * _exp2_hi_3;
        d_o[3] = d_o[3] * _exp2_hi_3;
        d_o[6] = d_o[6] * _exp2_hi_3;
        d_o[7] = d_o[7] * _exp2_hi_3;
        d_o[10] = d_o[10] * _exp2_hi_3;
        d_o[11] = d_o[11] * _exp2_hi_3;
        d_o[14] = d_o[14] * _exp2_hi_3;
        d_o[15] = d_o[15] * _exp2_hi_3;
        d_o[18] = d_o[18] * _exp2_hi_3;
        d_o[19] = d_o[19] * _exp2_hi_3;
        d_o[22] = d_o[22] * _exp2_hi_3;
        d_o[23] = d_o[23] * _exp2_hi_3;
        d_o[26] = d_o[26] * _exp2_hi_3;
        d_o[27] = d_o[27] * _exp2_hi_3;
        d_o[30] = d_o[30] * _exp2_hi_3;
        d_o[31] = d_o[31] * _exp2_hi_3;
        d_o[34] = d_o[34] * _exp2_hi_3;
        d_o[35] = d_o[35] * _exp2_hi_3;
        d_o[38] = d_o[38] * _exp2_hi_3;
        d_o[39] = d_o[39] * _exp2_hi_3;
        d_o[42] = d_o[42] * _exp2_hi_3;
        d_o[43] = d_o[43] * _exp2_hi_3;
        d_o[46] = d_o[46] * _exp2_hi_3;
        d_o[47] = d_o[47] * _exp2_hi_3;
        d_o[50] = d_o[50] * _exp2_hi_3;
        d_o[51] = d_o[51] * _exp2_hi_3;
        d_o[54] = d_o[54] * _exp2_hi_3;
        d_o[55] = d_o[55] * _exp2_hi_3;
        d_o[58] = d_o[58] * _exp2_hi_3;
        d_o[59] = d_o[59] * _exp2_hi_3;
        d_o[62] = d_o[62] * _exp2_hi_3;
        d_o[63] = d_o[63] * _exp2_hi_3;
        row_sum0 = row_sum0 * _exp2_lo_3;
        row_sum1 = row_sum1 * _exp2_hi_3;
        row_max0 = merged_max0_3;
        row_max1 = merged_max1_3;
        float exp2_64 = approx_exp2(d_qk[0] - row_max0);
        float exp2_1_64 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = exp2_64;
        d_qk[1] = exp2_1_64;
        row_sum0 += exp2_64 + exp2_1_64;
        float exp2_65 = approx_exp2(d_qk[2] - row_max1);
        float exp2_1_65 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = exp2_65;
        d_qk[3] = exp2_1_65;
        row_sum1 += exp2_65 + exp2_1_65;
        float exp2_66 = approx_exp2(d_qk[4] - row_max0);
        float exp2_1_66 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = exp2_66;
        d_qk[5] = exp2_1_66;
        row_sum0 += exp2_66 + exp2_1_66;
        float exp2_67 = approx_exp2(d_qk[6] - row_max1);
        float exp2_1_67 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = exp2_67;
        d_qk[7] = exp2_1_67;
        row_sum1 += exp2_67 + exp2_1_67;
        float exp2_68 = approx_exp2(d_qk[8] - row_max0);
        float exp2_1_68 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = exp2_68;
        d_qk[9] = exp2_1_68;
        row_sum0 += exp2_68 + exp2_1_68;
        float exp2_69 = approx_exp2(d_qk[10] - row_max1);
        float exp2_1_69 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = exp2_69;
        d_qk[11] = exp2_1_69;
        row_sum1 += exp2_69 + exp2_1_69;
        float exp2_70 = approx_exp2(d_qk[12] - row_max0);
        float exp2_1_70 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = exp2_70;
        d_qk[13] = exp2_1_70;
        row_sum0 += exp2_70 + exp2_1_70;
        float exp2_71 = approx_exp2(d_qk[14] - row_max1);
        float exp2_1_71 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = exp2_71;
        d_qk[15] = exp2_1_71;
        row_sum1 += exp2_71 + exp2_1_71;
        __nv_bfloat162 _bit_cast_src_101 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_101)[0];
        __nv_bfloat162 _bit_cast_src_102 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_102)[0];
        __nv_bfloat162 _bit_cast_src_103 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_103)[0];
        __nv_bfloat162 _bit_cast_src_104 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_104)[0];
        __nv_bfloat162 _bit_cast_src_105 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_105)[0];
        __nv_bfloat162 _bit_cast_src_106 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_106)[0];
        __nv_bfloat162 _bit_cast_src_107 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_107)[0];
        __nv_bfloat162 _bit_cast_src_108 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_108)[0];
        mbarrier_wait(v_full4_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_bv_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_bv_3 + 128)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_72 = approx_exp2(d_qk[16] - row_max0);
        float exp2_1_72 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = exp2_72;
        d_qk[17] = exp2_1_72;
        row_sum0 += exp2_72 + exp2_1_72;
        float exp2_73 = approx_exp2(d_qk[18] - row_max1);
        float exp2_1_73 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = exp2_73;
        d_qk[19] = exp2_1_73;
        row_sum1 += exp2_73 + exp2_1_73;
        float exp2_74 = approx_exp2(d_qk[20] - row_max0);
        float exp2_1_74 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = exp2_74;
        d_qk[21] = exp2_1_74;
        row_sum0 += exp2_74 + exp2_1_74;
        float exp2_75 = approx_exp2(d_qk[22] - row_max1);
        float exp2_1_75 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = exp2_75;
        d_qk[23] = exp2_1_75;
        row_sum1 += exp2_75 + exp2_1_75;
        float exp2_76 = approx_exp2(d_qk[24] - row_max0);
        float exp2_1_76 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = exp2_76;
        d_qk[25] = exp2_1_76;
        row_sum0 += exp2_76 + exp2_1_76;
        float exp2_77 = approx_exp2(d_qk[26] - row_max1);
        float exp2_1_77 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = exp2_77;
        d_qk[27] = exp2_1_77;
        row_sum1 += exp2_77 + exp2_1_77;
        float exp2_78 = approx_exp2(d_qk[28] - row_max0);
        float exp2_1_78 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = exp2_78;
        d_qk[29] = exp2_1_78;
        row_sum0 += exp2_78 + exp2_1_78;
        float exp2_79 = approx_exp2(d_qk[30] - row_max1);
        float exp2_1_79 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = exp2_79;
        d_qk[31] = exp2_1_79;
        row_sum1 += exp2_79 + exp2_1_79;
        __nv_bfloat162 _bit_cast_src_109 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_109)[0];
        __nv_bfloat162 _bit_cast_src_110 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_110)[0];
        __nv_bfloat162 _bit_cast_src_111 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_111)[0];
        __nv_bfloat162 _bit_cast_src_112 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_112)[0];
        __nv_bfloat162 _bit_cast_src_113 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_113)[0];
        __nv_bfloat162 _bit_cast_src_114 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_114)[0];
        __nv_bfloat162 _bit_cast_src_115 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_115)[0];
        __nv_bfloat162 _bit_cast_src_116 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_116)[0];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_bv_3 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_bv_3 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_117 = (((uint64_t)(((k_smem_addr + 81920)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_117 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_117);
    uint64_t _wgmma_desc_118 = (((uint64_t)(((k_smem_addr + 81920 + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bk1_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_118 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_118);
    uint64_t _wgmma_desc_119 = (((uint64_t)(((vt_smem_addr + 81920)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_bv_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_119 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_119);
    if (lim_odd > 5) {
        mbarrier_wait(k_full5_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_bk0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_bk0_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_bk0_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_bk0_4 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1), "l"(_wgmma_bk1_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 2), "l"(_wgmma_bk1_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 4), "l"(_wgmma_bk1_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_1 + 6), "l"(_wgmma_bk1_4 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_qk[0] = d_qk[0] * scale_log2;
        d_qk[1] = d_qk[1] * scale_log2;
        d_qk[2] = d_qk[2] * scale_log2;
        d_qk[3] = d_qk[3] * scale_log2;
        d_qk[4] = d_qk[4] * scale_log2;
        d_qk[5] = d_qk[5] * scale_log2;
        d_qk[6] = d_qk[6] * scale_log2;
        d_qk[7] = d_qk[7] * scale_log2;
        d_qk[8] = d_qk[8] * scale_log2;
        d_qk[9] = d_qk[9] * scale_log2;
        d_qk[10] = d_qk[10] * scale_log2;
        d_qk[11] = d_qk[11] * scale_log2;
        d_qk[12] = d_qk[12] * scale_log2;
        d_qk[13] = d_qk[13] * scale_log2;
        d_qk[14] = d_qk[14] * scale_log2;
        d_qk[15] = d_qk[15] * scale_log2;
        d_qk[16] = d_qk[16] * scale_log2;
        d_qk[17] = d_qk[17] * scale_log2;
        d_qk[18] = d_qk[18] * scale_log2;
        d_qk[19] = d_qk[19] * scale_log2;
        d_qk[20] = d_qk[20] * scale_log2;
        d_qk[21] = d_qk[21] * scale_log2;
        d_qk[22] = d_qk[22] * scale_log2;
        d_qk[23] = d_qk[23] * scale_log2;
        d_qk[24] = d_qk[24] * scale_log2;
        d_qk[25] = d_qk[25] * scale_log2;
        d_qk[26] = d_qk[26] * scale_log2;
        d_qk[27] = d_qk[27] * scale_log2;
        d_qk[28] = d_qk[28] * scale_log2;
        d_qk[29] = d_qk[29] * scale_log2;
        d_qk[30] = d_qk[30] * scale_log2;
        d_qk[31] = d_qk[31] * scale_log2;
        float new_max0_4 = -CUDART_INF_F;
        float new_max1_4 = -CUDART_INF_F;
        new_max0_4 = fmaxf(new_max0_4, d_qk[0]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[1]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[4]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[5]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[8]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[9]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[12]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[13]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[16]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[17]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[20]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[21]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[24]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[25]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[28]);
        new_max0_4 = fmaxf(new_max0_4, d_qk[29]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[2]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[3]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[6]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[7]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[10]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[11]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[14]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[15]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[18]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[19]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[22]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[23]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[26]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[27]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[30]);
        new_max1_4 = fmaxf(new_max1_4, d_qk[31]);
        float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, new_max0_4, 2);
        new_max0_4 = fmaxf(new_max0_4, _shfl_xor_sync_120);
        float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, new_max0_4, 1);
        new_max0_4 = fmaxf(new_max0_4, _shfl_xor_sync_121);
        float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, new_max1_4, 2);
        new_max1_4 = fmaxf(new_max1_4, _shfl_xor_sync_122);
        float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, new_max1_4, 1);
        new_max1_4 = fmaxf(new_max1_4, _shfl_xor_sync_123);
        float merged_max0_4 = fmaxf(row_max0, new_max0_4);
        float merged_max1_4 = fmaxf(row_max1, new_max1_4);
        float _exp2_lo_4 = approx_exp2(row_max0 - merged_max0_4);
        float _exp2_hi_4 = approx_exp2(row_max1 - merged_max1_4);
        d_o[0] = d_o[0] * _exp2_lo_4;
        d_o[1] = d_o[1] * _exp2_lo_4;
        d_o[4] = d_o[4] * _exp2_lo_4;
        d_o[5] = d_o[5] * _exp2_lo_4;
        d_o[8] = d_o[8] * _exp2_lo_4;
        d_o[9] = d_o[9] * _exp2_lo_4;
        d_o[12] = d_o[12] * _exp2_lo_4;
        d_o[13] = d_o[13] * _exp2_lo_4;
        d_o[16] = d_o[16] * _exp2_lo_4;
        d_o[17] = d_o[17] * _exp2_lo_4;
        d_o[20] = d_o[20] * _exp2_lo_4;
        d_o[21] = d_o[21] * _exp2_lo_4;
        d_o[24] = d_o[24] * _exp2_lo_4;
        d_o[25] = d_o[25] * _exp2_lo_4;
        d_o[28] = d_o[28] * _exp2_lo_4;
        d_o[29] = d_o[29] * _exp2_lo_4;
        d_o[32] = d_o[32] * _exp2_lo_4;
        d_o[33] = d_o[33] * _exp2_lo_4;
        d_o[36] = d_o[36] * _exp2_lo_4;
        d_o[37] = d_o[37] * _exp2_lo_4;
        d_o[40] = d_o[40] * _exp2_lo_4;
        d_o[41] = d_o[41] * _exp2_lo_4;
        d_o[44] = d_o[44] * _exp2_lo_4;
        d_o[45] = d_o[45] * _exp2_lo_4;
        d_o[48] = d_o[48] * _exp2_lo_4;
        d_o[49] = d_o[49] * _exp2_lo_4;
        d_o[52] = d_o[52] * _exp2_lo_4;
        d_o[53] = d_o[53] * _exp2_lo_4;
        d_o[56] = d_o[56] * _exp2_lo_4;
        d_o[57] = d_o[57] * _exp2_lo_4;
        d_o[60] = d_o[60] * _exp2_lo_4;
        d_o[61] = d_o[61] * _exp2_lo_4;
        d_o[2] = d_o[2] * _exp2_hi_4;
        d_o[3] = d_o[3] * _exp2_hi_4;
        d_o[6] = d_o[6] * _exp2_hi_4;
        d_o[7] = d_o[7] * _exp2_hi_4;
        d_o[10] = d_o[10] * _exp2_hi_4;
        d_o[11] = d_o[11] * _exp2_hi_4;
        d_o[14] = d_o[14] * _exp2_hi_4;
        d_o[15] = d_o[15] * _exp2_hi_4;
        d_o[18] = d_o[18] * _exp2_hi_4;
        d_o[19] = d_o[19] * _exp2_hi_4;
        d_o[22] = d_o[22] * _exp2_hi_4;
        d_o[23] = d_o[23] * _exp2_hi_4;
        d_o[26] = d_o[26] * _exp2_hi_4;
        d_o[27] = d_o[27] * _exp2_hi_4;
        d_o[30] = d_o[30] * _exp2_hi_4;
        d_o[31] = d_o[31] * _exp2_hi_4;
        d_o[34] = d_o[34] * _exp2_hi_4;
        d_o[35] = d_o[35] * _exp2_hi_4;
        d_o[38] = d_o[38] * _exp2_hi_4;
        d_o[39] = d_o[39] * _exp2_hi_4;
        d_o[42] = d_o[42] * _exp2_hi_4;
        d_o[43] = d_o[43] * _exp2_hi_4;
        d_o[46] = d_o[46] * _exp2_hi_4;
        d_o[47] = d_o[47] * _exp2_hi_4;
        d_o[50] = d_o[50] * _exp2_hi_4;
        d_o[51] = d_o[51] * _exp2_hi_4;
        d_o[54] = d_o[54] * _exp2_hi_4;
        d_o[55] = d_o[55] * _exp2_hi_4;
        d_o[58] = d_o[58] * _exp2_hi_4;
        d_o[59] = d_o[59] * _exp2_hi_4;
        d_o[62] = d_o[62] * _exp2_hi_4;
        d_o[63] = d_o[63] * _exp2_hi_4;
        row_sum0 = row_sum0 * _exp2_lo_4;
        row_sum1 = row_sum1 * _exp2_hi_4;
        row_max0 = merged_max0_4;
        row_max1 = merged_max1_4;
        float exp2_80 = approx_exp2(d_qk[0] - row_max0);
        float exp2_1_80 = approx_exp2(d_qk[1] - row_max0);
        d_qk[0] = exp2_80;
        d_qk[1] = exp2_1_80;
        row_sum0 += exp2_80 + exp2_1_80;
        float exp2_81 = approx_exp2(d_qk[2] - row_max1);
        float exp2_1_81 = approx_exp2(d_qk[3] - row_max1);
        d_qk[2] = exp2_81;
        d_qk[3] = exp2_1_81;
        row_sum1 += exp2_81 + exp2_1_81;
        float exp2_82 = approx_exp2(d_qk[4] - row_max0);
        float exp2_1_82 = approx_exp2(d_qk[5] - row_max0);
        d_qk[4] = exp2_82;
        d_qk[5] = exp2_1_82;
        row_sum0 += exp2_82 + exp2_1_82;
        float exp2_83 = approx_exp2(d_qk[6] - row_max1);
        float exp2_1_83 = approx_exp2(d_qk[7] - row_max1);
        d_qk[6] = exp2_83;
        d_qk[7] = exp2_1_83;
        row_sum1 += exp2_83 + exp2_1_83;
        float exp2_84 = approx_exp2(d_qk[8] - row_max0);
        float exp2_1_84 = approx_exp2(d_qk[9] - row_max0);
        d_qk[8] = exp2_84;
        d_qk[9] = exp2_1_84;
        row_sum0 += exp2_84 + exp2_1_84;
        float exp2_85 = approx_exp2(d_qk[10] - row_max1);
        float exp2_1_85 = approx_exp2(d_qk[11] - row_max1);
        d_qk[10] = exp2_85;
        d_qk[11] = exp2_1_85;
        row_sum1 += exp2_85 + exp2_1_85;
        float exp2_86 = approx_exp2(d_qk[12] - row_max0);
        float exp2_1_86 = approx_exp2(d_qk[13] - row_max0);
        d_qk[12] = exp2_86;
        d_qk[13] = exp2_1_86;
        row_sum0 += exp2_86 + exp2_1_86;
        float exp2_87 = approx_exp2(d_qk[14] - row_max1);
        float exp2_1_87 = approx_exp2(d_qk[15] - row_max1);
        d_qk[14] = exp2_87;
        d_qk[15] = exp2_1_87;
        row_sum1 += exp2_87 + exp2_1_87;
        __nv_bfloat162 _bit_cast_src_124 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_124)[0];
        __nv_bfloat162 _bit_cast_src_125 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_125)[0];
        __nv_bfloat162 _bit_cast_src_126 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_126)[0];
        __nv_bfloat162 _bit_cast_src_127 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_127)[0];
        __nv_bfloat162 _bit_cast_src_128 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_128)[0];
        __nv_bfloat162 _bit_cast_src_129 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
        __nv_bfloat162 _bit_cast_src_130 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
        __nv_bfloat162 _bit_cast_src_131 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_131)[0];
        mbarrier_wait(v_full5_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_bv_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_bv_4 + 128)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_88 = approx_exp2(d_qk[16] - row_max0);
        float exp2_1_88 = approx_exp2(d_qk[17] - row_max0);
        d_qk[16] = exp2_88;
        d_qk[17] = exp2_1_88;
        row_sum0 += exp2_88 + exp2_1_88;
        float exp2_89 = approx_exp2(d_qk[18] - row_max1);
        float exp2_1_89 = approx_exp2(d_qk[19] - row_max1);
        d_qk[18] = exp2_89;
        d_qk[19] = exp2_1_89;
        row_sum1 += exp2_89 + exp2_1_89;
        float exp2_90 = approx_exp2(d_qk[20] - row_max0);
        float exp2_1_90 = approx_exp2(d_qk[21] - row_max0);
        d_qk[20] = exp2_90;
        d_qk[21] = exp2_1_90;
        row_sum0 += exp2_90 + exp2_1_90;
        float exp2_91 = approx_exp2(d_qk[22] - row_max1);
        float exp2_1_91 = approx_exp2(d_qk[23] - row_max1);
        d_qk[22] = exp2_91;
        d_qk[23] = exp2_1_91;
        row_sum1 += exp2_91 + exp2_1_91;
        float exp2_92 = approx_exp2(d_qk[24] - row_max0);
        float exp2_1_92 = approx_exp2(d_qk[25] - row_max0);
        d_qk[24] = exp2_92;
        d_qk[25] = exp2_1_92;
        row_sum0 += exp2_92 + exp2_1_92;
        float exp2_93 = approx_exp2(d_qk[26] - row_max1);
        float exp2_1_93 = approx_exp2(d_qk[27] - row_max1);
        d_qk[26] = exp2_93;
        d_qk[27] = exp2_1_93;
        row_sum1 += exp2_93 + exp2_1_93;
        float exp2_94 = approx_exp2(d_qk[28] - row_max0);
        float exp2_1_94 = approx_exp2(d_qk[29] - row_max0);
        d_qk[28] = exp2_94;
        d_qk[29] = exp2_1_94;
        row_sum0 += exp2_94 + exp2_1_94;
        float exp2_95 = approx_exp2(d_qk[30] - row_max1);
        float exp2_1_95 = approx_exp2(d_qk[31] - row_max1);
        d_qk[30] = exp2_95;
        d_qk[31] = exp2_1_95;
        row_sum1 += exp2_95 + exp2_1_95;
        __nv_bfloat162 _bit_cast_src_132 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_132)[0];
        __nv_bfloat162 _bit_cast_src_133 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_133)[0];
        __nv_bfloat162 _bit_cast_src_134 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_134)[0];
        __nv_bfloat162 _bit_cast_src_135 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_135)[0];
        __nv_bfloat162 _bit_cast_src_136 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_136)[0];
        __nv_bfloat162 _bit_cast_src_137 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_137)[0];
        __nv_bfloat162 _bit_cast_src_138 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_138)[0];
        __nv_bfloat162 _bit_cast_src_139 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_139)[0];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_bv_4 + 256)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_bv_4 + 384)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_sync_140;
    float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_sync_141;
    float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_sync_142;
    float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_sync_143;
    int qj = lane & 3;
    if (wg == 1) {
        part_lo[tid_wg] = d_o[0];
        part_lo[128 + tid_wg] = d_o[1];
        part_lo[256 + tid_wg] = d_o[2];
        part_lo[384 + tid_wg] = d_o[3];
        part_lo[512 + tid_wg] = d_o[4];
        part_lo[640 + tid_wg] = d_o[5];
        part_lo[768 + tid_wg] = d_o[6];
        part_lo[896 + tid_wg] = d_o[7];
        part_lo[1024 + tid_wg] = d_o[8];
        part_lo[1152 + tid_wg] = d_o[9];
        part_lo[1280 + tid_wg] = d_o[10];
        part_lo[1408 + tid_wg] = d_o[11];
        part_lo[1536 + tid_wg] = d_o[12];
        part_lo[1664 + tid_wg] = d_o[13];
        part_lo[1792 + tid_wg] = d_o[14];
        part_lo[1920 + tid_wg] = d_o[15];
        part_lo[2048 + tid_wg] = d_o[16];
        part_lo[2176 + tid_wg] = d_o[17];
        part_lo[2304 + tid_wg] = d_o[18];
        part_lo[2432 + tid_wg] = d_o[19];
        part_lo[2560 + tid_wg] = d_o[20];
        part_lo[2688 + tid_wg] = d_o[21];
        part_lo[2816 + tid_wg] = d_o[22];
        part_lo[2944 + tid_wg] = d_o[23];
        part_lo[3072 + tid_wg] = d_o[24];
        part_lo[3200 + tid_wg] = d_o[25];
        part_lo[3328 + tid_wg] = d_o[26];
        part_lo[3456 + tid_wg] = d_o[27];
        part_lo[3584 + tid_wg] = d_o[28];
        part_lo[3712 + tid_wg] = d_o[29];
        part_lo[3840 + tid_wg] = d_o[30];
        part_lo[3968 + tid_wg] = d_o[31];
        part_hi[tid_wg] = d_o[32];
        part_hi[128 + tid_wg] = d_o[33];
        part_hi[256 + tid_wg] = d_o[34];
        part_hi[384 + tid_wg] = d_o[35];
        part_hi[512 + tid_wg] = d_o[36];
        part_hi[640 + tid_wg] = d_o[37];
        part_hi[768 + tid_wg] = d_o[38];
        part_hi[896 + tid_wg] = d_o[39];
        part_hi[1024 + tid_wg] = d_o[40];
        part_hi[1152 + tid_wg] = d_o[41];
        part_hi[1280 + tid_wg] = d_o[42];
        part_hi[1408 + tid_wg] = d_o[43];
        part_hi[1536 + tid_wg] = d_o[44];
        part_hi[1664 + tid_wg] = d_o[45];
        part_hi[1792 + tid_wg] = d_o[46];
        part_hi[1920 + tid_wg] = d_o[47];
        part_hi[2048 + tid_wg] = d_o[48];
        part_hi[2176 + tid_wg] = d_o[49];
        part_hi[2304 + tid_wg] = d_o[50];
        part_hi[2432 + tid_wg] = d_o[51];
        part_hi[2560 + tid_wg] = d_o[52];
        part_hi[2688 + tid_wg] = d_o[53];
        part_hi[2816 + tid_wg] = d_o[54];
        part_hi[2944 + tid_wg] = d_o[55];
        part_hi[3072 + tid_wg] = d_o[56];
        part_hi[3200 + tid_wg] = d_o[57];
        part_hi[3328 + tid_wg] = d_o[58];
        part_hi[3456 + tid_wg] = d_o[59];
        part_hi[3584 + tid_wg] = d_o[60];
        part_hi[3712 + tid_wg] = d_o[61];
        part_hi[3840 + tid_wg] = d_o[62];
        part_hi[3968 + tid_wg] = d_o[63];
        if (qj == 0) {
            mstats[quad * 4] = row_max0;
            mstats[quad * 4 + 1] = row_max1;
            mstats[quad * 4 + 2] = row_sum0;
            mstats[quad * 4 + 3] = row_sum1;
        }
    }
    asm volatile("barrier.sync 2, 256;" ::: "memory");
    if (wg == 0) {
        float pm0 = mstats[quad * 4];
        float pm1 = mstats[quad * 4 + 1];
        float pl0 = mstats[quad * 4 + 2];
        float pl1 = mstats[quad * 4 + 3];
        float mm0 = fmaxf(row_max0, pm0);
        float mm1 = fmaxf(row_max1, pm1);
        float _exp2_144 = approx_exp2(row_max0 - mm0);
        float a0 = ((row_max0 == -CUDART_INF_F) ? 0.0f : _exp2_144);
        float _exp2_145 = approx_exp2(row_max1 - mm1);
        float a1 = ((row_max1 == -CUDART_INF_F) ? 0.0f : _exp2_145);
        float _exp2_146 = approx_exp2(pm0 - mm0);
        float b0 = ((pm0 == -CUDART_INF_F) ? 0.0f : _exp2_146);
        float _exp2_147 = approx_exp2(pm1 - mm1);
        float b1 = ((pm1 == -CUDART_INF_F) ? 0.0f : _exp2_147);
        d_o[0] = __fmaf_rn(part_lo[tid_wg], b0, d_o[0] * a0);
        d_o[1] = __fmaf_rn(part_lo[128 + tid_wg], b0, d_o[1] * a0);
        d_o[2] = __fmaf_rn(part_lo[256 + tid_wg], b1, d_o[2] * a1);
        d_o[3] = __fmaf_rn(part_lo[384 + tid_wg], b1, d_o[3] * a1);
        d_o[4] = __fmaf_rn(part_lo[512 + tid_wg], b0, d_o[4] * a0);
        d_o[5] = __fmaf_rn(part_lo[640 + tid_wg], b0, d_o[5] * a0);
        d_o[6] = __fmaf_rn(part_lo[768 + tid_wg], b1, d_o[6] * a1);
        d_o[7] = __fmaf_rn(part_lo[896 + tid_wg], b1, d_o[7] * a1);
        d_o[8] = __fmaf_rn(part_lo[1024 + tid_wg], b0, d_o[8] * a0);
        d_o[9] = __fmaf_rn(part_lo[1152 + tid_wg], b0, d_o[9] * a0);
        d_o[10] = __fmaf_rn(part_lo[1280 + tid_wg], b1, d_o[10] * a1);
        d_o[11] = __fmaf_rn(part_lo[1408 + tid_wg], b1, d_o[11] * a1);
        d_o[12] = __fmaf_rn(part_lo[1536 + tid_wg], b0, d_o[12] * a0);
        d_o[13] = __fmaf_rn(part_lo[1664 + tid_wg], b0, d_o[13] * a0);
        d_o[14] = __fmaf_rn(part_lo[1792 + tid_wg], b1, d_o[14] * a1);
        d_o[15] = __fmaf_rn(part_lo[1920 + tid_wg], b1, d_o[15] * a1);
        d_o[16] = __fmaf_rn(part_lo[2048 + tid_wg], b0, d_o[16] * a0);
        d_o[17] = __fmaf_rn(part_lo[2176 + tid_wg], b0, d_o[17] * a0);
        d_o[18] = __fmaf_rn(part_lo[2304 + tid_wg], b1, d_o[18] * a1);
        d_o[19] = __fmaf_rn(part_lo[2432 + tid_wg], b1, d_o[19] * a1);
        d_o[20] = __fmaf_rn(part_lo[2560 + tid_wg], b0, d_o[20] * a0);
        d_o[21] = __fmaf_rn(part_lo[2688 + tid_wg], b0, d_o[21] * a0);
        d_o[22] = __fmaf_rn(part_lo[2816 + tid_wg], b1, d_o[22] * a1);
        d_o[23] = __fmaf_rn(part_lo[2944 + tid_wg], b1, d_o[23] * a1);
        d_o[24] = __fmaf_rn(part_lo[3072 + tid_wg], b0, d_o[24] * a0);
        d_o[25] = __fmaf_rn(part_lo[3200 + tid_wg], b0, d_o[25] * a0);
        d_o[26] = __fmaf_rn(part_lo[3328 + tid_wg], b1, d_o[26] * a1);
        d_o[27] = __fmaf_rn(part_lo[3456 + tid_wg], b1, d_o[27] * a1);
        d_o[28] = __fmaf_rn(part_lo[3584 + tid_wg], b0, d_o[28] * a0);
        d_o[29] = __fmaf_rn(part_lo[3712 + tid_wg], b0, d_o[29] * a0);
        d_o[30] = __fmaf_rn(part_lo[3840 + tid_wg], b1, d_o[30] * a1);
        d_o[31] = __fmaf_rn(part_lo[3968 + tid_wg], b1, d_o[31] * a1);
        d_o[32] = __fmaf_rn(part_hi[tid_wg], b0, d_o[32] * a0);
        d_o[33] = __fmaf_rn(part_hi[128 + tid_wg], b0, d_o[33] * a0);
        d_o[34] = __fmaf_rn(part_hi[256 + tid_wg], b1, d_o[34] * a1);
        d_o[35] = __fmaf_rn(part_hi[384 + tid_wg], b1, d_o[35] * a1);
        d_o[36] = __fmaf_rn(part_hi[512 + tid_wg], b0, d_o[36] * a0);
        d_o[37] = __fmaf_rn(part_hi[640 + tid_wg], b0, d_o[37] * a0);
        d_o[38] = __fmaf_rn(part_hi[768 + tid_wg], b1, d_o[38] * a1);
        d_o[39] = __fmaf_rn(part_hi[896 + tid_wg], b1, d_o[39] * a1);
        d_o[40] = __fmaf_rn(part_hi[1024 + tid_wg], b0, d_o[40] * a0);
        d_o[41] = __fmaf_rn(part_hi[1152 + tid_wg], b0, d_o[41] * a0);
        d_o[42] = __fmaf_rn(part_hi[1280 + tid_wg], b1, d_o[42] * a1);
        d_o[43] = __fmaf_rn(part_hi[1408 + tid_wg], b1, d_o[43] * a1);
        d_o[44] = __fmaf_rn(part_hi[1536 + tid_wg], b0, d_o[44] * a0);
        d_o[45] = __fmaf_rn(part_hi[1664 + tid_wg], b0, d_o[45] * a0);
        d_o[46] = __fmaf_rn(part_hi[1792 + tid_wg], b1, d_o[46] * a1);
        d_o[47] = __fmaf_rn(part_hi[1920 + tid_wg], b1, d_o[47] * a1);
        d_o[48] = __fmaf_rn(part_hi[2048 + tid_wg], b0, d_o[48] * a0);
        d_o[49] = __fmaf_rn(part_hi[2176 + tid_wg], b0, d_o[49] * a0);
        d_o[50] = __fmaf_rn(part_hi[2304 + tid_wg], b1, d_o[50] * a1);
        d_o[51] = __fmaf_rn(part_hi[2432 + tid_wg], b1, d_o[51] * a1);
        d_o[52] = __fmaf_rn(part_hi[2560 + tid_wg], b0, d_o[52] * a0);
        d_o[53] = __fmaf_rn(part_hi[2688 + tid_wg], b0, d_o[53] * a0);
        d_o[54] = __fmaf_rn(part_hi[2816 + tid_wg], b1, d_o[54] * a1);
        d_o[55] = __fmaf_rn(part_hi[2944 + tid_wg], b1, d_o[55] * a1);
        d_o[56] = __fmaf_rn(part_hi[3072 + tid_wg], b0, d_o[56] * a0);
        d_o[57] = __fmaf_rn(part_hi[3200 + tid_wg], b0, d_o[57] * a0);
        d_o[58] = __fmaf_rn(part_hi[3328 + tid_wg], b1, d_o[58] * a1);
        d_o[59] = __fmaf_rn(part_hi[3456 + tid_wg], b1, d_o[59] * a1);
        d_o[60] = __fmaf_rn(part_hi[3584 + tid_wg], b0, d_o[60] * a0);
        d_o[61] = __fmaf_rn(part_hi[3712 + tid_wg], b0, d_o[61] * a0);
        d_o[62] = __fmaf_rn(part_hi[3840 + tid_wg], b1, d_o[62] * a1);
        d_o[63] = __fmaf_rn(part_hi[3968 + tid_wg], b1, d_o[63] * a1);
        row_sum0 = __fmaf_rn(pl0, b0, row_sum0 * a0);
        row_sum1 = __fmaf_rn(pl1, b1, row_sum1 * a1);
        row_max0 = mm0;
        row_max1 = mm1;
    }
    int owner_wg = 1;
    if (wg != 0) {
        owner_wg = 0;
    }
    if (owner_wg != 0) {
        if (sc_nsplit > 1) {
            if (cnt > 0) {
                unsigned int w_vec[4];
                int item_base = item * 8192 + tid * 4;
                float _bit_cast_src_148 = d_o[0];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_148)[0];
                float _bit_cast_src_149 = d_o[1];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_149)[0];
                float _bit_cast_src_150 = d_o[2];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_150)[0];
                float _bit_cast_src_151 = d_o[3];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_151)[0];
                reinterpret_cast<int4*>(Wo + item_base)[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_152 = d_o[4];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_152)[0];
                float _bit_cast_src_153 = d_o[5];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_153)[0];
                float _bit_cast_src_154 = d_o[6];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_154)[0];
                float _bit_cast_src_155 = d_o[7];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_155)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 512))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_156 = d_o[8];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_156)[0];
                float _bit_cast_src_157 = d_o[9];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_157)[0];
                float _bit_cast_src_158 = d_o[10];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_158)[0];
                float _bit_cast_src_159 = d_o[11];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_159)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 1024))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_160 = d_o[12];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_160)[0];
                float _bit_cast_src_161 = d_o[13];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_161)[0];
                float _bit_cast_src_162 = d_o[14];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_162)[0];
                float _bit_cast_src_163 = d_o[15];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_163)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 1536))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_164 = d_o[16];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_164)[0];
                float _bit_cast_src_165 = d_o[17];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_165)[0];
                float _bit_cast_src_166 = d_o[18];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_166)[0];
                float _bit_cast_src_167 = d_o[19];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_167)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 2048))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_168 = d_o[20];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_168)[0];
                float _bit_cast_src_169 = d_o[21];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_169)[0];
                float _bit_cast_src_170 = d_o[22];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_170)[0];
                float _bit_cast_src_171 = d_o[23];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_171)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 2560))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_172 = d_o[24];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_172)[0];
                float _bit_cast_src_173 = d_o[25];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_173)[0];
                float _bit_cast_src_174 = d_o[26];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_174)[0];
                float _bit_cast_src_175 = d_o[27];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_175)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 3072))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_176 = d_o[28];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_176)[0];
                float _bit_cast_src_177 = d_o[29];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_177)[0];
                float _bit_cast_src_178 = d_o[30];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_178)[0];
                float _bit_cast_src_179 = d_o[31];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_179)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 3584))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_180 = d_o[32];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_180)[0];
                float _bit_cast_src_181 = d_o[33];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_181)[0];
                float _bit_cast_src_182 = d_o[34];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_182)[0];
                float _bit_cast_src_183 = d_o[35];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_183)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 4096))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_184 = d_o[36];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_184)[0];
                float _bit_cast_src_185 = d_o[37];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_185)[0];
                float _bit_cast_src_186 = d_o[38];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_186)[0];
                float _bit_cast_src_187 = d_o[39];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_187)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 4608))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_188 = d_o[40];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_188)[0];
                float _bit_cast_src_189 = d_o[41];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_189)[0];
                float _bit_cast_src_190 = d_o[42];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_190)[0];
                float _bit_cast_src_191 = d_o[43];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_191)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 5120))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_192 = d_o[44];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_192)[0];
                float _bit_cast_src_193 = d_o[45];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_193)[0];
                float _bit_cast_src_194 = d_o[46];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_194)[0];
                float _bit_cast_src_195 = d_o[47];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_195)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 5632))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_196 = d_o[48];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_196)[0];
                float _bit_cast_src_197 = d_o[49];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_197)[0];
                float _bit_cast_src_198 = d_o[50];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_198)[0];
                float _bit_cast_src_199 = d_o[51];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_199)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 6144))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_200 = d_o[52];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_200)[0];
                float _bit_cast_src_201 = d_o[53];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_201)[0];
                float _bit_cast_src_202 = d_o[54];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_202)[0];
                float _bit_cast_src_203 = d_o[55];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_203)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 6656))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_204 = d_o[56];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_204)[0];
                float _bit_cast_src_205 = d_o[57];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_205)[0];
                float _bit_cast_src_206 = d_o[58];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_206)[0];
                float _bit_cast_src_207 = d_o[59];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_207)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 7168))[0] = reinterpret_cast<int4*>(w_vec)[0];
                float _bit_cast_src_208 = d_o[60];
                w_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_208)[0];
                float _bit_cast_src_209 = d_o[61];
                w_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_209)[0];
                float _bit_cast_src_210 = d_o[62];
                w_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_210)[0];
                float _bit_cast_src_211 = d_o[63];
                w_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_211)[0];
                reinterpret_cast<int4*>(Wo + (item_base + 7680))[0] = reinterpret_cast<int4*>(w_vec)[0];
                int stats_base = item * 128;
                if (qj == 0) {
                    *(reinterpret_cast<float*>(Ws + (stats_base + m0_local * 2)) + (0)) = row_max0;
                    *(reinterpret_cast<float*>(Ws + (stats_base + m0_local * 2 + 1)) + (0)) = row_sum0;
                    *(reinterpret_cast<float*>(Ws + (stats_base + m1_local * 2)) + (0)) = row_max1;
                    *(reinterpret_cast<float*>(Ws + (stats_base + m1_local * 2 + 1)) + (0)) = row_sum1;
                }
            }
        } else if (sc_split == 0) {
            float rcp = approx_rcp(row_sum0);
            float rcp_1 = approx_rcp(row_sum1);
            d_o[0] = d_o[0] * rcp;
            d_o[1] = d_o[1] * rcp;
            d_o[2] = d_o[2] * rcp_1;
            d_o[3] = d_o[3] * rcp_1;
            d_o[4] = d_o[4] * rcp;
            d_o[5] = d_o[5] * rcp;
            d_o[6] = d_o[6] * rcp_1;
            d_o[7] = d_o[7] * rcp_1;
            d_o[8] = d_o[8] * rcp;
            d_o[9] = d_o[9] * rcp;
            d_o[10] = d_o[10] * rcp_1;
            d_o[11] = d_o[11] * rcp_1;
            d_o[12] = d_o[12] * rcp;
            d_o[13] = d_o[13] * rcp;
            d_o[14] = d_o[14] * rcp_1;
            d_o[15] = d_o[15] * rcp_1;
            d_o[16] = d_o[16] * rcp;
            d_o[17] = d_o[17] * rcp;
            d_o[18] = d_o[18] * rcp_1;
            d_o[19] = d_o[19] * rcp_1;
            d_o[20] = d_o[20] * rcp;
            d_o[21] = d_o[21] * rcp;
            d_o[22] = d_o[22] * rcp_1;
            d_o[23] = d_o[23] * rcp_1;
            d_o[24] = d_o[24] * rcp;
            d_o[25] = d_o[25] * rcp;
            d_o[26] = d_o[26] * rcp_1;
            d_o[27] = d_o[27] * rcp_1;
            d_o[28] = d_o[28] * rcp;
            d_o[29] = d_o[29] * rcp;
            d_o[30] = d_o[30] * rcp_1;
            d_o[31] = d_o[31] * rcp_1;
            d_o[32] = d_o[32] * rcp;
            d_o[33] = d_o[33] * rcp;
            d_o[34] = d_o[34] * rcp_1;
            d_o[35] = d_o[35] * rcp_1;
            d_o[36] = d_o[36] * rcp;
            d_o[37] = d_o[37] * rcp;
            d_o[38] = d_o[38] * rcp_1;
            d_o[39] = d_o[39] * rcp_1;
            d_o[40] = d_o[40] * rcp;
            d_o[41] = d_o[41] * rcp;
            d_o[42] = d_o[42] * rcp_1;
            d_o[43] = d_o[43] * rcp_1;
            d_o[44] = d_o[44] * rcp;
            d_o[45] = d_o[45] * rcp;
            d_o[46] = d_o[46] * rcp_1;
            d_o[47] = d_o[47] * rcp_1;
            d_o[48] = d_o[48] * rcp;
            d_o[49] = d_o[49] * rcp;
            d_o[50] = d_o[50] * rcp_1;
            d_o[51] = d_o[51] * rcp_1;
            d_o[52] = d_o[52] * rcp;
            d_o[53] = d_o[53] * rcp;
            d_o[54] = d_o[54] * rcp_1;
            d_o[55] = d_o[55] * rcp_1;
            d_o[56] = d_o[56] * rcp;
            d_o[57] = d_o[57] * rcp;
            d_o[58] = d_o[58] * rcp_1;
            d_o[59] = d_o[59] * rcp_1;
            d_o[60] = d_o[60] * rcp;
            d_o[61] = d_o[61] * rcp;
            d_o[62] = d_o[62] * rcp_1;
            d_o[63] = d_o[63] * rcp_1;
            int qj1 = qj & 1;
            int qj2 = qj & 2;
            unsigned int o_vec[4];
            unsigned int o_tmp[4];
            int o_row_base = q_row * 128;
            int m_local_r_1_0 = m0_local;
            __nv_bfloat162 _bit_cast_src_212 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_212)[0];
            __nv_bfloat162 _bit_cast_src_213 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_213)[0];
            __nv_bfloat162 _bit_cast_src_214 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_214)[0];
            __nv_bfloat162 _bit_cast_src_215 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_215)[0];
            unsigned int _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_216;
            unsigned int _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_217;
            unsigned int _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_218;
            unsigned int _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_219;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_220;
            unsigned int _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_221;
            unsigned int _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_222;
            unsigned int _shfl_xor_sync_223 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_223;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_0 = o_row_base + m_local_r_1_0 * 128 + qj * 8;
            reinterpret_cast<int4*>(O + o_off_3_0)[0] = reinterpret_cast<int4*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_224 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_224)[0];
            __nv_bfloat162 _bit_cast_src_225 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_225)[0];
            __nv_bfloat162 _bit_cast_src_226 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_226)[0];
            __nv_bfloat162 _bit_cast_src_227 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_227)[0];
            unsigned int _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_228;
            unsigned int _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_229;
            unsigned int _shfl_xor_sync_230 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_230;
            unsigned int _shfl_xor_sync_231 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_231;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_232;
            unsigned int _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_233;
            unsigned int _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_234;
            unsigned int _shfl_xor_sync_235 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_235;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_1 = o_row_base + m_local_r_1_0 * 128 + (4 + qj) * 8;
            reinterpret_cast<int4*>(O + o_off_3_1)[0] = reinterpret_cast<int4*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_236 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_236)[0];
            __nv_bfloat162 _bit_cast_src_237 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_237)[0];
            __nv_bfloat162 _bit_cast_src_238 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_238)[0];
            __nv_bfloat162 _bit_cast_src_239 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_239)[0];
            unsigned int _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_240;
            unsigned int _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_241;
            unsigned int _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_242;
            unsigned int _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_243;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_244;
            unsigned int _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_245;
            unsigned int _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_246;
            unsigned int _shfl_xor_sync_247 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_247;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_2 = o_row_base + m_local_r_1_0 * 128 + (8 + qj) * 8;
            reinterpret_cast<int4*>(O + o_off_3_2)[0] = reinterpret_cast<int4*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_248 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_248)[0];
            __nv_bfloat162 _bit_cast_src_249 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_249)[0];
            __nv_bfloat162 _bit_cast_src_250 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_250)[0];
            __nv_bfloat162 _bit_cast_src_251 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_251)[0];
            unsigned int _shfl_xor_sync_252 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_252;
            unsigned int _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_253;
            unsigned int _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_254;
            unsigned int _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_255;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_256 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_256;
            unsigned int _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_257;
            unsigned int _shfl_xor_sync_258 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_258;
            unsigned int _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_259;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_3 = o_row_base + m_local_r_1_0 * 128 + (12 + qj) * 8;
            reinterpret_cast<int4*>(O + o_off_3_3)[0] = reinterpret_cast<int4*>(o_vec)[0];
            int m_local_r_1_1 = m1_local;
            __nv_bfloat162 _bit_cast_src_260 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_260)[0];
            __nv_bfloat162 _bit_cast_src_261 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_261)[0];
            __nv_bfloat162 _bit_cast_src_262 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_262)[0];
            __nv_bfloat162 _bit_cast_src_263 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_263)[0];
            unsigned int _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_264;
            unsigned int _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_265;
            unsigned int _shfl_xor_sync_266 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_266;
            unsigned int _shfl_xor_sync_267 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_267;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_268 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_268;
            unsigned int _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_269;
            unsigned int _shfl_xor_sync_270 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_270;
            unsigned int _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_271;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_4 = o_row_base + m_local_r_1_1 * 128 + qj * 8;
            reinterpret_cast<int4*>(O + o_off_3_4)[0] = reinterpret_cast<int4*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_272 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_272)[0];
            __nv_bfloat162 _bit_cast_src_273 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_273)[0];
            __nv_bfloat162 _bit_cast_src_274 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_274)[0];
            __nv_bfloat162 _bit_cast_src_275 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_275)[0];
            unsigned int _shfl_xor_sync_276 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_276;
            unsigned int _shfl_xor_sync_277 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_277;
            unsigned int _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_278;
            unsigned int _shfl_xor_sync_279 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_279;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_280;
            unsigned int _shfl_xor_sync_281 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_281;
            unsigned int _shfl_xor_sync_282 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_282;
            unsigned int _shfl_xor_sync_283 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_283;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_5 = o_row_base + m_local_r_1_1 * 128 + (4 + qj) * 8;
            reinterpret_cast<int4*>(O + o_off_3_5)[0] = reinterpret_cast<int4*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_284 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_284)[0];
            __nv_bfloat162 _bit_cast_src_285 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_285)[0];
            __nv_bfloat162 _bit_cast_src_286 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_286)[0];
            __nv_bfloat162 _bit_cast_src_287 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_287)[0];
            unsigned int _shfl_xor_sync_288 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_288;
            unsigned int _shfl_xor_sync_289 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_289;
            unsigned int _shfl_xor_sync_290 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_290;
            unsigned int _shfl_xor_sync_291 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_291;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_292;
            unsigned int _shfl_xor_sync_293 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_293;
            unsigned int _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_294;
            unsigned int _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_295;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_6 = o_row_base + m_local_r_1_1 * 128 + (8 + qj) * 8;
            reinterpret_cast<int4*>(O + o_off_3_6)[0] = reinterpret_cast<int4*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_296 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
            o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_296)[0];
            __nv_bfloat162 _bit_cast_src_297 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
            o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_297)[0];
            __nv_bfloat162 _bit_cast_src_298 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
            o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_298)[0];
            __nv_bfloat162 _bit_cast_src_299 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
            o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_299)[0];
            unsigned int _shfl_xor_sync_300 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
            o_tmp[0] = _shfl_xor_sync_300;
            unsigned int _shfl_xor_sync_301 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
            o_tmp[1] = _shfl_xor_sync_301;
            unsigned int _shfl_xor_sync_302 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
            o_tmp[2] = _shfl_xor_sync_302;
            unsigned int _shfl_xor_sync_303 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
            o_tmp[3] = _shfl_xor_sync_303;
            o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
            unsigned int _shfl_xor_sync_304 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
            o_tmp[0] = _shfl_xor_sync_304;
            unsigned int _shfl_xor_sync_305 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
            o_tmp[1] = _shfl_xor_sync_305;
            unsigned int _shfl_xor_sync_306 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
            o_tmp[2] = _shfl_xor_sync_306;
            unsigned int _shfl_xor_sync_307 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
            o_tmp[3] = _shfl_xor_sync_307;
            o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
            o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
            o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
            o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
            int o_off_3_7 = o_row_base + m_local_r_1_1 * 128 + (12 + qj) * 8;
            reinterpret_cast<int4*>(O + o_off_3_7)[0] = reinterpret_cast<int4*>(o_vec)[0];
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (owner_wg != 0) {
        if (sc_nsplit > 1) {
            int n_owners = ((sc_nsplit < 4) ? sc_nsplit : 4);
            if (sc_split < n_owners) {
                int n_active = ((sc_nsplit < 4) ? sc_nsplit : 4);
                int stat_chunks = sc_nsplit * 32;
                int ws_base = sc_first * 128;
                float stat_buf[8];
                int sidx = tid;
                if (sidx < stat_chunks) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(Ws + (ws_base + sidx * 4) + 0);
                        stat_buf[0 + 0] = _v4.x;
                        stat_buf[0 + 1] = _v4.y;
                        stat_buf[0 + 2] = _v4.z;
                        stat_buf[0 + 3] = _v4.w;
                    }
                }
                int sidx_0 = 128 + tid;
                if (sidx_0 < stat_chunks) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(Ws + (ws_base + sidx_0 * 4) + 0);
                        stat_buf[4 + 0] = _v4.x;
                        stat_buf[4 + 1] = _v4.y;
                        stat_buf[4 + 2] = _v4.z;
                        stat_buf[4 + 3] = _v4.w;
                    }
                }
                int sidx_1 = tid;
                if (sidx_1 < stat_chunks) {
                    stats_smem[sidx_1 * 4] = stat_buf[0];
                    stats_smem[sidx_1 * 4 + 1] = stat_buf[1];
                    stats_smem[sidx_1 * 4 + 2] = stat_buf[2];
                    stats_smem[sidx_1 * 4 + 3] = stat_buf[3];
                }
                int sidx_2 = 128 + tid;
                if (sidx_2 < stat_chunks) {
                    stats_smem[sidx_2 * 4] = stat_buf[4];
                    stats_smem[sidx_2 * 4 + 1] = stat_buf[5];
                    stats_smem[sidx_2 * 4 + 2] = stat_buf[6];
                    stats_smem[sidx_2 * 4 + 3] = stat_buf[7];
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                float merged_max0 = row_max0;
                float merged_max1 = row_max1;
                #pragma unroll 1
                for (int j = 0; j < sc_nsplit; j++) {
                    int sb = j * 128;
                    merged_max0 = fmaxf(merged_max0, stats_smem[sb + m0_local * 2]);
                    merged_max1 = fmaxf(merged_max1, stats_smem[sb + m1_local * 2]);
                }
                float fsum0 = 0.0f;
                float fsum1 = 0.0f;
                #pragma unroll 1
                for (int j_1 = 0; j_1 < sc_nsplit; j_1++) {
                    int sb_1 = j_1 * 128;
                    float w0 = approx_exp2(stats_smem[sb_1 + m0_local * 2] - merged_max0);
                    float w1 = approx_exp2(stats_smem[sb_1 + m1_local * 2] - merged_max1);
                    fsum0 = __fmaf_rn(stats_smem[sb_1 + m0_local * 2 + 1], w0, fsum0);
                    fsum1 = __fmaf_rn(stats_smem[sb_1 + m1_local * 2 + 1], w1, fsum1);
                }
                float acc[16];
                int my_tid4 = tid * 4;
                int sset = sc_split;
                if (sset < 4) {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    #pragma unroll 1
                    for (int j_2 = 0; j_2 < sc_nsplit; j_2++) {
                        int sb_2 = j_2 * 128;
                        float w0_1 = approx_exp2(stats_smem[sb_2 + m0_local * 2] - merged_max0);
                        float w1_1 = approx_exp2(stats_smem[sb_2 + m1_local * 2] - merged_max1);
                        int base = (sc_first + j_2) * 8192 + sset * 2048 + my_tid4;
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + base + 0);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        acc[0] = __fmaf_rn(_vec_load_0[0], w0_1, acc[0]);
                        acc[1] = __fmaf_rn(_vec_load_0[1], w0_1, acc[1]);
                        acc[2] = __fmaf_rn(_vec_load_0[2], w1_1, acc[2]);
                        acc[3] = __fmaf_rn(_vec_load_0[3], w1_1, acc[3]);
                        float vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base + 512) + 0);
                            vec_load_0[0 + 0] = _v4.x;
                            vec_load_0[0 + 1] = _v4.y;
                            vec_load_0[0 + 2] = _v4.z;
                            vec_load_0[0 + 3] = _v4.w;
                        }
                        acc[4] = __fmaf_rn(vec_load_0[0], w0_1, acc[4]);
                        acc[5] = __fmaf_rn(vec_load_0[1], w0_1, acc[5]);
                        acc[6] = __fmaf_rn(vec_load_0[2], w1_1, acc[6]);
                        acc[7] = __fmaf_rn(vec_load_0[3], w1_1, acc[7]);
                        float vec_load_1[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base + 1024) + 0);
                            vec_load_1[0 + 0] = _v4.x;
                            vec_load_1[0 + 1] = _v4.y;
                            vec_load_1[0 + 2] = _v4.z;
                            vec_load_1[0 + 3] = _v4.w;
                        }
                        acc[8] = __fmaf_rn(vec_load_1[0], w0_1, acc[8]);
                        acc[9] = __fmaf_rn(vec_load_1[1], w0_1, acc[9]);
                        acc[10] = __fmaf_rn(vec_load_1[2], w1_1, acc[10]);
                        acc[11] = __fmaf_rn(vec_load_1[3], w1_1, acc[11]);
                        float vec_load_2[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base + 1536) + 0);
                            vec_load_2[0 + 0] = _v4.x;
                            vec_load_2[0 + 1] = _v4.y;
                            vec_load_2[0 + 2] = _v4.z;
                            vec_load_2[0 + 3] = _v4.w;
                        }
                        acc[12] = __fmaf_rn(vec_load_2[0], w0_1, acc[12]);
                        acc[13] = __fmaf_rn(vec_load_2[1], w0_1, acc[13]);
                        acc[14] = __fmaf_rn(vec_load_2[2], w1_1, acc[14]);
                        acc[15] = __fmaf_rn(vec_load_2[3], w1_1, acc[15]);
                    }
                    float rcp_2 = approx_rcp(fsum0);
                    float rcp_1_1 = approx_rcp(fsum1);
                    acc[0] = acc[0] * rcp_2;
                    acc[1] = acc[1] * rcp_2;
                    acc[2] = acc[2] * rcp_1_1;
                    acc[3] = acc[3] * rcp_1_1;
                    acc[4] = acc[4] * rcp_2;
                    acc[5] = acc[5] * rcp_2;
                    acc[6] = acc[6] * rcp_1_1;
                    acc[7] = acc[7] * rcp_1_1;
                    acc[8] = acc[8] * rcp_2;
                    acc[9] = acc[9] * rcp_2;
                    acc[10] = acc[10] * rcp_1_1;
                    acc[11] = acc[11] * rcp_1_1;
                    acc[12] = acc[12] * rcp_2;
                    acc[13] = acc[13] * rcp_2;
                    acc[14] = acc[14] * rcp_1_1;
                    acc[15] = acc[15] * rcp_1_1;
                    int qj1_1 = qj & 1;
                    int qj2_1 = qj & 2;
                    unsigned int o_vec_1[4];
                    unsigned int o_tmp_1[4];
                    int o_row_base_1 = q_row * 128;
                    int m_local_r_1_2 = m0_local;
                    __nv_bfloat162 _bit_cast_src_314 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                    o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_314)[0];
                    __nv_bfloat162 _bit_cast_src_315 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                    o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_315)[0];
                    __nv_bfloat162 _bit_cast_src_316 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                    o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_316)[0];
                    __nv_bfloat162 _bit_cast_src_317 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                    o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_317)[0];
                    unsigned int _shfl_xor_sync_318 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                    o_tmp_1[0] = _shfl_xor_sync_318;
                    unsigned int _shfl_xor_sync_319 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                    o_tmp_1[1] = _shfl_xor_sync_319;
                    unsigned int _shfl_xor_sync_320 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                    o_tmp_1[2] = _shfl_xor_sync_320;
                    unsigned int _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                    o_tmp_1[3] = _shfl_xor_sync_321;
                    o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                    o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                    o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                    o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                    unsigned int _shfl_xor_sync_322 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                    o_tmp_1[0] = _shfl_xor_sync_322;
                    unsigned int _shfl_xor_sync_323 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                    o_tmp_1[1] = _shfl_xor_sync_323;
                    unsigned int _shfl_xor_sync_324 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                    o_tmp_1[2] = _shfl_xor_sync_324;
                    unsigned int _shfl_xor_sync_325 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                    o_tmp_1[3] = _shfl_xor_sync_325;
                    o_vec_1[0] = ((qj2_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                    o_vec_1[1] = ((qj2_1 != 0) ? o_tmp_1[1] : o_vec_1[1]);
                    o_vec_1[2] = ((qj2_1 == 0) ? o_tmp_1[2] : o_vec_1[2]);
                    o_vec_1[3] = ((qj2_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                    int o_off_1_0 = o_row_base_1 + m_local_r_1_2 * 128 + (4 * sset + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_0)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
                    int m_local_r_1_3 = m1_local;
                    __nv_bfloat162 _bit_cast_src_326 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                    o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_326)[0];
                    __nv_bfloat162 _bit_cast_src_327 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                    o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_327)[0];
                    __nv_bfloat162 _bit_cast_src_328 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                    o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_328)[0];
                    __nv_bfloat162 _bit_cast_src_329 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                    o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_329)[0];
                    unsigned int _shfl_xor_sync_330 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                    o_tmp_1[0] = _shfl_xor_sync_330;
                    unsigned int _shfl_xor_sync_331 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                    o_tmp_1[1] = _shfl_xor_sync_331;
                    unsigned int _shfl_xor_sync_332 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                    o_tmp_1[2] = _shfl_xor_sync_332;
                    unsigned int _shfl_xor_sync_333 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                    o_tmp_1[3] = _shfl_xor_sync_333;
                    o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                    o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                    o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                    o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                    unsigned int _shfl_xor_sync_334 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                    o_tmp_1[0] = _shfl_xor_sync_334;
                    unsigned int _shfl_xor_sync_335 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                    o_tmp_1[1] = _shfl_xor_sync_335;
                    unsigned int _shfl_xor_sync_336 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                    o_tmp_1[2] = _shfl_xor_sync_336;
                    unsigned int _shfl_xor_sync_337 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                    o_tmp_1[3] = _shfl_xor_sync_337;
                    o_vec_1[0] = ((qj2_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                    o_vec_1[1] = ((qj2_1 != 0) ? o_tmp_1[1] : o_vec_1[1]);
                    o_vec_1[2] = ((qj2_1 == 0) ? o_tmp_1[2] : o_vec_1[2]);
                    o_vec_1[3] = ((qj2_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                    int o_off_1_1 = o_row_base_1 + m_local_r_1_3 * 128 + (4 * sset + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_1)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
                }
                int sset_3 = sc_split + n_active;
                if (sset_3 < 4) {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    #pragma unroll 1
                    for (int j_3 = 0; j_3 < sc_nsplit; j_3++) {
                        int sb_3 = j_3 * 128;
                        float w0_2 = approx_exp2(stats_smem[sb_3 + m0_local * 2] - merged_max0);
                        float w1_2 = approx_exp2(stats_smem[sb_3 + m1_local * 2] - merged_max1);
                        int base_1 = (sc_first + j_3) * 8192 + sset_3 * 2048 + my_tid4;
                        float _vec_load_4[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + base_1 + 0);
                            _vec_load_4[0 + 0] = _v4.x;
                            _vec_load_4[0 + 1] = _v4.y;
                            _vec_load_4[0 + 2] = _v4.z;
                            _vec_load_4[0 + 3] = _v4.w;
                        }
                        acc[0] = __fmaf_rn(_vec_load_4[0], w0_2, acc[0]);
                        acc[1] = __fmaf_rn(_vec_load_4[1], w0_2, acc[1]);
                        acc[2] = __fmaf_rn(_vec_load_4[2], w1_2, acc[2]);
                        acc[3] = __fmaf_rn(_vec_load_4[3], w1_2, acc[3]);
                        float _vec_load_5[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_1 + 512) + 0);
                            _vec_load_5[0 + 0] = _v4.x;
                            _vec_load_5[0 + 1] = _v4.y;
                            _vec_load_5[0 + 2] = _v4.z;
                            _vec_load_5[0 + 3] = _v4.w;
                        }
                        acc[4] = __fmaf_rn(_vec_load_5[0], w0_2, acc[4]);
                        acc[5] = __fmaf_rn(_vec_load_5[1], w0_2, acc[5]);
                        acc[6] = __fmaf_rn(_vec_load_5[2], w1_2, acc[6]);
                        acc[7] = __fmaf_rn(_vec_load_5[3], w1_2, acc[7]);
                        float _vec_load_6[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_1 + 1024) + 0);
                            _vec_load_6[0 + 0] = _v4.x;
                            _vec_load_6[0 + 1] = _v4.y;
                            _vec_load_6[0 + 2] = _v4.z;
                            _vec_load_6[0 + 3] = _v4.w;
                        }
                        acc[8] = __fmaf_rn(_vec_load_6[0], w0_2, acc[8]);
                        acc[9] = __fmaf_rn(_vec_load_6[1], w0_2, acc[9]);
                        acc[10] = __fmaf_rn(_vec_load_6[2], w1_2, acc[10]);
                        acc[11] = __fmaf_rn(_vec_load_6[3], w1_2, acc[11]);
                        float _vec_load_7[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_1 + 1536) + 0);
                            _vec_load_7[0 + 0] = _v4.x;
                            _vec_load_7[0 + 1] = _v4.y;
                            _vec_load_7[0 + 2] = _v4.z;
                            _vec_load_7[0 + 3] = _v4.w;
                        }
                        acc[12] = __fmaf_rn(_vec_load_7[0], w0_2, acc[12]);
                        acc[13] = __fmaf_rn(_vec_load_7[1], w0_2, acc[13]);
                        acc[14] = __fmaf_rn(_vec_load_7[2], w1_2, acc[14]);
                        acc[15] = __fmaf_rn(_vec_load_7[3], w1_2, acc[15]);
                    }
                    float rcp_3 = approx_rcp(fsum0);
                    float rcp_1_2 = approx_rcp(fsum1);
                    acc[0] = acc[0] * rcp_3;
                    acc[1] = acc[1] * rcp_3;
                    acc[2] = acc[2] * rcp_1_2;
                    acc[3] = acc[3] * rcp_1_2;
                    acc[4] = acc[4] * rcp_3;
                    acc[5] = acc[5] * rcp_3;
                    acc[6] = acc[6] * rcp_1_2;
                    acc[7] = acc[7] * rcp_1_2;
                    acc[8] = acc[8] * rcp_3;
                    acc[9] = acc[9] * rcp_3;
                    acc[10] = acc[10] * rcp_1_2;
                    acc[11] = acc[11] * rcp_1_2;
                    acc[12] = acc[12] * rcp_3;
                    acc[13] = acc[13] * rcp_3;
                    acc[14] = acc[14] * rcp_1_2;
                    acc[15] = acc[15] * rcp_1_2;
                    int qj1_2 = qj & 1;
                    int qj2_2 = qj & 2;
                    unsigned int o_vec_2[4];
                    unsigned int o_tmp_2[4];
                    int o_row_base_2 = q_row * 128;
                    int m_local_r_1_4 = m0_local;
                    __nv_bfloat162 _bit_cast_src_342 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                    o_vec_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_342)[0];
                    __nv_bfloat162 _bit_cast_src_343 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                    o_vec_2[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_343)[0];
                    __nv_bfloat162 _bit_cast_src_344 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                    o_vec_2[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_344)[0];
                    __nv_bfloat162 _bit_cast_src_345 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                    o_vec_2[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_345)[0];
                    unsigned int _shfl_xor_sync_346 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 1);
                    o_tmp_2[0] = _shfl_xor_sync_346;
                    unsigned int _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 1);
                    o_tmp_2[1] = _shfl_xor_sync_347;
                    unsigned int _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 1);
                    o_tmp_2[2] = _shfl_xor_sync_348;
                    unsigned int _shfl_xor_sync_349 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 1);
                    o_tmp_2[3] = _shfl_xor_sync_349;
                    o_vec_2[0] = ((qj1_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                    o_vec_2[1] = ((qj1_2 == 0) ? o_tmp_2[1] : o_vec_2[1]);
                    o_vec_2[2] = ((qj1_2 != 0) ? o_tmp_2[2] : o_vec_2[2]);
                    o_vec_2[3] = ((qj1_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                    unsigned int _shfl_xor_sync_350 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 2);
                    o_tmp_2[0] = _shfl_xor_sync_350;
                    unsigned int _shfl_xor_sync_351 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 2);
                    o_tmp_2[1] = _shfl_xor_sync_351;
                    unsigned int _shfl_xor_sync_352 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 2);
                    o_tmp_2[2] = _shfl_xor_sync_352;
                    unsigned int _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 2);
                    o_tmp_2[3] = _shfl_xor_sync_353;
                    o_vec_2[0] = ((qj2_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                    o_vec_2[1] = ((qj2_2 != 0) ? o_tmp_2[1] : o_vec_2[1]);
                    o_vec_2[2] = ((qj2_2 == 0) ? o_tmp_2[2] : o_vec_2[2]);
                    o_vec_2[3] = ((qj2_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                    int o_off_1_2 = o_row_base_2 + m_local_r_1_4 * 128 + (4 * sset_3 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_2)[0] = reinterpret_cast<int4*>(o_vec_2)[0];
                    int m_local_r_1_5 = m1_local;
                    __nv_bfloat162 _bit_cast_src_354 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                    o_vec_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_354)[0];
                    __nv_bfloat162 _bit_cast_src_355 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                    o_vec_2[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_355)[0];
                    __nv_bfloat162 _bit_cast_src_356 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                    o_vec_2[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_356)[0];
                    __nv_bfloat162 _bit_cast_src_357 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                    o_vec_2[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_357)[0];
                    unsigned int _shfl_xor_sync_358 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 1);
                    o_tmp_2[0] = _shfl_xor_sync_358;
                    unsigned int _shfl_xor_sync_359 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 1);
                    o_tmp_2[1] = _shfl_xor_sync_359;
                    unsigned int _shfl_xor_sync_360 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 1);
                    o_tmp_2[2] = _shfl_xor_sync_360;
                    unsigned int _shfl_xor_sync_361 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 1);
                    o_tmp_2[3] = _shfl_xor_sync_361;
                    o_vec_2[0] = ((qj1_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                    o_vec_2[1] = ((qj1_2 == 0) ? o_tmp_2[1] : o_vec_2[1]);
                    o_vec_2[2] = ((qj1_2 != 0) ? o_tmp_2[2] : o_vec_2[2]);
                    o_vec_2[3] = ((qj1_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                    unsigned int _shfl_xor_sync_362 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 2);
                    o_tmp_2[0] = _shfl_xor_sync_362;
                    unsigned int _shfl_xor_sync_363 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 2);
                    o_tmp_2[1] = _shfl_xor_sync_363;
                    unsigned int _shfl_xor_sync_364 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 2);
                    o_tmp_2[2] = _shfl_xor_sync_364;
                    unsigned int _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 2);
                    o_tmp_2[3] = _shfl_xor_sync_365;
                    o_vec_2[0] = ((qj2_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                    o_vec_2[1] = ((qj2_2 != 0) ? o_tmp_2[1] : o_vec_2[1]);
                    o_vec_2[2] = ((qj2_2 == 0) ? o_tmp_2[2] : o_vec_2[2]);
                    o_vec_2[3] = ((qj2_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                    int o_off_1_3 = o_row_base_2 + m_local_r_1_5 * 128 + (4 * sset_3 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_3)[0] = reinterpret_cast<int4*>(o_vec_2)[0];
                }
                int sset_4 = sc_split + 2 * n_active;
                if (sset_4 < 4) {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    #pragma unroll 1
                    for (int j_4 = 0; j_4 < sc_nsplit; j_4++) {
                        int sb_4 = j_4 * 128;
                        float w0_3 = approx_exp2(stats_smem[sb_4 + m0_local * 2] - merged_max0);
                        float w1_3 = approx_exp2(stats_smem[sb_4 + m1_local * 2] - merged_max1);
                        int base_2 = (sc_first + j_4) * 8192 + sset_4 * 2048 + my_tid4;
                        float _vec_load_8[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + base_2 + 0);
                            _vec_load_8[0 + 0] = _v4.x;
                            _vec_load_8[0 + 1] = _v4.y;
                            _vec_load_8[0 + 2] = _v4.z;
                            _vec_load_8[0 + 3] = _v4.w;
                        }
                        acc[0] = __fmaf_rn(_vec_load_8[0], w0_3, acc[0]);
                        acc[1] = __fmaf_rn(_vec_load_8[1], w0_3, acc[1]);
                        acc[2] = __fmaf_rn(_vec_load_8[2], w1_3, acc[2]);
                        acc[3] = __fmaf_rn(_vec_load_8[3], w1_3, acc[3]);
                        float _vec_load_9[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_2 + 512) + 0);
                            _vec_load_9[0 + 0] = _v4.x;
                            _vec_load_9[0 + 1] = _v4.y;
                            _vec_load_9[0 + 2] = _v4.z;
                            _vec_load_9[0 + 3] = _v4.w;
                        }
                        acc[4] = __fmaf_rn(_vec_load_9[0], w0_3, acc[4]);
                        acc[5] = __fmaf_rn(_vec_load_9[1], w0_3, acc[5]);
                        acc[6] = __fmaf_rn(_vec_load_9[2], w1_3, acc[6]);
                        acc[7] = __fmaf_rn(_vec_load_9[3], w1_3, acc[7]);
                        float _vec_load_10[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_2 + 1024) + 0);
                            _vec_load_10[0 + 0] = _v4.x;
                            _vec_load_10[0 + 1] = _v4.y;
                            _vec_load_10[0 + 2] = _v4.z;
                            _vec_load_10[0 + 3] = _v4.w;
                        }
                        acc[8] = __fmaf_rn(_vec_load_10[0], w0_3, acc[8]);
                        acc[9] = __fmaf_rn(_vec_load_10[1], w0_3, acc[9]);
                        acc[10] = __fmaf_rn(_vec_load_10[2], w1_3, acc[10]);
                        acc[11] = __fmaf_rn(_vec_load_10[3], w1_3, acc[11]);
                        float _vec_load_11[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_2 + 1536) + 0);
                            _vec_load_11[0 + 0] = _v4.x;
                            _vec_load_11[0 + 1] = _v4.y;
                            _vec_load_11[0 + 2] = _v4.z;
                            _vec_load_11[0 + 3] = _v4.w;
                        }
                        acc[12] = __fmaf_rn(_vec_load_11[0], w0_3, acc[12]);
                        acc[13] = __fmaf_rn(_vec_load_11[1], w0_3, acc[13]);
                        acc[14] = __fmaf_rn(_vec_load_11[2], w1_3, acc[14]);
                        acc[15] = __fmaf_rn(_vec_load_11[3], w1_3, acc[15]);
                    }
                    float rcp_4 = approx_rcp(fsum0);
                    float rcp_1_3 = approx_rcp(fsum1);
                    acc[0] = acc[0] * rcp_4;
                    acc[1] = acc[1] * rcp_4;
                    acc[2] = acc[2] * rcp_1_3;
                    acc[3] = acc[3] * rcp_1_3;
                    acc[4] = acc[4] * rcp_4;
                    acc[5] = acc[5] * rcp_4;
                    acc[6] = acc[6] * rcp_1_3;
                    acc[7] = acc[7] * rcp_1_3;
                    acc[8] = acc[8] * rcp_4;
                    acc[9] = acc[9] * rcp_4;
                    acc[10] = acc[10] * rcp_1_3;
                    acc[11] = acc[11] * rcp_1_3;
                    acc[12] = acc[12] * rcp_4;
                    acc[13] = acc[13] * rcp_4;
                    acc[14] = acc[14] * rcp_1_3;
                    acc[15] = acc[15] * rcp_1_3;
                    int qj1_3 = qj & 1;
                    int qj2_3 = qj & 2;
                    unsigned int o_vec_3[4];
                    unsigned int o_tmp_3[4];
                    int o_row_base_3 = q_row * 128;
                    int m_local_r_1_6 = m0_local;
                    __nv_bfloat162 _bit_cast_src_370 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                    o_vec_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_370)[0];
                    __nv_bfloat162 _bit_cast_src_371 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                    o_vec_3[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_371)[0];
                    __nv_bfloat162 _bit_cast_src_372 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                    o_vec_3[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_372)[0];
                    __nv_bfloat162 _bit_cast_src_373 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                    o_vec_3[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_373)[0];
                    unsigned int _shfl_xor_sync_374 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 1);
                    o_tmp_3[0] = _shfl_xor_sync_374;
                    unsigned int _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 1);
                    o_tmp_3[1] = _shfl_xor_sync_375;
                    unsigned int _shfl_xor_sync_376 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 1);
                    o_tmp_3[2] = _shfl_xor_sync_376;
                    unsigned int _shfl_xor_sync_377 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 1);
                    o_tmp_3[3] = _shfl_xor_sync_377;
                    o_vec_3[0] = ((qj1_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                    o_vec_3[1] = ((qj1_3 == 0) ? o_tmp_3[1] : o_vec_3[1]);
                    o_vec_3[2] = ((qj1_3 != 0) ? o_tmp_3[2] : o_vec_3[2]);
                    o_vec_3[3] = ((qj1_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                    unsigned int _shfl_xor_sync_378 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 2);
                    o_tmp_3[0] = _shfl_xor_sync_378;
                    unsigned int _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 2);
                    o_tmp_3[1] = _shfl_xor_sync_379;
                    unsigned int _shfl_xor_sync_380 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 2);
                    o_tmp_3[2] = _shfl_xor_sync_380;
                    unsigned int _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 2);
                    o_tmp_3[3] = _shfl_xor_sync_381;
                    o_vec_3[0] = ((qj2_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                    o_vec_3[1] = ((qj2_3 != 0) ? o_tmp_3[1] : o_vec_3[1]);
                    o_vec_3[2] = ((qj2_3 == 0) ? o_tmp_3[2] : o_vec_3[2]);
                    o_vec_3[3] = ((qj2_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                    int o_off_1_4 = o_row_base_3 + m_local_r_1_6 * 128 + (4 * sset_4 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_4)[0] = reinterpret_cast<int4*>(o_vec_3)[0];
                    int m_local_r_1_7 = m1_local;
                    __nv_bfloat162 _bit_cast_src_382 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                    o_vec_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_382)[0];
                    __nv_bfloat162 _bit_cast_src_383 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                    o_vec_3[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_383)[0];
                    __nv_bfloat162 _bit_cast_src_384 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                    o_vec_3[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_384)[0];
                    __nv_bfloat162 _bit_cast_src_385 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                    o_vec_3[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_385)[0];
                    unsigned int _shfl_xor_sync_386 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 1);
                    o_tmp_3[0] = _shfl_xor_sync_386;
                    unsigned int _shfl_xor_sync_387 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 1);
                    o_tmp_3[1] = _shfl_xor_sync_387;
                    unsigned int _shfl_xor_sync_388 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 1);
                    o_tmp_3[2] = _shfl_xor_sync_388;
                    unsigned int _shfl_xor_sync_389 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 1);
                    o_tmp_3[3] = _shfl_xor_sync_389;
                    o_vec_3[0] = ((qj1_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                    o_vec_3[1] = ((qj1_3 == 0) ? o_tmp_3[1] : o_vec_3[1]);
                    o_vec_3[2] = ((qj1_3 != 0) ? o_tmp_3[2] : o_vec_3[2]);
                    o_vec_3[3] = ((qj1_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                    unsigned int _shfl_xor_sync_390 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 2);
                    o_tmp_3[0] = _shfl_xor_sync_390;
                    unsigned int _shfl_xor_sync_391 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 2);
                    o_tmp_3[1] = _shfl_xor_sync_391;
                    unsigned int _shfl_xor_sync_392 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 2);
                    o_tmp_3[2] = _shfl_xor_sync_392;
                    unsigned int _shfl_xor_sync_393 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 2);
                    o_tmp_3[3] = _shfl_xor_sync_393;
                    o_vec_3[0] = ((qj2_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                    o_vec_3[1] = ((qj2_3 != 0) ? o_tmp_3[1] : o_vec_3[1]);
                    o_vec_3[2] = ((qj2_3 == 0) ? o_tmp_3[2] : o_vec_3[2]);
                    o_vec_3[3] = ((qj2_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                    int o_off_1_5 = o_row_base_3 + m_local_r_1_7 * 128 + (4 * sset_4 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_5)[0] = reinterpret_cast<int4*>(o_vec_3)[0];
                }
                int sset_5 = sc_split + 3 * n_active;
                if (sset_5 < 4) {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    #pragma unroll 1
                    for (int j_5 = 0; j_5 < sc_nsplit; j_5++) {
                        int sb_5 = j_5 * 128;
                        float w0_4 = approx_exp2(stats_smem[sb_5 + m0_local * 2] - merged_max0);
                        float w1_4 = approx_exp2(stats_smem[sb_5 + m1_local * 2] - merged_max1);
                        int base_3 = (sc_first + j_5) * 8192 + sset_5 * 2048 + my_tid4;
                        float _vec_load_12[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + base_3 + 0);
                            _vec_load_12[0 + 0] = _v4.x;
                            _vec_load_12[0 + 1] = _v4.y;
                            _vec_load_12[0 + 2] = _v4.z;
                            _vec_load_12[0 + 3] = _v4.w;
                        }
                        acc[0] = __fmaf_rn(_vec_load_12[0], w0_4, acc[0]);
                        acc[1] = __fmaf_rn(_vec_load_12[1], w0_4, acc[1]);
                        acc[2] = __fmaf_rn(_vec_load_12[2], w1_4, acc[2]);
                        acc[3] = __fmaf_rn(_vec_load_12[3], w1_4, acc[3]);
                        float _vec_load_13[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_3 + 512) + 0);
                            _vec_load_13[0 + 0] = _v4.x;
                            _vec_load_13[0 + 1] = _v4.y;
                            _vec_load_13[0 + 2] = _v4.z;
                            _vec_load_13[0 + 3] = _v4.w;
                        }
                        acc[4] = __fmaf_rn(_vec_load_13[0], w0_4, acc[4]);
                        acc[5] = __fmaf_rn(_vec_load_13[1], w0_4, acc[5]);
                        acc[6] = __fmaf_rn(_vec_load_13[2], w1_4, acc[6]);
                        acc[7] = __fmaf_rn(_vec_load_13[3], w1_4, acc[7]);
                        float _vec_load_14[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_3 + 1024) + 0);
                            _vec_load_14[0 + 0] = _v4.x;
                            _vec_load_14[0 + 1] = _v4.y;
                            _vec_load_14[0 + 2] = _v4.z;
                            _vec_load_14[0 + 3] = _v4.w;
                        }
                        acc[8] = __fmaf_rn(_vec_load_14[0], w0_4, acc[8]);
                        acc[9] = __fmaf_rn(_vec_load_14[1], w0_4, acc[9]);
                        acc[10] = __fmaf_rn(_vec_load_14[2], w1_4, acc[10]);
                        acc[11] = __fmaf_rn(_vec_load_14[3], w1_4, acc[11]);
                        float _vec_load_15[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(Wo + (base_3 + 1536) + 0);
                            _vec_load_15[0 + 0] = _v4.x;
                            _vec_load_15[0 + 1] = _v4.y;
                            _vec_load_15[0 + 2] = _v4.z;
                            _vec_load_15[0 + 3] = _v4.w;
                        }
                        acc[12] = __fmaf_rn(_vec_load_15[0], w0_4, acc[12]);
                        acc[13] = __fmaf_rn(_vec_load_15[1], w0_4, acc[13]);
                        acc[14] = __fmaf_rn(_vec_load_15[2], w1_4, acc[14]);
                        acc[15] = __fmaf_rn(_vec_load_15[3], w1_4, acc[15]);
                    }
                    float rcp_5 = approx_rcp(fsum0);
                    float rcp_1_4 = approx_rcp(fsum1);
                    acc[0] = acc[0] * rcp_5;
                    acc[1] = acc[1] * rcp_5;
                    acc[2] = acc[2] * rcp_1_4;
                    acc[3] = acc[3] * rcp_1_4;
                    acc[4] = acc[4] * rcp_5;
                    acc[5] = acc[5] * rcp_5;
                    acc[6] = acc[6] * rcp_1_4;
                    acc[7] = acc[7] * rcp_1_4;
                    acc[8] = acc[8] * rcp_5;
                    acc[9] = acc[9] * rcp_5;
                    acc[10] = acc[10] * rcp_1_4;
                    acc[11] = acc[11] * rcp_1_4;
                    acc[12] = acc[12] * rcp_5;
                    acc[13] = acc[13] * rcp_5;
                    acc[14] = acc[14] * rcp_1_4;
                    acc[15] = acc[15] * rcp_1_4;
                    int qj1_4 = qj & 1;
                    int qj2_4 = qj & 2;
                    unsigned int o_vec_4[4];
                    unsigned int o_tmp_4[4];
                    int o_row_base_4 = q_row * 128;
                    int m_local_r_1_8 = m0_local;
                    __nv_bfloat162 _bit_cast_src_398 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                    o_vec_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_398)[0];
                    __nv_bfloat162 _bit_cast_src_399 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                    o_vec_4[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_399)[0];
                    __nv_bfloat162 _bit_cast_src_400 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                    o_vec_4[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_400)[0];
                    __nv_bfloat162 _bit_cast_src_401 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                    o_vec_4[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_401)[0];
                    unsigned int _shfl_xor_sync_402 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[1], 1);
                    o_tmp_4[0] = _shfl_xor_sync_402;
                    unsigned int _shfl_xor_sync_403 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[0], 1);
                    o_tmp_4[1] = _shfl_xor_sync_403;
                    unsigned int _shfl_xor_sync_404 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[3], 1);
                    o_tmp_4[2] = _shfl_xor_sync_404;
                    unsigned int _shfl_xor_sync_405 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[2], 1);
                    o_tmp_4[3] = _shfl_xor_sync_405;
                    o_vec_4[0] = ((qj1_4 != 0) ? o_tmp_4[0] : o_vec_4[0]);
                    o_vec_4[1] = ((qj1_4 == 0) ? o_tmp_4[1] : o_vec_4[1]);
                    o_vec_4[2] = ((qj1_4 != 0) ? o_tmp_4[2] : o_vec_4[2]);
                    o_vec_4[3] = ((qj1_4 == 0) ? o_tmp_4[3] : o_vec_4[3]);
                    unsigned int _shfl_xor_sync_406 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[2], 2);
                    o_tmp_4[0] = _shfl_xor_sync_406;
                    unsigned int _shfl_xor_sync_407 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[3], 2);
                    o_tmp_4[1] = _shfl_xor_sync_407;
                    unsigned int _shfl_xor_sync_408 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[0], 2);
                    o_tmp_4[2] = _shfl_xor_sync_408;
                    unsigned int _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[1], 2);
                    o_tmp_4[3] = _shfl_xor_sync_409;
                    o_vec_4[0] = ((qj2_4 != 0) ? o_tmp_4[0] : o_vec_4[0]);
                    o_vec_4[1] = ((qj2_4 != 0) ? o_tmp_4[1] : o_vec_4[1]);
                    o_vec_4[2] = ((qj2_4 == 0) ? o_tmp_4[2] : o_vec_4[2]);
                    o_vec_4[3] = ((qj2_4 == 0) ? o_tmp_4[3] : o_vec_4[3]);
                    int o_off_1_6 = o_row_base_4 + m_local_r_1_8 * 128 + (4 * sset_5 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_6)[0] = reinterpret_cast<int4*>(o_vec_4)[0];
                    int m_local_r_1_9 = m1_local;
                    __nv_bfloat162 _bit_cast_src_410 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                    o_vec_4[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_410)[0];
                    __nv_bfloat162 _bit_cast_src_411 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                    o_vec_4[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_411)[0];
                    __nv_bfloat162 _bit_cast_src_412 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                    o_vec_4[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_412)[0];
                    __nv_bfloat162 _bit_cast_src_413 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                    o_vec_4[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_413)[0];
                    unsigned int _shfl_xor_sync_414 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[1], 1);
                    o_tmp_4[0] = _shfl_xor_sync_414;
                    unsigned int _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[0], 1);
                    o_tmp_4[1] = _shfl_xor_sync_415;
                    unsigned int _shfl_xor_sync_416 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[3], 1);
                    o_tmp_4[2] = _shfl_xor_sync_416;
                    unsigned int _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[2], 1);
                    o_tmp_4[3] = _shfl_xor_sync_417;
                    o_vec_4[0] = ((qj1_4 != 0) ? o_tmp_4[0] : o_vec_4[0]);
                    o_vec_4[1] = ((qj1_4 == 0) ? o_tmp_4[1] : o_vec_4[1]);
                    o_vec_4[2] = ((qj1_4 != 0) ? o_tmp_4[2] : o_vec_4[2]);
                    o_vec_4[3] = ((qj1_4 == 0) ? o_tmp_4[3] : o_vec_4[3]);
                    unsigned int _shfl_xor_sync_418 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[2], 2);
                    o_tmp_4[0] = _shfl_xor_sync_418;
                    unsigned int _shfl_xor_sync_419 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[3], 2);
                    o_tmp_4[1] = _shfl_xor_sync_419;
                    unsigned int _shfl_xor_sync_420 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[0], 2);
                    o_tmp_4[2] = _shfl_xor_sync_420;
                    unsigned int _shfl_xor_sync_421 = __shfl_xor_sync(0xFFFFFFFF, o_vec_4[1], 2);
                    o_tmp_4[3] = _shfl_xor_sync_421;
                    o_vec_4[0] = ((qj2_4 != 0) ? o_tmp_4[0] : o_vec_4[0]);
                    o_vec_4[1] = ((qj2_4 != 0) ? o_tmp_4[1] : o_vec_4[1]);
                    o_vec_4[2] = ((qj2_4 == 0) ? o_tmp_4[2] : o_vec_4[2]);
                    o_vec_4[3] = ((qj2_4 == 0) ? o_tmp_4[3] : o_vec_4[3]);
                    int o_off_1_7 = o_row_base_4 + m_local_r_1_9 * 128 + (4 * sset_5 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_1_7)[0] = reinterpret_cast<int4*>(o_vec_4)[0];
                }
            }
        }
    }
}

} // extern "C"
