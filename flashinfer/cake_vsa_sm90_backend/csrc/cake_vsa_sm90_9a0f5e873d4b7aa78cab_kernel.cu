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
#define SMEM_MSTATS_OFF 157696
#define SMEM_MSTATS_STAGE_BYTES 1024
#define SMEM_MSTATS_STRIDE 1024
#define SMEM_PART_LO_OFF 91136
#define SMEM_PART_LO_STAGE_BYTES 16384
#define SMEM_PART_LO_STRIDE 16384
#define SMEM_PART_HI0_OFF 74752
#define SMEM_PART_HI0_STAGE_BYTES 16384
#define SMEM_PART_HI0_STRIDE 16384
#define SMEM_RECV_SMEM_OFF 158720
#define SMEM_RECV_SMEM_STAGE_BYTES 17408
#define SMEM_RECV_SMEM_STRIDE 17408
#define SMEM_VS_SMEM_OFF 156672
#define SMEM_VS_SMEM_STAGE_BYTES 1024
#define SMEM_VS_SMEM_STRIDE 1024
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 8192
#define SMEM_Q_SMEM_STRIDE 8192
#define SMEM_K_SMEM_OFF 9216
#define SMEM_K_SMEM_STAGE_BYTES 8192
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VL_SMEM_OFF 17408
#define SMEM_VL_SMEM_STAGE_BYTES 8192
#define SMEM_VL_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 107520
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 176128
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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_vsa_sm90_9a0f5e873d4b7aa78cab(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ Qs, float* __restrict__ Ks, float* __restrict__ Vs, int k_scale_groups, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2)
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
    #define recv_full_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* mstats = reinterpret_cast<float*>(smem_raw + 157696);
    const int mstats_addr = smem + 157696;
    float* part_lo = reinterpret_cast<float*>(smem_raw + 91136);
    const int part_lo_addr = smem + 91136;
    float* part_hi0 = reinterpret_cast<float*>(smem_raw + 74752);
    const int part_hi0_addr = smem + 74752;
    float* recv_smem = reinterpret_cast<float*>(smem_raw + 158720);
    const int recv_smem_addr = smem + 158720;
    float* vs_smem = reinterpret_cast<float*>(smem_raw + 156672);
    const int vs_smem_addr = smem + 156672;
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* k_smem = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int k_smem_addr = smem + 9216;
    uint8_t* vl_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int vl_smem_addr = smem + 17408;
    uint8_t* vt_smem = reinterpret_cast<uint8_t*>(smem_raw + 107520);
    const int vt_smem_addr = smem + 107520;

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

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
            // recv_full: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // === Task calls (declaration order) ===
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }
    int item = bid;
    int mb = seqlen_q / 64;
    int plan_base = item * 8;
    int meta = plan[plan_base];
    int cnt = meta & 15;
    int tile = plan[plan_base + 1];
    int blk_base = plan_base + 2;
    int head = tile / mb;
    int qb = tile - head * mb;
    int q_row = head * seqlen_q + qb * 64;
    int kv_base = head * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            if (cta_rank < 2) {
                int nsets = 2;
                int recv_tx = nsets * 8192 + 512;
                mbarrier_arrive_expect_tx(recv_full_addr, recv_tx);
            }
            int blk_pre[2];
            blk_pre[0] = plan[blk_base];
            blk_pre[1] = plan[blk_base + 1];
            mbarrier_arrive_expect_tx(q_full_addr, 8192);
            tma_2d_gmem2smem(q_smem_addr, (&Q), 0, q_row, q_full_addr);
            if (cnt > 0) {
                int blk_row_1_0 = kv_base + blk_pre[0] * 64;
                mbarrier_arrive_expect_tx(k_full0_addr, 8192);
                tma_2d_gmem2smem(k_smem_addr, (&K), 0, blk_row_1_0, k_full0_addr);
                mbarrier_arrive_expect_tx(v_full0_addr, 8192);
                tma_2d_gmem2smem(vl_smem_addr, (&V), 0, blk_row_1_0, v_full0_addr);
            }
            if (cnt > 1) {
                int blk_row_1_1 = kv_base + blk_pre[1] * 64;
                mbarrier_arrive_expect_tx(k_full1_addr, 8192);
                tma_2d_gmem2smem(k_smem_addr + 16384, (&K), 0, blk_row_1_1, k_full1_addr);
                mbarrier_arrive_expect_tx(v_full1_addr, 8192);
                tma_2d_gmem2smem(vl_smem_addr + 16384, (&V), 0, blk_row_1_1, v_full1_addr);
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
    int vsl_two = wg * 128;
    int m0_local = warp_in_wg * 16 + lane / 4;
    int m1_local = m0_local + 8;
    int vs_base = vsl_two + (lane & 3) * 2;
    float q_log2[2];
    q_log2[0] = Qs[q_row + m0_local] * scale_log2;
    q_log2[1] = Qs[q_row + m1_local] * scale_log2;
    float ks_regs[12];
    int ks_head = head * k_scale_groups;
    if (lim_even > 0) {
        int ks_blk = plan[blk_base];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk * 4) + 0);
            ks_regs[0 + 0] = _v4.x;
            ks_regs[0 + 1] = _v4.y;
            ks_regs[0 + 2] = _v4.z;
            ks_regs[0 + 3] = _v4.w;
        }
    }
    if (lim_odd > 1) {
        int ks_blk_1 = plan[blk_base + 1];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk_1 * 4) + 0);
            ks_regs[0 + 0] = _v4.x;
            ks_regs[0 + 1] = _v4.y;
            ks_regs[0 + 2] = _v4.z;
            ks_regs[0 + 3] = _v4.w;
        }
    }
    if (lim_even > 2) {
        int ks_blk_2 = plan[blk_base + 2];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk_2 * 4) + 0);
            ks_regs[4 + 0] = _v4.x;
            ks_regs[4 + 1] = _v4.y;
            ks_regs[4 + 2] = _v4.z;
            ks_regs[4 + 3] = _v4.w;
        }
    }
    if (lim_odd > 3) {
        int ks_blk_3 = plan[blk_base + 3];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk_3 * 4) + 0);
            ks_regs[4 + 0] = _v4.x;
            ks_regs[4 + 1] = _v4.y;
            ks_regs[4 + 2] = _v4.z;
            ks_regs[4 + 3] = _v4.w;
        }
    }
    if (lim_even > 4) {
        int ks_blk_4 = plan[blk_base + 4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk_4 * 4) + 0);
            ks_regs[8 + 0] = _v4.x;
            ks_regs[8 + 1] = _v4.y;
            ks_regs[8 + 2] = _v4.z;
            ks_regs[8 + 3] = _v4.w;
        }
    }
    if (lim_odd > 5) {
        int ks_blk_5 = plan[blk_base + 5];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk_5 * 4) + 0);
            ks_regs[8 + 0] = _v4.x;
            ks_regs[8 + 1] = _v4.y;
            ks_regs[8 + 2] = _v4.z;
            ks_regs[8 + 3] = _v4.w;
        }
    }
    vs_smem[vsl_two + tid_wg] = Vs[head * 128 + tid_wg];
    float d_o[64];
    float d_qk[32];
    unsigned int p_e4m3[8];
    float qk_sc[8];
    float row_max0 = -CUDART_INF_F;
    float row_max1 = -CUDART_INF_F;
    float row_sum0 = 0.0f;
    float row_sum1 = 0.0f;
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
    int t_lane = lane;
    int t_mi = t_lane >> 3;
    int t_rr = t_lane & 7;
    int t_kl = t_mi * 8 + t_rr;
    int ld_row = t_kl * 128;
    int t_par = t_rr >> 2 ^ t_mi & 1;
    unsigned int sel_a = (unsigned int)(((t_lane < 16) ? 25632 : 30001));
    unsigned int sel_b = (unsigned int)(((t_lane < 16) ? 30001 : 25632));
    int t_dl = t_rr * 2 + t_par;
    int st_row = t_dl * 128;
    int t_mih = t_mi >> 1;
    int t_dl7 = t_dl & 7;
    int t_kq = warp_in_wg >> 1;
    int t_kq4096 = t_kq * 4096;
    int t_kq2 = t_kq * 2;
    int t_dc0 = (warp_in_wg & 1) * 4;
    unsigned int tw[16];
    unsigned int tpw[4];
    unsigned int _phase_q_full_0 = 0;
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;
    uint64_t _wgmma_desc_6 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
    uint64_t _wgmma_desc_7 = (((uint64_t)(((k_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_7 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_7);
    uint64_t _wgmma_desc_8 = (((uint64_t)(((vt_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_8 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_8);
    if (lim_even > 0) {
        mbarrier_wait(k_full0_addr, 0);
        if (cnt > 2) {
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    int nblk_0 = plan[blk_base + 2];
                    int nblk_row_0 = kv_base + nblk_0 * 64;
                    mbarrier_arrive_expect_tx(k_full2_addr, 8192);
                    tma_2d_gmem2smem(k_smem_addr + 32768, (&K), 0, nblk_row_0, k_full2_addr);
                    mbarrier_arrive_expect_tx(v_full2_addr, 8192);
                    tma_2d_gmem2smem(vl_smem_addr + 32768, (&V), 0, nblk_row_0, v_full2_addr);
                }
            }
        }
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(_wgmma_b_0)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_0 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_0 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_0 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        qk_sc[0] = q_log2[0] * ks_regs[0];
        qk_sc[4] = q_log2[1] * ks_regs[0];
        qk_sc[1] = q_log2[0] * ks_regs[1];
        qk_sc[5] = q_log2[1] * ks_regs[1];
        qk_sc[2] = q_log2[0] * ks_regs[2];
        qk_sc[6] = q_log2[1] * ks_regs[2];
        qk_sc[3] = q_log2[0] * ks_regs[3];
        qk_sc[7] = q_log2[1] * ks_regs[3];
        float new_max0_0 = -CUDART_INF_F;
        float new_max1_0 = -CUDART_INF_F;
        float m0 = -CUDART_INF_F;
        float m1 = -CUDART_INF_F;
        if (scale_log2 >= 0.0f) {
            m0 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])) * qk_sc[0], fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])) * qk_sc[2], fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])) * qk_sc[4], fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])) * qk_sc[6], fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31])) * qk_sc[7]));
        } else {
            m0 = fmaxf(fmaxf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])) * qk_sc[0], fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])) * qk_sc[2], fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1 = fmaxf(fmaxf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])) * qk_sc[4], fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])) * qk_sc[6], fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31])) * qk_sc[7]));
        }
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
        m0 = fmaxf(m0, _shfl_xor_sync_9);
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
        m0 = fmaxf(m0, _shfl_xor_sync_10);
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
        m1 = fmaxf(m1, _shfl_xor_sync_11);
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
        m1 = fmaxf(m1, _shfl_xor_sync_12);
        new_max0_0 = m0;
        new_max1_0 = m1;
        row_max0 = new_max0_0;
        row_max1 = new_max1_0;
        float bias0_0 = 8.0f - row_max0;
        float bias1_0 = 8.0f - row_max1;
        float exp2 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0_0));
        float exp2_1 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0_0));
        d_qk[0] = exp2;
        d_qk[1] = exp2_1;
        row_sum0 += exp2 + exp2_1;
        float exp2_2 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[4], bias1_0));
        float exp2_3 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[4], bias1_0));
        d_qk[2] = exp2_2;
        d_qk[3] = exp2_3;
        row_sum1 += exp2_2 + exp2_3;
        float exp2_4 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0_0));
        float exp2_5 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0_0));
        d_qk[4] = exp2_4;
        d_qk[5] = exp2_5;
        row_sum0 += exp2_4 + exp2_5;
        float exp2_6 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[4], bias1_0));
        float exp2_7 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[4], bias1_0));
        d_qk[6] = exp2_6;
        d_qk[7] = exp2_7;
        row_sum1 += exp2_6 + exp2_7;
        float exp2_8 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0_0));
        float exp2_9 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0_0));
        d_qk[8] = exp2_8;
        d_qk[9] = exp2_9;
        row_sum0 += exp2_8 + exp2_9;
        float exp2_10 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[5], bias1_0));
        float exp2_11 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[5], bias1_0));
        d_qk[10] = exp2_10;
        d_qk[11] = exp2_11;
        row_sum1 += exp2_10 + exp2_11;
        float exp2_12 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0_0));
        float exp2_13 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0_0));
        d_qk[12] = exp2_12;
        d_qk[13] = exp2_13;
        row_sum0 += exp2_12 + exp2_13;
        float exp2_14 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[5], bias1_0));
        float exp2_15 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[5], bias1_0));
        d_qk[14] = exp2_14;
        d_qk[15] = exp2_15;
        row_sum1 += exp2_14 + exp2_15;
        uint16_t _cvt_satfinite_e4m3x2_f32_14;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_15;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_15) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_13;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_13) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_14), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_15));
        p_e4m3[0] = _prmt_13;
        uint16_t _cvt_satfinite_e4m3x2_f32_17;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_17) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_18;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_16;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_16) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_17), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_18));
        p_e4m3[1] = _prmt_16;
        uint16_t _cvt_satfinite_e4m3x2_f32_20;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_20) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_21;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_19;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_19) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_20), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_21));
        p_e4m3[2] = _prmt_19;
        uint16_t _cvt_satfinite_e4m3x2_f32_23;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_23) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_24;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_24) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_22;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_22) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_23), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_24));
        p_e4m3[3] = _prmt_22;
        mbarrier_wait(v_full0_addr, 0);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
            : "r"(vl_smem_addr + (unsigned int)(t_kq4096 + ld_row + ((t_rr ^ t_dc0) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
            : "r"(vl_smem_addr + (unsigned int)(t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 1) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
            : "r"(vl_smem_addr + (unsigned int)(t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 2) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
            : "r"(vl_smem_addr + (unsigned int)(t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 3) << 4)))
            : "memory");
        uint32_t _prmt_25;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_25) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_25;
        uint32_t _prmt_26;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_26) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_26;
        uint32_t _prmt_27;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_27) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_27;
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_28) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_28;
        uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_30;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_30) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_30;
        uint32_t _prmt_31;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_31) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_31;
        uint32_t _prmt_32;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_32) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_32;
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_33) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_33;
        uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_35;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_35) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_35;
        uint32_t _prmt_36;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_36) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_36;
        uint32_t _prmt_37;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_37) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_37;
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_38;
        uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_40;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_40) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_40;
        uint32_t _prmt_41;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_41) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_41;
        uint32_t _prmt_42;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_42) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_42;
        uint32_t _prmt_43;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_43) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_43;
        uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 2, 128;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(_wgmma_b_1)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_0 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0_0));
        float exp2_1_0 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0_0));
        d_qk[16] = exp2_0;
        d_qk[17] = exp2_1_0;
        row_sum0 += exp2_0 + exp2_1_0;
        float exp2_2_0 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1_0));
        float exp2_3_0 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1_0));
        d_qk[18] = exp2_2_0;
        d_qk[19] = exp2_3_0;
        row_sum1 += exp2_2_0 + exp2_3_0;
        float exp2_4_0 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0_0));
        float exp2_5_0 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0_0));
        d_qk[20] = exp2_4_0;
        d_qk[21] = exp2_5_0;
        row_sum0 += exp2_4_0 + exp2_5_0;
        float exp2_6_0 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1_0));
        float exp2_7_0 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1_0));
        d_qk[22] = exp2_6_0;
        d_qk[23] = exp2_7_0;
        row_sum1 += exp2_6_0 + exp2_7_0;
        float exp2_8_0 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0_0));
        float exp2_9_0 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0_0));
        d_qk[24] = exp2_8_0;
        d_qk[25] = exp2_9_0;
        row_sum0 += exp2_8_0 + exp2_9_0;
        float exp2_10_0 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1_0));
        float exp2_11_0 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1_0));
        d_qk[26] = exp2_10_0;
        d_qk[27] = exp2_11_0;
        row_sum1 += exp2_10_0 + exp2_11_0;
        float exp2_12_0 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0_0));
        float exp2_13_0 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0_0));
        d_qk[28] = exp2_12_0;
        d_qk[29] = exp2_13_0;
        row_sum0 += exp2_12_0 + exp2_13_0;
        float exp2_14_0 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1_0));
        float exp2_15_0 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1_0));
        d_qk[30] = exp2_14_0;
        d_qk[31] = exp2_15_0;
        row_sum1 += exp2_14_0 + exp2_15_0;
        uint16_t _cvt_satfinite_e4m3x2_f32_46;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_46) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_47;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_47) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_45;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_45) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_46), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_47));
        p_e4m3[4] = _prmt_45;
        uint16_t _cvt_satfinite_e4m3x2_f32_49;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_50;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_50) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_48;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_48) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_49), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_50));
        p_e4m3[5] = _prmt_48;
        uint16_t _cvt_satfinite_e4m3x2_f32_52;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_52) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_53;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_51;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_51) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_52), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_53));
        p_e4m3[6] = _prmt_51;
        uint16_t _cvt_satfinite_e4m3x2_f32_55;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_55) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_56;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_56) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_54;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_54) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_55), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_56));
        p_e4m3[7] = _prmt_54;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_57 = (((uint64_t)(((k_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_57 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_57);
    if (lim_odd > 1) {
        mbarrier_wait(k_full1_addr, 0);
        if (cnt > 3) {
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    int nblk_1 = plan[blk_base + 3];
                    int nblk_row_1 = kv_base + nblk_1 * 64;
                    mbarrier_arrive_expect_tx(k_full3_addr, 8192);
                    tma_2d_gmem2smem(k_smem_addr + 49152, (&K), 0, nblk_row_1, k_full3_addr);
                    mbarrier_arrive_expect_tx(v_full3_addr, 8192);
                    tma_2d_gmem2smem(vl_smem_addr + 49152, (&V), 0, nblk_row_1, v_full3_addr);
                }
            }
        }
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(kd_0)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(kd_0 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(kd_0 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(kd_0 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        qk_sc[0] = q_log2[0] * ks_regs[0];
        qk_sc[4] = q_log2[1] * ks_regs[0];
        qk_sc[1] = q_log2[0] * ks_regs[1];
        qk_sc[5] = q_log2[1] * ks_regs[1];
        qk_sc[2] = q_log2[0] * ks_regs[2];
        qk_sc[6] = q_log2[1] * ks_regs[2];
        qk_sc[3] = q_log2[0] * ks_regs[3];
        qk_sc[7] = q_log2[1] * ks_regs[3];
        float new_max0_1 = -CUDART_INF_F;
        float new_max1_1 = -CUDART_INF_F;
        float m0_1 = -CUDART_INF_F;
        float m1_1 = -CUDART_INF_F;
        if (scale_log2 >= 0.0f) {
            m0_1 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])) * qk_sc[0], fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])) * qk_sc[2], fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_1 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])) * qk_sc[4], fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])) * qk_sc[6], fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31])) * qk_sc[7]));
        } else {
            m0_1 = fmaxf(fmaxf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])) * qk_sc[0], fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])) * qk_sc[2], fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_1 = fmaxf(fmaxf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])) * qk_sc[4], fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])) * qk_sc[6], fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31])) * qk_sc[7]));
        }
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 2);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_58);
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 1);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_59);
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 2);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_60);
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 1);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_61);
        new_max0_1 = m0_1;
        new_max1_1 = m1_1;
        row_max0 = new_max0_1;
        row_max1 = new_max1_1;
        float bias0_1 = 8.0f - row_max0;
        float bias1_1 = 8.0f - row_max1;
        float exp2_16 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0_1));
        float exp2_1_1 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0_1));
        d_qk[0] = exp2_16;
        d_qk[1] = exp2_1_1;
        row_sum0 += exp2_16 + exp2_1_1;
        float exp2_2_1 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[4], bias1_1));
        float exp2_3_1 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[4], bias1_1));
        d_qk[2] = exp2_2_1;
        d_qk[3] = exp2_3_1;
        row_sum1 += exp2_2_1 + exp2_3_1;
        float exp2_4_1 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0_1));
        float exp2_5_1 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0_1));
        d_qk[4] = exp2_4_1;
        d_qk[5] = exp2_5_1;
        row_sum0 += exp2_4_1 + exp2_5_1;
        float exp2_6_1 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[4], bias1_1));
        float exp2_7_1 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[4], bias1_1));
        d_qk[6] = exp2_6_1;
        d_qk[7] = exp2_7_1;
        row_sum1 += exp2_6_1 + exp2_7_1;
        float exp2_8_1 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0_1));
        float exp2_9_1 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0_1));
        d_qk[8] = exp2_8_1;
        d_qk[9] = exp2_9_1;
        row_sum0 += exp2_8_1 + exp2_9_1;
        float exp2_10_1 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[5], bias1_1));
        float exp2_11_1 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[5], bias1_1));
        d_qk[10] = exp2_10_1;
        d_qk[11] = exp2_11_1;
        row_sum1 += exp2_10_1 + exp2_11_1;
        float exp2_12_1 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0_1));
        float exp2_13_1 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0_1));
        d_qk[12] = exp2_12_1;
        d_qk[13] = exp2_13_1;
        row_sum0 += exp2_12_1 + exp2_13_1;
        float exp2_14_1 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[5], bias1_1));
        float exp2_15_1 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[5], bias1_1));
        d_qk[14] = exp2_14_1;
        d_qk[15] = exp2_15_1;
        row_sum1 += exp2_14_1 + exp2_15_1;
        uint16_t _cvt_satfinite_e4m3x2_f32_63;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_63) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_64;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_64) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_62;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_62) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_63), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_64));
        p_e4m3[0] = _prmt_62;
        uint16_t _cvt_satfinite_e4m3x2_f32_66;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_66) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_67;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_67) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_65;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_65) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_66), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_67));
        p_e4m3[1] = _prmt_65;
        uint16_t _cvt_satfinite_e4m3x2_f32_69;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_69) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_70;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_70) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_68;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_68) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_69), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_70));
        p_e4m3[2] = _prmt_68;
        uint16_t _cvt_satfinite_e4m3x2_f32_72;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_72) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_73;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_73) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_71;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_71) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_72), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_73));
        p_e4m3[3] = _prmt_71;
        mbarrier_wait(v_full1_addr, 0);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
            : "r"(vl_smem_addr + (unsigned int)(16384 + t_kq4096 + ld_row + ((t_rr ^ t_dc0) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
            : "r"(vl_smem_addr + (unsigned int)(16384 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 1) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
            : "r"(vl_smem_addr + (unsigned int)(16384 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 2) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
            : "r"(vl_smem_addr + (unsigned int)(16384 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 3) << 4)))
            : "memory");
        uint32_t _prmt_74;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_74) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_74;
        uint32_t _prmt_75;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_75) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_75;
        uint32_t _prmt_76;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_76) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_76;
        uint32_t _prmt_77;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_77) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_77;
        uint32_t _stmatrix_addr_78 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_78), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_79;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_79) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_79;
        uint32_t _prmt_80;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_80) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_80;
        uint32_t _prmt_81;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_81) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_81;
        uint32_t _prmt_82;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_82) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_82;
        uint32_t _stmatrix_addr_83 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_83), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_84;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_84) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_84;
        uint32_t _prmt_85;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_85) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_85;
        uint32_t _prmt_86;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_86) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_86;
        uint32_t _prmt_87;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_87) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_87;
        uint32_t _stmatrix_addr_88 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_88), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_89;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_89) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_89;
        uint32_t _prmt_90;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_90) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_90;
        uint32_t _prmt_91;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_91) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_91;
        uint32_t _prmt_92;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_92) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_92;
        uint32_t _stmatrix_addr_93 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_93), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 3, 128;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(_wgmma_b_1 + 4)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_16_1 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0_1));
        float exp2_1_1_1 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0_1));
        d_qk[16] = exp2_16_1;
        d_qk[17] = exp2_1_1_1;
        row_sum0 += exp2_16_1 + exp2_1_1_1;
        float exp2_2_1_1 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1_1));
        float exp2_3_1_1 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1_1));
        d_qk[18] = exp2_2_1_1;
        d_qk[19] = exp2_3_1_1;
        row_sum1 += exp2_2_1_1 + exp2_3_1_1;
        float exp2_4_1_1 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0_1));
        float exp2_5_1_1 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0_1));
        d_qk[20] = exp2_4_1_1;
        d_qk[21] = exp2_5_1_1;
        row_sum0 += exp2_4_1_1 + exp2_5_1_1;
        float exp2_6_1_1 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1_1));
        float exp2_7_1_1 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1_1));
        d_qk[22] = exp2_6_1_1;
        d_qk[23] = exp2_7_1_1;
        row_sum1 += exp2_6_1_1 + exp2_7_1_1;
        float exp2_8_1_1 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0_1));
        float exp2_9_1_1 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0_1));
        d_qk[24] = exp2_8_1_1;
        d_qk[25] = exp2_9_1_1;
        row_sum0 += exp2_8_1_1 + exp2_9_1_1;
        float exp2_10_1_1 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1_1));
        float exp2_11_1_1 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1_1));
        d_qk[26] = exp2_10_1_1;
        d_qk[27] = exp2_11_1_1;
        row_sum1 += exp2_10_1_1 + exp2_11_1_1;
        float exp2_12_1_1 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0_1));
        float exp2_13_1_1 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0_1));
        d_qk[28] = exp2_12_1_1;
        d_qk[29] = exp2_13_1_1;
        row_sum0 += exp2_12_1_1 + exp2_13_1_1;
        float exp2_14_1_1 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1_1));
        float exp2_15_1_1 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1_1));
        d_qk[30] = exp2_14_1_1;
        d_qk[31] = exp2_15_1_1;
        row_sum1 += exp2_14_1_1 + exp2_15_1_1;
        uint16_t _cvt_satfinite_e4m3x2_f32_95;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_95) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_96;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_96) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_94;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_94) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_95), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_96));
        p_e4m3[4] = _prmt_94;
        uint16_t _cvt_satfinite_e4m3x2_f32_98;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_98) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_99;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_99) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_97;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_97) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_98), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_99));
        p_e4m3[5] = _prmt_97;
        uint16_t _cvt_satfinite_e4m3x2_f32_101;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_101) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_102;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_102) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_100;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_100) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_101), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_102));
        p_e4m3[6] = _prmt_100;
        uint16_t _cvt_satfinite_e4m3x2_f32_104;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_104) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_105;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_105) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_103;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_103) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_104), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_105));
        p_e4m3[7] = _prmt_103;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_106 = (((uint64_t)(((k_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_106 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_106);
    uint64_t _wgmma_desc_107 = (((uint64_t)(((vt_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t vtd_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_107 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_107);
    if (lim_even > 2) {
        mbarrier_wait(k_full2_addr, 0);
        if (cnt > 4) {
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    int nblk_2 = plan[blk_base + 4];
                    int nblk_row_2 = kv_base + nblk_2 * 64;
                    mbarrier_arrive_expect_tx(k_full4_addr, 8192);
                    tma_2d_gmem2smem(k_smem_addr + 65536, (&K), 0, nblk_row_2, k_full4_addr);
                    mbarrier_arrive_expect_tx(v_full4_addr, 8192);
                    tma_2d_gmem2smem(vl_smem_addr + 65536, (&V), 0, nblk_row_2, v_full4_addr);
                }
            }
        }
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(kd_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(kd_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(kd_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(kd_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        qk_sc[0] = q_log2[0] * ks_regs[4];
        qk_sc[4] = q_log2[1] * ks_regs[4];
        qk_sc[1] = q_log2[0] * ks_regs[5];
        qk_sc[5] = q_log2[1] * ks_regs[5];
        qk_sc[2] = q_log2[0] * ks_regs[6];
        qk_sc[6] = q_log2[1] * ks_regs[6];
        qk_sc[3] = q_log2[0] * ks_regs[7];
        qk_sc[7] = q_log2[1] * ks_regs[7];
        float new_max0_2 = -CUDART_INF_F;
        float new_max1_2 = -CUDART_INF_F;
        float m0_2 = -CUDART_INF_F;
        float m1_2 = -CUDART_INF_F;
        if (scale_log2 >= 0.0f) {
            m0_2 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])) * qk_sc[0], fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])) * qk_sc[2], fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_2 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])) * qk_sc[4], fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])) * qk_sc[6], fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31])) * qk_sc[7]));
        } else {
            m0_2 = fmaxf(fmaxf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])) * qk_sc[0], fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])) * qk_sc[2], fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_2 = fmaxf(fmaxf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])) * qk_sc[4], fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])) * qk_sc[6], fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31])) * qk_sc[7]));
        }
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 2);
        m0_2 = fmaxf(m0_2, _shfl_xor_sync_108);
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 1);
        m0_2 = fmaxf(m0_2, _shfl_xor_sync_109);
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 2);
        m1_2 = fmaxf(m1_2, _shfl_xor_sync_110);
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 1);
        m1_2 = fmaxf(m1_2, _shfl_xor_sync_111);
        new_max0_2 = m0_2;
        new_max1_2 = m1_2;
        float merged_max0_0 = fmaxf(row_max0, new_max0_2);
        float merged_max1_0 = fmaxf(row_max1, new_max1_2);
        int grow0_0 = merged_max0_0 - row_max0 > 0.8073549f;
        int grow1_0 = merged_max1_0 - row_max1 > 0.8073549f;
        float _exp2_112 = approx_exp2(row_max0 - merged_max0_0);
        float rescale0_0 = ((grow0_0 != 0) ? _exp2_112 : 1.0f);
        float _exp2_113 = approx_exp2(row_max1 - merged_max1_0);
        float rescale1_0 = ((grow1_0 != 0) ? _exp2_113 : 1.0f);
        row_max0 = ((grow0_0 != 0) ? merged_max0_0 : row_max0);
        row_max1 = ((grow1_0 != 0) ? merged_max1_0 : row_max1);
        d_o[0] = d_o[0] * rescale0_0;
        d_o[1] = d_o[1] * rescale0_0;
        d_o[4] = d_o[4] * rescale0_0;
        d_o[5] = d_o[5] * rescale0_0;
        d_o[8] = d_o[8] * rescale0_0;
        d_o[9] = d_o[9] * rescale0_0;
        d_o[12] = d_o[12] * rescale0_0;
        d_o[13] = d_o[13] * rescale0_0;
        d_o[16] = d_o[16] * rescale0_0;
        d_o[17] = d_o[17] * rescale0_0;
        d_o[20] = d_o[20] * rescale0_0;
        d_o[21] = d_o[21] * rescale0_0;
        d_o[24] = d_o[24] * rescale0_0;
        d_o[25] = d_o[25] * rescale0_0;
        d_o[28] = d_o[28] * rescale0_0;
        d_o[29] = d_o[29] * rescale0_0;
        d_o[32] = d_o[32] * rescale0_0;
        d_o[33] = d_o[33] * rescale0_0;
        d_o[36] = d_o[36] * rescale0_0;
        d_o[37] = d_o[37] * rescale0_0;
        d_o[40] = d_o[40] * rescale0_0;
        d_o[41] = d_o[41] * rescale0_0;
        d_o[44] = d_o[44] * rescale0_0;
        d_o[45] = d_o[45] * rescale0_0;
        d_o[48] = d_o[48] * rescale0_0;
        d_o[49] = d_o[49] * rescale0_0;
        d_o[52] = d_o[52] * rescale0_0;
        d_o[53] = d_o[53] * rescale0_0;
        d_o[56] = d_o[56] * rescale0_0;
        d_o[57] = d_o[57] * rescale0_0;
        d_o[60] = d_o[60] * rescale0_0;
        d_o[61] = d_o[61] * rescale0_0;
        d_o[2] = d_o[2] * rescale1_0;
        d_o[3] = d_o[3] * rescale1_0;
        d_o[6] = d_o[6] * rescale1_0;
        d_o[7] = d_o[7] * rescale1_0;
        d_o[10] = d_o[10] * rescale1_0;
        d_o[11] = d_o[11] * rescale1_0;
        d_o[14] = d_o[14] * rescale1_0;
        d_o[15] = d_o[15] * rescale1_0;
        d_o[18] = d_o[18] * rescale1_0;
        d_o[19] = d_o[19] * rescale1_0;
        d_o[22] = d_o[22] * rescale1_0;
        d_o[23] = d_o[23] * rescale1_0;
        d_o[26] = d_o[26] * rescale1_0;
        d_o[27] = d_o[27] * rescale1_0;
        d_o[30] = d_o[30] * rescale1_0;
        d_o[31] = d_o[31] * rescale1_0;
        d_o[34] = d_o[34] * rescale1_0;
        d_o[35] = d_o[35] * rescale1_0;
        d_o[38] = d_o[38] * rescale1_0;
        d_o[39] = d_o[39] * rescale1_0;
        d_o[42] = d_o[42] * rescale1_0;
        d_o[43] = d_o[43] * rescale1_0;
        d_o[46] = d_o[46] * rescale1_0;
        d_o[47] = d_o[47] * rescale1_0;
        d_o[50] = d_o[50] * rescale1_0;
        d_o[51] = d_o[51] * rescale1_0;
        d_o[54] = d_o[54] * rescale1_0;
        d_o[55] = d_o[55] * rescale1_0;
        d_o[58] = d_o[58] * rescale1_0;
        d_o[59] = d_o[59] * rescale1_0;
        d_o[62] = d_o[62] * rescale1_0;
        d_o[63] = d_o[63] * rescale1_0;
        row_sum0 = row_sum0 * rescale0_0;
        row_sum1 = row_sum1 * rescale1_0;
        float bias0_2 = 8.0f - row_max0;
        float bias1_2 = 8.0f - row_max1;
        float exp2_17 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0_2));
        float exp2_1_2 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0_2));
        d_qk[0] = exp2_17;
        d_qk[1] = exp2_1_2;
        row_sum0 += exp2_17 + exp2_1_2;
        float exp2_2_2 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[4], bias1_2));
        float exp2_3_2 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[4], bias1_2));
        d_qk[2] = exp2_2_2;
        d_qk[3] = exp2_3_2;
        row_sum1 += exp2_2_2 + exp2_3_2;
        float exp2_4_2 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0_2));
        float exp2_5_2 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0_2));
        d_qk[4] = exp2_4_2;
        d_qk[5] = exp2_5_2;
        row_sum0 += exp2_4_2 + exp2_5_2;
        float exp2_6_2 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[4], bias1_2));
        float exp2_7_2 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[4], bias1_2));
        d_qk[6] = exp2_6_2;
        d_qk[7] = exp2_7_2;
        row_sum1 += exp2_6_2 + exp2_7_2;
        float exp2_8_2 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0_2));
        float exp2_9_2 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0_2));
        d_qk[8] = exp2_8_2;
        d_qk[9] = exp2_9_2;
        row_sum0 += exp2_8_2 + exp2_9_2;
        float exp2_10_2 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[5], bias1_2));
        float exp2_11_2 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[5], bias1_2));
        d_qk[10] = exp2_10_2;
        d_qk[11] = exp2_11_2;
        row_sum1 += exp2_10_2 + exp2_11_2;
        float exp2_12_2 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0_2));
        float exp2_13_2 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0_2));
        d_qk[12] = exp2_12_2;
        d_qk[13] = exp2_13_2;
        row_sum0 += exp2_12_2 + exp2_13_2;
        float exp2_14_2 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[5], bias1_2));
        float exp2_15_2 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[5], bias1_2));
        d_qk[14] = exp2_14_2;
        d_qk[15] = exp2_15_2;
        row_sum1 += exp2_14_2 + exp2_15_2;
        uint16_t _cvt_satfinite_e4m3x2_f32_115;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_115) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_116;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_116) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_114;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_114) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_115), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_116));
        p_e4m3[0] = _prmt_114;
        uint16_t _cvt_satfinite_e4m3x2_f32_118;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_118) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_119;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_119) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_117;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_117) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_118), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_119));
        p_e4m3[1] = _prmt_117;
        uint16_t _cvt_satfinite_e4m3x2_f32_121;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_121) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_122;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_122) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_120;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_120) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_121), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_122));
        p_e4m3[2] = _prmt_120;
        uint16_t _cvt_satfinite_e4m3x2_f32_124;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_124) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_125;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_125) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_123;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_123) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_124), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_125));
        p_e4m3[3] = _prmt_123;
        mbarrier_wait(v_full2_addr, 0);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
            : "r"(vl_smem_addr + (unsigned int)(32768 + t_kq4096 + ld_row + ((t_rr ^ t_dc0) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
            : "r"(vl_smem_addr + (unsigned int)(32768 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 1) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
            : "r"(vl_smem_addr + (unsigned int)(32768 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 2) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
            : "r"(vl_smem_addr + (unsigned int)(32768 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 3) << 4)))
            : "memory");
        uint32_t _prmt_126;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_126) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_126;
        uint32_t _prmt_127;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_127) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_127;
        uint32_t _prmt_128;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_128) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_128;
        uint32_t _prmt_129;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_129) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_129;
        uint32_t _stmatrix_addr_130 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_130), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_131;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_131) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_131;
        uint32_t _prmt_132;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_132) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_132;
        uint32_t _prmt_133;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_133) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_133;
        uint32_t _prmt_134;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_134) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_134;
        uint32_t _stmatrix_addr_135 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_135), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_136;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_136) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_136;
        uint32_t _prmt_137;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_137) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_137;
        uint32_t _prmt_138;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_138) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_138;
        uint32_t _prmt_139;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_139) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_139;
        uint32_t _stmatrix_addr_140 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_140), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_141;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_141) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_141;
        uint32_t _prmt_142;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_142) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_142;
        uint32_t _prmt_143;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_143) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_143;
        uint32_t _prmt_144;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_144) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_144;
        uint32_t _stmatrix_addr_145 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_145), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 2, 128;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(vtd_0)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_17_1 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0_2));
        float exp2_1_2_1 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0_2));
        d_qk[16] = exp2_17_1;
        d_qk[17] = exp2_1_2_1;
        row_sum0 += exp2_17_1 + exp2_1_2_1;
        float exp2_2_2_1 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1_2));
        float exp2_3_2_1 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1_2));
        d_qk[18] = exp2_2_2_1;
        d_qk[19] = exp2_3_2_1;
        row_sum1 += exp2_2_2_1 + exp2_3_2_1;
        float exp2_4_2_1 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0_2));
        float exp2_5_2_1 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0_2));
        d_qk[20] = exp2_4_2_1;
        d_qk[21] = exp2_5_2_1;
        row_sum0 += exp2_4_2_1 + exp2_5_2_1;
        float exp2_6_2_1 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1_2));
        float exp2_7_2_1 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1_2));
        d_qk[22] = exp2_6_2_1;
        d_qk[23] = exp2_7_2_1;
        row_sum1 += exp2_6_2_1 + exp2_7_2_1;
        float exp2_8_2_1 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0_2));
        float exp2_9_2_1 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0_2));
        d_qk[24] = exp2_8_2_1;
        d_qk[25] = exp2_9_2_1;
        row_sum0 += exp2_8_2_1 + exp2_9_2_1;
        float exp2_10_2_1 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1_2));
        float exp2_11_2_1 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1_2));
        d_qk[26] = exp2_10_2_1;
        d_qk[27] = exp2_11_2_1;
        row_sum1 += exp2_10_2_1 + exp2_11_2_1;
        float exp2_12_2_1 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0_2));
        float exp2_13_2_1 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0_2));
        d_qk[28] = exp2_12_2_1;
        d_qk[29] = exp2_13_2_1;
        row_sum0 += exp2_12_2_1 + exp2_13_2_1;
        float exp2_14_2_1 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1_2));
        float exp2_15_2_1 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1_2));
        d_qk[30] = exp2_14_2_1;
        d_qk[31] = exp2_15_2_1;
        row_sum1 += exp2_14_2_1 + exp2_15_2_1;
        uint16_t _cvt_satfinite_e4m3x2_f32_147;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_147) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_148;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_148) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_146;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_146) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_147), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_148));
        p_e4m3[4] = _prmt_146;
        uint16_t _cvt_satfinite_e4m3x2_f32_150;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_150) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_151;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_151) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_149;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_149) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_150), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_151));
        p_e4m3[5] = _prmt_149;
        uint16_t _cvt_satfinite_e4m3x2_f32_153;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_153) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_154;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_154) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_152;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_152) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_153), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_154));
        p_e4m3[6] = _prmt_152;
        uint16_t _cvt_satfinite_e4m3x2_f32_156;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_156) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_157;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_157) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_155;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_155) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_156), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_157));
        p_e4m3[7] = _prmt_155;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(vtd_0 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_158 = (((uint64_t)(((k_smem_addr + 49152)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_158 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_158);
    if (lim_odd > 3) {
        mbarrier_wait(k_full3_addr, 0);
        if (cnt > 5) {
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    int nblk_3 = plan[blk_base + 5];
                    int nblk_row_3 = kv_base + nblk_3 * 64;
                    mbarrier_arrive_expect_tx(k_full5_addr, 8192);
                    tma_2d_gmem2smem(k_smem_addr + 81920, (&K), 0, nblk_row_3, k_full5_addr);
                    mbarrier_arrive_expect_tx(v_full5_addr, 8192);
                    tma_2d_gmem2smem(vl_smem_addr + 81920, (&V), 0, nblk_row_3, v_full5_addr);
                }
            }
        }
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(kd_2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(kd_2 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(kd_2 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(kd_2 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        qk_sc[0] = q_log2[0] * ks_regs[4];
        qk_sc[4] = q_log2[1] * ks_regs[4];
        qk_sc[1] = q_log2[0] * ks_regs[5];
        qk_sc[5] = q_log2[1] * ks_regs[5];
        qk_sc[2] = q_log2[0] * ks_regs[6];
        qk_sc[6] = q_log2[1] * ks_regs[6];
        qk_sc[3] = q_log2[0] * ks_regs[7];
        qk_sc[7] = q_log2[1] * ks_regs[7];
        float new_max0_3 = -CUDART_INF_F;
        float new_max1_3 = -CUDART_INF_F;
        float m0_3 = -CUDART_INF_F;
        float m1_3 = -CUDART_INF_F;
        if (scale_log2 >= 0.0f) {
            m0_3 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])) * qk_sc[0], fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])) * qk_sc[2], fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_3 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])) * qk_sc[4], fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])) * qk_sc[6], fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31])) * qk_sc[7]));
        } else {
            m0_3 = fmaxf(fmaxf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])) * qk_sc[0], fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])) * qk_sc[2], fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_3 = fmaxf(fmaxf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])) * qk_sc[4], fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])) * qk_sc[6], fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31])) * qk_sc[7]));
        }
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, m0_3, 2);
        m0_3 = fmaxf(m0_3, _shfl_xor_sync_159);
        float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, m0_3, 1);
        m0_3 = fmaxf(m0_3, _shfl_xor_sync_160);
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, m1_3, 2);
        m1_3 = fmaxf(m1_3, _shfl_xor_sync_161);
        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, m1_3, 1);
        m1_3 = fmaxf(m1_3, _shfl_xor_sync_162);
        new_max0_3 = m0_3;
        new_max1_3 = m1_3;
        float merged_max0_1 = fmaxf(row_max0, new_max0_3);
        float merged_max1_1 = fmaxf(row_max1, new_max1_3);
        int grow0_1 = merged_max0_1 - row_max0 > 0.8073549f;
        int grow1_1 = merged_max1_1 - row_max1 > 0.8073549f;
        float _exp2_163 = approx_exp2(row_max0 - merged_max0_1);
        float rescale0_1 = ((grow0_1 != 0) ? _exp2_163 : 1.0f);
        float _exp2_164 = approx_exp2(row_max1 - merged_max1_1);
        float rescale1_1 = ((grow1_1 != 0) ? _exp2_164 : 1.0f);
        row_max0 = ((grow0_1 != 0) ? merged_max0_1 : row_max0);
        row_max1 = ((grow1_1 != 0) ? merged_max1_1 : row_max1);
        d_o[0] = d_o[0] * rescale0_1;
        d_o[1] = d_o[1] * rescale0_1;
        d_o[4] = d_o[4] * rescale0_1;
        d_o[5] = d_o[5] * rescale0_1;
        d_o[8] = d_o[8] * rescale0_1;
        d_o[9] = d_o[9] * rescale0_1;
        d_o[12] = d_o[12] * rescale0_1;
        d_o[13] = d_o[13] * rescale0_1;
        d_o[16] = d_o[16] * rescale0_1;
        d_o[17] = d_o[17] * rescale0_1;
        d_o[20] = d_o[20] * rescale0_1;
        d_o[21] = d_o[21] * rescale0_1;
        d_o[24] = d_o[24] * rescale0_1;
        d_o[25] = d_o[25] * rescale0_1;
        d_o[28] = d_o[28] * rescale0_1;
        d_o[29] = d_o[29] * rescale0_1;
        d_o[32] = d_o[32] * rescale0_1;
        d_o[33] = d_o[33] * rescale0_1;
        d_o[36] = d_o[36] * rescale0_1;
        d_o[37] = d_o[37] * rescale0_1;
        d_o[40] = d_o[40] * rescale0_1;
        d_o[41] = d_o[41] * rescale0_1;
        d_o[44] = d_o[44] * rescale0_1;
        d_o[45] = d_o[45] * rescale0_1;
        d_o[48] = d_o[48] * rescale0_1;
        d_o[49] = d_o[49] * rescale0_1;
        d_o[52] = d_o[52] * rescale0_1;
        d_o[53] = d_o[53] * rescale0_1;
        d_o[56] = d_o[56] * rescale0_1;
        d_o[57] = d_o[57] * rescale0_1;
        d_o[60] = d_o[60] * rescale0_1;
        d_o[61] = d_o[61] * rescale0_1;
        d_o[2] = d_o[2] * rescale1_1;
        d_o[3] = d_o[3] * rescale1_1;
        d_o[6] = d_o[6] * rescale1_1;
        d_o[7] = d_o[7] * rescale1_1;
        d_o[10] = d_o[10] * rescale1_1;
        d_o[11] = d_o[11] * rescale1_1;
        d_o[14] = d_o[14] * rescale1_1;
        d_o[15] = d_o[15] * rescale1_1;
        d_o[18] = d_o[18] * rescale1_1;
        d_o[19] = d_o[19] * rescale1_1;
        d_o[22] = d_o[22] * rescale1_1;
        d_o[23] = d_o[23] * rescale1_1;
        d_o[26] = d_o[26] * rescale1_1;
        d_o[27] = d_o[27] * rescale1_1;
        d_o[30] = d_o[30] * rescale1_1;
        d_o[31] = d_o[31] * rescale1_1;
        d_o[34] = d_o[34] * rescale1_1;
        d_o[35] = d_o[35] * rescale1_1;
        d_o[38] = d_o[38] * rescale1_1;
        d_o[39] = d_o[39] * rescale1_1;
        d_o[42] = d_o[42] * rescale1_1;
        d_o[43] = d_o[43] * rescale1_1;
        d_o[46] = d_o[46] * rescale1_1;
        d_o[47] = d_o[47] * rescale1_1;
        d_o[50] = d_o[50] * rescale1_1;
        d_o[51] = d_o[51] * rescale1_1;
        d_o[54] = d_o[54] * rescale1_1;
        d_o[55] = d_o[55] * rescale1_1;
        d_o[58] = d_o[58] * rescale1_1;
        d_o[59] = d_o[59] * rescale1_1;
        d_o[62] = d_o[62] * rescale1_1;
        d_o[63] = d_o[63] * rescale1_1;
        row_sum0 = row_sum0 * rescale0_1;
        row_sum1 = row_sum1 * rescale1_1;
        float bias0_3 = 8.0f - row_max0;
        float bias1_3 = 8.0f - row_max1;
        float exp2_18 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0_3));
        float exp2_1_3 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0_3));
        d_qk[0] = exp2_18;
        d_qk[1] = exp2_1_3;
        row_sum0 += exp2_18 + exp2_1_3;
        float exp2_2_3 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[4], bias1_3));
        float exp2_3_3 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[4], bias1_3));
        d_qk[2] = exp2_2_3;
        d_qk[3] = exp2_3_3;
        row_sum1 += exp2_2_3 + exp2_3_3;
        float exp2_4_3 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0_3));
        float exp2_5_3 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0_3));
        d_qk[4] = exp2_4_3;
        d_qk[5] = exp2_5_3;
        row_sum0 += exp2_4_3 + exp2_5_3;
        float exp2_6_3 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[4], bias1_3));
        float exp2_7_3 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[4], bias1_3));
        d_qk[6] = exp2_6_3;
        d_qk[7] = exp2_7_3;
        row_sum1 += exp2_6_3 + exp2_7_3;
        float exp2_8_3 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0_3));
        float exp2_9_3 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0_3));
        d_qk[8] = exp2_8_3;
        d_qk[9] = exp2_9_3;
        row_sum0 += exp2_8_3 + exp2_9_3;
        float exp2_10_3 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[5], bias1_3));
        float exp2_11_3 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[5], bias1_3));
        d_qk[10] = exp2_10_3;
        d_qk[11] = exp2_11_3;
        row_sum1 += exp2_10_3 + exp2_11_3;
        float exp2_12_3 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0_3));
        float exp2_13_3 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0_3));
        d_qk[12] = exp2_12_3;
        d_qk[13] = exp2_13_3;
        row_sum0 += exp2_12_3 + exp2_13_3;
        float exp2_14_3 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[5], bias1_3));
        float exp2_15_3 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[5], bias1_3));
        d_qk[14] = exp2_14_3;
        d_qk[15] = exp2_15_3;
        row_sum1 += exp2_14_3 + exp2_15_3;
        uint16_t _cvt_satfinite_e4m3x2_f32_166;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_166) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_167;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_167) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_165;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_165) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_166), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_167));
        p_e4m3[0] = _prmt_165;
        uint16_t _cvt_satfinite_e4m3x2_f32_169;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_169) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_170;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_170) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_168;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_168) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_169), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_170));
        p_e4m3[1] = _prmt_168;
        uint16_t _cvt_satfinite_e4m3x2_f32_172;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_172) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_173;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_173) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_171;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_171) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_172), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_173));
        p_e4m3[2] = _prmt_171;
        uint16_t _cvt_satfinite_e4m3x2_f32_175;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_175) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_176;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_176) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_174;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_174) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_175), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_176));
        p_e4m3[3] = _prmt_174;
        mbarrier_wait(v_full3_addr, 0);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
            : "r"(vl_smem_addr + (unsigned int)(49152 + t_kq4096 + ld_row + ((t_rr ^ t_dc0) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
            : "r"(vl_smem_addr + (unsigned int)(49152 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 1) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
            : "r"(vl_smem_addr + (unsigned int)(49152 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 2) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
            : "r"(vl_smem_addr + (unsigned int)(49152 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 3) << 4)))
            : "memory");
        uint32_t _prmt_177;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_177) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_177;
        uint32_t _prmt_178;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_178) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_178;
        uint32_t _prmt_179;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_179) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_179;
        uint32_t _prmt_180;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_180) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_180;
        uint32_t _stmatrix_addr_181 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + t_dc0 * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_181), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_182;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_182) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_182;
        uint32_t _prmt_183;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_183) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_183;
        uint32_t _prmt_184;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_184) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_184;
        uint32_t _prmt_185;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_185) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_185;
        uint32_t _stmatrix_addr_186 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 1) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_186), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_187;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_187) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_187;
        uint32_t _prmt_188;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_188) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_188;
        uint32_t _prmt_189;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_189) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_189;
        uint32_t _prmt_190;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_190) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_190;
        uint32_t _stmatrix_addr_191 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 2) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_191), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_192;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_192) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_192;
        uint32_t _prmt_193;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_193) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_193;
        uint32_t _prmt_194;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_194) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_194;
        uint32_t _prmt_195;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_195) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_195;
        uint32_t _stmatrix_addr_196 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 3) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_196), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 3, 128;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(vtd_0 + 4)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_18_1 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0_3));
        float exp2_1_3_1 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0_3));
        d_qk[16] = exp2_18_1;
        d_qk[17] = exp2_1_3_1;
        row_sum0 += exp2_18_1 + exp2_1_3_1;
        float exp2_2_3_1 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1_3));
        float exp2_3_3_1 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1_3));
        d_qk[18] = exp2_2_3_1;
        d_qk[19] = exp2_3_3_1;
        row_sum1 += exp2_2_3_1 + exp2_3_3_1;
        float exp2_4_3_1 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0_3));
        float exp2_5_3_1 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0_3));
        d_qk[20] = exp2_4_3_1;
        d_qk[21] = exp2_5_3_1;
        row_sum0 += exp2_4_3_1 + exp2_5_3_1;
        float exp2_6_3_1 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1_3));
        float exp2_7_3_1 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1_3));
        d_qk[22] = exp2_6_3_1;
        d_qk[23] = exp2_7_3_1;
        row_sum1 += exp2_6_3_1 + exp2_7_3_1;
        float exp2_8_3_1 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0_3));
        float exp2_9_3_1 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0_3));
        d_qk[24] = exp2_8_3_1;
        d_qk[25] = exp2_9_3_1;
        row_sum0 += exp2_8_3_1 + exp2_9_3_1;
        float exp2_10_3_1 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1_3));
        float exp2_11_3_1 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1_3));
        d_qk[26] = exp2_10_3_1;
        d_qk[27] = exp2_11_3_1;
        row_sum1 += exp2_10_3_1 + exp2_11_3_1;
        float exp2_12_3_1 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0_3));
        float exp2_13_3_1 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0_3));
        d_qk[28] = exp2_12_3_1;
        d_qk[29] = exp2_13_3_1;
        row_sum0 += exp2_12_3_1 + exp2_13_3_1;
        float exp2_14_3_1 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1_3));
        float exp2_15_3_1 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1_3));
        d_qk[30] = exp2_14_3_1;
        d_qk[31] = exp2_15_3_1;
        row_sum1 += exp2_14_3_1 + exp2_15_3_1;
        uint16_t _cvt_satfinite_e4m3x2_f32_198;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_198) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_199;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_199) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_197;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_197) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_198), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_199));
        p_e4m3[4] = _prmt_197;
        uint16_t _cvt_satfinite_e4m3x2_f32_201;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_201) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_202;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_202) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_200;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_200) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_201), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_202));
        p_e4m3[5] = _prmt_200;
        uint16_t _cvt_satfinite_e4m3x2_f32_204;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_204) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_205;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_205) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_203;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_203) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_204), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_205));
        p_e4m3[6] = _prmt_203;
        uint16_t _cvt_satfinite_e4m3x2_f32_207;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_207) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_208;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_208) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_206;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_206) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_207), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_208));
        p_e4m3[7] = _prmt_206;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(vtd_0 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_209 = (((uint64_t)(((k_smem_addr + 65536)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_209 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_209);
    uint64_t _wgmma_desc_210 = (((uint64_t)(((vt_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t vtd_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_210 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_210);
    if (lim_even > 4) {
        mbarrier_wait(k_full4_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(kd_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(kd_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(kd_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(kd_3 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        qk_sc[0] = q_log2[0] * ks_regs[8];
        qk_sc[4] = q_log2[1] * ks_regs[8];
        qk_sc[1] = q_log2[0] * ks_regs[9];
        qk_sc[5] = q_log2[1] * ks_regs[9];
        qk_sc[2] = q_log2[0] * ks_regs[10];
        qk_sc[6] = q_log2[1] * ks_regs[10];
        qk_sc[3] = q_log2[0] * ks_regs[11];
        qk_sc[7] = q_log2[1] * ks_regs[11];
        float new_max0_4 = -CUDART_INF_F;
        float new_max1_4 = -CUDART_INF_F;
        float m0_4 = -CUDART_INF_F;
        float m1_4 = -CUDART_INF_F;
        if (scale_log2 >= 0.0f) {
            m0_4 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])) * qk_sc[0], fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])) * qk_sc[2], fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_4 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])) * qk_sc[4], fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])) * qk_sc[6], fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31])) * qk_sc[7]));
        } else {
            m0_4 = fmaxf(fmaxf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])) * qk_sc[0], fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])) * qk_sc[2], fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_4 = fmaxf(fmaxf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])) * qk_sc[4], fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])) * qk_sc[6], fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31])) * qk_sc[7]));
        }
        float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, m0_4, 2);
        m0_4 = fmaxf(m0_4, _shfl_xor_sync_211);
        float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, m0_4, 1);
        m0_4 = fmaxf(m0_4, _shfl_xor_sync_212);
        float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, m1_4, 2);
        m1_4 = fmaxf(m1_4, _shfl_xor_sync_213);
        float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, m1_4, 1);
        m1_4 = fmaxf(m1_4, _shfl_xor_sync_214);
        new_max0_4 = m0_4;
        new_max1_4 = m1_4;
        float merged_max0_2 = fmaxf(row_max0, new_max0_4);
        float merged_max1_2 = fmaxf(row_max1, new_max1_4);
        int grow0_2 = merged_max0_2 - row_max0 > 0.8073549f;
        int grow1_2 = merged_max1_2 - row_max1 > 0.8073549f;
        float _exp2_215 = approx_exp2(row_max0 - merged_max0_2);
        float rescale0_2 = ((grow0_2 != 0) ? _exp2_215 : 1.0f);
        float _exp2_216 = approx_exp2(row_max1 - merged_max1_2);
        float rescale1_2 = ((grow1_2 != 0) ? _exp2_216 : 1.0f);
        row_max0 = ((grow0_2 != 0) ? merged_max0_2 : row_max0);
        row_max1 = ((grow1_2 != 0) ? merged_max1_2 : row_max1);
        d_o[0] = d_o[0] * rescale0_2;
        d_o[1] = d_o[1] * rescale0_2;
        d_o[4] = d_o[4] * rescale0_2;
        d_o[5] = d_o[5] * rescale0_2;
        d_o[8] = d_o[8] * rescale0_2;
        d_o[9] = d_o[9] * rescale0_2;
        d_o[12] = d_o[12] * rescale0_2;
        d_o[13] = d_o[13] * rescale0_2;
        d_o[16] = d_o[16] * rescale0_2;
        d_o[17] = d_o[17] * rescale0_2;
        d_o[20] = d_o[20] * rescale0_2;
        d_o[21] = d_o[21] * rescale0_2;
        d_o[24] = d_o[24] * rescale0_2;
        d_o[25] = d_o[25] * rescale0_2;
        d_o[28] = d_o[28] * rescale0_2;
        d_o[29] = d_o[29] * rescale0_2;
        d_o[32] = d_o[32] * rescale0_2;
        d_o[33] = d_o[33] * rescale0_2;
        d_o[36] = d_o[36] * rescale0_2;
        d_o[37] = d_o[37] * rescale0_2;
        d_o[40] = d_o[40] * rescale0_2;
        d_o[41] = d_o[41] * rescale0_2;
        d_o[44] = d_o[44] * rescale0_2;
        d_o[45] = d_o[45] * rescale0_2;
        d_o[48] = d_o[48] * rescale0_2;
        d_o[49] = d_o[49] * rescale0_2;
        d_o[52] = d_o[52] * rescale0_2;
        d_o[53] = d_o[53] * rescale0_2;
        d_o[56] = d_o[56] * rescale0_2;
        d_o[57] = d_o[57] * rescale0_2;
        d_o[60] = d_o[60] * rescale0_2;
        d_o[61] = d_o[61] * rescale0_2;
        d_o[2] = d_o[2] * rescale1_2;
        d_o[3] = d_o[3] * rescale1_2;
        d_o[6] = d_o[6] * rescale1_2;
        d_o[7] = d_o[7] * rescale1_2;
        d_o[10] = d_o[10] * rescale1_2;
        d_o[11] = d_o[11] * rescale1_2;
        d_o[14] = d_o[14] * rescale1_2;
        d_o[15] = d_o[15] * rescale1_2;
        d_o[18] = d_o[18] * rescale1_2;
        d_o[19] = d_o[19] * rescale1_2;
        d_o[22] = d_o[22] * rescale1_2;
        d_o[23] = d_o[23] * rescale1_2;
        d_o[26] = d_o[26] * rescale1_2;
        d_o[27] = d_o[27] * rescale1_2;
        d_o[30] = d_o[30] * rescale1_2;
        d_o[31] = d_o[31] * rescale1_2;
        d_o[34] = d_o[34] * rescale1_2;
        d_o[35] = d_o[35] * rescale1_2;
        d_o[38] = d_o[38] * rescale1_2;
        d_o[39] = d_o[39] * rescale1_2;
        d_o[42] = d_o[42] * rescale1_2;
        d_o[43] = d_o[43] * rescale1_2;
        d_o[46] = d_o[46] * rescale1_2;
        d_o[47] = d_o[47] * rescale1_2;
        d_o[50] = d_o[50] * rescale1_2;
        d_o[51] = d_o[51] * rescale1_2;
        d_o[54] = d_o[54] * rescale1_2;
        d_o[55] = d_o[55] * rescale1_2;
        d_o[58] = d_o[58] * rescale1_2;
        d_o[59] = d_o[59] * rescale1_2;
        d_o[62] = d_o[62] * rescale1_2;
        d_o[63] = d_o[63] * rescale1_2;
        row_sum0 = row_sum0 * rescale0_2;
        row_sum1 = row_sum1 * rescale1_2;
        float bias0_4 = 8.0f - row_max0;
        float bias1_4 = 8.0f - row_max1;
        float exp2_19 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0_4));
        float exp2_1_4 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0_4));
        d_qk[0] = exp2_19;
        d_qk[1] = exp2_1_4;
        row_sum0 += exp2_19 + exp2_1_4;
        float exp2_2_4 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[4], bias1_4));
        float exp2_3_4 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[4], bias1_4));
        d_qk[2] = exp2_2_4;
        d_qk[3] = exp2_3_4;
        row_sum1 += exp2_2_4 + exp2_3_4;
        float exp2_4_4 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0_4));
        float exp2_5_4 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0_4));
        d_qk[4] = exp2_4_4;
        d_qk[5] = exp2_5_4;
        row_sum0 += exp2_4_4 + exp2_5_4;
        float exp2_6_4 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[4], bias1_4));
        float exp2_7_4 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[4], bias1_4));
        d_qk[6] = exp2_6_4;
        d_qk[7] = exp2_7_4;
        row_sum1 += exp2_6_4 + exp2_7_4;
        float exp2_8_4 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0_4));
        float exp2_9_4 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0_4));
        d_qk[8] = exp2_8_4;
        d_qk[9] = exp2_9_4;
        row_sum0 += exp2_8_4 + exp2_9_4;
        float exp2_10_4 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[5], bias1_4));
        float exp2_11_4 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[5], bias1_4));
        d_qk[10] = exp2_10_4;
        d_qk[11] = exp2_11_4;
        row_sum1 += exp2_10_4 + exp2_11_4;
        float exp2_12_4 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0_4));
        float exp2_13_4 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0_4));
        d_qk[12] = exp2_12_4;
        d_qk[13] = exp2_13_4;
        row_sum0 += exp2_12_4 + exp2_13_4;
        float exp2_14_4 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[5], bias1_4));
        float exp2_15_4 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[5], bias1_4));
        d_qk[14] = exp2_14_4;
        d_qk[15] = exp2_15_4;
        row_sum1 += exp2_14_4 + exp2_15_4;
        uint16_t _cvt_satfinite_e4m3x2_f32_218;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_218) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_219;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_219) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_217;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_217) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_218), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_219));
        p_e4m3[0] = _prmt_217;
        uint16_t _cvt_satfinite_e4m3x2_f32_221;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_221) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_222;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_222) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_220;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_220) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_221), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_222));
        p_e4m3[1] = _prmt_220;
        uint16_t _cvt_satfinite_e4m3x2_f32_224;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_224) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_225;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_225) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_223;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_223) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_224), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_225));
        p_e4m3[2] = _prmt_223;
        uint16_t _cvt_satfinite_e4m3x2_f32_227;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_227) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_228;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_228) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_226;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_226) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_227), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_228));
        p_e4m3[3] = _prmt_226;
        mbarrier_wait(v_full4_addr, 0);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
            : "r"(vl_smem_addr + (unsigned int)(65536 + t_kq4096 + ld_row + ((t_rr ^ t_dc0) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
            : "r"(vl_smem_addr + (unsigned int)(65536 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 1) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
            : "r"(vl_smem_addr + (unsigned int)(65536 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 2) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
            : "r"(vl_smem_addr + (unsigned int)(65536 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 3) << 4)))
            : "memory");
        uint32_t _prmt_229;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_229) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_229;
        uint32_t _prmt_230;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_230) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_230;
        uint32_t _prmt_231;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_231) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_231;
        uint32_t _prmt_232;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_232) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_232;
        uint32_t _stmatrix_addr_233 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_233), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_234;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_234) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_234;
        uint32_t _prmt_235;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_235) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_235;
        uint32_t _prmt_236;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_236) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_236;
        uint32_t _prmt_237;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_237) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_237;
        uint32_t _stmatrix_addr_238 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + (t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_238), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_239;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_239) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_239;
        uint32_t _prmt_240;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_240) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_240;
        uint32_t _prmt_241;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_241) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_241;
        uint32_t _prmt_242;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_242) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_242;
        uint32_t _stmatrix_addr_243 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + (t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_243), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_244;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_244) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_244;
        uint32_t _prmt_245;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_245) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_245;
        uint32_t _prmt_246;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_246) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_246;
        uint32_t _prmt_247;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_247) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_247;
        uint32_t _stmatrix_addr_248 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + (t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_248), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 2, 128;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(vtd_1)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_19_1 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0_4));
        float exp2_1_4_1 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0_4));
        d_qk[16] = exp2_19_1;
        d_qk[17] = exp2_1_4_1;
        row_sum0 += exp2_19_1 + exp2_1_4_1;
        float exp2_2_4_1 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1_4));
        float exp2_3_4_1 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1_4));
        d_qk[18] = exp2_2_4_1;
        d_qk[19] = exp2_3_4_1;
        row_sum1 += exp2_2_4_1 + exp2_3_4_1;
        float exp2_4_4_1 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0_4));
        float exp2_5_4_1 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0_4));
        d_qk[20] = exp2_4_4_1;
        d_qk[21] = exp2_5_4_1;
        row_sum0 += exp2_4_4_1 + exp2_5_4_1;
        float exp2_6_4_1 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1_4));
        float exp2_7_4_1 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1_4));
        d_qk[22] = exp2_6_4_1;
        d_qk[23] = exp2_7_4_1;
        row_sum1 += exp2_6_4_1 + exp2_7_4_1;
        float exp2_8_4_1 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0_4));
        float exp2_9_4_1 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0_4));
        d_qk[24] = exp2_8_4_1;
        d_qk[25] = exp2_9_4_1;
        row_sum0 += exp2_8_4_1 + exp2_9_4_1;
        float exp2_10_4_1 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1_4));
        float exp2_11_4_1 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1_4));
        d_qk[26] = exp2_10_4_1;
        d_qk[27] = exp2_11_4_1;
        row_sum1 += exp2_10_4_1 + exp2_11_4_1;
        float exp2_12_4_1 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0_4));
        float exp2_13_4_1 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0_4));
        d_qk[28] = exp2_12_4_1;
        d_qk[29] = exp2_13_4_1;
        row_sum0 += exp2_12_4_1 + exp2_13_4_1;
        float exp2_14_4_1 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1_4));
        float exp2_15_4_1 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1_4));
        d_qk[30] = exp2_14_4_1;
        d_qk[31] = exp2_15_4_1;
        row_sum1 += exp2_14_4_1 + exp2_15_4_1;
        uint16_t _cvt_satfinite_e4m3x2_f32_250;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_250) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_251;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_251) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_249;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_249) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_250), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_251));
        p_e4m3[4] = _prmt_249;
        uint16_t _cvt_satfinite_e4m3x2_f32_253;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_253) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_254;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_254) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_252;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_252) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_253), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_254));
        p_e4m3[5] = _prmt_252;
        uint16_t _cvt_satfinite_e4m3x2_f32_256;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_256) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_257;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_257) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_255;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_255) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_256), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_257));
        p_e4m3[6] = _prmt_255;
        uint16_t _cvt_satfinite_e4m3x2_f32_259;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_259) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_260;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_260) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_258;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_258) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_259), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_260));
        p_e4m3[7] = _prmt_258;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(vtd_1 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_261 = (((uint64_t)(((k_smem_addr + 81920)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_261 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_261);
    if (lim_odd > 5) {
        mbarrier_wait(k_full5_addr, 0);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0), "l"(kd_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 2), "l"(kd_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 4), "l"(kd_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31])
            : "l"(_wgmma_a_0 + 6), "l"(kd_4 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        qk_sc[0] = q_log2[0] * ks_regs[8];
        qk_sc[4] = q_log2[1] * ks_regs[8];
        qk_sc[1] = q_log2[0] * ks_regs[9];
        qk_sc[5] = q_log2[1] * ks_regs[9];
        qk_sc[2] = q_log2[0] * ks_regs[10];
        qk_sc[6] = q_log2[1] * ks_regs[10];
        qk_sc[3] = q_log2[0] * ks_regs[11];
        qk_sc[7] = q_log2[1] * ks_regs[11];
        float new_max0_5 = -CUDART_INF_F;
        float new_max1_5 = -CUDART_INF_F;
        float m0_5 = -CUDART_INF_F;
        float m1_5 = -CUDART_INF_F;
        if (scale_log2 >= 0.0f) {
            m0_5 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])) * qk_sc[0], fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])) * qk_sc[2], fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_5 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])) * qk_sc[4], fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])) * qk_sc[6], fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31])) * qk_sc[7]));
        } else {
            m0_5 = fmaxf(fmaxf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])) * qk_sc[0], fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13])) * qk_sc[1]), fmaxf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])) * qk_sc[2], fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29])) * qk_sc[3]));
            m1_5 = fmaxf(fmaxf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])) * qk_sc[4], fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15])) * qk_sc[5]), fmaxf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])) * qk_sc[6], fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31])) * qk_sc[7]));
        }
        float _shfl_xor_sync_262 = __shfl_xor_sync(0xFFFFFFFF, m0_5, 2);
        m0_5 = fmaxf(m0_5, _shfl_xor_sync_262);
        float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, m0_5, 1);
        m0_5 = fmaxf(m0_5, _shfl_xor_sync_263);
        float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, m1_5, 2);
        m1_5 = fmaxf(m1_5, _shfl_xor_sync_264);
        float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, m1_5, 1);
        m1_5 = fmaxf(m1_5, _shfl_xor_sync_265);
        new_max0_5 = m0_5;
        new_max1_5 = m1_5;
        float merged_max0_3 = fmaxf(row_max0, new_max0_5);
        float merged_max1_3 = fmaxf(row_max1, new_max1_5);
        int grow0_3 = merged_max0_3 - row_max0 > 0.8073549f;
        int grow1_3 = merged_max1_3 - row_max1 > 0.8073549f;
        float _exp2_266 = approx_exp2(row_max0 - merged_max0_3);
        float rescale0_3 = ((grow0_3 != 0) ? _exp2_266 : 1.0f);
        float _exp2_267 = approx_exp2(row_max1 - merged_max1_3);
        float rescale1_3 = ((grow1_3 != 0) ? _exp2_267 : 1.0f);
        row_max0 = ((grow0_3 != 0) ? merged_max0_3 : row_max0);
        row_max1 = ((grow1_3 != 0) ? merged_max1_3 : row_max1);
        d_o[0] = d_o[0] * rescale0_3;
        d_o[1] = d_o[1] * rescale0_3;
        d_o[4] = d_o[4] * rescale0_3;
        d_o[5] = d_o[5] * rescale0_3;
        d_o[8] = d_o[8] * rescale0_3;
        d_o[9] = d_o[9] * rescale0_3;
        d_o[12] = d_o[12] * rescale0_3;
        d_o[13] = d_o[13] * rescale0_3;
        d_o[16] = d_o[16] * rescale0_3;
        d_o[17] = d_o[17] * rescale0_3;
        d_o[20] = d_o[20] * rescale0_3;
        d_o[21] = d_o[21] * rescale0_3;
        d_o[24] = d_o[24] * rescale0_3;
        d_o[25] = d_o[25] * rescale0_3;
        d_o[28] = d_o[28] * rescale0_3;
        d_o[29] = d_o[29] * rescale0_3;
        d_o[32] = d_o[32] * rescale0_3;
        d_o[33] = d_o[33] * rescale0_3;
        d_o[36] = d_o[36] * rescale0_3;
        d_o[37] = d_o[37] * rescale0_3;
        d_o[40] = d_o[40] * rescale0_3;
        d_o[41] = d_o[41] * rescale0_3;
        d_o[44] = d_o[44] * rescale0_3;
        d_o[45] = d_o[45] * rescale0_3;
        d_o[48] = d_o[48] * rescale0_3;
        d_o[49] = d_o[49] * rescale0_3;
        d_o[52] = d_o[52] * rescale0_3;
        d_o[53] = d_o[53] * rescale0_3;
        d_o[56] = d_o[56] * rescale0_3;
        d_o[57] = d_o[57] * rescale0_3;
        d_o[60] = d_o[60] * rescale0_3;
        d_o[61] = d_o[61] * rescale0_3;
        d_o[2] = d_o[2] * rescale1_3;
        d_o[3] = d_o[3] * rescale1_3;
        d_o[6] = d_o[6] * rescale1_3;
        d_o[7] = d_o[7] * rescale1_3;
        d_o[10] = d_o[10] * rescale1_3;
        d_o[11] = d_o[11] * rescale1_3;
        d_o[14] = d_o[14] * rescale1_3;
        d_o[15] = d_o[15] * rescale1_3;
        d_o[18] = d_o[18] * rescale1_3;
        d_o[19] = d_o[19] * rescale1_3;
        d_o[22] = d_o[22] * rescale1_3;
        d_o[23] = d_o[23] * rescale1_3;
        d_o[26] = d_o[26] * rescale1_3;
        d_o[27] = d_o[27] * rescale1_3;
        d_o[30] = d_o[30] * rescale1_3;
        d_o[31] = d_o[31] * rescale1_3;
        d_o[34] = d_o[34] * rescale1_3;
        d_o[35] = d_o[35] * rescale1_3;
        d_o[38] = d_o[38] * rescale1_3;
        d_o[39] = d_o[39] * rescale1_3;
        d_o[42] = d_o[42] * rescale1_3;
        d_o[43] = d_o[43] * rescale1_3;
        d_o[46] = d_o[46] * rescale1_3;
        d_o[47] = d_o[47] * rescale1_3;
        d_o[50] = d_o[50] * rescale1_3;
        d_o[51] = d_o[51] * rescale1_3;
        d_o[54] = d_o[54] * rescale1_3;
        d_o[55] = d_o[55] * rescale1_3;
        d_o[58] = d_o[58] * rescale1_3;
        d_o[59] = d_o[59] * rescale1_3;
        d_o[62] = d_o[62] * rescale1_3;
        d_o[63] = d_o[63] * rescale1_3;
        row_sum0 = row_sum0 * rescale0_3;
        row_sum1 = row_sum1 * rescale1_3;
        float bias0_5 = 8.0f - row_max0;
        float bias1_5 = 8.0f - row_max1;
        float exp2_20 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0_5));
        float exp2_1_5 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0_5));
        d_qk[0] = exp2_20;
        d_qk[1] = exp2_1_5;
        row_sum0 += exp2_20 + exp2_1_5;
        float exp2_2_5 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[4], bias1_5));
        float exp2_3_5 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[4], bias1_5));
        d_qk[2] = exp2_2_5;
        d_qk[3] = exp2_3_5;
        row_sum1 += exp2_2_5 + exp2_3_5;
        float exp2_4_5 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0_5));
        float exp2_5_5 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0_5));
        d_qk[4] = exp2_4_5;
        d_qk[5] = exp2_5_5;
        row_sum0 += exp2_4_5 + exp2_5_5;
        float exp2_6_5 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[4], bias1_5));
        float exp2_7_5 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[4], bias1_5));
        d_qk[6] = exp2_6_5;
        d_qk[7] = exp2_7_5;
        row_sum1 += exp2_6_5 + exp2_7_5;
        float exp2_8_5 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0_5));
        float exp2_9_5 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0_5));
        d_qk[8] = exp2_8_5;
        d_qk[9] = exp2_9_5;
        row_sum0 += exp2_8_5 + exp2_9_5;
        float exp2_10_5 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[5], bias1_5));
        float exp2_11_5 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[5], bias1_5));
        d_qk[10] = exp2_10_5;
        d_qk[11] = exp2_11_5;
        row_sum1 += exp2_10_5 + exp2_11_5;
        float exp2_12_5 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0_5));
        float exp2_13_5 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0_5));
        d_qk[12] = exp2_12_5;
        d_qk[13] = exp2_13_5;
        row_sum0 += exp2_12_5 + exp2_13_5;
        float exp2_14_5 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[5], bias1_5));
        float exp2_15_5 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[5], bias1_5));
        d_qk[14] = exp2_14_5;
        d_qk[15] = exp2_15_5;
        row_sum1 += exp2_14_5 + exp2_15_5;
        uint16_t _cvt_satfinite_e4m3x2_f32_269;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_269) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_270;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_270) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_268;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_268) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_269), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_270));
        p_e4m3[0] = _prmt_268;
        uint16_t _cvt_satfinite_e4m3x2_f32_272;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_272) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_273;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_273) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_271;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_271) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_272), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_273));
        p_e4m3[1] = _prmt_271;
        uint16_t _cvt_satfinite_e4m3x2_f32_275;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_275) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_276;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_276) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_274;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_274) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_275), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_276));
        p_e4m3[2] = _prmt_274;
        uint16_t _cvt_satfinite_e4m3x2_f32_278;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_278) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_279;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_279) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_277;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_277) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_278), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_279));
        p_e4m3[3] = _prmt_277;
        mbarrier_wait(v_full5_addr, 0);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
            : "r"(vl_smem_addr + (unsigned int)(81920 + t_kq4096 + ld_row + ((t_rr ^ t_dc0) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
            : "r"(vl_smem_addr + (unsigned int)(81920 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 1) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
            : "r"(vl_smem_addr + (unsigned int)(81920 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 2) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
            : "r"(vl_smem_addr + (unsigned int)(81920 + t_kq4096 + ld_row + ((t_rr ^ t_dc0 + 3) << 4)))
            : "memory");
        uint32_t _prmt_280;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_280) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_280;
        uint32_t _prmt_281;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_281) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_281;
        uint32_t _prmt_282;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_282) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_282;
        uint32_t _prmt_283;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_283) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_283;
        uint32_t _stmatrix_addr_284 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + t_dc0 * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_284), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_285;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_285) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_285;
        uint32_t _prmt_286;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_286) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_286;
        uint32_t _prmt_287;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_287) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_287;
        uint32_t _prmt_288;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_288) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_288;
        uint32_t _stmatrix_addr_289 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + (t_dc0 + 1) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_289), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_290;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_290) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_290;
        uint32_t _prmt_291;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_291) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_291;
        uint32_t _prmt_292;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_292) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_292;
        uint32_t _prmt_293;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_293) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_293;
        uint32_t _stmatrix_addr_294 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + (t_dc0 + 2) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_294), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_295;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_295) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_295;
        uint32_t _prmt_296;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_296) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_296;
        uint32_t _prmt_297;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_297) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_297;
        uint32_t _prmt_298;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_298) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_298;
        uint32_t _stmatrix_addr_299 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(32768 + (t_dc0 + 3) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_299), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 3, 128;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(vtd_1 + 4)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        float exp2_20_1 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0_5));
        float exp2_1_5_1 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0_5));
        d_qk[16] = exp2_20_1;
        d_qk[17] = exp2_1_5_1;
        row_sum0 += exp2_20_1 + exp2_1_5_1;
        float exp2_2_5_1 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1_5));
        float exp2_3_5_1 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1_5));
        d_qk[18] = exp2_2_5_1;
        d_qk[19] = exp2_3_5_1;
        row_sum1 += exp2_2_5_1 + exp2_3_5_1;
        float exp2_4_5_1 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0_5));
        float exp2_5_5_1 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0_5));
        d_qk[20] = exp2_4_5_1;
        d_qk[21] = exp2_5_5_1;
        row_sum0 += exp2_4_5_1 + exp2_5_5_1;
        float exp2_6_5_1 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1_5));
        float exp2_7_5_1 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1_5));
        d_qk[22] = exp2_6_5_1;
        d_qk[23] = exp2_7_5_1;
        row_sum1 += exp2_6_5_1 + exp2_7_5_1;
        float exp2_8_5_1 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0_5));
        float exp2_9_5_1 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0_5));
        d_qk[24] = exp2_8_5_1;
        d_qk[25] = exp2_9_5_1;
        row_sum0 += exp2_8_5_1 + exp2_9_5_1;
        float exp2_10_5_1 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1_5));
        float exp2_11_5_1 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1_5));
        d_qk[26] = exp2_10_5_1;
        d_qk[27] = exp2_11_5_1;
        row_sum1 += exp2_10_5_1 + exp2_11_5_1;
        float exp2_12_5_1 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0_5));
        float exp2_13_5_1 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0_5));
        d_qk[28] = exp2_12_5_1;
        d_qk[29] = exp2_13_5_1;
        row_sum0 += exp2_12_5_1 + exp2_13_5_1;
        float exp2_14_5_1 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1_5));
        float exp2_15_5_1 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1_5));
        d_qk[30] = exp2_14_5_1;
        d_qk[31] = exp2_15_5_1;
        row_sum1 += exp2_14_5_1 + exp2_15_5_1;
        uint16_t _cvt_satfinite_e4m3x2_f32_301;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_301) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_302;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_302) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_300;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_300) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_301), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_302));
        p_e4m3[4] = _prmt_300;
        uint16_t _cvt_satfinite_e4m3x2_f32_304;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_304) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_305;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_305) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_303;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_303) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_304), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_305));
        p_e4m3[5] = _prmt_303;
        uint16_t _cvt_satfinite_e4m3x2_f32_307;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_307) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_308;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_308) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_306;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_306) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_307), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_308));
        p_e4m3[6] = _prmt_306;
        uint16_t _cvt_satfinite_e4m3x2_f32_310;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_310) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_311;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_311) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_309;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_309) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_310), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_311));
        p_e4m3[7] = _prmt_309;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(vtd_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    float _shfl_xor_sync_312 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_sync_312;
    float _shfl_xor_sync_313 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_sync_313;
    float _shfl_xor_sync_314 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_sync_314;
    float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_sync_315;
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
        if (qj == 0) {
            mstats[quad * 4] = row_max0;
            mstats[quad * 4 + 1] = row_max1;
            mstats[quad * 4 + 2] = row_sum0;
            mstats[quad * 4 + 3] = row_sum1;
        }
    }
    if (wg == 0) {
        part_hi0[tid_wg] = d_o[32];
        part_hi0[128 + tid_wg] = d_o[33];
        part_hi0[256 + tid_wg] = d_o[34];
        part_hi0[384 + tid_wg] = d_o[35];
        part_hi0[512 + tid_wg] = d_o[36];
        part_hi0[640 + tid_wg] = d_o[37];
        part_hi0[768 + tid_wg] = d_o[38];
        part_hi0[896 + tid_wg] = d_o[39];
        part_hi0[1024 + tid_wg] = d_o[40];
        part_hi0[1152 + tid_wg] = d_o[41];
        part_hi0[1280 + tid_wg] = d_o[42];
        part_hi0[1408 + tid_wg] = d_o[43];
        part_hi0[1536 + tid_wg] = d_o[44];
        part_hi0[1664 + tid_wg] = d_o[45];
        part_hi0[1792 + tid_wg] = d_o[46];
        part_hi0[1920 + tid_wg] = d_o[47];
        part_hi0[2048 + tid_wg] = d_o[48];
        part_hi0[2176 + tid_wg] = d_o[49];
        part_hi0[2304 + tid_wg] = d_o[50];
        part_hi0[2432 + tid_wg] = d_o[51];
        part_hi0[2560 + tid_wg] = d_o[52];
        part_hi0[2688 + tid_wg] = d_o[53];
        part_hi0[2816 + tid_wg] = d_o[54];
        part_hi0[2944 + tid_wg] = d_o[55];
        part_hi0[3072 + tid_wg] = d_o[56];
        part_hi0[3200 + tid_wg] = d_o[57];
        part_hi0[3328 + tid_wg] = d_o[58];
        part_hi0[3456 + tid_wg] = d_o[59];
        part_hi0[3584 + tid_wg] = d_o[60];
        part_hi0[3712 + tid_wg] = d_o[61];
        part_hi0[3840 + tid_wg] = d_o[62];
        part_hi0[3968 + tid_wg] = d_o[63];
        if (qj == 0) {
            mstats[128 + quad * 4] = row_max0;
            mstats[128 + quad * 4 + 1] = row_max1;
            mstats[128 + quad * 4 + 2] = row_sum0;
            mstats[128 + quad * 4 + 3] = row_sum1;
        }
    }
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    int pbase = ((wg == 0) ? 0 : 128);
    float pm0 = mstats[pbase + quad * 4];
    float pm1 = mstats[pbase + quad * 4 + 1];
    float pl0 = mstats[pbase + quad * 4 + 2];
    float pl1 = mstats[pbase + quad * 4 + 3];
    float x0m0 = ((wg == 0) ? row_max0 : pm0);
    float x0m1 = ((wg == 0) ? row_max1 : pm1);
    float x1m0 = ((wg == 0) ? pm0 : row_max0);
    float x1m1 = ((wg == 0) ? pm1 : row_max1);
    float x0l0 = ((wg == 0) ? row_sum0 : pl0);
    float x0l1 = ((wg == 0) ? row_sum1 : pl1);
    float x1l0 = ((wg == 0) ? pl0 : row_sum0);
    float x1l1 = ((wg == 0) ? pl1 : row_sum1);
    float mm0 = fmaxf(x0m0, x1m0);
    float mm1 = fmaxf(x0m1, x1m1);
    float _exp2_316 = approx_exp2(x0m0 - mm0);
    float f00 = ((x0m0 == -CUDART_INF_F) ? 0.0f : _exp2_316);
    float _exp2_317 = approx_exp2(x0m1 - mm1);
    float f01 = ((x0m1 == -CUDART_INF_F) ? 0.0f : _exp2_317);
    float _exp2_318 = approx_exp2(x1m0 - mm0);
    float f10 = ((x1m0 == -CUDART_INF_F) ? 0.0f : _exp2_318);
    float _exp2_319 = approx_exp2(x1m1 - mm1);
    float f11 = ((x1m1 == -CUDART_INF_F) ? 0.0f : _exp2_319);
    float a0 = ((wg == 0) ? f00 : f10);
    float a1 = ((wg == 0) ? f01 : f11);
    float b0 = ((wg == 0) ? f10 : f00);
    float b1 = ((wg == 0) ? f11 : f01);
    if (wg == 0) {
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
    }
    if (wg == 1) {
        d_o[32] = __fmaf_rn(part_hi0[tid_wg], b0, d_o[32] * a0);
        d_o[33] = __fmaf_rn(part_hi0[128 + tid_wg], b0, d_o[33] * a0);
        d_o[34] = __fmaf_rn(part_hi0[256 + tid_wg], b1, d_o[34] * a1);
        d_o[35] = __fmaf_rn(part_hi0[384 + tid_wg], b1, d_o[35] * a1);
        d_o[36] = __fmaf_rn(part_hi0[512 + tid_wg], b0, d_o[36] * a0);
        d_o[37] = __fmaf_rn(part_hi0[640 + tid_wg], b0, d_o[37] * a0);
        d_o[38] = __fmaf_rn(part_hi0[768 + tid_wg], b1, d_o[38] * a1);
        d_o[39] = __fmaf_rn(part_hi0[896 + tid_wg], b1, d_o[39] * a1);
        d_o[40] = __fmaf_rn(part_hi0[1024 + tid_wg], b0, d_o[40] * a0);
        d_o[41] = __fmaf_rn(part_hi0[1152 + tid_wg], b0, d_o[41] * a0);
        d_o[42] = __fmaf_rn(part_hi0[1280 + tid_wg], b1, d_o[42] * a1);
        d_o[43] = __fmaf_rn(part_hi0[1408 + tid_wg], b1, d_o[43] * a1);
        d_o[44] = __fmaf_rn(part_hi0[1536 + tid_wg], b0, d_o[44] * a0);
        d_o[45] = __fmaf_rn(part_hi0[1664 + tid_wg], b0, d_o[45] * a0);
        d_o[46] = __fmaf_rn(part_hi0[1792 + tid_wg], b1, d_o[46] * a1);
        d_o[47] = __fmaf_rn(part_hi0[1920 + tid_wg], b1, d_o[47] * a1);
        d_o[48] = __fmaf_rn(part_hi0[2048 + tid_wg], b0, d_o[48] * a0);
        d_o[49] = __fmaf_rn(part_hi0[2176 + tid_wg], b0, d_o[49] * a0);
        d_o[50] = __fmaf_rn(part_hi0[2304 + tid_wg], b1, d_o[50] * a1);
        d_o[51] = __fmaf_rn(part_hi0[2432 + tid_wg], b1, d_o[51] * a1);
        d_o[52] = __fmaf_rn(part_hi0[2560 + tid_wg], b0, d_o[52] * a0);
        d_o[53] = __fmaf_rn(part_hi0[2688 + tid_wg], b0, d_o[53] * a0);
        d_o[54] = __fmaf_rn(part_hi0[2816 + tid_wg], b1, d_o[54] * a1);
        d_o[55] = __fmaf_rn(part_hi0[2944 + tid_wg], b1, d_o[55] * a1);
        d_o[56] = __fmaf_rn(part_hi0[3072 + tid_wg], b0, d_o[56] * a0);
        d_o[57] = __fmaf_rn(part_hi0[3200 + tid_wg], b0, d_o[57] * a0);
        d_o[58] = __fmaf_rn(part_hi0[3328 + tid_wg], b1, d_o[58] * a1);
        d_o[59] = __fmaf_rn(part_hi0[3456 + tid_wg], b1, d_o[59] * a1);
        d_o[60] = __fmaf_rn(part_hi0[3584 + tid_wg], b0, d_o[60] * a0);
        d_o[61] = __fmaf_rn(part_hi0[3712 + tid_wg], b0, d_o[61] * a0);
        d_o[62] = __fmaf_rn(part_hi0[3840 + tid_wg], b1, d_o[62] * a1);
        d_o[63] = __fmaf_rn(part_hi0[3968 + tid_wg], b1, d_o[63] * a1);
    }
    row_sum0 = __fmaf_rn(x1l0, f10, x0l0 * f00);
    row_sum1 = __fmaf_rn(x1l1, f11, x0l1 * f01);
    row_max0 = mm0;
    row_max1 = mm1;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (wg == 0) {
        int my_rank = cta_rank;
        int my_off = tid_wg * 16;
        if (my_rank != 0) {
            int pslot = my_rank;
            if (my_rank > 0) {
                pslot = my_rank - 1;
            }
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(recv_smem_addr + (unsigned int)(pslot * 17408) + (unsigned int)my_off), "r"(0));
            uint32_t _mbar_peer_320;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_320) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0), "r"(__float_as_uint(d_o[0])), "r"(__float_as_uint(d_o[1])), "r"(__float_as_uint(d_o[2])), "r"(__float_as_uint(d_o[3])), "r"(_mbar_peer_320) : "memory");
            uint32_t _mbar_peer_321;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_321) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(d_o[4])), "r"(__float_as_uint(d_o[5])), "r"(__float_as_uint(d_o[6])), "r"(__float_as_uint(d_o[7])), "r"(_mbar_peer_321) : "memory");
            uint32_t _mbar_peer_322;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_322) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(d_o[8])), "r"(__float_as_uint(d_o[9])), "r"(__float_as_uint(d_o[10])), "r"(__float_as_uint(d_o[11])), "r"(_mbar_peer_322) : "memory");
            uint32_t _mbar_peer_323;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_323) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(d_o[12])), "r"(__float_as_uint(d_o[13])), "r"(__float_as_uint(d_o[14])), "r"(__float_as_uint(d_o[15])), "r"(_mbar_peer_323) : "memory");
            if (qj == 0) {
                uint32_t _cluster_mapa_324;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_324) : "r"(recv_smem_addr + (unsigned int)(pslot * 17408) + 16384 + (unsigned int)(m0_local * 16)), "r"(0));
                uint32_t _mbar_peer_325;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_325) : "r"(recv_full_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_324), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_325) : "memory");
            }
        }
        if (my_rank != 1) {
            int pslot_1 = my_rank;
            if (my_rank > 1) {
                pslot_1 = my_rank - 1;
            }
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(recv_smem_addr + (unsigned int)(pslot_1 * 17408) + (unsigned int)my_off), "r"(1));
            uint32_t _mbar_peer_326;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_326) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1), "r"(__float_as_uint(d_o[16])), "r"(__float_as_uint(d_o[17])), "r"(__float_as_uint(d_o[18])), "r"(__float_as_uint(d_o[19])), "r"(_mbar_peer_326) : "memory");
            uint32_t _mbar_peer_327;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_327) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(d_o[20])), "r"(__float_as_uint(d_o[21])), "r"(__float_as_uint(d_o[22])), "r"(__float_as_uint(d_o[23])), "r"(_mbar_peer_327) : "memory");
            uint32_t _mbar_peer_328;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_328) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(d_o[24])), "r"(__float_as_uint(d_o[25])), "r"(__float_as_uint(d_o[26])), "r"(__float_as_uint(d_o[27])), "r"(_mbar_peer_328) : "memory");
            uint32_t _mbar_peer_329;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_329) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(d_o[28])), "r"(__float_as_uint(d_o[29])), "r"(__float_as_uint(d_o[30])), "r"(__float_as_uint(d_o[31])), "r"(_mbar_peer_329) : "memory");
            if (qj == 0) {
                uint32_t _cluster_mapa_330;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_330) : "r"(recv_smem_addr + (unsigned int)(pslot_1 * 17408) + 16384 + (unsigned int)(m0_local * 16)), "r"(1));
                uint32_t _mbar_peer_331;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_331) : "r"(recv_full_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_330), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_331) : "memory");
            }
        }
        if (my_rank < 2) {
            mbarrier_wait_cluster_hint(recv_full_addr, 0, 10000000);
            float merged_max0 = row_max0;
            float merged_max1 = row_max1;
            merged_max0 = fmaxf(merged_max0, recv_smem[4096 + m0_local * 4]);
            merged_max1 = fmaxf(merged_max1, recv_smem[4096 + m0_local * 4 + 2]);
            float exp2_21 = approx_exp2(row_max0 - merged_max0);
            float exp2_1_6 = approx_exp2(row_max1 - merged_max1);
            float msum0 = row_sum0 * exp2_21;
            float msum1 = row_sum1 * exp2_1_6;
            float w_peer0[1];
            float w_peer1[1];
            int sbase = 4096 + m0_local * 4;
            float _exp2_332 = approx_exp2(recv_smem[sbase] - merged_max0);
            w_peer0[0] = _exp2_332;
            float _exp2_333 = approx_exp2(recv_smem[sbase + 2] - merged_max1);
            w_peer1[0] = _exp2_333;
            msum0 = __fmaf_rn(recv_smem[sbase + 1], w_peer0[0], msum0);
            msum1 = __fmaf_rn(recv_smem[sbase + 3], w_peer1[0], msum1);
            int my_tid4 = tid_wg * 4;
            float acc[16];
            int sset_1_0 = my_rank;
            int fold_2_0 = ((sset_1_0 < 4) ? 1 : 0);
            if (sset_1_0 < 0) {
                fold_2_0 = 0;
            }
            if (sset_1_0 >= 2) {
                fold_2_0 = 0;
            }
            if (fold_2_0 != 0) {
                if (sset_1_0 == 0) {
                    acc[0] = d_o[0] * exp2_21;
                    acc[1] = d_o[1] * exp2_21;
                    acc[2] = d_o[2] * exp2_1_6;
                    acc[3] = d_o[3] * exp2_1_6;
                    acc[4] = d_o[4] * exp2_21;
                    acc[5] = d_o[5] * exp2_21;
                    acc[6] = d_o[6] * exp2_1_6;
                    acc[7] = d_o[7] * exp2_1_6;
                    acc[8] = d_o[8] * exp2_21;
                    acc[9] = d_o[9] * exp2_21;
                    acc[10] = d_o[10] * exp2_1_6;
                    acc[11] = d_o[11] * exp2_1_6;
                    acc[12] = d_o[12] * exp2_21;
                    acc[13] = d_o[13] * exp2_21;
                    acc[14] = d_o[14] * exp2_1_6;
                    acc[15] = d_o[15] * exp2_1_6;
                }
                if (sset_1_0 == 1) {
                    acc[0] = d_o[16] * exp2_21;
                    acc[1] = d_o[17] * exp2_21;
                    acc[2] = d_o[18] * exp2_1_6;
                    acc[3] = d_o[19] * exp2_1_6;
                    acc[4] = d_o[20] * exp2_21;
                    acc[5] = d_o[21] * exp2_21;
                    acc[6] = d_o[22] * exp2_1_6;
                    acc[7] = d_o[23] * exp2_1_6;
                    acc[8] = d_o[24] * exp2_21;
                    acc[9] = d_o[25] * exp2_21;
                    acc[10] = d_o[26] * exp2_1_6;
                    acc[11] = d_o[27] * exp2_1_6;
                    acc[12] = d_o[28] * exp2_21;
                    acc[13] = d_o[29] * exp2_21;
                    acc[14] = d_o[30] * exp2_1_6;
                    acc[15] = d_o[31] * exp2_1_6;
                }
                int base = my_tid4;
                acc[0] = __fmaf_rn(recv_smem[base], w_peer0[0], acc[0]);
                acc[1] = __fmaf_rn(recv_smem[base + 1], w_peer0[0], acc[1]);
                acc[2] = __fmaf_rn(recv_smem[base + 2], w_peer1[0], acc[2]);
                acc[3] = __fmaf_rn(recv_smem[base + 3], w_peer1[0], acc[3]);
                acc[4] = __fmaf_rn(recv_smem[base + 512], w_peer0[0], acc[4]);
                acc[5] = __fmaf_rn(recv_smem[base + 512 + 1], w_peer0[0], acc[5]);
                acc[6] = __fmaf_rn(recv_smem[base + 512 + 2], w_peer1[0], acc[6]);
                acc[7] = __fmaf_rn(recv_smem[base + 512 + 3], w_peer1[0], acc[7]);
                acc[8] = __fmaf_rn(recv_smem[base + 1024], w_peer0[0], acc[8]);
                acc[9] = __fmaf_rn(recv_smem[base + 1024 + 1], w_peer0[0], acc[9]);
                acc[10] = __fmaf_rn(recv_smem[base + 1024 + 2], w_peer1[0], acc[10]);
                acc[11] = __fmaf_rn(recv_smem[base + 1024 + 3], w_peer1[0], acc[11]);
                acc[12] = __fmaf_rn(recv_smem[base + 1536], w_peer0[0], acc[12]);
                acc[13] = __fmaf_rn(recv_smem[base + 1536 + 1], w_peer0[0], acc[13]);
                acc[14] = __fmaf_rn(recv_smem[base + 1536 + 2], w_peer1[0], acc[14]);
                acc[15] = __fmaf_rn(recv_smem[base + 1536 + 3], w_peer1[0], acc[15]);
                float rcp = approx_rcp(msum0);
                float rcp_1 = approx_rcp(msum1);
                int vs_set = vs_base + sset_1_0 * 32;
                float vs0_1_0 = vs_smem[vs_set];
                float vs1_1_0 = vs_smem[vs_set + 1];
                acc[0] = acc[0] * (vs0_1_0 * rcp);
                acc[1] = acc[1] * (vs1_1_0 * rcp);
                acc[2] = acc[2] * (vs0_1_0 * rcp_1);
                acc[3] = acc[3] * (vs1_1_0 * rcp_1);
                float vs0_1_1 = vs_smem[vs_set + 8];
                float vs1_1_1 = vs_smem[vs_set + 8 + 1];
                acc[4] = acc[4] * (vs0_1_1 * rcp);
                acc[5] = acc[5] * (vs1_1_1 * rcp);
                acc[6] = acc[6] * (vs0_1_1 * rcp_1);
                acc[7] = acc[7] * (vs1_1_1 * rcp_1);
                float vs0_1_2 = vs_smem[vs_set + 16];
                float vs1_1_2 = vs_smem[vs_set + 16 + 1];
                acc[8] = acc[8] * (vs0_1_2 * rcp);
                acc[9] = acc[9] * (vs1_1_2 * rcp);
                acc[10] = acc[10] * (vs0_1_2 * rcp_1);
                acc[11] = acc[11] * (vs1_1_2 * rcp_1);
                float vs0_1_3 = vs_smem[vs_set + 24];
                float vs1_1_3 = vs_smem[vs_set + 24 + 1];
                acc[12] = acc[12] * (vs0_1_3 * rcp);
                acc[13] = acc[13] * (vs1_1_3 * rcp);
                acc[14] = acc[14] * (vs0_1_3 * rcp_1);
                acc[15] = acc[15] * (vs1_1_3 * rcp_1);
                int qj1 = qj & 1;
                int qj2 = qj & 2;
                unsigned int o_vec[4];
                unsigned int o_tmp[4];
                int o_row_base = q_row * 128;
                int m_local_r_1_0 = m0_local;
                __nv_bfloat162 _bit_cast_src_334 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_334)[0];
                __nv_bfloat162 _bit_cast_src_335 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_335)[0];
                __nv_bfloat162 _bit_cast_src_336 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_336)[0];
                __nv_bfloat162 _bit_cast_src_337 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_337)[0];
                unsigned int _shfl_xor_sync_338 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                o_tmp[0] = _shfl_xor_sync_338;
                unsigned int _shfl_xor_sync_339 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                o_tmp[1] = _shfl_xor_sync_339;
                unsigned int _shfl_xor_sync_340 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                o_tmp[2] = _shfl_xor_sync_340;
                unsigned int _shfl_xor_sync_341 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                o_tmp[3] = _shfl_xor_sync_341;
                o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                unsigned int _shfl_xor_sync_342 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                o_tmp[0] = _shfl_xor_sync_342;
                unsigned int _shfl_xor_sync_343 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                o_tmp[1] = _shfl_xor_sync_343;
                unsigned int _shfl_xor_sync_344 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                o_tmp[2] = _shfl_xor_sync_344;
                unsigned int _shfl_xor_sync_345 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                o_tmp[3] = _shfl_xor_sync_345;
                o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                int o_off_1_0 = o_row_base + m_local_r_1_0 * 128 + (4 * sset_1_0 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_0)[0] = reinterpret_cast<int4*>(o_vec)[0];
                int m_local_r_1_1 = m1_local;
                __nv_bfloat162 _bit_cast_src_346 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_346)[0];
                __nv_bfloat162 _bit_cast_src_347 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_347)[0];
                __nv_bfloat162 _bit_cast_src_348 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_348)[0];
                __nv_bfloat162 _bit_cast_src_349 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_349)[0];
                unsigned int _shfl_xor_sync_350 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                o_tmp[0] = _shfl_xor_sync_350;
                unsigned int _shfl_xor_sync_351 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                o_tmp[1] = _shfl_xor_sync_351;
                unsigned int _shfl_xor_sync_352 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                o_tmp[2] = _shfl_xor_sync_352;
                unsigned int _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                o_tmp[3] = _shfl_xor_sync_353;
                o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                unsigned int _shfl_xor_sync_354 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                o_tmp[0] = _shfl_xor_sync_354;
                unsigned int _shfl_xor_sync_355 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                o_tmp[1] = _shfl_xor_sync_355;
                unsigned int _shfl_xor_sync_356 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                o_tmp[2] = _shfl_xor_sync_356;
                unsigned int _shfl_xor_sync_357 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                o_tmp[3] = _shfl_xor_sync_357;
                o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                int o_off_1_1 = o_row_base + m_local_r_1_1 * 128 + (4 * sset_1_0 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_1)[0] = reinterpret_cast<int4*>(o_vec)[0];
            }
            int sset_1_1 = my_rank + 2;
            int fold_2_1 = ((sset_1_1 < 4) ? 1 : 0);
            if (sset_1_1 < 0) {
                fold_2_1 = 0;
            }
            if (sset_1_1 >= 2) {
                fold_2_1 = 0;
            }
            if (fold_2_1 != 0) {
                if (sset_1_1 == 0) {
                    acc[0] = d_o[0] * exp2_21;
                    acc[1] = d_o[1] * exp2_21;
                    acc[2] = d_o[2] * exp2_1_6;
                    acc[3] = d_o[3] * exp2_1_6;
                    acc[4] = d_o[4] * exp2_21;
                    acc[5] = d_o[5] * exp2_21;
                    acc[6] = d_o[6] * exp2_1_6;
                    acc[7] = d_o[7] * exp2_1_6;
                    acc[8] = d_o[8] * exp2_21;
                    acc[9] = d_o[9] * exp2_21;
                    acc[10] = d_o[10] * exp2_1_6;
                    acc[11] = d_o[11] * exp2_1_6;
                    acc[12] = d_o[12] * exp2_21;
                    acc[13] = d_o[13] * exp2_21;
                    acc[14] = d_o[14] * exp2_1_6;
                    acc[15] = d_o[15] * exp2_1_6;
                }
                if (sset_1_1 == 1) {
                    acc[0] = d_o[16] * exp2_21;
                    acc[1] = d_o[17] * exp2_21;
                    acc[2] = d_o[18] * exp2_1_6;
                    acc[3] = d_o[19] * exp2_1_6;
                    acc[4] = d_o[20] * exp2_21;
                    acc[5] = d_o[21] * exp2_21;
                    acc[6] = d_o[22] * exp2_1_6;
                    acc[7] = d_o[23] * exp2_1_6;
                    acc[8] = d_o[24] * exp2_21;
                    acc[9] = d_o[25] * exp2_21;
                    acc[10] = d_o[26] * exp2_1_6;
                    acc[11] = d_o[27] * exp2_1_6;
                    acc[12] = d_o[28] * exp2_21;
                    acc[13] = d_o[29] * exp2_21;
                    acc[14] = d_o[30] * exp2_1_6;
                    acc[15] = d_o[31] * exp2_1_6;
                }
                int base_1 = 2048 + my_tid4;
                acc[0] = __fmaf_rn(recv_smem[base_1], w_peer0[0], acc[0]);
                acc[1] = __fmaf_rn(recv_smem[base_1 + 1], w_peer0[0], acc[1]);
                acc[2] = __fmaf_rn(recv_smem[base_1 + 2], w_peer1[0], acc[2]);
                acc[3] = __fmaf_rn(recv_smem[base_1 + 3], w_peer1[0], acc[3]);
                acc[4] = __fmaf_rn(recv_smem[base_1 + 512], w_peer0[0], acc[4]);
                acc[5] = __fmaf_rn(recv_smem[base_1 + 512 + 1], w_peer0[0], acc[5]);
                acc[6] = __fmaf_rn(recv_smem[base_1 + 512 + 2], w_peer1[0], acc[6]);
                acc[7] = __fmaf_rn(recv_smem[base_1 + 512 + 3], w_peer1[0], acc[7]);
                acc[8] = __fmaf_rn(recv_smem[base_1 + 1024], w_peer0[0], acc[8]);
                acc[9] = __fmaf_rn(recv_smem[base_1 + 1024 + 1], w_peer0[0], acc[9]);
                acc[10] = __fmaf_rn(recv_smem[base_1 + 1024 + 2], w_peer1[0], acc[10]);
                acc[11] = __fmaf_rn(recv_smem[base_1 + 1024 + 3], w_peer1[0], acc[11]);
                acc[12] = __fmaf_rn(recv_smem[base_1 + 1536], w_peer0[0], acc[12]);
                acc[13] = __fmaf_rn(recv_smem[base_1 + 1536 + 1], w_peer0[0], acc[13]);
                acc[14] = __fmaf_rn(recv_smem[base_1 + 1536 + 2], w_peer1[0], acc[14]);
                acc[15] = __fmaf_rn(recv_smem[base_1 + 1536 + 3], w_peer1[0], acc[15]);
                float rcp_2 = approx_rcp(msum0);
                float rcp_1_1 = approx_rcp(msum1);
                int vs_set_1 = vs_base + sset_1_1 * 32;
                float vs0_1_4 = vs_smem[vs_set_1];
                float vs1_1_4 = vs_smem[vs_set_1 + 1];
                acc[0] = acc[0] * (vs0_1_4 * rcp_2);
                acc[1] = acc[1] * (vs1_1_4 * rcp_2);
                acc[2] = acc[2] * (vs0_1_4 * rcp_1_1);
                acc[3] = acc[3] * (vs1_1_4 * rcp_1_1);
                float vs0_1_5 = vs_smem[vs_set_1 + 8];
                float vs1_1_5 = vs_smem[vs_set_1 + 8 + 1];
                acc[4] = acc[4] * (vs0_1_5 * rcp_2);
                acc[5] = acc[5] * (vs1_1_5 * rcp_2);
                acc[6] = acc[6] * (vs0_1_5 * rcp_1_1);
                acc[7] = acc[7] * (vs1_1_5 * rcp_1_1);
                float vs0_1_6 = vs_smem[vs_set_1 + 16];
                float vs1_1_6 = vs_smem[vs_set_1 + 16 + 1];
                acc[8] = acc[8] * (vs0_1_6 * rcp_2);
                acc[9] = acc[9] * (vs1_1_6 * rcp_2);
                acc[10] = acc[10] * (vs0_1_6 * rcp_1_1);
                acc[11] = acc[11] * (vs1_1_6 * rcp_1_1);
                float vs0_1_7 = vs_smem[vs_set_1 + 24];
                float vs1_1_7 = vs_smem[vs_set_1 + 24 + 1];
                acc[12] = acc[12] * (vs0_1_7 * rcp_2);
                acc[13] = acc[13] * (vs1_1_7 * rcp_2);
                acc[14] = acc[14] * (vs0_1_7 * rcp_1_1);
                acc[15] = acc[15] * (vs1_1_7 * rcp_1_1);
                int qj1_1 = qj & 1;
                int qj2_1 = qj & 2;
                unsigned int o_vec_1[4];
                unsigned int o_tmp_1[4];
                int o_row_base_1 = q_row * 128;
                int m_local_r_1_2 = m0_local;
                __nv_bfloat162 _bit_cast_src_358 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_358)[0];
                __nv_bfloat162 _bit_cast_src_359 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_359)[0];
                __nv_bfloat162 _bit_cast_src_360 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_360)[0];
                __nv_bfloat162 _bit_cast_src_361 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_361)[0];
                unsigned int _shfl_xor_sync_362 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                o_tmp_1[0] = _shfl_xor_sync_362;
                unsigned int _shfl_xor_sync_363 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                o_tmp_1[1] = _shfl_xor_sync_363;
                unsigned int _shfl_xor_sync_364 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                o_tmp_1[2] = _shfl_xor_sync_364;
                unsigned int _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                o_tmp_1[3] = _shfl_xor_sync_365;
                o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                unsigned int _shfl_xor_sync_366 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                o_tmp_1[0] = _shfl_xor_sync_366;
                unsigned int _shfl_xor_sync_367 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                o_tmp_1[1] = _shfl_xor_sync_367;
                unsigned int _shfl_xor_sync_368 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                o_tmp_1[2] = _shfl_xor_sync_368;
                unsigned int _shfl_xor_sync_369 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                o_tmp_1[3] = _shfl_xor_sync_369;
                o_vec_1[0] = ((qj2_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj2_1 != 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj2_1 == 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj2_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                int o_off_1_2 = o_row_base_1 + m_local_r_1_2 * 128 + (4 * sset_1_1 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_2)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
                int m_local_r_1_3 = m1_local;
                __nv_bfloat162 _bit_cast_src_370 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_370)[0];
                __nv_bfloat162 _bit_cast_src_371 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_371)[0];
                __nv_bfloat162 _bit_cast_src_372 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_372)[0];
                __nv_bfloat162 _bit_cast_src_373 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_373)[0];
                unsigned int _shfl_xor_sync_374 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                o_tmp_1[0] = _shfl_xor_sync_374;
                unsigned int _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                o_tmp_1[1] = _shfl_xor_sync_375;
                unsigned int _shfl_xor_sync_376 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                o_tmp_1[2] = _shfl_xor_sync_376;
                unsigned int _shfl_xor_sync_377 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                o_tmp_1[3] = _shfl_xor_sync_377;
                o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                unsigned int _shfl_xor_sync_378 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                o_tmp_1[0] = _shfl_xor_sync_378;
                unsigned int _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                o_tmp_1[1] = _shfl_xor_sync_379;
                unsigned int _shfl_xor_sync_380 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                o_tmp_1[2] = _shfl_xor_sync_380;
                unsigned int _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                o_tmp_1[3] = _shfl_xor_sync_381;
                o_vec_1[0] = ((qj2_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj2_1 != 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj2_1 == 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj2_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                int o_off_1_3 = o_row_base_1 + m_local_r_1_3 * 128 + (4 * sset_1_1 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_3)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            }
        }
    }
    if (wg == 1) {
        int my_rank_1 = cta_rank;
        int my_off_1 = tid_wg * 16;
        if (my_rank_1 != 0) {
            int pslot_2 = my_rank_1;
            if (my_rank_1 > 0) {
                pslot_2 = my_rank_1 - 1;
            }
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(recv_smem_addr + (unsigned int)(pslot_2 * 17408) + 8192 + (unsigned int)my_off_1), "r"(0));
            uint32_t _mbar_peer_382;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_382) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2), "r"(__float_as_uint(d_o[32])), "r"(__float_as_uint(d_o[33])), "r"(__float_as_uint(d_o[34])), "r"(__float_as_uint(d_o[35])), "r"(_mbar_peer_382) : "memory");
            uint32_t _mbar_peer_383;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_383) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 2048), "r"(__float_as_uint(d_o[36])), "r"(__float_as_uint(d_o[37])), "r"(__float_as_uint(d_o[38])), "r"(__float_as_uint(d_o[39])), "r"(_mbar_peer_383) : "memory");
            uint32_t _mbar_peer_384;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_384) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 4096), "r"(__float_as_uint(d_o[40])), "r"(__float_as_uint(d_o[41])), "r"(__float_as_uint(d_o[42])), "r"(__float_as_uint(d_o[43])), "r"(_mbar_peer_384) : "memory");
            uint32_t _mbar_peer_385;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_385) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 6144), "r"(__float_as_uint(d_o[44])), "r"(__float_as_uint(d_o[45])), "r"(__float_as_uint(d_o[46])), "r"(__float_as_uint(d_o[47])), "r"(_mbar_peer_385) : "memory");
        }
        if (my_rank_1 != 1) {
            int pslot_3 = my_rank_1;
            if (my_rank_1 > 1) {
                pslot_3 = my_rank_1 - 1;
            }
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(recv_smem_addr + (unsigned int)(pslot_3 * 17408) + 8192 + (unsigned int)my_off_1), "r"(1));
            uint32_t _mbar_peer_386;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_386) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3), "r"(__float_as_uint(d_o[48])), "r"(__float_as_uint(d_o[49])), "r"(__float_as_uint(d_o[50])), "r"(__float_as_uint(d_o[51])), "r"(_mbar_peer_386) : "memory");
            uint32_t _mbar_peer_387;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_387) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 2048), "r"(__float_as_uint(d_o[52])), "r"(__float_as_uint(d_o[53])), "r"(__float_as_uint(d_o[54])), "r"(__float_as_uint(d_o[55])), "r"(_mbar_peer_387) : "memory");
            uint32_t _mbar_peer_388;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_388) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 4096), "r"(__float_as_uint(d_o[56])), "r"(__float_as_uint(d_o[57])), "r"(__float_as_uint(d_o[58])), "r"(__float_as_uint(d_o[59])), "r"(_mbar_peer_388) : "memory");
            uint32_t _mbar_peer_389;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_389) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 6144), "r"(__float_as_uint(d_o[60])), "r"(__float_as_uint(d_o[61])), "r"(__float_as_uint(d_o[62])), "r"(__float_as_uint(d_o[63])), "r"(_mbar_peer_389) : "memory");
        }
        if (my_rank_1 < 2) {
            mbarrier_wait_cluster_hint(recv_full_addr, 0, 10000000);
            float merged_max0_4 = row_max0;
            float merged_max1_4 = row_max1;
            merged_max0_4 = fmaxf(merged_max0_4, recv_smem[4096 + m0_local * 4]);
            merged_max1_4 = fmaxf(merged_max1_4, recv_smem[4096 + m0_local * 4 + 2]);
            float exp2_22 = approx_exp2(row_max0 - merged_max0_4);
            float exp2_1_7 = approx_exp2(row_max1 - merged_max1_4);
            float msum0_1 = row_sum0 * exp2_22;
            float msum1_1 = row_sum1 * exp2_1_7;
            float w_peer0_1[1];
            float w_peer1_1[1];
            int sbase_1 = 4096 + m0_local * 4;
            float _exp2_390 = approx_exp2(recv_smem[sbase_1] - merged_max0_4);
            w_peer0_1[0] = _exp2_390;
            float _exp2_391 = approx_exp2(recv_smem[sbase_1 + 2] - merged_max1_4);
            w_peer1_1[0] = _exp2_391;
            msum0_1 = __fmaf_rn(recv_smem[sbase_1 + 1], w_peer0_1[0], msum0_1);
            msum1_1 = __fmaf_rn(recv_smem[sbase_1 + 3], w_peer1_1[0], msum1_1);
            int my_tid4_1 = tid_wg * 4;
            float acc_1[16];
            int sset_1_2 = my_rank_1;
            int fold_2_2 = ((sset_1_2 < 4) ? 1 : 0);
            if (sset_1_2 < 2) {
                fold_2_2 = 0;
            }
            if (sset_1_2 >= 4) {
                fold_2_2 = 0;
            }
            if (fold_2_2 != 0) {
                if (sset_1_2 == 2) {
                    acc_1[0] = d_o[32] * exp2_22;
                    acc_1[1] = d_o[33] * exp2_22;
                    acc_1[2] = d_o[34] * exp2_1_7;
                    acc_1[3] = d_o[35] * exp2_1_7;
                    acc_1[4] = d_o[36] * exp2_22;
                    acc_1[5] = d_o[37] * exp2_22;
                    acc_1[6] = d_o[38] * exp2_1_7;
                    acc_1[7] = d_o[39] * exp2_1_7;
                    acc_1[8] = d_o[40] * exp2_22;
                    acc_1[9] = d_o[41] * exp2_22;
                    acc_1[10] = d_o[42] * exp2_1_7;
                    acc_1[11] = d_o[43] * exp2_1_7;
                    acc_1[12] = d_o[44] * exp2_22;
                    acc_1[13] = d_o[45] * exp2_22;
                    acc_1[14] = d_o[46] * exp2_1_7;
                    acc_1[15] = d_o[47] * exp2_1_7;
                }
                if (sset_1_2 == 3) {
                    acc_1[0] = d_o[48] * exp2_22;
                    acc_1[1] = d_o[49] * exp2_22;
                    acc_1[2] = d_o[50] * exp2_1_7;
                    acc_1[3] = d_o[51] * exp2_1_7;
                    acc_1[4] = d_o[52] * exp2_22;
                    acc_1[5] = d_o[53] * exp2_22;
                    acc_1[6] = d_o[54] * exp2_1_7;
                    acc_1[7] = d_o[55] * exp2_1_7;
                    acc_1[8] = d_o[56] * exp2_22;
                    acc_1[9] = d_o[57] * exp2_22;
                    acc_1[10] = d_o[58] * exp2_1_7;
                    acc_1[11] = d_o[59] * exp2_1_7;
                    acc_1[12] = d_o[60] * exp2_22;
                    acc_1[13] = d_o[61] * exp2_22;
                    acc_1[14] = d_o[62] * exp2_1_7;
                    acc_1[15] = d_o[63] * exp2_1_7;
                }
                int base_2 = my_tid4_1;
                acc_1[0] = __fmaf_rn(recv_smem[base_2], w_peer0_1[0], acc_1[0]);
                acc_1[1] = __fmaf_rn(recv_smem[base_2 + 1], w_peer0_1[0], acc_1[1]);
                acc_1[2] = __fmaf_rn(recv_smem[base_2 + 2], w_peer1_1[0], acc_1[2]);
                acc_1[3] = __fmaf_rn(recv_smem[base_2 + 3], w_peer1_1[0], acc_1[3]);
                acc_1[4] = __fmaf_rn(recv_smem[base_2 + 512], w_peer0_1[0], acc_1[4]);
                acc_1[5] = __fmaf_rn(recv_smem[base_2 + 512 + 1], w_peer0_1[0], acc_1[5]);
                acc_1[6] = __fmaf_rn(recv_smem[base_2 + 512 + 2], w_peer1_1[0], acc_1[6]);
                acc_1[7] = __fmaf_rn(recv_smem[base_2 + 512 + 3], w_peer1_1[0], acc_1[7]);
                acc_1[8] = __fmaf_rn(recv_smem[base_2 + 1024], w_peer0_1[0], acc_1[8]);
                acc_1[9] = __fmaf_rn(recv_smem[base_2 + 1024 + 1], w_peer0_1[0], acc_1[9]);
                acc_1[10] = __fmaf_rn(recv_smem[base_2 + 1024 + 2], w_peer1_1[0], acc_1[10]);
                acc_1[11] = __fmaf_rn(recv_smem[base_2 + 1024 + 3], w_peer1_1[0], acc_1[11]);
                acc_1[12] = __fmaf_rn(recv_smem[base_2 + 1536], w_peer0_1[0], acc_1[12]);
                acc_1[13] = __fmaf_rn(recv_smem[base_2 + 1536 + 1], w_peer0_1[0], acc_1[13]);
                acc_1[14] = __fmaf_rn(recv_smem[base_2 + 1536 + 2], w_peer1_1[0], acc_1[14]);
                acc_1[15] = __fmaf_rn(recv_smem[base_2 + 1536 + 3], w_peer1_1[0], acc_1[15]);
                float rcp_3 = approx_rcp(msum0_1);
                float rcp_1_2 = approx_rcp(msum1_1);
                int vs_set_2 = vs_base + sset_1_2 * 32;
                float vs0_1_8 = vs_smem[vs_set_2];
                float vs1_1_8 = vs_smem[vs_set_2 + 1];
                acc_1[0] = acc_1[0] * (vs0_1_8 * rcp_3);
                acc_1[1] = acc_1[1] * (vs1_1_8 * rcp_3);
                acc_1[2] = acc_1[2] * (vs0_1_8 * rcp_1_2);
                acc_1[3] = acc_1[3] * (vs1_1_8 * rcp_1_2);
                float vs0_1_9 = vs_smem[vs_set_2 + 8];
                float vs1_1_9 = vs_smem[vs_set_2 + 8 + 1];
                acc_1[4] = acc_1[4] * (vs0_1_9 * rcp_3);
                acc_1[5] = acc_1[5] * (vs1_1_9 * rcp_3);
                acc_1[6] = acc_1[6] * (vs0_1_9 * rcp_1_2);
                acc_1[7] = acc_1[7] * (vs1_1_9 * rcp_1_2);
                float vs0_1_10 = vs_smem[vs_set_2 + 16];
                float vs1_1_10 = vs_smem[vs_set_2 + 16 + 1];
                acc_1[8] = acc_1[8] * (vs0_1_10 * rcp_3);
                acc_1[9] = acc_1[9] * (vs1_1_10 * rcp_3);
                acc_1[10] = acc_1[10] * (vs0_1_10 * rcp_1_2);
                acc_1[11] = acc_1[11] * (vs1_1_10 * rcp_1_2);
                float vs0_1_11 = vs_smem[vs_set_2 + 24];
                float vs1_1_11 = vs_smem[vs_set_2 + 24 + 1];
                acc_1[12] = acc_1[12] * (vs0_1_11 * rcp_3);
                acc_1[13] = acc_1[13] * (vs1_1_11 * rcp_3);
                acc_1[14] = acc_1[14] * (vs0_1_11 * rcp_1_2);
                acc_1[15] = acc_1[15] * (vs1_1_11 * rcp_1_2);
                int qj1_2 = qj & 1;
                int qj2_2 = qj & 2;
                unsigned int o_vec_2[4];
                unsigned int o_tmp_2[4];
                int o_row_base_2 = q_row * 128;
                int m_local_r_1_4 = m0_local;
                __nv_bfloat162 _bit_cast_src_392 = __float22bfloat162_rn(make_float2(acc_1[0], acc_1[1]));
                o_vec_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_392)[0];
                __nv_bfloat162 _bit_cast_src_393 = __float22bfloat162_rn(make_float2(acc_1[4], acc_1[5]));
                o_vec_2[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_393)[0];
                __nv_bfloat162 _bit_cast_src_394 = __float22bfloat162_rn(make_float2(acc_1[8], acc_1[9]));
                o_vec_2[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_394)[0];
                __nv_bfloat162 _bit_cast_src_395 = __float22bfloat162_rn(make_float2(acc_1[12], acc_1[13]));
                o_vec_2[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_395)[0];
                unsigned int _shfl_xor_sync_396 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 1);
                o_tmp_2[0] = _shfl_xor_sync_396;
                unsigned int _shfl_xor_sync_397 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 1);
                o_tmp_2[1] = _shfl_xor_sync_397;
                unsigned int _shfl_xor_sync_398 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 1);
                o_tmp_2[2] = _shfl_xor_sync_398;
                unsigned int _shfl_xor_sync_399 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 1);
                o_tmp_2[3] = _shfl_xor_sync_399;
                o_vec_2[0] = ((qj1_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                o_vec_2[1] = ((qj1_2 == 0) ? o_tmp_2[1] : o_vec_2[1]);
                o_vec_2[2] = ((qj1_2 != 0) ? o_tmp_2[2] : o_vec_2[2]);
                o_vec_2[3] = ((qj1_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                unsigned int _shfl_xor_sync_400 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 2);
                o_tmp_2[0] = _shfl_xor_sync_400;
                unsigned int _shfl_xor_sync_401 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 2);
                o_tmp_2[1] = _shfl_xor_sync_401;
                unsigned int _shfl_xor_sync_402 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 2);
                o_tmp_2[2] = _shfl_xor_sync_402;
                unsigned int _shfl_xor_sync_403 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 2);
                o_tmp_2[3] = _shfl_xor_sync_403;
                o_vec_2[0] = ((qj2_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                o_vec_2[1] = ((qj2_2 != 0) ? o_tmp_2[1] : o_vec_2[1]);
                o_vec_2[2] = ((qj2_2 == 0) ? o_tmp_2[2] : o_vec_2[2]);
                o_vec_2[3] = ((qj2_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                int o_off_1_4 = o_row_base_2 + m_local_r_1_4 * 128 + (4 * sset_1_2 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_4)[0] = reinterpret_cast<int4*>(o_vec_2)[0];
                int m_local_r_1_5 = m1_local;
                __nv_bfloat162 _bit_cast_src_404 = __float22bfloat162_rn(make_float2(acc_1[2], acc_1[3]));
                o_vec_2[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_404)[0];
                __nv_bfloat162 _bit_cast_src_405 = __float22bfloat162_rn(make_float2(acc_1[6], acc_1[7]));
                o_vec_2[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_405)[0];
                __nv_bfloat162 _bit_cast_src_406 = __float22bfloat162_rn(make_float2(acc_1[10], acc_1[11]));
                o_vec_2[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_406)[0];
                __nv_bfloat162 _bit_cast_src_407 = __float22bfloat162_rn(make_float2(acc_1[14], acc_1[15]));
                o_vec_2[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_407)[0];
                unsigned int _shfl_xor_sync_408 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 1);
                o_tmp_2[0] = _shfl_xor_sync_408;
                unsigned int _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 1);
                o_tmp_2[1] = _shfl_xor_sync_409;
                unsigned int _shfl_xor_sync_410 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 1);
                o_tmp_2[2] = _shfl_xor_sync_410;
                unsigned int _shfl_xor_sync_411 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 1);
                o_tmp_2[3] = _shfl_xor_sync_411;
                o_vec_2[0] = ((qj1_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                o_vec_2[1] = ((qj1_2 == 0) ? o_tmp_2[1] : o_vec_2[1]);
                o_vec_2[2] = ((qj1_2 != 0) ? o_tmp_2[2] : o_vec_2[2]);
                o_vec_2[3] = ((qj1_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                unsigned int _shfl_xor_sync_412 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[2], 2);
                o_tmp_2[0] = _shfl_xor_sync_412;
                unsigned int _shfl_xor_sync_413 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[3], 2);
                o_tmp_2[1] = _shfl_xor_sync_413;
                unsigned int _shfl_xor_sync_414 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[0], 2);
                o_tmp_2[2] = _shfl_xor_sync_414;
                unsigned int _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, o_vec_2[1], 2);
                o_tmp_2[3] = _shfl_xor_sync_415;
                o_vec_2[0] = ((qj2_2 != 0) ? o_tmp_2[0] : o_vec_2[0]);
                o_vec_2[1] = ((qj2_2 != 0) ? o_tmp_2[1] : o_vec_2[1]);
                o_vec_2[2] = ((qj2_2 == 0) ? o_tmp_2[2] : o_vec_2[2]);
                o_vec_2[3] = ((qj2_2 == 0) ? o_tmp_2[3] : o_vec_2[3]);
                int o_off_1_5 = o_row_base_2 + m_local_r_1_5 * 128 + (4 * sset_1_2 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_5)[0] = reinterpret_cast<int4*>(o_vec_2)[0];
            }
            int sset_1_3 = my_rank_1 + 2;
            int fold_2_3 = ((sset_1_3 < 4) ? 1 : 0);
            if (sset_1_3 < 2) {
                fold_2_3 = 0;
            }
            if (sset_1_3 >= 4) {
                fold_2_3 = 0;
            }
            if (fold_2_3 != 0) {
                if (sset_1_3 == 2) {
                    acc_1[0] = d_o[32] * exp2_22;
                    acc_1[1] = d_o[33] * exp2_22;
                    acc_1[2] = d_o[34] * exp2_1_7;
                    acc_1[3] = d_o[35] * exp2_1_7;
                    acc_1[4] = d_o[36] * exp2_22;
                    acc_1[5] = d_o[37] * exp2_22;
                    acc_1[6] = d_o[38] * exp2_1_7;
                    acc_1[7] = d_o[39] * exp2_1_7;
                    acc_1[8] = d_o[40] * exp2_22;
                    acc_1[9] = d_o[41] * exp2_22;
                    acc_1[10] = d_o[42] * exp2_1_7;
                    acc_1[11] = d_o[43] * exp2_1_7;
                    acc_1[12] = d_o[44] * exp2_22;
                    acc_1[13] = d_o[45] * exp2_22;
                    acc_1[14] = d_o[46] * exp2_1_7;
                    acc_1[15] = d_o[47] * exp2_1_7;
                }
                if (sset_1_3 == 3) {
                    acc_1[0] = d_o[48] * exp2_22;
                    acc_1[1] = d_o[49] * exp2_22;
                    acc_1[2] = d_o[50] * exp2_1_7;
                    acc_1[3] = d_o[51] * exp2_1_7;
                    acc_1[4] = d_o[52] * exp2_22;
                    acc_1[5] = d_o[53] * exp2_22;
                    acc_1[6] = d_o[54] * exp2_1_7;
                    acc_1[7] = d_o[55] * exp2_1_7;
                    acc_1[8] = d_o[56] * exp2_22;
                    acc_1[9] = d_o[57] * exp2_22;
                    acc_1[10] = d_o[58] * exp2_1_7;
                    acc_1[11] = d_o[59] * exp2_1_7;
                    acc_1[12] = d_o[60] * exp2_22;
                    acc_1[13] = d_o[61] * exp2_22;
                    acc_1[14] = d_o[62] * exp2_1_7;
                    acc_1[15] = d_o[63] * exp2_1_7;
                }
                int base_3 = 2048 + my_tid4_1;
                acc_1[0] = __fmaf_rn(recv_smem[base_3], w_peer0_1[0], acc_1[0]);
                acc_1[1] = __fmaf_rn(recv_smem[base_3 + 1], w_peer0_1[0], acc_1[1]);
                acc_1[2] = __fmaf_rn(recv_smem[base_3 + 2], w_peer1_1[0], acc_1[2]);
                acc_1[3] = __fmaf_rn(recv_smem[base_3 + 3], w_peer1_1[0], acc_1[3]);
                acc_1[4] = __fmaf_rn(recv_smem[base_3 + 512], w_peer0_1[0], acc_1[4]);
                acc_1[5] = __fmaf_rn(recv_smem[base_3 + 512 + 1], w_peer0_1[0], acc_1[5]);
                acc_1[6] = __fmaf_rn(recv_smem[base_3 + 512 + 2], w_peer1_1[0], acc_1[6]);
                acc_1[7] = __fmaf_rn(recv_smem[base_3 + 512 + 3], w_peer1_1[0], acc_1[7]);
                acc_1[8] = __fmaf_rn(recv_smem[base_3 + 1024], w_peer0_1[0], acc_1[8]);
                acc_1[9] = __fmaf_rn(recv_smem[base_3 + 1024 + 1], w_peer0_1[0], acc_1[9]);
                acc_1[10] = __fmaf_rn(recv_smem[base_3 + 1024 + 2], w_peer1_1[0], acc_1[10]);
                acc_1[11] = __fmaf_rn(recv_smem[base_3 + 1024 + 3], w_peer1_1[0], acc_1[11]);
                acc_1[12] = __fmaf_rn(recv_smem[base_3 + 1536], w_peer0_1[0], acc_1[12]);
                acc_1[13] = __fmaf_rn(recv_smem[base_3 + 1536 + 1], w_peer0_1[0], acc_1[13]);
                acc_1[14] = __fmaf_rn(recv_smem[base_3 + 1536 + 2], w_peer1_1[0], acc_1[14]);
                acc_1[15] = __fmaf_rn(recv_smem[base_3 + 1536 + 3], w_peer1_1[0], acc_1[15]);
                float rcp_4 = approx_rcp(msum0_1);
                float rcp_1_3 = approx_rcp(msum1_1);
                int vs_set_3 = vs_base + sset_1_3 * 32;
                float vs0_1_12 = vs_smem[vs_set_3];
                float vs1_1_12 = vs_smem[vs_set_3 + 1];
                acc_1[0] = acc_1[0] * (vs0_1_12 * rcp_4);
                acc_1[1] = acc_1[1] * (vs1_1_12 * rcp_4);
                acc_1[2] = acc_1[2] * (vs0_1_12 * rcp_1_3);
                acc_1[3] = acc_1[3] * (vs1_1_12 * rcp_1_3);
                float vs0_1_13 = vs_smem[vs_set_3 + 8];
                float vs1_1_13 = vs_smem[vs_set_3 + 8 + 1];
                acc_1[4] = acc_1[4] * (vs0_1_13 * rcp_4);
                acc_1[5] = acc_1[5] * (vs1_1_13 * rcp_4);
                acc_1[6] = acc_1[6] * (vs0_1_13 * rcp_1_3);
                acc_1[7] = acc_1[7] * (vs1_1_13 * rcp_1_3);
                float vs0_1_14 = vs_smem[vs_set_3 + 16];
                float vs1_1_14 = vs_smem[vs_set_3 + 16 + 1];
                acc_1[8] = acc_1[8] * (vs0_1_14 * rcp_4);
                acc_1[9] = acc_1[9] * (vs1_1_14 * rcp_4);
                acc_1[10] = acc_1[10] * (vs0_1_14 * rcp_1_3);
                acc_1[11] = acc_1[11] * (vs1_1_14 * rcp_1_3);
                float vs0_1_15 = vs_smem[vs_set_3 + 24];
                float vs1_1_15 = vs_smem[vs_set_3 + 24 + 1];
                acc_1[12] = acc_1[12] * (vs0_1_15 * rcp_4);
                acc_1[13] = acc_1[13] * (vs1_1_15 * rcp_4);
                acc_1[14] = acc_1[14] * (vs0_1_15 * rcp_1_3);
                acc_1[15] = acc_1[15] * (vs1_1_15 * rcp_1_3);
                int qj1_3 = qj & 1;
                int qj2_3 = qj & 2;
                unsigned int o_vec_3[4];
                unsigned int o_tmp_3[4];
                int o_row_base_3 = q_row * 128;
                int m_local_r_1_6 = m0_local;
                __nv_bfloat162 _bit_cast_src_416 = __float22bfloat162_rn(make_float2(acc_1[0], acc_1[1]));
                o_vec_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_416)[0];
                __nv_bfloat162 _bit_cast_src_417 = __float22bfloat162_rn(make_float2(acc_1[4], acc_1[5]));
                o_vec_3[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_417)[0];
                __nv_bfloat162 _bit_cast_src_418 = __float22bfloat162_rn(make_float2(acc_1[8], acc_1[9]));
                o_vec_3[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_418)[0];
                __nv_bfloat162 _bit_cast_src_419 = __float22bfloat162_rn(make_float2(acc_1[12], acc_1[13]));
                o_vec_3[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_419)[0];
                unsigned int _shfl_xor_sync_420 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 1);
                o_tmp_3[0] = _shfl_xor_sync_420;
                unsigned int _shfl_xor_sync_421 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 1);
                o_tmp_3[1] = _shfl_xor_sync_421;
                unsigned int _shfl_xor_sync_422 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 1);
                o_tmp_3[2] = _shfl_xor_sync_422;
                unsigned int _shfl_xor_sync_423 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 1);
                o_tmp_3[3] = _shfl_xor_sync_423;
                o_vec_3[0] = ((qj1_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                o_vec_3[1] = ((qj1_3 == 0) ? o_tmp_3[1] : o_vec_3[1]);
                o_vec_3[2] = ((qj1_3 != 0) ? o_tmp_3[2] : o_vec_3[2]);
                o_vec_3[3] = ((qj1_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                unsigned int _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 2);
                o_tmp_3[0] = _shfl_xor_sync_424;
                unsigned int _shfl_xor_sync_425 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 2);
                o_tmp_3[1] = _shfl_xor_sync_425;
                unsigned int _shfl_xor_sync_426 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 2);
                o_tmp_3[2] = _shfl_xor_sync_426;
                unsigned int _shfl_xor_sync_427 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 2);
                o_tmp_3[3] = _shfl_xor_sync_427;
                o_vec_3[0] = ((qj2_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                o_vec_3[1] = ((qj2_3 != 0) ? o_tmp_3[1] : o_vec_3[1]);
                o_vec_3[2] = ((qj2_3 == 0) ? o_tmp_3[2] : o_vec_3[2]);
                o_vec_3[3] = ((qj2_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                int o_off_1_6 = o_row_base_3 + m_local_r_1_6 * 128 + (4 * sset_1_3 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_6)[0] = reinterpret_cast<int4*>(o_vec_3)[0];
                int m_local_r_1_7 = m1_local;
                __nv_bfloat162 _bit_cast_src_428 = __float22bfloat162_rn(make_float2(acc_1[2], acc_1[3]));
                o_vec_3[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_428)[0];
                __nv_bfloat162 _bit_cast_src_429 = __float22bfloat162_rn(make_float2(acc_1[6], acc_1[7]));
                o_vec_3[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_429)[0];
                __nv_bfloat162 _bit_cast_src_430 = __float22bfloat162_rn(make_float2(acc_1[10], acc_1[11]));
                o_vec_3[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_430)[0];
                __nv_bfloat162 _bit_cast_src_431 = __float22bfloat162_rn(make_float2(acc_1[14], acc_1[15]));
                o_vec_3[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_431)[0];
                unsigned int _shfl_xor_sync_432 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 1);
                o_tmp_3[0] = _shfl_xor_sync_432;
                unsigned int _shfl_xor_sync_433 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 1);
                o_tmp_3[1] = _shfl_xor_sync_433;
                unsigned int _shfl_xor_sync_434 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 1);
                o_tmp_3[2] = _shfl_xor_sync_434;
                unsigned int _shfl_xor_sync_435 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 1);
                o_tmp_3[3] = _shfl_xor_sync_435;
                o_vec_3[0] = ((qj1_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                o_vec_3[1] = ((qj1_3 == 0) ? o_tmp_3[1] : o_vec_3[1]);
                o_vec_3[2] = ((qj1_3 != 0) ? o_tmp_3[2] : o_vec_3[2]);
                o_vec_3[3] = ((qj1_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                unsigned int _shfl_xor_sync_436 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[2], 2);
                o_tmp_3[0] = _shfl_xor_sync_436;
                unsigned int _shfl_xor_sync_437 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[3], 2);
                o_tmp_3[1] = _shfl_xor_sync_437;
                unsigned int _shfl_xor_sync_438 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[0], 2);
                o_tmp_3[2] = _shfl_xor_sync_438;
                unsigned int _shfl_xor_sync_439 = __shfl_xor_sync(0xFFFFFFFF, o_vec_3[1], 2);
                o_tmp_3[3] = _shfl_xor_sync_439;
                o_vec_3[0] = ((qj2_3 != 0) ? o_tmp_3[0] : o_vec_3[0]);
                o_vec_3[1] = ((qj2_3 != 0) ? o_tmp_3[1] : o_vec_3[1]);
                o_vec_3[2] = ((qj2_3 == 0) ? o_tmp_3[2] : o_vec_3[2]);
                o_vec_3[3] = ((qj2_3 == 0) ? o_tmp_3[3] : o_vec_3[3]);
                int o_off_1_7 = o_row_base_3 + m_local_r_1_7 * 128 + (4 * sset_1_3 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_7)[0] = reinterpret_cast<int4*>(o_vec_3)[0];
            }
        }
    }
}

} // extern "C"
