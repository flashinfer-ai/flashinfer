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
#define SMEM_MSTATS_OFF 59392
#define SMEM_MSTATS_STAGE_BYTES 1024
#define SMEM_MSTATS_STRIDE 1024
#define SMEM_PART_LO_OFF 25600
#define SMEM_PART_LO_STAGE_BYTES 16384
#define SMEM_PART_LO_STRIDE 16384
#define SMEM_PART_HI0_OFF 9216
#define SMEM_PART_HI0_STAGE_BYTES 16384
#define SMEM_PART_HI0_STRIDE 16384
#define SMEM_RECV_SMEM_OFF 60416
#define SMEM_RECV_SMEM_STAGE_BYTES 27648
#define SMEM_RECV_SMEM_STRIDE 27648
#define SMEM_VS_SMEM_OFF 58368
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
#define SMEM_VT_SMEM_OFF 41984
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 88064
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_vsa_sm90_438ea44a7e2aba916483(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ Qs, float* __restrict__ Ks, float* __restrict__ Vs, int k_scale_groups, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, int g_shift)
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
    #define v_full0_addr (mbar_base + 24)
    #define v_full1_addr (mbar_base + 32)
    #define recv_full_addr (mbar_base + 40)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* mstats = reinterpret_cast<float*>(smem_raw + 59392);
    const int mstats_addr = smem + 59392;
    float* part_lo = reinterpret_cast<float*>(smem_raw + 25600);
    const int part_lo_addr = smem + 25600;
    float* part_hi0 = reinterpret_cast<float*>(smem_raw + 9216);
    const int part_hi0_addr = smem + 9216;
    float* recv_smem = reinterpret_cast<float*>(smem_raw + 60416);
    const int recv_smem_addr = smem + 60416;
    float* vs_smem = reinterpret_cast<float*>(smem_raw + 58368);
    const int vs_smem_addr = smem + 58368;
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* k_smem = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int k_smem_addr = smem + 9216;
    uint8_t* vl_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int vl_smem_addr = smem + 17408;
    uint8_t* vt_smem = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int vt_smem_addr = smem + 41984;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // k_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // v_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // v_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // recv_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
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
    int plan_base = item * 4;
    int meta = plan[plan_base];
    int cnt = meta & 15;
    int tile = plan[plan_base + 1];
    int blk_base = plan_base + 2;
    int head = tile / mb;
    int qb = tile - head * mb;
    int q_row = head * seqlen_q + qb * 64;
    int kv_base = (head >> g_shift) * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            if (cta_rank < 4) {
                int nsets = 1;
                int recv_tx = 3 * (nsets * 8192 + 512);
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
    float ks_regs[4];
    int ks_head = (head >> g_shift) * k_scale_groups;
    if (lim_even > 0) {
        int ks_blk_1_0 = plan[blk_base];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk_1_0 * 4) + 0);
            ks_regs[0 + 0] = _v4.x;
            ks_regs[0 + 1] = _v4.y;
            ks_regs[0 + 2] = _v4.z;
            ks_regs[0 + 3] = _v4.w;
        }
    }
    if (lim_odd > 1) {
        int ks_blk_1_1 = plan[blk_base + 1];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk_1_1 * 4) + 0);
            ks_regs[0 + 0] = _v4.x;
            ks_regs[0 + 1] = _v4.y;
            ks_regs[0 + 2] = _v4.z;
            ks_regs[0 + 3] = _v4.w;
        }
    }
    vs_smem[vsl_two + tid_wg] = Vs[(head >> g_shift) * 128 + tid_wg];
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
    uint64_t _wgmma_desc_2 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    uint64_t _wgmma_desc_3 = (((uint64_t)(((k_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
    uint64_t _wgmma_desc_4 = (((uint64_t)(((vt_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    if (lim_even > 0) {
        mbarrier_wait(k_full0_addr, 0);
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
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
        m0 = fmaxf(m0, _shfl_xor_sync_5);
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
        m0 = fmaxf(m0, _shfl_xor_sync_6);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
        m1 = fmaxf(m1, _shfl_xor_sync_7);
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
        m1 = fmaxf(m1, _shfl_xor_sync_8);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_10;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_10) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_11;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_11) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_9;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_9) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_10), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_11));
        p_e4m3[0] = _prmt_9;
        uint16_t _cvt_satfinite_e4m3x2_f32_13;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_13) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_14;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_12;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_12) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_13), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_14));
        p_e4m3[1] = _prmt_12;
        uint16_t _cvt_satfinite_e4m3x2_f32_16;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_16) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_17;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_17) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_15;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_15) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_16), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_17));
        p_e4m3[2] = _prmt_15;
        uint16_t _cvt_satfinite_e4m3x2_f32_19;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_20;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_20) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_18;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_18) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_19), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_20));
        p_e4m3[3] = _prmt_18;
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
        uint32_t _prmt_21;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_21) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_21;
        uint32_t _prmt_22;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_22) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_22;
        uint32_t _prmt_23;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_23) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_23;
        uint32_t _prmt_24;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_24) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_24;
        uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_26;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_26) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_26;
        uint32_t _prmt_27;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_27) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_27;
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_28) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_28;
        uint32_t _prmt_29;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_29) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_29;
        uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_31;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_31) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_31;
        uint32_t _prmt_32;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_32) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_32;
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_33) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_33;
        uint32_t _prmt_34;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_34) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_34;
        uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_36;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_36) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_36;
        uint32_t _prmt_37;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_37) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_37;
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_38;
        uint32_t _prmt_39;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_39) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_39;
        uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_42;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_42) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_43;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_43) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_41;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_41) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_42), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_43));
        p_e4m3[4] = _prmt_41;
        uint16_t _cvt_satfinite_e4m3x2_f32_45;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_45) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_46;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_46) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_44;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_44) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_45), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_46));
        p_e4m3[5] = _prmt_44;
        uint16_t _cvt_satfinite_e4m3x2_f32_48;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_48) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_49;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_47;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_47) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_48), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_49));
        p_e4m3[6] = _prmt_47;
        uint16_t _cvt_satfinite_e4m3x2_f32_51;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_51) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_52;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_52) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_50;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_50) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_51), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_52));
        p_e4m3[7] = _prmt_50;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_53 = (((uint64_t)(((k_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_53 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_53);
    if (lim_odd > 1) {
        mbarrier_wait(k_full1_addr, 0);
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
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 2);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_54);
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 1);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_55);
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 2);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_56);
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 1);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_57);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_59;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_59) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_60;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_60) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_58;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_58) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_59), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_60));
        p_e4m3[0] = _prmt_58;
        uint16_t _cvt_satfinite_e4m3x2_f32_62;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_62) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_63;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_63) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_61;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_61) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_62), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_63));
        p_e4m3[1] = _prmt_61;
        uint16_t _cvt_satfinite_e4m3x2_f32_65;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_65) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_66;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_66) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_64;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_64) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_65), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_66));
        p_e4m3[2] = _prmt_64;
        uint16_t _cvt_satfinite_e4m3x2_f32_68;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_68) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_69;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_69) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_67;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_67) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_68), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_69));
        p_e4m3[3] = _prmt_67;
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
        uint32_t _prmt_70;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_70) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_70;
        uint32_t _prmt_71;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_71) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_71;
        uint32_t _prmt_72;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_72) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_72;
        uint32_t _prmt_73;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_73) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_73;
        uint32_t _stmatrix_addr_74 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_75;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_75) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_75;
        uint32_t _prmt_76;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_76) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_76;
        uint32_t _prmt_77;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_77) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_77;
        uint32_t _prmt_78;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_78) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_78;
        uint32_t _stmatrix_addr_79 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_79), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_80;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_80) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_80;
        uint32_t _prmt_81;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_81) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_81;
        uint32_t _prmt_82;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_82) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_82;
        uint32_t _prmt_83;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_83) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_83;
        uint32_t _stmatrix_addr_84 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_84), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_85;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_85) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_85;
        uint32_t _prmt_86;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_86) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_86;
        uint32_t _prmt_87;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_87) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_87;
        uint32_t _prmt_88;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_88) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_88;
        uint32_t _stmatrix_addr_89 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_89), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_91;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_91) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_92;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_92) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_90;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_90) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_91), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_92));
        p_e4m3[4] = _prmt_90;
        uint16_t _cvt_satfinite_e4m3x2_f32_94;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_94) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_95;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_95) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_93;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_93) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_94), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_95));
        p_e4m3[5] = _prmt_93;
        uint16_t _cvt_satfinite_e4m3x2_f32_97;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_97) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_98;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_98) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_96;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_96) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_97), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_98));
        p_e4m3[6] = _prmt_96;
        uint16_t _cvt_satfinite_e4m3x2_f32_100;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_100) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_101;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_101) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_99;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_99) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_100), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_101));
        p_e4m3[7] = _prmt_99;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_sync_102;
    float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_sync_103;
    float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_sync_104;
    float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_sync_105;
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
    float _exp2_106 = approx_exp2(x0m0 - mm0);
    float f00 = ((x0m0 == -CUDART_INF_F) ? 0.0f : _exp2_106);
    float _exp2_107 = approx_exp2(x0m1 - mm1);
    float f01 = ((x0m1 == -CUDART_INF_F) ? 0.0f : _exp2_107);
    float _exp2_108 = approx_exp2(x1m0 - mm0);
    float f10 = ((x1m0 == -CUDART_INF_F) ? 0.0f : _exp2_108);
    float _exp2_109 = approx_exp2(x1m1 - mm1);
    float f11 = ((x1m1 == -CUDART_INF_F) ? 0.0f : _exp2_109);
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
                : "=r"(_mapa_0) : "r"(recv_smem_addr + (unsigned int)(pslot * 9216) + (unsigned int)my_off), "r"(0));
            uint32_t _mbar_peer_110;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_110) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0), "r"(__float_as_uint(d_o[0])), "r"(__float_as_uint(d_o[1])), "r"(__float_as_uint(d_o[2])), "r"(__float_as_uint(d_o[3])), "r"(_mbar_peer_110) : "memory");
            uint32_t _mbar_peer_111;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_111) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(d_o[4])), "r"(__float_as_uint(d_o[5])), "r"(__float_as_uint(d_o[6])), "r"(__float_as_uint(d_o[7])), "r"(_mbar_peer_111) : "memory");
            uint32_t _mbar_peer_112;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_112) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(d_o[8])), "r"(__float_as_uint(d_o[9])), "r"(__float_as_uint(d_o[10])), "r"(__float_as_uint(d_o[11])), "r"(_mbar_peer_112) : "memory");
            uint32_t _mbar_peer_113;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_113) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(d_o[12])), "r"(__float_as_uint(d_o[13])), "r"(__float_as_uint(d_o[14])), "r"(__float_as_uint(d_o[15])), "r"(_mbar_peer_113) : "memory");
            if (qj == 0) {
                uint32_t _cluster_mapa_114;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_114) : "r"(recv_smem_addr + (unsigned int)(pslot * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(0));
                uint32_t _mbar_peer_115;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_115) : "r"(recv_full_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_114), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_115) : "memory");
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
                : "=r"(_mapa_1) : "r"(recv_smem_addr + (unsigned int)(pslot_1 * 9216) + (unsigned int)my_off), "r"(1));
            uint32_t _mbar_peer_116;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_116) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1), "r"(__float_as_uint(d_o[16])), "r"(__float_as_uint(d_o[17])), "r"(__float_as_uint(d_o[18])), "r"(__float_as_uint(d_o[19])), "r"(_mbar_peer_116) : "memory");
            uint32_t _mbar_peer_117;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_117) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(d_o[20])), "r"(__float_as_uint(d_o[21])), "r"(__float_as_uint(d_o[22])), "r"(__float_as_uint(d_o[23])), "r"(_mbar_peer_117) : "memory");
            uint32_t _mbar_peer_118;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_118) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(d_o[24])), "r"(__float_as_uint(d_o[25])), "r"(__float_as_uint(d_o[26])), "r"(__float_as_uint(d_o[27])), "r"(_mbar_peer_118) : "memory");
            uint32_t _mbar_peer_119;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_119) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(d_o[28])), "r"(__float_as_uint(d_o[29])), "r"(__float_as_uint(d_o[30])), "r"(__float_as_uint(d_o[31])), "r"(_mbar_peer_119) : "memory");
            if (qj == 0) {
                uint32_t _cluster_mapa_120;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_120) : "r"(recv_smem_addr + (unsigned int)(pslot_1 * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(1));
                uint32_t _mbar_peer_121;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_121) : "r"(recv_full_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_120), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_121) : "memory");
            }
        }
        if (my_rank != 2) {
            int pslot_2 = my_rank;
            if (my_rank > 2) {
                pslot_2 = my_rank - 1;
            }
            if (qj == 0) {
                uint32_t _cluster_mapa_122;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_122) : "r"(recv_smem_addr + (unsigned int)(pslot_2 * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(2));
                uint32_t _mbar_peer_123;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_123) : "r"(recv_full_addr), "r"(2));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_122), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_123) : "memory");
            }
        }
        if (my_rank != 3) {
            int pslot_3 = my_rank;
            if (my_rank > 3) {
                pslot_3 = my_rank - 1;
            }
            if (qj == 0) {
                uint32_t _cluster_mapa_124;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_124) : "r"(recv_smem_addr + (unsigned int)(pslot_3 * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(3));
                uint32_t _mbar_peer_125;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_125) : "r"(recv_full_addr), "r"(3));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_124), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_125) : "memory");
            }
        }
        if (my_rank < 4) {
            mbarrier_wait_cluster_hint(recv_full_addr, 0, 10000000);
            float merged_max0 = row_max0;
            float merged_max1 = row_max1;
            merged_max0 = fmaxf(merged_max0, recv_smem[2048 + m0_local * 4]);
            merged_max1 = fmaxf(merged_max1, recv_smem[2048 + m0_local * 4 + 2]);
            merged_max0 = fmaxf(merged_max0, recv_smem[4352 + m0_local * 4]);
            merged_max1 = fmaxf(merged_max1, recv_smem[4352 + m0_local * 4 + 2]);
            merged_max0 = fmaxf(merged_max0, recv_smem[6656 + m0_local * 4]);
            merged_max1 = fmaxf(merged_max1, recv_smem[6656 + m0_local * 4 + 2]);
            float exp2_17 = approx_exp2(row_max0 - merged_max0);
            float exp2_1_2 = approx_exp2(row_max1 - merged_max1);
            float msum0 = row_sum0 * exp2_17;
            float msum1 = row_sum1 * exp2_1_2;
            float w_peer0[3];
            float w_peer1[3];
            int sbase_2_0 = 2048 + m0_local * 4;
            float _exp2_126 = approx_exp2(recv_smem[sbase_2_0] - merged_max0);
            w_peer0[0] = _exp2_126;
            float _exp2_127 = approx_exp2(recv_smem[sbase_2_0 + 2] - merged_max1);
            w_peer1[0] = _exp2_127;
            msum0 = __fmaf_rn(recv_smem[sbase_2_0 + 1], w_peer0[0], msum0);
            msum1 = __fmaf_rn(recv_smem[sbase_2_0 + 3], w_peer1[0], msum1);
            int sbase_2_1 = 4352 + m0_local * 4;
            float _exp2_128 = approx_exp2(recv_smem[sbase_2_1] - merged_max0);
            w_peer0[1] = _exp2_128;
            float _exp2_129 = approx_exp2(recv_smem[sbase_2_1 + 2] - merged_max1);
            w_peer1[1] = _exp2_129;
            msum0 = __fmaf_rn(recv_smem[sbase_2_1 + 1], w_peer0[1], msum0);
            msum1 = __fmaf_rn(recv_smem[sbase_2_1 + 3], w_peer1[1], msum1);
            int sbase_2_2 = 6656 + m0_local * 4;
            float _exp2_130 = approx_exp2(recv_smem[sbase_2_2] - merged_max0);
            w_peer0[2] = _exp2_130;
            float _exp2_131 = approx_exp2(recv_smem[sbase_2_2 + 2] - merged_max1);
            w_peer1[2] = _exp2_131;
            msum0 = __fmaf_rn(recv_smem[sbase_2_2 + 1], w_peer0[2], msum0);
            msum1 = __fmaf_rn(recv_smem[sbase_2_2 + 3], w_peer1[2], msum1);
            int my_tid4 = tid_wg * 4;
            float acc[16];
            int sset = my_rank;
            int fold = ((sset < 4) ? 1 : 0);
            if (sset < 0) {
                fold = 0;
            }
            if (sset >= 2) {
                fold = 0;
            }
            if (fold != 0) {
                if (sset == 0) {
                    acc[0] = d_o[0] * exp2_17;
                    acc[1] = d_o[1] * exp2_17;
                    acc[2] = d_o[2] * exp2_1_2;
                    acc[3] = d_o[3] * exp2_1_2;
                    acc[4] = d_o[4] * exp2_17;
                    acc[5] = d_o[5] * exp2_17;
                    acc[6] = d_o[6] * exp2_1_2;
                    acc[7] = d_o[7] * exp2_1_2;
                    acc[8] = d_o[8] * exp2_17;
                    acc[9] = d_o[9] * exp2_17;
                    acc[10] = d_o[10] * exp2_1_2;
                    acc[11] = d_o[11] * exp2_1_2;
                    acc[12] = d_o[12] * exp2_17;
                    acc[13] = d_o[13] * exp2_17;
                    acc[14] = d_o[14] * exp2_1_2;
                    acc[15] = d_o[15] * exp2_1_2;
                }
                if (sset == 1) {
                    acc[0] = d_o[16] * exp2_17;
                    acc[1] = d_o[17] * exp2_17;
                    acc[2] = d_o[18] * exp2_1_2;
                    acc[3] = d_o[19] * exp2_1_2;
                    acc[4] = d_o[20] * exp2_17;
                    acc[5] = d_o[21] * exp2_17;
                    acc[6] = d_o[22] * exp2_1_2;
                    acc[7] = d_o[23] * exp2_1_2;
                    acc[8] = d_o[24] * exp2_17;
                    acc[9] = d_o[25] * exp2_17;
                    acc[10] = d_o[26] * exp2_1_2;
                    acc[11] = d_o[27] * exp2_1_2;
                    acc[12] = d_o[28] * exp2_17;
                    acc[13] = d_o[29] * exp2_17;
                    acc[14] = d_o[30] * exp2_1_2;
                    acc[15] = d_o[31] * exp2_1_2;
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
                int base_0 = 2304 + my_tid4;
                acc[0] = __fmaf_rn(recv_smem[base_0], w_peer0[1], acc[0]);
                acc[1] = __fmaf_rn(recv_smem[base_0 + 1], w_peer0[1], acc[1]);
                acc[2] = __fmaf_rn(recv_smem[base_0 + 2], w_peer1[1], acc[2]);
                acc[3] = __fmaf_rn(recv_smem[base_0 + 3], w_peer1[1], acc[3]);
                acc[4] = __fmaf_rn(recv_smem[base_0 + 512], w_peer0[1], acc[4]);
                acc[5] = __fmaf_rn(recv_smem[base_0 + 512 + 1], w_peer0[1], acc[5]);
                acc[6] = __fmaf_rn(recv_smem[base_0 + 512 + 2], w_peer1[1], acc[6]);
                acc[7] = __fmaf_rn(recv_smem[base_0 + 512 + 3], w_peer1[1], acc[7]);
                acc[8] = __fmaf_rn(recv_smem[base_0 + 1024], w_peer0[1], acc[8]);
                acc[9] = __fmaf_rn(recv_smem[base_0 + 1024 + 1], w_peer0[1], acc[9]);
                acc[10] = __fmaf_rn(recv_smem[base_0 + 1024 + 2], w_peer1[1], acc[10]);
                acc[11] = __fmaf_rn(recv_smem[base_0 + 1024 + 3], w_peer1[1], acc[11]);
                acc[12] = __fmaf_rn(recv_smem[base_0 + 1536], w_peer0[1], acc[12]);
                acc[13] = __fmaf_rn(recv_smem[base_0 + 1536 + 1], w_peer0[1], acc[13]);
                acc[14] = __fmaf_rn(recv_smem[base_0 + 1536 + 2], w_peer1[1], acc[14]);
                acc[15] = __fmaf_rn(recv_smem[base_0 + 1536 + 3], w_peer1[1], acc[15]);
                int base_1 = 4608 + my_tid4;
                acc[0] = __fmaf_rn(recv_smem[base_1], w_peer0[2], acc[0]);
                acc[1] = __fmaf_rn(recv_smem[base_1 + 1], w_peer0[2], acc[1]);
                acc[2] = __fmaf_rn(recv_smem[base_1 + 2], w_peer1[2], acc[2]);
                acc[3] = __fmaf_rn(recv_smem[base_1 + 3], w_peer1[2], acc[3]);
                acc[4] = __fmaf_rn(recv_smem[base_1 + 512], w_peer0[2], acc[4]);
                acc[5] = __fmaf_rn(recv_smem[base_1 + 512 + 1], w_peer0[2], acc[5]);
                acc[6] = __fmaf_rn(recv_smem[base_1 + 512 + 2], w_peer1[2], acc[6]);
                acc[7] = __fmaf_rn(recv_smem[base_1 + 512 + 3], w_peer1[2], acc[7]);
                acc[8] = __fmaf_rn(recv_smem[base_1 + 1024], w_peer0[2], acc[8]);
                acc[9] = __fmaf_rn(recv_smem[base_1 + 1024 + 1], w_peer0[2], acc[9]);
                acc[10] = __fmaf_rn(recv_smem[base_1 + 1024 + 2], w_peer1[2], acc[10]);
                acc[11] = __fmaf_rn(recv_smem[base_1 + 1024 + 3], w_peer1[2], acc[11]);
                acc[12] = __fmaf_rn(recv_smem[base_1 + 1536], w_peer0[2], acc[12]);
                acc[13] = __fmaf_rn(recv_smem[base_1 + 1536 + 1], w_peer0[2], acc[13]);
                acc[14] = __fmaf_rn(recv_smem[base_1 + 1536 + 2], w_peer1[2], acc[14]);
                acc[15] = __fmaf_rn(recv_smem[base_1 + 1536 + 3], w_peer1[2], acc[15]);
                float rcp = approx_rcp(msum0);
                float rcp_1 = approx_rcp(msum1);
                int vs_set = vs_base + sset * 32;
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
                __nv_bfloat162 _bit_cast_src_132 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_132)[0];
                __nv_bfloat162 _bit_cast_src_133 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_133)[0];
                __nv_bfloat162 _bit_cast_src_134 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_134)[0];
                __nv_bfloat162 _bit_cast_src_135 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_135)[0];
                unsigned int _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                o_tmp[0] = _shfl_xor_sync_136;
                unsigned int _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                o_tmp[1] = _shfl_xor_sync_137;
                unsigned int _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                o_tmp[2] = _shfl_xor_sync_138;
                unsigned int _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                o_tmp[3] = _shfl_xor_sync_139;
                o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                unsigned int _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                o_tmp[0] = _shfl_xor_sync_140;
                unsigned int _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                o_tmp[1] = _shfl_xor_sync_141;
                unsigned int _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                o_tmp[2] = _shfl_xor_sync_142;
                unsigned int _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                o_tmp[3] = _shfl_xor_sync_143;
                o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                int o_off_1_0 = o_row_base + m_local_r_1_0 * 128 + (4 * sset + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_0)[0] = reinterpret_cast<int4*>(o_vec)[0];
                int m_local_r_1_1 = m1_local;
                __nv_bfloat162 _bit_cast_src_144 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_144)[0];
                __nv_bfloat162 _bit_cast_src_145 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_145)[0];
                __nv_bfloat162 _bit_cast_src_146 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_146)[0];
                __nv_bfloat162 _bit_cast_src_147 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_147)[0];
                unsigned int _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                o_tmp[0] = _shfl_xor_sync_148;
                unsigned int _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                o_tmp[1] = _shfl_xor_sync_149;
                unsigned int _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                o_tmp[2] = _shfl_xor_sync_150;
                unsigned int _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                o_tmp[3] = _shfl_xor_sync_151;
                o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                unsigned int _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                o_tmp[0] = _shfl_xor_sync_152;
                unsigned int _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                o_tmp[1] = _shfl_xor_sync_153;
                unsigned int _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                o_tmp[2] = _shfl_xor_sync_154;
                unsigned int _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                o_tmp[3] = _shfl_xor_sync_155;
                o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                int o_off_1_1 = o_row_base + m_local_r_1_1 * 128 + (4 * sset + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_1)[0] = reinterpret_cast<int4*>(o_vec)[0];
            }
        }
    }
    if (wg == 1) {
        int my_rank_1 = cta_rank;
        int my_off_1 = tid_wg * 16;
        if (my_rank_1 != 0) {
            int pslot_4 = my_rank_1;
            if (my_rank_1 > 0) {
                pslot_4 = my_rank_1 - 1;
            }
        }
        if (my_rank_1 != 1) {
            int pslot_5 = my_rank_1;
            if (my_rank_1 > 1) {
                pslot_5 = my_rank_1 - 1;
            }
        }
        if (my_rank_1 != 2) {
            int pslot_6 = my_rank_1;
            if (my_rank_1 > 2) {
                pslot_6 = my_rank_1 - 1;
            }
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(recv_smem_addr + (unsigned int)(pslot_6 * 9216) + (unsigned int)my_off_1), "r"(2));
            uint32_t _mbar_peer_156;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_156) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2), "r"(__float_as_uint(d_o[32])), "r"(__float_as_uint(d_o[33])), "r"(__float_as_uint(d_o[34])), "r"(__float_as_uint(d_o[35])), "r"(_mbar_peer_156) : "memory");
            uint32_t _mbar_peer_157;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_157) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 2048), "r"(__float_as_uint(d_o[36])), "r"(__float_as_uint(d_o[37])), "r"(__float_as_uint(d_o[38])), "r"(__float_as_uint(d_o[39])), "r"(_mbar_peer_157) : "memory");
            uint32_t _mbar_peer_158;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_158) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 4096), "r"(__float_as_uint(d_o[40])), "r"(__float_as_uint(d_o[41])), "r"(__float_as_uint(d_o[42])), "r"(__float_as_uint(d_o[43])), "r"(_mbar_peer_158) : "memory");
            uint32_t _mbar_peer_159;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_159) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 6144), "r"(__float_as_uint(d_o[44])), "r"(__float_as_uint(d_o[45])), "r"(__float_as_uint(d_o[46])), "r"(__float_as_uint(d_o[47])), "r"(_mbar_peer_159) : "memory");
        }
        if (my_rank_1 != 3) {
            int pslot_7 = my_rank_1;
            if (my_rank_1 > 3) {
                pslot_7 = my_rank_1 - 1;
            }
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(recv_smem_addr + (unsigned int)(pslot_7 * 9216) + (unsigned int)my_off_1), "r"(3));
            uint32_t _mbar_peer_160;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_160) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3), "r"(__float_as_uint(d_o[48])), "r"(__float_as_uint(d_o[49])), "r"(__float_as_uint(d_o[50])), "r"(__float_as_uint(d_o[51])), "r"(_mbar_peer_160) : "memory");
            uint32_t _mbar_peer_161;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_161) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 2048), "r"(__float_as_uint(d_o[52])), "r"(__float_as_uint(d_o[53])), "r"(__float_as_uint(d_o[54])), "r"(__float_as_uint(d_o[55])), "r"(_mbar_peer_161) : "memory");
            uint32_t _mbar_peer_162;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_162) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 4096), "r"(__float_as_uint(d_o[56])), "r"(__float_as_uint(d_o[57])), "r"(__float_as_uint(d_o[58])), "r"(__float_as_uint(d_o[59])), "r"(_mbar_peer_162) : "memory");
            uint32_t _mbar_peer_163;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_163) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 6144), "r"(__float_as_uint(d_o[60])), "r"(__float_as_uint(d_o[61])), "r"(__float_as_uint(d_o[62])), "r"(__float_as_uint(d_o[63])), "r"(_mbar_peer_163) : "memory");
        }
        if (my_rank_1 < 4) {
            mbarrier_wait_cluster_hint(recv_full_addr, 0, 10000000);
            float merged_max0_1 = row_max0;
            float merged_max1_1 = row_max1;
            merged_max0_1 = fmaxf(merged_max0_1, recv_smem[2048 + m0_local * 4]);
            merged_max1_1 = fmaxf(merged_max1_1, recv_smem[2048 + m0_local * 4 + 2]);
            merged_max0_1 = fmaxf(merged_max0_1, recv_smem[4352 + m0_local * 4]);
            merged_max1_1 = fmaxf(merged_max1_1, recv_smem[4352 + m0_local * 4 + 2]);
            merged_max0_1 = fmaxf(merged_max0_1, recv_smem[6656 + m0_local * 4]);
            merged_max1_1 = fmaxf(merged_max1_1, recv_smem[6656 + m0_local * 4 + 2]);
            float exp2_18 = approx_exp2(row_max0 - merged_max0_1);
            float exp2_1_3 = approx_exp2(row_max1 - merged_max1_1);
            float msum0_1 = row_sum0 * exp2_18;
            float msum1_1 = row_sum1 * exp2_1_3;
            float w_peer0_1[3];
            float w_peer1_1[3];
            int sbase_2_3 = 2048 + m0_local * 4;
            float _exp2_164 = approx_exp2(recv_smem[sbase_2_3] - merged_max0_1);
            w_peer0_1[0] = _exp2_164;
            float _exp2_165 = approx_exp2(recv_smem[sbase_2_3 + 2] - merged_max1_1);
            w_peer1_1[0] = _exp2_165;
            msum0_1 = __fmaf_rn(recv_smem[sbase_2_3 + 1], w_peer0_1[0], msum0_1);
            msum1_1 = __fmaf_rn(recv_smem[sbase_2_3 + 3], w_peer1_1[0], msum1_1);
            int sbase_2_4 = 4352 + m0_local * 4;
            float _exp2_166 = approx_exp2(recv_smem[sbase_2_4] - merged_max0_1);
            w_peer0_1[1] = _exp2_166;
            float _exp2_167 = approx_exp2(recv_smem[sbase_2_4 + 2] - merged_max1_1);
            w_peer1_1[1] = _exp2_167;
            msum0_1 = __fmaf_rn(recv_smem[sbase_2_4 + 1], w_peer0_1[1], msum0_1);
            msum1_1 = __fmaf_rn(recv_smem[sbase_2_4 + 3], w_peer1_1[1], msum1_1);
            int sbase_2_5 = 6656 + m0_local * 4;
            float _exp2_168 = approx_exp2(recv_smem[sbase_2_5] - merged_max0_1);
            w_peer0_1[2] = _exp2_168;
            float _exp2_169 = approx_exp2(recv_smem[sbase_2_5 + 2] - merged_max1_1);
            w_peer1_1[2] = _exp2_169;
            msum0_1 = __fmaf_rn(recv_smem[sbase_2_5 + 1], w_peer0_1[2], msum0_1);
            msum1_1 = __fmaf_rn(recv_smem[sbase_2_5 + 3], w_peer1_1[2], msum1_1);
            int my_tid4_1 = tid_wg * 4;
            float acc_1[16];
            int sset_1 = my_rank_1;
            int fold_1 = ((sset_1 < 4) ? 1 : 0);
            if (sset_1 < 2) {
                fold_1 = 0;
            }
            if (sset_1 >= 4) {
                fold_1 = 0;
            }
            if (fold_1 != 0) {
                if (sset_1 == 2) {
                    acc_1[0] = d_o[32] * exp2_18;
                    acc_1[1] = d_o[33] * exp2_18;
                    acc_1[2] = d_o[34] * exp2_1_3;
                    acc_1[3] = d_o[35] * exp2_1_3;
                    acc_1[4] = d_o[36] * exp2_18;
                    acc_1[5] = d_o[37] * exp2_18;
                    acc_1[6] = d_o[38] * exp2_1_3;
                    acc_1[7] = d_o[39] * exp2_1_3;
                    acc_1[8] = d_o[40] * exp2_18;
                    acc_1[9] = d_o[41] * exp2_18;
                    acc_1[10] = d_o[42] * exp2_1_3;
                    acc_1[11] = d_o[43] * exp2_1_3;
                    acc_1[12] = d_o[44] * exp2_18;
                    acc_1[13] = d_o[45] * exp2_18;
                    acc_1[14] = d_o[46] * exp2_1_3;
                    acc_1[15] = d_o[47] * exp2_1_3;
                }
                if (sset_1 == 3) {
                    acc_1[0] = d_o[48] * exp2_18;
                    acc_1[1] = d_o[49] * exp2_18;
                    acc_1[2] = d_o[50] * exp2_1_3;
                    acc_1[3] = d_o[51] * exp2_1_3;
                    acc_1[4] = d_o[52] * exp2_18;
                    acc_1[5] = d_o[53] * exp2_18;
                    acc_1[6] = d_o[54] * exp2_1_3;
                    acc_1[7] = d_o[55] * exp2_1_3;
                    acc_1[8] = d_o[56] * exp2_18;
                    acc_1[9] = d_o[57] * exp2_18;
                    acc_1[10] = d_o[58] * exp2_1_3;
                    acc_1[11] = d_o[59] * exp2_1_3;
                    acc_1[12] = d_o[60] * exp2_18;
                    acc_1[13] = d_o[61] * exp2_18;
                    acc_1[14] = d_o[62] * exp2_1_3;
                    acc_1[15] = d_o[63] * exp2_1_3;
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
                int base_0_1 = 2304 + my_tid4_1;
                acc_1[0] = __fmaf_rn(recv_smem[base_0_1], w_peer0_1[1], acc_1[0]);
                acc_1[1] = __fmaf_rn(recv_smem[base_0_1 + 1], w_peer0_1[1], acc_1[1]);
                acc_1[2] = __fmaf_rn(recv_smem[base_0_1 + 2], w_peer1_1[1], acc_1[2]);
                acc_1[3] = __fmaf_rn(recv_smem[base_0_1 + 3], w_peer1_1[1], acc_1[3]);
                acc_1[4] = __fmaf_rn(recv_smem[base_0_1 + 512], w_peer0_1[1], acc_1[4]);
                acc_1[5] = __fmaf_rn(recv_smem[base_0_1 + 512 + 1], w_peer0_1[1], acc_1[5]);
                acc_1[6] = __fmaf_rn(recv_smem[base_0_1 + 512 + 2], w_peer1_1[1], acc_1[6]);
                acc_1[7] = __fmaf_rn(recv_smem[base_0_1 + 512 + 3], w_peer1_1[1], acc_1[7]);
                acc_1[8] = __fmaf_rn(recv_smem[base_0_1 + 1024], w_peer0_1[1], acc_1[8]);
                acc_1[9] = __fmaf_rn(recv_smem[base_0_1 + 1024 + 1], w_peer0_1[1], acc_1[9]);
                acc_1[10] = __fmaf_rn(recv_smem[base_0_1 + 1024 + 2], w_peer1_1[1], acc_1[10]);
                acc_1[11] = __fmaf_rn(recv_smem[base_0_1 + 1024 + 3], w_peer1_1[1], acc_1[11]);
                acc_1[12] = __fmaf_rn(recv_smem[base_0_1 + 1536], w_peer0_1[1], acc_1[12]);
                acc_1[13] = __fmaf_rn(recv_smem[base_0_1 + 1536 + 1], w_peer0_1[1], acc_1[13]);
                acc_1[14] = __fmaf_rn(recv_smem[base_0_1 + 1536 + 2], w_peer1_1[1], acc_1[14]);
                acc_1[15] = __fmaf_rn(recv_smem[base_0_1 + 1536 + 3], w_peer1_1[1], acc_1[15]);
                int base_1_1 = 4608 + my_tid4_1;
                acc_1[0] = __fmaf_rn(recv_smem[base_1_1], w_peer0_1[2], acc_1[0]);
                acc_1[1] = __fmaf_rn(recv_smem[base_1_1 + 1], w_peer0_1[2], acc_1[1]);
                acc_1[2] = __fmaf_rn(recv_smem[base_1_1 + 2], w_peer1_1[2], acc_1[2]);
                acc_1[3] = __fmaf_rn(recv_smem[base_1_1 + 3], w_peer1_1[2], acc_1[3]);
                acc_1[4] = __fmaf_rn(recv_smem[base_1_1 + 512], w_peer0_1[2], acc_1[4]);
                acc_1[5] = __fmaf_rn(recv_smem[base_1_1 + 512 + 1], w_peer0_1[2], acc_1[5]);
                acc_1[6] = __fmaf_rn(recv_smem[base_1_1 + 512 + 2], w_peer1_1[2], acc_1[6]);
                acc_1[7] = __fmaf_rn(recv_smem[base_1_1 + 512 + 3], w_peer1_1[2], acc_1[7]);
                acc_1[8] = __fmaf_rn(recv_smem[base_1_1 + 1024], w_peer0_1[2], acc_1[8]);
                acc_1[9] = __fmaf_rn(recv_smem[base_1_1 + 1024 + 1], w_peer0_1[2], acc_1[9]);
                acc_1[10] = __fmaf_rn(recv_smem[base_1_1 + 1024 + 2], w_peer1_1[2], acc_1[10]);
                acc_1[11] = __fmaf_rn(recv_smem[base_1_1 + 1024 + 3], w_peer1_1[2], acc_1[11]);
                acc_1[12] = __fmaf_rn(recv_smem[base_1_1 + 1536], w_peer0_1[2], acc_1[12]);
                acc_1[13] = __fmaf_rn(recv_smem[base_1_1 + 1536 + 1], w_peer0_1[2], acc_1[13]);
                acc_1[14] = __fmaf_rn(recv_smem[base_1_1 + 1536 + 2], w_peer1_1[2], acc_1[14]);
                acc_1[15] = __fmaf_rn(recv_smem[base_1_1 + 1536 + 3], w_peer1_1[2], acc_1[15]);
                float rcp_2 = approx_rcp(msum0_1);
                float rcp_1_1 = approx_rcp(msum1_1);
                int vs_set_1 = vs_base + sset_1 * 32;
                float vs0_1_4 = vs_smem[vs_set_1];
                float vs1_1_4 = vs_smem[vs_set_1 + 1];
                acc_1[0] = acc_1[0] * (vs0_1_4 * rcp_2);
                acc_1[1] = acc_1[1] * (vs1_1_4 * rcp_2);
                acc_1[2] = acc_1[2] * (vs0_1_4 * rcp_1_1);
                acc_1[3] = acc_1[3] * (vs1_1_4 * rcp_1_1);
                float vs0_1_5 = vs_smem[vs_set_1 + 8];
                float vs1_1_5 = vs_smem[vs_set_1 + 8 + 1];
                acc_1[4] = acc_1[4] * (vs0_1_5 * rcp_2);
                acc_1[5] = acc_1[5] * (vs1_1_5 * rcp_2);
                acc_1[6] = acc_1[6] * (vs0_1_5 * rcp_1_1);
                acc_1[7] = acc_1[7] * (vs1_1_5 * rcp_1_1);
                float vs0_1_6 = vs_smem[vs_set_1 + 16];
                float vs1_1_6 = vs_smem[vs_set_1 + 16 + 1];
                acc_1[8] = acc_1[8] * (vs0_1_6 * rcp_2);
                acc_1[9] = acc_1[9] * (vs1_1_6 * rcp_2);
                acc_1[10] = acc_1[10] * (vs0_1_6 * rcp_1_1);
                acc_1[11] = acc_1[11] * (vs1_1_6 * rcp_1_1);
                float vs0_1_7 = vs_smem[vs_set_1 + 24];
                float vs1_1_7 = vs_smem[vs_set_1 + 24 + 1];
                acc_1[12] = acc_1[12] * (vs0_1_7 * rcp_2);
                acc_1[13] = acc_1[13] * (vs1_1_7 * rcp_2);
                acc_1[14] = acc_1[14] * (vs0_1_7 * rcp_1_1);
                acc_1[15] = acc_1[15] * (vs1_1_7 * rcp_1_1);
                int qj1_1 = qj & 1;
                int qj2_1 = qj & 2;
                unsigned int o_vec_1[4];
                unsigned int o_tmp_1[4];
                int o_row_base_1 = q_row * 128;
                int m_local_r_1_2 = m0_local;
                __nv_bfloat162 _bit_cast_src_170 = __float22bfloat162_rn(make_float2(acc_1[0], acc_1[1]));
                o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_170)[0];
                __nv_bfloat162 _bit_cast_src_171 = __float22bfloat162_rn(make_float2(acc_1[4], acc_1[5]));
                o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_171)[0];
                __nv_bfloat162 _bit_cast_src_172 = __float22bfloat162_rn(make_float2(acc_1[8], acc_1[9]));
                o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_172)[0];
                __nv_bfloat162 _bit_cast_src_173 = __float22bfloat162_rn(make_float2(acc_1[12], acc_1[13]));
                o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_173)[0];
                unsigned int _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                o_tmp_1[0] = _shfl_xor_sync_174;
                unsigned int _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                o_tmp_1[1] = _shfl_xor_sync_175;
                unsigned int _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                o_tmp_1[2] = _shfl_xor_sync_176;
                unsigned int _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                o_tmp_1[3] = _shfl_xor_sync_177;
                o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                unsigned int _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                o_tmp_1[0] = _shfl_xor_sync_178;
                unsigned int _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                o_tmp_1[1] = _shfl_xor_sync_179;
                unsigned int _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                o_tmp_1[2] = _shfl_xor_sync_180;
                unsigned int _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                o_tmp_1[3] = _shfl_xor_sync_181;
                o_vec_1[0] = ((qj2_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj2_1 != 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj2_1 == 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj2_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                int o_off_1_2 = o_row_base_1 + m_local_r_1_2 * 128 + (4 * sset_1 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_2)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
                int m_local_r_1_3 = m1_local;
                __nv_bfloat162 _bit_cast_src_182 = __float22bfloat162_rn(make_float2(acc_1[2], acc_1[3]));
                o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_182)[0];
                __nv_bfloat162 _bit_cast_src_183 = __float22bfloat162_rn(make_float2(acc_1[6], acc_1[7]));
                o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_183)[0];
                __nv_bfloat162 _bit_cast_src_184 = __float22bfloat162_rn(make_float2(acc_1[10], acc_1[11]));
                o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_184)[0];
                __nv_bfloat162 _bit_cast_src_185 = __float22bfloat162_rn(make_float2(acc_1[14], acc_1[15]));
                o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_185)[0];
                unsigned int _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                o_tmp_1[0] = _shfl_xor_sync_186;
                unsigned int _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                o_tmp_1[1] = _shfl_xor_sync_187;
                unsigned int _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                o_tmp_1[2] = _shfl_xor_sync_188;
                unsigned int _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                o_tmp_1[3] = _shfl_xor_sync_189;
                o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                unsigned int _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                o_tmp_1[0] = _shfl_xor_sync_190;
                unsigned int _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                o_tmp_1[1] = _shfl_xor_sync_191;
                unsigned int _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                o_tmp_1[2] = _shfl_xor_sync_192;
                unsigned int _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                o_tmp_1[3] = _shfl_xor_sync_193;
                o_vec_1[0] = ((qj2_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj2_1 != 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj2_1 == 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj2_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                int o_off_1_3 = o_row_base_1 + m_local_r_1_3 * 128 + (4 * sset_1 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_3)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            }
        }
    }
}

} // extern "C"
