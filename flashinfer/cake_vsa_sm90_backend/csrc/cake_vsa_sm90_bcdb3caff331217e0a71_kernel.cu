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
#define SMEM_MSTATS_OFF 92160
#define SMEM_MSTATS_STAGE_BYTES 1024
#define SMEM_MSTATS_STRIDE 1024
#define SMEM_PART_LO_OFF 25600
#define SMEM_PART_LO_STAGE_BYTES 16384
#define SMEM_PART_LO_STRIDE 16384
#define SMEM_PART_HI0_OFF 41984
#define SMEM_PART_HI0_STAGE_BYTES 16384
#define SMEM_PART_HI0_STRIDE 16384
#define SMEM_VS_SMEM_OFF 91136
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
#define SMEM_VT_SMEM_OFF 58368
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 93184
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_vsa_sm90_bcdb3caff331217e0a71(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ Qs, float* __restrict__ Ks, float* __restrict__ Vs, int k_scale_groups, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, int g_shift)
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
    #define v_full0_addr (mbar_base + 32)
    #define v_full1_addr (mbar_base + 40)
    #define v_full2_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* mstats = reinterpret_cast<float*>(smem_raw + 92160);
    const int mstats_addr = smem + 92160;
    float* part_lo = reinterpret_cast<float*>(smem_raw + 25600);
    const int part_lo_addr = smem + 25600;
    float* part_hi0 = reinterpret_cast<float*>(smem_raw + 41984);
    const int part_hi0_addr = smem + 41984;
    float* vs_smem = reinterpret_cast<float*>(smem_raw + 91136);
    const int vs_smem_addr = smem + 91136;
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* k_smem = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int k_smem_addr = smem + 9216;
    uint8_t* vl_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int vl_smem_addr = smem + 17408;
    uint8_t* vt_smem = reinterpret_cast<uint8_t*>(smem_raw + 58368);
    const int vt_smem_addr = smem + 58368;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 7 barriers)
    // Mbarriers at smem_raw[0..56)

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
            // v_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // v_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // v_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // === Task calls (declaration order) ===
    int item = bid;
    int mb = seqlen_q / 64;
    int plan_base = item * 4;
    int meta = plan[plan_base];
    int cnt = meta;
    int tile = item;
    int blk_base = plan_base + 1;
    int head = tile / mb;
    int qb = tile - head * mb;
    int q_row = head * seqlen_q + qb * 64;
    int kv_base = (head >> g_shift) * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            int blk_pre[3];
            blk_pre[0] = plan[blk_base];
            blk_pre[1] = plan[blk_base + 1];
            blk_pre[2] = plan[blk_base + 2];
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
            if (cnt > 2) {
                int blk_row_1_2 = kv_base + blk_pre[2] * 64;
                mbarrier_arrive_expect_tx(k_full2_addr, 8192);
                tma_2d_gmem2smem(k_smem_addr + 32768, (&K), 0, blk_row_1_2, k_full2_addr);
                mbarrier_arrive_expect_tx(v_full2_addr, 8192);
                tma_2d_gmem2smem(vl_smem_addr + 32768, (&V), 0, blk_row_1_2, v_full2_addr);
            }
        }
    }
    __syncthreads();
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
    float ks_regs[8];
    int ks_head = (head >> g_shift) * k_scale_groups;
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
    uint64_t _wgmma_desc_3 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
    uint64_t _wgmma_desc_4 = (((uint64_t)(((k_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    uint64_t _wgmma_desc_5 = (((uint64_t)(((vt_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
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
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
        m0 = fmaxf(m0, _shfl_xor_sync_6);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
        m0 = fmaxf(m0, _shfl_xor_sync_7);
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
        m1 = fmaxf(m1, _shfl_xor_sync_8);
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
        m1 = fmaxf(m1, _shfl_xor_sync_9);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_11;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_11) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_12;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_12) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_10;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_10) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_11), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_12));
        p_e4m3[0] = _prmt_10;
        uint16_t _cvt_satfinite_e4m3x2_f32_14;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_15;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_15) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_13;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_13) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_14), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_15));
        p_e4m3[1] = _prmt_13;
        uint16_t _cvt_satfinite_e4m3x2_f32_17;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_17) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_18;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_16;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_16) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_17), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_18));
        p_e4m3[2] = _prmt_16;
        uint16_t _cvt_satfinite_e4m3x2_f32_20;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_20) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_21;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_19;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_19) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_20), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_21));
        p_e4m3[3] = _prmt_19;
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
        uint32_t _prmt_22;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_22) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_22;
        uint32_t _prmt_23;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_23) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_23;
        uint32_t _prmt_24;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_24) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_24;
        uint32_t _prmt_25;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_25) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_25;
        uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_27;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_27) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_27;
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_28) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_28;
        uint32_t _prmt_29;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_29) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_29;
        uint32_t _prmt_30;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_30) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_30;
        uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_32;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_32) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_32;
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_33) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_33;
        uint32_t _prmt_34;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_34) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_34;
        uint32_t _prmt_35;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_35) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_35;
        uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_37;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_37) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_37;
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_38;
        uint32_t _prmt_39;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_39) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_39;
        uint32_t _prmt_40;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_40) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_40;
        uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_43;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_43) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_44;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_44) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_42;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_42) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_43), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_44));
        p_e4m3[4] = _prmt_42;
        uint16_t _cvt_satfinite_e4m3x2_f32_46;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_46) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_47;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_47) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_45;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_45) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_46), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_47));
        p_e4m3[5] = _prmt_45;
        uint16_t _cvt_satfinite_e4m3x2_f32_49;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_50;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_50) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_48;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_48) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_49), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_50));
        p_e4m3[6] = _prmt_48;
        uint16_t _cvt_satfinite_e4m3x2_f32_52;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_52) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_53;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_51;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_51) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_52), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_53));
        p_e4m3[7] = _prmt_51;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_54 = (((uint64_t)(((k_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_54 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_54);
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
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 2);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_55);
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 1);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_56);
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 2);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_57);
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 1);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_58);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_60;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_60) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_61;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_61) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_59;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_59) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_60), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_61));
        p_e4m3[0] = _prmt_59;
        uint16_t _cvt_satfinite_e4m3x2_f32_63;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_63) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_64;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_64) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_62;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_62) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_63), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_64));
        p_e4m3[1] = _prmt_62;
        uint16_t _cvt_satfinite_e4m3x2_f32_66;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_66) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_67;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_67) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_65;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_65) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_66), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_67));
        p_e4m3[2] = _prmt_65;
        uint16_t _cvt_satfinite_e4m3x2_f32_69;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_69) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_70;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_70) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_68;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_68) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_69), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_70));
        p_e4m3[3] = _prmt_68;
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
        uint32_t _prmt_71;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_71) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_71;
        uint32_t _prmt_72;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_72) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_72;
        uint32_t _prmt_73;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_73) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_73;
        uint32_t _prmt_74;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_74) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_74;
        uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_76;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_76) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_76;
        uint32_t _prmt_77;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_77) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_77;
        uint32_t _prmt_78;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_78) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_78;
        uint32_t _prmt_79;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_79) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_79;
        uint32_t _stmatrix_addr_80 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_80), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_81;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_81) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_81;
        uint32_t _prmt_82;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_82) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_82;
        uint32_t _prmt_83;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_83) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_83;
        uint32_t _prmt_84;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_84) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_84;
        uint32_t _stmatrix_addr_85 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_85), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_86;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_86) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_86;
        uint32_t _prmt_87;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_87) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_87;
        uint32_t _prmt_88;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_88) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_88;
        uint32_t _prmt_89;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_89) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_89;
        uint32_t _stmatrix_addr_90 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_90), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_92;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_92) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_93;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_93) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_91;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_91) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_92), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_93));
        p_e4m3[4] = _prmt_91;
        uint16_t _cvt_satfinite_e4m3x2_f32_95;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_95) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_96;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_96) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_94;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_94) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_95), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_96));
        p_e4m3[5] = _prmt_94;
        uint16_t _cvt_satfinite_e4m3x2_f32_98;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_98) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_99;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_99) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_97;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_97) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_98), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_99));
        p_e4m3[6] = _prmt_97;
        uint16_t _cvt_satfinite_e4m3x2_f32_101;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_101) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_102;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_102) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_100;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_100) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_101), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_102));
        p_e4m3[7] = _prmt_100;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_103 = (((uint64_t)(((k_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_103 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_103);
    uint64_t _wgmma_desc_104 = (((uint64_t)(((vt_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t vtd_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_104 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_104);
    if (lim_even > 2) {
        mbarrier_wait(k_full2_addr, 0);
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
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 2);
        m0_2 = fmaxf(m0_2, _shfl_xor_sync_105);
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 1);
        m0_2 = fmaxf(m0_2, _shfl_xor_sync_106);
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 2);
        m1_2 = fmaxf(m1_2, _shfl_xor_sync_107);
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 1);
        m1_2 = fmaxf(m1_2, _shfl_xor_sync_108);
        new_max0_2 = m0_2;
        new_max1_2 = m1_2;
        float merged_max0_0 = fmaxf(row_max0, new_max0_2);
        float merged_max1_0 = fmaxf(row_max1, new_max1_2);
        int grow0_0 = merged_max0_0 - row_max0 > 0.8073549f;
        int grow1_0 = merged_max1_0 - row_max1 > 0.8073549f;
        float _exp2_109 = approx_exp2(row_max0 - merged_max0_0);
        float rescale0_0 = ((grow0_0 != 0) ? _exp2_109 : 1.0f);
        float _exp2_110 = approx_exp2(row_max1 - merged_max1_0);
        float rescale1_0 = ((grow1_0 != 0) ? _exp2_110 : 1.0f);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_112;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_112) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_113;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_113) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_111;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_111) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_112), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_113));
        p_e4m3[0] = _prmt_111;
        uint16_t _cvt_satfinite_e4m3x2_f32_115;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_115) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_116;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_116) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_114;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_114) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_115), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_116));
        p_e4m3[1] = _prmt_114;
        uint16_t _cvt_satfinite_e4m3x2_f32_118;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_118) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_119;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_119) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_117;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_117) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_118), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_119));
        p_e4m3[2] = _prmt_117;
        uint16_t _cvt_satfinite_e4m3x2_f32_121;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_121) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_122;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_122) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_120;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_120) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_121), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_122));
        p_e4m3[3] = _prmt_120;
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
        uint32_t _prmt_123;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_123) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_123;
        uint32_t _prmt_124;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_124) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_124;
        uint32_t _prmt_125;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_125) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_125;
        uint32_t _prmt_126;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_126) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_126;
        uint32_t _stmatrix_addr_127 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_127), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_128;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_128) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_128;
        uint32_t _prmt_129;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_129) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_129;
        uint32_t _prmt_130;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_130) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_130;
        uint32_t _prmt_131;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_131) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_131;
        uint32_t _stmatrix_addr_132 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_132), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_133;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_133) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_133;
        uint32_t _prmt_134;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_134) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_134;
        uint32_t _prmt_135;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_135) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_135;
        uint32_t _prmt_136;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_136) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_136;
        uint32_t _stmatrix_addr_137 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_137), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_138;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_138) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_138;
        uint32_t _prmt_139;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_139) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_139;
        uint32_t _prmt_140;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_140) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_140;
        uint32_t _prmt_141;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_141) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_141;
        uint32_t _stmatrix_addr_142 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_142), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_144;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_144) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_145;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_145) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_143;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_143) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_144), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_145));
        p_e4m3[4] = _prmt_143;
        uint16_t _cvt_satfinite_e4m3x2_f32_147;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_147) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_148;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_148) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_146;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_146) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_147), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_148));
        p_e4m3[5] = _prmt_146;
        uint16_t _cvt_satfinite_e4m3x2_f32_150;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_150) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_151;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_151) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_149;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_149) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_150), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_151));
        p_e4m3[6] = _prmt_149;
        uint16_t _cvt_satfinite_e4m3x2_f32_153;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_153) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_154;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_154) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_152;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_152) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_153), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_154));
        p_e4m3[7] = _prmt_152;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(vtd_0 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_sync_155;
    float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_sync_156;
    float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_sync_157;
    float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_sync_158;
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
    float _exp2_159 = approx_exp2(x0m0 - mm0);
    float f00 = ((x0m0 == -CUDART_INF_F) ? 0.0f : _exp2_159);
    float _exp2_160 = approx_exp2(x0m1 - mm1);
    float f01 = ((x0m1 == -CUDART_INF_F) ? 0.0f : _exp2_160);
    float _exp2_161 = approx_exp2(x1m0 - mm0);
    float f10 = ((x1m0 == -CUDART_INF_F) ? 0.0f : _exp2_161);
    float _exp2_162 = approx_exp2(x1m1 - mm1);
    float f11 = ((x1m1 == -CUDART_INF_F) ? 0.0f : _exp2_162);
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
    if (wg == 0) {
        float _rcp_0 = approx_rcp(row_sum0);
        float _rcp_1 = approx_rcp(row_sum1);
        float vs0_1_0 = vs_smem[vs_base];
        float vs1_2_0 = vs_smem[vs_base + 1];
        d_o[0] = d_o[0] * (vs0_1_0 * _rcp_0);
        d_o[1] = d_o[1] * (vs1_2_0 * _rcp_0);
        d_o[2] = d_o[2] * (vs0_1_0 * _rcp_1);
        d_o[3] = d_o[3] * (vs1_2_0 * _rcp_1);
        float vs0_1_1 = vs_smem[vs_base + 8];
        float vs1_2_1 = vs_smem[vs_base + 8 + 1];
        d_o[4] = d_o[4] * (vs0_1_1 * _rcp_0);
        d_o[5] = d_o[5] * (vs1_2_1 * _rcp_0);
        d_o[6] = d_o[6] * (vs0_1_1 * _rcp_1);
        d_o[7] = d_o[7] * (vs1_2_1 * _rcp_1);
        float vs0_1_2 = vs_smem[vs_base + 16];
        float vs1_2_2 = vs_smem[vs_base + 16 + 1];
        d_o[8] = d_o[8] * (vs0_1_2 * _rcp_0);
        d_o[9] = d_o[9] * (vs1_2_2 * _rcp_0);
        d_o[10] = d_o[10] * (vs0_1_2 * _rcp_1);
        d_o[11] = d_o[11] * (vs1_2_2 * _rcp_1);
        float vs0_1_3 = vs_smem[vs_base + 24];
        float vs1_2_3 = vs_smem[vs_base + 24 + 1];
        d_o[12] = d_o[12] * (vs0_1_3 * _rcp_0);
        d_o[13] = d_o[13] * (vs1_2_3 * _rcp_0);
        d_o[14] = d_o[14] * (vs0_1_3 * _rcp_1);
        d_o[15] = d_o[15] * (vs1_2_3 * _rcp_1);
        float vs0_1_4 = vs_smem[vs_base + 32];
        float vs1_2_4 = vs_smem[vs_base + 32 + 1];
        d_o[16] = d_o[16] * (vs0_1_4 * _rcp_0);
        d_o[17] = d_o[17] * (vs1_2_4 * _rcp_0);
        d_o[18] = d_o[18] * (vs0_1_4 * _rcp_1);
        d_o[19] = d_o[19] * (vs1_2_4 * _rcp_1);
        float vs0_1_5 = vs_smem[vs_base + 40];
        float vs1_2_5 = vs_smem[vs_base + 40 + 1];
        d_o[20] = d_o[20] * (vs0_1_5 * _rcp_0);
        d_o[21] = d_o[21] * (vs1_2_5 * _rcp_0);
        d_o[22] = d_o[22] * (vs0_1_5 * _rcp_1);
        d_o[23] = d_o[23] * (vs1_2_5 * _rcp_1);
        float vs0_1_6 = vs_smem[vs_base + 48];
        float vs1_2_6 = vs_smem[vs_base + 48 + 1];
        d_o[24] = d_o[24] * (vs0_1_6 * _rcp_0);
        d_o[25] = d_o[25] * (vs1_2_6 * _rcp_0);
        d_o[26] = d_o[26] * (vs0_1_6 * _rcp_1);
        d_o[27] = d_o[27] * (vs1_2_6 * _rcp_1);
        float vs0_1_7 = vs_smem[vs_base + 56];
        float vs1_2_7 = vs_smem[vs_base + 56 + 1];
        d_o[28] = d_o[28] * (vs0_1_7 * _rcp_0);
        d_o[29] = d_o[29] * (vs1_2_7 * _rcp_0);
        d_o[30] = d_o[30] * (vs0_1_7 * _rcp_1);
        d_o[31] = d_o[31] * (vs1_2_7 * _rcp_1);
        int qodd = qj & 1;
        unsigned int o_pk[2];
        unsigned int o_vec[2];
        int o_row_base = q_row * 128 + qj * 2 + qodd * 6;
        int m_local_r_1_0 = m0_local;
        int row_off_1_0 = o_row_base + m_local_r_1_0 * 128;
        __nv_bfloat162 _bit_cast_src_163 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_163)[0];
        __nv_bfloat162 _bit_cast_src_164 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_164)[0];
        unsigned int send_1_0 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_0);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_0 : o_pk[1]);
        reinterpret_cast<int2*>(O + row_off_1_0)[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_165 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_165)[0];
        __nv_bfloat162 _bit_cast_src_166 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_166)[0];
        unsigned int send_1_1 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_1);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_1 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_167 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_167)[0];
        __nv_bfloat162 _bit_cast_src_168 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_168)[0];
        unsigned int send_1_2 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_2);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_2 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_169 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_169)[0];
        __nv_bfloat162 _bit_cast_src_170 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_170)[0];
        unsigned int send_1_3 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_3);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_3 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
        int m_local_r_1_1 = m1_local;
        int row_off_1_1 = o_row_base + m_local_r_1_1 * 128;
        __nv_bfloat162 _bit_cast_src_171 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_171)[0];
        __nv_bfloat162 _bit_cast_src_172 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_172)[0];
        unsigned int send_1_4 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_4);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_4 : o_pk[1]);
        reinterpret_cast<int2*>(O + row_off_1_1)[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_173 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_173)[0];
        __nv_bfloat162 _bit_cast_src_174 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_174)[0];
        unsigned int send_1_5 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_5);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_5 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_175 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_175)[0];
        __nv_bfloat162 _bit_cast_src_176 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_176)[0];
        unsigned int send_1_6 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_6);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_6 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_177 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_177)[0];
        __nv_bfloat162 _bit_cast_src_178 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_178)[0];
        unsigned int send_1_7 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_1_7, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_7);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_7 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
    }
    if (wg == 1) {
        float _rcp_2 = approx_rcp(row_sum0);
        float _rcp_3 = approx_rcp(row_sum1);
        float vs0_1_8 = vs_smem[vs_base + 64];
        float vs1_2_8 = vs_smem[vs_base + 64 + 1];
        d_o[32] = d_o[32] * (vs0_1_8 * _rcp_2);
        d_o[33] = d_o[33] * (vs1_2_8 * _rcp_2);
        d_o[34] = d_o[34] * (vs0_1_8 * _rcp_3);
        d_o[35] = d_o[35] * (vs1_2_8 * _rcp_3);
        float vs0_1_9 = vs_smem[vs_base + 72];
        float vs1_2_9 = vs_smem[vs_base + 72 + 1];
        d_o[36] = d_o[36] * (vs0_1_9 * _rcp_2);
        d_o[37] = d_o[37] * (vs1_2_9 * _rcp_2);
        d_o[38] = d_o[38] * (vs0_1_9 * _rcp_3);
        d_o[39] = d_o[39] * (vs1_2_9 * _rcp_3);
        float vs0_1_10 = vs_smem[vs_base + 80];
        float vs1_2_10 = vs_smem[vs_base + 80 + 1];
        d_o[40] = d_o[40] * (vs0_1_10 * _rcp_2);
        d_o[41] = d_o[41] * (vs1_2_10 * _rcp_2);
        d_o[42] = d_o[42] * (vs0_1_10 * _rcp_3);
        d_o[43] = d_o[43] * (vs1_2_10 * _rcp_3);
        float vs0_1_11 = vs_smem[vs_base + 88];
        float vs1_2_11 = vs_smem[vs_base + 88 + 1];
        d_o[44] = d_o[44] * (vs0_1_11 * _rcp_2);
        d_o[45] = d_o[45] * (vs1_2_11 * _rcp_2);
        d_o[46] = d_o[46] * (vs0_1_11 * _rcp_3);
        d_o[47] = d_o[47] * (vs1_2_11 * _rcp_3);
        float vs0_1_12 = vs_smem[vs_base + 96];
        float vs1_2_12 = vs_smem[vs_base + 96 + 1];
        d_o[48] = d_o[48] * (vs0_1_12 * _rcp_2);
        d_o[49] = d_o[49] * (vs1_2_12 * _rcp_2);
        d_o[50] = d_o[50] * (vs0_1_12 * _rcp_3);
        d_o[51] = d_o[51] * (vs1_2_12 * _rcp_3);
        float vs0_1_13 = vs_smem[vs_base + 104];
        float vs1_2_13 = vs_smem[vs_base + 104 + 1];
        d_o[52] = d_o[52] * (vs0_1_13 * _rcp_2);
        d_o[53] = d_o[53] * (vs1_2_13 * _rcp_2);
        d_o[54] = d_o[54] * (vs0_1_13 * _rcp_3);
        d_o[55] = d_o[55] * (vs1_2_13 * _rcp_3);
        float vs0_1_14 = vs_smem[vs_base + 112];
        float vs1_2_14 = vs_smem[vs_base + 112 + 1];
        d_o[56] = d_o[56] * (vs0_1_14 * _rcp_2);
        d_o[57] = d_o[57] * (vs1_2_14 * _rcp_2);
        d_o[58] = d_o[58] * (vs0_1_14 * _rcp_3);
        d_o[59] = d_o[59] * (vs1_2_14 * _rcp_3);
        float vs0_1_15 = vs_smem[vs_base + 120];
        float vs1_2_15 = vs_smem[vs_base + 120 + 1];
        d_o[60] = d_o[60] * (vs0_1_15 * _rcp_2);
        d_o[61] = d_o[61] * (vs1_2_15 * _rcp_2);
        d_o[62] = d_o[62] * (vs0_1_15 * _rcp_3);
        d_o[63] = d_o[63] * (vs1_2_15 * _rcp_3);
        int qodd_1 = qj & 1;
        unsigned int o_pk_1[2];
        unsigned int o_vec_1[2];
        int o_row_base_1 = q_row * 128 + qj * 2 + qodd_1 * 6;
        int m_local_r_1_2 = m0_local;
        int row_off_1_2 = o_row_base_1 + m_local_r_1_2 * 128;
        __nv_bfloat162 _bit_cast_src_179 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_179)[0];
        __nv_bfloat162 _bit_cast_src_180 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_180)[0];
        unsigned int send_1_8 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send_1_8, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_8);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_8 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 64))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_181 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_181)[0];
        __nv_bfloat162 _bit_cast_src_182 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_182)[0];
        unsigned int send_1_9 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_1_9, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_9);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_9 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 80))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_183 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_183)[0];
        __nv_bfloat162 _bit_cast_src_184 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_184)[0];
        unsigned int send_1_10 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_1_10, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_10);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_10 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 96))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_185 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_185)[0];
        __nv_bfloat162 _bit_cast_src_186 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_186)[0];
        unsigned int send_1_11 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_1_11, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_11);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_11 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 112))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        int m_local_r_1_3 = m1_local;
        int row_off_1_3 = o_row_base_1 + m_local_r_1_3 * 128;
        __nv_bfloat162 _bit_cast_src_187 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_187)[0];
        __nv_bfloat162 _bit_cast_src_188 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_188)[0];
        unsigned int send_1_12 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_1_12, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_12);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_12 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 64))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_189 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_189)[0];
        __nv_bfloat162 _bit_cast_src_190 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_190)[0];
        unsigned int send_1_13 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_1_13, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_13);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_13 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 80))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_191 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_191)[0];
        __nv_bfloat162 _bit_cast_src_192 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_192)[0];
        unsigned int send_1_14 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_1_14, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_14);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_14 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 96))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_193 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_193)[0];
        __nv_bfloat162 _bit_cast_src_194 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_194)[0];
        unsigned int send_1_15 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_1_15, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_15);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_15 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 112))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
