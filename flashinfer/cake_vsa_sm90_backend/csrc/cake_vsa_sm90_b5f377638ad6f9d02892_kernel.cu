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
#define SMEM_MSTATS_OFF 108544
#define SMEM_MSTATS_STAGE_BYTES 1024
#define SMEM_MSTATS_STRIDE 1024
#define SMEM_PART_LO_OFF 58368
#define SMEM_PART_LO_STAGE_BYTES 16384
#define SMEM_PART_LO_STRIDE 16384
#define SMEM_PART_HI0_OFF 41984
#define SMEM_PART_HI0_STAGE_BYTES 16384
#define SMEM_PART_HI0_STRIDE 16384
#define SMEM_VS_SMEM_OFF 107520
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
#define SMEM_VT_SMEM_OFF 74752
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 109568
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
kernel_cake_vsa_sm90_b5f377638ad6f9d02892(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ Qs, float* __restrict__ Ks, float* __restrict__ Vs, int k_scale_groups, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, int g_shift)
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
    #define v_full0_addr (mbar_base + 40)
    #define v_full1_addr (mbar_base + 48)
    #define v_full2_addr (mbar_base + 56)
    #define v_full3_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* mstats = reinterpret_cast<float*>(smem_raw + 108544);
    const int mstats_addr = smem + 108544;
    float* part_lo = reinterpret_cast<float*>(smem_raw + 58368);
    const int part_lo_addr = smem + 58368;
    float* part_hi0 = reinterpret_cast<float*>(smem_raw + 41984);
    const int part_hi0_addr = smem + 41984;
    float* vs_smem = reinterpret_cast<float*>(smem_raw + 107520);
    const int vs_smem_addr = smem + 107520;
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* k_smem = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int k_smem_addr = smem + 9216;
    uint8_t* vl_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int vl_smem_addr = smem + 17408;
    uint8_t* vt_smem = reinterpret_cast<uint8_t*>(smem_raw + 74752);
    const int vt_smem_addr = smem + 74752;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

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
            // v_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // v_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // v_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // v_full3: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // === Task calls (declaration order) ===
    int item = bid;
    int mb = seqlen_q / 64;
    int plan_base = item * 5;
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
    uint64_t _wgmma_desc_4 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    uint64_t _wgmma_desc_5 = (((uint64_t)(((k_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
    uint64_t _wgmma_desc_6 = (((uint64_t)(((vt_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
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
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
        m0 = fmaxf(m0, _shfl_xor_sync_7);
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
        m0 = fmaxf(m0, _shfl_xor_sync_8);
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
        m1 = fmaxf(m1, _shfl_xor_sync_9);
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
        m1 = fmaxf(m1, _shfl_xor_sync_10);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_12;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_12) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_13;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_13) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_11;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_11) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_12), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_13));
        p_e4m3[0] = _prmt_11;
        uint16_t _cvt_satfinite_e4m3x2_f32_15;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_15) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_16;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_16) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_14;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_14) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_15), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_16));
        p_e4m3[1] = _prmt_14;
        uint16_t _cvt_satfinite_e4m3x2_f32_18;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_19;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_17;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_17) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_18), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_19));
        p_e4m3[2] = _prmt_17;
        uint16_t _cvt_satfinite_e4m3x2_f32_21;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_22;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_22) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_20;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_20) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_21), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_22));
        p_e4m3[3] = _prmt_20;
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
        uint32_t _prmt_23;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_23) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_23;
        uint32_t _prmt_24;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_24) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_24;
        uint32_t _prmt_25;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_25) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_25;
        uint32_t _prmt_26;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_26) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_26;
        uint32_t _stmatrix_addr_27 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_28;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_28) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_28;
        uint32_t _prmt_29;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_29) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_29;
        uint32_t _prmt_30;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_30) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_30;
        uint32_t _prmt_31;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_31) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_31;
        uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_33;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_33) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_33;
        uint32_t _prmt_34;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_34) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_34;
        uint32_t _prmt_35;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_35) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_35;
        uint32_t _prmt_36;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_36) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_36;
        uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_38;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_38;
        uint32_t _prmt_39;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_39) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_39;
        uint32_t _prmt_40;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_40) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_40;
        uint32_t _prmt_41;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_41) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_41;
        uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_44;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_44) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_45;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_45) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_43;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_43) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_44), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_45));
        p_e4m3[4] = _prmt_43;
        uint16_t _cvt_satfinite_e4m3x2_f32_47;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_47) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_48;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_48) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_46;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_46) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_47), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_48));
        p_e4m3[5] = _prmt_46;
        uint16_t _cvt_satfinite_e4m3x2_f32_50;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_50) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_51;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_51) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_49;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_49) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_50), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_51));
        p_e4m3[6] = _prmt_49;
        uint16_t _cvt_satfinite_e4m3x2_f32_53;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_54;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_54) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_52;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_52) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_53), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_54));
        p_e4m3[7] = _prmt_52;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_55 = (((uint64_t)(((k_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_55 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_55);
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
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 2);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_56);
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 1);
        m0_1 = fmaxf(m0_1, _shfl_xor_sync_57);
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 2);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_58);
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, m1_1, 1);
        m1_1 = fmaxf(m1_1, _shfl_xor_sync_59);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_61;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_61) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_62;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_62) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_60;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_60) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_61), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_62));
        p_e4m3[0] = _prmt_60;
        uint16_t _cvt_satfinite_e4m3x2_f32_64;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_64) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_65;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_65) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_63;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_63) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_64), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_65));
        p_e4m3[1] = _prmt_63;
        uint16_t _cvt_satfinite_e4m3x2_f32_67;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_67) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_68;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_68) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_66;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_66) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_67), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_68));
        p_e4m3[2] = _prmt_66;
        uint16_t _cvt_satfinite_e4m3x2_f32_70;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_70) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_71;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_71) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_69;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_69) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_70), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_71));
        p_e4m3[3] = _prmt_69;
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
        uint32_t _prmt_72;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_72) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_72;
        uint32_t _prmt_73;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_73) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_73;
        uint32_t _prmt_74;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_74) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_74;
        uint32_t _prmt_75;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_75) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_75;
        uint32_t _stmatrix_addr_76 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(t_dc0 * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_76), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_77;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_77) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_77;
        uint32_t _prmt_78;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_78) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_78;
        uint32_t _prmt_79;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_79) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_79;
        uint32_t _prmt_80;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_80) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_80;
        uint32_t _stmatrix_addr_81 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 1) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_81), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_82;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_82) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_82;
        uint32_t _prmt_83;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_83) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_83;
        uint32_t _prmt_84;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_84) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_84;
        uint32_t _prmt_85;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_85) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_85;
        uint32_t _stmatrix_addr_86 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 2) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_86), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_87;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_87) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_87;
        uint32_t _prmt_88;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_88) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_88;
        uint32_t _prmt_89;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_89) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_89;
        uint32_t _prmt_90;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_90) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_90;
        uint32_t _stmatrix_addr_91 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)((t_dc0 + 3) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_91), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_93;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_93) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_94;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_94) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_92;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_92) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_93), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_94));
        p_e4m3[4] = _prmt_92;
        uint16_t _cvt_satfinite_e4m3x2_f32_96;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_96) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_97;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_97) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_95;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_95) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_96), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_97));
        p_e4m3[5] = _prmt_95;
        uint16_t _cvt_satfinite_e4m3x2_f32_99;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_99) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_100;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_100) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_98;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_98) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_99), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_100));
        p_e4m3[6] = _prmt_98;
        uint16_t _cvt_satfinite_e4m3x2_f32_102;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_102) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_103;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_103) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_101;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_101) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_102), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_103));
        p_e4m3[7] = _prmt_101;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_104 = (((uint64_t)(((k_smem_addr + 32768)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_104 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_104);
    uint64_t _wgmma_desc_105 = (((uint64_t)(((vt_smem_addr + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t vtd_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_105 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_105);
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
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 2);
        m0_2 = fmaxf(m0_2, _shfl_xor_sync_106);
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 1);
        m0_2 = fmaxf(m0_2, _shfl_xor_sync_107);
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 2);
        m1_2 = fmaxf(m1_2, _shfl_xor_sync_108);
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 1);
        m1_2 = fmaxf(m1_2, _shfl_xor_sync_109);
        new_max0_2 = m0_2;
        new_max1_2 = m1_2;
        float merged_max0_0 = fmaxf(row_max0, new_max0_2);
        float merged_max1_0 = fmaxf(row_max1, new_max1_2);
        int grow0_0 = merged_max0_0 - row_max0 > 0.8073549f;
        int grow1_0 = merged_max1_0 - row_max1 > 0.8073549f;
        float _exp2_110 = approx_exp2(row_max0 - merged_max0_0);
        float rescale0_0 = ((grow0_0 != 0) ? _exp2_110 : 1.0f);
        float _exp2_111 = approx_exp2(row_max1 - merged_max1_0);
        float rescale1_0 = ((grow1_0 != 0) ? _exp2_111 : 1.0f);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_113;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_113) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_114;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_114) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_112;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_112) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_113), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_114));
        p_e4m3[0] = _prmt_112;
        uint16_t _cvt_satfinite_e4m3x2_f32_116;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_116) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_117;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_117) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_115;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_115) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_116), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_117));
        p_e4m3[1] = _prmt_115;
        uint16_t _cvt_satfinite_e4m3x2_f32_119;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_119) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_120;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_120) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_118;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_118) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_119), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_120));
        p_e4m3[2] = _prmt_118;
        uint16_t _cvt_satfinite_e4m3x2_f32_122;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_122) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_123;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_123) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_121;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_121) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_122), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_123));
        p_e4m3[3] = _prmt_121;
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
        uint32_t _prmt_124;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_124) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_124;
        uint32_t _prmt_125;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_125) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_125;
        uint32_t _prmt_126;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_126) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_126;
        uint32_t _prmt_127;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_127) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_127;
        uint32_t _stmatrix_addr_128 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + t_dc0 * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_128), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_129;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_129) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_129;
        uint32_t _prmt_130;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_130) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_130;
        uint32_t _prmt_131;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_131) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_131;
        uint32_t _prmt_132;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_132) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_132;
        uint32_t _stmatrix_addr_133 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 1) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_133), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_134;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_134) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_134;
        uint32_t _prmt_135;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_135) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_135;
        uint32_t _prmt_136;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_136) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_136;
        uint32_t _prmt_137;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_137) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_137;
        uint32_t _stmatrix_addr_138 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 2) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_138), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_139;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_139) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_139;
        uint32_t _prmt_140;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_140) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_140;
        uint32_t _prmt_141;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_141) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_141;
        uint32_t _prmt_142;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_142) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_142;
        uint32_t _stmatrix_addr_143 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 3) * 2048 + st_row + ((t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_143), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_145;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_145) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_146;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_146) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_144;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_144) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_145), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_146));
        p_e4m3[4] = _prmt_144;
        uint16_t _cvt_satfinite_e4m3x2_f32_148;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_148) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_149;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_149) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_147;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_147) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_148), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_149));
        p_e4m3[5] = _prmt_147;
        uint16_t _cvt_satfinite_e4m3x2_f32_151;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_151) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_152;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_152) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_150;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_150) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_151), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_152));
        p_e4m3[6] = _prmt_150;
        uint16_t _cvt_satfinite_e4m3x2_f32_154;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_154) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_155;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_155) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_153;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_153) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_154), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_155));
        p_e4m3[7] = _prmt_153;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(vtd_0 + 2)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    uint64_t _wgmma_desc_156 = (((uint64_t)(((k_smem_addr + 49152)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t kd_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_156 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_156);
    if (lim_odd > 3) {
        mbarrier_wait(k_full3_addr, 0);
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
        float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, m0_3, 2);
        m0_3 = fmaxf(m0_3, _shfl_xor_sync_157);
        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, m0_3, 1);
        m0_3 = fmaxf(m0_3, _shfl_xor_sync_158);
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, m1_3, 2);
        m1_3 = fmaxf(m1_3, _shfl_xor_sync_159);
        float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, m1_3, 1);
        m1_3 = fmaxf(m1_3, _shfl_xor_sync_160);
        new_max0_3 = m0_3;
        new_max1_3 = m1_3;
        float merged_max0_1 = fmaxf(row_max0, new_max0_3);
        float merged_max1_1 = fmaxf(row_max1, new_max1_3);
        int grow0_1 = merged_max0_1 - row_max0 > 0.8073549f;
        int grow1_1 = merged_max1_1 - row_max1 > 0.8073549f;
        float _exp2_161 = approx_exp2(row_max0 - merged_max0_1);
        float rescale0_1 = ((grow0_1 != 0) ? _exp2_161 : 1.0f);
        float _exp2_162 = approx_exp2(row_max1 - merged_max1_1);
        float rescale1_1 = ((grow1_1 != 0) ? _exp2_162 : 1.0f);
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
        uint16_t _cvt_satfinite_e4m3x2_f32_164;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_164) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_165;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_165) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_163;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_163) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_164), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_165));
        p_e4m3[0] = _prmt_163;
        uint16_t _cvt_satfinite_e4m3x2_f32_167;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_167) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_168;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_168) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_166;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_166) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_167), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_168));
        p_e4m3[1] = _prmt_166;
        uint16_t _cvt_satfinite_e4m3x2_f32_170;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_170) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_171;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_171) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_169;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_169) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_170), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_171));
        p_e4m3[2] = _prmt_169;
        uint16_t _cvt_satfinite_e4m3x2_f32_173;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_173) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_174;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_174) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_172;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_172) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_173), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_174));
        p_e4m3[3] = _prmt_172;
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
        uint32_t _prmt_175;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_175) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
        tpw[0] = _prmt_175;
        uint32_t _prmt_176;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_176) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
        tpw[1] = _prmt_176;
        uint32_t _prmt_177;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_177) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
        tpw[2] = _prmt_177;
        uint32_t _prmt_178;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_178) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
        tpw[3] = _prmt_178;
        uint32_t _stmatrix_addr_179 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + t_dc0 * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_179), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_180;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_180) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
        tpw[0] = _prmt_180;
        uint32_t _prmt_181;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_181) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
        tpw[1] = _prmt_181;
        uint32_t _prmt_182;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_182) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
        tpw[2] = _prmt_182;
        uint32_t _prmt_183;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_183) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
        tpw[3] = _prmt_183;
        uint32_t _stmatrix_addr_184 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 1) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_184), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_185;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_185) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
        tpw[0] = _prmt_185;
        uint32_t _prmt_186;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_186) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
        tpw[1] = _prmt_186;
        uint32_t _prmt_187;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_187) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
        tpw[2] = _prmt_187;
        uint32_t _prmt_188;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_188) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
        tpw[3] = _prmt_188;
        uint32_t _stmatrix_addr_189 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 2) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_189), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
            : "memory");
        uint32_t _prmt_190;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_190) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
        tpw[0] = _prmt_190;
        uint32_t _prmt_191;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_191) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
        tpw[1] = _prmt_191;
        uint32_t _prmt_192;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_192) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
        tpw[2] = _prmt_192;
        uint32_t _prmt_193;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_193) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
        tpw[3] = _prmt_193;
        uint32_t _stmatrix_addr_194 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(16384 + (t_dc0 + 3) * 2048 + st_row + ((4 + t_kq2 + t_mih ^ t_dl7) << 4)));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_194), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
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
        uint16_t _cvt_satfinite_e4m3x2_f32_196;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_196) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_197;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_197) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_195;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_195) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_196), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_197));
        p_e4m3[4] = _prmt_195;
        uint16_t _cvt_satfinite_e4m3x2_f32_199;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_199) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_200;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_200) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_198;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_198) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_199), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_200));
        p_e4m3[5] = _prmt_198;
        uint16_t _cvt_satfinite_e4m3x2_f32_202;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_202) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_203;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_203) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_201;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_201) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_202), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_203));
        p_e4m3[6] = _prmt_201;
        uint16_t _cvt_satfinite_e4m3x2_f32_205;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_205) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_206;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_206) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_204;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_204) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_205), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_206));
        p_e4m3[7] = _prmt_204;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(vtd_0 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    }
    float _shfl_xor_sync_207 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_sync_207;
    float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_sync_208;
    float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_sync_209;
    float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_sync_210;
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
    float _exp2_211 = approx_exp2(x0m0 - mm0);
    float f00 = ((x0m0 == -CUDART_INF_F) ? 0.0f : _exp2_211);
    float _exp2_212 = approx_exp2(x0m1 - mm1);
    float f01 = ((x0m1 == -CUDART_INF_F) ? 0.0f : _exp2_212);
    float _exp2_213 = approx_exp2(x1m0 - mm0);
    float f10 = ((x1m0 == -CUDART_INF_F) ? 0.0f : _exp2_213);
    float _exp2_214 = approx_exp2(x1m1 - mm1);
    float f11 = ((x1m1 == -CUDART_INF_F) ? 0.0f : _exp2_214);
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
        __nv_bfloat162 _bit_cast_src_215 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_215)[0];
        __nv_bfloat162 _bit_cast_src_216 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_216)[0];
        unsigned int send_1_0 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_0);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_0 : o_pk[1]);
        reinterpret_cast<int2*>(O + row_off_1_0)[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_217 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_217)[0];
        __nv_bfloat162 _bit_cast_src_218 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_218)[0];
        unsigned int send_1_1 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_1);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_1 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_219 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_219)[0];
        __nv_bfloat162 _bit_cast_src_220 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_220)[0];
        unsigned int send_1_2 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_2);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_2 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_221 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_221)[0];
        __nv_bfloat162 _bit_cast_src_222 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_222)[0];
        unsigned int send_1_3 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_3);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_3 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
        int m_local_r_1_1 = m1_local;
        int row_off_1_1 = o_row_base + m_local_r_1_1 * 128;
        __nv_bfloat162 _bit_cast_src_223 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_223)[0];
        __nv_bfloat162 _bit_cast_src_224 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_224)[0];
        unsigned int send_1_4 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_4);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_4 : o_pk[1]);
        reinterpret_cast<int2*>(O + row_off_1_1)[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_225 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_225)[0];
        __nv_bfloat162 _bit_cast_src_226 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_226)[0];
        unsigned int send_1_5 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_5);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_5 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_227 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_227)[0];
        __nv_bfloat162 _bit_cast_src_228 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_228)[0];
        unsigned int send_1_6 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_6);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_6 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_229 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_229)[0];
        __nv_bfloat162 _bit_cast_src_230 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_230)[0];
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
        __nv_bfloat162 _bit_cast_src_231 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_231)[0];
        __nv_bfloat162 _bit_cast_src_232 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_232)[0];
        unsigned int send_1_8 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send_1_8, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_8);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_8 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 64))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_233 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_233)[0];
        __nv_bfloat162 _bit_cast_src_234 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_234)[0];
        unsigned int send_1_9 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_1_9, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_9);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_9 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 80))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_235 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_235)[0];
        __nv_bfloat162 _bit_cast_src_236 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_236)[0];
        unsigned int send_1_10 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_1_10, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_10);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_10 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 96))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_237 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_237)[0];
        __nv_bfloat162 _bit_cast_src_238 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_238)[0];
        unsigned int send_1_11 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_1_11, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_11);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_11 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 112))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        int m_local_r_1_3 = m1_local;
        int row_off_1_3 = o_row_base_1 + m_local_r_1_3 * 128;
        __nv_bfloat162 _bit_cast_src_239 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_239)[0];
        __nv_bfloat162 _bit_cast_src_240 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_240)[0];
        unsigned int send_1_12 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_1_12, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_12);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_12 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 64))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_241 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_241)[0];
        __nv_bfloat162 _bit_cast_src_242 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_242)[0];
        unsigned int send_1_13 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_1_13, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_13);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_13 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 80))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_243 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_243)[0];
        __nv_bfloat162 _bit_cast_src_244 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_244)[0];
        unsigned int send_1_14 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_1_14, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_14);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_14 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 96))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_245 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_245)[0];
        __nv_bfloat162 _bit_cast_src_246 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_246)[0];
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
