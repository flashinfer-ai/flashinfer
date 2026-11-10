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
#define SMEM_VS_SMEM_OFF 41984
#define SMEM_VS_SMEM_STAGE_BYTES 512
#define SMEM_VS_SMEM_STRIDE 512
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 8192
#define SMEM_Q_SMEM_STRIDE 8192
#define SMEM_K_SMEM_OFF 9216
#define SMEM_K_SMEM_STAGE_BYTES 8192
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VL_SMEM_OFF 17408
#define SMEM_VL_SMEM_STAGE_BYTES 8192
#define SMEM_VL_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 25600
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 42496
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_vsa_sm90_355b9eaa1f9bc36a2dc4(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ Qs, float* __restrict__ Ks, float* __restrict__ Vs, int k_scale_groups, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2)
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
    #define v_full0_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* vs_smem = reinterpret_cast<float*>(smem_raw + 41984);
    const int vs_smem_addr = smem + 41984;
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* k_smem = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int k_smem_addr = smem + 9216;
    uint8_t* vl_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int vl_smem_addr = smem + 17408;
    uint8_t* vt_smem = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int vt_smem_addr = smem + 25600;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // v_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // === Task calls (declaration order) ===
    int item = bid;
    int mb = seqlen_q / 64;
    int plan_base = item * 2;
    int meta = plan[plan_base];
    int cnt = meta;
    int tile = item;
    int blk_base = plan_base + 1;
    int head = tile / mb;
    int qb = tile - head * mb;
    int q_row = head * seqlen_q + qb * 64;
    int kv_base = head * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            int blk_pre[1];
            blk_pre[0] = plan[blk_base];
            mbarrier_arrive_expect_tx(q_full_addr, 8192);
            tma_2d_gmem2smem(q_smem_addr, (&Q), 0, q_row, q_full_addr);
            if (cnt > 0) {
                int blk_row = kv_base + blk_pre[0] * 64;
                mbarrier_arrive_expect_tx(k_full0_addr, 8192);
                tma_2d_gmem2smem(k_smem_addr, (&K), 0, blk_row, k_full0_addr);
                mbarrier_arrive_expect_tx(v_full0_addr, 8192);
                tma_2d_gmem2smem(vl_smem_addr, (&V), 0, blk_row, v_full0_addr);
            }
        }
    }
    __syncthreads();
    int vsl_one = 0;
    int m0_local = warp * 16 + lane / 4;
    int m1_local = m0_local + 8;
    int vs_base = vsl_one + (lane & 3) * 2;
    float q_log2[2];
    q_log2[0] = Qs[q_row + m0_local] * scale_log2;
    q_log2[1] = Qs[q_row + m1_local] * scale_log2;
    float ks_regs[4];
    int ks_head = head * k_scale_groups;
    if (cnt > 0) {
        int ks_blk = plan[blk_base];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(Ks + (ks_head + ks_blk * 4) + 0);
            ks_regs[0 + 0] = _v4.x;
            ks_regs[0 + 1] = _v4.y;
            ks_regs[0 + 2] = _v4.z;
            ks_regs[0 + 3] = _v4.w;
        }
    }
    vs_smem[vsl_one + tid] = Vs[head * 128 + tid];
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
    int t_kq = warp >> 1;
    int t_kq4096 = t_kq * 4096;
    int t_kq2 = t_kq * 2;
    int t_dc0 = (warp & 1) * 4;
    unsigned int tw[16];
    unsigned int tpw[4];
    unsigned int _phase_q_full_0 = 0;
    mbarrier_wait(q_full_addr, _phase_q_full_0);
    _phase_q_full_0 ^= 1;
    uint64_t _wgmma_desc_1 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
    uint64_t _wgmma_desc_2 = (((uint64_t)(((k_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    uint64_t _wgmma_desc_3 = (((uint64_t)(((vt_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
    uint64_t _wgmma_desc_4 = (((uint64_t)(((vt_smem_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    if (cnt > 0) {
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
        float new_max0 = -CUDART_INF_F;
        float new_max1 = -CUDART_INF_F;
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
        new_max0 = m0;
        new_max1 = m1;
        row_max0 = new_max0;
        row_max1 = new_max1;
        float bias0 = 8.0f - row_max0;
        float bias1 = 8.0f - row_max1;
        float exp2 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0));
        float exp2_1 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0));
        d_qk[0] = exp2;
        d_qk[1] = exp2_1;
        row_sum0 += exp2 + exp2_1;
        float exp2_2 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[4], bias1));
        float exp2_3 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[4], bias1));
        d_qk[2] = exp2_2;
        d_qk[3] = exp2_3;
        row_sum1 += exp2_2 + exp2_3;
        float exp2_4 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0));
        float exp2_5 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0));
        d_qk[4] = exp2_4;
        d_qk[5] = exp2_5;
        row_sum0 += exp2_4 + exp2_5;
        float exp2_6 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[4], bias1));
        float exp2_7 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[4], bias1));
        d_qk[6] = exp2_6;
        d_qk[7] = exp2_7;
        row_sum1 += exp2_6 + exp2_7;
        float exp2_8 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0));
        float exp2_9 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0));
        d_qk[8] = exp2_8;
        d_qk[9] = exp2_9;
        row_sum0 += exp2_8 + exp2_9;
        float exp2_10 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[5], bias1));
        float exp2_11 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[5], bias1));
        d_qk[10] = exp2_10;
        d_qk[11] = exp2_11;
        row_sum1 += exp2_10 + exp2_11;
        float exp2_12 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0));
        float exp2_13 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0));
        d_qk[12] = exp2_12;
        d_qk[13] = exp2_13;
        row_sum0 += exp2_12 + exp2_13;
        float exp2_14 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[5], bias1));
        float exp2_15 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[5], bias1));
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
        asm volatile("barrier.sync 1, 128;" ::: "memory");
        if (gridDim.x <= 132) {
            float d_o_lo[32];
            float d_o_hi[32];
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1;\n}\n"
                : "+f"(d_o_lo[0]), "+f"(d_o_lo[1]), "+f"(d_o_lo[2]), "+f"(d_o_lo[3]), "+f"(d_o_lo[4]), "+f"(d_o_lo[5]), "+f"(d_o_lo[6]), "+f"(d_o_lo[7]), "+f"(d_o_lo[8]), "+f"(d_o_lo[9]), "+f"(d_o_lo[10]), "+f"(d_o_lo[11]), "+f"(d_o_lo[12]), "+f"(d_o_lo[13]), "+f"(d_o_lo[14]), "+f"(d_o_lo[15]), "+f"(d_o_lo[16]), "+f"(d_o_lo[17]), "+f"(d_o_lo[18]), "+f"(d_o_lo[19]), "+f"(d_o_lo[20]), "+f"(d_o_lo[21]), "+f"(d_o_lo[22]), "+f"(d_o_lo[23]), "+f"(d_o_lo[24]), "+f"(d_o_lo[25]), "+f"(d_o_lo[26]), "+f"(d_o_lo[27]), "+f"(d_o_lo[28]), "+f"(d_o_lo[29]), "+f"(d_o_lo[30]), "+f"(d_o_lo[31])
                : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(_wgmma_b_1)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1;\n}\n"
                : "+f"(d_o_hi[0]), "+f"(d_o_hi[1]), "+f"(d_o_hi[2]), "+f"(d_o_hi[3]), "+f"(d_o_hi[4]), "+f"(d_o_hi[5]), "+f"(d_o_hi[6]), "+f"(d_o_hi[7]), "+f"(d_o_hi[8]), "+f"(d_o_hi[9]), "+f"(d_o_hi[10]), "+f"(d_o_hi[11]), "+f"(d_o_hi[12]), "+f"(d_o_hi[13]), "+f"(d_o_hi[14]), "+f"(d_o_hi[15]), "+f"(d_o_hi[16]), "+f"(d_o_hi[17]), "+f"(d_o_hi[18]), "+f"(d_o_hi[19]), "+f"(d_o_hi[20]), "+f"(d_o_hi[21]), "+f"(d_o_hi[22]), "+f"(d_o_hi[23]), "+f"(d_o_hi[24]), "+f"(d_o_hi[25]), "+f"(d_o_hi[26]), "+f"(d_o_hi[27]), "+f"(d_o_hi[28]), "+f"(d_o_hi[29]), "+f"(d_o_hi[30]), "+f"(d_o_hi[31])
                : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(_wgmma_b_2)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            float _exp2_16 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0));
            float _exp2_17 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0));
            d_qk[16] = _exp2_16;
            d_qk[17] = _exp2_17;
            row_sum0 += _exp2_16 + _exp2_17;
            float _exp2_18 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1));
            float _exp2_19 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1));
            d_qk[18] = _exp2_18;
            d_qk[19] = _exp2_19;
            row_sum1 += _exp2_18 + _exp2_19;
            float _exp2_20 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0));
            float _exp2_21 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0));
            d_qk[20] = _exp2_20;
            d_qk[21] = _exp2_21;
            row_sum0 += _exp2_20 + _exp2_21;
            float _exp2_22 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1));
            float _exp2_23 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1));
            d_qk[22] = _exp2_22;
            d_qk[23] = _exp2_23;
            row_sum1 += _exp2_22 + _exp2_23;
            float _exp2_24 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0));
            float _exp2_25 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0));
            d_qk[24] = _exp2_24;
            d_qk[25] = _exp2_25;
            row_sum0 += _exp2_24 + _exp2_25;
            float _exp2_26 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1));
            float _exp2_27 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1));
            d_qk[26] = _exp2_26;
            d_qk[27] = _exp2_27;
            row_sum1 += _exp2_26 + _exp2_27;
            float _exp2_28 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0));
            float _exp2_29 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0));
            d_qk[28] = _exp2_28;
            d_qk[29] = _exp2_29;
            row_sum0 += _exp2_28 + _exp2_29;
            float _exp2_30 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1));
            float _exp2_31 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1));
            d_qk[30] = _exp2_30;
            d_qk[31] = _exp2_31;
            row_sum1 += _exp2_30 + _exp2_31;
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
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1;\n}\n"
                : "+f"(d_o_lo[0]), "+f"(d_o_lo[1]), "+f"(d_o_lo[2]), "+f"(d_o_lo[3]), "+f"(d_o_lo[4]), "+f"(d_o_lo[5]), "+f"(d_o_lo[6]), "+f"(d_o_lo[7]), "+f"(d_o_lo[8]), "+f"(d_o_lo[9]), "+f"(d_o_lo[10]), "+f"(d_o_lo[11]), "+f"(d_o_lo[12]), "+f"(d_o_lo[13]), "+f"(d_o_lo[14]), "+f"(d_o_lo[15]), "+f"(d_o_lo[16]), "+f"(d_o_lo[17]), "+f"(d_o_lo[18]), "+f"(d_o_lo[19]), "+f"(d_o_lo[20]), "+f"(d_o_lo[21]), "+f"(d_o_lo[22]), "+f"(d_o_lo[23]), "+f"(d_o_lo[24]), "+f"(d_o_lo[25]), "+f"(d_o_lo[26]), "+f"(d_o_lo[27]), "+f"(d_o_lo[28]), "+f"(d_o_lo[29]), "+f"(d_o_lo[30]), "+f"(d_o_lo[31])
                : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 2)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1;\n}\n"
                : "+f"(d_o_hi[0]), "+f"(d_o_hi[1]), "+f"(d_o_hi[2]), "+f"(d_o_hi[3]), "+f"(d_o_hi[4]), "+f"(d_o_hi[5]), "+f"(d_o_hi[6]), "+f"(d_o_hi[7]), "+f"(d_o_hi[8]), "+f"(d_o_hi[9]), "+f"(d_o_hi[10]), "+f"(d_o_hi[11]), "+f"(d_o_hi[12]), "+f"(d_o_hi[13]), "+f"(d_o_hi[14]), "+f"(d_o_hi[15]), "+f"(d_o_hi[16]), "+f"(d_o_hi[17]), "+f"(d_o_hi[18]), "+f"(d_o_hi[19]), "+f"(d_o_hi[20]), "+f"(d_o_hi[21]), "+f"(d_o_hi[22]), "+f"(d_o_hi[23]), "+f"(d_o_hi[24]), "+f"(d_o_hi[25]), "+f"(d_o_hi[26]), "+f"(d_o_hi[27]), "+f"(d_o_hi[28]), "+f"(d_o_hi[29]), "+f"(d_o_hi[30]), "+f"(d_o_hi[31])
                : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_2 + 2)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
            row_sum0 += _shfl_xor_sync_53;
            float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
            row_sum0 += _shfl_xor_sync_54;
            float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
            row_sum1 += _shfl_xor_sync_55;
            float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
            row_sum1 += _shfl_xor_sync_56;
            float _rcp_0 = approx_rcp(row_sum0);
            float _rcp_1 = approx_rcp(row_sum1);
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
            float vs0_1_0 = vs_smem[vs_base];
            float vs1_2_0 = vs_smem[vs_base + 1];
            d_o_lo[0] = d_o_lo[0] * (vs0_1_0 * _rcp_0);
            d_o_lo[1] = d_o_lo[1] * (vs1_2_0 * _rcp_0);
            d_o_lo[2] = d_o_lo[2] * (vs0_1_0 * _rcp_1);
            d_o_lo[3] = d_o_lo[3] * (vs1_2_0 * _rcp_1);
            float vs0_1_1 = vs_smem[vs_base + 8];
            float vs1_2_1 = vs_smem[vs_base + 8 + 1];
            d_o_lo[4] = d_o_lo[4] * (vs0_1_1 * _rcp_0);
            d_o_lo[5] = d_o_lo[5] * (vs1_2_1 * _rcp_0);
            d_o_lo[6] = d_o_lo[6] * (vs0_1_1 * _rcp_1);
            d_o_lo[7] = d_o_lo[7] * (vs1_2_1 * _rcp_1);
            float vs0_1_2 = vs_smem[vs_base + 16];
            float vs1_2_2 = vs_smem[vs_base + 16 + 1];
            d_o_lo[8] = d_o_lo[8] * (vs0_1_2 * _rcp_0);
            d_o_lo[9] = d_o_lo[9] * (vs1_2_2 * _rcp_0);
            d_o_lo[10] = d_o_lo[10] * (vs0_1_2 * _rcp_1);
            d_o_lo[11] = d_o_lo[11] * (vs1_2_2 * _rcp_1);
            float vs0_1_3 = vs_smem[vs_base + 24];
            float vs1_2_3 = vs_smem[vs_base + 24 + 1];
            d_o_lo[12] = d_o_lo[12] * (vs0_1_3 * _rcp_0);
            d_o_lo[13] = d_o_lo[13] * (vs1_2_3 * _rcp_0);
            d_o_lo[14] = d_o_lo[14] * (vs0_1_3 * _rcp_1);
            d_o_lo[15] = d_o_lo[15] * (vs1_2_3 * _rcp_1);
            float vs0_1_4 = vs_smem[vs_base + 32];
            float vs1_2_4 = vs_smem[vs_base + 32 + 1];
            d_o_lo[16] = d_o_lo[16] * (vs0_1_4 * _rcp_0);
            d_o_lo[17] = d_o_lo[17] * (vs1_2_4 * _rcp_0);
            d_o_lo[18] = d_o_lo[18] * (vs0_1_4 * _rcp_1);
            d_o_lo[19] = d_o_lo[19] * (vs1_2_4 * _rcp_1);
            float vs0_1_5 = vs_smem[vs_base + 40];
            float vs1_2_5 = vs_smem[vs_base + 40 + 1];
            d_o_lo[20] = d_o_lo[20] * (vs0_1_5 * _rcp_0);
            d_o_lo[21] = d_o_lo[21] * (vs1_2_5 * _rcp_0);
            d_o_lo[22] = d_o_lo[22] * (vs0_1_5 * _rcp_1);
            d_o_lo[23] = d_o_lo[23] * (vs1_2_5 * _rcp_1);
            float vs0_1_6 = vs_smem[vs_base + 48];
            float vs1_2_6 = vs_smem[vs_base + 48 + 1];
            d_o_lo[24] = d_o_lo[24] * (vs0_1_6 * _rcp_0);
            d_o_lo[25] = d_o_lo[25] * (vs1_2_6 * _rcp_0);
            d_o_lo[26] = d_o_lo[26] * (vs0_1_6 * _rcp_1);
            d_o_lo[27] = d_o_lo[27] * (vs1_2_6 * _rcp_1);
            float vs0_1_7 = vs_smem[vs_base + 56];
            float vs1_2_7 = vs_smem[vs_base + 56 + 1];
            d_o_lo[28] = d_o_lo[28] * (vs0_1_7 * _rcp_0);
            d_o_lo[29] = d_o_lo[29] * (vs1_2_7 * _rcp_0);
            d_o_lo[30] = d_o_lo[30] * (vs0_1_7 * _rcp_1);
            d_o_lo[31] = d_o_lo[31] * (vs1_2_7 * _rcp_1);
            int qj = lane & 3;
            int qodd = qj & 1;
            unsigned int o_pk[2];
            unsigned int o_vec[2];
            int o_row_base = q_row * 128 + qj * 2 + qodd * 6;
            int m_local_r_1_0 = m0_local;
            int row_off_1_0 = o_row_base + m_local_r_1_0 * 128;
            __nv_bfloat162 _bit_cast_src_57 = __float22bfloat162_rn(make_float2(d_o_lo[0], d_o_lo[1]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_57)[0];
            __nv_bfloat162 _bit_cast_src_58 = __float22bfloat162_rn(make_float2(d_o_lo[4], d_o_lo[5]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_58)[0];
            unsigned int send_1_0 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_0);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_0 : o_pk[1]);
            reinterpret_cast<int2*>(O + row_off_1_0)[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_59 = __float22bfloat162_rn(make_float2(d_o_lo[8], d_o_lo[9]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_59)[0];
            __nv_bfloat162 _bit_cast_src_60 = __float22bfloat162_rn(make_float2(d_o_lo[12], d_o_lo[13]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_60)[0];
            unsigned int send_1_1 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_1);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_1 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_0 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_61 = __float22bfloat162_rn(make_float2(d_o_lo[16], d_o_lo[17]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_61)[0];
            __nv_bfloat162 _bit_cast_src_62 = __float22bfloat162_rn(make_float2(d_o_lo[20], d_o_lo[21]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_62)[0];
            unsigned int send_1_2 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_2);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_2 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_0 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_63 = __float22bfloat162_rn(make_float2(d_o_lo[24], d_o_lo[25]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_63)[0];
            __nv_bfloat162 _bit_cast_src_64 = __float22bfloat162_rn(make_float2(d_o_lo[28], d_o_lo[29]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_64)[0];
            unsigned int send_1_3 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_3);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_3 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_0 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
            int m_local_r_1_1 = m1_local;
            int row_off_1_1 = o_row_base + m_local_r_1_1 * 128;
            __nv_bfloat162 _bit_cast_src_65 = __float22bfloat162_rn(make_float2(d_o_lo[2], d_o_lo[3]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_65)[0];
            __nv_bfloat162 _bit_cast_src_66 = __float22bfloat162_rn(make_float2(d_o_lo[6], d_o_lo[7]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_66)[0];
            unsigned int send_1_4 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_4);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_4 : o_pk[1]);
            reinterpret_cast<int2*>(O + row_off_1_1)[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_67 = __float22bfloat162_rn(make_float2(d_o_lo[10], d_o_lo[11]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_67)[0];
            __nv_bfloat162 _bit_cast_src_68 = __float22bfloat162_rn(make_float2(d_o_lo[14], d_o_lo[15]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_68)[0];
            unsigned int send_1_5 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_5);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_5 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_1 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_69 = __float22bfloat162_rn(make_float2(d_o_lo[18], d_o_lo[19]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_69)[0];
            __nv_bfloat162 _bit_cast_src_70 = __float22bfloat162_rn(make_float2(d_o_lo[22], d_o_lo[23]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_70)[0];
            unsigned int send_1_6 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_6);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_6 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_1 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_71 = __float22bfloat162_rn(make_float2(d_o_lo[26], d_o_lo[27]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_71)[0];
            __nv_bfloat162 _bit_cast_src_72 = __float22bfloat162_rn(make_float2(d_o_lo[30], d_o_lo[31]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_72)[0];
            unsigned int send_1_7 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_1_7, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_7);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_7 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_1 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            float vs0_1_8 = vs_smem[vs_base + 64];
            float vs1_2_8 = vs_smem[vs_base + 64 + 1];
            d_o_hi[0] = d_o_hi[0] * (vs0_1_8 * _rcp_0);
            d_o_hi[1] = d_o_hi[1] * (vs1_2_8 * _rcp_0);
            d_o_hi[2] = d_o_hi[2] * (vs0_1_8 * _rcp_1);
            d_o_hi[3] = d_o_hi[3] * (vs1_2_8 * _rcp_1);
            float vs0_1_9 = vs_smem[vs_base + 64 + 8];
            float vs1_2_9 = vs_smem[vs_base + 64 + 8 + 1];
            d_o_hi[4] = d_o_hi[4] * (vs0_1_9 * _rcp_0);
            d_o_hi[5] = d_o_hi[5] * (vs1_2_9 * _rcp_0);
            d_o_hi[6] = d_o_hi[6] * (vs0_1_9 * _rcp_1);
            d_o_hi[7] = d_o_hi[7] * (vs1_2_9 * _rcp_1);
            float vs0_1_10 = vs_smem[vs_base + 64 + 16];
            float vs1_2_10 = vs_smem[vs_base + 64 + 16 + 1];
            d_o_hi[8] = d_o_hi[8] * (vs0_1_10 * _rcp_0);
            d_o_hi[9] = d_o_hi[9] * (vs1_2_10 * _rcp_0);
            d_o_hi[10] = d_o_hi[10] * (vs0_1_10 * _rcp_1);
            d_o_hi[11] = d_o_hi[11] * (vs1_2_10 * _rcp_1);
            float vs0_1_11 = vs_smem[vs_base + 64 + 24];
            float vs1_2_11 = vs_smem[vs_base + 64 + 24 + 1];
            d_o_hi[12] = d_o_hi[12] * (vs0_1_11 * _rcp_0);
            d_o_hi[13] = d_o_hi[13] * (vs1_2_11 * _rcp_0);
            d_o_hi[14] = d_o_hi[14] * (vs0_1_11 * _rcp_1);
            d_o_hi[15] = d_o_hi[15] * (vs1_2_11 * _rcp_1);
            float vs0_1_12 = vs_smem[vs_base + 64 + 32];
            float vs1_2_12 = vs_smem[vs_base + 64 + 32 + 1];
            d_o_hi[16] = d_o_hi[16] * (vs0_1_12 * _rcp_0);
            d_o_hi[17] = d_o_hi[17] * (vs1_2_12 * _rcp_0);
            d_o_hi[18] = d_o_hi[18] * (vs0_1_12 * _rcp_1);
            d_o_hi[19] = d_o_hi[19] * (vs1_2_12 * _rcp_1);
            float vs0_1_13 = vs_smem[vs_base + 64 + 40];
            float vs1_2_13 = vs_smem[vs_base + 64 + 40 + 1];
            d_o_hi[20] = d_o_hi[20] * (vs0_1_13 * _rcp_0);
            d_o_hi[21] = d_o_hi[21] * (vs1_2_13 * _rcp_0);
            d_o_hi[22] = d_o_hi[22] * (vs0_1_13 * _rcp_1);
            d_o_hi[23] = d_o_hi[23] * (vs1_2_13 * _rcp_1);
            float vs0_1_14 = vs_smem[vs_base + 64 + 48];
            float vs1_2_14 = vs_smem[vs_base + 64 + 48 + 1];
            d_o_hi[24] = d_o_hi[24] * (vs0_1_14 * _rcp_0);
            d_o_hi[25] = d_o_hi[25] * (vs1_2_14 * _rcp_0);
            d_o_hi[26] = d_o_hi[26] * (vs0_1_14 * _rcp_1);
            d_o_hi[27] = d_o_hi[27] * (vs1_2_14 * _rcp_1);
            float vs0_1_15 = vs_smem[vs_base + 64 + 56];
            float vs1_2_15 = vs_smem[vs_base + 64 + 56 + 1];
            d_o_hi[28] = d_o_hi[28] * (vs0_1_15 * _rcp_0);
            d_o_hi[29] = d_o_hi[29] * (vs1_2_15 * _rcp_0);
            d_o_hi[30] = d_o_hi[30] * (vs0_1_15 * _rcp_1);
            d_o_hi[31] = d_o_hi[31] * (vs1_2_15 * _rcp_1);
            int qj_39 = lane & 3;
            int qodd_40 = qj_39 & 1;
            unsigned int o_pk_41[2];
            unsigned int o_vec_42[2];
            int o_row_base_43 = q_row * 128 + 64 + qj_39 * 2 + qodd_40 * 6;
            int m_local_r_1_2 = m0_local;
            int row_off_1_2 = o_row_base_43 + m_local_r_1_2 * 128;
            __nv_bfloat162 _bit_cast_src_73 = __float22bfloat162_rn(make_float2(d_o_hi[0], d_o_hi[1]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_73)[0];
            __nv_bfloat162 _bit_cast_src_74 = __float22bfloat162_rn(make_float2(d_o_hi[4], d_o_hi[5]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_74)[0];
            unsigned int send_1_8 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send_1_8, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_8);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_8 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + row_off_1_2)[0] = reinterpret_cast<int2*>(o_vec_42)[0];
            __nv_bfloat162 _bit_cast_src_75 = __float22bfloat162_rn(make_float2(d_o_hi[8], d_o_hi[9]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_75)[0];
            __nv_bfloat162 _bit_cast_src_76 = __float22bfloat162_rn(make_float2(d_o_hi[12], d_o_hi[13]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_76)[0];
            unsigned int send_1_9 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_1_9, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_9);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_9 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + (row_off_1_2 + 16))[0] = reinterpret_cast<int2*>(o_vec_42)[0];
            __nv_bfloat162 _bit_cast_src_77 = __float22bfloat162_rn(make_float2(d_o_hi[16], d_o_hi[17]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_77)[0];
            __nv_bfloat162 _bit_cast_src_78 = __float22bfloat162_rn(make_float2(d_o_hi[20], d_o_hi[21]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_78)[0];
            unsigned int send_1_10 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_1_10, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_10);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_10 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + (row_off_1_2 + 32))[0] = reinterpret_cast<int2*>(o_vec_42)[0];
            __nv_bfloat162 _bit_cast_src_79 = __float22bfloat162_rn(make_float2(d_o_hi[24], d_o_hi[25]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_79)[0];
            __nv_bfloat162 _bit_cast_src_80 = __float22bfloat162_rn(make_float2(d_o_hi[28], d_o_hi[29]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_80)[0];
            unsigned int send_1_11 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_1_11, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_11);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_11 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + (row_off_1_2 + 48))[0] = reinterpret_cast<int2*>(o_vec_42)[0];
            int m_local_r_1_3 = m1_local;
            int row_off_1_3 = o_row_base_43 + m_local_r_1_3 * 128;
            __nv_bfloat162 _bit_cast_src_81 = __float22bfloat162_rn(make_float2(d_o_hi[2], d_o_hi[3]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_81)[0];
            __nv_bfloat162 _bit_cast_src_82 = __float22bfloat162_rn(make_float2(d_o_hi[6], d_o_hi[7]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_82)[0];
            unsigned int send_1_12 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_1_12, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_12);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_12 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + row_off_1_3)[0] = reinterpret_cast<int2*>(o_vec_42)[0];
            __nv_bfloat162 _bit_cast_src_83 = __float22bfloat162_rn(make_float2(d_o_hi[10], d_o_hi[11]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_83)[0];
            __nv_bfloat162 _bit_cast_src_84 = __float22bfloat162_rn(make_float2(d_o_hi[14], d_o_hi[15]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_84)[0];
            unsigned int send_1_13 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_1_13, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_13);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_13 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + (row_off_1_3 + 16))[0] = reinterpret_cast<int2*>(o_vec_42)[0];
            __nv_bfloat162 _bit_cast_src_85 = __float22bfloat162_rn(make_float2(d_o_hi[18], d_o_hi[19]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_85)[0];
            __nv_bfloat162 _bit_cast_src_86 = __float22bfloat162_rn(make_float2(d_o_hi[22], d_o_hi[23]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_86)[0];
            unsigned int send_1_14 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_1_14, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_14);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_14 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + (row_off_1_3 + 32))[0] = reinterpret_cast<int2*>(o_vec_42)[0];
            __nv_bfloat162 _bit_cast_src_87 = __float22bfloat162_rn(make_float2(d_o_hi[26], d_o_hi[27]));
            o_pk_41[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_87)[0];
            __nv_bfloat162 _bit_cast_src_88 = __float22bfloat162_rn(make_float2(d_o_hi[30], d_o_hi[31]));
            o_pk_41[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_88)[0];
            unsigned int send_1_15 = ((qodd_40 != 0) ? o_pk_41[0] : o_pk_41[1]);
            unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_1_15, 1);
            o_vec_42[0] = ((qodd_40 == 0) ? o_pk_41[0] : _shfl_xor_15);
            o_vec_42[1] = ((qodd_40 == 0) ? _shfl_xor_15 : o_pk_41[1]);
            reinterpret_cast<int2*>(O + (row_off_1_3 + 48))[0] = reinterpret_cast<int2*>(o_vec_42)[0];
        } else {
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1;\n}\n"
                : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(_wgmma_b_1)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            float _exp2_32 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0));
            float _exp2_33 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0));
            d_qk[16] = _exp2_32;
            d_qk[17] = _exp2_33;
            row_sum0 += _exp2_32 + _exp2_33;
            float _exp2_34 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[6], bias1));
            float _exp2_35 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[6], bias1));
            d_qk[18] = _exp2_34;
            d_qk[19] = _exp2_35;
            row_sum1 += _exp2_34 + _exp2_35;
            float _exp2_36 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0));
            float _exp2_37 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0));
            d_qk[20] = _exp2_36;
            d_qk[21] = _exp2_37;
            row_sum0 += _exp2_36 + _exp2_37;
            float _exp2_38 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[6], bias1));
            float _exp2_39 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[6], bias1));
            d_qk[22] = _exp2_38;
            d_qk[23] = _exp2_39;
            row_sum1 += _exp2_38 + _exp2_39;
            float _exp2_40 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0));
            float _exp2_41 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0));
            d_qk[24] = _exp2_40;
            d_qk[25] = _exp2_41;
            row_sum0 += _exp2_40 + _exp2_41;
            float _exp2_42 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[7], bias1));
            float _exp2_43 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[7], bias1));
            d_qk[26] = _exp2_42;
            d_qk[27] = _exp2_43;
            row_sum1 += _exp2_42 + _exp2_43;
            float _exp2_44 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0));
            float _exp2_45 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0));
            d_qk[28] = _exp2_44;
            d_qk[29] = _exp2_45;
            row_sum0 += _exp2_44 + _exp2_45;
            float _exp2_46 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[7], bias1));
            float _exp2_47 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[7], bias1));
            d_qk[30] = _exp2_46;
            d_qk[31] = _exp2_47;
            row_sum1 += _exp2_46 + _exp2_47;
            uint16_t _cvt_satfinite_e4m3x2_f32_90;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_90) : "f"(d_qk[17]), "f"(d_qk[16]));
            uint16_t _cvt_satfinite_e4m3x2_f32_91;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_91) : "f"(d_qk[21]), "f"(d_qk[20]));
            uint32_t _prmt_89;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_89) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_90), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_91));
            p_e4m3[4] = _prmt_89;
            uint16_t _cvt_satfinite_e4m3x2_f32_93;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_93) : "f"(d_qk[19]), "f"(d_qk[18]));
            uint16_t _cvt_satfinite_e4m3x2_f32_94;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_94) : "f"(d_qk[23]), "f"(d_qk[22]));
            uint32_t _prmt_92;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_92) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_93), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_94));
            p_e4m3[5] = _prmt_92;
            uint16_t _cvt_satfinite_e4m3x2_f32_96;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_96) : "f"(d_qk[25]), "f"(d_qk[24]));
            uint16_t _cvt_satfinite_e4m3x2_f32_97;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_97) : "f"(d_qk[29]), "f"(d_qk[28]));
            uint32_t _prmt_95;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_95) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_96), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_97));
            p_e4m3[6] = _prmt_95;
            uint16_t _cvt_satfinite_e4m3x2_f32_99;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_99) : "f"(d_qk[27]), "f"(d_qk[26]));
            uint16_t _cvt_satfinite_e4m3x2_f32_100;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_100) : "f"(d_qk[31]), "f"(d_qk[30]));
            uint32_t _prmt_98;
            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_98) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_99), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_100));
            p_e4m3[7] = _prmt_98;
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(_wgmma_b_1 + 2)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
            row_sum0 += _shfl_xor_sync_101;
            float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
            row_sum0 += _shfl_xor_sync_102;
            float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
            row_sum1 += _shfl_xor_sync_103;
            float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
            row_sum1 += _shfl_xor_sync_104;
            int qj_k1 = lane & 3;
            float _rcp_2 = approx_rcp(row_sum0);
            float _rcp_3 = approx_rcp(row_sum1);
            float vs0_1_16 = vs_smem[vs_base];
            float vs1_2_16 = vs_smem[vs_base + 1];
            d_o[0] = d_o[0] * (vs0_1_16 * _rcp_2);
            d_o[1] = d_o[1] * (vs1_2_16 * _rcp_2);
            d_o[2] = d_o[2] * (vs0_1_16 * _rcp_3);
            d_o[3] = d_o[3] * (vs1_2_16 * _rcp_3);
            float vs0_1_17 = vs_smem[vs_base + 8];
            float vs1_2_17 = vs_smem[vs_base + 8 + 1];
            d_o[4] = d_o[4] * (vs0_1_17 * _rcp_2);
            d_o[5] = d_o[5] * (vs1_2_17 * _rcp_2);
            d_o[6] = d_o[6] * (vs0_1_17 * _rcp_3);
            d_o[7] = d_o[7] * (vs1_2_17 * _rcp_3);
            float vs0_1_18 = vs_smem[vs_base + 16];
            float vs1_2_18 = vs_smem[vs_base + 16 + 1];
            d_o[8] = d_o[8] * (vs0_1_18 * _rcp_2);
            d_o[9] = d_o[9] * (vs1_2_18 * _rcp_2);
            d_o[10] = d_o[10] * (vs0_1_18 * _rcp_3);
            d_o[11] = d_o[11] * (vs1_2_18 * _rcp_3);
            float vs0_1_19 = vs_smem[vs_base + 24];
            float vs1_2_19 = vs_smem[vs_base + 24 + 1];
            d_o[12] = d_o[12] * (vs0_1_19 * _rcp_2);
            d_o[13] = d_o[13] * (vs1_2_19 * _rcp_2);
            d_o[14] = d_o[14] * (vs0_1_19 * _rcp_3);
            d_o[15] = d_o[15] * (vs1_2_19 * _rcp_3);
            float vs0_1_20 = vs_smem[vs_base + 32];
            float vs1_2_20 = vs_smem[vs_base + 32 + 1];
            d_o[16] = d_o[16] * (vs0_1_20 * _rcp_2);
            d_o[17] = d_o[17] * (vs1_2_20 * _rcp_2);
            d_o[18] = d_o[18] * (vs0_1_20 * _rcp_3);
            d_o[19] = d_o[19] * (vs1_2_20 * _rcp_3);
            float vs0_1_21 = vs_smem[vs_base + 40];
            float vs1_2_21 = vs_smem[vs_base + 40 + 1];
            d_o[20] = d_o[20] * (vs0_1_21 * _rcp_2);
            d_o[21] = d_o[21] * (vs1_2_21 * _rcp_2);
            d_o[22] = d_o[22] * (vs0_1_21 * _rcp_3);
            d_o[23] = d_o[23] * (vs1_2_21 * _rcp_3);
            float vs0_1_22 = vs_smem[vs_base + 48];
            float vs1_2_22 = vs_smem[vs_base + 48 + 1];
            d_o[24] = d_o[24] * (vs0_1_22 * _rcp_2);
            d_o[25] = d_o[25] * (vs1_2_22 * _rcp_2);
            d_o[26] = d_o[26] * (vs0_1_22 * _rcp_3);
            d_o[27] = d_o[27] * (vs1_2_22 * _rcp_3);
            float vs0_1_23 = vs_smem[vs_base + 56];
            float vs1_2_23 = vs_smem[vs_base + 56 + 1];
            d_o[28] = d_o[28] * (vs0_1_23 * _rcp_2);
            d_o[29] = d_o[29] * (vs1_2_23 * _rcp_2);
            d_o[30] = d_o[30] * (vs0_1_23 * _rcp_3);
            d_o[31] = d_o[31] * (vs1_2_23 * _rcp_3);
            float vs0_1_24 = vs_smem[vs_base + 64];
            float vs1_2_24 = vs_smem[vs_base + 64 + 1];
            d_o[32] = d_o[32] * (vs0_1_24 * _rcp_2);
            d_o[33] = d_o[33] * (vs1_2_24 * _rcp_2);
            d_o[34] = d_o[34] * (vs0_1_24 * _rcp_3);
            d_o[35] = d_o[35] * (vs1_2_24 * _rcp_3);
            float vs0_1_25 = vs_smem[vs_base + 72];
            float vs1_2_25 = vs_smem[vs_base + 72 + 1];
            d_o[36] = d_o[36] * (vs0_1_25 * _rcp_2);
            d_o[37] = d_o[37] * (vs1_2_25 * _rcp_2);
            d_o[38] = d_o[38] * (vs0_1_25 * _rcp_3);
            d_o[39] = d_o[39] * (vs1_2_25 * _rcp_3);
            float vs0_1_26 = vs_smem[vs_base + 80];
            float vs1_2_26 = vs_smem[vs_base + 80 + 1];
            d_o[40] = d_o[40] * (vs0_1_26 * _rcp_2);
            d_o[41] = d_o[41] * (vs1_2_26 * _rcp_2);
            d_o[42] = d_o[42] * (vs0_1_26 * _rcp_3);
            d_o[43] = d_o[43] * (vs1_2_26 * _rcp_3);
            float vs0_1_27 = vs_smem[vs_base + 88];
            float vs1_2_27 = vs_smem[vs_base + 88 + 1];
            d_o[44] = d_o[44] * (vs0_1_27 * _rcp_2);
            d_o[45] = d_o[45] * (vs1_2_27 * _rcp_2);
            d_o[46] = d_o[46] * (vs0_1_27 * _rcp_3);
            d_o[47] = d_o[47] * (vs1_2_27 * _rcp_3);
            float vs0_1_28 = vs_smem[vs_base + 96];
            float vs1_2_28 = vs_smem[vs_base + 96 + 1];
            d_o[48] = d_o[48] * (vs0_1_28 * _rcp_2);
            d_o[49] = d_o[49] * (vs1_2_28 * _rcp_2);
            d_o[50] = d_o[50] * (vs0_1_28 * _rcp_3);
            d_o[51] = d_o[51] * (vs1_2_28 * _rcp_3);
            float vs0_1_29 = vs_smem[vs_base + 104];
            float vs1_2_29 = vs_smem[vs_base + 104 + 1];
            d_o[52] = d_o[52] * (vs0_1_29 * _rcp_2);
            d_o[53] = d_o[53] * (vs1_2_29 * _rcp_2);
            d_o[54] = d_o[54] * (vs0_1_29 * _rcp_3);
            d_o[55] = d_o[55] * (vs1_2_29 * _rcp_3);
            float vs0_1_30 = vs_smem[vs_base + 112];
            float vs1_2_30 = vs_smem[vs_base + 112 + 1];
            d_o[56] = d_o[56] * (vs0_1_30 * _rcp_2);
            d_o[57] = d_o[57] * (vs1_2_30 * _rcp_2);
            d_o[58] = d_o[58] * (vs0_1_30 * _rcp_3);
            d_o[59] = d_o[59] * (vs1_2_30 * _rcp_3);
            float vs0_1_31 = vs_smem[vs_base + 120];
            float vs1_2_31 = vs_smem[vs_base + 120 + 1];
            d_o[60] = d_o[60] * (vs0_1_31 * _rcp_2);
            d_o[61] = d_o[61] * (vs1_2_31 * _rcp_2);
            d_o[62] = d_o[62] * (vs0_1_31 * _rcp_3);
            d_o[63] = d_o[63] * (vs1_2_31 * _rcp_3);
            int qj1 = qj_k1 & 1;
            int qj2 = qj_k1 & 2;
            unsigned int o_vec_1[4];
            unsigned int o_tmp[4];
            int o_row_base_1 = q_row * 128;
            int m_local_r_1_4 = m0_local;
            __nv_bfloat162 _bit_cast_src_105 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_105)[0];
            __nv_bfloat162 _bit_cast_src_106 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_106)[0];
            __nv_bfloat162 _bit_cast_src_107 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_107)[0];
            __nv_bfloat162 _bit_cast_src_108 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_108)[0];
            unsigned int _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_109;
            unsigned int _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_110;
            unsigned int _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_111;
            unsigned int _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_112;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_113;
            unsigned int _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_114;
            unsigned int _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_115;
            unsigned int _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_116;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_0 = o_row_base_1 + m_local_r_1_4 * 128 + qj_k1 * 8;
            reinterpret_cast<int4*>(O + o_off_1_0)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_117 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_117)[0];
            __nv_bfloat162 _bit_cast_src_118 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_118)[0];
            __nv_bfloat162 _bit_cast_src_119 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_119)[0];
            __nv_bfloat162 _bit_cast_src_120 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_120)[0];
            unsigned int _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_121;
            unsigned int _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_122;
            unsigned int _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_123;
            unsigned int _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_124;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_125;
            unsigned int _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_126;
            unsigned int _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_127;
            unsigned int _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_128;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_1 = o_row_base_1 + m_local_r_1_4 * 128 + (4 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_1_1)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_129 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
            __nv_bfloat162 _bit_cast_src_130 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
            __nv_bfloat162 _bit_cast_src_131 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_131)[0];
            __nv_bfloat162 _bit_cast_src_132 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_132)[0];
            unsigned int _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_133;
            unsigned int _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_134;
            unsigned int _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_135;
            unsigned int _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_136;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_137;
            unsigned int _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_138;
            unsigned int _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_139;
            unsigned int _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_140;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_2 = o_row_base_1 + m_local_r_1_4 * 128 + (8 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_1_2)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_141 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_141)[0];
            __nv_bfloat162 _bit_cast_src_142 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_142)[0];
            __nv_bfloat162 _bit_cast_src_143 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_143)[0];
            __nv_bfloat162 _bit_cast_src_144 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_144)[0];
            unsigned int _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_145;
            unsigned int _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_146;
            unsigned int _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_147;
            unsigned int _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_148;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_149;
            unsigned int _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_150;
            unsigned int _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_151;
            unsigned int _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_152;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_3 = o_row_base_1 + m_local_r_1_4 * 128 + (12 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_1_3)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            int m_local_r_1_5 = m1_local;
            __nv_bfloat162 _bit_cast_src_153 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_153)[0];
            __nv_bfloat162 _bit_cast_src_154 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_154)[0];
            __nv_bfloat162 _bit_cast_src_155 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_155)[0];
            __nv_bfloat162 _bit_cast_src_156 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_156)[0];
            unsigned int _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_157;
            unsigned int _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_158;
            unsigned int _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_159;
            unsigned int _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_160;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_161;
            unsigned int _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_162;
            unsigned int _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_163;
            unsigned int _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_164;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_4 = o_row_base_1 + m_local_r_1_5 * 128 + qj_k1 * 8;
            reinterpret_cast<int4*>(O + o_off_1_4)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_165 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_165)[0];
            __nv_bfloat162 _bit_cast_src_166 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_166)[0];
            __nv_bfloat162 _bit_cast_src_167 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_167)[0];
            __nv_bfloat162 _bit_cast_src_168 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_168)[0];
            unsigned int _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_169;
            unsigned int _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_170;
            unsigned int _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_171;
            unsigned int _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_172;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_173;
            unsigned int _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_174;
            unsigned int _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_175;
            unsigned int _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_176;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_5 = o_row_base_1 + m_local_r_1_5 * 128 + (4 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_1_5)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_177 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_177)[0];
            __nv_bfloat162 _bit_cast_src_178 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_178)[0];
            __nv_bfloat162 _bit_cast_src_179 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_179)[0];
            __nv_bfloat162 _bit_cast_src_180 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_180)[0];
            unsigned int _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_181;
            unsigned int _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_182;
            unsigned int _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_183;
            unsigned int _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_184;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_185;
            unsigned int _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_186;
            unsigned int _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_187;
            unsigned int _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_188;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_6 = o_row_base_1 + m_local_r_1_5 * 128 + (8 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_1_6)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_189 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_189)[0];
            __nv_bfloat162 _bit_cast_src_190 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_190)[0];
            __nv_bfloat162 _bit_cast_src_191 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_191)[0];
            __nv_bfloat162 _bit_cast_src_192 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_192)[0];
            unsigned int _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_193;
            unsigned int _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_194;
            unsigned int _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_195;
            unsigned int _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_196;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_197;
            unsigned int _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_198;
            unsigned int _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_199;
            unsigned int _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_200;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_1_7 = o_row_base_1 + m_local_r_1_5 * 128 + (12 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_1_7)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
        }
    }
    int qj_1 = lane & 3;

    // Cleanup
    __syncthreads();
}

} // extern "C"
