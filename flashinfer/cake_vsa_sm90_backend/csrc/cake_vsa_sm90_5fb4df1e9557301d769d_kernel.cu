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
#define SMEM_MSTATS_OFF 148480
#define SMEM_MSTATS_STAGE_BYTES 1024
#define SMEM_MSTATS_STRIDE 1024
#define SMEM_PART_LO_OFF 66560
#define SMEM_PART_LO_STAGE_BYTES 16384
#define SMEM_PART_LO_STRIDE 16384
#define SMEM_PART_HI_OFF 132096
#define SMEM_PART_HI_STAGE_BYTES 16384
#define SMEM_PART_HI_STRIDE 16384
#define SMEM_PART_HI0_OFF 50176
#define SMEM_PART_HI0_STAGE_BYTES 16384
#define SMEM_PART_HI0_STRIDE 16384
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 82944
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 149504
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_vsa_sm90_5fb4df1e9557301d769d(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, int g_shift)
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
    float* mstats = reinterpret_cast<float*>(smem_raw + 148480);
    const int mstats_addr = smem + 148480;
    float* part_lo = reinterpret_cast<float*>(smem_raw + 66560);
    const int part_lo_addr = smem + 66560;
    float* part_hi = reinterpret_cast<float*>(smem_raw + 132096);
    const int part_hi_addr = smem + 132096;
    float* part_hi0 = reinterpret_cast<float*>(smem_raw + 50176);
    const int part_hi0_addr = smem + 50176;
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    __nv_bfloat16* vt_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int vt_smem_addr = smem + 82944;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vt))) : "memory"); }

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
        if (cnt > 2) {
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    int nblk = plan[blk_base + 2];
                    int nblk_row = kv_base + nblk * 64;
                    mbarrier_arrive_expect_tx(k_full2_addr, 16384);
                    tma_3d_gmem2smem(k_smem_addr + 32768, (&K), 0, nblk_row, 0, k_full2_addr);
                    mbarrier_arrive_expect_tx(v_full2_addr, 16384);
                    tma_4d_gmem2smem(vt_smem_addr + 32768, (&Vt), 0, 0, nblk_row / 8, 0, v_full2_addr);
                }
            }
        }
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
        if (cnt > 3) {
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    int nblk_0 = plan[blk_base + 3];
                    int nblk_row_0 = kv_base + nblk_0 * 64;
                    mbarrier_arrive_expect_tx(k_full3_addr, 16384);
                    tma_3d_gmem2smem(k_smem_addr + 49152, (&K), 0, nblk_row_0, 0, k_full3_addr);
                    mbarrier_arrive_expect_tx(v_full3_addr, 16384);
                    tma_4d_gmem2smem(vt_smem_addr + 49152, (&Vt), 0, 0, nblk_row_0 / 8, 0, v_full3_addr);
                }
            }
        }
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
    float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_sync_94;
    float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_sync_95;
    float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_sync_96;
    float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_sync_97;
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
    float _exp2_98 = approx_exp2(x0m0 - mm0);
    float f00 = ((x0m0 == -CUDART_INF_F) ? 0.0f : _exp2_98);
    float _exp2_99 = approx_exp2(x0m1 - mm1);
    float f01 = ((x0m1 == -CUDART_INF_F) ? 0.0f : _exp2_99);
    float _exp2_100 = approx_exp2(x1m0 - mm0);
    float f10 = ((x1m0 == -CUDART_INF_F) ? 0.0f : _exp2_100);
    float _exp2_101 = approx_exp2(x1m1 - mm1);
    float f11 = ((x1m1 == -CUDART_INF_F) ? 0.0f : _exp2_101);
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
        d_o[0] = d_o[0] * _rcp_0;
        d_o[1] = d_o[1] * _rcp_0;
        d_o[2] = d_o[2] * _rcp_1;
        d_o[3] = d_o[3] * _rcp_1;
        d_o[4] = d_o[4] * _rcp_0;
        d_o[5] = d_o[5] * _rcp_0;
        d_o[6] = d_o[6] * _rcp_1;
        d_o[7] = d_o[7] * _rcp_1;
        d_o[8] = d_o[8] * _rcp_0;
        d_o[9] = d_o[9] * _rcp_0;
        d_o[10] = d_o[10] * _rcp_1;
        d_o[11] = d_o[11] * _rcp_1;
        d_o[12] = d_o[12] * _rcp_0;
        d_o[13] = d_o[13] * _rcp_0;
        d_o[14] = d_o[14] * _rcp_1;
        d_o[15] = d_o[15] * _rcp_1;
        d_o[16] = d_o[16] * _rcp_0;
        d_o[17] = d_o[17] * _rcp_0;
        d_o[18] = d_o[18] * _rcp_1;
        d_o[19] = d_o[19] * _rcp_1;
        d_o[20] = d_o[20] * _rcp_0;
        d_o[21] = d_o[21] * _rcp_0;
        d_o[22] = d_o[22] * _rcp_1;
        d_o[23] = d_o[23] * _rcp_1;
        d_o[24] = d_o[24] * _rcp_0;
        d_o[25] = d_o[25] * _rcp_0;
        d_o[26] = d_o[26] * _rcp_1;
        d_o[27] = d_o[27] * _rcp_1;
        d_o[28] = d_o[28] * _rcp_0;
        d_o[29] = d_o[29] * _rcp_0;
        d_o[30] = d_o[30] * _rcp_1;
        d_o[31] = d_o[31] * _rcp_1;
        int qodd = qj & 1;
        unsigned int o_pk[2];
        unsigned int o_vec[2];
        int o_row_base = q_row * 128 + qj * 2 + qodd * 6;
        int m_local_r_1_0 = m0_local;
        int row_off_1_0 = o_row_base + m_local_r_1_0 * 128;
        __nv_bfloat162 _bit_cast_src_102 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_102)[0];
        __nv_bfloat162 _bit_cast_src_103 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_103)[0];
        unsigned int send_3_0 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send_3_0, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_0);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_0 : o_pk[1]);
        reinterpret_cast<int2*>(O + row_off_1_0)[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_104 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_104)[0];
        __nv_bfloat162 _bit_cast_src_105 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_105)[0];
        unsigned int send_3_1 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_3_1, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_1);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_1 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_106 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_106)[0];
        __nv_bfloat162 _bit_cast_src_107 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_107)[0];
        unsigned int send_3_2 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_3_2, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_2);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_2 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_108 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_108)[0];
        __nv_bfloat162 _bit_cast_src_109 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_109)[0];
        unsigned int send_3_3 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_3_3, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_3);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_3 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_0 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
        int m_local_r_1_1 = m1_local;
        int row_off_1_1 = o_row_base + m_local_r_1_1 * 128;
        __nv_bfloat162 _bit_cast_src_110 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_110)[0];
        __nv_bfloat162 _bit_cast_src_111 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_111)[0];
        unsigned int send_3_4 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_3_4, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_4);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_4 : o_pk[1]);
        reinterpret_cast<int2*>(O + row_off_1_1)[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_112 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_112)[0];
        __nv_bfloat162 _bit_cast_src_113 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_113)[0];
        unsigned int send_3_5 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_3_5, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_5);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_5 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_114 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_114)[0];
        __nv_bfloat162 _bit_cast_src_115 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_115)[0];
        unsigned int send_3_6 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_3_6, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_6);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_6 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
        __nv_bfloat162 _bit_cast_src_116 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
        o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_116)[0];
        __nv_bfloat162 _bit_cast_src_117 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
        o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_117)[0];
        unsigned int send_3_7 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
        unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_3_7, 1);
        o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_7);
        o_vec[1] = ((qodd == 0) ? _shfl_xor_7 : o_pk[1]);
        reinterpret_cast<int2*>(O + (row_off_1_1 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
    }
    if (wg == 1) {
        float _rcp_2 = approx_rcp(row_sum0);
        float _rcp_3 = approx_rcp(row_sum1);
        d_o[32] = d_o[32] * _rcp_2;
        d_o[33] = d_o[33] * _rcp_2;
        d_o[34] = d_o[34] * _rcp_3;
        d_o[35] = d_o[35] * _rcp_3;
        d_o[36] = d_o[36] * _rcp_2;
        d_o[37] = d_o[37] * _rcp_2;
        d_o[38] = d_o[38] * _rcp_3;
        d_o[39] = d_o[39] * _rcp_3;
        d_o[40] = d_o[40] * _rcp_2;
        d_o[41] = d_o[41] * _rcp_2;
        d_o[42] = d_o[42] * _rcp_3;
        d_o[43] = d_o[43] * _rcp_3;
        d_o[44] = d_o[44] * _rcp_2;
        d_o[45] = d_o[45] * _rcp_2;
        d_o[46] = d_o[46] * _rcp_3;
        d_o[47] = d_o[47] * _rcp_3;
        d_o[48] = d_o[48] * _rcp_2;
        d_o[49] = d_o[49] * _rcp_2;
        d_o[50] = d_o[50] * _rcp_3;
        d_o[51] = d_o[51] * _rcp_3;
        d_o[52] = d_o[52] * _rcp_2;
        d_o[53] = d_o[53] * _rcp_2;
        d_o[54] = d_o[54] * _rcp_3;
        d_o[55] = d_o[55] * _rcp_3;
        d_o[56] = d_o[56] * _rcp_2;
        d_o[57] = d_o[57] * _rcp_2;
        d_o[58] = d_o[58] * _rcp_3;
        d_o[59] = d_o[59] * _rcp_3;
        d_o[60] = d_o[60] * _rcp_2;
        d_o[61] = d_o[61] * _rcp_2;
        d_o[62] = d_o[62] * _rcp_3;
        d_o[63] = d_o[63] * _rcp_3;
        int qodd_1 = qj & 1;
        unsigned int o_pk_1[2];
        unsigned int o_vec_1[2];
        int o_row_base_1 = q_row * 128 + qj * 2 + qodd_1 * 6;
        int m_local_r_1_2 = m0_local;
        int row_off_1_2 = o_row_base_1 + m_local_r_1_2 * 128;
        __nv_bfloat162 _bit_cast_src_118 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_118)[0];
        __nv_bfloat162 _bit_cast_src_119 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_119)[0];
        unsigned int send_3_8 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send_3_8, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_8);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_8 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 64))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_120 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_120)[0];
        __nv_bfloat162 _bit_cast_src_121 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_121)[0];
        unsigned int send_3_9 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_3_9, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_9);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_9 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 80))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_122 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_122)[0];
        __nv_bfloat162 _bit_cast_src_123 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_123)[0];
        unsigned int send_3_10 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_3_10, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_10);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_10 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 96))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_124 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_124)[0];
        __nv_bfloat162 _bit_cast_src_125 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_125)[0];
        unsigned int send_3_11 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_3_11, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_11);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_11 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_2 + 112))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        int m_local_r_1_3 = m1_local;
        int row_off_1_3 = o_row_base_1 + m_local_r_1_3 * 128;
        __nv_bfloat162 _bit_cast_src_126 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_126)[0];
        __nv_bfloat162 _bit_cast_src_127 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_127)[0];
        unsigned int send_3_12 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_3_12, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_12);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_12 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 64))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_128 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_128)[0];
        __nv_bfloat162 _bit_cast_src_129 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
        unsigned int send_3_13 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_3_13, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_13);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_13 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 80))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_130 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
        __nv_bfloat162 _bit_cast_src_131 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_131)[0];
        unsigned int send_3_14 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_3_14, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_14);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_14 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 96))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
        __nv_bfloat162 _bit_cast_src_132 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
        o_pk_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_132)[0];
        __nv_bfloat162 _bit_cast_src_133 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
        o_pk_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_133)[0];
        unsigned int send_3_15 = ((qodd_1 != 0) ? o_pk_1[0] : o_pk_1[1]);
        unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_3_15, 1);
        o_vec_1[0] = ((qodd_1 == 0) ? o_pk_1[0] : _shfl_xor_15);
        o_vec_1[1] = ((qodd_1 == 0) ? _shfl_xor_15 : o_pk_1[1]);
        reinterpret_cast<int2*>(O + (row_off_1_3 + 112))[0] = reinterpret_cast<int2*>(o_vec_1)[0];
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
