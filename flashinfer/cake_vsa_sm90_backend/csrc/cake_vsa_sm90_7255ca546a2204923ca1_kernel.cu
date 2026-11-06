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
#define SMEM_MSTATS_OFF 82944
#define SMEM_MSTATS_STAGE_BYTES 1024
#define SMEM_MSTATS_STRIDE 1024
#define SMEM_PART_LO_OFF 33792
#define SMEM_PART_LO_STAGE_BYTES 16384
#define SMEM_PART_LO_STRIDE 16384
#define SMEM_PART_HI_OFF 66560
#define SMEM_PART_HI_STAGE_BYTES 16384
#define SMEM_PART_HI_STRIDE 16384
#define SMEM_PART_HI0_OFF 17408
#define SMEM_PART_HI0_STAGE_BYTES 16384
#define SMEM_PART_HI0_STRIDE 16384
#define SMEM_RECV_SMEM_OFF 83968
#define SMEM_RECV_SMEM_STAGE_BYTES 27648
#define SMEM_RECV_SMEM_STRIDE 27648
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 50176
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 111616
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_vsa_sm90_7255ca546a2204923ca1(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, int g_shift)
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
    float* mstats = reinterpret_cast<float*>(smem_raw + 82944);
    const int mstats_addr = smem + 82944;
    float* part_lo = reinterpret_cast<float*>(smem_raw + 33792);
    const int part_lo_addr = smem + 33792;
    float* part_hi = reinterpret_cast<float*>(smem_raw + 66560);
    const int part_hi_addr = smem + 66560;
    float* part_hi0 = reinterpret_cast<float*>(smem_raw + 17408);
    const int part_hi0_addr = smem + 17408;
    float* recv_smem = reinterpret_cast<float*>(smem_raw + 83968);
    const int recv_smem_addr = smem + 83968;
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    __nv_bfloat16* vt_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int vt_smem_addr = smem + 50176;

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
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vt))) : "memory"); }
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
    float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 += _shfl_xor_sync_48;
    float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 += _shfl_xor_sync_49;
    float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 += _shfl_xor_sync_50;
    float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 += _shfl_xor_sync_51;
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
    float _exp2_52 = approx_exp2(x0m0 - mm0);
    float f00 = ((x0m0 == -CUDART_INF_F) ? 0.0f : _exp2_52);
    float _exp2_53 = approx_exp2(x0m1 - mm1);
    float f01 = ((x0m1 == -CUDART_INF_F) ? 0.0f : _exp2_53);
    float _exp2_54 = approx_exp2(x1m0 - mm0);
    float f10 = ((x1m0 == -CUDART_INF_F) ? 0.0f : _exp2_54);
    float _exp2_55 = approx_exp2(x1m1 - mm1);
    float f11 = ((x1m1 == -CUDART_INF_F) ? 0.0f : _exp2_55);
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
            uint32_t _mbar_peer_56;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_56) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0), "r"(__float_as_uint(d_o[0])), "r"(__float_as_uint(d_o[1])), "r"(__float_as_uint(d_o[2])), "r"(__float_as_uint(d_o[3])), "r"(_mbar_peer_56) : "memory");
            uint32_t _mbar_peer_57;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_57) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(d_o[4])), "r"(__float_as_uint(d_o[5])), "r"(__float_as_uint(d_o[6])), "r"(__float_as_uint(d_o[7])), "r"(_mbar_peer_57) : "memory");
            uint32_t _mbar_peer_58;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_58) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(d_o[8])), "r"(__float_as_uint(d_o[9])), "r"(__float_as_uint(d_o[10])), "r"(__float_as_uint(d_o[11])), "r"(_mbar_peer_58) : "memory");
            uint32_t _mbar_peer_59;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_59) : "r"(recv_full_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(d_o[12])), "r"(__float_as_uint(d_o[13])), "r"(__float_as_uint(d_o[14])), "r"(__float_as_uint(d_o[15])), "r"(_mbar_peer_59) : "memory");
            if (qj == 0) {
                uint32_t _cluster_mapa_60;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_60) : "r"(recv_smem_addr + (unsigned int)(pslot * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(0));
                uint32_t _mbar_peer_61;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_61) : "r"(recv_full_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_60), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_61) : "memory");
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
            uint32_t _mbar_peer_62;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_62) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1), "r"(__float_as_uint(d_o[16])), "r"(__float_as_uint(d_o[17])), "r"(__float_as_uint(d_o[18])), "r"(__float_as_uint(d_o[19])), "r"(_mbar_peer_62) : "memory");
            uint32_t _mbar_peer_63;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_63) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(d_o[20])), "r"(__float_as_uint(d_o[21])), "r"(__float_as_uint(d_o[22])), "r"(__float_as_uint(d_o[23])), "r"(_mbar_peer_63) : "memory");
            uint32_t _mbar_peer_64;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_64) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(d_o[24])), "r"(__float_as_uint(d_o[25])), "r"(__float_as_uint(d_o[26])), "r"(__float_as_uint(d_o[27])), "r"(_mbar_peer_64) : "memory");
            uint32_t _mbar_peer_65;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_65) : "r"(recv_full_addr), "r"(1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(d_o[28])), "r"(__float_as_uint(d_o[29])), "r"(__float_as_uint(d_o[30])), "r"(__float_as_uint(d_o[31])), "r"(_mbar_peer_65) : "memory");
            if (qj == 0) {
                uint32_t _cluster_mapa_66;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_66) : "r"(recv_smem_addr + (unsigned int)(pslot_1 * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(1));
                uint32_t _mbar_peer_67;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_67) : "r"(recv_full_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_66), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_67) : "memory");
            }
        }
        if (my_rank != 2) {
            int pslot_2 = my_rank;
            if (my_rank > 2) {
                pslot_2 = my_rank - 1;
            }
            if (qj == 0) {
                uint32_t _cluster_mapa_68;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_68) : "r"(recv_smem_addr + (unsigned int)(pslot_2 * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(2));
                uint32_t _mbar_peer_69;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_69) : "r"(recv_full_addr), "r"(2));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_68), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_69) : "memory");
            }
        }
        if (my_rank != 3) {
            int pslot_3 = my_rank;
            if (my_rank > 3) {
                pslot_3 = my_rank - 1;
            }
            if (qj == 0) {
                uint32_t _cluster_mapa_70;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_70) : "r"(recv_smem_addr + (unsigned int)(pslot_3 * 9216) + 8192 + (unsigned int)(m0_local * 16)), "r"(3));
                uint32_t _mbar_peer_71;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_71) : "r"(recv_full_addr), "r"(3));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_70), "r"(__float_as_uint(row_max0)), "r"(__float_as_uint(row_sum0)), "r"(__float_as_uint(row_max1)), "r"(__float_as_uint(row_sum1)), "r"(_mbar_peer_71) : "memory");
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
            float exp2 = approx_exp2(row_max0 - merged_max0);
            float exp2_1_32 = approx_exp2(row_max1 - merged_max1);
            float msum0 = row_sum0 * exp2;
            float msum1 = row_sum1 * exp2_1_32;
            float w_peer0[3];
            float w_peer1[3];
            int sbase_2_0 = 2048 + m0_local * 4;
            float _exp2_72 = approx_exp2(recv_smem[sbase_2_0] - merged_max0);
            w_peer0[0] = _exp2_72;
            float _exp2_73 = approx_exp2(recv_smem[sbase_2_0 + 2] - merged_max1);
            w_peer1[0] = _exp2_73;
            msum0 = __fmaf_rn(recv_smem[sbase_2_0 + 1], w_peer0[0], msum0);
            msum1 = __fmaf_rn(recv_smem[sbase_2_0 + 3], w_peer1[0], msum1);
            int sbase_2_1 = 4352 + m0_local * 4;
            float _exp2_74 = approx_exp2(recv_smem[sbase_2_1] - merged_max0);
            w_peer0[1] = _exp2_74;
            float _exp2_75 = approx_exp2(recv_smem[sbase_2_1 + 2] - merged_max1);
            w_peer1[1] = _exp2_75;
            msum0 = __fmaf_rn(recv_smem[sbase_2_1 + 1], w_peer0[1], msum0);
            msum1 = __fmaf_rn(recv_smem[sbase_2_1 + 3], w_peer1[1], msum1);
            int sbase_2_2 = 6656 + m0_local * 4;
            float _exp2_76 = approx_exp2(recv_smem[sbase_2_2] - merged_max0);
            w_peer0[2] = _exp2_76;
            float _exp2_77 = approx_exp2(recv_smem[sbase_2_2 + 2] - merged_max1);
            w_peer1[2] = _exp2_77;
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
                    acc[0] = d_o[0] * exp2;
                    acc[1] = d_o[1] * exp2;
                    acc[2] = d_o[2] * exp2_1_32;
                    acc[3] = d_o[3] * exp2_1_32;
                    acc[4] = d_o[4] * exp2;
                    acc[5] = d_o[5] * exp2;
                    acc[6] = d_o[6] * exp2_1_32;
                    acc[7] = d_o[7] * exp2_1_32;
                    acc[8] = d_o[8] * exp2;
                    acc[9] = d_o[9] * exp2;
                    acc[10] = d_o[10] * exp2_1_32;
                    acc[11] = d_o[11] * exp2_1_32;
                    acc[12] = d_o[12] * exp2;
                    acc[13] = d_o[13] * exp2;
                    acc[14] = d_o[14] * exp2_1_32;
                    acc[15] = d_o[15] * exp2_1_32;
                }
                if (sset == 1) {
                    acc[0] = d_o[16] * exp2;
                    acc[1] = d_o[17] * exp2;
                    acc[2] = d_o[18] * exp2_1_32;
                    acc[3] = d_o[19] * exp2_1_32;
                    acc[4] = d_o[20] * exp2;
                    acc[5] = d_o[21] * exp2;
                    acc[6] = d_o[22] * exp2_1_32;
                    acc[7] = d_o[23] * exp2_1_32;
                    acc[8] = d_o[24] * exp2;
                    acc[9] = d_o[25] * exp2;
                    acc[10] = d_o[26] * exp2_1_32;
                    acc[11] = d_o[27] * exp2_1_32;
                    acc[12] = d_o[28] * exp2;
                    acc[13] = d_o[29] * exp2;
                    acc[14] = d_o[30] * exp2_1_32;
                    acc[15] = d_o[31] * exp2_1_32;
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
                acc[0] = acc[0] * rcp;
                acc[1] = acc[1] * rcp;
                acc[2] = acc[2] * rcp_1;
                acc[3] = acc[3] * rcp_1;
                acc[4] = acc[4] * rcp;
                acc[5] = acc[5] * rcp;
                acc[6] = acc[6] * rcp_1;
                acc[7] = acc[7] * rcp_1;
                acc[8] = acc[8] * rcp;
                acc[9] = acc[9] * rcp;
                acc[10] = acc[10] * rcp_1;
                acc[11] = acc[11] * rcp_1;
                acc[12] = acc[12] * rcp;
                acc[13] = acc[13] * rcp;
                acc[14] = acc[14] * rcp_1;
                acc[15] = acc[15] * rcp_1;
                int qj1 = qj & 1;
                int qj2 = qj & 2;
                unsigned int o_vec[4];
                unsigned int o_tmp[4];
                int o_row_base = q_row * 128;
                int m_local_r_1_0 = m0_local;
                __nv_bfloat162 _bit_cast_src_78 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
                o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_78)[0];
                __nv_bfloat162 _bit_cast_src_79 = __float22bfloat162_rn(make_float2(acc[4], acc[5]));
                o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_79)[0];
                __nv_bfloat162 _bit_cast_src_80 = __float22bfloat162_rn(make_float2(acc[8], acc[9]));
                o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_80)[0];
                __nv_bfloat162 _bit_cast_src_81 = __float22bfloat162_rn(make_float2(acc[12], acc[13]));
                o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_81)[0];
                unsigned int _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                o_tmp[0] = _shfl_xor_sync_82;
                unsigned int _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                o_tmp[1] = _shfl_xor_sync_83;
                unsigned int _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                o_tmp[2] = _shfl_xor_sync_84;
                unsigned int _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                o_tmp[3] = _shfl_xor_sync_85;
                o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                unsigned int _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                o_tmp[0] = _shfl_xor_sync_86;
                unsigned int _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                o_tmp[1] = _shfl_xor_sync_87;
                unsigned int _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                o_tmp[2] = _shfl_xor_sync_88;
                unsigned int _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                o_tmp[3] = _shfl_xor_sync_89;
                o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                int o_off_1_0 = o_row_base + m_local_r_1_0 * 128 + (4 * sset + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_0)[0] = reinterpret_cast<int4*>(o_vec)[0];
                int m_local_r_1_1 = m1_local;
                __nv_bfloat162 _bit_cast_src_90 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
                o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_90)[0];
                __nv_bfloat162 _bit_cast_src_91 = __float22bfloat162_rn(make_float2(acc[6], acc[7]));
                o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_91)[0];
                __nv_bfloat162 _bit_cast_src_92 = __float22bfloat162_rn(make_float2(acc[10], acc[11]));
                o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_92)[0];
                __nv_bfloat162 _bit_cast_src_93 = __float22bfloat162_rn(make_float2(acc[14], acc[15]));
                o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_93)[0];
                unsigned int _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                o_tmp[0] = _shfl_xor_sync_94;
                unsigned int _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                o_tmp[1] = _shfl_xor_sync_95;
                unsigned int _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                o_tmp[2] = _shfl_xor_sync_96;
                unsigned int _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                o_tmp[3] = _shfl_xor_sync_97;
                o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                unsigned int _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                o_tmp[0] = _shfl_xor_sync_98;
                unsigned int _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                o_tmp[1] = _shfl_xor_sync_99;
                unsigned int _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                o_tmp[2] = _shfl_xor_sync_100;
                unsigned int _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                o_tmp[3] = _shfl_xor_sync_101;
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
            uint32_t _mbar_peer_102;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_102) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2), "r"(__float_as_uint(d_o[32])), "r"(__float_as_uint(d_o[33])), "r"(__float_as_uint(d_o[34])), "r"(__float_as_uint(d_o[35])), "r"(_mbar_peer_102) : "memory");
            uint32_t _mbar_peer_103;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_103) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 2048), "r"(__float_as_uint(d_o[36])), "r"(__float_as_uint(d_o[37])), "r"(__float_as_uint(d_o[38])), "r"(__float_as_uint(d_o[39])), "r"(_mbar_peer_103) : "memory");
            uint32_t _mbar_peer_104;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_104) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 4096), "r"(__float_as_uint(d_o[40])), "r"(__float_as_uint(d_o[41])), "r"(__float_as_uint(d_o[42])), "r"(__float_as_uint(d_o[43])), "r"(_mbar_peer_104) : "memory");
            uint32_t _mbar_peer_105;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_105) : "r"(recv_full_addr), "r"(2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 6144), "r"(__float_as_uint(d_o[44])), "r"(__float_as_uint(d_o[45])), "r"(__float_as_uint(d_o[46])), "r"(__float_as_uint(d_o[47])), "r"(_mbar_peer_105) : "memory");
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
            uint32_t _mbar_peer_106;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_106) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3), "r"(__float_as_uint(d_o[48])), "r"(__float_as_uint(d_o[49])), "r"(__float_as_uint(d_o[50])), "r"(__float_as_uint(d_o[51])), "r"(_mbar_peer_106) : "memory");
            uint32_t _mbar_peer_107;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_107) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 2048), "r"(__float_as_uint(d_o[52])), "r"(__float_as_uint(d_o[53])), "r"(__float_as_uint(d_o[54])), "r"(__float_as_uint(d_o[55])), "r"(_mbar_peer_107) : "memory");
            uint32_t _mbar_peer_108;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_108) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 4096), "r"(__float_as_uint(d_o[56])), "r"(__float_as_uint(d_o[57])), "r"(__float_as_uint(d_o[58])), "r"(__float_as_uint(d_o[59])), "r"(_mbar_peer_108) : "memory");
            uint32_t _mbar_peer_109;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_109) : "r"(recv_full_addr), "r"(3));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3 + 6144), "r"(__float_as_uint(d_o[60])), "r"(__float_as_uint(d_o[61])), "r"(__float_as_uint(d_o[62])), "r"(__float_as_uint(d_o[63])), "r"(_mbar_peer_109) : "memory");
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
            float exp2_32 = approx_exp2(row_max0 - merged_max0_1);
            float exp2_1_33 = approx_exp2(row_max1 - merged_max1_1);
            float msum0_1 = row_sum0 * exp2_32;
            float msum1_1 = row_sum1 * exp2_1_33;
            float w_peer0_1[3];
            float w_peer1_1[3];
            int sbase_2_3 = 2048 + m0_local * 4;
            float _exp2_110 = approx_exp2(recv_smem[sbase_2_3] - merged_max0_1);
            w_peer0_1[0] = _exp2_110;
            float _exp2_111 = approx_exp2(recv_smem[sbase_2_3 + 2] - merged_max1_1);
            w_peer1_1[0] = _exp2_111;
            msum0_1 = __fmaf_rn(recv_smem[sbase_2_3 + 1], w_peer0_1[0], msum0_1);
            msum1_1 = __fmaf_rn(recv_smem[sbase_2_3 + 3], w_peer1_1[0], msum1_1);
            int sbase_2_4 = 4352 + m0_local * 4;
            float _exp2_112 = approx_exp2(recv_smem[sbase_2_4] - merged_max0_1);
            w_peer0_1[1] = _exp2_112;
            float _exp2_113 = approx_exp2(recv_smem[sbase_2_4 + 2] - merged_max1_1);
            w_peer1_1[1] = _exp2_113;
            msum0_1 = __fmaf_rn(recv_smem[sbase_2_4 + 1], w_peer0_1[1], msum0_1);
            msum1_1 = __fmaf_rn(recv_smem[sbase_2_4 + 3], w_peer1_1[1], msum1_1);
            int sbase_2_5 = 6656 + m0_local * 4;
            float _exp2_114 = approx_exp2(recv_smem[sbase_2_5] - merged_max0_1);
            w_peer0_1[2] = _exp2_114;
            float _exp2_115 = approx_exp2(recv_smem[sbase_2_5 + 2] - merged_max1_1);
            w_peer1_1[2] = _exp2_115;
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
                    acc_1[0] = d_o[32] * exp2_32;
                    acc_1[1] = d_o[33] * exp2_32;
                    acc_1[2] = d_o[34] * exp2_1_33;
                    acc_1[3] = d_o[35] * exp2_1_33;
                    acc_1[4] = d_o[36] * exp2_32;
                    acc_1[5] = d_o[37] * exp2_32;
                    acc_1[6] = d_o[38] * exp2_1_33;
                    acc_1[7] = d_o[39] * exp2_1_33;
                    acc_1[8] = d_o[40] * exp2_32;
                    acc_1[9] = d_o[41] * exp2_32;
                    acc_1[10] = d_o[42] * exp2_1_33;
                    acc_1[11] = d_o[43] * exp2_1_33;
                    acc_1[12] = d_o[44] * exp2_32;
                    acc_1[13] = d_o[45] * exp2_32;
                    acc_1[14] = d_o[46] * exp2_1_33;
                    acc_1[15] = d_o[47] * exp2_1_33;
                }
                if (sset_1 == 3) {
                    acc_1[0] = d_o[48] * exp2_32;
                    acc_1[1] = d_o[49] * exp2_32;
                    acc_1[2] = d_o[50] * exp2_1_33;
                    acc_1[3] = d_o[51] * exp2_1_33;
                    acc_1[4] = d_o[52] * exp2_32;
                    acc_1[5] = d_o[53] * exp2_32;
                    acc_1[6] = d_o[54] * exp2_1_33;
                    acc_1[7] = d_o[55] * exp2_1_33;
                    acc_1[8] = d_o[56] * exp2_32;
                    acc_1[9] = d_o[57] * exp2_32;
                    acc_1[10] = d_o[58] * exp2_1_33;
                    acc_1[11] = d_o[59] * exp2_1_33;
                    acc_1[12] = d_o[60] * exp2_32;
                    acc_1[13] = d_o[61] * exp2_32;
                    acc_1[14] = d_o[62] * exp2_1_33;
                    acc_1[15] = d_o[63] * exp2_1_33;
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
                acc_1[0] = acc_1[0] * rcp_2;
                acc_1[1] = acc_1[1] * rcp_2;
                acc_1[2] = acc_1[2] * rcp_1_1;
                acc_1[3] = acc_1[3] * rcp_1_1;
                acc_1[4] = acc_1[4] * rcp_2;
                acc_1[5] = acc_1[5] * rcp_2;
                acc_1[6] = acc_1[6] * rcp_1_1;
                acc_1[7] = acc_1[7] * rcp_1_1;
                acc_1[8] = acc_1[8] * rcp_2;
                acc_1[9] = acc_1[9] * rcp_2;
                acc_1[10] = acc_1[10] * rcp_1_1;
                acc_1[11] = acc_1[11] * rcp_1_1;
                acc_1[12] = acc_1[12] * rcp_2;
                acc_1[13] = acc_1[13] * rcp_2;
                acc_1[14] = acc_1[14] * rcp_1_1;
                acc_1[15] = acc_1[15] * rcp_1_1;
                int qj1_1 = qj & 1;
                int qj2_1 = qj & 2;
                unsigned int o_vec_1[4];
                unsigned int o_tmp_1[4];
                int o_row_base_1 = q_row * 128;
                int m_local_r_1_2 = m0_local;
                __nv_bfloat162 _bit_cast_src_116 = __float22bfloat162_rn(make_float2(acc_1[0], acc_1[1]));
                o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_116)[0];
                __nv_bfloat162 _bit_cast_src_117 = __float22bfloat162_rn(make_float2(acc_1[4], acc_1[5]));
                o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_117)[0];
                __nv_bfloat162 _bit_cast_src_118 = __float22bfloat162_rn(make_float2(acc_1[8], acc_1[9]));
                o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_118)[0];
                __nv_bfloat162 _bit_cast_src_119 = __float22bfloat162_rn(make_float2(acc_1[12], acc_1[13]));
                o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_119)[0];
                unsigned int _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                o_tmp_1[0] = _shfl_xor_sync_120;
                unsigned int _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                o_tmp_1[1] = _shfl_xor_sync_121;
                unsigned int _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                o_tmp_1[2] = _shfl_xor_sync_122;
                unsigned int _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                o_tmp_1[3] = _shfl_xor_sync_123;
                o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                unsigned int _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                o_tmp_1[0] = _shfl_xor_sync_124;
                unsigned int _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                o_tmp_1[1] = _shfl_xor_sync_125;
                unsigned int _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                o_tmp_1[2] = _shfl_xor_sync_126;
                unsigned int _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                o_tmp_1[3] = _shfl_xor_sync_127;
                o_vec_1[0] = ((qj2_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj2_1 != 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj2_1 == 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj2_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                int o_off_1_2 = o_row_base_1 + m_local_r_1_2 * 128 + (4 * sset_1 + qj) * 8;
                reinterpret_cast<int4*>(O + o_off_1_2)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
                int m_local_r_1_3 = m1_local;
                __nv_bfloat162 _bit_cast_src_128 = __float22bfloat162_rn(make_float2(acc_1[2], acc_1[3]));
                o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_128)[0];
                __nv_bfloat162 _bit_cast_src_129 = __float22bfloat162_rn(make_float2(acc_1[6], acc_1[7]));
                o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
                __nv_bfloat162 _bit_cast_src_130 = __float22bfloat162_rn(make_float2(acc_1[10], acc_1[11]));
                o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
                __nv_bfloat162 _bit_cast_src_131 = __float22bfloat162_rn(make_float2(acc_1[14], acc_1[15]));
                o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_131)[0];
                unsigned int _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
                o_tmp_1[0] = _shfl_xor_sync_132;
                unsigned int _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
                o_tmp_1[1] = _shfl_xor_sync_133;
                unsigned int _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
                o_tmp_1[2] = _shfl_xor_sync_134;
                unsigned int _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
                o_tmp_1[3] = _shfl_xor_sync_135;
                o_vec_1[0] = ((qj1_1 != 0) ? o_tmp_1[0] : o_vec_1[0]);
                o_vec_1[1] = ((qj1_1 == 0) ? o_tmp_1[1] : o_vec_1[1]);
                o_vec_1[2] = ((qj1_1 != 0) ? o_tmp_1[2] : o_vec_1[2]);
                o_vec_1[3] = ((qj1_1 == 0) ? o_tmp_1[3] : o_vec_1[3]);
                unsigned int _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
                o_tmp_1[0] = _shfl_xor_sync_136;
                unsigned int _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
                o_tmp_1[1] = _shfl_xor_sync_137;
                unsigned int _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
                o_tmp_1[2] = _shfl_xor_sync_138;
                unsigned int _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
                o_tmp_1[3] = _shfl_xor_sync_139;
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
