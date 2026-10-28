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
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 33792
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 50176
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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_vsa_sm90_fff66236353a558b6d21(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, const CakeParamArray<int16_t, 1750> plan, int seqlen_q, int seqlen_k, float scale_log2, int g_shift)
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
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    __nv_bfloat16* vt_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vt_smem_addr = smem + 33792;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vt))) : "memory"); }

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
    int kv_base = (head >> g_shift) * seqlen_k;
    if (warp == 0) {
        if (elect_sync()) {
            int blk_pre[1];
            blk_pre[0] = plan[blk_base];
            mbarrier_arrive_expect_tx(q_full_addr, 16384);
            tma_3d_gmem2smem(q_smem_addr, (&Q), 0, q_row, 0, q_full_addr);
            if (cnt > 0) {
                int blk_row = kv_base + blk_pre[0] * 64;
                mbarrier_arrive_expect_tx(k_full0_addr, 16384);
                tma_3d_gmem2smem(k_smem_addr, (&K), 0, blk_row, 0, k_full0_addr);
                mbarrier_arrive_expect_tx(v_full0_addr, 16384);
                tma_4d_gmem2smem(vt_smem_addr, (&Vt), 0, 0, blk_row / 8, 0, v_full0_addr);
                tma_4d_gmem2smem(vt_smem_addr + 8192, (&Vt), 0, 0, blk_row / 8, 1, v_full0_addr);
                tma_4d_gmem2smem(vt_smem_addr + 4096, (&Vt), 0, 0, blk_row / 8 + 4, 0, v_full0_addr);
                tma_4d_gmem2smem(vt_smem_addr + 8192 + 4096, (&Vt), 0, 0, blk_row / 8 + 4, 1, v_full0_addr);
            }
        }
    }
    __syncthreads();
    int m0_local = warp * 16 + lane / 4;
    int m1_local = m0_local + 8;
    float d_o[64];
    float d_qk[32];
    unsigned int p_bf16[16];
    float row_max0 = -CUDART_INF_F;
    float row_max1 = -CUDART_INF_F;
    float row_sum0 = 0.0f;
    float row_sum1 = 0.0f;
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
    uint64_t _wgmma_desc_5 = (((uint64_t)(((vt_smem_addr + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
    if (cnt > 0) {
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
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, new_max0, 2);
        new_max0 = fmaxf(new_max0, _shfl_xor_sync_6);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, new_max0, 1);
        new_max0 = fmaxf(new_max0, _shfl_xor_sync_7);
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, new_max1, 2);
        new_max1 = fmaxf(new_max1, _shfl_xor_sync_8);
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, new_max1, 1);
        new_max1 = fmaxf(new_max1, _shfl_xor_sync_9);
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
        __nv_bfloat162 _bit_cast_src_10 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_10)[0];
        __nv_bfloat162 _bit_cast_src_11 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_11)[0];
        __nv_bfloat162 _bit_cast_src_12 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_12)[0];
        __nv_bfloat162 _bit_cast_src_13 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_13)[0];
        __nv_bfloat162 _bit_cast_src_14 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_14)[0];
        __nv_bfloat162 _bit_cast_src_15 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_15)[0];
        __nv_bfloat162 _bit_cast_src_16 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_16)[0];
        __nv_bfloat162 _bit_cast_src_17 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_17)[0];
        mbarrier_wait(v_full0_addr, 0);
        if (gridDim.x <= 132) {
            float d_o_lo[32];
            float d_o_hi[32];
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 1;\n}\n"
                : "+f"(d_o_lo[0]), "+f"(d_o_lo[1]), "+f"(d_o_lo[2]), "+f"(d_o_lo[3]), "+f"(d_o_lo[4]), "+f"(d_o_lo[5]), "+f"(d_o_lo[6]), "+f"(d_o_lo[7]), "+f"(d_o_lo[8]), "+f"(d_o_lo[9]), "+f"(d_o_lo[10]), "+f"(d_o_lo[11]), "+f"(d_o_lo[12]), "+f"(d_o_lo[13]), "+f"(d_o_lo[14]), "+f"(d_o_lo[15]), "+f"(d_o_lo[16]), "+f"(d_o_lo[17]), "+f"(d_o_lo[18]), "+f"(d_o_lo[19]), "+f"(d_o_lo[20]), "+f"(d_o_lo[21]), "+f"(d_o_lo[22]), "+f"(d_o_lo[23]), "+f"(d_o_lo[24]), "+f"(d_o_lo[25]), "+f"(d_o_lo[26]), "+f"(d_o_lo[27]), "+f"(d_o_lo[28]), "+f"(d_o_lo[29]), "+f"(d_o_lo[30]), "+f"(d_o_lo[31])
                : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
                : "+f"(d_o_lo[0]), "+f"(d_o_lo[1]), "+f"(d_o_lo[2]), "+f"(d_o_lo[3]), "+f"(d_o_lo[4]), "+f"(d_o_lo[5]), "+f"(d_o_lo[6]), "+f"(d_o_lo[7]), "+f"(d_o_lo[8]), "+f"(d_o_lo[9]), "+f"(d_o_lo[10]), "+f"(d_o_lo[11]), "+f"(d_o_lo[12]), "+f"(d_o_lo[13]), "+f"(d_o_lo[14]), "+f"(d_o_lo[15]), "+f"(d_o_lo[16]), "+f"(d_o_lo[17]), "+f"(d_o_lo[18]), "+f"(d_o_lo[19]), "+f"(d_o_lo[20]), "+f"(d_o_lo[21]), "+f"(d_o_lo[22]), "+f"(d_o_lo[23]), "+f"(d_o_lo[24]), "+f"(d_o_lo[25]), "+f"(d_o_lo[26]), "+f"(d_o_lo[27]), "+f"(d_o_lo[28]), "+f"(d_o_lo[29]), "+f"(d_o_lo[30]), "+f"(d_o_lo[31])
                : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_2 + 128)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 1;\n}\n"
                : "+f"(d_o_hi[0]), "+f"(d_o_hi[1]), "+f"(d_o_hi[2]), "+f"(d_o_hi[3]), "+f"(d_o_hi[4]), "+f"(d_o_hi[5]), "+f"(d_o_hi[6]), "+f"(d_o_hi[7]), "+f"(d_o_hi[8]), "+f"(d_o_hi[9]), "+f"(d_o_hi[10]), "+f"(d_o_hi[11]), "+f"(d_o_hi[12]), "+f"(d_o_hi[13]), "+f"(d_o_hi[14]), "+f"(d_o_hi[15]), "+f"(d_o_hi[16]), "+f"(d_o_hi[17]), "+f"(d_o_hi[18]), "+f"(d_o_hi[19]), "+f"(d_o_hi[20]), "+f"(d_o_hi[21]), "+f"(d_o_hi[22]), "+f"(d_o_hi[23]), "+f"(d_o_hi[24]), "+f"(d_o_hi[25]), "+f"(d_o_hi[26]), "+f"(d_o_hi[27]), "+f"(d_o_hi[28]), "+f"(d_o_hi[29]), "+f"(d_o_hi[30]), "+f"(d_o_hi[31])
                : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_3)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
                : "+f"(d_o_hi[0]), "+f"(d_o_hi[1]), "+f"(d_o_hi[2]), "+f"(d_o_hi[3]), "+f"(d_o_hi[4]), "+f"(d_o_hi[5]), "+f"(d_o_hi[6]), "+f"(d_o_hi[7]), "+f"(d_o_hi[8]), "+f"(d_o_hi[9]), "+f"(d_o_hi[10]), "+f"(d_o_hi[11]), "+f"(d_o_hi[12]), "+f"(d_o_hi[13]), "+f"(d_o_hi[14]), "+f"(d_o_hi[15]), "+f"(d_o_hi[16]), "+f"(d_o_hi[17]), "+f"(d_o_hi[18]), "+f"(d_o_hi[19]), "+f"(d_o_hi[20]), "+f"(d_o_hi[21]), "+f"(d_o_hi[22]), "+f"(d_o_hi[23]), "+f"(d_o_hi[24]), "+f"(d_o_hi[25]), "+f"(d_o_hi[26]), "+f"(d_o_hi[27]), "+f"(d_o_hi[28]), "+f"(d_o_hi[29]), "+f"(d_o_hi[30]), "+f"(d_o_hi[31])
                : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_3 + 128)
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
            __nv_bfloat162 _bit_cast_src_18 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
            p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_18)[0];
            __nv_bfloat162 _bit_cast_src_19 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
            p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_19)[0];
            __nv_bfloat162 _bit_cast_src_20 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
            p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_20)[0];
            __nv_bfloat162 _bit_cast_src_21 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
            p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_21)[0];
            __nv_bfloat162 _bit_cast_src_22 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
            p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_22)[0];
            __nv_bfloat162 _bit_cast_src_23 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
            p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_23)[0];
            __nv_bfloat162 _bit_cast_src_24 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
            p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_24)[0];
            __nv_bfloat162 _bit_cast_src_25 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
            p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_25)[0];
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
                : "+f"(d_o_lo[0]), "+f"(d_o_lo[1]), "+f"(d_o_lo[2]), "+f"(d_o_lo[3]), "+f"(d_o_lo[4]), "+f"(d_o_lo[5]), "+f"(d_o_lo[6]), "+f"(d_o_lo[7]), "+f"(d_o_lo[8]), "+f"(d_o_lo[9]), "+f"(d_o_lo[10]), "+f"(d_o_lo[11]), "+f"(d_o_lo[12]), "+f"(d_o_lo[13]), "+f"(d_o_lo[14]), "+f"(d_o_lo[15]), "+f"(d_o_lo[16]), "+f"(d_o_lo[17]), "+f"(d_o_lo[18]), "+f"(d_o_lo[19]), "+f"(d_o_lo[20]), "+f"(d_o_lo[21]), "+f"(d_o_lo[22]), "+f"(d_o_lo[23]), "+f"(d_o_lo[24]), "+f"(d_o_lo[25]), "+f"(d_o_lo[26]), "+f"(d_o_lo[27]), "+f"(d_o_lo[28]), "+f"(d_o_lo[29]), "+f"(d_o_lo[30]), "+f"(d_o_lo[31])
                : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_2 + 256)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
                : "+f"(d_o_lo[0]), "+f"(d_o_lo[1]), "+f"(d_o_lo[2]), "+f"(d_o_lo[3]), "+f"(d_o_lo[4]), "+f"(d_o_lo[5]), "+f"(d_o_lo[6]), "+f"(d_o_lo[7]), "+f"(d_o_lo[8]), "+f"(d_o_lo[9]), "+f"(d_o_lo[10]), "+f"(d_o_lo[11]), "+f"(d_o_lo[12]), "+f"(d_o_lo[13]), "+f"(d_o_lo[14]), "+f"(d_o_lo[15]), "+f"(d_o_lo[16]), "+f"(d_o_lo[17]), "+f"(d_o_lo[18]), "+f"(d_o_lo[19]), "+f"(d_o_lo[20]), "+f"(d_o_lo[21]), "+f"(d_o_lo[22]), "+f"(d_o_lo[23]), "+f"(d_o_lo[24]), "+f"(d_o_lo[25]), "+f"(d_o_lo[26]), "+f"(d_o_lo[27]), "+f"(d_o_lo[28]), "+f"(d_o_lo[29]), "+f"(d_o_lo[30]), "+f"(d_o_lo[31])
                : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_2 + 384)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
                : "+f"(d_o_hi[0]), "+f"(d_o_hi[1]), "+f"(d_o_hi[2]), "+f"(d_o_hi[3]), "+f"(d_o_hi[4]), "+f"(d_o_hi[5]), "+f"(d_o_hi[6]), "+f"(d_o_hi[7]), "+f"(d_o_hi[8]), "+f"(d_o_hi[9]), "+f"(d_o_hi[10]), "+f"(d_o_hi[11]), "+f"(d_o_hi[12]), "+f"(d_o_hi[13]), "+f"(d_o_hi[14]), "+f"(d_o_hi[15]), "+f"(d_o_hi[16]), "+f"(d_o_hi[17]), "+f"(d_o_hi[18]), "+f"(d_o_hi[19]), "+f"(d_o_hi[20]), "+f"(d_o_hi[21]), "+f"(d_o_hi[22]), "+f"(d_o_hi[23]), "+f"(d_o_hi[24]), "+f"(d_o_hi[25]), "+f"(d_o_hi[26]), "+f"(d_o_hi[27]), "+f"(d_o_hi[28]), "+f"(d_o_hi[29]), "+f"(d_o_hi[30]), "+f"(d_o_hi[31])
                : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_3 + 256)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
                : "+f"(d_o_hi[0]), "+f"(d_o_hi[1]), "+f"(d_o_hi[2]), "+f"(d_o_hi[3]), "+f"(d_o_hi[4]), "+f"(d_o_hi[5]), "+f"(d_o_hi[6]), "+f"(d_o_hi[7]), "+f"(d_o_hi[8]), "+f"(d_o_hi[9]), "+f"(d_o_hi[10]), "+f"(d_o_hi[11]), "+f"(d_o_hi[12]), "+f"(d_o_hi[13]), "+f"(d_o_hi[14]), "+f"(d_o_hi[15]), "+f"(d_o_hi[16]), "+f"(d_o_hi[17]), "+f"(d_o_hi[18]), "+f"(d_o_hi[19]), "+f"(d_o_hi[20]), "+f"(d_o_hi[21]), "+f"(d_o_hi[22]), "+f"(d_o_hi[23]), "+f"(d_o_hi[24]), "+f"(d_o_hi[25]), "+f"(d_o_hi[26]), "+f"(d_o_hi[27]), "+f"(d_o_hi[28]), "+f"(d_o_hi[29]), "+f"(d_o_hi[30]), "+f"(d_o_hi[31])
                : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_3 + 384)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
            row_sum0 += _shfl_xor_sync_26;
            float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
            row_sum0 += _shfl_xor_sync_27;
            float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
            row_sum1 += _shfl_xor_sync_28;
            float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
            row_sum1 += _shfl_xor_sync_29;
            float _rcp_0 = approx_rcp(row_sum0);
            float _rcp_1 = approx_rcp(row_sum1);
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
            d_o_lo[0] = d_o_lo[0] * _rcp_0;
            d_o_lo[1] = d_o_lo[1] * _rcp_0;
            d_o_lo[2] = d_o_lo[2] * _rcp_1;
            d_o_lo[3] = d_o_lo[3] * _rcp_1;
            d_o_lo[4] = d_o_lo[4] * _rcp_0;
            d_o_lo[5] = d_o_lo[5] * _rcp_0;
            d_o_lo[6] = d_o_lo[6] * _rcp_1;
            d_o_lo[7] = d_o_lo[7] * _rcp_1;
            d_o_lo[8] = d_o_lo[8] * _rcp_0;
            d_o_lo[9] = d_o_lo[9] * _rcp_0;
            d_o_lo[10] = d_o_lo[10] * _rcp_1;
            d_o_lo[11] = d_o_lo[11] * _rcp_1;
            d_o_lo[12] = d_o_lo[12] * _rcp_0;
            d_o_lo[13] = d_o_lo[13] * _rcp_0;
            d_o_lo[14] = d_o_lo[14] * _rcp_1;
            d_o_lo[15] = d_o_lo[15] * _rcp_1;
            d_o_lo[16] = d_o_lo[16] * _rcp_0;
            d_o_lo[17] = d_o_lo[17] * _rcp_0;
            d_o_lo[18] = d_o_lo[18] * _rcp_1;
            d_o_lo[19] = d_o_lo[19] * _rcp_1;
            d_o_lo[20] = d_o_lo[20] * _rcp_0;
            d_o_lo[21] = d_o_lo[21] * _rcp_0;
            d_o_lo[22] = d_o_lo[22] * _rcp_1;
            d_o_lo[23] = d_o_lo[23] * _rcp_1;
            d_o_lo[24] = d_o_lo[24] * _rcp_0;
            d_o_lo[25] = d_o_lo[25] * _rcp_0;
            d_o_lo[26] = d_o_lo[26] * _rcp_1;
            d_o_lo[27] = d_o_lo[27] * _rcp_1;
            d_o_lo[28] = d_o_lo[28] * _rcp_0;
            d_o_lo[29] = d_o_lo[29] * _rcp_0;
            d_o_lo[30] = d_o_lo[30] * _rcp_1;
            d_o_lo[31] = d_o_lo[31] * _rcp_1;
            int qj = lane & 3;
            int qodd = qj & 1;
            unsigned int o_pk[2];
            unsigned int o_vec[2];
            int o_row_base = q_row * 128 + qj * 2 + qodd * 6;
            int m_local_r_1_0 = m0_local;
            int row_off_1_0 = o_row_base + m_local_r_1_0 * 128;
            __nv_bfloat162 _bit_cast_src_30 = __float22bfloat162_rn(make_float2(d_o_lo[0], d_o_lo[1]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_30)[0];
            __nv_bfloat162 _bit_cast_src_31 = __float22bfloat162_rn(make_float2(d_o_lo[4], d_o_lo[5]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_31)[0];
            unsigned int send_3_0 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send_3_0, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_0);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_0 : o_pk[1]);
            reinterpret_cast<int2*>(O + row_off_1_0)[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_32 = __float22bfloat162_rn(make_float2(d_o_lo[8], d_o_lo[9]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_32)[0];
            __nv_bfloat162 _bit_cast_src_33 = __float22bfloat162_rn(make_float2(d_o_lo[12], d_o_lo[13]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_33)[0];
            unsigned int send_3_1 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_3_1, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_1);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_1 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_0 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_34 = __float22bfloat162_rn(make_float2(d_o_lo[16], d_o_lo[17]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_34)[0];
            __nv_bfloat162 _bit_cast_src_35 = __float22bfloat162_rn(make_float2(d_o_lo[20], d_o_lo[21]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_35)[0];
            unsigned int send_3_2 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_3_2, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_2);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_2 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_0 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_36 = __float22bfloat162_rn(make_float2(d_o_lo[24], d_o_lo[25]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_36)[0];
            __nv_bfloat162 _bit_cast_src_37 = __float22bfloat162_rn(make_float2(d_o_lo[28], d_o_lo[29]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_37)[0];
            unsigned int send_3_3 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_3_3, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_3);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_3 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_0 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
            int m_local_r_1_1 = m1_local;
            int row_off_1_1 = o_row_base + m_local_r_1_1 * 128;
            __nv_bfloat162 _bit_cast_src_38 = __float22bfloat162_rn(make_float2(d_o_lo[2], d_o_lo[3]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_38)[0];
            __nv_bfloat162 _bit_cast_src_39 = __float22bfloat162_rn(make_float2(d_o_lo[6], d_o_lo[7]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_39)[0];
            unsigned int send_3_4 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_3_4, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_4);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_4 : o_pk[1]);
            reinterpret_cast<int2*>(O + row_off_1_1)[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_40 = __float22bfloat162_rn(make_float2(d_o_lo[10], d_o_lo[11]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_40)[0];
            __nv_bfloat162 _bit_cast_src_41 = __float22bfloat162_rn(make_float2(d_o_lo[14], d_o_lo[15]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_41)[0];
            unsigned int send_3_5 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_3_5, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_5);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_5 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_1 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_42 = __float22bfloat162_rn(make_float2(d_o_lo[18], d_o_lo[19]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_42)[0];
            __nv_bfloat162 _bit_cast_src_43 = __float22bfloat162_rn(make_float2(d_o_lo[22], d_o_lo[23]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
            unsigned int send_3_6 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_3_6, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_6);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_6 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_1 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
            __nv_bfloat162 _bit_cast_src_44 = __float22bfloat162_rn(make_float2(d_o_lo[26], d_o_lo[27]));
            o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
            __nv_bfloat162 _bit_cast_src_45 = __float22bfloat162_rn(make_float2(d_o_lo[30], d_o_lo[31]));
            o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
            unsigned int send_3_7 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
            unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_3_7, 1);
            o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_7);
            o_vec[1] = ((qodd == 0) ? _shfl_xor_7 : o_pk[1]);
            reinterpret_cast<int2*>(O + (row_off_1_1 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            d_o_hi[0] = d_o_hi[0] * _rcp_0;
            d_o_hi[1] = d_o_hi[1] * _rcp_0;
            d_o_hi[2] = d_o_hi[2] * _rcp_1;
            d_o_hi[3] = d_o_hi[3] * _rcp_1;
            d_o_hi[4] = d_o_hi[4] * _rcp_0;
            d_o_hi[5] = d_o_hi[5] * _rcp_0;
            d_o_hi[6] = d_o_hi[6] * _rcp_1;
            d_o_hi[7] = d_o_hi[7] * _rcp_1;
            d_o_hi[8] = d_o_hi[8] * _rcp_0;
            d_o_hi[9] = d_o_hi[9] * _rcp_0;
            d_o_hi[10] = d_o_hi[10] * _rcp_1;
            d_o_hi[11] = d_o_hi[11] * _rcp_1;
            d_o_hi[12] = d_o_hi[12] * _rcp_0;
            d_o_hi[13] = d_o_hi[13] * _rcp_0;
            d_o_hi[14] = d_o_hi[14] * _rcp_1;
            d_o_hi[15] = d_o_hi[15] * _rcp_1;
            d_o_hi[16] = d_o_hi[16] * _rcp_0;
            d_o_hi[17] = d_o_hi[17] * _rcp_0;
            d_o_hi[18] = d_o_hi[18] * _rcp_1;
            d_o_hi[19] = d_o_hi[19] * _rcp_1;
            d_o_hi[20] = d_o_hi[20] * _rcp_0;
            d_o_hi[21] = d_o_hi[21] * _rcp_0;
            d_o_hi[22] = d_o_hi[22] * _rcp_1;
            d_o_hi[23] = d_o_hi[23] * _rcp_1;
            d_o_hi[24] = d_o_hi[24] * _rcp_0;
            d_o_hi[25] = d_o_hi[25] * _rcp_0;
            d_o_hi[26] = d_o_hi[26] * _rcp_1;
            d_o_hi[27] = d_o_hi[27] * _rcp_1;
            d_o_hi[28] = d_o_hi[28] * _rcp_0;
            d_o_hi[29] = d_o_hi[29] * _rcp_0;
            d_o_hi[30] = d_o_hi[30] * _rcp_1;
            d_o_hi[31] = d_o_hi[31] * _rcp_1;
            int qj_9 = lane & 3;
            int qodd_10 = qj_9 & 1;
            unsigned int o_pk_11[2];
            unsigned int o_vec_12[2];
            int o_row_base_13 = q_row * 128 + 64 + qj_9 * 2 + qodd_10 * 6;
            int m_local_r_0 = m0_local;
            int row_off_0 = o_row_base_13 + m_local_r_0 * 128;
            __nv_bfloat162 _bit_cast_src_46 = __float22bfloat162_rn(make_float2(d_o_hi[0], d_o_hi[1]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_46)[0];
            __nv_bfloat162 _bit_cast_src_47 = __float22bfloat162_rn(make_float2(d_o_hi[4], d_o_hi[5]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_47)[0];
            unsigned int send_0 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send_0, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_8);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_8 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + row_off_0)[0] = reinterpret_cast<int2*>(o_vec_12)[0];
            __nv_bfloat162 _bit_cast_src_48 = __float22bfloat162_rn(make_float2(d_o_hi[8], d_o_hi[9]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_48)[0];
            __nv_bfloat162 _bit_cast_src_49 = __float22bfloat162_rn(make_float2(d_o_hi[12], d_o_hi[13]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_49)[0];
            unsigned int send_1 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_1, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_9);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_9 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + (row_off_0 + 16))[0] = reinterpret_cast<int2*>(o_vec_12)[0];
            __nv_bfloat162 _bit_cast_src_50 = __float22bfloat162_rn(make_float2(d_o_hi[16], d_o_hi[17]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_50)[0];
            __nv_bfloat162 _bit_cast_src_51 = __float22bfloat162_rn(make_float2(d_o_hi[20], d_o_hi[21]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_51)[0];
            unsigned int send_2 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_2, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_10);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_10 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + (row_off_0 + 32))[0] = reinterpret_cast<int2*>(o_vec_12)[0];
            __nv_bfloat162 _bit_cast_src_52 = __float22bfloat162_rn(make_float2(d_o_hi[24], d_o_hi[25]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_52)[0];
            __nv_bfloat162 _bit_cast_src_53 = __float22bfloat162_rn(make_float2(d_o_hi[28], d_o_hi[29]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_53)[0];
            unsigned int send_3 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_3, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_11);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_11 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + (row_off_0 + 48))[0] = reinterpret_cast<int2*>(o_vec_12)[0];
            int m_local_r_1 = m1_local;
            int row_off_1 = o_row_base_13 + m_local_r_1 * 128;
            __nv_bfloat162 _bit_cast_src_54 = __float22bfloat162_rn(make_float2(d_o_hi[2], d_o_hi[3]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_54)[0];
            __nv_bfloat162 _bit_cast_src_55 = __float22bfloat162_rn(make_float2(d_o_hi[6], d_o_hi[7]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_55)[0];
            unsigned int send_4 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_4, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_12);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_12 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + row_off_1)[0] = reinterpret_cast<int2*>(o_vec_12)[0];
            __nv_bfloat162 _bit_cast_src_56 = __float22bfloat162_rn(make_float2(d_o_hi[10], d_o_hi[11]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_56)[0];
            __nv_bfloat162 _bit_cast_src_57 = __float22bfloat162_rn(make_float2(d_o_hi[14], d_o_hi[15]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_57)[0];
            unsigned int send_5 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_5, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_13);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_13 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + (row_off_1 + 16))[0] = reinterpret_cast<int2*>(o_vec_12)[0];
            __nv_bfloat162 _bit_cast_src_58 = __float22bfloat162_rn(make_float2(d_o_hi[18], d_o_hi[19]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_58)[0];
            __nv_bfloat162 _bit_cast_src_59 = __float22bfloat162_rn(make_float2(d_o_hi[22], d_o_hi[23]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_59)[0];
            unsigned int send_6 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_6, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_14);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_14 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + (row_off_1 + 32))[0] = reinterpret_cast<int2*>(o_vec_12)[0];
            __nv_bfloat162 _bit_cast_src_60 = __float22bfloat162_rn(make_float2(d_o_hi[26], d_o_hi[27]));
            o_pk_11[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_60)[0];
            __nv_bfloat162 _bit_cast_src_61 = __float22bfloat162_rn(make_float2(d_o_hi[30], d_o_hi[31]));
            o_pk_11[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_61)[0];
            unsigned int send_7 = ((qodd_10 != 0) ? o_pk_11[0] : o_pk_11[1]);
            unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_7, 1);
            o_vec_12[0] = ((qodd_10 == 0) ? o_pk_11[0] : _shfl_xor_15);
            o_vec_12[1] = ((qodd_10 == 0) ? _shfl_xor_15 : o_pk_11[1]);
            reinterpret_cast<int2*>(O + (row_off_1 + 48))[0] = reinterpret_cast<int2*>(o_vec_12)[0];
        } else {
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
            float exp2_16 = approx_exp2(d_qk[16] - row_max0);
            float exp2_1_16 = approx_exp2(d_qk[17] - row_max0);
            d_qk[16] = exp2_16;
            d_qk[17] = exp2_1_16;
            row_sum0 += exp2_16 + exp2_1_16;
            float exp2_17 = approx_exp2(d_qk[18] - row_max1);
            float exp2_1_17 = approx_exp2(d_qk[19] - row_max1);
            d_qk[18] = exp2_17;
            d_qk[19] = exp2_1_17;
            row_sum1 += exp2_17 + exp2_1_17;
            float exp2_18 = approx_exp2(d_qk[20] - row_max0);
            float exp2_1_18 = approx_exp2(d_qk[21] - row_max0);
            d_qk[20] = exp2_18;
            d_qk[21] = exp2_1_18;
            row_sum0 += exp2_18 + exp2_1_18;
            float exp2_19 = approx_exp2(d_qk[22] - row_max1);
            float exp2_1_19 = approx_exp2(d_qk[23] - row_max1);
            d_qk[22] = exp2_19;
            d_qk[23] = exp2_1_19;
            row_sum1 += exp2_19 + exp2_1_19;
            float exp2_20 = approx_exp2(d_qk[24] - row_max0);
            float exp2_1_20 = approx_exp2(d_qk[25] - row_max0);
            d_qk[24] = exp2_20;
            d_qk[25] = exp2_1_20;
            row_sum0 += exp2_20 + exp2_1_20;
            float exp2_21 = approx_exp2(d_qk[26] - row_max1);
            float exp2_1_21 = approx_exp2(d_qk[27] - row_max1);
            d_qk[26] = exp2_21;
            d_qk[27] = exp2_1_21;
            row_sum1 += exp2_21 + exp2_1_21;
            float exp2_22 = approx_exp2(d_qk[28] - row_max0);
            float exp2_1_22 = approx_exp2(d_qk[29] - row_max0);
            d_qk[28] = exp2_22;
            d_qk[29] = exp2_1_22;
            row_sum0 += exp2_22 + exp2_1_22;
            float exp2_23 = approx_exp2(d_qk[30] - row_max1);
            float exp2_1_23 = approx_exp2(d_qk[31] - row_max1);
            d_qk[30] = exp2_23;
            d_qk[31] = exp2_1_23;
            row_sum1 += exp2_23 + exp2_1_23;
            __nv_bfloat162 _bit_cast_src_62 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
            p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_62)[0];
            __nv_bfloat162 _bit_cast_src_63 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
            p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_63)[0];
            __nv_bfloat162 _bit_cast_src_64 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
            p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_64)[0];
            __nv_bfloat162 _bit_cast_src_65 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
            p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_65)[0];
            __nv_bfloat162 _bit_cast_src_66 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
            p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_66)[0];
            __nv_bfloat162 _bit_cast_src_67 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
            p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_67)[0];
            __nv_bfloat162 _bit_cast_src_68 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
            p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_68)[0];
            __nv_bfloat162 _bit_cast_src_69 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
            p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_69)[0];
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
            float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
            row_sum0 += _shfl_xor_sync_70;
            float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
            row_sum0 += _shfl_xor_sync_71;
            float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
            row_sum1 += _shfl_xor_sync_72;
            float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
            row_sum1 += _shfl_xor_sync_73;
            int qj_k1 = lane & 3;
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
            int qj1 = qj_k1 & 1;
            int qj2 = qj_k1 & 2;
            unsigned int o_vec_1[4];
            unsigned int o_tmp[4];
            int o_row_base_1 = q_row * 128;
            int m_local_r_1_2 = m0_local;
            __nv_bfloat162 _bit_cast_src_74 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_74)[0];
            __nv_bfloat162 _bit_cast_src_75 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_75)[0];
            __nv_bfloat162 _bit_cast_src_76 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_76)[0];
            __nv_bfloat162 _bit_cast_src_77 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_77)[0];
            unsigned int _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_78;
            unsigned int _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_79;
            unsigned int _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_80;
            unsigned int _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_81;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_82;
            unsigned int _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_83;
            unsigned int _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_84;
            unsigned int _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_85;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_0 = o_row_base_1 + m_local_r_1_2 * 128 + qj_k1 * 8;
            reinterpret_cast<int4*>(O + o_off_3_0)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_86 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_86)[0];
            __nv_bfloat162 _bit_cast_src_87 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_87)[0];
            __nv_bfloat162 _bit_cast_src_88 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_88)[0];
            __nv_bfloat162 _bit_cast_src_89 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_89)[0];
            unsigned int _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_90;
            unsigned int _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_91;
            unsigned int _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_92;
            unsigned int _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_93;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_94;
            unsigned int _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_95;
            unsigned int _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_96;
            unsigned int _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_97;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_1 = o_row_base_1 + m_local_r_1_2 * 128 + (4 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_3_1)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_98 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_98)[0];
            __nv_bfloat162 _bit_cast_src_99 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_99)[0];
            __nv_bfloat162 _bit_cast_src_100 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_100)[0];
            __nv_bfloat162 _bit_cast_src_101 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_101)[0];
            unsigned int _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_102;
            unsigned int _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_103;
            unsigned int _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_104;
            unsigned int _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_105;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_106;
            unsigned int _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_107;
            unsigned int _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_108;
            unsigned int _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_109;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_2 = o_row_base_1 + m_local_r_1_2 * 128 + (8 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_3_2)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_110 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_110)[0];
            __nv_bfloat162 _bit_cast_src_111 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_111)[0];
            __nv_bfloat162 _bit_cast_src_112 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_112)[0];
            __nv_bfloat162 _bit_cast_src_113 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_113)[0];
            unsigned int _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_114;
            unsigned int _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_115;
            unsigned int _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_116;
            unsigned int _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_117;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_118;
            unsigned int _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_119;
            unsigned int _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_120;
            unsigned int _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_121;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_3 = o_row_base_1 + m_local_r_1_2 * 128 + (12 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_3_3)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            int m_local_r_1_3 = m1_local;
            __nv_bfloat162 _bit_cast_src_122 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_122)[0];
            __nv_bfloat162 _bit_cast_src_123 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_123)[0];
            __nv_bfloat162 _bit_cast_src_124 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_124)[0];
            __nv_bfloat162 _bit_cast_src_125 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_125)[0];
            unsigned int _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_126;
            unsigned int _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_127;
            unsigned int _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_128;
            unsigned int _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_129;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_130;
            unsigned int _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_131;
            unsigned int _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_132;
            unsigned int _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_133;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_4 = o_row_base_1 + m_local_r_1_3 * 128 + qj_k1 * 8;
            reinterpret_cast<int4*>(O + o_off_3_4)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_134 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_134)[0];
            __nv_bfloat162 _bit_cast_src_135 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_135)[0];
            __nv_bfloat162 _bit_cast_src_136 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_136)[0];
            __nv_bfloat162 _bit_cast_src_137 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_137)[0];
            unsigned int _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_138;
            unsigned int _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_139;
            unsigned int _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_140;
            unsigned int _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_141;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_142;
            unsigned int _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_143;
            unsigned int _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_144;
            unsigned int _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_145;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_5 = o_row_base_1 + m_local_r_1_3 * 128 + (4 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_3_5)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_146 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_146)[0];
            __nv_bfloat162 _bit_cast_src_147 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_147)[0];
            __nv_bfloat162 _bit_cast_src_148 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_148)[0];
            __nv_bfloat162 _bit_cast_src_149 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_149)[0];
            unsigned int _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_150;
            unsigned int _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_151;
            unsigned int _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_152;
            unsigned int _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_153;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_154;
            unsigned int _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_155;
            unsigned int _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_156;
            unsigned int _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_157;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_6 = o_row_base_1 + m_local_r_1_3 * 128 + (8 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_3_6)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
            __nv_bfloat162 _bit_cast_src_158 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
            o_vec_1[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_158)[0];
            __nv_bfloat162 _bit_cast_src_159 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
            o_vec_1[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_159)[0];
            __nv_bfloat162 _bit_cast_src_160 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
            o_vec_1[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_160)[0];
            __nv_bfloat162 _bit_cast_src_161 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
            o_vec_1[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_161)[0];
            unsigned int _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 1);
            o_tmp[0] = _shfl_xor_sync_162;
            unsigned int _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 1);
            o_tmp[1] = _shfl_xor_sync_163;
            unsigned int _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 1);
            o_tmp[2] = _shfl_xor_sync_164;
            unsigned int _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 1);
            o_tmp[3] = _shfl_xor_sync_165;
            o_vec_1[0] = ((qj1 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj1 == 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj1 != 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj1 == 0) ? o_tmp[3] : o_vec_1[3]);
            unsigned int _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[2], 2);
            o_tmp[0] = _shfl_xor_sync_166;
            unsigned int _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[3], 2);
            o_tmp[1] = _shfl_xor_sync_167;
            unsigned int _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[0], 2);
            o_tmp[2] = _shfl_xor_sync_168;
            unsigned int _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, o_vec_1[1], 2);
            o_tmp[3] = _shfl_xor_sync_169;
            o_vec_1[0] = ((qj2 != 0) ? o_tmp[0] : o_vec_1[0]);
            o_vec_1[1] = ((qj2 != 0) ? o_tmp[1] : o_vec_1[1]);
            o_vec_1[2] = ((qj2 == 0) ? o_tmp[2] : o_vec_1[2]);
            o_vec_1[3] = ((qj2 == 0) ? o_tmp[3] : o_vec_1[3]);
            int o_off_3_7 = o_row_base_1 + m_local_r_1_3 * 128 + (12 + qj_k1) * 8;
            reinterpret_cast<int4*>(O + o_off_3_7)[0] = reinterpret_cast<int4*>(o_vec_1)[0];
        }
    }
    int qj_1 = lane & 3;

    // Cleanup
    __syncthreads();
}

} // extern "C"
