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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_1_STAGES 1
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 8192
#define SMEM_Q_SMEM_STRIDE 8192
#define SMEM_K_SMEM_OFF 9216
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 41984
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_TOTAL 74752
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 3

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






__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_fp8_attention_c0e2681a52a8b76b0256(const __grid_constant__ CUtensorMap Q8, const __grid_constant__ CUtensorMap K8, const __grid_constant__ CUtensorMap VT8, __nv_bfloat16* __restrict__ O, float* __restrict__ q_scale, float* __restrict__ k_scale, float* __restrict__ v_scale, int seqlen, int num_heads, int o_stride_b, int o_stride_s, int o_stride_h, float softmax_scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_ready_addr (mbar_base + 0)
    #define k_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int8_t* q_smem = reinterpret_cast<int8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    int8_t* k_smem = reinterpret_cast<int8_t*>(smem_raw + 9216);
    const int k_smem_addr = smem + 9216;
    uint8_t* vt_smem = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int vt_smem_addr = smem + 41984;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 5 barriers)
    // Mbarriers at smem_raw[0..40)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (declaration order) ===
    int num_blocks = (seqlen + 128 - 1) / 128;
    int nq = (seqlen + 64 - 1) / 64;
    int qi = bid % nq;
    int bh = bid / nq;
    int kbase = bh * num_blocks;
    int blk = kbase + qi / 2;
    int q_row_start = qi * 64;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(q_ready_addr, 8192);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(q_smem_addr), "l"((&Q8)), "r"(0), "r"(q_row_start), "r"(bh),
                   "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(k_full_addr, 16384);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(k_smem_addr), "l"((&K8)), "r"(0), "r"(0), "r"(bh),
                   "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(v_full_addr, 16384);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(vt_smem_addr), "l"((&VT8)), "r"(0), "r"(0), "r"(bh),
                   "r"(v_full_addr), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(k_full_addr + 8, 16384);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(k_smem_addr + 16384), "l"((&K8)), "r"(0), "r"(128), "r"(bh),
                   "r"(k_full_addr + 8), "l"(0x14F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(v_full_addr + 8, 16384);
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(vt_smem_addr + 16384), "l"((&VT8)), "r"(128), "r"(0), "r"(bh),
                   "r"(v_full_addr + 8), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int lane_q = lane % 4;
    int lane_r = lane / 4;
    int m0_local = warp * 16 + lane_r;
    int row0 = q_row_start + m0_local;
    int row1 = row0 + 8;
    float sq = q_scale[blk];
    float cq = softmax_scale_log2 * sq;
    float d_o[64];
    float d_qk[64];
    unsigned int p8[16];
    float o_tmp[4];
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
    float row_max0 = -CUDART_INF_F;
    float row_max1 = -CUDART_INF_F;
    float row_sum0 = 0.0f;
    float row_sum1 = 0.0f;
    mbarrier_wait(q_ready_addr, 0);
    uint64_t _wgmma_desc_0 = (((uint64_t)(((q_smem_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_a_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
    #pragma unroll 1
    for (int kt = 0; kt < num_blocks; kt++) {
        int stage = kt % 2;
        int full_phase = kt / 2 & 1;
        float ck0 = cq * k_scale[2 * (kbase + kt)];
        float ck1 = cq * k_scale[2 * (kbase + kt) + 1];
        mbarrier_wait(k_full_addr + (stage) * 8, full_phase);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_1 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0;\n}\n"
            : "+r"(reinterpret_cast<unsigned&>(d_qk[0])), "+r"(reinterpret_cast<unsigned&>(d_qk[1])), "+r"(reinterpret_cast<unsigned&>(d_qk[2])), "+r"(reinterpret_cast<unsigned&>(d_qk[3])), "+r"(reinterpret_cast<unsigned&>(d_qk[4])), "+r"(reinterpret_cast<unsigned&>(d_qk[5])), "+r"(reinterpret_cast<unsigned&>(d_qk[6])), "+r"(reinterpret_cast<unsigned&>(d_qk[7])), "+r"(reinterpret_cast<unsigned&>(d_qk[8])), "+r"(reinterpret_cast<unsigned&>(d_qk[9])), "+r"(reinterpret_cast<unsigned&>(d_qk[10])), "+r"(reinterpret_cast<unsigned&>(d_qk[11])), "+r"(reinterpret_cast<unsigned&>(d_qk[12])), "+r"(reinterpret_cast<unsigned&>(d_qk[13])), "+r"(reinterpret_cast<unsigned&>(d_qk[14])), "+r"(reinterpret_cast<unsigned&>(d_qk[15])), "+r"(reinterpret_cast<unsigned&>(d_qk[16])), "+r"(reinterpret_cast<unsigned&>(d_qk[17])), "+r"(reinterpret_cast<unsigned&>(d_qk[18])), "+r"(reinterpret_cast<unsigned&>(d_qk[19])), "+r"(reinterpret_cast<unsigned&>(d_qk[20])), "+r"(reinterpret_cast<unsigned&>(d_qk[21])), "+r"(reinterpret_cast<unsigned&>(d_qk[22])), "+r"(reinterpret_cast<unsigned&>(d_qk[23])), "+r"(reinterpret_cast<unsigned&>(d_qk[24])), "+r"(reinterpret_cast<unsigned&>(d_qk[25])), "+r"(reinterpret_cast<unsigned&>(d_qk[26])), "+r"(reinterpret_cast<unsigned&>(d_qk[27])), "+r"(reinterpret_cast<unsigned&>(d_qk[28])), "+r"(reinterpret_cast<unsigned&>(d_qk[29])), "+r"(reinterpret_cast<unsigned&>(d_qk[30])), "+r"(reinterpret_cast<unsigned&>(d_qk[31])), "+r"(reinterpret_cast<unsigned&>(d_qk[32])), "+r"(reinterpret_cast<unsigned&>(d_qk[33])), "+r"(reinterpret_cast<unsigned&>(d_qk[34])), "+r"(reinterpret_cast<unsigned&>(d_qk[35])), "+r"(reinterpret_cast<unsigned&>(d_qk[36])), "+r"(reinterpret_cast<unsigned&>(d_qk[37])), "+r"(reinterpret_cast<unsigned&>(d_qk[38])), "+r"(reinterpret_cast<unsigned&>(d_qk[39])), "+r"(reinterpret_cast<unsigned&>(d_qk[40])), "+r"(reinterpret_cast<unsigned&>(d_qk[41])), "+r"(reinterpret_cast<unsigned&>(d_qk[42])), "+r"(reinterpret_cast<unsigned&>(d_qk[43])), "+r"(reinterpret_cast<unsigned&>(d_qk[44])), "+r"(reinterpret_cast<unsigned&>(d_qk[45])), "+r"(reinterpret_cast<unsigned&>(d_qk[46])), "+r"(reinterpret_cast<unsigned&>(d_qk[47])), "+r"(reinterpret_cast<unsigned&>(d_qk[48])), "+r"(reinterpret_cast<unsigned&>(d_qk[49])), "+r"(reinterpret_cast<unsigned&>(d_qk[50])), "+r"(reinterpret_cast<unsigned&>(d_qk[51])), "+r"(reinterpret_cast<unsigned&>(d_qk[52])), "+r"(reinterpret_cast<unsigned&>(d_qk[53])), "+r"(reinterpret_cast<unsigned&>(d_qk[54])), "+r"(reinterpret_cast<unsigned&>(d_qk[55])), "+r"(reinterpret_cast<unsigned&>(d_qk[56])), "+r"(reinterpret_cast<unsigned&>(d_qk[57])), "+r"(reinterpret_cast<unsigned&>(d_qk[58])), "+r"(reinterpret_cast<unsigned&>(d_qk[59])), "+r"(reinterpret_cast<unsigned&>(d_qk[60])), "+r"(reinterpret_cast<unsigned&>(d_qk[61])), "+r"(reinterpret_cast<unsigned&>(d_qk[62])), "+r"(reinterpret_cast<unsigned&>(d_qk[63]))
            : "l"(_wgmma_a_0), "l"(_wgmma_b_0)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1;\n}\n"
            : "+r"(reinterpret_cast<unsigned&>(d_qk[0])), "+r"(reinterpret_cast<unsigned&>(d_qk[1])), "+r"(reinterpret_cast<unsigned&>(d_qk[2])), "+r"(reinterpret_cast<unsigned&>(d_qk[3])), "+r"(reinterpret_cast<unsigned&>(d_qk[4])), "+r"(reinterpret_cast<unsigned&>(d_qk[5])), "+r"(reinterpret_cast<unsigned&>(d_qk[6])), "+r"(reinterpret_cast<unsigned&>(d_qk[7])), "+r"(reinterpret_cast<unsigned&>(d_qk[8])), "+r"(reinterpret_cast<unsigned&>(d_qk[9])), "+r"(reinterpret_cast<unsigned&>(d_qk[10])), "+r"(reinterpret_cast<unsigned&>(d_qk[11])), "+r"(reinterpret_cast<unsigned&>(d_qk[12])), "+r"(reinterpret_cast<unsigned&>(d_qk[13])), "+r"(reinterpret_cast<unsigned&>(d_qk[14])), "+r"(reinterpret_cast<unsigned&>(d_qk[15])), "+r"(reinterpret_cast<unsigned&>(d_qk[16])), "+r"(reinterpret_cast<unsigned&>(d_qk[17])), "+r"(reinterpret_cast<unsigned&>(d_qk[18])), "+r"(reinterpret_cast<unsigned&>(d_qk[19])), "+r"(reinterpret_cast<unsigned&>(d_qk[20])), "+r"(reinterpret_cast<unsigned&>(d_qk[21])), "+r"(reinterpret_cast<unsigned&>(d_qk[22])), "+r"(reinterpret_cast<unsigned&>(d_qk[23])), "+r"(reinterpret_cast<unsigned&>(d_qk[24])), "+r"(reinterpret_cast<unsigned&>(d_qk[25])), "+r"(reinterpret_cast<unsigned&>(d_qk[26])), "+r"(reinterpret_cast<unsigned&>(d_qk[27])), "+r"(reinterpret_cast<unsigned&>(d_qk[28])), "+r"(reinterpret_cast<unsigned&>(d_qk[29])), "+r"(reinterpret_cast<unsigned&>(d_qk[30])), "+r"(reinterpret_cast<unsigned&>(d_qk[31])), "+r"(reinterpret_cast<unsigned&>(d_qk[32])), "+r"(reinterpret_cast<unsigned&>(d_qk[33])), "+r"(reinterpret_cast<unsigned&>(d_qk[34])), "+r"(reinterpret_cast<unsigned&>(d_qk[35])), "+r"(reinterpret_cast<unsigned&>(d_qk[36])), "+r"(reinterpret_cast<unsigned&>(d_qk[37])), "+r"(reinterpret_cast<unsigned&>(d_qk[38])), "+r"(reinterpret_cast<unsigned&>(d_qk[39])), "+r"(reinterpret_cast<unsigned&>(d_qk[40])), "+r"(reinterpret_cast<unsigned&>(d_qk[41])), "+r"(reinterpret_cast<unsigned&>(d_qk[42])), "+r"(reinterpret_cast<unsigned&>(d_qk[43])), "+r"(reinterpret_cast<unsigned&>(d_qk[44])), "+r"(reinterpret_cast<unsigned&>(d_qk[45])), "+r"(reinterpret_cast<unsigned&>(d_qk[46])), "+r"(reinterpret_cast<unsigned&>(d_qk[47])), "+r"(reinterpret_cast<unsigned&>(d_qk[48])), "+r"(reinterpret_cast<unsigned&>(d_qk[49])), "+r"(reinterpret_cast<unsigned&>(d_qk[50])), "+r"(reinterpret_cast<unsigned&>(d_qk[51])), "+r"(reinterpret_cast<unsigned&>(d_qk[52])), "+r"(reinterpret_cast<unsigned&>(d_qk[53])), "+r"(reinterpret_cast<unsigned&>(d_qk[54])), "+r"(reinterpret_cast<unsigned&>(d_qk[55])), "+r"(reinterpret_cast<unsigned&>(d_qk[56])), "+r"(reinterpret_cast<unsigned&>(d_qk[57])), "+r"(reinterpret_cast<unsigned&>(d_qk[58])), "+r"(reinterpret_cast<unsigned&>(d_qk[59])), "+r"(reinterpret_cast<unsigned&>(d_qk[60])), "+r"(reinterpret_cast<unsigned&>(d_qk[61])), "+r"(reinterpret_cast<unsigned&>(d_qk[62])), "+r"(reinterpret_cast<unsigned&>(d_qk[63]))
            : "l"(_wgmma_a_0 + 2), "l"(_wgmma_b_0 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1;\n}\n"
            : "+r"(reinterpret_cast<unsigned&>(d_qk[0])), "+r"(reinterpret_cast<unsigned&>(d_qk[1])), "+r"(reinterpret_cast<unsigned&>(d_qk[2])), "+r"(reinterpret_cast<unsigned&>(d_qk[3])), "+r"(reinterpret_cast<unsigned&>(d_qk[4])), "+r"(reinterpret_cast<unsigned&>(d_qk[5])), "+r"(reinterpret_cast<unsigned&>(d_qk[6])), "+r"(reinterpret_cast<unsigned&>(d_qk[7])), "+r"(reinterpret_cast<unsigned&>(d_qk[8])), "+r"(reinterpret_cast<unsigned&>(d_qk[9])), "+r"(reinterpret_cast<unsigned&>(d_qk[10])), "+r"(reinterpret_cast<unsigned&>(d_qk[11])), "+r"(reinterpret_cast<unsigned&>(d_qk[12])), "+r"(reinterpret_cast<unsigned&>(d_qk[13])), "+r"(reinterpret_cast<unsigned&>(d_qk[14])), "+r"(reinterpret_cast<unsigned&>(d_qk[15])), "+r"(reinterpret_cast<unsigned&>(d_qk[16])), "+r"(reinterpret_cast<unsigned&>(d_qk[17])), "+r"(reinterpret_cast<unsigned&>(d_qk[18])), "+r"(reinterpret_cast<unsigned&>(d_qk[19])), "+r"(reinterpret_cast<unsigned&>(d_qk[20])), "+r"(reinterpret_cast<unsigned&>(d_qk[21])), "+r"(reinterpret_cast<unsigned&>(d_qk[22])), "+r"(reinterpret_cast<unsigned&>(d_qk[23])), "+r"(reinterpret_cast<unsigned&>(d_qk[24])), "+r"(reinterpret_cast<unsigned&>(d_qk[25])), "+r"(reinterpret_cast<unsigned&>(d_qk[26])), "+r"(reinterpret_cast<unsigned&>(d_qk[27])), "+r"(reinterpret_cast<unsigned&>(d_qk[28])), "+r"(reinterpret_cast<unsigned&>(d_qk[29])), "+r"(reinterpret_cast<unsigned&>(d_qk[30])), "+r"(reinterpret_cast<unsigned&>(d_qk[31])), "+r"(reinterpret_cast<unsigned&>(d_qk[32])), "+r"(reinterpret_cast<unsigned&>(d_qk[33])), "+r"(reinterpret_cast<unsigned&>(d_qk[34])), "+r"(reinterpret_cast<unsigned&>(d_qk[35])), "+r"(reinterpret_cast<unsigned&>(d_qk[36])), "+r"(reinterpret_cast<unsigned&>(d_qk[37])), "+r"(reinterpret_cast<unsigned&>(d_qk[38])), "+r"(reinterpret_cast<unsigned&>(d_qk[39])), "+r"(reinterpret_cast<unsigned&>(d_qk[40])), "+r"(reinterpret_cast<unsigned&>(d_qk[41])), "+r"(reinterpret_cast<unsigned&>(d_qk[42])), "+r"(reinterpret_cast<unsigned&>(d_qk[43])), "+r"(reinterpret_cast<unsigned&>(d_qk[44])), "+r"(reinterpret_cast<unsigned&>(d_qk[45])), "+r"(reinterpret_cast<unsigned&>(d_qk[46])), "+r"(reinterpret_cast<unsigned&>(d_qk[47])), "+r"(reinterpret_cast<unsigned&>(d_qk[48])), "+r"(reinterpret_cast<unsigned&>(d_qk[49])), "+r"(reinterpret_cast<unsigned&>(d_qk[50])), "+r"(reinterpret_cast<unsigned&>(d_qk[51])), "+r"(reinterpret_cast<unsigned&>(d_qk[52])), "+r"(reinterpret_cast<unsigned&>(d_qk[53])), "+r"(reinterpret_cast<unsigned&>(d_qk[54])), "+r"(reinterpret_cast<unsigned&>(d_qk[55])), "+r"(reinterpret_cast<unsigned&>(d_qk[56])), "+r"(reinterpret_cast<unsigned&>(d_qk[57])), "+r"(reinterpret_cast<unsigned&>(d_qk[58])), "+r"(reinterpret_cast<unsigned&>(d_qk[59])), "+r"(reinterpret_cast<unsigned&>(d_qk[60])), "+r"(reinterpret_cast<unsigned&>(d_qk[61])), "+r"(reinterpret_cast<unsigned&>(d_qk[62])), "+r"(reinterpret_cast<unsigned&>(d_qk[63]))
            : "l"(_wgmma_a_0 + 4), "l"(_wgmma_b_0 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1;\n}\n"
            : "+r"(reinterpret_cast<unsigned&>(d_qk[0])), "+r"(reinterpret_cast<unsigned&>(d_qk[1])), "+r"(reinterpret_cast<unsigned&>(d_qk[2])), "+r"(reinterpret_cast<unsigned&>(d_qk[3])), "+r"(reinterpret_cast<unsigned&>(d_qk[4])), "+r"(reinterpret_cast<unsigned&>(d_qk[5])), "+r"(reinterpret_cast<unsigned&>(d_qk[6])), "+r"(reinterpret_cast<unsigned&>(d_qk[7])), "+r"(reinterpret_cast<unsigned&>(d_qk[8])), "+r"(reinterpret_cast<unsigned&>(d_qk[9])), "+r"(reinterpret_cast<unsigned&>(d_qk[10])), "+r"(reinterpret_cast<unsigned&>(d_qk[11])), "+r"(reinterpret_cast<unsigned&>(d_qk[12])), "+r"(reinterpret_cast<unsigned&>(d_qk[13])), "+r"(reinterpret_cast<unsigned&>(d_qk[14])), "+r"(reinterpret_cast<unsigned&>(d_qk[15])), "+r"(reinterpret_cast<unsigned&>(d_qk[16])), "+r"(reinterpret_cast<unsigned&>(d_qk[17])), "+r"(reinterpret_cast<unsigned&>(d_qk[18])), "+r"(reinterpret_cast<unsigned&>(d_qk[19])), "+r"(reinterpret_cast<unsigned&>(d_qk[20])), "+r"(reinterpret_cast<unsigned&>(d_qk[21])), "+r"(reinterpret_cast<unsigned&>(d_qk[22])), "+r"(reinterpret_cast<unsigned&>(d_qk[23])), "+r"(reinterpret_cast<unsigned&>(d_qk[24])), "+r"(reinterpret_cast<unsigned&>(d_qk[25])), "+r"(reinterpret_cast<unsigned&>(d_qk[26])), "+r"(reinterpret_cast<unsigned&>(d_qk[27])), "+r"(reinterpret_cast<unsigned&>(d_qk[28])), "+r"(reinterpret_cast<unsigned&>(d_qk[29])), "+r"(reinterpret_cast<unsigned&>(d_qk[30])), "+r"(reinterpret_cast<unsigned&>(d_qk[31])), "+r"(reinterpret_cast<unsigned&>(d_qk[32])), "+r"(reinterpret_cast<unsigned&>(d_qk[33])), "+r"(reinterpret_cast<unsigned&>(d_qk[34])), "+r"(reinterpret_cast<unsigned&>(d_qk[35])), "+r"(reinterpret_cast<unsigned&>(d_qk[36])), "+r"(reinterpret_cast<unsigned&>(d_qk[37])), "+r"(reinterpret_cast<unsigned&>(d_qk[38])), "+r"(reinterpret_cast<unsigned&>(d_qk[39])), "+r"(reinterpret_cast<unsigned&>(d_qk[40])), "+r"(reinterpret_cast<unsigned&>(d_qk[41])), "+r"(reinterpret_cast<unsigned&>(d_qk[42])), "+r"(reinterpret_cast<unsigned&>(d_qk[43])), "+r"(reinterpret_cast<unsigned&>(d_qk[44])), "+r"(reinterpret_cast<unsigned&>(d_qk[45])), "+r"(reinterpret_cast<unsigned&>(d_qk[46])), "+r"(reinterpret_cast<unsigned&>(d_qk[47])), "+r"(reinterpret_cast<unsigned&>(d_qk[48])), "+r"(reinterpret_cast<unsigned&>(d_qk[49])), "+r"(reinterpret_cast<unsigned&>(d_qk[50])), "+r"(reinterpret_cast<unsigned&>(d_qk[51])), "+r"(reinterpret_cast<unsigned&>(d_qk[52])), "+r"(reinterpret_cast<unsigned&>(d_qk[53])), "+r"(reinterpret_cast<unsigned&>(d_qk[54])), "+r"(reinterpret_cast<unsigned&>(d_qk[55])), "+r"(reinterpret_cast<unsigned&>(d_qk[56])), "+r"(reinterpret_cast<unsigned&>(d_qk[57])), "+r"(reinterpret_cast<unsigned&>(d_qk[58])), "+r"(reinterpret_cast<unsigned&>(d_qk[59])), "+r"(reinterpret_cast<unsigned&>(d_qk[60])), "+r"(reinterpret_cast<unsigned&>(d_qk[61])), "+r"(reinterpret_cast<unsigned&>(d_qk[62])), "+r"(reinterpret_cast<unsigned&>(d_qk[63]))
            : "l"(_wgmma_a_0 + 6), "l"(_wgmma_b_0 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        if (num_blocks > kt + 2) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + (stage) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(k_smem_addr + (unsigned int)(stage * 16384)), "l"((&K8)), "r"(0), "r"((kt + 2) * 128), "r"(bh),
                           "r"(k_full_addr + (stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        d_qk[0] = __uint_as_float(__as_u32(d_qk[0]) + 1262485504);
        d_qk[1] = __uint_as_float(__as_u32(d_qk[1]) + 1262485504);
        d_qk[2] = __uint_as_float(__as_u32(d_qk[2]) + 1262485504);
        d_qk[3] = __uint_as_float(__as_u32(d_qk[3]) + 1262485504);
        d_qk[4] = __uint_as_float(__as_u32(d_qk[4]) + 1262485504);
        d_qk[5] = __uint_as_float(__as_u32(d_qk[5]) + 1262485504);
        d_qk[6] = __uint_as_float(__as_u32(d_qk[6]) + 1262485504);
        d_qk[7] = __uint_as_float(__as_u32(d_qk[7]) + 1262485504);
        d_qk[8] = __uint_as_float(__as_u32(d_qk[8]) + 1262485504);
        d_qk[9] = __uint_as_float(__as_u32(d_qk[9]) + 1262485504);
        d_qk[10] = __uint_as_float(__as_u32(d_qk[10]) + 1262485504);
        d_qk[11] = __uint_as_float(__as_u32(d_qk[11]) + 1262485504);
        d_qk[12] = __uint_as_float(__as_u32(d_qk[12]) + 1262485504);
        d_qk[13] = __uint_as_float(__as_u32(d_qk[13]) + 1262485504);
        d_qk[14] = __uint_as_float(__as_u32(d_qk[14]) + 1262485504);
        d_qk[15] = __uint_as_float(__as_u32(d_qk[15]) + 1262485504);
        d_qk[16] = __uint_as_float(__as_u32(d_qk[16]) + 1262485504);
        d_qk[17] = __uint_as_float(__as_u32(d_qk[17]) + 1262485504);
        d_qk[18] = __uint_as_float(__as_u32(d_qk[18]) + 1262485504);
        d_qk[19] = __uint_as_float(__as_u32(d_qk[19]) + 1262485504);
        d_qk[20] = __uint_as_float(__as_u32(d_qk[20]) + 1262485504);
        d_qk[21] = __uint_as_float(__as_u32(d_qk[21]) + 1262485504);
        d_qk[22] = __uint_as_float(__as_u32(d_qk[22]) + 1262485504);
        d_qk[23] = __uint_as_float(__as_u32(d_qk[23]) + 1262485504);
        d_qk[24] = __uint_as_float(__as_u32(d_qk[24]) + 1262485504);
        d_qk[25] = __uint_as_float(__as_u32(d_qk[25]) + 1262485504);
        d_qk[26] = __uint_as_float(__as_u32(d_qk[26]) + 1262485504);
        d_qk[27] = __uint_as_float(__as_u32(d_qk[27]) + 1262485504);
        d_qk[28] = __uint_as_float(__as_u32(d_qk[28]) + 1262485504);
        d_qk[29] = __uint_as_float(__as_u32(d_qk[29]) + 1262485504);
        d_qk[30] = __uint_as_float(__as_u32(d_qk[30]) + 1262485504);
        d_qk[31] = __uint_as_float(__as_u32(d_qk[31]) + 1262485504);
        d_qk[32] = __uint_as_float(__as_u32(d_qk[32]) + 1262485504);
        d_qk[33] = __uint_as_float(__as_u32(d_qk[33]) + 1262485504);
        d_qk[34] = __uint_as_float(__as_u32(d_qk[34]) + 1262485504);
        d_qk[35] = __uint_as_float(__as_u32(d_qk[35]) + 1262485504);
        d_qk[36] = __uint_as_float(__as_u32(d_qk[36]) + 1262485504);
        d_qk[37] = __uint_as_float(__as_u32(d_qk[37]) + 1262485504);
        d_qk[38] = __uint_as_float(__as_u32(d_qk[38]) + 1262485504);
        d_qk[39] = __uint_as_float(__as_u32(d_qk[39]) + 1262485504);
        d_qk[40] = __uint_as_float(__as_u32(d_qk[40]) + 1262485504);
        d_qk[41] = __uint_as_float(__as_u32(d_qk[41]) + 1262485504);
        d_qk[42] = __uint_as_float(__as_u32(d_qk[42]) + 1262485504);
        d_qk[43] = __uint_as_float(__as_u32(d_qk[43]) + 1262485504);
        d_qk[44] = __uint_as_float(__as_u32(d_qk[44]) + 1262485504);
        d_qk[45] = __uint_as_float(__as_u32(d_qk[45]) + 1262485504);
        d_qk[46] = __uint_as_float(__as_u32(d_qk[46]) + 1262485504);
        d_qk[47] = __uint_as_float(__as_u32(d_qk[47]) + 1262485504);
        d_qk[48] = __uint_as_float(__as_u32(d_qk[48]) + 1262485504);
        d_qk[49] = __uint_as_float(__as_u32(d_qk[49]) + 1262485504);
        d_qk[50] = __uint_as_float(__as_u32(d_qk[50]) + 1262485504);
        d_qk[51] = __uint_as_float(__as_u32(d_qk[51]) + 1262485504);
        d_qk[52] = __uint_as_float(__as_u32(d_qk[52]) + 1262485504);
        d_qk[53] = __uint_as_float(__as_u32(d_qk[53]) + 1262485504);
        d_qk[54] = __uint_as_float(__as_u32(d_qk[54]) + 1262485504);
        d_qk[55] = __uint_as_float(__as_u32(d_qk[55]) + 1262485504);
        d_qk[56] = __uint_as_float(__as_u32(d_qk[56]) + 1262485504);
        d_qk[57] = __uint_as_float(__as_u32(d_qk[57]) + 1262485504);
        d_qk[58] = __uint_as_float(__as_u32(d_qk[58]) + 1262485504);
        d_qk[59] = __uint_as_float(__as_u32(d_qk[59]) + 1262485504);
        d_qk[60] = __uint_as_float(__as_u32(d_qk[60]) + 1262485504);
        d_qk[61] = __uint_as_float(__as_u32(d_qk[61]) + 1262485504);
        d_qk[62] = __uint_as_float(__as_u32(d_qk[62]) + 1262485504);
        d_qk[63] = __uint_as_float(__as_u32(d_qk[63]) + 1262485504);
        if (kt == num_blocks - 1) {
            int key_base = kt * 128 + 2 * lane_q;
            int key = key_base;
            d_qk[0] = ((key >= seqlen) ? -CUDART_INF_F : d_qk[0]);
            int key_0 = key_base + 1;
            d_qk[1] = ((key_0 >= seqlen) ? -CUDART_INF_F : d_qk[1]);
            int key_1 = key_base;
            d_qk[2] = ((key_1 >= seqlen) ? -CUDART_INF_F : d_qk[2]);
            int key_2 = key_base + 1;
            d_qk[3] = ((key_2 >= seqlen) ? -CUDART_INF_F : d_qk[3]);
            int key_3 = key_base + 8;
            d_qk[4] = ((key_3 >= seqlen) ? -CUDART_INF_F : d_qk[4]);
            int key_4 = key_base + 8 + 1;
            d_qk[5] = ((key_4 >= seqlen) ? -CUDART_INF_F : d_qk[5]);
            int key_5 = key_base + 8;
            d_qk[6] = ((key_5 >= seqlen) ? -CUDART_INF_F : d_qk[6]);
            int key_6 = key_base + 8 + 1;
            d_qk[7] = ((key_6 >= seqlen) ? -CUDART_INF_F : d_qk[7]);
            int key_7 = key_base + 16;
            d_qk[8] = ((key_7 >= seqlen) ? -CUDART_INF_F : d_qk[8]);
            int key_8 = key_base + 16 + 1;
            d_qk[9] = ((key_8 >= seqlen) ? -CUDART_INF_F : d_qk[9]);
            int key_9 = key_base + 16;
            d_qk[10] = ((key_9 >= seqlen) ? -CUDART_INF_F : d_qk[10]);
            int key_10 = key_base + 16 + 1;
            d_qk[11] = ((key_10 >= seqlen) ? -CUDART_INF_F : d_qk[11]);
            int key_11 = key_base + 24;
            d_qk[12] = ((key_11 >= seqlen) ? -CUDART_INF_F : d_qk[12]);
            int key_12 = key_base + 24 + 1;
            d_qk[13] = ((key_12 >= seqlen) ? -CUDART_INF_F : d_qk[13]);
            int key_13 = key_base + 24;
            d_qk[14] = ((key_13 >= seqlen) ? -CUDART_INF_F : d_qk[14]);
            int key_14 = key_base + 24 + 1;
            d_qk[15] = ((key_14 >= seqlen) ? -CUDART_INF_F : d_qk[15]);
            int key_15 = key_base + 32;
            d_qk[16] = ((key_15 >= seqlen) ? -CUDART_INF_F : d_qk[16]);
            int key_16 = key_base + 32 + 1;
            d_qk[17] = ((key_16 >= seqlen) ? -CUDART_INF_F : d_qk[17]);
            int key_17 = key_base + 32;
            d_qk[18] = ((key_17 >= seqlen) ? -CUDART_INF_F : d_qk[18]);
            int key_18 = key_base + 32 + 1;
            d_qk[19] = ((key_18 >= seqlen) ? -CUDART_INF_F : d_qk[19]);
            int key_19 = key_base + 40;
            d_qk[20] = ((key_19 >= seqlen) ? -CUDART_INF_F : d_qk[20]);
            int key_20 = key_base + 40 + 1;
            d_qk[21] = ((key_20 >= seqlen) ? -CUDART_INF_F : d_qk[21]);
            int key_21 = key_base + 40;
            d_qk[22] = ((key_21 >= seqlen) ? -CUDART_INF_F : d_qk[22]);
            int key_22 = key_base + 40 + 1;
            d_qk[23] = ((key_22 >= seqlen) ? -CUDART_INF_F : d_qk[23]);
            int key_23 = key_base + 48;
            d_qk[24] = ((key_23 >= seqlen) ? -CUDART_INF_F : d_qk[24]);
            int key_24 = key_base + 48 + 1;
            d_qk[25] = ((key_24 >= seqlen) ? -CUDART_INF_F : d_qk[25]);
            int key_25 = key_base + 48;
            d_qk[26] = ((key_25 >= seqlen) ? -CUDART_INF_F : d_qk[26]);
            int key_26 = key_base + 48 + 1;
            d_qk[27] = ((key_26 >= seqlen) ? -CUDART_INF_F : d_qk[27]);
            int key_27 = key_base + 56;
            d_qk[28] = ((key_27 >= seqlen) ? -CUDART_INF_F : d_qk[28]);
            int key_28 = key_base + 56 + 1;
            d_qk[29] = ((key_28 >= seqlen) ? -CUDART_INF_F : d_qk[29]);
            int key_29 = key_base + 56;
            d_qk[30] = ((key_29 >= seqlen) ? -CUDART_INF_F : d_qk[30]);
            int key_30 = key_base + 56 + 1;
            d_qk[31] = ((key_30 >= seqlen) ? -CUDART_INF_F : d_qk[31]);
            int key_31 = key_base + 64;
            d_qk[32] = ((key_31 >= seqlen) ? -CUDART_INF_F : d_qk[32]);
            int key_32 = key_base + 64 + 1;
            d_qk[33] = ((key_32 >= seqlen) ? -CUDART_INF_F : d_qk[33]);
            int key_33 = key_base + 64;
            d_qk[34] = ((key_33 >= seqlen) ? -CUDART_INF_F : d_qk[34]);
            int key_34 = key_base + 64 + 1;
            d_qk[35] = ((key_34 >= seqlen) ? -CUDART_INF_F : d_qk[35]);
            int key_35 = key_base + 72;
            d_qk[36] = ((key_35 >= seqlen) ? -CUDART_INF_F : d_qk[36]);
            int key_36 = key_base + 72 + 1;
            d_qk[37] = ((key_36 >= seqlen) ? -CUDART_INF_F : d_qk[37]);
            int key_37 = key_base + 72;
            d_qk[38] = ((key_37 >= seqlen) ? -CUDART_INF_F : d_qk[38]);
            int key_38 = key_base + 72 + 1;
            d_qk[39] = ((key_38 >= seqlen) ? -CUDART_INF_F : d_qk[39]);
            int key_39 = key_base + 80;
            d_qk[40] = ((key_39 >= seqlen) ? -CUDART_INF_F : d_qk[40]);
            int key_40 = key_base + 80 + 1;
            d_qk[41] = ((key_40 >= seqlen) ? -CUDART_INF_F : d_qk[41]);
            int key_41 = key_base + 80;
            d_qk[42] = ((key_41 >= seqlen) ? -CUDART_INF_F : d_qk[42]);
            int key_42 = key_base + 80 + 1;
            d_qk[43] = ((key_42 >= seqlen) ? -CUDART_INF_F : d_qk[43]);
            int key_43 = key_base + 88;
            d_qk[44] = ((key_43 >= seqlen) ? -CUDART_INF_F : d_qk[44]);
            int key_44 = key_base + 88 + 1;
            d_qk[45] = ((key_44 >= seqlen) ? -CUDART_INF_F : d_qk[45]);
            int key_45 = key_base + 88;
            d_qk[46] = ((key_45 >= seqlen) ? -CUDART_INF_F : d_qk[46]);
            int key_46 = key_base + 88 + 1;
            d_qk[47] = ((key_46 >= seqlen) ? -CUDART_INF_F : d_qk[47]);
            int key_47 = key_base + 96;
            d_qk[48] = ((key_47 >= seqlen) ? -CUDART_INF_F : d_qk[48]);
            int key_48 = key_base + 96 + 1;
            d_qk[49] = ((key_48 >= seqlen) ? -CUDART_INF_F : d_qk[49]);
            int key_49 = key_base + 96;
            d_qk[50] = ((key_49 >= seqlen) ? -CUDART_INF_F : d_qk[50]);
            int key_50 = key_base + 96 + 1;
            d_qk[51] = ((key_50 >= seqlen) ? -CUDART_INF_F : d_qk[51]);
            int key_51 = key_base + 104;
            d_qk[52] = ((key_51 >= seqlen) ? -CUDART_INF_F : d_qk[52]);
            int key_52 = key_base + 104 + 1;
            d_qk[53] = ((key_52 >= seqlen) ? -CUDART_INF_F : d_qk[53]);
            int key_53 = key_base + 104;
            d_qk[54] = ((key_53 >= seqlen) ? -CUDART_INF_F : d_qk[54]);
            int key_54 = key_base + 104 + 1;
            d_qk[55] = ((key_54 >= seqlen) ? -CUDART_INF_F : d_qk[55]);
            int key_55 = key_base + 112;
            d_qk[56] = ((key_55 >= seqlen) ? -CUDART_INF_F : d_qk[56]);
            int key_56 = key_base + 112 + 1;
            d_qk[57] = ((key_56 >= seqlen) ? -CUDART_INF_F : d_qk[57]);
            int key_57 = key_base + 112;
            d_qk[58] = ((key_57 >= seqlen) ? -CUDART_INF_F : d_qk[58]);
            int key_58 = key_base + 112 + 1;
            d_qk[59] = ((key_58 >= seqlen) ? -CUDART_INF_F : d_qk[59]);
            int key_59 = key_base + 120;
            d_qk[60] = ((key_59 >= seqlen) ? -CUDART_INF_F : d_qk[60]);
            int key_60 = key_base + 120 + 1;
            d_qk[61] = ((key_60 >= seqlen) ? -CUDART_INF_F : d_qk[61]);
            int key_61 = key_base + 120;
            d_qk[62] = ((key_61 >= seqlen) ? -CUDART_INF_F : d_qk[62]);
            int key_62 = key_base + 120 + 1;
            d_qk[63] = ((key_62 >= seqlen) ? -CUDART_INF_F : d_qk[63]);
        }
        float mx00 = -CUDART_INF_F;
        float mx01 = -CUDART_INF_F;
        float mx10 = -CUDART_INF_F;
        float mx11 = -CUDART_INF_F;
        mx00 = fmaxf(mx00, fmaxf(d_qk[0], d_qk[1]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[2], d_qk[3]));
        mx00 = fmaxf(mx00, fmaxf(d_qk[4], d_qk[5]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[6], d_qk[7]));
        mx00 = fmaxf(mx00, fmaxf(d_qk[8], d_qk[9]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[10], d_qk[11]));
        mx00 = fmaxf(mx00, fmaxf(d_qk[12], d_qk[13]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[14], d_qk[15]));
        mx00 = fmaxf(mx00, fmaxf(d_qk[16], d_qk[17]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[18], d_qk[19]));
        mx00 = fmaxf(mx00, fmaxf(d_qk[20], d_qk[21]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[22], d_qk[23]));
        mx00 = fmaxf(mx00, fmaxf(d_qk[24], d_qk[25]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[26], d_qk[27]));
        mx00 = fmaxf(mx00, fmaxf(d_qk[28], d_qk[29]));
        mx10 = fmaxf(mx10, fmaxf(d_qk[30], d_qk[31]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[32], d_qk[33]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[34], d_qk[35]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[36], d_qk[37]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[38], d_qk[39]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[40], d_qk[41]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[42], d_qk[43]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[44], d_qk[45]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[46], d_qk[47]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[48], d_qk[49]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[50], d_qk[51]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[52], d_qk[53]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[54], d_qk[55]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[56], d_qk[57]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[58], d_qk[59]));
        mx01 = fmaxf(mx01, fmaxf(d_qk[60], d_qk[61]));
        mx11 = fmaxf(mx11, fmaxf(d_qk[62], d_qk[63]));
        float mxn00 = mx00 - 1.2582912e+07f;
        float mxn01 = mx01 - 1.2582912e+07f;
        float mxn10 = mx10 - 1.2582912e+07f;
        float mxn11 = mx11 - 1.2582912e+07f;
        float mx0 = fmaxf(mxn00 * ck0, mxn01 * ck1);
        float mx1 = fmaxf(mxn10 * ck0, mxn11 * ck1);
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, mx0, 1);
        mx0 = fmaxf(mx0, _shfl_xor_sync_2);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, mx0, 2);
        mx0 = fmaxf(mx0, _shfl_xor_sync_3);
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, mx1, 1);
        mx1 = fmaxf(mx1, _shfl_xor_sync_4);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, mx1, 2);
        mx1 = fmaxf(mx1, _shfl_xor_sync_5);
        float m_new0 = fmaxf(row_max0, mx0);
        float m_new1 = fmaxf(row_max1, mx1);
        float alpha0 = approx_exp2(row_max0 - m_new0);
        float alpha1 = approx_exp2(row_max1 - m_new1);
        float bias0 = 8.0f - m_new0;
        float bias1 = 8.0f - m_new1;
        float bias00;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(bias00) : "f"(ck0), "f"(-1.2582912e+07f), "f"(bias0));
        float bias01;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(bias01) : "f"(ck1), "f"(-1.2582912e+07f), "f"(bias0));
        float bias10;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(bias10) : "f"(ck0), "f"(-1.2582912e+07f), "f"(bias1));
        float bias11;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(bias11) : "f"(ck1), "f"(-1.2582912e+07f), "f"(bias1));
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float _fma_7;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(d_qk[0]), "f"(ck0), "f"(bias00));
        float _exp2_6 = approx_exp2(_fma_7);
        d_qk[0] = _exp2_6;
        float _fma_9;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(d_qk[1]), "f"(ck0), "f"(bias00));
        float _exp2_8 = approx_exp2(_fma_9);
        d_qk[1] = _exp2_8;
        float _fma_11;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(d_qk[2]), "f"(ck0), "f"(bias10));
        float _exp2_10 = approx_exp2(_fma_11);
        d_qk[2] = _exp2_10;
        float _fma_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(d_qk[3]), "f"(ck0), "f"(bias10));
        float _exp2_12 = approx_exp2(_fma_13);
        d_qk[3] = _exp2_12;
        float _fma_15;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(d_qk[4]), "f"(ck0), "f"(bias00));
        float _exp2_14 = approx_exp2(_fma_15);
        d_qk[4] = _exp2_14;
        float _fma_17;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(d_qk[5]), "f"(ck0), "f"(bias00));
        float _exp2_16 = approx_exp2(_fma_17);
        d_qk[5] = _exp2_16;
        float _fma_19;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(d_qk[6]), "f"(ck0), "f"(bias10));
        float _exp2_18 = approx_exp2(_fma_19);
        d_qk[6] = _exp2_18;
        float _fma_21;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(d_qk[7]), "f"(ck0), "f"(bias10));
        float _exp2_20 = approx_exp2(_fma_21);
        d_qk[7] = _exp2_20;
        float _fma_23;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(d_qk[8]), "f"(ck0), "f"(bias00));
        float _exp2_22 = approx_exp2(_fma_23);
        d_qk[8] = _exp2_22;
        float _fma_25;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(d_qk[9]), "f"(ck0), "f"(bias00));
        float _exp2_24 = approx_exp2(_fma_25);
        d_qk[9] = _exp2_24;
        float _fma_27;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(d_qk[10]), "f"(ck0), "f"(bias10));
        float _exp2_26 = approx_exp2(_fma_27);
        d_qk[10] = _exp2_26;
        float _fma_29;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(d_qk[11]), "f"(ck0), "f"(bias10));
        float _exp2_28 = approx_exp2(_fma_29);
        d_qk[11] = _exp2_28;
        float _fma_31;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(d_qk[12]), "f"(ck0), "f"(bias00));
        float _exp2_30 = approx_exp2(_fma_31);
        d_qk[12] = _exp2_30;
        float _fma_33;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(d_qk[13]), "f"(ck0), "f"(bias00));
        float _exp2_32 = approx_exp2(_fma_33);
        d_qk[13] = _exp2_32;
        float _fma_35;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(d_qk[14]), "f"(ck0), "f"(bias10));
        float _exp2_34 = approx_exp2(_fma_35);
        d_qk[14] = _exp2_34;
        float _fma_37;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(d_qk[15]), "f"(ck0), "f"(bias10));
        float _exp2_36 = approx_exp2(_fma_37);
        d_qk[15] = _exp2_36;
        float _fma_39;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(d_qk[16]), "f"(ck0), "f"(bias00));
        float _exp2_38 = approx_exp2(_fma_39);
        d_qk[16] = _exp2_38;
        float _fma_41;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(d_qk[17]), "f"(ck0), "f"(bias00));
        float _exp2_40 = approx_exp2(_fma_41);
        d_qk[17] = _exp2_40;
        float _fma_43;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(d_qk[18]), "f"(ck0), "f"(bias10));
        float _exp2_42 = approx_exp2(_fma_43);
        d_qk[18] = _exp2_42;
        float _fma_45;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(d_qk[19]), "f"(ck0), "f"(bias10));
        float _exp2_44 = approx_exp2(_fma_45);
        d_qk[19] = _exp2_44;
        float _fma_47;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(d_qk[20]), "f"(ck0), "f"(bias00));
        float _exp2_46 = approx_exp2(_fma_47);
        d_qk[20] = _exp2_46;
        float _fma_49;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(d_qk[21]), "f"(ck0), "f"(bias00));
        float _exp2_48 = approx_exp2(_fma_49);
        d_qk[21] = _exp2_48;
        float _fma_51;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(d_qk[22]), "f"(ck0), "f"(bias10));
        float _exp2_50 = approx_exp2(_fma_51);
        d_qk[22] = _exp2_50;
        float _fma_53;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(d_qk[23]), "f"(ck0), "f"(bias10));
        float _exp2_52 = approx_exp2(_fma_53);
        d_qk[23] = _exp2_52;
        float _fma_55;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(d_qk[24]), "f"(ck0), "f"(bias00));
        float _exp2_54 = approx_exp2(_fma_55);
        d_qk[24] = _exp2_54;
        float _fma_57;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(d_qk[25]), "f"(ck0), "f"(bias00));
        float _exp2_56 = approx_exp2(_fma_57);
        d_qk[25] = _exp2_56;
        float _fma_59;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(d_qk[26]), "f"(ck0), "f"(bias10));
        float _exp2_58 = approx_exp2(_fma_59);
        d_qk[26] = _exp2_58;
        float _fma_61;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(d_qk[27]), "f"(ck0), "f"(bias10));
        float _exp2_60 = approx_exp2(_fma_61);
        d_qk[27] = _exp2_60;
        float _fma_63;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(d_qk[28]), "f"(ck0), "f"(bias00));
        float _exp2_62 = approx_exp2(_fma_63);
        d_qk[28] = _exp2_62;
        float _fma_65;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(d_qk[29]), "f"(ck0), "f"(bias00));
        float _exp2_64 = approx_exp2(_fma_65);
        d_qk[29] = _exp2_64;
        float _fma_67;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(d_qk[30]), "f"(ck0), "f"(bias10));
        float _exp2_66 = approx_exp2(_fma_67);
        d_qk[30] = _exp2_66;
        float _fma_69;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(d_qk[31]), "f"(ck0), "f"(bias10));
        float _exp2_68 = approx_exp2(_fma_69);
        d_qk[31] = _exp2_68;
        float _fma_71;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(d_qk[32]), "f"(ck1), "f"(bias01));
        float _exp2_70 = approx_exp2(_fma_71);
        d_qk[32] = _exp2_70;
        float _fma_73;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(d_qk[33]), "f"(ck1), "f"(bias01));
        float _exp2_72 = approx_exp2(_fma_73);
        d_qk[33] = _exp2_72;
        float _fma_75;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(d_qk[34]), "f"(ck1), "f"(bias11));
        float _exp2_74 = approx_exp2(_fma_75);
        d_qk[34] = _exp2_74;
        float _fma_77;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(d_qk[35]), "f"(ck1), "f"(bias11));
        float _exp2_76 = approx_exp2(_fma_77);
        d_qk[35] = _exp2_76;
        float _fma_79;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(d_qk[36]), "f"(ck1), "f"(bias01));
        float _exp2_78 = approx_exp2(_fma_79);
        d_qk[36] = _exp2_78;
        float _fma_81;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(d_qk[37]), "f"(ck1), "f"(bias01));
        float _exp2_80 = approx_exp2(_fma_81);
        d_qk[37] = _exp2_80;
        float _fma_83;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(d_qk[38]), "f"(ck1), "f"(bias11));
        float _exp2_82 = approx_exp2(_fma_83);
        d_qk[38] = _exp2_82;
        float _fma_85;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(d_qk[39]), "f"(ck1), "f"(bias11));
        float _exp2_84 = approx_exp2(_fma_85);
        d_qk[39] = _exp2_84;
        float _fma_87;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(d_qk[40]), "f"(ck1), "f"(bias01));
        float _exp2_86 = approx_exp2(_fma_87);
        d_qk[40] = _exp2_86;
        float _fma_89;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(d_qk[41]), "f"(ck1), "f"(bias01));
        float _exp2_88 = approx_exp2(_fma_89);
        d_qk[41] = _exp2_88;
        float _fma_91;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(d_qk[42]), "f"(ck1), "f"(bias11));
        float _exp2_90 = approx_exp2(_fma_91);
        d_qk[42] = _exp2_90;
        float _fma_93;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(d_qk[43]), "f"(ck1), "f"(bias11));
        float _exp2_92 = approx_exp2(_fma_93);
        d_qk[43] = _exp2_92;
        float _fma_95;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(d_qk[44]), "f"(ck1), "f"(bias01));
        float _exp2_94 = approx_exp2(_fma_95);
        d_qk[44] = _exp2_94;
        float _fma_97;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(d_qk[45]), "f"(ck1), "f"(bias01));
        float _exp2_96 = approx_exp2(_fma_97);
        d_qk[45] = _exp2_96;
        float _fma_99;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(d_qk[46]), "f"(ck1), "f"(bias11));
        float _exp2_98 = approx_exp2(_fma_99);
        d_qk[46] = _exp2_98;
        float _fma_101;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(d_qk[47]), "f"(ck1), "f"(bias11));
        float _exp2_100 = approx_exp2(_fma_101);
        d_qk[47] = _exp2_100;
        float _fma_103;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(d_qk[48]), "f"(ck1), "f"(bias01));
        float _exp2_102 = approx_exp2(_fma_103);
        d_qk[48] = _exp2_102;
        float _fma_105;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(d_qk[49]), "f"(ck1), "f"(bias01));
        float _exp2_104 = approx_exp2(_fma_105);
        d_qk[49] = _exp2_104;
        float _fma_107;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(d_qk[50]), "f"(ck1), "f"(bias11));
        float _exp2_106 = approx_exp2(_fma_107);
        d_qk[50] = _exp2_106;
        float _fma_109;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(d_qk[51]), "f"(ck1), "f"(bias11));
        float _exp2_108 = approx_exp2(_fma_109);
        d_qk[51] = _exp2_108;
        float _fma_111;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(d_qk[52]), "f"(ck1), "f"(bias01));
        float _exp2_110 = approx_exp2(_fma_111);
        d_qk[52] = _exp2_110;
        float _fma_113;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(d_qk[53]), "f"(ck1), "f"(bias01));
        float _exp2_112 = approx_exp2(_fma_113);
        d_qk[53] = _exp2_112;
        float _fma_115;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(d_qk[54]), "f"(ck1), "f"(bias11));
        float _exp2_114 = approx_exp2(_fma_115);
        d_qk[54] = _exp2_114;
        float _fma_117;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(d_qk[55]), "f"(ck1), "f"(bias11));
        float _exp2_116 = approx_exp2(_fma_117);
        d_qk[55] = _exp2_116;
        float _fma_119;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(d_qk[56]), "f"(ck1), "f"(bias01));
        float _exp2_118 = approx_exp2(_fma_119);
        d_qk[56] = _exp2_118;
        float _fma_121;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(d_qk[57]), "f"(ck1), "f"(bias01));
        float _exp2_120 = approx_exp2(_fma_121);
        d_qk[57] = _exp2_120;
        float _fma_123;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(d_qk[58]), "f"(ck1), "f"(bias11));
        float _exp2_122 = approx_exp2(_fma_123);
        d_qk[58] = _exp2_122;
        float _fma_125;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(d_qk[59]), "f"(ck1), "f"(bias11));
        float _exp2_124 = approx_exp2(_fma_125);
        d_qk[59] = _exp2_124;
        float _fma_127;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(d_qk[60]), "f"(ck1), "f"(bias01));
        float _exp2_126 = approx_exp2(_fma_127);
        d_qk[60] = _exp2_126;
        float _fma_129;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(d_qk[61]), "f"(ck1), "f"(bias01));
        float _exp2_128 = approx_exp2(_fma_129);
        d_qk[61] = _exp2_128;
        float _fma_131;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(d_qk[62]), "f"(ck1), "f"(bias11));
        float _exp2_130 = approx_exp2(_fma_131);
        d_qk[62] = _exp2_130;
        float _fma_133;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(d_qk[63]), "f"(ck1), "f"(bias11));
        float _exp2_132 = approx_exp2(_fma_133);
        d_qk[63] = _exp2_132;
        sum0 = sum0 + d_qk[0];
        sum0 = sum0 + d_qk[1];
        sum1 = sum1 + d_qk[2];
        sum1 = sum1 + d_qk[3];
        sum0 = sum0 + d_qk[4];
        sum0 = sum0 + d_qk[5];
        sum1 = sum1 + d_qk[6];
        sum1 = sum1 + d_qk[7];
        sum0 = sum0 + d_qk[8];
        sum0 = sum0 + d_qk[9];
        sum1 = sum1 + d_qk[10];
        sum1 = sum1 + d_qk[11];
        sum0 = sum0 + d_qk[12];
        sum0 = sum0 + d_qk[13];
        sum1 = sum1 + d_qk[14];
        sum1 = sum1 + d_qk[15];
        sum0 = sum0 + d_qk[16];
        sum0 = sum0 + d_qk[17];
        sum1 = sum1 + d_qk[18];
        sum1 = sum1 + d_qk[19];
        sum0 = sum0 + d_qk[20];
        sum0 = sum0 + d_qk[21];
        sum1 = sum1 + d_qk[22];
        sum1 = sum1 + d_qk[23];
        sum0 = sum0 + d_qk[24];
        sum0 = sum0 + d_qk[25];
        sum1 = sum1 + d_qk[26];
        sum1 = sum1 + d_qk[27];
        sum0 = sum0 + d_qk[28];
        sum0 = sum0 + d_qk[29];
        sum1 = sum1 + d_qk[30];
        sum1 = sum1 + d_qk[31];
        sum0 = sum0 + d_qk[32];
        sum0 = sum0 + d_qk[33];
        sum1 = sum1 + d_qk[34];
        sum1 = sum1 + d_qk[35];
        sum0 = sum0 + d_qk[36];
        sum0 = sum0 + d_qk[37];
        sum1 = sum1 + d_qk[38];
        sum1 = sum1 + d_qk[39];
        sum0 = sum0 + d_qk[40];
        sum0 = sum0 + d_qk[41];
        sum1 = sum1 + d_qk[42];
        sum1 = sum1 + d_qk[43];
        sum0 = sum0 + d_qk[44];
        sum0 = sum0 + d_qk[45];
        sum1 = sum1 + d_qk[46];
        sum1 = sum1 + d_qk[47];
        sum0 = sum0 + d_qk[48];
        sum0 = sum0 + d_qk[49];
        sum1 = sum1 + d_qk[50];
        sum1 = sum1 + d_qk[51];
        sum0 = sum0 + d_qk[52];
        sum0 = sum0 + d_qk[53];
        sum1 = sum1 + d_qk[54];
        sum1 = sum1 + d_qk[55];
        sum0 = sum0 + d_qk[56];
        sum0 = sum0 + d_qk[57];
        sum1 = sum1 + d_qk[58];
        sum1 = sum1 + d_qk[59];
        sum0 = sum0 + d_qk[60];
        sum0 = sum0 + d_qk[61];
        sum1 = sum1 + d_qk[62];
        sum1 = sum1 + d_qk[63];
        row_sum0 = __fmaf_rn(row_sum0, alpha0, sum0);
        row_sum1 = __fmaf_rn(row_sum1, alpha1, sum1);
        row_max0 = m_new0;
        row_max1 = m_new1;
        uint16_t _cvt_satfinite_e4m3x2_f32_135;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_135) : "f"(d_qk[1]), "f"(d_qk[0]));
        uint16_t _cvt_satfinite_e4m3x2_f32_136;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_136) : "f"(d_qk[5]), "f"(d_qk[4]));
        uint32_t _prmt_134;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_134) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_135), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_136));
        p8[0] = _prmt_134;
        uint16_t _cvt_satfinite_e4m3x2_f32_138;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_138) : "f"(d_qk[3]), "f"(d_qk[2]));
        uint16_t _cvt_satfinite_e4m3x2_f32_139;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_139) : "f"(d_qk[7]), "f"(d_qk[6]));
        uint32_t _prmt_137;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_137) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_138), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_139));
        p8[1] = _prmt_137;
        uint16_t _cvt_satfinite_e4m3x2_f32_141;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_141) : "f"(d_qk[9]), "f"(d_qk[8]));
        uint16_t _cvt_satfinite_e4m3x2_f32_142;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_142) : "f"(d_qk[13]), "f"(d_qk[12]));
        uint32_t _prmt_140;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_140) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_141), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_142));
        p8[2] = _prmt_140;
        uint16_t _cvt_satfinite_e4m3x2_f32_144;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_144) : "f"(d_qk[11]), "f"(d_qk[10]));
        uint16_t _cvt_satfinite_e4m3x2_f32_145;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_145) : "f"(d_qk[15]), "f"(d_qk[14]));
        uint32_t _prmt_143;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_143) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_144), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_145));
        p8[3] = _prmt_143;
        uint16_t _cvt_satfinite_e4m3x2_f32_147;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_147) : "f"(d_qk[17]), "f"(d_qk[16]));
        uint16_t _cvt_satfinite_e4m3x2_f32_148;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_148) : "f"(d_qk[21]), "f"(d_qk[20]));
        uint32_t _prmt_146;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_146) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_147), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_148));
        p8[4] = _prmt_146;
        uint16_t _cvt_satfinite_e4m3x2_f32_150;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_150) : "f"(d_qk[19]), "f"(d_qk[18]));
        uint16_t _cvt_satfinite_e4m3x2_f32_151;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_151) : "f"(d_qk[23]), "f"(d_qk[22]));
        uint32_t _prmt_149;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_149) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_150), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_151));
        p8[5] = _prmt_149;
        uint16_t _cvt_satfinite_e4m3x2_f32_153;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_153) : "f"(d_qk[25]), "f"(d_qk[24]));
        uint16_t _cvt_satfinite_e4m3x2_f32_154;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_154) : "f"(d_qk[29]), "f"(d_qk[28]));
        uint32_t _prmt_152;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_152) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_153), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_154));
        p8[6] = _prmt_152;
        uint16_t _cvt_satfinite_e4m3x2_f32_156;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_156) : "f"(d_qk[27]), "f"(d_qk[26]));
        uint16_t _cvt_satfinite_e4m3x2_f32_157;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_157) : "f"(d_qk[31]), "f"(d_qk[30]));
        uint32_t _prmt_155;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_155) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_156), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_157));
        p8[7] = _prmt_155;
        uint16_t _cvt_satfinite_e4m3x2_f32_159;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_159) : "f"(d_qk[33]), "f"(d_qk[32]));
        uint16_t _cvt_satfinite_e4m3x2_f32_160;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_160) : "f"(d_qk[37]), "f"(d_qk[36]));
        uint32_t _prmt_158;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_158) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_159), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_160));
        p8[8] = _prmt_158;
        uint16_t _cvt_satfinite_e4m3x2_f32_162;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_162) : "f"(d_qk[35]), "f"(d_qk[34]));
        uint16_t _cvt_satfinite_e4m3x2_f32_163;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_163) : "f"(d_qk[39]), "f"(d_qk[38]));
        uint32_t _prmt_161;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_161) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_162), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_163));
        p8[9] = _prmt_161;
        uint16_t _cvt_satfinite_e4m3x2_f32_165;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_165) : "f"(d_qk[41]), "f"(d_qk[40]));
        uint16_t _cvt_satfinite_e4m3x2_f32_166;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_166) : "f"(d_qk[45]), "f"(d_qk[44]));
        uint32_t _prmt_164;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_164) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_165), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_166));
        p8[10] = _prmt_164;
        uint16_t _cvt_satfinite_e4m3x2_f32_168;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_168) : "f"(d_qk[43]), "f"(d_qk[42]));
        uint16_t _cvt_satfinite_e4m3x2_f32_169;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_169) : "f"(d_qk[47]), "f"(d_qk[46]));
        uint32_t _prmt_167;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_167) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_168), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_169));
        p8[11] = _prmt_167;
        uint16_t _cvt_satfinite_e4m3x2_f32_171;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_171) : "f"(d_qk[49]), "f"(d_qk[48]));
        uint16_t _cvt_satfinite_e4m3x2_f32_172;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_172) : "f"(d_qk[53]), "f"(d_qk[52]));
        uint32_t _prmt_170;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_170) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_171), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_172));
        p8[12] = _prmt_170;
        uint16_t _cvt_satfinite_e4m3x2_f32_174;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_174) : "f"(d_qk[51]), "f"(d_qk[50]));
        uint16_t _cvt_satfinite_e4m3x2_f32_175;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_175) : "f"(d_qk[55]), "f"(d_qk[54]));
        uint32_t _prmt_173;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_173) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_174), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_175));
        p8[13] = _prmt_173;
        uint16_t _cvt_satfinite_e4m3x2_f32_177;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_177) : "f"(d_qk[57]), "f"(d_qk[56]));
        uint16_t _cvt_satfinite_e4m3x2_f32_178;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_178) : "f"(d_qk[61]), "f"(d_qk[60]));
        uint32_t _prmt_176;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_176) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_177), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_178));
        p8[14] = _prmt_176;
        uint16_t _cvt_satfinite_e4m3x2_f32_180;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_180) : "f"(d_qk[59]), "f"(d_qk[58]));
        uint16_t _cvt_satfinite_e4m3x2_f32_181;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_181) : "f"(d_qk[63]), "f"(d_qk[62]));
        uint32_t _prmt_179;
        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_179) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_180), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_181));
        p8[15] = _prmt_179;
        mbarrier_wait(v_full_addr + (stage) * 8, full_phase);
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_182 = (((uint64_t)(((vt_smem_addr + (unsigned int)(stage * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_182 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_182);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
            : "r"(p8[0]), "r"(p8[1]), "r"(p8[2]), "r"(p8[3]), "l"(_wgmma_b_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
            : "r"(p8[4]), "r"(p8[(4) + 1]), "r"(p8[(4) + 2]), "r"(p8[(4) + 3]), "l"(_wgmma_b_1 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
            : "r"(p8[8]), "r"(p8[(8) + 1]), "r"(p8[(8) + 2]), "r"(p8[(8) + 3]), "l"(_wgmma_b_1 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
            : "r"(p8[12]), "r"(p8[(12) + 1]), "r"(p8[(12) + 2]), "r"(p8[(12) + 3]), "l"(_wgmma_b_1 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        if (num_blocks > kt + 2) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(v_full_addr + (stage) * 8, 16384);
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(vt_smem_addr + (unsigned int)(stage * 16384)), "l"((&VT8)), "r"((kt + 2) * 128), "r"(0), "r"(bh),
                           "r"(v_full_addr + (stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
        d_o[0] = __fmaf_rn(d_o[0], alpha0, d_qk[0]);
        d_o[1] = __fmaf_rn(d_o[1], alpha0, d_qk[1]);
        d_o[2] = __fmaf_rn(d_o[2], alpha1, d_qk[2]);
        d_o[3] = __fmaf_rn(d_o[3], alpha1, d_qk[3]);
        d_o[4] = __fmaf_rn(d_o[4], alpha0, d_qk[4]);
        d_o[5] = __fmaf_rn(d_o[5], alpha0, d_qk[5]);
        d_o[6] = __fmaf_rn(d_o[6], alpha1, d_qk[6]);
        d_o[7] = __fmaf_rn(d_o[7], alpha1, d_qk[7]);
        d_o[8] = __fmaf_rn(d_o[8], alpha0, d_qk[8]);
        d_o[9] = __fmaf_rn(d_o[9], alpha0, d_qk[9]);
        d_o[10] = __fmaf_rn(d_o[10], alpha1, d_qk[10]);
        d_o[11] = __fmaf_rn(d_o[11], alpha1, d_qk[11]);
        d_o[12] = __fmaf_rn(d_o[12], alpha0, d_qk[12]);
        d_o[13] = __fmaf_rn(d_o[13], alpha0, d_qk[13]);
        d_o[14] = __fmaf_rn(d_o[14], alpha1, d_qk[14]);
        d_o[15] = __fmaf_rn(d_o[15], alpha1, d_qk[15]);
        d_o[16] = __fmaf_rn(d_o[16], alpha0, d_qk[16]);
        d_o[17] = __fmaf_rn(d_o[17], alpha0, d_qk[17]);
        d_o[18] = __fmaf_rn(d_o[18], alpha1, d_qk[18]);
        d_o[19] = __fmaf_rn(d_o[19], alpha1, d_qk[19]);
        d_o[20] = __fmaf_rn(d_o[20], alpha0, d_qk[20]);
        d_o[21] = __fmaf_rn(d_o[21], alpha0, d_qk[21]);
        d_o[22] = __fmaf_rn(d_o[22], alpha1, d_qk[22]);
        d_o[23] = __fmaf_rn(d_o[23], alpha1, d_qk[23]);
        d_o[24] = __fmaf_rn(d_o[24], alpha0, d_qk[24]);
        d_o[25] = __fmaf_rn(d_o[25], alpha0, d_qk[25]);
        d_o[26] = __fmaf_rn(d_o[26], alpha1, d_qk[26]);
        d_o[27] = __fmaf_rn(d_o[27], alpha1, d_qk[27]);
        d_o[28] = __fmaf_rn(d_o[28], alpha0, d_qk[28]);
        d_o[29] = __fmaf_rn(d_o[29], alpha0, d_qk[29]);
        d_o[30] = __fmaf_rn(d_o[30], alpha1, d_qk[30]);
        d_o[31] = __fmaf_rn(d_o[31], alpha1, d_qk[31]);
        d_o[32] = __fmaf_rn(d_o[32], alpha0, d_qk[32]);
        d_o[33] = __fmaf_rn(d_o[33], alpha0, d_qk[33]);
        d_o[34] = __fmaf_rn(d_o[34], alpha1, d_qk[34]);
        d_o[35] = __fmaf_rn(d_o[35], alpha1, d_qk[35]);
        d_o[36] = __fmaf_rn(d_o[36], alpha0, d_qk[36]);
        d_o[37] = __fmaf_rn(d_o[37], alpha0, d_qk[37]);
        d_o[38] = __fmaf_rn(d_o[38], alpha1, d_qk[38]);
        d_o[39] = __fmaf_rn(d_o[39], alpha1, d_qk[39]);
        d_o[40] = __fmaf_rn(d_o[40], alpha0, d_qk[40]);
        d_o[41] = __fmaf_rn(d_o[41], alpha0, d_qk[41]);
        d_o[42] = __fmaf_rn(d_o[42], alpha1, d_qk[42]);
        d_o[43] = __fmaf_rn(d_o[43], alpha1, d_qk[43]);
        d_o[44] = __fmaf_rn(d_o[44], alpha0, d_qk[44]);
        d_o[45] = __fmaf_rn(d_o[45], alpha0, d_qk[45]);
        d_o[46] = __fmaf_rn(d_o[46], alpha1, d_qk[46]);
        d_o[47] = __fmaf_rn(d_o[47], alpha1, d_qk[47]);
        d_o[48] = __fmaf_rn(d_o[48], alpha0, d_qk[48]);
        d_o[49] = __fmaf_rn(d_o[49], alpha0, d_qk[49]);
        d_o[50] = __fmaf_rn(d_o[50], alpha1, d_qk[50]);
        d_o[51] = __fmaf_rn(d_o[51], alpha1, d_qk[51]);
        d_o[52] = __fmaf_rn(d_o[52], alpha0, d_qk[52]);
        d_o[53] = __fmaf_rn(d_o[53], alpha0, d_qk[53]);
        d_o[54] = __fmaf_rn(d_o[54], alpha1, d_qk[54]);
        d_o[55] = __fmaf_rn(d_o[55], alpha1, d_qk[55]);
        d_o[56] = __fmaf_rn(d_o[56], alpha0, d_qk[56]);
        d_o[57] = __fmaf_rn(d_o[57], alpha0, d_qk[57]);
        d_o[58] = __fmaf_rn(d_o[58], alpha1, d_qk[58]);
        d_o[59] = __fmaf_rn(d_o[59], alpha1, d_qk[59]);
        d_o[60] = __fmaf_rn(d_o[60], alpha0, d_qk[60]);
        d_o[61] = __fmaf_rn(d_o[61], alpha0, d_qk[61]);
        d_o[62] = __fmaf_rn(d_o[62], alpha1, d_qk[62]);
        d_o[63] = __fmaf_rn(d_o[63], alpha1, d_qk[63]);
    }
    float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
    row_sum0 = row_sum0 + _shfl_xor_sync_183;
    float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
    row_sum0 = row_sum0 + _shfl_xor_sync_184;
    float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
    row_sum1 = row_sum1 + _shfl_xor_sync_185;
    float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);
    row_sum1 = row_sum1 + _shfl_xor_sync_186;
    float inv0 = approx_rcp(row_sum0);
    float inv1 = approx_rcp(row_sum1);
    int batch = bh / num_heads;
    int head = bh % num_heads;
    int o_row0 = batch * o_stride_b + head * o_stride_h + row0 * o_stride_s + 2 * lane_q;
    int o_row1 = o_row0 + 8 * o_stride_s;
    int vs_base = bh * 128 + 2 * lane_q;
    float _vec_load_0[2];
    {
        float2 _v2_187 = *reinterpret_cast<const float2*>(v_scale + vs_base + 0);
        _vec_load_0[0] = _v2_187.x;
        _vec_load_0[0 + 1] = _v2_187.y;
    }
    o_tmp[0] = d_o[0] * inv0 * _vec_load_0[0];
    o_tmp[1] = d_o[1] * inv0 * _vec_load_0[1];
    o_tmp[2] = d_o[2] * inv1 * _vec_load_0[0];
    o_tmp[3] = d_o[3] * inv1 * _vec_load_0[1];
    if (row0 < seqlen) {
        unsigned int _packed_0[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + o_row0)[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_1[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + o_row1)[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
    }
    float _vec_load_1[2];
    {
        float2 _v2_188 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 8) + 0);
        _vec_load_1[0] = _v2_188.x;
        _vec_load_1[0 + 1] = _v2_188.y;
    }
    o_tmp[0] = d_o[4] * inv0 * _vec_load_1[0];
    o_tmp[1] = d_o[5] * inv0 * _vec_load_1[1];
    o_tmp[2] = d_o[6] * inv1 * _vec_load_1[0];
    o_tmp[3] = d_o[7] * inv1 * _vec_load_1[1];
    if (row0 < seqlen) {
        unsigned int _packed_2[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 8))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_3[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 8))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
    }
    float _vec_load_2[2];
    {
        float2 _v2_189 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 16) + 0);
        _vec_load_2[0] = _v2_189.x;
        _vec_load_2[0 + 1] = _v2_189.y;
    }
    o_tmp[0] = d_o[8] * inv0 * _vec_load_2[0];
    o_tmp[1] = d_o[9] * inv0 * _vec_load_2[1];
    o_tmp[2] = d_o[10] * inv1 * _vec_load_2[0];
    o_tmp[3] = d_o[11] * inv1 * _vec_load_2[1];
    if (row0 < seqlen) {
        unsigned int _packed_4[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_4[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 16))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_5[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_5[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 16))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
    }
    float _vec_load_3[2];
    {
        float2 _v2_190 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 24) + 0);
        _vec_load_3[0] = _v2_190.x;
        _vec_load_3[0 + 1] = _v2_190.y;
    }
    o_tmp[0] = d_o[12] * inv0 * _vec_load_3[0];
    o_tmp[1] = d_o[13] * inv0 * _vec_load_3[1];
    o_tmp[2] = d_o[14] * inv1 * _vec_load_3[0];
    o_tmp[3] = d_o[15] * inv1 * _vec_load_3[1];
    if (row0 < seqlen) {
        unsigned int _packed_6[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_6[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 24))[0] = reinterpret_cast<unsigned*>(_packed_6)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_7[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_7[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 24))[0] = reinterpret_cast<unsigned*>(_packed_7)[0];
    }
    float _vec_load_4[2];
    {
        float2 _v2_191 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 32) + 0);
        _vec_load_4[0] = _v2_191.x;
        _vec_load_4[0 + 1] = _v2_191.y;
    }
    o_tmp[0] = d_o[16] * inv0 * _vec_load_4[0];
    o_tmp[1] = d_o[17] * inv0 * _vec_load_4[1];
    o_tmp[2] = d_o[18] * inv1 * _vec_load_4[0];
    o_tmp[3] = d_o[19] * inv1 * _vec_load_4[1];
    if (row0 < seqlen) {
        unsigned int _packed_8[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_8[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 32))[0] = reinterpret_cast<unsigned*>(_packed_8)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_9[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_9[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 32))[0] = reinterpret_cast<unsigned*>(_packed_9)[0];
    }
    float _vec_load_5[2];
    {
        float2 _v2_192 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 40) + 0);
        _vec_load_5[0] = _v2_192.x;
        _vec_load_5[0 + 1] = _v2_192.y;
    }
    o_tmp[0] = d_o[20] * inv0 * _vec_load_5[0];
    o_tmp[1] = d_o[21] * inv0 * _vec_load_5[1];
    o_tmp[2] = d_o[22] * inv1 * _vec_load_5[0];
    o_tmp[3] = d_o[23] * inv1 * _vec_load_5[1];
    if (row0 < seqlen) {
        unsigned int _packed_10[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_10[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 40))[0] = reinterpret_cast<unsigned*>(_packed_10)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_11[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_11[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 40))[0] = reinterpret_cast<unsigned*>(_packed_11)[0];
    }
    float _vec_load_6[2];
    {
        float2 _v2_193 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 48) + 0);
        _vec_load_6[0] = _v2_193.x;
        _vec_load_6[0 + 1] = _v2_193.y;
    }
    o_tmp[0] = d_o[24] * inv0 * _vec_load_6[0];
    o_tmp[1] = d_o[25] * inv0 * _vec_load_6[1];
    o_tmp[2] = d_o[26] * inv1 * _vec_load_6[0];
    o_tmp[3] = d_o[27] * inv1 * _vec_load_6[1];
    if (row0 < seqlen) {
        unsigned int _packed_12[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_12[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 48))[0] = reinterpret_cast<unsigned*>(_packed_12)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_13[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_13[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 48))[0] = reinterpret_cast<unsigned*>(_packed_13)[0];
    }
    float _vec_load_7[2];
    {
        float2 _v2_194 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 56) + 0);
        _vec_load_7[0] = _v2_194.x;
        _vec_load_7[0 + 1] = _v2_194.y;
    }
    o_tmp[0] = d_o[28] * inv0 * _vec_load_7[0];
    o_tmp[1] = d_o[29] * inv0 * _vec_load_7[1];
    o_tmp[2] = d_o[30] * inv1 * _vec_load_7[0];
    o_tmp[3] = d_o[31] * inv1 * _vec_load_7[1];
    if (row0 < seqlen) {
        unsigned int _packed_14[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_14[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 56))[0] = reinterpret_cast<unsigned*>(_packed_14)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_15[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_15[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 56))[0] = reinterpret_cast<unsigned*>(_packed_15)[0];
    }
    float _vec_load_8[2];
    {
        float2 _v2_195 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 64) + 0);
        _vec_load_8[0] = _v2_195.x;
        _vec_load_8[0 + 1] = _v2_195.y;
    }
    o_tmp[0] = d_o[32] * inv0 * _vec_load_8[0];
    o_tmp[1] = d_o[33] * inv0 * _vec_load_8[1];
    o_tmp[2] = d_o[34] * inv1 * _vec_load_8[0];
    o_tmp[3] = d_o[35] * inv1 * _vec_load_8[1];
    if (row0 < seqlen) {
        unsigned int _packed_16[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_16[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 64))[0] = reinterpret_cast<unsigned*>(_packed_16)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_17[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_17[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 64))[0] = reinterpret_cast<unsigned*>(_packed_17)[0];
    }
    float _vec_load_9[2];
    {
        float2 _v2_196 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 72) + 0);
        _vec_load_9[0] = _v2_196.x;
        _vec_load_9[0 + 1] = _v2_196.y;
    }
    o_tmp[0] = d_o[36] * inv0 * _vec_load_9[0];
    o_tmp[1] = d_o[37] * inv0 * _vec_load_9[1];
    o_tmp[2] = d_o[38] * inv1 * _vec_load_9[0];
    o_tmp[3] = d_o[39] * inv1 * _vec_load_9[1];
    if (row0 < seqlen) {
        unsigned int _packed_18[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_18[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 72))[0] = reinterpret_cast<unsigned*>(_packed_18)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_19[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_19[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 72))[0] = reinterpret_cast<unsigned*>(_packed_19)[0];
    }
    float _vec_load_10[2];
    {
        float2 _v2_197 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 80) + 0);
        _vec_load_10[0] = _v2_197.x;
        _vec_load_10[0 + 1] = _v2_197.y;
    }
    o_tmp[0] = d_o[40] * inv0 * _vec_load_10[0];
    o_tmp[1] = d_o[41] * inv0 * _vec_load_10[1];
    o_tmp[2] = d_o[42] * inv1 * _vec_load_10[0];
    o_tmp[3] = d_o[43] * inv1 * _vec_load_10[1];
    if (row0 < seqlen) {
        unsigned int _packed_20[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_20[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 80))[0] = reinterpret_cast<unsigned*>(_packed_20)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_21[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_21[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 80))[0] = reinterpret_cast<unsigned*>(_packed_21)[0];
    }
    float _vec_load_11[2];
    {
        float2 _v2_198 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 88) + 0);
        _vec_load_11[0] = _v2_198.x;
        _vec_load_11[0 + 1] = _v2_198.y;
    }
    o_tmp[0] = d_o[44] * inv0 * _vec_load_11[0];
    o_tmp[1] = d_o[45] * inv0 * _vec_load_11[1];
    o_tmp[2] = d_o[46] * inv1 * _vec_load_11[0];
    o_tmp[3] = d_o[47] * inv1 * _vec_load_11[1];
    if (row0 < seqlen) {
        unsigned int _packed_22[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_22[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 88))[0] = reinterpret_cast<unsigned*>(_packed_22)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_23[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_23[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 88))[0] = reinterpret_cast<unsigned*>(_packed_23)[0];
    }
    float _vec_load_12[2];
    {
        float2 _v2_199 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 96) + 0);
        _vec_load_12[0] = _v2_199.x;
        _vec_load_12[0 + 1] = _v2_199.y;
    }
    o_tmp[0] = d_o[48] * inv0 * _vec_load_12[0];
    o_tmp[1] = d_o[49] * inv0 * _vec_load_12[1];
    o_tmp[2] = d_o[50] * inv1 * _vec_load_12[0];
    o_tmp[3] = d_o[51] * inv1 * _vec_load_12[1];
    if (row0 < seqlen) {
        unsigned int _packed_24[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_24[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 96))[0] = reinterpret_cast<unsigned*>(_packed_24)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_25[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_25[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 96))[0] = reinterpret_cast<unsigned*>(_packed_25)[0];
    }
    float _vec_load_13[2];
    {
        float2 _v2_200 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 104) + 0);
        _vec_load_13[0] = _v2_200.x;
        _vec_load_13[0 + 1] = _v2_200.y;
    }
    o_tmp[0] = d_o[52] * inv0 * _vec_load_13[0];
    o_tmp[1] = d_o[53] * inv0 * _vec_load_13[1];
    o_tmp[2] = d_o[54] * inv1 * _vec_load_13[0];
    o_tmp[3] = d_o[55] * inv1 * _vec_load_13[1];
    if (row0 < seqlen) {
        unsigned int _packed_26[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_26[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 104))[0] = reinterpret_cast<unsigned*>(_packed_26)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_27[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_27[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 104))[0] = reinterpret_cast<unsigned*>(_packed_27)[0];
    }
    float _vec_load_14[2];
    {
        float2 _v2_201 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 112) + 0);
        _vec_load_14[0] = _v2_201.x;
        _vec_load_14[0 + 1] = _v2_201.y;
    }
    o_tmp[0] = d_o[56] * inv0 * _vec_load_14[0];
    o_tmp[1] = d_o[57] * inv0 * _vec_load_14[1];
    o_tmp[2] = d_o[58] * inv1 * _vec_load_14[0];
    o_tmp[3] = d_o[59] * inv1 * _vec_load_14[1];
    if (row0 < seqlen) {
        unsigned int _packed_28[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_28[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 112))[0] = reinterpret_cast<unsigned*>(_packed_28)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_29[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_29[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 112))[0] = reinterpret_cast<unsigned*>(_packed_29)[0];
    }
    float _vec_load_15[2];
    {
        float2 _v2_202 = *reinterpret_cast<const float2*>(v_scale + (vs_base + 120) + 0);
        _vec_load_15[0] = _v2_202.x;
        _vec_load_15[0 + 1] = _v2_202.y;
    }
    o_tmp[0] = d_o[60] * inv0 * _vec_load_15[0];
    o_tmp[1] = d_o[61] * inv0 * _vec_load_15[1];
    o_tmp[2] = d_o[62] * inv1 * _vec_load_15[0];
    o_tmp[3] = d_o[63] * inv1 * _vec_load_15[1];
    if (row0 < seqlen) {
        unsigned int _packed_30[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 0], o_tmp[_lp*2+1 + 0]));
            _packed_30[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row0 + 120))[0] = reinterpret_cast<unsigned*>(_packed_30)[0];
    }
    if (row1 < seqlen) {
        unsigned int _packed_31[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_tmp[_lp*2 + 2], o_tmp[_lp*2+1 + 2]));
            _packed_31[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (o_row1 + 120))[0] = reinterpret_cast<unsigned*>(_packed_31)[0];
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
