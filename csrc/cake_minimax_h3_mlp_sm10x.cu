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
// clang-format off
// MiniMax-H3 full MLP block (RMSNorm + indexed AdaLN + FC1 + SwiGLU + FC2 + indexed gate + residual)
// for sm_100a and sm_103a (Blackwell, compute capability 10.0 and 10.3; one translation unit compiled
// once per target, MINIMAX_H3_SM_MINOR naming the minor).  Generated device code; three operator
// variants share one translation unit, each a chain of three launches on the current stream:
//
//   a   = BF16(BF16(RMSNorm_fp32(x, x_norm_weight, eps)) * BF16(1 + adaln_scale[idx]) + adaln_shift[idx])
//         adaln_scale / adaln_shift / gate are bf16 [rows, 5376] tables read as base + idx * row_stride
//         (contiguous tables or column chunks of one wider [rows, 6 * 5376] projection); ONE int64
//         index table idx = adaln_index[m] drives the modulation and the output gate; a row whose idx
//         lies outside [0, rows) has a = 0 and gate = 0 (out = residual)
//   BF16 : h = BF16(a @ fc1_weight^T)                       FP32 accumulation
//   MXFP8: a_q, a_sf = mxfp8_quantize(a)                     (E4M3 + UE8M0 per 32, FlashInfer recipe)
//          h = BF16(dequant(a_q, a_sf) @ dequant(w1_q, w1_sf)^T)
//   NVFP4: a_q, a_sf = nvfp4_quantize(a, a_global_scale)     (E2M1 + UE4M3 per 16, FlashInfer recipe)
//          h = BF16(alpha1 * ((a_q * a_sf) @ (w1_q * w1_sf)^T)),  alpha1 = 1 / (a_global_scale * w1_global_scale)
//   y   = BF16(BF16(silu(h[:, :14336])) * h[:, 14336:])      fc1 rows [0, 14336) = gate, [14336, 28672) = up
//   BF16 : o = BF16(y @ fc2_weight^T)
//   MXFP8: y_q, y_sf = mxfp8_quantize(y)                     written by the FC1 epilogue in registers
//          o = BF16(dequant(y_q, y_sf) @ dequant(w2_q, w2_sf)^T)
//   NVFP4: y_q, y_sf = nvfp4_quantize(y, y_global_scale)     written by the FC1 epilogue in registers
//          o = BF16(alpha2 * ((y_q * y_sf) @ (w2_q * w2_sf)^T)),  alpha2 = 1 / (y_global_scale * w2_global_scale)
//   p   = BF16(gate[idx] * o)
//   out = BF16(residual + p)                                  out may alias residual
//
// Kernel 1 of the BF16 and MXFP8 variants (norm + AdaLN [+ quantize]) is a plain 128-thread launch,
// one warp per row; the NVFP4 kernel 1 is a persistent row stream: one CTA per SM takes a contiguous
// run of rows, eight rows in flight, each row read by one cp.async.bulk copy that overlaps the work
// on the rows before it.  Kernel 2 (FC1 + SwiGLU [+ quantize y]) and kernel 3 (FC2 + gate + residual)
// are persistent 2-CTA (cta_group::2) tcgen05 GEMMs with TMEM accumulators, TMA operand loads and a
// cluster-launch-control tile scheduler; both require a cluster launch of (2, 1, 1) and do not
// compile for SM90 or SM120.  Kernel 2 executes griddepcontrol.launch_dependents on entry and
// kernel 3 is launched with programmatic stream serialization: its prologue (barrier / TMEM setup,
// scheduler ring) overlaps the FC1 tail and its load warp executes griddepcontrol.wait before the
// first y / scale fetch.
// The MXFP8 variant has two FC1 GEMMs, chosen by the FC1 weight scale layout: the 256x256 pair tile,
// and the 256x448 pair tile (two 224-column accumulator halves, half 1 trailing half 0 by two K
// stages).  The 256x448 MXFP8 FC1 and the NVFP4 FC1 are launched with programmatic stream
// serialization after the norm (the NVFP4 norm executes griddepcontrol.launch_dependents on entry):
// their prologue overlaps the norm and their load warp executes griddepcontrol.wait before the first
// activation fetch.  The caller computes every launch grid (see CheckGrids).
// Kernel 3 may compute the tiles of its final partial wave as two K halves on two clusters (tail
// split-K; FP32 partial sums and release/acquire counters in a caller-provided per-device
// workspace whose counters start at zero and are re-armed by the kernel).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

static_assert(sizeof(CUtensorMap) == 128, "CUDA tensor-map ABI size mismatch");
static_assert(alignof(CUtensorMap) >= 64, "CUDA tensor-map ABI requires at least 64-byte alignment");


__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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

__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}
__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}

__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
}

__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}

__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}

__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
}

__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
}

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1


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

#include "cake_minimax_h3_norm_adaln_bf16_sm10x.cuh"

#undef MINIMAX_H3_MLP_INF
#undef NUM_MAIN_STAGES

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_WORK_RESPONSE_OFF 230400
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 230528
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 7
#define GROUP_M 32
#define WORK_STAGES 4
#define WORK_CONSUMERS 290
#define NUM_K_ITERS 84
#define N_TILES 112
#define FFN 14336
#define num_cluster_tiles ((m_tiles / CTA_GROUP) * N_TILES)
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_minimax_h3_mlp_fc1_bf16(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ y, int M, int m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)
    #define work_full_addr (mbar_base + 144)
    #define work_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // mma_done: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=290
            mbarrier_init(smem + 176, 290);
            mbarrier_init(smem + 184, 290);
            mbarrier_init(smem + 192, 290);
            mbarrier_init(smem + 200, 290);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int weight_row_base = cta_rank * FFN;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * B_HALF_N;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int off_k = iter_k * BLOCK_K;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), 0, weight_row_base + off_n, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        load_stage += 1;
                        if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 7; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)));
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * B_HALF_N;
                int global_row = off_m_1 + local_row;
                unsigned long long row_out = (unsigned long long)global_row * (unsigned long long)FFN + (unsigned long long)off_n_1;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N;
                #pragma unroll 1
                for (int n_chunk = 0; n_chunk < B_HALF_N / 16; n_chunk++) {
                    int col = n_chunk * 16;
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], lane_addr + col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], lane_addr + B_HALF_N + col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float out_vals[16];
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[j]);
                        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                        float gate_b = _cvt_f32_0;
                        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_1[j]);
                        float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                        float up_b = _cvt_f32_1;
                        float _exp2_0 = approx_exp2((-gate_b) * 1.4426950408889634f);
                        float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                        float sig = _rcp_0;
                        __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(gate_b * sig);
                        float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                        float silu_b = _cvt_f32_2;
                        out_vals[j] = silu_b * up_b;
                    }
                    if (global_row < M) {
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals[0 + 0], out_vals[0 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals[0 + 2], out_vals[0 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals[0 + 4], out_vals[0 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals[0 + 6], out_vals[0 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals[0 + 8], out_vals[0 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals[0 + 10], out_vals[0 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals[0 + 12], out_vals[0 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals[0 + 14], out_vals[0 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(y + (row_out + (unsigned long long)col)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef FFN
#undef GROUP_M
#undef MINIMAX_H3_MLP_INF
#undef MMA_K
#undef NUM_K_ITERS
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef N_TILES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef WORK_CONSUMERS
#undef WORK_STAGES
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef num_cluster_tiles
#undef smem_a_addr
#undef smem_b_addr
#undef tiles_per_group
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_WORK_RESPONSE_OFF 230400
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 230528
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 7
#define WORK_STAGES 4
#define NUM_K_ITERS 224
#define K_HALF 112
#define N_TILES 21
#define HIDDEN 5376
#define EPI_WARPS 4
#define num_cluster_tiles (((m_tiles / CTA_GROUP) * N_TILES) + split_count)
#define tiles_per_group (group_m * N_TILES)

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_minimax_h3_mlp_fc2_bf16(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride, float* __restrict__ partial, unsigned int* __restrict__ flags, int tail_base, int split_count, int group_m)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)
    #define work_full_addr (mbar_base + 144)
    #define work_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // mma_done: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=290
            mbarrier_init(smem + 176, 290);
            mbarrier_init(smem + 184, 290);
            mbarrier_init(smem + 192, 290);
            mbarrier_init(smem + 200, 290);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int this_bid_0 = (int)this_bid;
                    int tail_u = this_bid_0 - tail_base;
                    int in_tail = ((tail_u >= 0) ? 1 : 0);
                    int q = tail_u >> 1;
                    int is_h0 = ((q < split_count) ? 1 : 0);
                    int is_h1 = ((q >= split_count && q < 2 * split_count) ? 1 : 0);
                    int half_tail = ((is_h0 == 1) ? 0 : ((is_h1 == 1) ? 1 : -1));
                    int half = ((in_tail == 1) ? half_tail : -1);
                    int j = ((is_h0 == 1) ? q : q - split_count);
                    int slot = ((half >= 0) ? j : 0);
                    int tile_tail = tail_base + 2 * j + (tail_u & 1);
                    int tile = ((in_tail == 1) ? tile_tail : this_bid_0);
                    int k0 = ((half == 1) ? K_HALF : 0);
                    int k1 = ((half == 0) ? K_HALF : NUM_K_ITERS);
                    int group = tile / tiles_per_group;
                    int first_m = group * group_m;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= group_m) ? group_m : remaining);
                    int local = tile % tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    #pragma unroll 1
                    for (int iter_k = k0; iter_k < k1; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), 0, weight_row, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        load_stage += 1;
                        if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 7; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int this_bid_m = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    int this_bid_1 = (int)this_bid_m;
                    int tail_u_1 = this_bid_1 - tail_base;
                    int in_tail_1 = ((tail_u_1 >= 0) ? 1 : 0);
                    int q_1 = tail_u_1 >> 1;
                    int is_h0_1 = ((q_1 < split_count) ? 1 : 0);
                    int is_h1_1 = ((q_1 >= split_count && q_1 < 2 * split_count) ? 1 : 0);
                    int half_tail_1 = ((is_h0_1 == 1) ? 0 : ((is_h1_1 == 1) ? 1 : -1));
                    int half_1 = ((in_tail_1 == 1) ? half_tail_1 : -1);
                    int j_1 = ((is_h0_1 == 1) ? q_1 : q_1 - split_count);
                    int slot_1 = ((half_1 >= 0) ? j_1 : 0);
                    int tile_tail_1 = tail_base + 2 * j_1 + (tail_u_1 & 1);
                    int tile_1 = ((in_tail_1 == 1) ? tile_tail_1 : this_bid_1);
                    int k0_1 = ((half_1 == 1) ? K_HALF : 0);
                    int k1_1 = ((half_1 == 0) ? K_HALF : NUM_K_ITERS);
                    int group_1 = tile_1 / tiles_per_group;
                    int first_m_1 = group_1 * group_m;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= group_m) ? group_m : remaining_1);
                    int local_1 = tile_1 % tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * BLOCK_M;
                    int off_n_1 = bid_n_1 * BLOCK_N;
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = k0_1; iter_k_1 < k1_1; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == k0_1) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)));
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                    this_bid_m = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int local_row = epi_warp * 32 + lane;
            int epi_cta_rank = (int)cta_rank;
            unsigned int this_bid_2 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                int this_bid_0_1 = (int)this_bid_2;
                int tail_u_2 = this_bid_0_1 - tail_base;
                int in_tail_2 = ((tail_u_2 >= 0) ? 1 : 0);
                int q_2 = tail_u_2 >> 1;
                int is_h0_2 = ((q_2 < split_count) ? 1 : 0);
                int is_h1_2 = ((q_2 >= split_count && q_2 < 2 * split_count) ? 1 : 0);
                int half_tail_2 = ((is_h0_2 == 1) ? 0 : ((is_h1_2 == 1) ? 1 : -1));
                int half_2 = ((in_tail_2 == 1) ? half_tail_2 : -1);
                int j_2 = ((is_h0_2 == 1) ? q_2 : q_2 - split_count);
                int slot_2 = ((half_2 >= 0) ? j_2 : 0);
                int tile_tail_2 = tail_base + 2 * j_2 + (tail_u_2 & 1);
                int tile_2 = ((in_tail_2 == 1) ? tile_tail_2 : this_bid_0_1);
                int k0_2 = ((half_2 == 1) ? K_HALF : 0);
                int k1_2 = ((half_2 == 0) ? K_HALF : NUM_K_ITERS);
                int group_2 = tile_2 / tiles_per_group;
                int first_m_2 = group_2 * group_m;
                int remaining_2 = m_tiles - first_m_2;
                int group_size_2 = ((remaining_2 >= group_m) ? group_m : remaining_2);
                int local_2 = tile_2 % tiles_per_group;
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int off_m_2 = bid_m_2 * BLOCK_M;
                int off_n_2 = bid_n_2 * BLOCK_N;
                int global_row = off_m_2 + local_row;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N;
                int flag_idx = slot_2 * 2 + epi_cta_rank;
                int partial_row = flag_idx * BLOCK_M + local_row;
                unsigned long long partial_base = (unsigned long long)partial_row * (unsigned long long)BLOCK_N;
                int _min_0 = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                int load_row = _min_0;
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_2;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_2;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (half_2 == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(EPI_WARPS)) >= static_cast<unsigned int>(1));
                    }
                }
                float acc_chunk[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(acc_chunk[0]), "=f"(acc_chunk[1]), "=f"(acc_chunk[2]), "=f"(acc_chunk[3]), "=f"(acc_chunk[4]), "=f"(acc_chunk[5]), "=f"(acc_chunk[6]), "=f"(acc_chunk[7]), "=f"(acc_chunk[8]), "=f"(acc_chunk[9]), "=f"(acc_chunk[10]), "=f"(acc_chunk[11]), "=f"(acc_chunk[12]), "=f"(acc_chunk[13]), "=f"(acc_chunk[14]), "=f"(acc_chunk[15]), "=f"(acc_chunk[16]), "=f"(acc_chunk[17]), "=f"(acc_chunk[18]), "=f"(acc_chunk[19]), "=f"(acc_chunk[20]), "=f"(acc_chunk[21]), "=f"(acc_chunk[22]), "=f"(acc_chunk[23]), "=f"(acc_chunk[24]), "=f"(acc_chunk[25]), "=f"(acc_chunk[26]), "=f"(acc_chunk[27]), "=f"(acc_chunk[28]), "=f"(acc_chunk[29]), "=f"(acc_chunk[30]), "=f"(acc_chunk[31]), "=f"(acc_chunk[32]), "=f"(acc_chunk[33]), "=f"(acc_chunk[34]), "=f"(acc_chunk[35]), "=f"(acc_chunk[36]), "=f"(acc_chunk[37]), "=f"(acc_chunk[38]), "=f"(acc_chunk[39]), "=f"(acc_chunk[40]), "=f"(acc_chunk[41]), "=f"(acc_chunk[42]), "=f"(acc_chunk[43]), "=f"(acc_chunk[44]), "=f"(acc_chunk[45]), "=f"(acc_chunk[46]), "=f"(acc_chunk[47]), "=f"(acc_chunk[48]), "=f"(acc_chunk[49]), "=f"(acc_chunk[50]), "=f"(acc_chunk[51]), "=f"(acc_chunk[52]), "=f"(acc_chunk[53]), "=f"(acc_chunk[54]), "=f"(acc_chunk[55]), "=f"(acc_chunk[56]), "=f"(acc_chunk[57]), "=f"(acc_chunk[58]), "=f"(acc_chunk[59]), "=f"(acc_chunk[60]), "=f"(acc_chunk[61]), "=f"(acc_chunk[62]), "=f"(acc_chunk[63])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (half_2 == 0) {
                    #pragma unroll
                    for (int q_3 = 0; q_3 < 16; q_3++) {
                        float part_q[4];
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 4; j_3++) {
                            part_q[j_3] = acc_chunk[q_3 * 4 + j_3];
                        }
                        {
                            float4 _v4 = make_float4(part_q[0 + 0], part_q[0 + 1], part_q[0 + 2], part_q[0 + 3]);
                            *reinterpret_cast<float4*>((partial + (partial_base + (unsigned long long)(q_3 * 4))) + 0) = _v4;
                        }
                    }
                }
                if (half_2 == 1) {
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 16; q_4++) {
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (partial_base + (unsigned long long)(q_4 * 4)) + 0);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 4; j_4++) {
                            acc_chunk[q_4 * 4 + j_4] = acc_chunk[q_4 * 4 + j_4] + _vec_load_0[j_4];
                        }
                    }
                }
                uint32_t acc_chunk_bf16[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_chunk[_lp*2 + 0], acc_chunk[_lp*2+1 + 0]));
                    acc_chunk_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(acc_chunk[0]), "=f"(acc_chunk[1]), "=f"(acc_chunk[2]), "=f"(acc_chunk[3]), "=f"(acc_chunk[4]), "=f"(acc_chunk[5]), "=f"(acc_chunk[6]), "=f"(acc_chunk[7]), "=f"(acc_chunk[8]), "=f"(acc_chunk[9]), "=f"(acc_chunk[10]), "=f"(acc_chunk[11]), "=f"(acc_chunk[12]), "=f"(acc_chunk[13]), "=f"(acc_chunk[14]), "=f"(acc_chunk[15]), "=f"(acc_chunk[16]), "=f"(acc_chunk[17]), "=f"(acc_chunk[18]), "=f"(acc_chunk[19]), "=f"(acc_chunk[20]), "=f"(acc_chunk[21]), "=f"(acc_chunk[22]), "=f"(acc_chunk[23]), "=f"(acc_chunk[24]), "=f"(acc_chunk[25]), "=f"(acc_chunk[26]), "=f"(acc_chunk[27]), "=f"(acc_chunk[28]), "=f"(acc_chunk[29]), "=f"(acc_chunk[30]), "=f"(acc_chunk[31]), "=f"(acc_chunk[32]), "=f"(acc_chunk[33]), "=f"(acc_chunk[34]), "=f"(acc_chunk[35]), "=f"(acc_chunk[36]), "=f"(acc_chunk[37]), "=f"(acc_chunk[38]), "=f"(acc_chunk[39]), "=f"(acc_chunk[40]), "=f"(acc_chunk[41]), "=f"(acc_chunk[42]), "=f"(acc_chunk[43]), "=f"(acc_chunk[44]), "=f"(acc_chunk[45]), "=f"(acc_chunk[46]), "=f"(acc_chunk[47]), "=f"(acc_chunk[48]), "=f"(acc_chunk[49]), "=f"(acc_chunk[50]), "=f"(acc_chunk[51]), "=f"(acc_chunk[52]), "=f"(acc_chunk[53]), "=f"(acc_chunk[54]), "=f"(acc_chunk[55]), "=f"(acc_chunk[56]), "=f"(acc_chunk[57]), "=f"(acc_chunk[58]), "=f"(acc_chunk[59]), "=f"(acc_chunk[60]), "=f"(acc_chunk[61]), "=f"(acc_chunk[62]), "=f"(acc_chunk[63])
                    : "r"(lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (half_2 == 0) {
                    #pragma unroll
                    for (int q_5 = 0; q_5 < 16; q_5++) {
                        float part_q_1[4];
                        #pragma unroll
                        for (int j_5 = 0; j_5 < 4; j_5++) {
                            part_q_1[j_5] = acc_chunk[q_5 * 4 + j_5];
                        }
                        {
                            float4 _v4 = make_float4(part_q_1[0 + 0], part_q_1[0 + 1], part_q_1[0 + 2], part_q_1[0 + 3]);
                            *reinterpret_cast<float4*>((partial + (partial_base + 64 + (unsigned long long)(q_5 * 4))) + 0) = _v4;
                        }
                    }
                }
                if (half_2 == 1) {
                    #pragma unroll
                    for (int q_6 = 0; q_6 < 16; q_6++) {
                        float _vec_load_1[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (partial_base + 64 + (unsigned long long)(q_6 * 4)) + 0);
                            _vec_load_1[0 + 0] = _v4.x;
                            _vec_load_1[0 + 1] = _v4.y;
                            _vec_load_1[0 + 2] = _v4.z;
                            _vec_load_1[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 4; j_6++) {
                            acc_chunk[q_6 * 4 + j_6] = acc_chunk[q_6 * 4 + j_6] + _vec_load_1[j_6];
                        }
                    }
                }
                uint32_t acc_chunk_bf16_1[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_chunk[_lp*2 + 0], acc_chunk[_lp*2+1 + 0]));
                    acc_chunk_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(acc_chunk[0]), "=f"(acc_chunk[1]), "=f"(acc_chunk[2]), "=f"(acc_chunk[3]), "=f"(acc_chunk[4]), "=f"(acc_chunk[5]), "=f"(acc_chunk[6]), "=f"(acc_chunk[7]), "=f"(acc_chunk[8]), "=f"(acc_chunk[9]), "=f"(acc_chunk[10]), "=f"(acc_chunk[11]), "=f"(acc_chunk[12]), "=f"(acc_chunk[13]), "=f"(acc_chunk[14]), "=f"(acc_chunk[15]), "=f"(acc_chunk[16]), "=f"(acc_chunk[17]), "=f"(acc_chunk[18]), "=f"(acc_chunk[19]), "=f"(acc_chunk[20]), "=f"(acc_chunk[21]), "=f"(acc_chunk[22]), "=f"(acc_chunk[23]), "=f"(acc_chunk[24]), "=f"(acc_chunk[25]), "=f"(acc_chunk[26]), "=f"(acc_chunk[27]), "=f"(acc_chunk[28]), "=f"(acc_chunk[29]), "=f"(acc_chunk[30]), "=f"(acc_chunk[31]), "=f"(acc_chunk[32]), "=f"(acc_chunk[33]), "=f"(acc_chunk[34]), "=f"(acc_chunk[35]), "=f"(acc_chunk[36]), "=f"(acc_chunk[37]), "=f"(acc_chunk[38]), "=f"(acc_chunk[39]), "=f"(acc_chunk[40]), "=f"(acc_chunk[41]), "=f"(acc_chunk[42]), "=f"(acc_chunk[43]), "=f"(acc_chunk[44]), "=f"(acc_chunk[45]), "=f"(acc_chunk[46]), "=f"(acc_chunk[47]), "=f"(acc_chunk[48]), "=f"(acc_chunk[49]), "=f"(acc_chunk[50]), "=f"(acc_chunk[51]), "=f"(acc_chunk[52]), "=f"(acc_chunk[53]), "=f"(acc_chunk[54]), "=f"(acc_chunk[55]), "=f"(acc_chunk[56]), "=f"(acc_chunk[57]), "=f"(acc_chunk[58]), "=f"(acc_chunk[59]), "=f"(acc_chunk[60]), "=f"(acc_chunk[61]), "=f"(acc_chunk[62]), "=f"(acc_chunk[63])
                    : "r"(lane_addr + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (half_2 == 0) {
                    #pragma unroll
                    for (int q_7 = 0; q_7 < 16; q_7++) {
                        float part_q_2[4];
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 4; j_7++) {
                            part_q_2[j_7] = acc_chunk[q_7 * 4 + j_7];
                        }
                        {
                            float4 _v4 = make_float4(part_q_2[0 + 0], part_q_2[0 + 1], part_q_2[0 + 2], part_q_2[0 + 3]);
                            *reinterpret_cast<float4*>((partial + (partial_base + 128 + (unsigned long long)(q_7 * 4))) + 0) = _v4;
                        }
                    }
                }
                if (half_2 == 1) {
                    #pragma unroll
                    for (int q_8 = 0; q_8 < 16; q_8++) {
                        float _vec_load_2[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (partial_base + 128 + (unsigned long long)(q_8 * 4)) + 0);
                            _vec_load_2[0 + 0] = _v4.x;
                            _vec_load_2[0 + 1] = _v4.y;
                            _vec_load_2[0 + 2] = _v4.z;
                            _vec_load_2[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int j_8 = 0; j_8 < 4; j_8++) {
                            acc_chunk[q_8 * 4 + j_8] = acc_chunk[q_8 * 4 + j_8] + _vec_load_2[j_8];
                        }
                    }
                }
                uint32_t acc_chunk_bf16_2[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_chunk[_lp*2 + 0], acc_chunk[_lp*2+1 + 0]));
                    acc_chunk_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(acc_chunk[0]), "=f"(acc_chunk[1]), "=f"(acc_chunk[2]), "=f"(acc_chunk[3]), "=f"(acc_chunk[4]), "=f"(acc_chunk[5]), "=f"(acc_chunk[6]), "=f"(acc_chunk[7]), "=f"(acc_chunk[8]), "=f"(acc_chunk[9]), "=f"(acc_chunk[10]), "=f"(acc_chunk[11]), "=f"(acc_chunk[12]), "=f"(acc_chunk[13]), "=f"(acc_chunk[14]), "=f"(acc_chunk[15]), "=f"(acc_chunk[16]), "=f"(acc_chunk[17]), "=f"(acc_chunk[18]), "=f"(acc_chunk[19]), "=f"(acc_chunk[20]), "=f"(acc_chunk[21]), "=f"(acc_chunk[22]), "=f"(acc_chunk[23]), "=f"(acc_chunk[24]), "=f"(acc_chunk[25]), "=f"(acc_chunk[26]), "=f"(acc_chunk[27]), "=f"(acc_chunk[28]), "=f"(acc_chunk[29]), "=f"(acc_chunk[30]), "=f"(acc_chunk[31]), "=f"(acc_chunk[32]), "=f"(acc_chunk[33]), "=f"(acc_chunk[34]), "=f"(acc_chunk[35]), "=f"(acc_chunk[36]), "=f"(acc_chunk[37]), "=f"(acc_chunk[38]), "=f"(acc_chunk[39]), "=f"(acc_chunk[40]), "=f"(acc_chunk[41]), "=f"(acc_chunk[42]), "=f"(acc_chunk[43]), "=f"(acc_chunk[44]), "=f"(acc_chunk[45]), "=f"(acc_chunk[46]), "=f"(acc_chunk[47]), "=f"(acc_chunk[48]), "=f"(acc_chunk[49]), "=f"(acc_chunk[50]), "=f"(acc_chunk[51]), "=f"(acc_chunk[52]), "=f"(acc_chunk[53]), "=f"(acc_chunk[54]), "=f"(acc_chunk[55]), "=f"(acc_chunk[56]), "=f"(acc_chunk[57]), "=f"(acc_chunk[58]), "=f"(acc_chunk[59]), "=f"(acc_chunk[60]), "=f"(acc_chunk[61]), "=f"(acc_chunk[62]), "=f"(acc_chunk[63])
                    : "r"(lane_addr + 192));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                if (half_2 == 0) {
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 16; q_9++) {
                        float part_q_3[4];
                        #pragma unroll
                        for (int j_9 = 0; j_9 < 4; j_9++) {
                            part_q_3[j_9] = acc_chunk[q_9 * 4 + j_9];
                        }
                        {
                            float4 _v4 = make_float4(part_q_3[0 + 0], part_q_3[0 + 1], part_q_3[0 + 2], part_q_3[0 + 3]);
                            *reinterpret_cast<float4*>((partial + (partial_base + 192 + (unsigned long long)(q_9 * 4))) + 0) = _v4;
                        }
                    }
                }
                if (half_2 == 1) {
                    #pragma unroll
                    for (int q_10 = 0; q_10 < 16; q_10++) {
                        float _vec_load_3[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (partial_base + 192 + (unsigned long long)(q_10 * 4)) + 0);
                            _vec_load_3[0 + 0] = _v4.x;
                            _vec_load_3[0 + 1] = _v4.y;
                            _vec_load_3[0 + 2] = _v4.z;
                            _vec_load_3[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int j_10 = 0; j_10 < 4; j_10++) {
                            acc_chunk[q_10 * 4 + j_10] = acc_chunk[q_10 * 4 + j_10] + _vec_load_3[j_10];
                        }
                    }
                }
                uint32_t acc_chunk_bf16_3[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_chunk[_lp*2 + 0], acc_chunk[_lp*2+1 + 0]));
                    acc_chunk_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                if (half_2 == 0) {
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
                if (half_2 != 0) {
                    if (global_row < M) {
                        #pragma unroll
                        for (int n_chunk = 0; n_chunk < 4; n_chunk++) {
                            int col = n_chunk * 16;
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_5[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_6[8];
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_7[8];
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float acc_chunk_bf16_f32[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&acc_chunk_bf16_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16 + n_chunk * 8)[_pair]) << 16);
                                (&acc_chunk_bf16_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16 + n_chunk * 8)[_pair]) & 0xffff0000u);
                            }
                            float out_vals[16];
                            #pragma unroll
                            for (int j_11 = 0; j_11 < 8; j_11++) {
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_vec_load_4[j_11] * gate_mul * acc_chunk_bf16_f32[j_11]);
                                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                float p_lo = _cvt_f32_0;
                                out_vals[j_11] = _vec_load_6[j_11] + p_lo;
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_vec_load_5[j_11] * gate_mul * acc_chunk_bf16_f32[j_11 + 8]);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                float p_hi = _cvt_f32_1;
                                out_vals[j_11 + 8] = _vec_load_7[j_11] + p_hi;
                            }
                            {
                                {
                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals[0 + 0], out_vals[0 + 1]);
                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals[0 + 2], out_vals[0 + 3]);
                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals[0 + 4], out_vals[0 + 5]);
                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals[0 + 6], out_vals[0 + 7]);
                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals[0 + 8], out_vals[0 + 9]);
                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals[0 + 10], out_vals[0 + 11]);
                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals[0 + 12], out_vals[0 + 13]);
                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals[0 + 14], out_vals[0 + 15]);
                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                }
                            }
                        }
                        #pragma unroll
                        for (int n_chunk_1 = 0; n_chunk_1 < 4; n_chunk_1++) {
                            int col_1 = n_chunk_1 * 16 + 64;
                            float _vec_load_8[8];
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_1) + 0);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_9[8];
                            {
                                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_9[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_9[_blk] = _vptr_9[_blk];
                                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_10[8];
                            {
                                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col_1) + 0);
                                uint4 _vld_10[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_10[_blk] = _vptr_10[_blk];
                                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_11[8];
                            {
                                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_11[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_11[_blk] = _vptr_11[_blk];
                                    uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float acc_chunk_bf16_1_f32[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&acc_chunk_bf16_1_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16_1 + n_chunk_1 * 8)[_pair]) << 16);
                                (&acc_chunk_bf16_1_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16_1 + n_chunk_1 * 8)[_pair]) & 0xffff0000u);
                            }
                            float out_vals_1[16];
                            #pragma unroll
                            for (int j_12 = 0; j_12 < 8; j_12++) {
                                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_vec_load_8[j_12] * gate_mul * acc_chunk_bf16_1_f32[j_12]);
                                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                                float p_lo_1 = _cvt_f32_2;
                                out_vals_1[j_12] = _vec_load_10[j_12] + p_lo_1;
                                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_vec_load_9[j_12] * gate_mul * acc_chunk_bf16_1_f32[j_12 + 8]);
                                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                                float p_hi_1 = _cvt_f32_3;
                                out_vals_1[j_12 + 8] = _vec_load_11[j_12] + p_hi_1;
                            }
                            {
                                {
                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals_1[0 + 0], out_vals_1[0 + 1]);
                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals_1[0 + 2], out_vals_1[0 + 3]);
                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals_1[0 + 4], out_vals_1[0 + 5]);
                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals_1[0 + 6], out_vals_1[0 + 7]);
                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals_1[0 + 8], out_vals_1[0 + 9]);
                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals_1[0 + 10], out_vals_1[0 + 11]);
                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals_1[0 + 12], out_vals_1[0 + 13]);
                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals_1[0 + 14], out_vals_1[0 + 15]);
                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                }
                            }
                        }
                        #pragma unroll
                        for (int n_chunk_2 = 0; n_chunk_2 < 4; n_chunk_2++) {
                            int col_2 = n_chunk_2 * 16 + 128;
                            float _vec_load_12[8];
                            {
                                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_2) + 0);
                                uint4 _vld_12[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_12[_blk] = _vptr_12[_blk];
                                    uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                                        (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_13[8];
                            {
                                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_2 + 8) + 0);
                                uint4 _vld_13[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_13[_blk] = _vptr_13[_blk];
                                    uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                                        (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_14[8];
                            {
                                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col_2) + 0);
                                uint4 _vld_14[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_14[_blk] = _vptr_14[_blk];
                                    uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                                        (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_15[8];
                            {
                                const uint4* _vptr_15 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col_2 + 8) + 0);
                                uint4 _vld_15[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_15[_blk] = _vptr_15[_blk];
                                    uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                                        (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float acc_chunk_bf16_2_f32[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&acc_chunk_bf16_2_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16_2 + n_chunk_2 * 8)[_pair]) << 16);
                                (&acc_chunk_bf16_2_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16_2 + n_chunk_2 * 8)[_pair]) & 0xffff0000u);
                            }
                            float out_vals_2[16];
                            #pragma unroll
                            for (int j_13 = 0; j_13 < 8; j_13++) {
                                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(_vec_load_12[j_13] * gate_mul * acc_chunk_bf16_2_f32[j_13]);
                                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                                float p_lo_2 = _cvt_f32_4;
                                out_vals_2[j_13] = _vec_load_14[j_13] + p_lo_2;
                                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(_vec_load_13[j_13] * gate_mul * acc_chunk_bf16_2_f32[j_13 + 8]);
                                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                                float p_hi_2 = _cvt_f32_5;
                                out_vals_2[j_13 + 8] = _vec_load_15[j_13] + p_hi_2;
                            }
                            {
                                {
                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals_2[0 + 0], out_vals_2[0 + 1]);
                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals_2[0 + 2], out_vals_2[0 + 3]);
                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals_2[0 + 4], out_vals_2[0 + 5]);
                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals_2[0 + 6], out_vals_2[0 + 7]);
                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals_2[0 + 8], out_vals_2[0 + 9]);
                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals_2[0 + 10], out_vals_2[0 + 11]);
                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals_2[0 + 12], out_vals_2[0 + 13]);
                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals_2[0 + 14], out_vals_2[0 + 15]);
                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col_2)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                }
                            }
                        }
                        #pragma unroll
                        for (int n_chunk_3 = 0; n_chunk_3 < 4; n_chunk_3++) {
                            int col_3 = n_chunk_3 * 16 + 192;
                            float _vec_load_16[8];
                            {
                                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_3) + 0);
                                uint4 _vld_16[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_16[_blk] = _vptr_16[_blk];
                                    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                        (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_17[8];
                            {
                                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_3 + 8) + 0);
                                uint4 _vld_17[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_17[_blk] = _vptr_17[_blk];
                                    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                                        (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_18[8];
                            {
                                const uint4* _vptr_18 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col_3) + 0);
                                uint4 _vld_18[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_18[_blk] = _vptr_18[_blk];
                                    uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                        (&_vec_load_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_19[8];
                            {
                                const uint4* _vptr_19 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col_3 + 8) + 0);
                                uint4 _vld_19[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_19[_blk] = _vptr_19[_blk];
                                    uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                        (&_vec_load_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float acc_chunk_bf16_3_f32[16];
                            #pragma unroll
                            for (int _pair = 0; _pair < 8; _pair++) {
                                (&acc_chunk_bf16_3_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16_3 + n_chunk_3 * 8)[_pair]) << 16);
                                (&acc_chunk_bf16_3_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((acc_chunk_bf16_3 + n_chunk_3 * 8)[_pair]) & 0xffff0000u);
                            }
                            float out_vals_3[16];
                            #pragma unroll
                            for (int j_14 = 0; j_14 < 8; j_14++) {
                                __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(_vec_load_16[j_14] * gate_mul * acc_chunk_bf16_3_f32[j_14]);
                                float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
                                float p_lo_3 = _cvt_f32_6;
                                out_vals_3[j_14] = _vec_load_18[j_14] + p_lo_3;
                                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_vec_load_17[j_14] * gate_mul * acc_chunk_bf16_3_f32[j_14 + 8]);
                                float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
                                float p_hi_3 = _cvt_f32_7;
                                out_vals_3[j_14 + 8] = _vec_load_19[j_14] + p_hi_3;
                            }
                            {
                                {
                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals_3[0 + 0], out_vals_3[0 + 1]);
                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals_3[0 + 2], out_vals_3[0 + 3]);
                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals_3[0 + 4], out_vals_3[0 + 5]);
                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals_3[0 + 6], out_vals_3[0 + 7]);
                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals_3[0 + 8], out_vals_3[0 + 9]);
                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals_3[0 + 10], out_vals_3[0 + 11]);
                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals_3[0 + 12], out_vals_3[0 + 13]);
                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals_3[0 + 14], out_vals_3[0 + 15]);
                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col_3)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                }
                            }
                        }
                    }
                }
                if (half_2 == 1) {
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))), "r"(static_cast<unsigned int>(4294967295)) : "memory");
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_2 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef EPI_WARPS
#undef HIDDEN
#undef K_HALF
#undef MINIMAX_H3_MLP_INF
#undef MMA_K
#undef NUM_K_ITERS
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef N_TILES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef WORK_STAGES
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef num_cluster_tiles
#undef smem_a_addr
#undef smem_b_addr
#undef tiles_per_group
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1

#include "cake_minimax_h3_norm_adaln_mxfp8_sm10x.cuh"

#undef MINIMAX_H3_MLP_INF
#undef NUM_MAIN_STAGES

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define TMEM_NCOLS 280
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 68608
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 68608
#define SMEM_SMEM_SFA_ALL_OFF 66560
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_ALL_STRIDE 68608
#define SMEM_SMEM_SFB_ALL_OFF 67584
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_ALL_STRIDE 68608
#define SMEM_SMEM_SFA0_OFF 66560
#define SMEM_SMEM_SFA0_STAGE_BYTES 512
#define SMEM_SMEM_SFA0_STRIDE 68608
#define SMEM_SMEM_SFA1_OFF 67072
#define SMEM_SMEM_SFA1_STAGE_BYTES 512
#define SMEM_SMEM_SFA1_STRIDE 68608
#define SMEM_SMEM_SFB0_OFF 67584
#define SMEM_SMEM_SFB0_STAGE_BYTES 1024
#define SMEM_SMEM_SFB0_STRIDE 68608
#define SMEM_SMEM_SFB1_OFF 68608
#define SMEM_SMEM_SFB1_STAGE_BYTES 1024
#define SMEM_SMEM_SFB1_STRIDE 68608
#define SMEM_WORK_RESPONSE_OFF 206848
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 206976
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 3
#define GROUP_M 64
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 21
#define N_TILES 112
#define SF_K_TILES 42
#define FFN 14336
#define Y_SF_K_TILES 112
#define Y_SF_TILE_BYTES 512
#define num_cluster_tiles ((m_tiles / CTA_GROUP) * N_TILES)
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_mlp_fc1_e4m3(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, uint8_t* __restrict__ y_q, uint8_t* __restrict__ y_sf, int M, int m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 24)
    #define mainloop_done_addr (mbar_base + 48)
    #define epilogue_done_addr (mbar_base + 56)
    #define work_full_addr (mbar_base + 64)
    #define work_empty_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA_ALL_OFF);
    const int smem_sfa_all_addr = smem + SMEM_SMEM_SFA_ALL_OFF;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_ALL_OFF);
    const int smem_sfb_all_addr = smem + SMEM_SMEM_SFB_ALL_OFF;
    uint8_t* smem_sfa0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA0_OFF);
    const int smem_sfa0_addr = smem + SMEM_SMEM_SFA0_OFF;
    uint8_t* smem_sfa1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA1_OFF);
    const int smem_sfa1_addr = smem + SMEM_SMEM_SFA1_OFF;
    uint8_t* smem_sfb0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB0_OFF);
    const int smem_sfb0_addr = smem + SMEM_SMEM_SFB0_OFF;
    uint8_t* smem_sfb1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB1_OFF);
    const int smem_sfb1_addr = smem + SMEM_SMEM_SFB1_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 56, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 96, 546);
            mbarrier_init(smem + 104, 546);
            mbarrier_init(smem + 112, 546);
            mbarrier_init(smem + 120, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 280 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 264;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int weight_row_base = cta_rank * FFN;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * B_HALF_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 68608, (&A), 0, off_m, k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 68608, (&B), 0, weight_row_base + off_n, k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 68608, (&SFA), 0, 0, sfa_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 68608, (&SFB), 0, 0, sfb_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(68608)) : "memory");
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 3; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _mma_b_lo_1 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (acc_stage) * 8, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr + (acc_stage_1) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * B_HALF_N;
                int global_row = off_m_1 + local_row;
                int col0 = col_part * 64;
                unsigned long long row_out = (unsigned long long)global_row * (unsigned long long)FFN + (unsigned long long)off_n_1 + (unsigned long long)col0;
                unsigned long long sf_row_base = (unsigned long long)bid_m_1 * (unsigned long long)Y_SF_K_TILES * (unsigned long long)Y_SF_TILE_BYTES + (unsigned long long)(local_row & 31) * 16 + (unsigned long long)(local_row >> 5) * 4;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(lane_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(lane_addr + B_HALF_N));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                    : "r"(lane_addr + B_HALF_N + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                #pragma unroll
                for (int blk = 0; blk < 2; blk++) {
                    float vals[32];
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[blk * 32 + j]);
                        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                        float gate_b = _cvt_f32_0;
                        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_1[blk * 32 + j]);
                        float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                        float up_b = _cvt_f32_1;
                        float _exp2_0 = approx_exp2((-gate_b) * 1.4426950408889634f);
                        float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                        float sig = _rcp_0;
                        __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(gate_b * sig);
                        float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                        float silu_b = _cvt_f32_2;
                        __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(silu_b * up_b);
                        float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                        vals[j] = _cvt_f32_3;
                    }
                    float mags[32];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 32; j_1++) {
                        mags[j_1] = vals[j_1];
                    }
                    float _fabs_0 = fabsf(mags[0]);
                    mags[0] = _fabs_0;
                    float _fabs_1 = fabsf(mags[1]);
                    mags[1] = _fabs_1;
                    float _fabs_2 = fabsf(mags[2]);
                    mags[2] = _fabs_2;
                    float _fabs_3 = fabsf(mags[3]);
                    mags[3] = _fabs_3;
                    float _fabs_4 = fabsf(mags[4]);
                    mags[4] = _fabs_4;
                    float _fabs_5 = fabsf(mags[5]);
                    mags[5] = _fabs_5;
                    float _fabs_6 = fabsf(mags[6]);
                    mags[6] = _fabs_6;
                    float _fabs_7 = fabsf(mags[7]);
                    mags[7] = _fabs_7;
                    float _fabs_8 = fabsf(mags[8]);
                    mags[8] = _fabs_8;
                    float _fabs_9 = fabsf(mags[9]);
                    mags[9] = _fabs_9;
                    float _fabs_10 = fabsf(mags[10]);
                    mags[10] = _fabs_10;
                    float _fabs_11 = fabsf(mags[11]);
                    mags[11] = _fabs_11;
                    float _fabs_12 = fabsf(mags[12]);
                    mags[12] = _fabs_12;
                    float _fabs_13 = fabsf(mags[13]);
                    mags[13] = _fabs_13;
                    float _fabs_14 = fabsf(mags[14]);
                    mags[14] = _fabs_14;
                    float _fabs_15 = fabsf(mags[15]);
                    mags[15] = _fabs_15;
                    float _fabs_16 = fabsf(mags[16]);
                    mags[16] = _fabs_16;
                    float _fabs_17 = fabsf(mags[17]);
                    mags[17] = _fabs_17;
                    float _fabs_18 = fabsf(mags[18]);
                    mags[18] = _fabs_18;
                    float _fabs_19 = fabsf(mags[19]);
                    mags[19] = _fabs_19;
                    float _fabs_20 = fabsf(mags[20]);
                    mags[20] = _fabs_20;
                    float _fabs_21 = fabsf(mags[21]);
                    mags[21] = _fabs_21;
                    float _fabs_22 = fabsf(mags[22]);
                    mags[22] = _fabs_22;
                    float _fabs_23 = fabsf(mags[23]);
                    mags[23] = _fabs_23;
                    float _fabs_24 = fabsf(mags[24]);
                    mags[24] = _fabs_24;
                    float _fabs_25 = fabsf(mags[25]);
                    mags[25] = _fabs_25;
                    float _fabs_26 = fabsf(mags[26]);
                    mags[26] = _fabs_26;
                    float _fabs_27 = fabsf(mags[27]);
                    mags[27] = _fabs_27;
                    float _fabs_28 = fabsf(mags[28]);
                    mags[28] = _fabs_28;
                    float _fabs_29 = fabsf(mags[29]);
                    mags[29] = _fabs_29;
                    float _fabs_30 = fabsf(mags[30]);
                    mags[30] = _fabs_30;
                    float _fabs_31 = fabsf(mags[31]);
                    mags[31] = _fabs_31;
                    float2 _reg_reduce_max2_0 = {-MINIMAX_H3_MLP_INF, -MINIMAX_H3_MLP_INF};
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[0], mags[1]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[2], mags[3]));
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[4], mags[5]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[6], mags[7]));
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[8], mags[9]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[10], mags[11]));
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[12], mags[13]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[14], mags[15]));
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[16], mags[17]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[18], mags[19]));
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[20], mags[21]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[22], mags[23]));
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[24], mags[25]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[26], mags[27]));
                    _reg_reduce_max2_0.x = max_noftz(_reg_reduce_max2_0.x, max_noftz(mags[28], mags[29]));
                    _reg_reduce_max2_0.y = max_noftz(_reg_reduce_max2_0.y, max_noftz(mags[30], mags[31]));
                    float mags_max = row_max_reduce(_reg_reduce_max2_0);
                    float absmax = mags_max;
                    float scale = absmax / 448.0f;
                    unsigned int scale_bits = __as_u32(scale);
                    unsigned int exponent = scale_bits >> 23 & 255;
                    unsigned int mantissa = scale_bits & 8388607;
                    unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
                    unsigned int normal = ((exponent != 0) ? 1 : 0);
                    unsigned int large_subnormal = ((mantissa > 4194304) ? 1 : 0);
                    unsigned int _min_0 = ((exponent + (has_mantissa & (normal | large_subnormal))) < (254) ? (exponent + (has_mantissa & (normal | large_subnormal))) : (254));
                    unsigned int scale_byte = _min_0;
                    unsigned int inverse_nonzero_bits = 254 - scale_byte << 23;
                    unsigned int zero_bits = 0;
                    unsigned int inverse_bits = ((scale_byte == 0) ? zero_bits : inverse_nonzero_bits);
                    float inverse = 0.0f;
                    inverse = reinterpret_cast<float*>(&inverse_bits)[0];
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_1 = {inverse, inverse};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(vals)[_ls], _scale2_1);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++) {
                        vals[_ls] = vals[_ls] * inverse;
                    }
                    #endif
                    if (global_row < M) {
                        {
                            unsigned int _fp8_pk[8];
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[0]) : "f"(vals[0 + 0]), "f"(vals[0 + 1]), "f"(vals[0 + 2]), "f"(vals[0 + 3]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[1]) : "f"(vals[0 + 4]), "f"(vals[0 + 5]), "f"(vals[0 + 6]), "f"(vals[0 + 7]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[2]) : "f"(vals[0 + 8]), "f"(vals[0 + 9]), "f"(vals[0 + 10]), "f"(vals[0 + 11]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[3]) : "f"(vals[0 + 12]), "f"(vals[0 + 13]), "f"(vals[0 + 14]), "f"(vals[0 + 15]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[4]) : "f"(vals[0 + 16]), "f"(vals[0 + 17]), "f"(vals[0 + 18]), "f"(vals[0 + 19]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[5]) : "f"(vals[0 + 20]), "f"(vals[0 + 21]), "f"(vals[0 + 22]), "f"(vals[0 + 23]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[6]) : "f"(vals[0 + 24]), "f"(vals[0 + 25]), "f"(vals[0 + 26]), "f"(vals[0 + 27]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[7]) : "f"(vals[0 + 28]), "f"(vals[0 + 29]), "f"(vals[0 + 30]), "f"(vals[0 + 31]));
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"(reinterpret_cast<unsigned char*>(y_q + (row_out + (unsigned long long)(blk * 32))) + (0)), "r"(_fp8_pk[0]), "r"(_fp8_pk[1]), "r"(_fp8_pk[2]), "r"(_fp8_pk[3]), "r"(_fp8_pk[4]), "r"(_fp8_pk[5]), "r"(_fp8_pk[6]), "r"(_fp8_pk[7]) : "memory");
                        }
                        int block = bid_n_1 * 4 + col_part * 2 + blk;
                        int k_tile = block >> 2;
                        int sf_col = block & 3;
                        *(reinterpret_cast<unsigned char*>(y_sf + (sf_row_base + (unsigned long long)k_tile * (unsigned long long)Y_SF_TILE_BYTES + (unsigned long long)sf_col)) + (0)) = (unsigned char)(scale_byte);
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef FFN
#undef GROUP_M
#undef MINIMAX_H3_MLP_INF
#undef NUM_K_ITERS
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef N_TILES
#undef SF_K_TILES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_SFA0_OFF
#undef SMEM_SMEM_SFA0_STAGE_BYTES
#undef SMEM_SMEM_SFA0_STRIDE
#undef SMEM_SMEM_SFA1_OFF
#undef SMEM_SMEM_SFA1_STAGE_BYTES
#undef SMEM_SMEM_SFA1_STRIDE
#undef SMEM_SMEM_SFA_ALL_OFF
#undef SMEM_SMEM_SFA_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFA_ALL_STRIDE
#undef SMEM_SMEM_SFB0_OFF
#undef SMEM_SMEM_SFB0_STAGE_BYTES
#undef SMEM_SMEM_SFB0_STRIDE
#undef SMEM_SMEM_SFB1_OFF
#undef SMEM_SMEM_SFB1_STAGE_BYTES
#undef SMEM_SMEM_SFB1_STRIDE
#undef SMEM_SMEM_SFB_ALL_OFF
#undef SMEM_SMEM_SFB_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFB_ALL_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef TMEM_TMEM_SFA_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef WORK_CONSUMERS
#undef WORK_STAGES
#undef Y_SF_K_TILES
#undef Y_SF_TILE_BYTES
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef num_cluster_tiles
#undef smem_a_addr
#undef smem_b_addr
#undef smem_sfa0_addr
#undef smem_sfa1_addr
#undef smem_sfa_all_addr
#undef smem_sfb0_addr
#undef smem_sfb1_addr
#undef smem_sfb_all_addr
#undef tiles_per_group
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define TMEM_NCOLS 504
#define TMEM_ACCUM0_OFFSET 0
#define TMEM_ACCUM1_OFFSET 224
#define TMEM_TMEM_SFA_OFFSET 448
#define TMEM_TMEM_SFB_OFFSET 472
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 48128
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 28672
#define SMEM_SMEM_B_STRIDE 48128
#define SMEM_SMEM_V2_OFF 17408
#define SMEM_SMEM_V2_STAGE_BYTES 14336
#define SMEM_SMEM_V2_STRIDE 48128
#define SMEM_SMEM_V3_OFF 31744
#define SMEM_SMEM_V3_STAGE_BYTES 14336
#define SMEM_SMEM_V3_STRIDE 48128
#define SMEM_SMEM_SFA_ALL_OFF 46080
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 512
#define SMEM_SMEM_SFA_ALL_STRIDE 48128
#define SMEM_SMEM_SFA_OFF 46080
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 48128
#define SMEM_SMEM_V6_OFF 46592
#define SMEM_SMEM_V6_STAGE_BYTES 1024
#define SMEM_SMEM_V6_STRIDE 48128
#define SMEM_SMEM_V7_OFF 47616
#define SMEM_SMEM_V7_STAGE_BYTES 1024
#define SMEM_SMEM_V7_STRIDE 48128
#define SMEM_SMEM_V8_OFF 46592
#define SMEM_SMEM_V8_STAGE_BYTES 1024
#define SMEM_SMEM_V8_STRIDE 48128
#define SMEM_SMEM_V9_OFF 47616
#define SMEM_SMEM_V9_STAGE_BYTES 1024
#define SMEM_SMEM_V9_STRIDE 48128
#define SMEM_WORK_RESPONSE_OFF 193536
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_XCHG_OFF 193600
#define SMEM_XCHG_STAGE_BYTES 2048
#define SMEM_XCHG_STRIDE 2048
#define SMEM_TOTAL 195712
#define tiles_per_group 4096

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_mlp_fc1_e4m3_w448(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, uint8_t* __restrict__ y_q, uint8_t* __restrict__ y_sf, int M, int m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 32)
    #define mainloop_done_half_0_addr (mbar_base + 64)
    #define mainloop_done_half_1_addr (mbar_base + 72)
    #define epilogue_done_half_0_addr (mbar_base + 80)
    #define epilogue_done_half_1_addr (mbar_base + 88)
    #define work_full_addr (mbar_base + 96)
    #define work_empty_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_v2 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v2_addr = smem + 17408;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 31744);
    const int smem_v3_addr = smem + 31744;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 46080);
    const int smem_sfa_all_addr = smem + 46080;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 46080);
    const int smem_sfa_addr = smem + 46080;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 46592);
    const int smem_v6_addr = smem + 46592;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 47616);
    const int smem_v7_addr = smem + 47616;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 46592);
    const int smem_v8_addr = smem + 46592;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 47616);
    const int smem_v9_addr = smem + 47616;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 193536);
    const int work_response_addr = smem + 193536;
    float* xchg = reinterpret_cast<float*>(smem_raw + 193600);
    const int xchg_addr = smem + 193600;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done_half_0: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // mainloop_done_half_1: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // epilogue_done_half_0: 1 barriers, init_count=8
            mbarrier_init(smem + 80, 8);
            // epilogue_done_half_1: 1 barriers, init_count=8
            mbarrier_init(smem + 88, 8);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 128, 546);
            mbarrier_init(smem + 136, 546);
            mbarrier_init(smem + 144, 546);
            mbarrier_init(smem + 152, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 504 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum0 = taddr;
    const int tmem_accum1 = taddr + 224;
    const int tmem_tmem_sfa = taddr + 448;
    const int tmem_tmem_sfb = taddr + 472;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int weight_row_base = cta_rank * 14336;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < m_tiles / 2 * 64; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * 64;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= 64) ? 64 : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * 128;
                    int off_n = bid_n * 224;
                    int sfa_tile_row = bid_m * 42;
                    int sfb_tile_row0 = bid_n * 2 * 42;
                    int sfb_tile_row1 = (bid_n * 2 + 1) * 42;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < 42; iter_k++) {
                        mbarrier_wait_hint(mma_done_addr + (load_stage) * 8, _phase_mma_done, 10000000);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 48128, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 48128, (&B), 0, weight_row_base + off_n, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 48128, (&SFA), 0, 0, sfa_tile_row + iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_v6_addr + load_stage * 48128, (&SFB), 0, 0, sfb_tile_row0 + iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_v7_addr + load_stage * 48128, (&SFB), 0, 0, sfb_tile_row1 + iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(47616)) : "memory");
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 10000000);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 4; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int half1_stage = 0;
            unsigned int _phase_epilogue_done_half_0 = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_epilogue_done_half_1 = 1;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < m_tiles / 2 * 64; _tile_iter_1++) {
                    mbarrier_wait_hint(epilogue_done_half_0_addr, _phase_epilogue_done_half_0, 10000000);
                    mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_0 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_0);
                    int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    int _smem_desc_lo_2 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa, tmem_tmem_sfb, 0, _elected_0);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                    }
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_3 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_3);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_3);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_3);
                    int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    int _smem_desc_lo_5 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_3);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_3);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_3);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_3);
                    }
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    mbarrier_wait_hint(epilogue_done_half_1_addr, _phase_epilogue_done_half_1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_6 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_6);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_6);
                    int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    int _smem_desc_lo_8 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 0, _elected_6);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_6);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_6);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_6);
                    }
                    elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                    half1_stage += 1;
                    if (half1_stage == 4) { half1_stage = 0; }
                    mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_9 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_9);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_9);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_9);
                    int _smem_desc_lo_10 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    int _smem_desc_lo_11 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_9);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_9);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_9);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_9);
                    }
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    uint32_t _elected_12 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_12);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_12);
                    int _smem_desc_lo_13 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    int _smem_desc_lo_14 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_13) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_14) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_12);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_12);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_12);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_12);
                    }
                    elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                    half1_stage += 1;
                    if (half1_stage == 4) { half1_stage = 0; }
                    mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_15 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_15);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_15);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_15);
                    int _smem_desc_lo_16 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    int _smem_desc_lo_17 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_17) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_15);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_15);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_15);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_15);
                    }
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    uint32_t _elected_18 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_18);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_18);
                    int _smem_desc_lo_19 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    int _smem_desc_lo_20 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_18);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_18);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_18);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_18);
                    }
                    elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                    half1_stage += 1;
                    if (half1_stage == 4) { half1_stage = 0; }
                    mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_21 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_21);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_21);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_21);
                    int _smem_desc_lo_22 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    int _smem_desc_lo_23 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_22) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_23) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_21);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_21);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_21);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_21);
                    }
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    uint32_t _elected_24 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_24);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_24);
                    int _smem_desc_lo_25 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    int _smem_desc_lo_26 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_26) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_24);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_24);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_24);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_24);
                    }
                    elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                    half1_stage += 1;
                    if (half1_stage == 4) { half1_stage = 0; }
                    mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_27 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_27);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_27);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_27);
                    int _smem_desc_lo_28 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    int _smem_desc_lo_29 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_28) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_29) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_27);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_27);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_27);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_27);
                    }
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    #pragma unroll 1
                    for (int _group = 1; _group < 7; _group++) {
                        uint32_t _elected_30 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_30);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_30);
                        int _smem_desc_lo_31 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        int _smem_desc_lo_32 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_31) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_32) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_30);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_30);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_30);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_30);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                        half1_stage += 1;
                        if (half1_stage == 4) { half1_stage = 0; }
                        mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_33 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_33);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_33);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_33);
                        int _smem_desc_lo_34 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        int _smem_desc_lo_35 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_34) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_35) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_33);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_33);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_33);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_33);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        uint32_t _elected_36 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_36);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_36);
                        int _smem_desc_lo_37 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        int _smem_desc_lo_38 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_37) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_38) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_36);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_36);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_36);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_36);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                        half1_stage += 1;
                        if (half1_stage == 4) { half1_stage = 0; }
                        mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_39 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_39);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_39);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_39);
                        int _smem_desc_lo_40 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        int _smem_desc_lo_41 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_40) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_41) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_39);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_39);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_39);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 16, 1, _elected_39);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        uint32_t _elected_42 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_42);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_42);
                        int _smem_desc_lo_43 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        int _smem_desc_lo_44 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_43) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_44) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_42);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_42);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_42);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa, tmem_tmem_sfb + 8, 1, _elected_42);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                        half1_stage += 1;
                        if (half1_stage == 4) { half1_stage = 0; }
                        mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_45 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_45);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_45);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_45);
                        int _smem_desc_lo_46 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        int _smem_desc_lo_47 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_46) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_47) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_45);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_45);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_45);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 8, tmem_tmem_sfb, 1, _elected_45);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        uint32_t _elected_48 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_48);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_48);
                        int _smem_desc_lo_49 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        int _smem_desc_lo_50 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_49) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_50) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_48);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_48);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_48);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 24, 1, _elected_48);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                        half1_stage += 1;
                        if (half1_stage == 4) { half1_stage = 0; }
                        mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_51 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_51);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_51);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_51);
                        int _smem_desc_lo_52 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        int _smem_desc_lo_53 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_52) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_53) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_51);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_51);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_51);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 16, 1, _elected_51);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        uint32_t _elected_54 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_54);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_54);
                        int _smem_desc_lo_55 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        int _smem_desc_lo_56 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_55) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_56) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_54);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_54);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_54);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1, _elected_54);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                        half1_stage += 1;
                        if (half1_stage == 4) { half1_stage = 0; }
                        mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_57 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_57);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_57);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_57);
                        int _smem_desc_lo_58 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        int _smem_desc_lo_59 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_58) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_59) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_57);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_57);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_57);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 16, tmem_tmem_sfb, 1, _elected_57);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        uint32_t _elected_60 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_60);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_60);
                        int _smem_desc_lo_61 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        int _smem_desc_lo_62 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_61) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_62) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_60);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_60);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_60);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 24, 1, _elected_60);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                        half1_stage += 1;
                        if (half1_stage == 4) { half1_stage = 0; }
                        mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_63 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_63);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008)), _elected_63);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3008 + 32)), _elected_63);
                        int _smem_desc_lo_64 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        int _smem_desc_lo_65 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3008;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_64) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_65) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10b80000U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_63);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 2, b_desc + 2,
                                0x30b80010U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_63);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 4, b_desc + 4,
                                0x50b80020U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_63);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum0, a_desc + 6, b_desc + 6,
                                0x70b80030U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 16, 1, _elected_63);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_half_0_addr, (uint16_t)(3));
                    uint32_t _elected_66 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_66);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_66);
                    int _smem_desc_lo_67 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    int _smem_desc_lo_68 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_67) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_68) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_66);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_66);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_66);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 16, tmem_tmem_sfb + 8, 1, _elected_66);
                    }
                    elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                    half1_stage += 1;
                    if (half1_stage == 4) { half1_stage = 0; }
                    uint32_t _elected_69 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008)), _elected_69);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (half1_stage) * 3008 + 32)), _elected_69);
                    int _smem_desc_lo_70 = (((smem_a_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    int _smem_desc_lo_71 = (((smem_v3_addr) >> 4) & 0x3FFF) + (half1_stage) * 3008;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_70) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_71) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                            0x10b80000U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_69);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 2, b_desc + 2,
                            0x30b80010U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_69);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 4, b_desc + 4,
                            0x50b80020U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_69);
                        tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum1, a_desc + 6, b_desc + 6,
                            0x70b80030U, tmem_tmem_sfa + 20, tmem_tmem_sfb + 24, 1, _elected_69);
                    }
                    elect_commit_cg2_multicast(mma_done_addr + (half1_stage) * 8, (uint16_t)(3));
                    half1_stage += 1;
                    if (half1_stage == 4) { half1_stage = 0; }
                    elect_commit_cg2_multicast(mainloop_done_half_1_addr, (uint16_t)(3));
                    _phase_epilogue_done_half_0 ^= 1;
                    _phase_epilogue_done_half_1 ^= 1;
                    mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 10000000);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int acc_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            int parity = 0;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done_half_0 = 0;
            unsigned int _phase_mainloop_done_half_1 = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < m_tiles / 2 * 64; _tile_iter_2++) {
                if (acc_half == 0) {
                    mbarrier_wait_hint(mainloop_done_half_0_addr, _phase_mainloop_done_half_0, 10000000);
                } else {
                    mbarrier_wait_hint(mainloop_done_half_1_addr, _phase_mainloop_done_half_1, 10000000);
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * 64;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= 64) ? 64 : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * 128;
                int off_n_1 = bid_n_1 * 224;
                int global_row = off_m_1 + local_row;
                unsigned long long row_out = (unsigned long long)global_row * 14336 + (unsigned long long)off_n_1;
                unsigned long long sf_row_base = (unsigned long long)bid_m_1 * 112 * 512 + (unsigned long long)(local_row & 31) * 16 + (unsigned long long)(local_row >> 5) * 4;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(acc_half * 224);
                unsigned int gate_pk[56];
                unsigned int up_pk[56];
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], lane_addr + acc_half * 16);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], lane_addr + acc_half * 16 + 112);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], lane_addr + (acc_half + 1) * 16);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], lane_addr + (acc_half + 1) * 16 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    gate_pk[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    up_pk[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    gate_pk[_lp + 8] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    up_pk[_lp + 8] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], lane_addr + (acc_half + 2) * 16);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], lane_addr + (acc_half + 2) * 16 + 112);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], lane_addr + (acc_half + 3) * 16);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], lane_addr + (acc_half + 3) * 16 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    gate_pk[_lp + 16] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    up_pk[_lp + 16] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    gate_pk[_lp + 24] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                    up_pk[_lp + 24] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], lane_addr + (acc_half + 4) * 16);
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], lane_addr + (acc_half + 4) * 16 + 112);
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], lane_addr + (acc_half + 5) * 16);
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], lane_addr + (acc_half + 5) * 16 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                    gate_pk[_lp + 32] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                    up_pk[_lp + 32] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                    gate_pk[_lp + 40] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                    up_pk[_lp + 40] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], lane_addr + (1 - acc_half) * 96);
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], lane_addr + (1 - acc_half) * 96 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                    gate_pk[_lp + 48] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 0], _tmem_load_13[_lp*2+1 + 0]));
                    up_pk[_lp + 48] = *(uint32_t*)&_bf2;
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (acc_half == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_half_0_addr) & 0xFEFFFFFF) : "memory");
                    } else {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_half_1_addr) & 0xFEFFFFFF) : "memory");
                    }
                }
                _phase_mainloop_done_half_0 ^= 1;
                _phase_mainloop_done_half_1 ^= 1;
                float vals[16];
                #pragma unroll
                for (int jp = 0; jp < 8; jp++) {
                    float gate_lo = __uint_as_float(gate_pk[48 + jp] << 16);
                    float up_lo = __uint_as_float(up_pk[48 + jp] << 16);
                    float _exp2_0 = approx_exp2((-gate_lo) * 1.442695f);
                    float sig_lo = approx_rcp(1.0f + _exp2_0);
                    float pair[2];
                    pair[0] = 0.0f;
                    pair[1] = gate_lo * sig_lo;
                    unsigned int word[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                        word[_lp] = *(uint32_t*)&_bf2;
                    }
                    float silu_lo = __uint_as_float(word[0]);
                    float pair_0[2];
                    pair_0[0] = 0.0f;
                    pair_0[1] = silu_lo * up_lo;
                    unsigned int word_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_0[_lp*2 + 0], pair_0[_lp*2+1 + 0]));
                        word_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    vals[jp * 2] = __uint_as_float(word_1[0]);
                    float gate_hi = __uint_as_float(gate_pk[48 + jp] & 4294901760u);
                    float up_hi = __uint_as_float(up_pk[48 + jp] & 4294901760u);
                    float _exp2_1 = approx_exp2((-gate_hi) * 1.442695f);
                    float sig_hi = approx_rcp(1.0f + _exp2_1);
                    float pair_2[2];
                    pair_2[0] = 0.0f;
                    pair_2[1] = gate_hi * sig_hi;
                    unsigned int word_3[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_2[_lp*2 + 0], pair_2[_lp*2+1 + 0]));
                        word_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    float silu_hi = __uint_as_float(word_3[0]);
                    float pair_4[2];
                    pair_4[0] = 0.0f;
                    pair_4[1] = silu_hi * up_hi;
                    unsigned int word_5[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_4[_lp*2 + 0], pair_4[_lp*2+1 + 0]));
                        word_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    vals[jp * 2 + 1] = __uint_as_float(word_5[0]);
                }
                float mags[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    mags[j] = vals[j];
                }
                mags[0] = fabsf(mags[0]);
                mags[1] = fabsf(mags[1]);
                mags[2] = fabsf(mags[2]);
                mags[3] = fabsf(mags[3]);
                mags[4] = fabsf(mags[4]);
                mags[5] = fabsf(mags[5]);
                mags[6] = fabsf(mags[6]);
                mags[7] = fabsf(mags[7]);
                mags[8] = fabsf(mags[8]);
                mags[9] = fabsf(mags[9]);
                mags[10] = fabsf(mags[10]);
                mags[11] = fabsf(mags[11]);
                mags[12] = fabsf(mags[12]);
                mags[13] = fabsf(mags[13]);
                mags[14] = fabsf(mags[14]);
                mags[15] = fabsf(mags[15]);
                float amax = mags[0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    amax = max_noftz(amax, mags[_lr]);
                }
                int slot = ((parity * 2 + acc_half) * 128 + local_row) * 4;
                int other_slot = ((parity * 2 + (1 - acc_half)) * 128 + local_row) * 4;
                {
                    uint32_t _addr_2 = static_cast<uint32_t>(xchg_addr + (unsigned int)slot);
                    asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(amax) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 64;" :: "r"(epi_warp + 1) : "memory");
                float other[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&other[0])) : "r"(xchg_addr + (unsigned int)other_slot));
                float both[2];
                both[0] = amax;
                both[1] = other[0];
                float shared_absmax = both[0];
                #pragma unroll
                for (int _lr = 1; _lr < 2; _lr++) {
                    shared_absmax = max_noftz(shared_absmax, both[_lr]);
                }
                float scale = shared_absmax / 448.0f;
                unsigned int scale_bits = __as_u32(scale);
                unsigned int exponent = scale_bits >> 23 & 255;
                unsigned int mantissa = scale_bits & 8388607;
                unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
                unsigned int normal = ((exponent != 0) ? 1 : 0);
                unsigned int large_subnormal = ((mantissa > 4194304) ? 1 : 0);
                unsigned int scale_byte = ((exponent + (has_mantissa & (normal | large_subnormal))) < (254) ? (exponent + (has_mantissa & (normal | large_subnormal))) : (254));
                unsigned int inverse_nonzero_bits = 254 - scale_byte << 23;
                unsigned int zero_bits = 0;
                unsigned int inverse_bits = ((scale_byte == 0) ? zero_bits : inverse_nonzero_bits);
                float inverse = 0.0f;
                inverse = reinterpret_cast<float*>(&inverse_bits)[0];
                const float2 _scale2_3 = {inverse, inverse};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vals)[_ls], _scale2_3);
                if (global_row < M) {
                    unsigned int _packed_0[4];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[0]), "f"(vals[1]),
                                               "f"(vals[2]), "f"(vals[3]));
                        _packed_0[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[4]), "f"(vals[5]),
                                               "f"(vals[6]), "f"(vals[7]));
                        _packed_0[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[8]), "f"(vals[9]),
                                               "f"(vals[10]), "f"(vals[11]));
                        _packed_0[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[12]), "f"(vals[13]),
                                               "f"(vals[14]), "f"(vals[15]));
                        _packed_0[3] = _packed;
                    }
                    reinterpret_cast<int4*>(y_q + (row_out + (unsigned long long)(96 + acc_half * 16)))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                    if (acc_half == 0) {
                        int shared_block = bid_n_1 * 7 + 3;
                        int k_tile = shared_block >> 2;
                        int sf_col = shared_block & 3;
                        *(reinterpret_cast<unsigned char*>(y_sf + (sf_row_base + (unsigned long long)k_tile * 512 + (unsigned long long)sf_col)) + (0)) = (unsigned char)(scale_byte);
                    }
                }
                parity = parity ^ 1;
                #pragma unroll
                for (int blk = 0; blk < 3; blk++) {
                    float vals_0[32];
                    #pragma unroll
                    for (int jp_1 = 0; jp_1 < 16; jp_1++) {
                        float gate_lo_1 = __uint_as_float(gate_pk[blk * 16 + jp_1] << 16);
                        float up_lo_1 = __uint_as_float(up_pk[blk * 16 + jp_1] << 16);
                        float _exp2_4 = approx_exp2((-gate_lo_1) * 1.442695f);
                        float sig_lo_1 = approx_rcp(1.0f + _exp2_4);
                        float pair_1[2];
                        pair_1[0] = 0.0f;
                        pair_1[1] = gate_lo_1 * sig_lo_1;
                        unsigned int word_2[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_1[_lp*2 + 0], pair_1[_lp*2+1 + 0]));
                            word_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        float silu_lo_1 = __uint_as_float(word_2[0]);
                        float pair_0_1[2];
                        pair_0_1[0] = 0.0f;
                        pair_0_1[1] = silu_lo_1 * up_lo_1;
                        unsigned int word_1_1[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_0_1[_lp*2 + 0], pair_0_1[_lp*2+1 + 0]));
                            word_1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        vals_0[jp_1 * 2] = __uint_as_float(word_1_1[0]);
                        float gate_hi_1 = __uint_as_float(gate_pk[blk * 16 + jp_1] & 4294901760u);
                        float up_hi_1 = __uint_as_float(up_pk[blk * 16 + jp_1] & 4294901760u);
                        float _exp2_5 = approx_exp2((-gate_hi_1) * 1.442695f);
                        float sig_hi_1 = approx_rcp(1.0f + _exp2_5);
                        float pair_2_1[2];
                        pair_2_1[0] = 0.0f;
                        pair_2_1[1] = gate_hi_1 * sig_hi_1;
                        unsigned int word_3_1[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_2_1[_lp*2 + 0], pair_2_1[_lp*2+1 + 0]));
                            word_3_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        float silu_hi_1 = __uint_as_float(word_3_1[0]);
                        float pair_4_1[2];
                        pair_4_1[0] = 0.0f;
                        pair_4_1[1] = silu_hi_1 * up_hi_1;
                        unsigned int word_5_1[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_4_1[_lp*2 + 0], pair_4_1[_lp*2+1 + 0]));
                            word_5_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        vals_0[jp_1 * 2 + 1] = __uint_as_float(word_5_1[0]);
                    }
                    float mags_1[32];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 32; j_1++) {
                        mags_1[j_1] = vals_0[j_1];
                    }
                    mags_1[0] = fabsf(mags_1[0]);
                    mags_1[1] = fabsf(mags_1[1]);
                    mags_1[2] = fabsf(mags_1[2]);
                    mags_1[3] = fabsf(mags_1[3]);
                    mags_1[4] = fabsf(mags_1[4]);
                    mags_1[5] = fabsf(mags_1[5]);
                    mags_1[6] = fabsf(mags_1[6]);
                    mags_1[7] = fabsf(mags_1[7]);
                    mags_1[8] = fabsf(mags_1[8]);
                    mags_1[9] = fabsf(mags_1[9]);
                    mags_1[10] = fabsf(mags_1[10]);
                    mags_1[11] = fabsf(mags_1[11]);
                    mags_1[12] = fabsf(mags_1[12]);
                    mags_1[13] = fabsf(mags_1[13]);
                    mags_1[14] = fabsf(mags_1[14]);
                    mags_1[15] = fabsf(mags_1[15]);
                    mags_1[16] = fabsf(mags_1[16]);
                    mags_1[17] = fabsf(mags_1[17]);
                    mags_1[18] = fabsf(mags_1[18]);
                    mags_1[19] = fabsf(mags_1[19]);
                    mags_1[20] = fabsf(mags_1[20]);
                    mags_1[21] = fabsf(mags_1[21]);
                    mags_1[22] = fabsf(mags_1[22]);
                    mags_1[23] = fabsf(mags_1[23]);
                    mags_1[24] = fabsf(mags_1[24]);
                    mags_1[25] = fabsf(mags_1[25]);
                    mags_1[26] = fabsf(mags_1[26]);
                    mags_1[27] = fabsf(mags_1[27]);
                    mags_1[28] = fabsf(mags_1[28]);
                    mags_1[29] = fabsf(mags_1[29]);
                    mags_1[30] = fabsf(mags_1[30]);
                    mags_1[31] = fabsf(mags_1[31]);
                    float2 _reg_reduce_max2_6 = {-MINIMAX_H3_MLP_INF, -MINIMAX_H3_MLP_INF};
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[0], mags_1[1]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[2], mags_1[3]));
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[4], mags_1[5]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[6], mags_1[7]));
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[8], mags_1[9]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[10], mags_1[11]));
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[12], mags_1[13]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[14], mags_1[15]));
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[16], mags_1[17]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[18], mags_1[19]));
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[20], mags_1[21]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[22], mags_1[23]));
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[24], mags_1[25]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[26], mags_1[27]));
                    _reg_reduce_max2_6.x = max_noftz(_reg_reduce_max2_6.x, max_noftz(mags_1[28], mags_1[29]));
                    _reg_reduce_max2_6.y = max_noftz(_reg_reduce_max2_6.y, max_noftz(mags_1[30], mags_1[31]));
                    float amax_2 = row_max_reduce(_reg_reduce_max2_6);
                    float scale_3 = amax_2 / 448.0f;
                    unsigned int scale_bits_4 = __as_u32(scale_3);
                    unsigned int exponent_5 = scale_bits_4 >> 23 & 255;
                    unsigned int mantissa_6 = scale_bits_4 & 8388607;
                    unsigned int has_mantissa_7 = ((mantissa_6 != 0) ? 1 : 0);
                    unsigned int normal_8 = ((exponent_5 != 0) ? 1 : 0);
                    unsigned int large_subnormal_9 = ((mantissa_6 > 4194304) ? 1 : 0);
                    unsigned int scale_byte_10 = ((exponent_5 + (has_mantissa_7 & (normal_8 | large_subnormal_9))) < (254) ? (exponent_5 + (has_mantissa_7 & (normal_8 | large_subnormal_9))) : (254));
                    unsigned int inverse_nonzero_bits_11 = 254 - scale_byte_10 << 23;
                    unsigned int zero_bits_12 = 0;
                    unsigned int inverse_bits_13 = ((scale_byte_10 == 0) ? zero_bits_12 : inverse_nonzero_bits_11);
                    float inverse_14 = 0.0f;
                    inverse_14 = reinterpret_cast<float*>(&inverse_bits_13)[0];
                    const float2 _scale2_7 = {inverse_14, inverse_14};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vals_0)[_ls], _scale2_7);
                    if (global_row < M) {
                        unsigned int _packed_1[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[0]), "f"(vals_0[1]),
                                                   "f"(vals_0[2]), "f"(vals_0[3]));
                            _packed_1[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[4]), "f"(vals_0[5]),
                                                   "f"(vals_0[6]), "f"(vals_0[7]));
                            _packed_1[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[8]), "f"(vals_0[9]),
                                                   "f"(vals_0[10]), "f"(vals_0[11]));
                            _packed_1[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[12]), "f"(vals_0[13]),
                                                   "f"(vals_0[14]), "f"(vals_0[15]));
                            _packed_1[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[16]), "f"(vals_0[17]),
                                                   "f"(vals_0[18]), "f"(vals_0[19]));
                            _packed_1[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[20]), "f"(vals_0[21]),
                                                   "f"(vals_0[22]), "f"(vals_0[23]));
                            _packed_1[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[24]), "f"(vals_0[25]),
                                                   "f"(vals_0[26]), "f"(vals_0[27]));
                            _packed_1[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(vals_0[28]), "f"(vals_0[29]),
                                                   "f"(vals_0[30]), "f"(vals_0[31]));
                            _packed_1[7] = _packed;
                        }
                        {
                            const unsigned* _raw_stv8_8 = reinterpret_cast<const unsigned*>(_packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(y_q + (row_out + (unsigned long long)(acc_half * 128 + blk * 32)))), "r"(_raw_stv8_8[0]), "r"(_raw_stv8_8[1]), "r"(_raw_stv8_8[2]), "r"(_raw_stv8_8[3]), "r"(_raw_stv8_8[4]), "r"(_raw_stv8_8[5]), "r"(_raw_stv8_8[6]), "r"(_raw_stv8_8[7]) : "memory");
                        }
                        int whole_block = bid_n_1 * 7 + acc_half * 4 + blk;
                        int k_tile_1 = whole_block >> 2;
                        int sf_col_1 = whole_block & 3;
                        *(reinterpret_cast<unsigned char*>(y_sf + (sf_row_base + (unsigned long long)k_tile_1 * 512 + (unsigned long long)sf_col_1)) + (0)) = (unsigned char)(scale_byte_10);
                    }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 10000000);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef MINIMAX_H3_MLP_INF
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_SFA_ALL_OFF
#undef SMEM_SMEM_SFA_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFA_ALL_STRIDE
#undef SMEM_SMEM_SFA_OFF
#undef SMEM_SMEM_SFA_STAGE_BYTES
#undef SMEM_SMEM_SFA_STRIDE
#undef SMEM_SMEM_V2_OFF
#undef SMEM_SMEM_V2_STAGE_BYTES
#undef SMEM_SMEM_V2_STRIDE
#undef SMEM_SMEM_V3_OFF
#undef SMEM_SMEM_V3_STAGE_BYTES
#undef SMEM_SMEM_V3_STRIDE
#undef SMEM_SMEM_V6_OFF
#undef SMEM_SMEM_V6_STAGE_BYTES
#undef SMEM_SMEM_V6_STRIDE
#undef SMEM_SMEM_V7_OFF
#undef SMEM_SMEM_V7_STAGE_BYTES
#undef SMEM_SMEM_V7_STRIDE
#undef SMEM_SMEM_V8_OFF
#undef SMEM_SMEM_V8_STAGE_BYTES
#undef SMEM_SMEM_V8_STRIDE
#undef SMEM_SMEM_V9_OFF
#undef SMEM_SMEM_V9_STAGE_BYTES
#undef SMEM_SMEM_V9_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef SMEM_XCHG_OFF
#undef SMEM_XCHG_STAGE_BYTES
#undef SMEM_XCHG_STRIDE
#undef TMEM_ACCUM0_OFFSET
#undef TMEM_ACCUM1_OFFSET
#undef TMEM_NCOLS
#undef TMEM_TMEM_SFA_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef tiles_per_group

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define TMEM_NCOLS 280
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 68608
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 68608
#define SMEM_SMEM_SFA_ALL_OFF 66560
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_ALL_STRIDE 68608
#define SMEM_SMEM_SFB_ALL_OFF 67584
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_ALL_STRIDE 68608
#define SMEM_SMEM_SFA0_OFF 66560
#define SMEM_SMEM_SFA0_STAGE_BYTES 512
#define SMEM_SMEM_SFA0_STRIDE 68608
#define SMEM_SMEM_SFA1_OFF 67072
#define SMEM_SMEM_SFA1_STAGE_BYTES 512
#define SMEM_SMEM_SFA1_STRIDE 68608
#define SMEM_SMEM_SFB0_OFF 67584
#define SMEM_SMEM_SFB0_STAGE_BYTES 1024
#define SMEM_SMEM_SFB0_STRIDE 68608
#define SMEM_SMEM_SFB1_OFF 68608
#define SMEM_SMEM_SFB1_STAGE_BYTES 1024
#define SMEM_SMEM_SFB1_STRIDE 68608
#define SMEM_WORK_RESPONSE_OFF 206848
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 206976
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 3
#define WORK_STAGES 4
#define NUM_K_ITERS 56
#define K_HALF 28
#define EPI_WARPS 8
#define N_TILES 21
#define SF_K_TILES 112
#define HIDDEN 5376
#define num_cluster_tiles (((m_tiles / CTA_GROUP) * N_TILES) + split_count)
#define tiles_per_group (group_m * N_TILES)

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_mlp_fc2_e4m3(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride, float* __restrict__ partial, unsigned int* __restrict__ flags, int tail_base, int split_count, int group_m)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 24)
    #define mainloop_done_addr (mbar_base + 48)
    #define epilogue_done_addr (mbar_base + 56)
    #define work_full_addr (mbar_base + 64)
    #define work_empty_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA_ALL_OFF);
    const int smem_sfa_all_addr = smem + SMEM_SMEM_SFA_ALL_OFF;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_ALL_OFF);
    const int smem_sfb_all_addr = smem + SMEM_SMEM_SFB_ALL_OFF;
    uint8_t* smem_sfa0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA0_OFF);
    const int smem_sfa0_addr = smem + SMEM_SMEM_SFA0_OFF;
    uint8_t* smem_sfa1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA1_OFF);
    const int smem_sfa1_addr = smem + SMEM_SMEM_SFA1_OFF;
    uint8_t* smem_sfb0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB0_OFF);
    const int smem_sfb0_addr = smem + SMEM_SMEM_SFB0_OFF;
    uint8_t* smem_sfb1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB1_OFF);
    const int smem_sfb1_addr = smem + SMEM_SMEM_SFB1_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 56, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 96, 546);
            mbarrier_init(smem + 104, 546);
            mbarrier_init(smem + 112, 546);
            mbarrier_init(smem + 120, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 280 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 264;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int this_bid_0 = (int)this_bid;
                    int tail_u = this_bid_0 - tail_base;
                    int in_tail = ((tail_u >= 0) ? 1 : 0);
                    int q = tail_u >> 1;
                    int is_h0 = ((q < split_count) ? 1 : 0);
                    int is_h1 = ((q >= split_count && q < 2 * split_count) ? 1 : 0);
                    int half_tail = ((is_h0 == 1) ? 0 : ((is_h1 == 1) ? 1 : -1));
                    int half = ((in_tail == 1) ? half_tail : -1);
                    int j = ((is_h0 == 1) ? q : q - split_count);
                    int slot = ((half >= 0) ? j : 0);
                    int tile_tail = tail_base + 2 * j + (tail_u & 1);
                    int tile = ((in_tail == 1) ? tile_tail : this_bid_0);
                    int k0 = ((half == 1) ? K_HALF : 0);
                    int k1 = ((half == 0) ? K_HALF : NUM_K_ITERS);
                    int group = tile / tiles_per_group;
                    int first_m = group * group_m;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= group_m) ? group_m : remaining);
                    int local = tile % tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll 1
                    for (int iter_k = k0; iter_k < k1; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 68608, (&A), 0, off_m, k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 68608, (&B), 0, weight_row, k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 68608, (&SFA), 0, 0, sfa_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 68608, (&SFB), 0, 0, sfb_tile_row + k_group, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(68608)) : "memory");
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 3; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int this_bid_m = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    int this_bid_1 = (int)this_bid_m;
                    int tail_u_1 = this_bid_1 - tail_base;
                    int in_tail_1 = ((tail_u_1 >= 0) ? 1 : 0);
                    int q_1 = tail_u_1 >> 1;
                    int is_h0_1 = ((q_1 < split_count) ? 1 : 0);
                    int is_h1_1 = ((q_1 >= split_count && q_1 < 2 * split_count) ? 1 : 0);
                    int half_tail_1 = ((is_h0_1 == 1) ? 0 : ((is_h1_1 == 1) ? 1 : -1));
                    int half_1 = ((in_tail_1 == 1) ? half_tail_1 : -1);
                    int j_1 = ((is_h0_1 == 1) ? q_1 : q_1 - split_count);
                    int slot_1 = ((half_1 >= 0) ? j_1 : 0);
                    int tile_tail_1 = tail_base + 2 * j_1 + (tail_u_1 & 1);
                    int tile_1 = ((in_tail_1 == 1) ? tile_tail_1 : this_bid_1);
                    int k0_1 = ((half_1 == 1) ? K_HALF : 0);
                    int k1_1 = ((half_1 == 0) ? K_HALF : NUM_K_ITERS);
                    int group_1 = tile_1 / tiles_per_group;
                    int first_m_1 = group_1 * group_m;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= group_m) ? group_m : remaining_1);
                    int local_1 = tile_1 % tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * BLOCK_M;
                    int off_n_1 = bid_n_1 * BLOCK_N;
                    mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = k0_1; iter_k_1 < k1_1; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == k0_1) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288 + 32)));
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            int _mma_b_lo_1 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10c00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 2, b_desc + 2,
                                    0x30c00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 4, b_desc + 4,
                                    0x50c00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                                tcgen05_mma_mxf8_bs_cta2(tmem_accum, a_desc + 6, b_desc + 6,
                                    0x70c00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (acc_stage) * 8, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                    this_bid_m = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            int epi_cta_rank = (int)cta_rank;
            unsigned int this_bid_2 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                int this_bid_0_1 = (int)this_bid_2;
                int tail_u_2 = this_bid_0_1 - tail_base;
                int in_tail_2 = ((tail_u_2 >= 0) ? 1 : 0);
                int q_2 = tail_u_2 >> 1;
                int is_h0_2 = ((q_2 < split_count) ? 1 : 0);
                int is_h1_2 = ((q_2 >= split_count && q_2 < 2 * split_count) ? 1 : 0);
                int half_tail_2 = ((is_h0_2 == 1) ? 0 : ((is_h1_2 == 1) ? 1 : -1));
                int half_2 = ((in_tail_2 == 1) ? half_tail_2 : -1);
                int j_2 = ((is_h0_2 == 1) ? q_2 : q_2 - split_count);
                int slot_2 = ((half_2 >= 0) ? j_2 : 0);
                int tile_tail_2 = tail_base + 2 * j_2 + (tail_u_2 & 1);
                int tile_2 = ((in_tail_2 == 1) ? tile_tail_2 : this_bid_0_1);
                int k0_2 = ((half_2 == 1) ? K_HALF : 0);
                int k1_2 = ((half_2 == 0) ? K_HALF : NUM_K_ITERS);
                int group_2 = tile_2 / tiles_per_group;
                int first_m_2 = group_2 * group_m;
                int remaining_2 = m_tiles - first_m_2;
                int group_size_2 = ((remaining_2 >= group_m) ? group_m : remaining_2);
                int local_2 = tile_2 % tiles_per_group;
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int off_m_2 = bid_m_2 * BLOCK_M;
                int off_n_2 = bid_n_2 * BLOCK_N;
                int global_row = off_m_2 + local_row;
                int col0 = col_part * 128;
                int flag_idx = slot_2 * 2 + epi_cta_rank;
                int partial_row = flag_idx * BLOCK_M + local_row;
                unsigned long long partial_base = (unsigned long long)partial_row * (unsigned long long)BLOCK_N + (unsigned long long)col0;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                int _min_0 = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                int load_row = _min_0;
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_2 + (unsigned long long)col0;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_2 + (unsigned long long)col0;
                mbarrier_wait(mainloop_done_addr + (acc_stage_1) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                    : "r"(lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                if (half_2 == 0) {
                    #pragma unroll
                    for (int q_3 = 0; q_3 < 32; q_3++) {
                        float part_q[4];
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 4; j_3++) {
                            part_q[j_3] = _tmem_load_0[q_3 * 4 + j_3];
                        }
                        {
                            float4 _v4 = make_float4(part_q[0 + 0], part_q[0 + 1], part_q[0 + 2], part_q[0 + 3]);
                            *reinterpret_cast<float4*>((partial + (partial_base + (unsigned long long)(q_3 * 4))) + 0) = _v4;
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
                if (half_2 == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(EPI_WARPS)) >= static_cast<unsigned int>(1));
                    }
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 32; q_4++) {
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (partial_base + (unsigned long long)(q_4 * 4)) + 0);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 4; j_4++) {
                            _tmem_load_0[q_4 * 4 + j_4] = _tmem_load_0[q_4 * 4 + j_4] + _vec_load_0[j_4];
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))), "r"(static_cast<unsigned int>(4294967295)) : "memory");
                    }
                }
                if (half_2 != 0) {
                    if (global_row < M) {
                        #pragma unroll
                        for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                            int col = n_chunk * 16;
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                                uint4 _vld_1[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_1[_blk] = _vptr_1[_blk];
                                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_2[8];
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
                                uint4 _vld_2[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_3[8];
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
                                uint4 _vld_3[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float out_vals[16];
                            #pragma unroll
                            for (int j_5 = 0; j_5 < 8; j_5++) {
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[n_chunk * 16 + j_5]);
                                float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                float o_lo = _cvt_f32_0;
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_vec_load_1[j_5] * gate_mul * o_lo);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                float p_lo = _cvt_f32_1;
                                out_vals[j_5] = _vec_load_3[j_5] + p_lo;
                                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_tmem_load_0[n_chunk * 16 + j_5 + 8]);
                                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                                float o_hi = _cvt_f32_2;
                                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_vec_load_2[j_5] * gate_mul * o_hi);
                                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                                float p_hi = _cvt_f32_3;
                                out_vals[j_5 + 8] = _vec_load_4[j_5] + p_hi;
                            }
                            {
                                {
                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals[0 + 0], out_vals[0 + 1]);
                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals[0 + 2], out_vals[0 + 3]);
                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals[0 + 4], out_vals[0 + 5]);
                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals[0 + 6], out_vals[0 + 7]);
                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals[0 + 8], out_vals[0 + 9]);
                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals[0 + 10], out_vals[0 + 11]);
                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals[0 + 12], out_vals[0 + 13]);
                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals[0 + 14], out_vals[0 + 15]);
                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(&((__nv_bfloat16*)(out + (row_base + (unsigned long long)col)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                }
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_2 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef EPI_WARPS
#undef HIDDEN
#undef K_HALF
#undef MINIMAX_H3_MLP_INF
#undef NUM_K_ITERS
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef N_TILES
#undef SF_K_TILES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_SFA0_OFF
#undef SMEM_SMEM_SFA0_STAGE_BYTES
#undef SMEM_SMEM_SFA0_STRIDE
#undef SMEM_SMEM_SFA1_OFF
#undef SMEM_SMEM_SFA1_STAGE_BYTES
#undef SMEM_SMEM_SFA1_STRIDE
#undef SMEM_SMEM_SFA_ALL_OFF
#undef SMEM_SMEM_SFA_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFA_ALL_STRIDE
#undef SMEM_SMEM_SFB0_OFF
#undef SMEM_SMEM_SFB0_STAGE_BYTES
#undef SMEM_SMEM_SFB0_STRIDE
#undef SMEM_SMEM_SFB1_OFF
#undef SMEM_SMEM_SFB1_STAGE_BYTES
#undef SMEM_SMEM_SFB1_STRIDE
#undef SMEM_SMEM_SFB_ALL_OFF
#undef SMEM_SMEM_SFB_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFB_ALL_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef TMEM_TMEM_SFA_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef WORK_STAGES
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef num_cluster_tiles
#undef smem_a_addr
#undef smem_b_addr
#undef smem_sfa0_addr
#undef smem_sfa1_addr
#undef smem_sfa_all_addr
#undef smem_sfb0_addr
#undef smem_sfb1_addr
#undef smem_sfb_all_addr
#undef tiles_per_group
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_X_RING_OFF 1024
#define SMEM_X_RING_STAGE_BYTES 10752
#define SMEM_X_RING_STRIDE 10752
#define SMEM_W_SMEM_OFF 87040
#define SMEM_W_SMEM_STAGE_BYTES 21504
#define SMEM_W_SMEM_STRIDE 21504
#define SMEM_RSTD_SMEM_OFF 108544
#define SMEM_RSTD_SMEM_STAGE_BYTES 64
#define SMEM_RSTD_SMEM_STRIDE 64
#define SMEM_TOTAL 108672
#define THREADS 768
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(768, LAUNCH_MIN_BLOCKS) void
kernel_minimax_h3_mlp_norm_adaln_nvfp4(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, long long* __restrict__ adaln_index, float* __restrict__ a_global_scale, unsigned int* __restrict__ a_q, uint8_t* __restrict__ a_sf, int M, int adaln_rows, long long adaln_row_stride, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define w_full_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* x_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int x_ring_addr = smem + 1024;
    float* w_smem = reinterpret_cast<float*>(smem_raw + 87040);
    const int w_smem_addr = smem + 87040;
    float* rstd_smem = reinterpret_cast<float*>(smem_raw + 108544);
    const int rstd_smem_addr = smem + 108544;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // x_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int group = warp / 3;
    int member = warp - group * 3;
    long long ctas = (long long)gridDim.x;
    int run_lo = (int)((long long)bid * (long long)M / ctas);
    int run_hi = (int)((long long)(bid + 1) * (long long)M / ctas);
    int n_rows = (run_hi - run_lo - group + 8 - 1) / 8;
    int first_row = run_lo + group;
    int slot0 = group;
    if (elect_sync()) {
        if (member == 0 && n_rows > 0) {
            mbarrier_arrive_expect_tx(x_full_addr + (slot0) * 8, 10752);
            cp_async_bulk_gmem2smem(x_ring_addr + (unsigned int)(slot0 * 10752), x + ((unsigned long long)first_row * 5376), 10752, x_full_addr + (slot0) * 8);
        }
    }
    float g = a_global_scale[0];
    float inv_g = approx_rcp(g);
    float sixth = approx_rcp(6.0f);
    float zero_f32 = 0.0f;
    int own_first = member * 7;
    int own_col = lane * 8 + own_first * 256;
    unsigned int lane_bytes = (unsigned int)(lane * 16);
    unsigned int own_bytes = lane_bytes + (unsigned int)(own_first * 512);
    unsigned int w_lane = w_smem_addr + own_bytes;
    int pair_lane = lane & 1;
    int block_in_vec = lane >> 1;
    unsigned long long sf_lane_col = (unsigned long long)(block_in_vec >> 2) * 512 + (unsigned long long)(block_in_vec & 3) + (unsigned long long)(own_first * 2048);
    int w_vec_id = tid;
    if (w_vec_id < 672) {
        unsigned int _vec_load_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(x_norm_weight + (unsigned long long)(w_vec_id * 8) + 0);
            _vec_load_0[0 + 0] = _uv4_0.x;
            _vec_load_0[0 + 1] = _uv4_0.y;
            _vec_load_0[0 + 2] = _uv4_0.z;
            _vec_load_0[0 + 3] = _uv4_0.w;
        }
        float _unpack_bf16_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_0[_pair]) << 16);
            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_0[_pair]) & 0xffff0000u);
        }
        unsigned int w_at = w_smem_addr + (unsigned int)(w_vec_id * 16);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(w_at), "r"(__float_as_uint(_unpack_bf16_0[0])), "r"(__float_as_uint(_unpack_bf16_0[1])), "r"(__float_as_uint(_unpack_bf16_0[2])), "r"(__float_as_uint(_unpack_bf16_0[3])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(w_at + 10752), "r"(__float_as_uint(_unpack_bf16_0[4])), "r"(__float_as_uint(_unpack_bf16_0[5])), "r"(__float_as_uint(_unpack_bf16_0[6])), "r"(__float_as_uint(_unpack_bf16_0[7])) : "memory");
    }
    __syncthreads();
    for (int t = 0; t < n_rows; t++) {
        int row = first_row + t * 8;
        int slot = slot0;
        long long table_row = adaln_index[row];
        mbarrier_wait(x_full_addr + (slot) * 8, t & 1);
        unsigned int slot_addr = x_ring_addr + (unsigned int)(slot * 10752);
        unsigned int x_own[28];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&x_own[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(0) + 3]))
            : "r"(slot_addr + own_bytes));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&x_own[4])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(4) + 3]))
            : "r"(slot_addr + own_bytes + 512));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&x_own[8])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(8) + 3]))
            : "r"(slot_addr + own_bytes + 1024));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&x_own[12])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(12) + 3]))
            : "r"(slot_addr + own_bytes + 1536));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&x_own[16])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(16) + 3]))
            : "r"(slot_addr + own_bytes + 2048));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&x_own[20])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(20) + 3]))
            : "r"(slot_addr + own_bytes + 2560));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&x_own[24])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_own[(24) + 3]))
            : "r"(slot_addr + own_bytes + 3072));
        float rstd = 0.0f;
        unsigned int rstd_at = rstd_smem_addr + (unsigned int)((group * 2 + (t & 1)) * 4);
        if (member == t % 3) {
            float split_sum = 0.0f;
            unsigned int row_words[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words[(0) + 3]))
                : "r"(slot_addr + lane_bytes));
            float _dot_1;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_1) : "r"(row_words[0]), "r"(row_words[0]), "f"(split_sum));
            split_sum = _dot_1;
            float _dot_2;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_2) : "r"(row_words[1]), "r"(row_words[1]), "f"(split_sum));
            split_sum = _dot_2;
            float _dot_3;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_3) : "r"(row_words[2]), "r"(row_words[2]), "f"(split_sum));
            split_sum = _dot_3;
            float _dot_4;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_4) : "r"(row_words[3]), "r"(row_words[3]), "f"(split_sum));
            split_sum = _dot_4;
            unsigned int row_words_0[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_0[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 512));
            float _dot_5;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_5) : "r"(row_words_0[0]), "r"(row_words_0[0]), "f"(split_sum));
            split_sum = _dot_5;
            float _dot_6;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_6) : "r"(row_words_0[1]), "r"(row_words_0[1]), "f"(split_sum));
            split_sum = _dot_6;
            float _dot_7;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_7) : "r"(row_words_0[2]), "r"(row_words_0[2]), "f"(split_sum));
            split_sum = _dot_7;
            float _dot_8;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_8) : "r"(row_words_0[3]), "r"(row_words_0[3]), "f"(split_sum));
            split_sum = _dot_8;
            unsigned int row_words_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_1[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 1024));
            float _dot_9;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_9) : "r"(row_words_1[0]), "r"(row_words_1[0]), "f"(split_sum));
            split_sum = _dot_9;
            float _dot_10;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_10) : "r"(row_words_1[1]), "r"(row_words_1[1]), "f"(split_sum));
            split_sum = _dot_10;
            float _dot_11;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_11) : "r"(row_words_1[2]), "r"(row_words_1[2]), "f"(split_sum));
            split_sum = _dot_11;
            float _dot_12;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_12) : "r"(row_words_1[3]), "r"(row_words_1[3]), "f"(split_sum));
            split_sum = _dot_12;
            unsigned int row_words_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_2[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 1536));
            float _dot_13;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_13) : "r"(row_words_2[0]), "r"(row_words_2[0]), "f"(split_sum));
            split_sum = _dot_13;
            float _dot_14;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_14) : "r"(row_words_2[1]), "r"(row_words_2[1]), "f"(split_sum));
            split_sum = _dot_14;
            float _dot_15;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_15) : "r"(row_words_2[2]), "r"(row_words_2[2]), "f"(split_sum));
            split_sum = _dot_15;
            float _dot_16;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_16) : "r"(row_words_2[3]), "r"(row_words_2[3]), "f"(split_sum));
            split_sum = _dot_16;
            unsigned int row_words_3[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_3[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 2048));
            float _dot_17;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_17) : "r"(row_words_3[0]), "r"(row_words_3[0]), "f"(split_sum));
            split_sum = _dot_17;
            float _dot_18;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_18) : "r"(row_words_3[1]), "r"(row_words_3[1]), "f"(split_sum));
            split_sum = _dot_18;
            float _dot_19;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_19) : "r"(row_words_3[2]), "r"(row_words_3[2]), "f"(split_sum));
            split_sum = _dot_19;
            float _dot_20;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_20) : "r"(row_words_3[3]), "r"(row_words_3[3]), "f"(split_sum));
            split_sum = _dot_20;
            unsigned int row_words_4[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_4[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 2560));
            float _dot_21;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_21) : "r"(row_words_4[0]), "r"(row_words_4[0]), "f"(split_sum));
            split_sum = _dot_21;
            float _dot_22;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_22) : "r"(row_words_4[1]), "r"(row_words_4[1]), "f"(split_sum));
            split_sum = _dot_22;
            float _dot_23;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_23) : "r"(row_words_4[2]), "r"(row_words_4[2]), "f"(split_sum));
            split_sum = _dot_23;
            float _dot_24;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_24) : "r"(row_words_4[3]), "r"(row_words_4[3]), "f"(split_sum));
            split_sum = _dot_24;
            unsigned int row_words_5[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_5[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 3072));
            float _dot_25;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_25) : "r"(row_words_5[0]), "r"(row_words_5[0]), "f"(split_sum));
            split_sum = _dot_25;
            float _dot_26;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_26) : "r"(row_words_5[1]), "r"(row_words_5[1]), "f"(split_sum));
            split_sum = _dot_26;
            float _dot_27;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_27) : "r"(row_words_5[2]), "r"(row_words_5[2]), "f"(split_sum));
            split_sum = _dot_27;
            float _dot_28;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_28) : "r"(row_words_5[3]), "r"(row_words_5[3]), "f"(split_sum));
            split_sum = _dot_28;
            unsigned int row_words_6[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_6[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 3584));
            float _dot_29;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_29) : "r"(row_words_6[0]), "r"(row_words_6[0]), "f"(split_sum));
            split_sum = _dot_29;
            float _dot_30;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_30) : "r"(row_words_6[1]), "r"(row_words_6[1]), "f"(split_sum));
            split_sum = _dot_30;
            float _dot_31;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_31) : "r"(row_words_6[2]), "r"(row_words_6[2]), "f"(split_sum));
            split_sum = _dot_31;
            float _dot_32;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_32) : "r"(row_words_6[3]), "r"(row_words_6[3]), "f"(split_sum));
            split_sum = _dot_32;
            unsigned int row_words_7[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_7[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 4096));
            float _dot_33;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_33) : "r"(row_words_7[0]), "r"(row_words_7[0]), "f"(split_sum));
            split_sum = _dot_33;
            float _dot_34;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_34) : "r"(row_words_7[1]), "r"(row_words_7[1]), "f"(split_sum));
            split_sum = _dot_34;
            float _dot_35;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_35) : "r"(row_words_7[2]), "r"(row_words_7[2]), "f"(split_sum));
            split_sum = _dot_35;
            float _dot_36;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_36) : "r"(row_words_7[3]), "r"(row_words_7[3]), "f"(split_sum));
            split_sum = _dot_36;
            unsigned int row_words_8[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_8[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 4608));
            float _dot_37;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_37) : "r"(row_words_8[0]), "r"(row_words_8[0]), "f"(split_sum));
            split_sum = _dot_37;
            float _dot_38;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_38) : "r"(row_words_8[1]), "r"(row_words_8[1]), "f"(split_sum));
            split_sum = _dot_38;
            float _dot_39;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_39) : "r"(row_words_8[2]), "r"(row_words_8[2]), "f"(split_sum));
            split_sum = _dot_39;
            float _dot_40;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_40) : "r"(row_words_8[3]), "r"(row_words_8[3]), "f"(split_sum));
            split_sum = _dot_40;
            unsigned int row_words_9[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_9[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 5120));
            float _dot_41;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_41) : "r"(row_words_9[0]), "r"(row_words_9[0]), "f"(split_sum));
            split_sum = _dot_41;
            float _dot_42;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_42) : "r"(row_words_9[1]), "r"(row_words_9[1]), "f"(split_sum));
            split_sum = _dot_42;
            float _dot_43;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_43) : "r"(row_words_9[2]), "r"(row_words_9[2]), "f"(split_sum));
            split_sum = _dot_43;
            float _dot_44;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_44) : "r"(row_words_9[3]), "r"(row_words_9[3]), "f"(split_sum));
            split_sum = _dot_44;
            unsigned int row_words_10[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_10[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 5632));
            float _dot_45;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_45) : "r"(row_words_10[0]), "r"(row_words_10[0]), "f"(split_sum));
            split_sum = _dot_45;
            float _dot_46;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_46) : "r"(row_words_10[1]), "r"(row_words_10[1]), "f"(split_sum));
            split_sum = _dot_46;
            float _dot_47;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_47) : "r"(row_words_10[2]), "r"(row_words_10[2]), "f"(split_sum));
            split_sum = _dot_47;
            float _dot_48;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_48) : "r"(row_words_10[3]), "r"(row_words_10[3]), "f"(split_sum));
            split_sum = _dot_48;
            unsigned int row_words_11[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_11[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 6144));
            float _dot_49;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_49) : "r"(row_words_11[0]), "r"(row_words_11[0]), "f"(split_sum));
            split_sum = _dot_49;
            float _dot_50;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_50) : "r"(row_words_11[1]), "r"(row_words_11[1]), "f"(split_sum));
            split_sum = _dot_50;
            float _dot_51;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_51) : "r"(row_words_11[2]), "r"(row_words_11[2]), "f"(split_sum));
            split_sum = _dot_51;
            float _dot_52;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_52) : "r"(row_words_11[3]), "r"(row_words_11[3]), "f"(split_sum));
            split_sum = _dot_52;
            unsigned int row_words_12[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_12[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 6656));
            float _dot_53;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_53) : "r"(row_words_12[0]), "r"(row_words_12[0]), "f"(split_sum));
            split_sum = _dot_53;
            float _dot_54;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_54) : "r"(row_words_12[1]), "r"(row_words_12[1]), "f"(split_sum));
            split_sum = _dot_54;
            float _dot_55;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_55) : "r"(row_words_12[2]), "r"(row_words_12[2]), "f"(split_sum));
            split_sum = _dot_55;
            float _dot_56;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_56) : "r"(row_words_12[3]), "r"(row_words_12[3]), "f"(split_sum));
            split_sum = _dot_56;
            unsigned int row_words_13[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_13[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 7168));
            float _dot_57;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_57) : "r"(row_words_13[0]), "r"(row_words_13[0]), "f"(split_sum));
            split_sum = _dot_57;
            float _dot_58;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_58) : "r"(row_words_13[1]), "r"(row_words_13[1]), "f"(split_sum));
            split_sum = _dot_58;
            float _dot_59;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_59) : "r"(row_words_13[2]), "r"(row_words_13[2]), "f"(split_sum));
            split_sum = _dot_59;
            float _dot_60;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_60) : "r"(row_words_13[3]), "r"(row_words_13[3]), "f"(split_sum));
            split_sum = _dot_60;
            unsigned int row_words_14[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_14[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 7680));
            float _dot_61;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_61) : "r"(row_words_14[0]), "r"(row_words_14[0]), "f"(split_sum));
            split_sum = _dot_61;
            float _dot_62;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_62) : "r"(row_words_14[1]), "r"(row_words_14[1]), "f"(split_sum));
            split_sum = _dot_62;
            float _dot_63;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_63) : "r"(row_words_14[2]), "r"(row_words_14[2]), "f"(split_sum));
            split_sum = _dot_63;
            float _dot_64;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_64) : "r"(row_words_14[3]), "r"(row_words_14[3]), "f"(split_sum));
            split_sum = _dot_64;
            unsigned int row_words_15[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_15[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 8192));
            float _dot_65;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_65) : "r"(row_words_15[0]), "r"(row_words_15[0]), "f"(split_sum));
            split_sum = _dot_65;
            float _dot_66;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_66) : "r"(row_words_15[1]), "r"(row_words_15[1]), "f"(split_sum));
            split_sum = _dot_66;
            float _dot_67;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_67) : "r"(row_words_15[2]), "r"(row_words_15[2]), "f"(split_sum));
            split_sum = _dot_67;
            float _dot_68;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_68) : "r"(row_words_15[3]), "r"(row_words_15[3]), "f"(split_sum));
            split_sum = _dot_68;
            unsigned int row_words_16[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_16[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 8704));
            float _dot_69;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_69) : "r"(row_words_16[0]), "r"(row_words_16[0]), "f"(split_sum));
            split_sum = _dot_69;
            float _dot_70;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_70) : "r"(row_words_16[1]), "r"(row_words_16[1]), "f"(split_sum));
            split_sum = _dot_70;
            float _dot_71;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_71) : "r"(row_words_16[2]), "r"(row_words_16[2]), "f"(split_sum));
            split_sum = _dot_71;
            float _dot_72;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_72) : "r"(row_words_16[3]), "r"(row_words_16[3]), "f"(split_sum));
            split_sum = _dot_72;
            unsigned int row_words_17[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_17[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 9216));
            float _dot_73;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_73) : "r"(row_words_17[0]), "r"(row_words_17[0]), "f"(split_sum));
            split_sum = _dot_73;
            float _dot_74;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_74) : "r"(row_words_17[1]), "r"(row_words_17[1]), "f"(split_sum));
            split_sum = _dot_74;
            float _dot_75;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_75) : "r"(row_words_17[2]), "r"(row_words_17[2]), "f"(split_sum));
            split_sum = _dot_75;
            float _dot_76;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_76) : "r"(row_words_17[3]), "r"(row_words_17[3]), "f"(split_sum));
            split_sum = _dot_76;
            unsigned int row_words_18[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_18[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 9728));
            float _dot_77;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_77) : "r"(row_words_18[0]), "r"(row_words_18[0]), "f"(split_sum));
            split_sum = _dot_77;
            float _dot_78;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_78) : "r"(row_words_18[1]), "r"(row_words_18[1]), "f"(split_sum));
            split_sum = _dot_78;
            float _dot_79;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_79) : "r"(row_words_18[2]), "r"(row_words_18[2]), "f"(split_sum));
            split_sum = _dot_79;
            float _dot_80;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_80) : "r"(row_words_18[3]), "r"(row_words_18[3]), "f"(split_sum));
            split_sum = _dot_80;
            unsigned int row_words_19[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&row_words_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_words_19[(0) + 3]))
                : "r"(slot_addr + lane_bytes + 10240));
            float _dot_81;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_81) : "r"(row_words_19[0]), "r"(row_words_19[0]), "f"(split_sum));
            split_sum = _dot_81;
            float _dot_82;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_82) : "r"(row_words_19[1]), "r"(row_words_19[1]), "f"(split_sum));
            split_sum = _dot_82;
            float _dot_83;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_83) : "r"(row_words_19[2]), "r"(row_words_19[2]), "f"(split_sum));
            split_sum = _dot_83;
            float _dot_84;
            asm(
                "{\n\t"
                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                "}\n"
                : "=f"(_dot_84) : "r"(row_words_19[3]), "r"(row_words_19[3]), "f"(split_sum));
            split_sum = _dot_84;
            float split_total = split_sum;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                split_total += __shfl_xor_sync(0xFFFFFFFF, split_total, offset);
            float split_rstd = rsqrtf(split_total / 5376.0f + eps);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(rstd_at), "r"(__as_u32(split_rstd)) : "memory");
        }
        asm volatile("barrier.sync %0, 96;" :: "r"(group + 1) : "memory");
        unsigned int rstd_word[1];
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&rstd_word[0])) : "r"(rstd_at));
        rstd = __uint_as_float(rstd_word[0]);
        if (member == t % 3) {
            if (elect_sync()) {
                if (n_rows > t + 1) {
                    mbarrier_arrive_expect_tx(x_full_addr + (slot) * 8, 10752);
                    cp_async_bulk_gmem2smem(slot_addr, x + ((unsigned long long)(row + 8) * 5376), 10752, x_full_addr + (slot) * 8);
                }
            }
        }
        unsigned long long row_base = (unsigned long long)row * 5376;
        int m_tile = row / 128;
        int row_in_tile = row - m_tile * 128;
        unsigned long long sf_lane_base = (unsigned long long)m_tile * 84 * 512 + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4 + sf_lane_col;
        unsigned int* q_ptr = a_q + (row_base + (unsigned long long)own_col >> 3);
        if (table_row >= 0 && table_row < (long long)adaln_rows) {
            float2 rstd_pair = make_float2(rstd, rstd);
            unsigned long long table_base = (unsigned long long)table_row * (unsigned long long)adaln_row_stride;
            const __nv_bfloat16* s_ptr = adaln_scale + (table_base + (unsigned long long)own_col);
            const __nv_bfloat16* b_ptr = adaln_shift + (table_base + (unsigned long long)own_col);
            float w_vals[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals[(0) + 3]))
                : "r"(w_lane));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals[(4) + 3]))
                : "r"(w_lane + 10752));
            unsigned int _vec_load_1[4];
            {
                uint4 _uv4_85 = *reinterpret_cast<const uint4*>(s_ptr + 0 + 0);
                _vec_load_1[0 + 0] = _uv4_85.x;
                _vec_load_1[0 + 1] = _uv4_85.y;
                _vec_load_1[0 + 2] = _uv4_85.z;
                _vec_load_1[0 + 3] = _uv4_85.w;
            }
            unsigned int _vec_load_2[4];
            {
                uint4 _uv4_86 = *reinterpret_cast<const uint4*>(b_ptr + 0 + 0);
                _vec_load_2[0 + 0] = _uv4_86.x;
                _vec_load_2[0 + 1] = _uv4_86.y;
                _vec_load_2[0 + 2] = _uv4_86.z;
                _vec_load_2[0 + 3] = _uv4_86.w;
            }
            float _unpack_bf16_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_own[_pair]) << 16);
                (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_own[_pair]) & 0xffff0000u);
            }
            float _unpack_bf16_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_2[_pair]) << 16);
                (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_2[_pair]) & 0xffff0000u);
            }
            float prod[8];
            float2 _f32x2_mul_a_87 = make_float2(_unpack_bf16_1[0], _unpack_bf16_1[1]);
            float2 x_rstd;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_88 = make_float2(w_vals[0], w_vals[1]);
            float2 x_rstd_w;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&x_rstd));
            prod[0] = x_rstd_w.x;
            prod[1] = x_rstd_w.y;
            float2 _f32x2_mul_a_89 = make_float2(_unpack_bf16_1[2], _unpack_bf16_1[3]);
            float2 x_rstd_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_90 = make_float2(w_vals[2], w_vals[3]);
            float2 x_rstd_w_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_90), "l"(*(const unsigned long long*)&x_rstd_0));
            prod[2] = x_rstd_w_1.x;
            prod[3] = x_rstd_w_1.y;
            float2 _f32x2_mul_a_91 = make_float2(_unpack_bf16_1[4], _unpack_bf16_1[5]);
            float2 x_rstd_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_91), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_92 = make_float2(w_vals[4], w_vals[5]);
            float2 x_rstd_w_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&x_rstd_2));
            prod[4] = x_rstd_w_3.x;
            prod[5] = x_rstd_w_3.y;
            float2 _f32x2_mul_a_93 = make_float2(_unpack_bf16_1[6], _unpack_bf16_1[7]);
            float2 x_rstd_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_94 = make_float2(w_vals[6], w_vals[7]);
            float2 x_rstd_w_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&x_rstd_4));
            prod[6] = x_rstd_w_5.x;
            prod[7] = x_rstd_w_5.y;
            unsigned int prod_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod[_lp*2 + 0], prod[_lp*2+1 + 0]));
                prod_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float vals[8];
            uint32_t _bf16x2_add_0;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(_vec_load_1[0]), "r"(1065369472));
            unsigned int s_plus_one = _bf16x2_add_0;
            float adaln[2] = {_unpack_bf16_2[0], _unpack_bf16_2[1]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln[0]) : "h"(static_cast<uint16_t>((prod_bf16[0]))), "h"(static_cast<uint16_t>((s_plus_one))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln[1]) : "h"(static_cast<uint16_t>((prod_bf16[0]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one) >> 16)));
            float val_lo_pair[2];
            val_lo_pair[0] = 0.0f;
            val_lo_pair[1] = adaln[0];
            unsigned int val_lo_word[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair[_lp*2 + 0], val_lo_pair[_lp*2+1 + 0]));
                val_lo_word[_lp] = *(uint32_t*)&_bf2;
            }
            vals[0] = __uint_as_float(val_lo_word[0]);
            float val_hi_pair[2];
            val_hi_pair[0] = 0.0f;
            val_hi_pair[1] = adaln[1];
            unsigned int val_hi_word[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair[_lp*2 + 0], val_hi_pair[_lp*2+1 + 0]));
                val_hi_word[_lp] = *(uint32_t*)&_bf2;
            }
            vals[1] = __uint_as_float(val_hi_word[0]);
            uint32_t _bf16x2_add_1;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_1) : "r"(_vec_load_1[1]), "r"(1065369472));
            unsigned int s_plus_one_6 = _bf16x2_add_1;
            float adaln_7[2] = {_unpack_bf16_2[2], _unpack_bf16_2[3]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_7[0]) : "h"(static_cast<uint16_t>((prod_bf16[1]))), "h"(static_cast<uint16_t>((s_plus_one_6))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_7[1]) : "h"(static_cast<uint16_t>((prod_bf16[1]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_6) >> 16)));
            float val_lo_pair_8[2];
            val_lo_pair_8[0] = 0.0f;
            val_lo_pair_8[1] = adaln_7[0];
            unsigned int val_lo_word_9[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_8[_lp*2 + 0], val_lo_pair_8[_lp*2+1 + 0]));
                val_lo_word_9[_lp] = *(uint32_t*)&_bf2;
            }
            vals[2] = __uint_as_float(val_lo_word_9[0]);
            float val_hi_pair_10[2];
            val_hi_pair_10[0] = 0.0f;
            val_hi_pair_10[1] = adaln_7[1];
            unsigned int val_hi_word_11[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_10[_lp*2 + 0], val_hi_pair_10[_lp*2+1 + 0]));
                val_hi_word_11[_lp] = *(uint32_t*)&_bf2;
            }
            vals[3] = __uint_as_float(val_hi_word_11[0]);
            uint32_t _bf16x2_add_2;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_2) : "r"(_vec_load_1[2]), "r"(1065369472));
            unsigned int s_plus_one_12 = _bf16x2_add_2;
            float adaln_13[2] = {_unpack_bf16_2[4], _unpack_bf16_2[5]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_13[0]) : "h"(static_cast<uint16_t>((prod_bf16[2]))), "h"(static_cast<uint16_t>((s_plus_one_12))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_13[1]) : "h"(static_cast<uint16_t>((prod_bf16[2]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_12) >> 16)));
            float val_lo_pair_14[2];
            val_lo_pair_14[0] = 0.0f;
            val_lo_pair_14[1] = adaln_13[0];
            unsigned int val_lo_word_15[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_14[_lp*2 + 0], val_lo_pair_14[_lp*2+1 + 0]));
                val_lo_word_15[_lp] = *(uint32_t*)&_bf2;
            }
            vals[4] = __uint_as_float(val_lo_word_15[0]);
            float val_hi_pair_16[2];
            val_hi_pair_16[0] = 0.0f;
            val_hi_pair_16[1] = adaln_13[1];
            unsigned int val_hi_word_17[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_16[_lp*2 + 0], val_hi_pair_16[_lp*2+1 + 0]));
                val_hi_word_17[_lp] = *(uint32_t*)&_bf2;
            }
            vals[5] = __uint_as_float(val_hi_word_17[0]);
            uint32_t _bf16x2_add_3;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_3) : "r"(_vec_load_1[3]), "r"(1065369472));
            unsigned int s_plus_one_18 = _bf16x2_add_3;
            float adaln_19[2] = {_unpack_bf16_2[6], _unpack_bf16_2[7]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_19[0]) : "h"(static_cast<uint16_t>((prod_bf16[3]))), "h"(static_cast<uint16_t>((s_plus_one_18))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_19[1]) : "h"(static_cast<uint16_t>((prod_bf16[3]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_18) >> 16)));
            float val_lo_pair_20[2];
            val_lo_pair_20[0] = 0.0f;
            val_lo_pair_20[1] = adaln_19[0];
            unsigned int val_lo_word_21[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_20[_lp*2 + 0], val_lo_pair_20[_lp*2+1 + 0]));
                val_lo_word_21[_lp] = *(uint32_t*)&_bf2;
            }
            vals[6] = __uint_as_float(val_lo_word_21[0]);
            float val_hi_pair_22[2];
            val_hi_pair_22[0] = 0.0f;
            val_hi_pair_22[1] = adaln_19[1];
            unsigned int val_hi_word_23[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_22[_lp*2 + 0], val_hi_pair_22[_lp*2+1 + 0]));
                val_hi_word_23[_lp] = *(uint32_t*)&_bf2;
            }
            vals[7] = __uint_as_float(val_hi_word_23[0]);
            float mags[8];
            mags[0] = vals[0];
            mags[1] = vals[1];
            mags[2] = vals[2];
            mags[3] = vals[3];
            mags[4] = vals[4];
            mags[5] = vals[5];
            mags[6] = vals[6];
            mags[7] = vals[7];
            mags[0] = fabsf(mags[0]);
            mags[1] = fabsf(mags[1]);
            mags[2] = fabsf(mags[2]);
            mags[3] = fabsf(mags[3]);
            mags[4] = fabsf(mags[4]);
            mags[5] = fabsf(mags[5]);
            mags[6] = fabsf(mags[6]);
            mags[7] = fabsf(mags[7]);
            float absmax = mags[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                absmax = max_noftz(absmax, mags[_lr]);
            }
            float o1 = __shfl_xor_sync(0xFFFFFFFF, absmax, 1);
            absmax = fmaxf(absmax, o1);
            float sf_value = g * (absmax * sixth);
            uint16_t sf_pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair) : "f"(zero_f32), "f"(sf_value));
            unsigned int sf_byte = (unsigned int)sf_pair;
            uint32_t sf_f16x2;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16x2) : "h"((uint16_t)(sf_pair)));
            unsigned int sf_f16_lo = sf_f16x2 & 65535;
            float sf_f;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f) : "h"((uint16_t)(sf_f16_lo)));
            float out_scale_nonzero = approx_rcp(sf_f * inv_g);
            float out_scale = ((absmax == 0.0f) ? 0.0f : out_scale_nonzero);
            float2 out_scale_pair = make_float2(out_scale, out_scale);
            float scaled[8];
            float2 _f32x2_mul_a_95 = make_float2(vals[0], vals[1]);
            float2 q_pair;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&out_scale_pair));
            scaled[0] = q_pair.x;
            scaled[1] = q_pair.y;
            float2 _f32x2_mul_a_96 = make_float2(vals[2], vals[3]);
            float2 q_pair_24;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&out_scale_pair));
            scaled[2] = q_pair_24.x;
            scaled[3] = q_pair_24.y;
            float2 _f32x2_mul_a_97 = make_float2(vals[4], vals[5]);
            float2 q_pair_25;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_97), "l"(*(const unsigned long long*)&out_scale_pair));
            scaled[4] = q_pair_25.x;
            scaled[5] = q_pair_25.y;
            float2 _f32x2_mul_a_98 = make_float2(vals[6], vals[7]);
            float2 q_pair_26;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&out_scale_pair));
            scaled[6] = q_pair_26.x;
            scaled[7] = q_pair_26.y;
            unsigned int packed[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(scaled[0]), "f"(scaled[1]), "f"(scaled[2]), "f"(scaled[3]), "f"(scaled[4]), "f"(scaled[5]), "f"(scaled[6]), "f"(scaled[7]));
            {
                auto _gst_99 = q_ptr + 0;
                __builtin_assume(__isGlobal(_gst_99));
                *(reinterpret_cast<unsigned int*>(_gst_99) + (0)) = packed[0];
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + sf_lane_base) + (0)) = (unsigned char)(sf_byte);
            }
            float w_vals_27[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[(0) + 3]))
                : "r"(w_lane + 512));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_27[(4) + 3]))
                : "r"(w_lane + 10752 + 512));
            unsigned int _vec_load_3[4];
            {
                uint4 _uv4_100 = *reinterpret_cast<const uint4*>(s_ptr + 256 + 0);
                _vec_load_3[0 + 0] = _uv4_100.x;
                _vec_load_3[0 + 1] = _uv4_100.y;
                _vec_load_3[0 + 2] = _uv4_100.z;
                _vec_load_3[0 + 3] = _uv4_100.w;
            }
            unsigned int _vec_load_4[4];
            {
                uint4 _uv4_101 = *reinterpret_cast<const uint4*>(b_ptr + 256 + 0);
                _vec_load_4[0 + 0] = _uv4_101.x;
                _vec_load_4[0 + 1] = _uv4_101.y;
                _vec_load_4[0 + 2] = _uv4_101.z;
                _vec_load_4[0 + 3] = _uv4_101.w;
            }
            float _unpack_bf16_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_own[4 + _pair]) << 16);
                (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_own[4 + _pair]) & 0xffff0000u);
            }
            float _unpack_bf16_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_4[_pair]) << 16);
                (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_4[_pair]) & 0xffff0000u);
            }
            float prod_28[8];
            float2 _f32x2_mul_a_102 = make_float2(_unpack_bf16_3[0], _unpack_bf16_3[1]);
            float2 x_rstd_29;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_103 = make_float2(w_vals_27[0], w_vals_27[1]);
            float2 x_rstd_w_30;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_103), "l"(*(const unsigned long long*)&x_rstd_29));
            prod_28[0] = x_rstd_w_30.x;
            prod_28[1] = x_rstd_w_30.y;
            float2 _f32x2_mul_a_104 = make_float2(_unpack_bf16_3[2], _unpack_bf16_3[3]);
            float2 x_rstd_31;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_105 = make_float2(w_vals_27[2], w_vals_27[3]);
            float2 x_rstd_w_32;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_105), "l"(*(const unsigned long long*)&x_rstd_31));
            prod_28[2] = x_rstd_w_32.x;
            prod_28[3] = x_rstd_w_32.y;
            float2 _f32x2_mul_a_106 = make_float2(_unpack_bf16_3[4], _unpack_bf16_3[5]);
            float2 x_rstd_33;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_107 = make_float2(w_vals_27[4], w_vals_27[5]);
            float2 x_rstd_w_34;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&x_rstd_33));
            prod_28[4] = x_rstd_w_34.x;
            prod_28[5] = x_rstd_w_34.y;
            float2 _f32x2_mul_a_108 = make_float2(_unpack_bf16_3[6], _unpack_bf16_3[7]);
            float2 x_rstd_35;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_108), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_109 = make_float2(w_vals_27[6], w_vals_27[7]);
            float2 x_rstd_w_36;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&x_rstd_35));
            prod_28[6] = x_rstd_w_36.x;
            prod_28[7] = x_rstd_w_36.y;
            unsigned int prod_28_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_28[_lp*2 + 0], prod_28[_lp*2+1 + 0]));
                prod_28_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float vals_37[8];
            uint32_t _bf16x2_add_4;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_4) : "r"(_vec_load_3[0]), "r"(1065369472));
            unsigned int s_plus_one_38 = _bf16x2_add_4;
            float adaln_39[2] = {_unpack_bf16_4[0], _unpack_bf16_4[1]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_39[0]) : "h"(static_cast<uint16_t>((prod_28_bf16[0]))), "h"(static_cast<uint16_t>((s_plus_one_38))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_39[1]) : "h"(static_cast<uint16_t>((prod_28_bf16[0]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_38) >> 16)));
            float val_lo_pair_40[2];
            val_lo_pair_40[0] = 0.0f;
            val_lo_pair_40[1] = adaln_39[0];
            unsigned int val_lo_word_41[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_40[_lp*2 + 0], val_lo_pair_40[_lp*2+1 + 0]));
                val_lo_word_41[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[0] = __uint_as_float(val_lo_word_41[0]);
            float val_hi_pair_42[2];
            val_hi_pair_42[0] = 0.0f;
            val_hi_pair_42[1] = adaln_39[1];
            unsigned int val_hi_word_43[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_42[_lp*2 + 0], val_hi_pair_42[_lp*2+1 + 0]));
                val_hi_word_43[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[1] = __uint_as_float(val_hi_word_43[0]);
            uint32_t _bf16x2_add_5;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_5) : "r"(_vec_load_3[1]), "r"(1065369472));
            unsigned int s_plus_one_44 = _bf16x2_add_5;
            float adaln_45[2] = {_unpack_bf16_4[2], _unpack_bf16_4[3]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_45[0]) : "h"(static_cast<uint16_t>((prod_28_bf16[1]))), "h"(static_cast<uint16_t>((s_plus_one_44))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_45[1]) : "h"(static_cast<uint16_t>((prod_28_bf16[1]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_44) >> 16)));
            float val_lo_pair_46[2];
            val_lo_pair_46[0] = 0.0f;
            val_lo_pair_46[1] = adaln_45[0];
            unsigned int val_lo_word_47[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_46[_lp*2 + 0], val_lo_pair_46[_lp*2+1 + 0]));
                val_lo_word_47[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[2] = __uint_as_float(val_lo_word_47[0]);
            float val_hi_pair_48[2];
            val_hi_pair_48[0] = 0.0f;
            val_hi_pair_48[1] = adaln_45[1];
            unsigned int val_hi_word_49[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_48[_lp*2 + 0], val_hi_pair_48[_lp*2+1 + 0]));
                val_hi_word_49[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[3] = __uint_as_float(val_hi_word_49[0]);
            uint32_t _bf16x2_add_6;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_6) : "r"(_vec_load_3[2]), "r"(1065369472));
            unsigned int s_plus_one_50 = _bf16x2_add_6;
            float adaln_51[2] = {_unpack_bf16_4[4], _unpack_bf16_4[5]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_51[0]) : "h"(static_cast<uint16_t>((prod_28_bf16[2]))), "h"(static_cast<uint16_t>((s_plus_one_50))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_51[1]) : "h"(static_cast<uint16_t>((prod_28_bf16[2]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_50) >> 16)));
            float val_lo_pair_52[2];
            val_lo_pair_52[0] = 0.0f;
            val_lo_pair_52[1] = adaln_51[0];
            unsigned int val_lo_word_53[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_52[_lp*2 + 0], val_lo_pair_52[_lp*2+1 + 0]));
                val_lo_word_53[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[4] = __uint_as_float(val_lo_word_53[0]);
            float val_hi_pair_54[2];
            val_hi_pair_54[0] = 0.0f;
            val_hi_pair_54[1] = adaln_51[1];
            unsigned int val_hi_word_55[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_54[_lp*2 + 0], val_hi_pair_54[_lp*2+1 + 0]));
                val_hi_word_55[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[5] = __uint_as_float(val_hi_word_55[0]);
            uint32_t _bf16x2_add_7;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_7) : "r"(_vec_load_3[3]), "r"(1065369472));
            unsigned int s_plus_one_56 = _bf16x2_add_7;
            float adaln_57[2] = {_unpack_bf16_4[6], _unpack_bf16_4[7]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_57[0]) : "h"(static_cast<uint16_t>((prod_28_bf16[3]))), "h"(static_cast<uint16_t>((s_plus_one_56))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_57[1]) : "h"(static_cast<uint16_t>((prod_28_bf16[3]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_56) >> 16)));
            float val_lo_pair_58[2];
            val_lo_pair_58[0] = 0.0f;
            val_lo_pair_58[1] = adaln_57[0];
            unsigned int val_lo_word_59[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_58[_lp*2 + 0], val_lo_pair_58[_lp*2+1 + 0]));
                val_lo_word_59[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[6] = __uint_as_float(val_lo_word_59[0]);
            float val_hi_pair_60[2];
            val_hi_pair_60[0] = 0.0f;
            val_hi_pair_60[1] = adaln_57[1];
            unsigned int val_hi_word_61[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_60[_lp*2 + 0], val_hi_pair_60[_lp*2+1 + 0]));
                val_hi_word_61[_lp] = *(uint32_t*)&_bf2;
            }
            vals_37[7] = __uint_as_float(val_hi_word_61[0]);
            float mags_62[8];
            mags_62[0] = vals_37[0];
            mags_62[1] = vals_37[1];
            mags_62[2] = vals_37[2];
            mags_62[3] = vals_37[3];
            mags_62[4] = vals_37[4];
            mags_62[5] = vals_37[5];
            mags_62[6] = vals_37[6];
            mags_62[7] = vals_37[7];
            mags_62[0] = fabsf(mags_62[0]);
            mags_62[1] = fabsf(mags_62[1]);
            mags_62[2] = fabsf(mags_62[2]);
            mags_62[3] = fabsf(mags_62[3]);
            mags_62[4] = fabsf(mags_62[4]);
            mags_62[5] = fabsf(mags_62[5]);
            mags_62[6] = fabsf(mags_62[6]);
            mags_62[7] = fabsf(mags_62[7]);
            float absmax_63 = mags_62[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                absmax_63 = max_noftz(absmax_63, mags_62[_lr]);
            }
            float o1_64 = __shfl_xor_sync(0xFFFFFFFF, absmax_63, 1);
            absmax_63 = fmaxf(absmax_63, o1_64);
            float sf_value_65 = g * (absmax_63 * sixth);
            uint16_t sf_pair_66;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_66) : "f"(zero_f32), "f"(sf_value_65));
            unsigned int sf_byte_67 = (unsigned int)sf_pair_66;
            uint32_t sf_f16x2_68;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16x2_68) : "h"((uint16_t)(sf_pair_66)));
            unsigned int sf_f16_lo_69 = sf_f16x2_68 & 65535;
            float sf_f_70;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f_70) : "h"((uint16_t)(sf_f16_lo_69)));
            float out_scale_nonzero_71 = approx_rcp(sf_f_70 * inv_g);
            float out_scale_72 = ((absmax_63 == 0.0f) ? 0.0f : out_scale_nonzero_71);
            float2 out_scale_pair_73 = make_float2(out_scale_72, out_scale_72);
            float scaled_74[8];
            float2 _f32x2_mul_a_110 = make_float2(vals_37[0], vals_37[1]);
            float2 q_pair_75;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_75) : "l"(*(const unsigned long long*)&_f32x2_mul_a_110), "l"(*(const unsigned long long*)&out_scale_pair_73));
            scaled_74[0] = q_pair_75.x;
            scaled_74[1] = q_pair_75.y;
            float2 _f32x2_mul_a_111 = make_float2(vals_37[2], vals_37[3]);
            float2 q_pair_76;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_76) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&out_scale_pair_73));
            scaled_74[2] = q_pair_76.x;
            scaled_74[3] = q_pair_76.y;
            float2 _f32x2_mul_a_112 = make_float2(vals_37[4], vals_37[5]);
            float2 q_pair_77;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_77) : "l"(*(const unsigned long long*)&_f32x2_mul_a_112), "l"(*(const unsigned long long*)&out_scale_pair_73));
            scaled_74[4] = q_pair_77.x;
            scaled_74[5] = q_pair_77.y;
            float2 _f32x2_mul_a_113 = make_float2(vals_37[6], vals_37[7]);
            float2 q_pair_78;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_78) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&out_scale_pair_73));
            scaled_74[6] = q_pair_78.x;
            scaled_74[7] = q_pair_78.y;
            unsigned int packed_79[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_79[0]) : "f"(scaled_74[0]), "f"(scaled_74[1]), "f"(scaled_74[2]), "f"(scaled_74[3]), "f"(scaled_74[4]), "f"(scaled_74[5]), "f"(scaled_74[6]), "f"(scaled_74[7]));
            {
                auto _gst_114 = q_ptr + 32;
                __builtin_assume(__isGlobal(_gst_114));
                *(reinterpret_cast<unsigned int*>(_gst_114) + (0)) = packed_79[0];
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 2048)) + (0)) = (unsigned char)(sf_byte_67);
            }
            float w_vals_80[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[(0) + 3]))
                : "r"(w_lane + 1024));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_80[(4) + 3]))
                : "r"(w_lane + 10752 + 1024));
            unsigned int _vec_load_5[4];
            {
                uint4 _uv4_115 = *reinterpret_cast<const uint4*>(s_ptr + 512 + 0);
                _vec_load_5[0 + 0] = _uv4_115.x;
                _vec_load_5[0 + 1] = _uv4_115.y;
                _vec_load_5[0 + 2] = _uv4_115.z;
                _vec_load_5[0 + 3] = _uv4_115.w;
            }
            unsigned int _vec_load_6[4];
            {
                uint4 _uv4_116 = *reinterpret_cast<const uint4*>(b_ptr + 512 + 0);
                _vec_load_6[0 + 0] = _uv4_116.x;
                _vec_load_6[0 + 1] = _uv4_116.y;
                _vec_load_6[0 + 2] = _uv4_116.z;
                _vec_load_6[0 + 3] = _uv4_116.w;
            }
            float _unpack_bf16_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_own[8 + _pair]) << 16);
                (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_own[8 + _pair]) & 0xffff0000u);
            }
            float _unpack_bf16_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_6[_pair]) << 16);
                (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_6[_pair]) & 0xffff0000u);
            }
            float prod_81[8];
            float2 _f32x2_mul_a_117 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
            float2 x_rstd_82;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_82) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_118 = make_float2(w_vals_80[0], w_vals_80[1]);
            float2 x_rstd_w_83;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_83) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&x_rstd_82));
            prod_81[0] = x_rstd_w_83.x;
            prod_81[1] = x_rstd_w_83.y;
            float2 _f32x2_mul_a_119 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
            float2 x_rstd_84;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_84) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_120 = make_float2(w_vals_80[2], w_vals_80[3]);
            float2 x_rstd_w_85;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_85) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&x_rstd_84));
            prod_81[2] = x_rstd_w_85.x;
            prod_81[3] = x_rstd_w_85.y;
            float2 _f32x2_mul_a_121 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
            float2 x_rstd_86;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_86) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_122 = make_float2(w_vals_80[4], w_vals_80[5]);
            float2 x_rstd_w_87;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_87) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&x_rstd_86));
            prod_81[4] = x_rstd_w_87.x;
            prod_81[5] = x_rstd_w_87.y;
            float2 _f32x2_mul_a_123 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
            float2 x_rstd_88;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_88) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_124 = make_float2(w_vals_80[6], w_vals_80[7]);
            float2 x_rstd_w_89;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_89) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&x_rstd_88));
            prod_81[6] = x_rstd_w_89.x;
            prod_81[7] = x_rstd_w_89.y;
            unsigned int prod_81_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_81[_lp*2 + 0], prod_81[_lp*2+1 + 0]));
                prod_81_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float vals_90[8];
            uint32_t _bf16x2_add_8;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_8) : "r"(_vec_load_5[0]), "r"(1065369472));
            unsigned int s_plus_one_91 = _bf16x2_add_8;
            float adaln_92[2] = {_unpack_bf16_6[0], _unpack_bf16_6[1]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_92[0]) : "h"(static_cast<uint16_t>((prod_81_bf16[0]))), "h"(static_cast<uint16_t>((s_plus_one_91))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_92[1]) : "h"(static_cast<uint16_t>((prod_81_bf16[0]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_91) >> 16)));
            float val_lo_pair_93[2];
            val_lo_pair_93[0] = 0.0f;
            val_lo_pair_93[1] = adaln_92[0];
            unsigned int val_lo_word_94[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_93[_lp*2 + 0], val_lo_pair_93[_lp*2+1 + 0]));
                val_lo_word_94[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[0] = __uint_as_float(val_lo_word_94[0]);
            float val_hi_pair_95[2];
            val_hi_pair_95[0] = 0.0f;
            val_hi_pair_95[1] = adaln_92[1];
            unsigned int val_hi_word_96[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_95[_lp*2 + 0], val_hi_pair_95[_lp*2+1 + 0]));
                val_hi_word_96[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[1] = __uint_as_float(val_hi_word_96[0]);
            uint32_t _bf16x2_add_9;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_9) : "r"(_vec_load_5[1]), "r"(1065369472));
            unsigned int s_plus_one_97 = _bf16x2_add_9;
            float adaln_98[2] = {_unpack_bf16_6[2], _unpack_bf16_6[3]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_98[0]) : "h"(static_cast<uint16_t>((prod_81_bf16[1]))), "h"(static_cast<uint16_t>((s_plus_one_97))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_98[1]) : "h"(static_cast<uint16_t>((prod_81_bf16[1]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_97) >> 16)));
            float val_lo_pair_99[2];
            val_lo_pair_99[0] = 0.0f;
            val_lo_pair_99[1] = adaln_98[0];
            unsigned int val_lo_word_100[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_99[_lp*2 + 0], val_lo_pair_99[_lp*2+1 + 0]));
                val_lo_word_100[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[2] = __uint_as_float(val_lo_word_100[0]);
            float val_hi_pair_101[2];
            val_hi_pair_101[0] = 0.0f;
            val_hi_pair_101[1] = adaln_98[1];
            unsigned int val_hi_word_102[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_101[_lp*2 + 0], val_hi_pair_101[_lp*2+1 + 0]));
                val_hi_word_102[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[3] = __uint_as_float(val_hi_word_102[0]);
            uint32_t _bf16x2_add_10;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_10) : "r"(_vec_load_5[2]), "r"(1065369472));
            unsigned int s_plus_one_103 = _bf16x2_add_10;
            float adaln_104[2] = {_unpack_bf16_6[4], _unpack_bf16_6[5]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_104[0]) : "h"(static_cast<uint16_t>((prod_81_bf16[2]))), "h"(static_cast<uint16_t>((s_plus_one_103))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_104[1]) : "h"(static_cast<uint16_t>((prod_81_bf16[2]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_103) >> 16)));
            float val_lo_pair_105[2];
            val_lo_pair_105[0] = 0.0f;
            val_lo_pair_105[1] = adaln_104[0];
            unsigned int val_lo_word_106[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_105[_lp*2 + 0], val_lo_pair_105[_lp*2+1 + 0]));
                val_lo_word_106[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[4] = __uint_as_float(val_lo_word_106[0]);
            float val_hi_pair_107[2];
            val_hi_pair_107[0] = 0.0f;
            val_hi_pair_107[1] = adaln_104[1];
            unsigned int val_hi_word_108[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_107[_lp*2 + 0], val_hi_pair_107[_lp*2+1 + 0]));
                val_hi_word_108[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[5] = __uint_as_float(val_hi_word_108[0]);
            uint32_t _bf16x2_add_11;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_11) : "r"(_vec_load_5[3]), "r"(1065369472));
            unsigned int s_plus_one_109 = _bf16x2_add_11;
            float adaln_110[2] = {_unpack_bf16_6[6], _unpack_bf16_6[7]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_110[0]) : "h"(static_cast<uint16_t>((prod_81_bf16[3]))), "h"(static_cast<uint16_t>((s_plus_one_109))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_110[1]) : "h"(static_cast<uint16_t>((prod_81_bf16[3]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_109) >> 16)));
            float val_lo_pair_111[2];
            val_lo_pair_111[0] = 0.0f;
            val_lo_pair_111[1] = adaln_110[0];
            unsigned int val_lo_word_112[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_111[_lp*2 + 0], val_lo_pair_111[_lp*2+1 + 0]));
                val_lo_word_112[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[6] = __uint_as_float(val_lo_word_112[0]);
            float val_hi_pair_113[2];
            val_hi_pair_113[0] = 0.0f;
            val_hi_pair_113[1] = adaln_110[1];
            unsigned int val_hi_word_114[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_113[_lp*2 + 0], val_hi_pair_113[_lp*2+1 + 0]));
                val_hi_word_114[_lp] = *(uint32_t*)&_bf2;
            }
            vals_90[7] = __uint_as_float(val_hi_word_114[0]);
            float mags_115[8];
            mags_115[0] = vals_90[0];
            mags_115[1] = vals_90[1];
            mags_115[2] = vals_90[2];
            mags_115[3] = vals_90[3];
            mags_115[4] = vals_90[4];
            mags_115[5] = vals_90[5];
            mags_115[6] = vals_90[6];
            mags_115[7] = vals_90[7];
            mags_115[0] = fabsf(mags_115[0]);
            mags_115[1] = fabsf(mags_115[1]);
            mags_115[2] = fabsf(mags_115[2]);
            mags_115[3] = fabsf(mags_115[3]);
            mags_115[4] = fabsf(mags_115[4]);
            mags_115[5] = fabsf(mags_115[5]);
            mags_115[6] = fabsf(mags_115[6]);
            mags_115[7] = fabsf(mags_115[7]);
            float absmax_116 = mags_115[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                absmax_116 = max_noftz(absmax_116, mags_115[_lr]);
            }
            float o1_117 = __shfl_xor_sync(0xFFFFFFFF, absmax_116, 1);
            absmax_116 = fmaxf(absmax_116, o1_117);
            float sf_value_118 = g * (absmax_116 * sixth);
            uint16_t sf_pair_119;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_119) : "f"(zero_f32), "f"(sf_value_118));
            unsigned int sf_byte_120 = (unsigned int)sf_pair_119;
            uint32_t sf_f16x2_121;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16x2_121) : "h"((uint16_t)(sf_pair_119)));
            unsigned int sf_f16_lo_122 = sf_f16x2_121 & 65535;
            float sf_f_123;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f_123) : "h"((uint16_t)(sf_f16_lo_122)));
            float out_scale_nonzero_124 = approx_rcp(sf_f_123 * inv_g);
            float out_scale_125 = ((absmax_116 == 0.0f) ? 0.0f : out_scale_nonzero_124);
            float2 out_scale_pair_126 = make_float2(out_scale_125, out_scale_125);
            float scaled_127[8];
            float2 _f32x2_mul_a_125 = make_float2(vals_90[0], vals_90[1]);
            float2 q_pair_128;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_128) : "l"(*(const unsigned long long*)&_f32x2_mul_a_125), "l"(*(const unsigned long long*)&out_scale_pair_126));
            scaled_127[0] = q_pair_128.x;
            scaled_127[1] = q_pair_128.y;
            float2 _f32x2_mul_a_126 = make_float2(vals_90[2], vals_90[3]);
            float2 q_pair_129;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_129) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&out_scale_pair_126));
            scaled_127[2] = q_pair_129.x;
            scaled_127[3] = q_pair_129.y;
            float2 _f32x2_mul_a_127 = make_float2(vals_90[4], vals_90[5]);
            float2 q_pair_130;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_130) : "l"(*(const unsigned long long*)&_f32x2_mul_a_127), "l"(*(const unsigned long long*)&out_scale_pair_126));
            scaled_127[4] = q_pair_130.x;
            scaled_127[5] = q_pair_130.y;
            float2 _f32x2_mul_a_128 = make_float2(vals_90[6], vals_90[7]);
            float2 q_pair_131;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_131) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&out_scale_pair_126));
            scaled_127[6] = q_pair_131.x;
            scaled_127[7] = q_pair_131.y;
            unsigned int packed_132[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_132[0]) : "f"(scaled_127[0]), "f"(scaled_127[1]), "f"(scaled_127[2]), "f"(scaled_127[3]), "f"(scaled_127[4]), "f"(scaled_127[5]), "f"(scaled_127[6]), "f"(scaled_127[7]));
            {
                auto _gst_129 = q_ptr + 64;
                __builtin_assume(__isGlobal(_gst_129));
                *(reinterpret_cast<unsigned int*>(_gst_129) + (0)) = packed_132[0];
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 4096)) + (0)) = (unsigned char)(sf_byte_120);
            }
            float w_vals_133[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[(0) + 3]))
                : "r"(w_lane + 1536));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_133[(4) + 3]))
                : "r"(w_lane + 10752 + 1536));
            unsigned int _vec_load_7[4];
            {
                uint4 _uv4_130 = *reinterpret_cast<const uint4*>(s_ptr + 768 + 0);
                _vec_load_7[0 + 0] = _uv4_130.x;
                _vec_load_7[0 + 1] = _uv4_130.y;
                _vec_load_7[0 + 2] = _uv4_130.z;
                _vec_load_7[0 + 3] = _uv4_130.w;
            }
            unsigned int _vec_load_8[4];
            {
                uint4 _uv4_131 = *reinterpret_cast<const uint4*>(b_ptr + 768 + 0);
                _vec_load_8[0 + 0] = _uv4_131.x;
                _vec_load_8[0 + 1] = _uv4_131.y;
                _vec_load_8[0 + 2] = _uv4_131.z;
                _vec_load_8[0 + 3] = _uv4_131.w;
            }
            float _unpack_bf16_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_own[12 + _pair]) << 16);
                (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_own[12 + _pair]) & 0xffff0000u);
            }
            float _unpack_bf16_8[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_8[_pair]) << 16);
                (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_8[_pair]) & 0xffff0000u);
            }
            float prod_134[8];
            float2 _f32x2_mul_a_132 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
            float2 x_rstd_135;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_135) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_133 = make_float2(w_vals_133[0], w_vals_133[1]);
            float2 x_rstd_w_136;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_136) : "l"(*(const unsigned long long*)&_f32x2_mul_a_133), "l"(*(const unsigned long long*)&x_rstd_135));
            prod_134[0] = x_rstd_w_136.x;
            prod_134[1] = x_rstd_w_136.y;
            float2 _f32x2_mul_a_134 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
            float2 x_rstd_137;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_137) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_135 = make_float2(w_vals_133[2], w_vals_133[3]);
            float2 x_rstd_w_138;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_138) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&x_rstd_137));
            prod_134[2] = x_rstd_w_138.x;
            prod_134[3] = x_rstd_w_138.y;
            float2 _f32x2_mul_a_136 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
            float2 x_rstd_139;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_139) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_137 = make_float2(w_vals_133[4], w_vals_133[5]);
            float2 x_rstd_w_140;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_140) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&x_rstd_139));
            prod_134[4] = x_rstd_w_140.x;
            prod_134[5] = x_rstd_w_140.y;
            float2 _f32x2_mul_a_138 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
            float2 x_rstd_141;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_141) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_139 = make_float2(w_vals_133[6], w_vals_133[7]);
            float2 x_rstd_w_142;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_142) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&x_rstd_141));
            prod_134[6] = x_rstd_w_142.x;
            prod_134[7] = x_rstd_w_142.y;
            unsigned int prod_134_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_134[_lp*2 + 0], prod_134[_lp*2+1 + 0]));
                prod_134_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float vals_143[8];
            uint32_t _bf16x2_add_12;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_12) : "r"(_vec_load_7[0]), "r"(1065369472));
            unsigned int s_plus_one_144 = _bf16x2_add_12;
            float adaln_145[2] = {_unpack_bf16_8[0], _unpack_bf16_8[1]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_145[0]) : "h"(static_cast<uint16_t>((prod_134_bf16[0]))), "h"(static_cast<uint16_t>((s_plus_one_144))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_145[1]) : "h"(static_cast<uint16_t>((prod_134_bf16[0]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_144) >> 16)));
            float val_lo_pair_146[2];
            val_lo_pair_146[0] = 0.0f;
            val_lo_pair_146[1] = adaln_145[0];
            unsigned int val_lo_word_147[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_146[_lp*2 + 0], val_lo_pair_146[_lp*2+1 + 0]));
                val_lo_word_147[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[0] = __uint_as_float(val_lo_word_147[0]);
            float val_hi_pair_148[2];
            val_hi_pair_148[0] = 0.0f;
            val_hi_pair_148[1] = adaln_145[1];
            unsigned int val_hi_word_149[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_148[_lp*2 + 0], val_hi_pair_148[_lp*2+1 + 0]));
                val_hi_word_149[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[1] = __uint_as_float(val_hi_word_149[0]);
            uint32_t _bf16x2_add_13;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_13) : "r"(_vec_load_7[1]), "r"(1065369472));
            unsigned int s_plus_one_150 = _bf16x2_add_13;
            float adaln_151[2] = {_unpack_bf16_8[2], _unpack_bf16_8[3]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_151[0]) : "h"(static_cast<uint16_t>((prod_134_bf16[1]))), "h"(static_cast<uint16_t>((s_plus_one_150))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_151[1]) : "h"(static_cast<uint16_t>((prod_134_bf16[1]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_150) >> 16)));
            float val_lo_pair_152[2];
            val_lo_pair_152[0] = 0.0f;
            val_lo_pair_152[1] = adaln_151[0];
            unsigned int val_lo_word_153[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_152[_lp*2 + 0], val_lo_pair_152[_lp*2+1 + 0]));
                val_lo_word_153[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[2] = __uint_as_float(val_lo_word_153[0]);
            float val_hi_pair_154[2];
            val_hi_pair_154[0] = 0.0f;
            val_hi_pair_154[1] = adaln_151[1];
            unsigned int val_hi_word_155[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_154[_lp*2 + 0], val_hi_pair_154[_lp*2+1 + 0]));
                val_hi_word_155[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[3] = __uint_as_float(val_hi_word_155[0]);
            uint32_t _bf16x2_add_14;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_14) : "r"(_vec_load_7[2]), "r"(1065369472));
            unsigned int s_plus_one_156 = _bf16x2_add_14;
            float adaln_157[2] = {_unpack_bf16_8[4], _unpack_bf16_8[5]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_157[0]) : "h"(static_cast<uint16_t>((prod_134_bf16[2]))), "h"(static_cast<uint16_t>((s_plus_one_156))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_157[1]) : "h"(static_cast<uint16_t>((prod_134_bf16[2]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_156) >> 16)));
            float val_lo_pair_158[2];
            val_lo_pair_158[0] = 0.0f;
            val_lo_pair_158[1] = adaln_157[0];
            unsigned int val_lo_word_159[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_158[_lp*2 + 0], val_lo_pair_158[_lp*2+1 + 0]));
                val_lo_word_159[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[4] = __uint_as_float(val_lo_word_159[0]);
            float val_hi_pair_160[2];
            val_hi_pair_160[0] = 0.0f;
            val_hi_pair_160[1] = adaln_157[1];
            unsigned int val_hi_word_161[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_160[_lp*2 + 0], val_hi_pair_160[_lp*2+1 + 0]));
                val_hi_word_161[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[5] = __uint_as_float(val_hi_word_161[0]);
            uint32_t _bf16x2_add_15;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_15) : "r"(_vec_load_7[3]), "r"(1065369472));
            unsigned int s_plus_one_162 = _bf16x2_add_15;
            float adaln_163[2] = {_unpack_bf16_8[6], _unpack_bf16_8[7]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_163[0]) : "h"(static_cast<uint16_t>((prod_134_bf16[3]))), "h"(static_cast<uint16_t>((s_plus_one_162))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_163[1]) : "h"(static_cast<uint16_t>((prod_134_bf16[3]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_162) >> 16)));
            float val_lo_pair_164[2];
            val_lo_pair_164[0] = 0.0f;
            val_lo_pair_164[1] = adaln_163[0];
            unsigned int val_lo_word_165[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_164[_lp*2 + 0], val_lo_pair_164[_lp*2+1 + 0]));
                val_lo_word_165[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[6] = __uint_as_float(val_lo_word_165[0]);
            float val_hi_pair_166[2];
            val_hi_pair_166[0] = 0.0f;
            val_hi_pair_166[1] = adaln_163[1];
            unsigned int val_hi_word_167[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_166[_lp*2 + 0], val_hi_pair_166[_lp*2+1 + 0]));
                val_hi_word_167[_lp] = *(uint32_t*)&_bf2;
            }
            vals_143[7] = __uint_as_float(val_hi_word_167[0]);
            float mags_168[8];
            mags_168[0] = vals_143[0];
            mags_168[1] = vals_143[1];
            mags_168[2] = vals_143[2];
            mags_168[3] = vals_143[3];
            mags_168[4] = vals_143[4];
            mags_168[5] = vals_143[5];
            mags_168[6] = vals_143[6];
            mags_168[7] = vals_143[7];
            mags_168[0] = fabsf(mags_168[0]);
            mags_168[1] = fabsf(mags_168[1]);
            mags_168[2] = fabsf(mags_168[2]);
            mags_168[3] = fabsf(mags_168[3]);
            mags_168[4] = fabsf(mags_168[4]);
            mags_168[5] = fabsf(mags_168[5]);
            mags_168[6] = fabsf(mags_168[6]);
            mags_168[7] = fabsf(mags_168[7]);
            float absmax_169 = mags_168[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                absmax_169 = max_noftz(absmax_169, mags_168[_lr]);
            }
            float o1_170 = __shfl_xor_sync(0xFFFFFFFF, absmax_169, 1);
            absmax_169 = fmaxf(absmax_169, o1_170);
            float sf_value_171 = g * (absmax_169 * sixth);
            uint16_t sf_pair_172;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_172) : "f"(zero_f32), "f"(sf_value_171));
            unsigned int sf_byte_173 = (unsigned int)sf_pair_172;
            uint32_t sf_f16x2_174;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16x2_174) : "h"((uint16_t)(sf_pair_172)));
            unsigned int sf_f16_lo_175 = sf_f16x2_174 & 65535;
            float sf_f_176;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f_176) : "h"((uint16_t)(sf_f16_lo_175)));
            float out_scale_nonzero_177 = approx_rcp(sf_f_176 * inv_g);
            float out_scale_178 = ((absmax_169 == 0.0f) ? 0.0f : out_scale_nonzero_177);
            float2 out_scale_pair_179 = make_float2(out_scale_178, out_scale_178);
            float scaled_180[8];
            float2 _f32x2_mul_a_140 = make_float2(vals_143[0], vals_143[1]);
            float2 q_pair_181;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_181) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&out_scale_pair_179));
            scaled_180[0] = q_pair_181.x;
            scaled_180[1] = q_pair_181.y;
            float2 _f32x2_mul_a_141 = make_float2(vals_143[2], vals_143[3]);
            float2 q_pair_182;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_182) : "l"(*(const unsigned long long*)&_f32x2_mul_a_141), "l"(*(const unsigned long long*)&out_scale_pair_179));
            scaled_180[2] = q_pair_182.x;
            scaled_180[3] = q_pair_182.y;
            float2 _f32x2_mul_a_142 = make_float2(vals_143[4], vals_143[5]);
            float2 q_pair_183;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_183) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&out_scale_pair_179));
            scaled_180[4] = q_pair_183.x;
            scaled_180[5] = q_pair_183.y;
            float2 _f32x2_mul_a_143 = make_float2(vals_143[6], vals_143[7]);
            float2 q_pair_184;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_184) : "l"(*(const unsigned long long*)&_f32x2_mul_a_143), "l"(*(const unsigned long long*)&out_scale_pair_179));
            scaled_180[6] = q_pair_184.x;
            scaled_180[7] = q_pair_184.y;
            unsigned int packed_185[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_185[0]) : "f"(scaled_180[0]), "f"(scaled_180[1]), "f"(scaled_180[2]), "f"(scaled_180[3]), "f"(scaled_180[4]), "f"(scaled_180[5]), "f"(scaled_180[6]), "f"(scaled_180[7]));
            {
                auto _gst_144 = q_ptr + 96;
                __builtin_assume(__isGlobal(_gst_144));
                *(reinterpret_cast<unsigned int*>(_gst_144) + (0)) = packed_185[0];
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 6144)) + (0)) = (unsigned char)(sf_byte_173);
            }
            float w_vals_186[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[(0) + 3]))
                : "r"(w_lane + 2048));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_186[(4) + 3]))
                : "r"(w_lane + 10752 + 2048));
            unsigned int _vec_load_9[4];
            {
                uint4 _uv4_145 = *reinterpret_cast<const uint4*>(s_ptr + 1024 + 0);
                _vec_load_9[0 + 0] = _uv4_145.x;
                _vec_load_9[0 + 1] = _uv4_145.y;
                _vec_load_9[0 + 2] = _uv4_145.z;
                _vec_load_9[0 + 3] = _uv4_145.w;
            }
            unsigned int _vec_load_10[4];
            {
                uint4 _uv4_146 = *reinterpret_cast<const uint4*>(b_ptr + 1024 + 0);
                _vec_load_10[0 + 0] = _uv4_146.x;
                _vec_load_10[0 + 1] = _uv4_146.y;
                _vec_load_10[0 + 2] = _uv4_146.z;
                _vec_load_10[0 + 3] = _uv4_146.w;
            }
            float _unpack_bf16_9[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_own[16 + _pair]) << 16);
                (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_own[16 + _pair]) & 0xffff0000u);
            }
            float _unpack_bf16_10[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_10[_pair]) << 16);
                (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_10[_pair]) & 0xffff0000u);
            }
            float prod_187[8];
            float2 _f32x2_mul_a_147 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
            float2 x_rstd_188;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_188) : "l"(*(const unsigned long long*)&_f32x2_mul_a_147), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_148 = make_float2(w_vals_186[0], w_vals_186[1]);
            float2 x_rstd_w_189;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_189) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&x_rstd_188));
            prod_187[0] = x_rstd_w_189.x;
            prod_187[1] = x_rstd_w_189.y;
            float2 _f32x2_mul_a_149 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
            float2 x_rstd_190;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_190) : "l"(*(const unsigned long long*)&_f32x2_mul_a_149), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_150 = make_float2(w_vals_186[2], w_vals_186[3]);
            float2 x_rstd_w_191;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_191) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&x_rstd_190));
            prod_187[2] = x_rstd_w_191.x;
            prod_187[3] = x_rstd_w_191.y;
            float2 _f32x2_mul_a_151 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
            float2 x_rstd_192;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_192) : "l"(*(const unsigned long long*)&_f32x2_mul_a_151), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_152 = make_float2(w_vals_186[4], w_vals_186[5]);
            float2 x_rstd_w_193;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_193) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&x_rstd_192));
            prod_187[4] = x_rstd_w_193.x;
            prod_187[5] = x_rstd_w_193.y;
            float2 _f32x2_mul_a_153 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
            float2 x_rstd_194;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_194) : "l"(*(const unsigned long long*)&_f32x2_mul_a_153), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_154 = make_float2(w_vals_186[6], w_vals_186[7]);
            float2 x_rstd_w_195;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_195) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&x_rstd_194));
            prod_187[6] = x_rstd_w_195.x;
            prod_187[7] = x_rstd_w_195.y;
            unsigned int prod_187_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_187[_lp*2 + 0], prod_187[_lp*2+1 + 0]));
                prod_187_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float vals_196[8];
            uint32_t _bf16x2_add_16;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_16) : "r"(_vec_load_9[0]), "r"(1065369472));
            unsigned int s_plus_one_197 = _bf16x2_add_16;
            float adaln_198[2] = {_unpack_bf16_10[0], _unpack_bf16_10[1]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_198[0]) : "h"(static_cast<uint16_t>((prod_187_bf16[0]))), "h"(static_cast<uint16_t>((s_plus_one_197))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_198[1]) : "h"(static_cast<uint16_t>((prod_187_bf16[0]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_197) >> 16)));
            float val_lo_pair_199[2];
            val_lo_pair_199[0] = 0.0f;
            val_lo_pair_199[1] = adaln_198[0];
            unsigned int val_lo_word_200[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_199[_lp*2 + 0], val_lo_pair_199[_lp*2+1 + 0]));
                val_lo_word_200[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[0] = __uint_as_float(val_lo_word_200[0]);
            float val_hi_pair_201[2];
            val_hi_pair_201[0] = 0.0f;
            val_hi_pair_201[1] = adaln_198[1];
            unsigned int val_hi_word_202[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_201[_lp*2 + 0], val_hi_pair_201[_lp*2+1 + 0]));
                val_hi_word_202[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[1] = __uint_as_float(val_hi_word_202[0]);
            uint32_t _bf16x2_add_17;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_17) : "r"(_vec_load_9[1]), "r"(1065369472));
            unsigned int s_plus_one_203 = _bf16x2_add_17;
            float adaln_204[2] = {_unpack_bf16_10[2], _unpack_bf16_10[3]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_204[0]) : "h"(static_cast<uint16_t>((prod_187_bf16[1]))), "h"(static_cast<uint16_t>((s_plus_one_203))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_204[1]) : "h"(static_cast<uint16_t>((prod_187_bf16[1]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_203) >> 16)));
            float val_lo_pair_205[2];
            val_lo_pair_205[0] = 0.0f;
            val_lo_pair_205[1] = adaln_204[0];
            unsigned int val_lo_word_206[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_205[_lp*2 + 0], val_lo_pair_205[_lp*2+1 + 0]));
                val_lo_word_206[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[2] = __uint_as_float(val_lo_word_206[0]);
            float val_hi_pair_207[2];
            val_hi_pair_207[0] = 0.0f;
            val_hi_pair_207[1] = adaln_204[1];
            unsigned int val_hi_word_208[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_207[_lp*2 + 0], val_hi_pair_207[_lp*2+1 + 0]));
                val_hi_word_208[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[3] = __uint_as_float(val_hi_word_208[0]);
            uint32_t _bf16x2_add_18;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_18) : "r"(_vec_load_9[2]), "r"(1065369472));
            unsigned int s_plus_one_209 = _bf16x2_add_18;
            float adaln_210[2] = {_unpack_bf16_10[4], _unpack_bf16_10[5]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_210[0]) : "h"(static_cast<uint16_t>((prod_187_bf16[2]))), "h"(static_cast<uint16_t>((s_plus_one_209))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_210[1]) : "h"(static_cast<uint16_t>((prod_187_bf16[2]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_209) >> 16)));
            float val_lo_pair_211[2];
            val_lo_pair_211[0] = 0.0f;
            val_lo_pair_211[1] = adaln_210[0];
            unsigned int val_lo_word_212[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_211[_lp*2 + 0], val_lo_pair_211[_lp*2+1 + 0]));
                val_lo_word_212[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[4] = __uint_as_float(val_lo_word_212[0]);
            float val_hi_pair_213[2];
            val_hi_pair_213[0] = 0.0f;
            val_hi_pair_213[1] = adaln_210[1];
            unsigned int val_hi_word_214[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_213[_lp*2 + 0], val_hi_pair_213[_lp*2+1 + 0]));
                val_hi_word_214[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[5] = __uint_as_float(val_hi_word_214[0]);
            uint32_t _bf16x2_add_19;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_19) : "r"(_vec_load_9[3]), "r"(1065369472));
            unsigned int s_plus_one_215 = _bf16x2_add_19;
            float adaln_216[2] = {_unpack_bf16_10[6], _unpack_bf16_10[7]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_216[0]) : "h"(static_cast<uint16_t>((prod_187_bf16[3]))), "h"(static_cast<uint16_t>((s_plus_one_215))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_216[1]) : "h"(static_cast<uint16_t>((prod_187_bf16[3]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_215) >> 16)));
            float val_lo_pair_217[2];
            val_lo_pair_217[0] = 0.0f;
            val_lo_pair_217[1] = adaln_216[0];
            unsigned int val_lo_word_218[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_217[_lp*2 + 0], val_lo_pair_217[_lp*2+1 + 0]));
                val_lo_word_218[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[6] = __uint_as_float(val_lo_word_218[0]);
            float val_hi_pair_219[2];
            val_hi_pair_219[0] = 0.0f;
            val_hi_pair_219[1] = adaln_216[1];
            unsigned int val_hi_word_220[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_219[_lp*2 + 0], val_hi_pair_219[_lp*2+1 + 0]));
                val_hi_word_220[_lp] = *(uint32_t*)&_bf2;
            }
            vals_196[7] = __uint_as_float(val_hi_word_220[0]);
            float mags_221[8];
            mags_221[0] = vals_196[0];
            mags_221[1] = vals_196[1];
            mags_221[2] = vals_196[2];
            mags_221[3] = vals_196[3];
            mags_221[4] = vals_196[4];
            mags_221[5] = vals_196[5];
            mags_221[6] = vals_196[6];
            mags_221[7] = vals_196[7];
            mags_221[0] = fabsf(mags_221[0]);
            mags_221[1] = fabsf(mags_221[1]);
            mags_221[2] = fabsf(mags_221[2]);
            mags_221[3] = fabsf(mags_221[3]);
            mags_221[4] = fabsf(mags_221[4]);
            mags_221[5] = fabsf(mags_221[5]);
            mags_221[6] = fabsf(mags_221[6]);
            mags_221[7] = fabsf(mags_221[7]);
            float absmax_222 = mags_221[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                absmax_222 = max_noftz(absmax_222, mags_221[_lr]);
            }
            float o1_223 = __shfl_xor_sync(0xFFFFFFFF, absmax_222, 1);
            absmax_222 = fmaxf(absmax_222, o1_223);
            float sf_value_224 = g * (absmax_222 * sixth);
            uint16_t sf_pair_225;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_225) : "f"(zero_f32), "f"(sf_value_224));
            unsigned int sf_byte_226 = (unsigned int)sf_pair_225;
            uint32_t sf_f16x2_227;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16x2_227) : "h"((uint16_t)(sf_pair_225)));
            unsigned int sf_f16_lo_228 = sf_f16x2_227 & 65535;
            float sf_f_229;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f_229) : "h"((uint16_t)(sf_f16_lo_228)));
            float out_scale_nonzero_230 = approx_rcp(sf_f_229 * inv_g);
            float out_scale_231 = ((absmax_222 == 0.0f) ? 0.0f : out_scale_nonzero_230);
            float2 out_scale_pair_232 = make_float2(out_scale_231, out_scale_231);
            float scaled_233[8];
            float2 _f32x2_mul_a_155 = make_float2(vals_196[0], vals_196[1]);
            float2 q_pair_234;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_234) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&out_scale_pair_232));
            scaled_233[0] = q_pair_234.x;
            scaled_233[1] = q_pair_234.y;
            float2 _f32x2_mul_a_156 = make_float2(vals_196[2], vals_196[3]);
            float2 q_pair_235;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_235) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&out_scale_pair_232));
            scaled_233[2] = q_pair_235.x;
            scaled_233[3] = q_pair_235.y;
            float2 _f32x2_mul_a_157 = make_float2(vals_196[4], vals_196[5]);
            float2 q_pair_236;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_236) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&out_scale_pair_232));
            scaled_233[4] = q_pair_236.x;
            scaled_233[5] = q_pair_236.y;
            float2 _f32x2_mul_a_158 = make_float2(vals_196[6], vals_196[7]);
            float2 q_pair_237;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_237) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&out_scale_pair_232));
            scaled_233[6] = q_pair_237.x;
            scaled_233[7] = q_pair_237.y;
            unsigned int packed_238[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_238[0]) : "f"(scaled_233[0]), "f"(scaled_233[1]), "f"(scaled_233[2]), "f"(scaled_233[3]), "f"(scaled_233[4]), "f"(scaled_233[5]), "f"(scaled_233[6]), "f"(scaled_233[7]));
            {
                auto _gst_159 = q_ptr + 128;
                __builtin_assume(__isGlobal(_gst_159));
                *(reinterpret_cast<unsigned int*>(_gst_159) + (0)) = packed_238[0];
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 8192)) + (0)) = (unsigned char)(sf_byte_226);
            }
            float w_vals_239[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[(0) + 3]))
                : "r"(w_lane + 2560));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_239[(4) + 3]))
                : "r"(w_lane + 10752 + 2560));
            unsigned int _vec_load_11[4];
            {
                uint4 _uv4_160 = *reinterpret_cast<const uint4*>(s_ptr + 1280 + 0);
                _vec_load_11[0 + 0] = _uv4_160.x;
                _vec_load_11[0 + 1] = _uv4_160.y;
                _vec_load_11[0 + 2] = _uv4_160.z;
                _vec_load_11[0 + 3] = _uv4_160.w;
            }
            unsigned int _vec_load_12[4];
            {
                uint4 _uv4_161 = *reinterpret_cast<const uint4*>(b_ptr + 1280 + 0);
                _vec_load_12[0 + 0] = _uv4_161.x;
                _vec_load_12[0 + 1] = _uv4_161.y;
                _vec_load_12[0 + 2] = _uv4_161.z;
                _vec_load_12[0 + 3] = _uv4_161.w;
            }
            float _unpack_bf16_11[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_own[20 + _pair]) << 16);
                (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_own[20 + _pair]) & 0xffff0000u);
            }
            float _unpack_bf16_12[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_12[_pair]) << 16);
                (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_12[_pair]) & 0xffff0000u);
            }
            float prod_240[8];
            float2 _f32x2_mul_a_162 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
            float2 x_rstd_241;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_241) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_163 = make_float2(w_vals_239[0], w_vals_239[1]);
            float2 x_rstd_w_242;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_242) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&x_rstd_241));
            prod_240[0] = x_rstd_w_242.x;
            prod_240[1] = x_rstd_w_242.y;
            float2 _f32x2_mul_a_164 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
            float2 x_rstd_243;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_243) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_165 = make_float2(w_vals_239[2], w_vals_239[3]);
            float2 x_rstd_w_244;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_244) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&x_rstd_243));
            prod_240[2] = x_rstd_w_244.x;
            prod_240[3] = x_rstd_w_244.y;
            float2 _f32x2_mul_a_166 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
            float2 x_rstd_245;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_245) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_167 = make_float2(w_vals_239[4], w_vals_239[5]);
            float2 x_rstd_w_246;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_246) : "l"(*(const unsigned long long*)&_f32x2_mul_a_167), "l"(*(const unsigned long long*)&x_rstd_245));
            prod_240[4] = x_rstd_w_246.x;
            prod_240[5] = x_rstd_w_246.y;
            float2 _f32x2_mul_a_168 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
            float2 x_rstd_247;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_247) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_169 = make_float2(w_vals_239[6], w_vals_239[7]);
            float2 x_rstd_w_248;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_248) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&x_rstd_247));
            prod_240[6] = x_rstd_w_248.x;
            prod_240[7] = x_rstd_w_248.y;
            unsigned int prod_240_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_240[_lp*2 + 0], prod_240[_lp*2+1 + 0]));
                prod_240_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float vals_249[8];
            uint32_t _bf16x2_add_20;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_20) : "r"(_vec_load_11[0]), "r"(1065369472));
            unsigned int s_plus_one_250 = _bf16x2_add_20;
            float adaln_251[2] = {_unpack_bf16_12[0], _unpack_bf16_12[1]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_251[0]) : "h"(static_cast<uint16_t>((prod_240_bf16[0]))), "h"(static_cast<uint16_t>((s_plus_one_250))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_251[1]) : "h"(static_cast<uint16_t>((prod_240_bf16[0]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_250) >> 16)));
            float val_lo_pair_252[2];
            val_lo_pair_252[0] = 0.0f;
            val_lo_pair_252[1] = adaln_251[0];
            unsigned int val_lo_word_253[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_252[_lp*2 + 0], val_lo_pair_252[_lp*2+1 + 0]));
                val_lo_word_253[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[0] = __uint_as_float(val_lo_word_253[0]);
            float val_hi_pair_254[2];
            val_hi_pair_254[0] = 0.0f;
            val_hi_pair_254[1] = adaln_251[1];
            unsigned int val_hi_word_255[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_254[_lp*2 + 0], val_hi_pair_254[_lp*2+1 + 0]));
                val_hi_word_255[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[1] = __uint_as_float(val_hi_word_255[0]);
            uint32_t _bf16x2_add_21;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_21) : "r"(_vec_load_11[1]), "r"(1065369472));
            unsigned int s_plus_one_256 = _bf16x2_add_21;
            float adaln_257[2] = {_unpack_bf16_12[2], _unpack_bf16_12[3]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_257[0]) : "h"(static_cast<uint16_t>((prod_240_bf16[1]))), "h"(static_cast<uint16_t>((s_plus_one_256))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_257[1]) : "h"(static_cast<uint16_t>((prod_240_bf16[1]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_256) >> 16)));
            float val_lo_pair_258[2];
            val_lo_pair_258[0] = 0.0f;
            val_lo_pair_258[1] = adaln_257[0];
            unsigned int val_lo_word_259[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_258[_lp*2 + 0], val_lo_pair_258[_lp*2+1 + 0]));
                val_lo_word_259[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[2] = __uint_as_float(val_lo_word_259[0]);
            float val_hi_pair_260[2];
            val_hi_pair_260[0] = 0.0f;
            val_hi_pair_260[1] = adaln_257[1];
            unsigned int val_hi_word_261[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_260[_lp*2 + 0], val_hi_pair_260[_lp*2+1 + 0]));
                val_hi_word_261[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[3] = __uint_as_float(val_hi_word_261[0]);
            uint32_t _bf16x2_add_22;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_22) : "r"(_vec_load_11[2]), "r"(1065369472));
            unsigned int s_plus_one_262 = _bf16x2_add_22;
            float adaln_263[2] = {_unpack_bf16_12[4], _unpack_bf16_12[5]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_263[0]) : "h"(static_cast<uint16_t>((prod_240_bf16[2]))), "h"(static_cast<uint16_t>((s_plus_one_262))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_263[1]) : "h"(static_cast<uint16_t>((prod_240_bf16[2]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_262) >> 16)));
            float val_lo_pair_264[2];
            val_lo_pair_264[0] = 0.0f;
            val_lo_pair_264[1] = adaln_263[0];
            unsigned int val_lo_word_265[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_264[_lp*2 + 0], val_lo_pair_264[_lp*2+1 + 0]));
                val_lo_word_265[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[4] = __uint_as_float(val_lo_word_265[0]);
            float val_hi_pair_266[2];
            val_hi_pair_266[0] = 0.0f;
            val_hi_pair_266[1] = adaln_263[1];
            unsigned int val_hi_word_267[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_266[_lp*2 + 0], val_hi_pair_266[_lp*2+1 + 0]));
                val_hi_word_267[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[5] = __uint_as_float(val_hi_word_267[0]);
            uint32_t _bf16x2_add_23;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_23) : "r"(_vec_load_11[3]), "r"(1065369472));
            unsigned int s_plus_one_268 = _bf16x2_add_23;
            float adaln_269[2] = {_unpack_bf16_12[6], _unpack_bf16_12[7]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_269[0]) : "h"(static_cast<uint16_t>((prod_240_bf16[3]))), "h"(static_cast<uint16_t>((s_plus_one_268))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_269[1]) : "h"(static_cast<uint16_t>((prod_240_bf16[3]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_268) >> 16)));
            float val_lo_pair_270[2];
            val_lo_pair_270[0] = 0.0f;
            val_lo_pair_270[1] = adaln_269[0];
            unsigned int val_lo_word_271[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_270[_lp*2 + 0], val_lo_pair_270[_lp*2+1 + 0]));
                val_lo_word_271[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[6] = __uint_as_float(val_lo_word_271[0]);
            float val_hi_pair_272[2];
            val_hi_pair_272[0] = 0.0f;
            val_hi_pair_272[1] = adaln_269[1];
            unsigned int val_hi_word_273[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_272[_lp*2 + 0], val_hi_pair_272[_lp*2+1 + 0]));
                val_hi_word_273[_lp] = *(uint32_t*)&_bf2;
            }
            vals_249[7] = __uint_as_float(val_hi_word_273[0]);
            float mags_274[8];
            mags_274[0] = vals_249[0];
            mags_274[1] = vals_249[1];
            mags_274[2] = vals_249[2];
            mags_274[3] = vals_249[3];
            mags_274[4] = vals_249[4];
            mags_274[5] = vals_249[5];
            mags_274[6] = vals_249[6];
            mags_274[7] = vals_249[7];
            mags_274[0] = fabsf(mags_274[0]);
            mags_274[1] = fabsf(mags_274[1]);
            mags_274[2] = fabsf(mags_274[2]);
            mags_274[3] = fabsf(mags_274[3]);
            mags_274[4] = fabsf(mags_274[4]);
            mags_274[5] = fabsf(mags_274[5]);
            mags_274[6] = fabsf(mags_274[6]);
            mags_274[7] = fabsf(mags_274[7]);
            float absmax_275 = mags_274[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                absmax_275 = max_noftz(absmax_275, mags_274[_lr]);
            }
            float o1_276 = __shfl_xor_sync(0xFFFFFFFF, absmax_275, 1);
            absmax_275 = fmaxf(absmax_275, o1_276);
            float sf_value_277 = g * (absmax_275 * sixth);
            uint16_t sf_pair_278;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_278) : "f"(zero_f32), "f"(sf_value_277));
            unsigned int sf_byte_279 = (unsigned int)sf_pair_278;
            uint32_t sf_f16x2_280;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16x2_280) : "h"((uint16_t)(sf_pair_278)));
            unsigned int sf_f16_lo_281 = sf_f16x2_280 & 65535;
            float sf_f_282;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f_282) : "h"((uint16_t)(sf_f16_lo_281)));
            float out_scale_nonzero_283 = approx_rcp(sf_f_282 * inv_g);
            float out_scale_284 = ((absmax_275 == 0.0f) ? 0.0f : out_scale_nonzero_283);
            float2 out_scale_pair_285 = make_float2(out_scale_284, out_scale_284);
            float scaled_286[8];
            float2 _f32x2_mul_a_170 = make_float2(vals_249[0], vals_249[1]);
            float2 q_pair_287;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_287) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&out_scale_pair_285));
            scaled_286[0] = q_pair_287.x;
            scaled_286[1] = q_pair_287.y;
            float2 _f32x2_mul_a_171 = make_float2(vals_249[2], vals_249[3]);
            float2 q_pair_288;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_288) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&out_scale_pair_285));
            scaled_286[2] = q_pair_288.x;
            scaled_286[3] = q_pair_288.y;
            float2 _f32x2_mul_a_172 = make_float2(vals_249[4], vals_249[5]);
            float2 q_pair_289;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_289) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&out_scale_pair_285));
            scaled_286[4] = q_pair_289.x;
            scaled_286[5] = q_pair_289.y;
            float2 _f32x2_mul_a_173 = make_float2(vals_249[6], vals_249[7]);
            float2 q_pair_290;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_290) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&out_scale_pair_285));
            scaled_286[6] = q_pair_290.x;
            scaled_286[7] = q_pair_290.y;
            unsigned int packed_291[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_291[0]) : "f"(scaled_286[0]), "f"(scaled_286[1]), "f"(scaled_286[2]), "f"(scaled_286[3]), "f"(scaled_286[4]), "f"(scaled_286[5]), "f"(scaled_286[6]), "f"(scaled_286[7]));
            {
                auto _gst_174 = q_ptr + 160;
                __builtin_assume(__isGlobal(_gst_174));
                *(reinterpret_cast<unsigned int*>(_gst_174) + (0)) = packed_291[0];
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 10240)) + (0)) = (unsigned char)(sf_byte_279);
            }
            float w_vals_292[8];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[(0) + 3]))
                : "r"(w_lane + 3072));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_vals_292[(4) + 3]))
                : "r"(w_lane + 10752 + 3072));
            unsigned int _vec_load_13[4];
            {
                uint4 _uv4_175 = *reinterpret_cast<const uint4*>(s_ptr + 1536 + 0);
                _vec_load_13[0 + 0] = _uv4_175.x;
                _vec_load_13[0 + 1] = _uv4_175.y;
                _vec_load_13[0 + 2] = _uv4_175.z;
                _vec_load_13[0 + 3] = _uv4_175.w;
            }
            unsigned int _vec_load_14[4];
            {
                uint4 _uv4_176 = *reinterpret_cast<const uint4*>(b_ptr + 1536 + 0);
                _vec_load_14[0 + 0] = _uv4_176.x;
                _vec_load_14[0 + 1] = _uv4_176.y;
                _vec_load_14[0 + 2] = _uv4_176.z;
                _vec_load_14[0 + 3] = _uv4_176.w;
            }
            float _unpack_bf16_13[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_own[24 + _pair]) << 16);
                (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_own[24 + _pair]) & 0xffff0000u);
            }
            float _unpack_bf16_14[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_14[_pair]) << 16);
                (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_14[_pair]) & 0xffff0000u);
            }
            float prod_293[8];
            float2 _f32x2_mul_a_177 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
            float2 x_rstd_294;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_294) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_178 = make_float2(w_vals_292[0], w_vals_292[1]);
            float2 x_rstd_w_295;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_295) : "l"(*(const unsigned long long*)&_f32x2_mul_a_178), "l"(*(const unsigned long long*)&x_rstd_294));
            prod_293[0] = x_rstd_w_295.x;
            prod_293[1] = x_rstd_w_295.y;
            float2 _f32x2_mul_a_179 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
            float2 x_rstd_296;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_296) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_180 = make_float2(w_vals_292[2], w_vals_292[3]);
            float2 x_rstd_w_297;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_297) : "l"(*(const unsigned long long*)&_f32x2_mul_a_180), "l"(*(const unsigned long long*)&x_rstd_296));
            prod_293[2] = x_rstd_w_297.x;
            prod_293[3] = x_rstd_w_297.y;
            float2 _f32x2_mul_a_181 = make_float2(_unpack_bf16_13[4], _unpack_bf16_13[5]);
            float2 x_rstd_298;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_298) : "l"(*(const unsigned long long*)&_f32x2_mul_a_181), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_182 = make_float2(w_vals_292[4], w_vals_292[5]);
            float2 x_rstd_w_299;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_299) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&x_rstd_298));
            prod_293[4] = x_rstd_w_299.x;
            prod_293[5] = x_rstd_w_299.y;
            float2 _f32x2_mul_a_183 = make_float2(_unpack_bf16_13[6], _unpack_bf16_13[7]);
            float2 x_rstd_300;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_300) : "l"(*(const unsigned long long*)&_f32x2_mul_a_183), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_a_184 = make_float2(w_vals_292[6], w_vals_292[7]);
            float2 x_rstd_w_301;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_301) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&x_rstd_300));
            prod_293[6] = x_rstd_w_301.x;
            prod_293[7] = x_rstd_w_301.y;
            unsigned int prod_293_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_293[_lp*2 + 0], prod_293[_lp*2+1 + 0]));
                prod_293_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            float vals_302[8];
            uint32_t _bf16x2_add_24;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_24) : "r"(_vec_load_13[0]), "r"(1065369472));
            unsigned int s_plus_one_303 = _bf16x2_add_24;
            float adaln_304[2] = {_unpack_bf16_14[0], _unpack_bf16_14[1]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_304[0]) : "h"(static_cast<uint16_t>((prod_293_bf16[0]))), "h"(static_cast<uint16_t>((s_plus_one_303))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_304[1]) : "h"(static_cast<uint16_t>((prod_293_bf16[0]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_303) >> 16)));
            float val_lo_pair_305[2];
            val_lo_pair_305[0] = 0.0f;
            val_lo_pair_305[1] = adaln_304[0];
            unsigned int val_lo_word_306[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_305[_lp*2 + 0], val_lo_pair_305[_lp*2+1 + 0]));
                val_lo_word_306[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[0] = __uint_as_float(val_lo_word_306[0]);
            float val_hi_pair_307[2];
            val_hi_pair_307[0] = 0.0f;
            val_hi_pair_307[1] = adaln_304[1];
            unsigned int val_hi_word_308[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_307[_lp*2 + 0], val_hi_pair_307[_lp*2+1 + 0]));
                val_hi_word_308[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[1] = __uint_as_float(val_hi_word_308[0]);
            uint32_t _bf16x2_add_25;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_25) : "r"(_vec_load_13[1]), "r"(1065369472));
            unsigned int s_plus_one_309 = _bf16x2_add_25;
            float adaln_310[2] = {_unpack_bf16_14[2], _unpack_bf16_14[3]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_310[0]) : "h"(static_cast<uint16_t>((prod_293_bf16[1]))), "h"(static_cast<uint16_t>((s_plus_one_309))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_310[1]) : "h"(static_cast<uint16_t>((prod_293_bf16[1]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_309) >> 16)));
            float val_lo_pair_311[2];
            val_lo_pair_311[0] = 0.0f;
            val_lo_pair_311[1] = adaln_310[0];
            unsigned int val_lo_word_312[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_311[_lp*2 + 0], val_lo_pair_311[_lp*2+1 + 0]));
                val_lo_word_312[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[2] = __uint_as_float(val_lo_word_312[0]);
            float val_hi_pair_313[2];
            val_hi_pair_313[0] = 0.0f;
            val_hi_pair_313[1] = adaln_310[1];
            unsigned int val_hi_word_314[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_313[_lp*2 + 0], val_hi_pair_313[_lp*2+1 + 0]));
                val_hi_word_314[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[3] = __uint_as_float(val_hi_word_314[0]);
            uint32_t _bf16x2_add_26;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_26) : "r"(_vec_load_13[2]), "r"(1065369472));
            unsigned int s_plus_one_315 = _bf16x2_add_26;
            float adaln_316[2] = {_unpack_bf16_14[4], _unpack_bf16_14[5]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_316[0]) : "h"(static_cast<uint16_t>((prod_293_bf16[2]))), "h"(static_cast<uint16_t>((s_plus_one_315))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_316[1]) : "h"(static_cast<uint16_t>((prod_293_bf16[2]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_315) >> 16)));
            float val_lo_pair_317[2];
            val_lo_pair_317[0] = 0.0f;
            val_lo_pair_317[1] = adaln_316[0];
            unsigned int val_lo_word_318[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_317[_lp*2 + 0], val_lo_pair_317[_lp*2+1 + 0]));
                val_lo_word_318[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[4] = __uint_as_float(val_lo_word_318[0]);
            float val_hi_pair_319[2];
            val_hi_pair_319[0] = 0.0f;
            val_hi_pair_319[1] = adaln_316[1];
            unsigned int val_hi_word_320[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_319[_lp*2 + 0], val_hi_pair_319[_lp*2+1 + 0]));
                val_hi_word_320[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[5] = __uint_as_float(val_hi_word_320[0]);
            uint32_t _bf16x2_add_27;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_27) : "r"(_vec_load_13[3]), "r"(1065369472));
            unsigned int s_plus_one_321 = _bf16x2_add_27;
            float adaln_322[2] = {_unpack_bf16_14[6], _unpack_bf16_14[7]};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_322[0]) : "h"(static_cast<uint16_t>((prod_293_bf16[3]))), "h"(static_cast<uint16_t>((s_plus_one_321))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(adaln_322[1]) : "h"(static_cast<uint16_t>((prod_293_bf16[3]) >> 16)), "h"(static_cast<uint16_t>((s_plus_one_321) >> 16)));
            float val_lo_pair_323[2];
            val_lo_pair_323[0] = 0.0f;
            val_lo_pair_323[1] = adaln_322[0];
            unsigned int val_lo_word_324[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_lo_pair_323[_lp*2 + 0], val_lo_pair_323[_lp*2+1 + 0]));
                val_lo_word_324[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[6] = __uint_as_float(val_lo_word_324[0]);
            float val_hi_pair_325[2];
            val_hi_pair_325[0] = 0.0f;
            val_hi_pair_325[1] = adaln_322[1];
            unsigned int val_hi_word_326[1];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(val_hi_pair_325[_lp*2 + 0], val_hi_pair_325[_lp*2+1 + 0]));
                val_hi_word_326[_lp] = *(uint32_t*)&_bf2;
            }
            vals_302[7] = __uint_as_float(val_hi_word_326[0]);
            float mags_327[8];
            mags_327[0] = vals_302[0];
            mags_327[1] = vals_302[1];
            mags_327[2] = vals_302[2];
            mags_327[3] = vals_302[3];
            mags_327[4] = vals_302[4];
            mags_327[5] = vals_302[5];
            mags_327[6] = vals_302[6];
            mags_327[7] = vals_302[7];
            mags_327[0] = fabsf(mags_327[0]);
            mags_327[1] = fabsf(mags_327[1]);
            mags_327[2] = fabsf(mags_327[2]);
            mags_327[3] = fabsf(mags_327[3]);
            mags_327[4] = fabsf(mags_327[4]);
            mags_327[5] = fabsf(mags_327[5]);
            mags_327[6] = fabsf(mags_327[6]);
            mags_327[7] = fabsf(mags_327[7]);
            float absmax_328 = mags_327[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                absmax_328 = max_noftz(absmax_328, mags_327[_lr]);
            }
            float o1_329 = __shfl_xor_sync(0xFFFFFFFF, absmax_328, 1);
            absmax_328 = fmaxf(absmax_328, o1_329);
            float sf_value_330 = g * (absmax_328 * sixth);
            uint16_t sf_pair_331;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_331) : "f"(zero_f32), "f"(sf_value_330));
            unsigned int sf_byte_332 = (unsigned int)sf_pair_331;
            uint32_t sf_f16x2_333;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16x2_333) : "h"((uint16_t)(sf_pair_331)));
            unsigned int sf_f16_lo_334 = sf_f16x2_333 & 65535;
            float sf_f_335;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f_335) : "h"((uint16_t)(sf_f16_lo_334)));
            float out_scale_nonzero_336 = approx_rcp(sf_f_335 * inv_g);
            float out_scale_337 = ((absmax_328 == 0.0f) ? 0.0f : out_scale_nonzero_336);
            float2 out_scale_pair_338 = make_float2(out_scale_337, out_scale_337);
            float scaled_339[8];
            float2 _f32x2_mul_a_185 = make_float2(vals_302[0], vals_302[1]);
            float2 q_pair_340;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_340) : "l"(*(const unsigned long long*)&_f32x2_mul_a_185), "l"(*(const unsigned long long*)&out_scale_pair_338));
            scaled_339[0] = q_pair_340.x;
            scaled_339[1] = q_pair_340.y;
            float2 _f32x2_mul_a_186 = make_float2(vals_302[2], vals_302[3]);
            float2 q_pair_341;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_341) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&out_scale_pair_338));
            scaled_339[2] = q_pair_341.x;
            scaled_339[3] = q_pair_341.y;
            float2 _f32x2_mul_a_187 = make_float2(vals_302[4], vals_302[5]);
            float2 q_pair_342;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_342) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&out_scale_pair_338));
            scaled_339[4] = q_pair_342.x;
            scaled_339[5] = q_pair_342.y;
            float2 _f32x2_mul_a_188 = make_float2(vals_302[6], vals_302[7]);
            float2 q_pair_343;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair_343) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&out_scale_pair_338));
            scaled_339[6] = q_pair_343.x;
            scaled_339[7] = q_pair_343.y;
            unsigned int packed_344[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_344[0]) : "f"(scaled_339[0]), "f"(scaled_339[1]), "f"(scaled_339[2]), "f"(scaled_339[3]), "f"(scaled_339[4]), "f"(scaled_339[5]), "f"(scaled_339[6]), "f"(scaled_339[7]));
            {
                auto _gst_189 = q_ptr + 192;
                __builtin_assume(__isGlobal(_gst_189));
                *(reinterpret_cast<unsigned int*>(_gst_189) + (0)) = packed_344[0];
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 12288)) + (0)) = (unsigned char)(sf_byte_332);
            }
        } else {
            unsigned int zero_word = 0;
            {
                auto _gst_190 = q_ptr + 0;
                __builtin_assume(__isGlobal(_gst_190));
                *(reinterpret_cast<unsigned int*>(_gst_190) + (0)) = zero_word;
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + sf_lane_base) + (0)) = (unsigned char)(zero_word);
            }
            {
                auto _gst_191 = q_ptr + 32;
                __builtin_assume(__isGlobal(_gst_191));
                *(reinterpret_cast<unsigned int*>(_gst_191) + (0)) = zero_word;
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 2048)) + (0)) = (unsigned char)(zero_word);
            }
            {
                auto _gst_192 = q_ptr + 64;
                __builtin_assume(__isGlobal(_gst_192));
                *(reinterpret_cast<unsigned int*>(_gst_192) + (0)) = zero_word;
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 4096)) + (0)) = (unsigned char)(zero_word);
            }
            {
                auto _gst_193 = q_ptr + 96;
                __builtin_assume(__isGlobal(_gst_193));
                *(reinterpret_cast<unsigned int*>(_gst_193) + (0)) = zero_word;
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 6144)) + (0)) = (unsigned char)(zero_word);
            }
            {
                auto _gst_194 = q_ptr + 128;
                __builtin_assume(__isGlobal(_gst_194));
                *(reinterpret_cast<unsigned int*>(_gst_194) + (0)) = zero_word;
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 8192)) + (0)) = (unsigned char)(zero_word);
            }
            {
                auto _gst_195 = q_ptr + 160;
                __builtin_assume(__isGlobal(_gst_195));
                *(reinterpret_cast<unsigned int*>(_gst_195) + (0)) = zero_word;
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 10240)) + (0)) = (unsigned char)(zero_word);
            }
            {
                auto _gst_196 = q_ptr + 192;
                __builtin_assume(__isGlobal(_gst_196));
                *(reinterpret_cast<unsigned int*>(_gst_196) + (0)) = zero_word;
            }
            if (pair_lane == 0) {
                *(reinterpret_cast<unsigned char*>(a_sf + (sf_lane_base + 12288)) + (0)) = (unsigned char)(zero_word);
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

#undef LAUNCH_MIN_BLOCKS
#undef MINIMAX_H3_MLP_INF
#undef NUM_MAIN_STAGES
#undef SMEM_RSTD_SMEM_OFF
#undef SMEM_RSTD_SMEM_STAGE_BYTES
#undef SMEM_RSTD_SMEM_STRIDE
#undef SMEM_TOTAL
#undef SMEM_W_SMEM_OFF
#undef SMEM_W_SMEM_STAGE_BYTES
#undef SMEM_W_SMEM_STRIDE
#undef SMEM_X_RING_OFF
#undef SMEM_X_RING_STAGE_BYTES
#undef SMEM_X_RING_STRIDE
#undef THREADS
#undef w_full_addr
#undef x_full_addr

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define TMEM_NCOLS 488
#define TMEM_ACCUM0_OFFSET 0
#define TMEM_ACCUM1_OFFSET 224
#define TMEM_TMEM_SFA_OFFSET 448
#define TMEM_TMEM_SFB_OFFSET 456
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 55296
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 28672
#define SMEM_SMEM_B_STRIDE 55296
#define SMEM_SMEM_V2_OFF 17408
#define SMEM_SMEM_V2_STAGE_BYTES 14336
#define SMEM_SMEM_V2_STRIDE 55296
#define SMEM_SMEM_V3_OFF 31744
#define SMEM_SMEM_V3_STAGE_BYTES 14336
#define SMEM_SMEM_V3_STRIDE 55296
#define SMEM_SMEM_SFA_ALL_OFF 46080
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_ALL_STRIDE 55296
#define SMEM_SMEM_V5_OFF 48128
#define SMEM_SMEM_V5_STAGE_BYTES 4096
#define SMEM_SMEM_V5_STRIDE 55296
#define SMEM_SMEM_V6_OFF 52224
#define SMEM_SMEM_V6_STAGE_BYTES 4096
#define SMEM_SMEM_V6_STRIDE 55296
#define SMEM_SMEM_V7_OFF 46080
#define SMEM_SMEM_V7_STAGE_BYTES 512
#define SMEM_SMEM_V7_STRIDE 55296
#define SMEM_SMEM_V8_OFF 46592
#define SMEM_SMEM_V8_STAGE_BYTES 512
#define SMEM_SMEM_V8_STRIDE 55296
#define SMEM_SMEM_V9_OFF 47104
#define SMEM_SMEM_V9_STAGE_BYTES 512
#define SMEM_SMEM_V9_STRIDE 55296
#define SMEM_SMEM_V10_OFF 47616
#define SMEM_SMEM_V10_STAGE_BYTES 512
#define SMEM_SMEM_V10_STRIDE 55296
#define SMEM_SMEM_V11_OFF 48128
#define SMEM_SMEM_V11_STAGE_BYTES 1024
#define SMEM_SMEM_V11_STRIDE 55296
#define SMEM_SMEM_V12_OFF 49152
#define SMEM_SMEM_V12_STAGE_BYTES 1024
#define SMEM_SMEM_V12_STRIDE 55296
#define SMEM_SMEM_V13_OFF 50176
#define SMEM_SMEM_V13_STAGE_BYTES 1024
#define SMEM_SMEM_V13_STRIDE 55296
#define SMEM_SMEM_V14_OFF 51200
#define SMEM_SMEM_V14_STAGE_BYTES 1024
#define SMEM_SMEM_V14_STRIDE 55296
#define SMEM_SMEM_V15_OFF 52224
#define SMEM_SMEM_V15_STAGE_BYTES 1024
#define SMEM_SMEM_V15_STRIDE 55296
#define SMEM_SMEM_V16_OFF 53248
#define SMEM_SMEM_V16_STAGE_BYTES 1024
#define SMEM_SMEM_V16_STRIDE 55296
#define SMEM_SMEM_V17_OFF 54272
#define SMEM_SMEM_V17_STAGE_BYTES 1024
#define SMEM_SMEM_V17_STRIDE 55296
#define SMEM_SMEM_V18_OFF 55296
#define SMEM_SMEM_V18_STAGE_BYTES 1024
#define SMEM_SMEM_V18_STRIDE 55296
#define SMEM_WORK_RESPONSE_OFF 222208
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 222336
#define tiles_per_group 4096

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_mlp_fc1_e2m1(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, float* __restrict__ y_global_scale, unsigned int* __restrict__ y_q, uint8_t* __restrict__ y_sf, int M, int m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 32)
    #define mainloop_done_addr (mbar_base + 64)
    #define epilogue_done_addr (mbar_base + 72)
    #define work_full_addr (mbar_base + 80)
    #define work_empty_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_v2 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v2_addr = smem + 17408;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 31744);
    const int smem_v3_addr = smem + 31744;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 46080);
    const int smem_sfa_all_addr = smem + 46080;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 48128);
    const int smem_v5_addr = smem + 48128;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 52224);
    const int smem_v6_addr = smem + 52224;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 46080);
    const int smem_v7_addr = smem + 46080;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 46592);
    const int smem_v8_addr = smem + 46592;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 47104);
    const int smem_v9_addr = smem + 47104;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 47616);
    const int smem_v10_addr = smem + 47616;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 48128);
    const int smem_v11_addr = smem + 48128;
    uint8_t* smem_v12 = reinterpret_cast<uint8_t*>(smem_raw + 49152);
    const int smem_v12_addr = smem + 49152;
    uint8_t* smem_v13 = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_v13_addr = smem + 50176;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_v14_addr = smem + 51200;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 52224);
    const int smem_v15_addr = smem + 52224;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 53248);
    const int smem_v16_addr = smem + 53248;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 54272);
    const int smem_v17_addr = smem + 54272;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 55296);
    const int smem_v18_addr = smem + 55296;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    const int work_response_addr = smem + 222208;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 72, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 112, 546);
            mbarrier_init(smem + 120, 546);
            mbarrier_init(smem + 128, 546);
            mbarrier_init(smem + 136, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 488 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum0 = taddr;
    const int tmem_accum1 = taddr + 224;
    const int tmem_tmem_sfa = taddr + 448;
    const int tmem_tmem_sfb = taddr + 456;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int weight_row_base = cta_rank * 14336;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < m_tiles / 2 * 64; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * 64;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= 64) ? 64 : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * 128;
                    int off_n = bid_n * 224;
                    int sfa_tile_row = bid_m * 84;
                    int sfb_tile_row0 = bid_n * 2 * 84;
                    int sfb_tile_row1 = (bid_n * 2 + 1) * 84;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < 21; iter_k++) {
                        mbarrier_wait_hint(mma_done_addr + (load_stage) * 8, _phase_mma_done, 10000000);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 55296, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 55296, (&B), 0, weight_row_base + off_n, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        int k_set_base = iter_k * 4;
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 55296, (&SFA), 0, 0, sfa_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_v5_addr + load_stage * 55296, (&SFB), 0, 0, sfb_tile_row0 + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_v6_addr + load_stage * 55296, (&SFB), 0, 0, sfb_tile_row1 + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(55296)) : "memory");
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 10000000);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 4; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < m_tiles / 2 * 64; _tile_iter_1++) {
                    mbarrier_wait_hint(epilogue_done_addr, _phase_epilogue_done, 10000000);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < 21; iter_k_1++) {
                        mbarrier_wait_hint(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_2 = (((smem_v2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
                        }
                        int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_4 = (((smem_v3_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 8 + 0, ((init_flag) ? 0 : 1), _elected_0);
                        }
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v12_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v12_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        int _smem_desc_lo_5 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_6 = (((smem_v2_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_0);
                        }
                        int _smem_desc_lo_7 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_8 = (((smem_v3_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_0);
                        }
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v17_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v17_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        int _smem_desc_lo_9 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_10 = (((smem_v2_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_9) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_0);
                        }
                        int _smem_desc_lo_11 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_12 = (((smem_v3_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_12) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
                        }
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v18_addr) >> 4) + (mma_tma_stage) * 3456)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v18_addr) >> 4) + (mma_tma_stage) * 3456 + 32)), _elected_0);
                        int _smem_desc_lo_13 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_14 = (((smem_v2_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_13) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_14) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum0, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_0);
                        }
                        int _smem_desc_lo_15 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        int _smem_desc_lo_16 = (((smem_v3_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 3456;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum1, a_desc + 0, b_desc + 0,
                                0x10380480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_0);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 10000000);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int acc_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            float alpha_v = alpha[0];
            float g = y_global_scale[0];
            float inv_g = approx_rcp(g);
            float sixth = approx_rcp(6.0f);
            float zero_f32 = 0.0f;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < m_tiles / 2 * 64; _tile_iter_2++) {
                mbarrier_wait_hint(mainloop_done_addr, _phase_mainloop_done, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * 64;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= 64) ? 64 : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * 128;
                int off_n_1 = bid_n_1 * 224;
                int global_row = off_m_1 + local_row;
                int col0 = acc_half * 112;
                unsigned long long word_base = (unsigned long long)global_row * 14336 + (unsigned long long)off_n_1 + (unsigned long long)col0 >> 3;
                unsigned long long sf_row_base = (unsigned long long)bid_m_1 * 224 * 512 + (unsigned long long)(local_row & 31) * 16 + (unsigned long long)(local_row >> 5) * 4;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(acc_half * 224);
                unsigned int gate_pk[56];
                unsigned int up_pk[56];
                float _tmem_load_0[32];
                tmem_ld_x16(&_tmem_load_0[0], lane_addr);
                tmem_ld_x16(&_tmem_load_0[16], lane_addr + 16);
                float _tmem_load_1[32];
                tmem_ld_x16(&_tmem_load_1[0], lane_addr + 112);
                tmem_ld_x16(&_tmem_load_1[16], lane_addr + 112 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                _tmem_load_0[0] = _tmem_load_0[0] * alpha_v;
                _tmem_load_1[0] = _tmem_load_1[0] * alpha_v;
                _tmem_load_0[1] = _tmem_load_0[1] * alpha_v;
                _tmem_load_1[1] = _tmem_load_1[1] * alpha_v;
                _tmem_load_0[2] = _tmem_load_0[2] * alpha_v;
                _tmem_load_1[2] = _tmem_load_1[2] * alpha_v;
                _tmem_load_0[3] = _tmem_load_0[3] * alpha_v;
                _tmem_load_1[3] = _tmem_load_1[3] * alpha_v;
                _tmem_load_0[4] = _tmem_load_0[4] * alpha_v;
                _tmem_load_1[4] = _tmem_load_1[4] * alpha_v;
                _tmem_load_0[5] = _tmem_load_0[5] * alpha_v;
                _tmem_load_1[5] = _tmem_load_1[5] * alpha_v;
                _tmem_load_0[6] = _tmem_load_0[6] * alpha_v;
                _tmem_load_1[6] = _tmem_load_1[6] * alpha_v;
                _tmem_load_0[7] = _tmem_load_0[7] * alpha_v;
                _tmem_load_1[7] = _tmem_load_1[7] * alpha_v;
                _tmem_load_0[8] = _tmem_load_0[8] * alpha_v;
                _tmem_load_1[8] = _tmem_load_1[8] * alpha_v;
                _tmem_load_0[9] = _tmem_load_0[9] * alpha_v;
                _tmem_load_1[9] = _tmem_load_1[9] * alpha_v;
                _tmem_load_0[10] = _tmem_load_0[10] * alpha_v;
                _tmem_load_1[10] = _tmem_load_1[10] * alpha_v;
                _tmem_load_0[11] = _tmem_load_0[11] * alpha_v;
                _tmem_load_1[11] = _tmem_load_1[11] * alpha_v;
                _tmem_load_0[12] = _tmem_load_0[12] * alpha_v;
                _tmem_load_1[12] = _tmem_load_1[12] * alpha_v;
                _tmem_load_0[13] = _tmem_load_0[13] * alpha_v;
                _tmem_load_1[13] = _tmem_load_1[13] * alpha_v;
                _tmem_load_0[14] = _tmem_load_0[14] * alpha_v;
                _tmem_load_1[14] = _tmem_load_1[14] * alpha_v;
                _tmem_load_0[15] = _tmem_load_0[15] * alpha_v;
                _tmem_load_1[15] = _tmem_load_1[15] * alpha_v;
                _tmem_load_0[16] = _tmem_load_0[16] * alpha_v;
                _tmem_load_1[16] = _tmem_load_1[16] * alpha_v;
                _tmem_load_0[17] = _tmem_load_0[17] * alpha_v;
                _tmem_load_1[17] = _tmem_load_1[17] * alpha_v;
                _tmem_load_0[18] = _tmem_load_0[18] * alpha_v;
                _tmem_load_1[18] = _tmem_load_1[18] * alpha_v;
                _tmem_load_0[19] = _tmem_load_0[19] * alpha_v;
                _tmem_load_1[19] = _tmem_load_1[19] * alpha_v;
                _tmem_load_0[20] = _tmem_load_0[20] * alpha_v;
                _tmem_load_1[20] = _tmem_load_1[20] * alpha_v;
                _tmem_load_0[21] = _tmem_load_0[21] * alpha_v;
                _tmem_load_1[21] = _tmem_load_1[21] * alpha_v;
                _tmem_load_0[22] = _tmem_load_0[22] * alpha_v;
                _tmem_load_1[22] = _tmem_load_1[22] * alpha_v;
                _tmem_load_0[23] = _tmem_load_0[23] * alpha_v;
                _tmem_load_1[23] = _tmem_load_1[23] * alpha_v;
                _tmem_load_0[24] = _tmem_load_0[24] * alpha_v;
                _tmem_load_1[24] = _tmem_load_1[24] * alpha_v;
                _tmem_load_0[25] = _tmem_load_0[25] * alpha_v;
                _tmem_load_1[25] = _tmem_load_1[25] * alpha_v;
                _tmem_load_0[26] = _tmem_load_0[26] * alpha_v;
                _tmem_load_1[26] = _tmem_load_1[26] * alpha_v;
                _tmem_load_0[27] = _tmem_load_0[27] * alpha_v;
                _tmem_load_1[27] = _tmem_load_1[27] * alpha_v;
                _tmem_load_0[28] = _tmem_load_0[28] * alpha_v;
                _tmem_load_1[28] = _tmem_load_1[28] * alpha_v;
                _tmem_load_0[29] = _tmem_load_0[29] * alpha_v;
                _tmem_load_1[29] = _tmem_load_1[29] * alpha_v;
                _tmem_load_0[30] = _tmem_load_0[30] * alpha_v;
                _tmem_load_1[30] = _tmem_load_1[30] * alpha_v;
                _tmem_load_0[31] = _tmem_load_0[31] * alpha_v;
                _tmem_load_1[31] = _tmem_load_1[31] * alpha_v;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    gate_pk[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    up_pk[_lp] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_2[32];
                tmem_ld_x16(&_tmem_load_2[0], lane_addr + 32);
                tmem_ld_x16(&_tmem_load_2[16], lane_addr + 32 + 16);
                float _tmem_load_3[32];
                tmem_ld_x16(&_tmem_load_3[0], lane_addr + 112 + 32);
                tmem_ld_x16(&_tmem_load_3[16], lane_addr + 112 + 32 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                _tmem_load_2[0] = _tmem_load_2[0] * alpha_v;
                _tmem_load_3[0] = _tmem_load_3[0] * alpha_v;
                _tmem_load_2[1] = _tmem_load_2[1] * alpha_v;
                _tmem_load_3[1] = _tmem_load_3[1] * alpha_v;
                _tmem_load_2[2] = _tmem_load_2[2] * alpha_v;
                _tmem_load_3[2] = _tmem_load_3[2] * alpha_v;
                _tmem_load_2[3] = _tmem_load_2[3] * alpha_v;
                _tmem_load_3[3] = _tmem_load_3[3] * alpha_v;
                _tmem_load_2[4] = _tmem_load_2[4] * alpha_v;
                _tmem_load_3[4] = _tmem_load_3[4] * alpha_v;
                _tmem_load_2[5] = _tmem_load_2[5] * alpha_v;
                _tmem_load_3[5] = _tmem_load_3[5] * alpha_v;
                _tmem_load_2[6] = _tmem_load_2[6] * alpha_v;
                _tmem_load_3[6] = _tmem_load_3[6] * alpha_v;
                _tmem_load_2[7] = _tmem_load_2[7] * alpha_v;
                _tmem_load_3[7] = _tmem_load_3[7] * alpha_v;
                _tmem_load_2[8] = _tmem_load_2[8] * alpha_v;
                _tmem_load_3[8] = _tmem_load_3[8] * alpha_v;
                _tmem_load_2[9] = _tmem_load_2[9] * alpha_v;
                _tmem_load_3[9] = _tmem_load_3[9] * alpha_v;
                _tmem_load_2[10] = _tmem_load_2[10] * alpha_v;
                _tmem_load_3[10] = _tmem_load_3[10] * alpha_v;
                _tmem_load_2[11] = _tmem_load_2[11] * alpha_v;
                _tmem_load_3[11] = _tmem_load_3[11] * alpha_v;
                _tmem_load_2[12] = _tmem_load_2[12] * alpha_v;
                _tmem_load_3[12] = _tmem_load_3[12] * alpha_v;
                _tmem_load_2[13] = _tmem_load_2[13] * alpha_v;
                _tmem_load_3[13] = _tmem_load_3[13] * alpha_v;
                _tmem_load_2[14] = _tmem_load_2[14] * alpha_v;
                _tmem_load_3[14] = _tmem_load_3[14] * alpha_v;
                _tmem_load_2[15] = _tmem_load_2[15] * alpha_v;
                _tmem_load_3[15] = _tmem_load_3[15] * alpha_v;
                _tmem_load_2[16] = _tmem_load_2[16] * alpha_v;
                _tmem_load_3[16] = _tmem_load_3[16] * alpha_v;
                _tmem_load_2[17] = _tmem_load_2[17] * alpha_v;
                _tmem_load_3[17] = _tmem_load_3[17] * alpha_v;
                _tmem_load_2[18] = _tmem_load_2[18] * alpha_v;
                _tmem_load_3[18] = _tmem_load_3[18] * alpha_v;
                _tmem_load_2[19] = _tmem_load_2[19] * alpha_v;
                _tmem_load_3[19] = _tmem_load_3[19] * alpha_v;
                _tmem_load_2[20] = _tmem_load_2[20] * alpha_v;
                _tmem_load_3[20] = _tmem_load_3[20] * alpha_v;
                _tmem_load_2[21] = _tmem_load_2[21] * alpha_v;
                _tmem_load_3[21] = _tmem_load_3[21] * alpha_v;
                _tmem_load_2[22] = _tmem_load_2[22] * alpha_v;
                _tmem_load_3[22] = _tmem_load_3[22] * alpha_v;
                _tmem_load_2[23] = _tmem_load_2[23] * alpha_v;
                _tmem_load_3[23] = _tmem_load_3[23] * alpha_v;
                _tmem_load_2[24] = _tmem_load_2[24] * alpha_v;
                _tmem_load_3[24] = _tmem_load_3[24] * alpha_v;
                _tmem_load_2[25] = _tmem_load_2[25] * alpha_v;
                _tmem_load_3[25] = _tmem_load_3[25] * alpha_v;
                _tmem_load_2[26] = _tmem_load_2[26] * alpha_v;
                _tmem_load_3[26] = _tmem_load_3[26] * alpha_v;
                _tmem_load_2[27] = _tmem_load_2[27] * alpha_v;
                _tmem_load_3[27] = _tmem_load_3[27] * alpha_v;
                _tmem_load_2[28] = _tmem_load_2[28] * alpha_v;
                _tmem_load_3[28] = _tmem_load_3[28] * alpha_v;
                _tmem_load_2[29] = _tmem_load_2[29] * alpha_v;
                _tmem_load_3[29] = _tmem_load_3[29] * alpha_v;
                _tmem_load_2[30] = _tmem_load_2[30] * alpha_v;
                _tmem_load_3[30] = _tmem_load_3[30] * alpha_v;
                _tmem_load_2[31] = _tmem_load_2[31] * alpha_v;
                _tmem_load_3[31] = _tmem_load_3[31] * alpha_v;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    gate_pk[_lp + 16] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    up_pk[_lp + 16] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_4[32];
                tmem_ld_x16(&_tmem_load_4[0], lane_addr + 64);
                tmem_ld_x16(&_tmem_load_4[16], lane_addr + 64 + 16);
                float _tmem_load_5[32];
                tmem_ld_x16(&_tmem_load_5[0], lane_addr + 112 + 64);
                tmem_ld_x16(&_tmem_load_5[16], lane_addr + 112 + 64 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                _tmem_load_4[0] = _tmem_load_4[0] * alpha_v;
                _tmem_load_5[0] = _tmem_load_5[0] * alpha_v;
                _tmem_load_4[1] = _tmem_load_4[1] * alpha_v;
                _tmem_load_5[1] = _tmem_load_5[1] * alpha_v;
                _tmem_load_4[2] = _tmem_load_4[2] * alpha_v;
                _tmem_load_5[2] = _tmem_load_5[2] * alpha_v;
                _tmem_load_4[3] = _tmem_load_4[3] * alpha_v;
                _tmem_load_5[3] = _tmem_load_5[3] * alpha_v;
                _tmem_load_4[4] = _tmem_load_4[4] * alpha_v;
                _tmem_load_5[4] = _tmem_load_5[4] * alpha_v;
                _tmem_load_4[5] = _tmem_load_4[5] * alpha_v;
                _tmem_load_5[5] = _tmem_load_5[5] * alpha_v;
                _tmem_load_4[6] = _tmem_load_4[6] * alpha_v;
                _tmem_load_5[6] = _tmem_load_5[6] * alpha_v;
                _tmem_load_4[7] = _tmem_load_4[7] * alpha_v;
                _tmem_load_5[7] = _tmem_load_5[7] * alpha_v;
                _tmem_load_4[8] = _tmem_load_4[8] * alpha_v;
                _tmem_load_5[8] = _tmem_load_5[8] * alpha_v;
                _tmem_load_4[9] = _tmem_load_4[9] * alpha_v;
                _tmem_load_5[9] = _tmem_load_5[9] * alpha_v;
                _tmem_load_4[10] = _tmem_load_4[10] * alpha_v;
                _tmem_load_5[10] = _tmem_load_5[10] * alpha_v;
                _tmem_load_4[11] = _tmem_load_4[11] * alpha_v;
                _tmem_load_5[11] = _tmem_load_5[11] * alpha_v;
                _tmem_load_4[12] = _tmem_load_4[12] * alpha_v;
                _tmem_load_5[12] = _tmem_load_5[12] * alpha_v;
                _tmem_load_4[13] = _tmem_load_4[13] * alpha_v;
                _tmem_load_5[13] = _tmem_load_5[13] * alpha_v;
                _tmem_load_4[14] = _tmem_load_4[14] * alpha_v;
                _tmem_load_5[14] = _tmem_load_5[14] * alpha_v;
                _tmem_load_4[15] = _tmem_load_4[15] * alpha_v;
                _tmem_load_5[15] = _tmem_load_5[15] * alpha_v;
                _tmem_load_4[16] = _tmem_load_4[16] * alpha_v;
                _tmem_load_5[16] = _tmem_load_5[16] * alpha_v;
                _tmem_load_4[17] = _tmem_load_4[17] * alpha_v;
                _tmem_load_5[17] = _tmem_load_5[17] * alpha_v;
                _tmem_load_4[18] = _tmem_load_4[18] * alpha_v;
                _tmem_load_5[18] = _tmem_load_5[18] * alpha_v;
                _tmem_load_4[19] = _tmem_load_4[19] * alpha_v;
                _tmem_load_5[19] = _tmem_load_5[19] * alpha_v;
                _tmem_load_4[20] = _tmem_load_4[20] * alpha_v;
                _tmem_load_5[20] = _tmem_load_5[20] * alpha_v;
                _tmem_load_4[21] = _tmem_load_4[21] * alpha_v;
                _tmem_load_5[21] = _tmem_load_5[21] * alpha_v;
                _tmem_load_4[22] = _tmem_load_4[22] * alpha_v;
                _tmem_load_5[22] = _tmem_load_5[22] * alpha_v;
                _tmem_load_4[23] = _tmem_load_4[23] * alpha_v;
                _tmem_load_5[23] = _tmem_load_5[23] * alpha_v;
                _tmem_load_4[24] = _tmem_load_4[24] * alpha_v;
                _tmem_load_5[24] = _tmem_load_5[24] * alpha_v;
                _tmem_load_4[25] = _tmem_load_4[25] * alpha_v;
                _tmem_load_5[25] = _tmem_load_5[25] * alpha_v;
                _tmem_load_4[26] = _tmem_load_4[26] * alpha_v;
                _tmem_load_5[26] = _tmem_load_5[26] * alpha_v;
                _tmem_load_4[27] = _tmem_load_4[27] * alpha_v;
                _tmem_load_5[27] = _tmem_load_5[27] * alpha_v;
                _tmem_load_4[28] = _tmem_load_4[28] * alpha_v;
                _tmem_load_5[28] = _tmem_load_5[28] * alpha_v;
                _tmem_load_4[29] = _tmem_load_4[29] * alpha_v;
                _tmem_load_5[29] = _tmem_load_5[29] * alpha_v;
                _tmem_load_4[30] = _tmem_load_4[30] * alpha_v;
                _tmem_load_5[30] = _tmem_load_5[30] * alpha_v;
                _tmem_load_4[31] = _tmem_load_4[31] * alpha_v;
                _tmem_load_5[31] = _tmem_load_5[31] * alpha_v;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    gate_pk[_lp + 32] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    up_pk[_lp + 32] = *(uint32_t*)&_bf2;
                }
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], lane_addr + 96);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], lane_addr + 112 + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                _tmem_load_6[0] = _tmem_load_6[0] * alpha_v;
                _tmem_load_7[0] = _tmem_load_7[0] * alpha_v;
                _tmem_load_6[1] = _tmem_load_6[1] * alpha_v;
                _tmem_load_7[1] = _tmem_load_7[1] * alpha_v;
                _tmem_load_6[2] = _tmem_load_6[2] * alpha_v;
                _tmem_load_7[2] = _tmem_load_7[2] * alpha_v;
                _tmem_load_6[3] = _tmem_load_6[3] * alpha_v;
                _tmem_load_7[3] = _tmem_load_7[3] * alpha_v;
                _tmem_load_6[4] = _tmem_load_6[4] * alpha_v;
                _tmem_load_7[4] = _tmem_load_7[4] * alpha_v;
                _tmem_load_6[5] = _tmem_load_6[5] * alpha_v;
                _tmem_load_7[5] = _tmem_load_7[5] * alpha_v;
                _tmem_load_6[6] = _tmem_load_6[6] * alpha_v;
                _tmem_load_7[6] = _tmem_load_7[6] * alpha_v;
                _tmem_load_6[7] = _tmem_load_6[7] * alpha_v;
                _tmem_load_7[7] = _tmem_load_7[7] * alpha_v;
                _tmem_load_6[8] = _tmem_load_6[8] * alpha_v;
                _tmem_load_7[8] = _tmem_load_7[8] * alpha_v;
                _tmem_load_6[9] = _tmem_load_6[9] * alpha_v;
                _tmem_load_7[9] = _tmem_load_7[9] * alpha_v;
                _tmem_load_6[10] = _tmem_load_6[10] * alpha_v;
                _tmem_load_7[10] = _tmem_load_7[10] * alpha_v;
                _tmem_load_6[11] = _tmem_load_6[11] * alpha_v;
                _tmem_load_7[11] = _tmem_load_7[11] * alpha_v;
                _tmem_load_6[12] = _tmem_load_6[12] * alpha_v;
                _tmem_load_7[12] = _tmem_load_7[12] * alpha_v;
                _tmem_load_6[13] = _tmem_load_6[13] * alpha_v;
                _tmem_load_7[13] = _tmem_load_7[13] * alpha_v;
                _tmem_load_6[14] = _tmem_load_6[14] * alpha_v;
                _tmem_load_7[14] = _tmem_load_7[14] * alpha_v;
                _tmem_load_6[15] = _tmem_load_6[15] * alpha_v;
                _tmem_load_7[15] = _tmem_load_7[15] * alpha_v;
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    gate_pk[_lp + 48] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                    up_pk[_lp + 48] = *(uint32_t*)&_bf2;
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                asm volatile("" : "+r"(gate_pk[0]) :: "memory");
                asm volatile("" : "+r"(gate_pk[1]) :: "memory");
                asm volatile("" : "+r"(gate_pk[2]) :: "memory");
                asm volatile("" : "+r"(gate_pk[3]) :: "memory");
                asm volatile("" : "+r"(gate_pk[4]) :: "memory");
                asm volatile("" : "+r"(gate_pk[5]) :: "memory");
                asm volatile("" : "+r"(gate_pk[6]) :: "memory");
                asm volatile("" : "+r"(gate_pk[7]) :: "memory");
                asm volatile("" : "+r"(gate_pk[8]) :: "memory");
                asm volatile("" : "+r"(gate_pk[9]) :: "memory");
                asm volatile("" : "+r"(gate_pk[10]) :: "memory");
                asm volatile("" : "+r"(gate_pk[11]) :: "memory");
                asm volatile("" : "+r"(gate_pk[12]) :: "memory");
                asm volatile("" : "+r"(gate_pk[13]) :: "memory");
                asm volatile("" : "+r"(gate_pk[14]) :: "memory");
                asm volatile("" : "+r"(gate_pk[15]) :: "memory");
                asm volatile("" : "+r"(gate_pk[16]) :: "memory");
                asm volatile("" : "+r"(gate_pk[17]) :: "memory");
                asm volatile("" : "+r"(gate_pk[18]) :: "memory");
                asm volatile("" : "+r"(gate_pk[19]) :: "memory");
                asm volatile("" : "+r"(gate_pk[20]) :: "memory");
                asm volatile("" : "+r"(gate_pk[21]) :: "memory");
                asm volatile("" : "+r"(gate_pk[22]) :: "memory");
                asm volatile("" : "+r"(gate_pk[23]) :: "memory");
                asm volatile("" : "+r"(gate_pk[24]) :: "memory");
                asm volatile("" : "+r"(gate_pk[25]) :: "memory");
                asm volatile("" : "+r"(gate_pk[26]) :: "memory");
                asm volatile("" : "+r"(gate_pk[27]) :: "memory");
                asm volatile("" : "+r"(gate_pk[28]) :: "memory");
                asm volatile("" : "+r"(gate_pk[29]) :: "memory");
                asm volatile("" : "+r"(gate_pk[30]) :: "memory");
                asm volatile("" : "+r"(gate_pk[31]) :: "memory");
                asm volatile("" : "+r"(gate_pk[32]) :: "memory");
                asm volatile("" : "+r"(gate_pk[33]) :: "memory");
                asm volatile("" : "+r"(gate_pk[34]) :: "memory");
                asm volatile("" : "+r"(gate_pk[35]) :: "memory");
                asm volatile("" : "+r"(gate_pk[36]) :: "memory");
                asm volatile("" : "+r"(gate_pk[37]) :: "memory");
                asm volatile("" : "+r"(gate_pk[38]) :: "memory");
                asm volatile("" : "+r"(gate_pk[39]) :: "memory");
                asm volatile("" : "+r"(gate_pk[40]) :: "memory");
                asm volatile("" : "+r"(gate_pk[41]) :: "memory");
                asm volatile("" : "+r"(gate_pk[42]) :: "memory");
                asm volatile("" : "+r"(gate_pk[43]) :: "memory");
                asm volatile("" : "+r"(gate_pk[44]) :: "memory");
                asm volatile("" : "+r"(gate_pk[45]) :: "memory");
                asm volatile("" : "+r"(gate_pk[46]) :: "memory");
                asm volatile("" : "+r"(gate_pk[47]) :: "memory");
                asm volatile("" : "+r"(gate_pk[48]) :: "memory");
                asm volatile("" : "+r"(gate_pk[49]) :: "memory");
                asm volatile("" : "+r"(gate_pk[50]) :: "memory");
                asm volatile("" : "+r"(gate_pk[51]) :: "memory");
                asm volatile("" : "+r"(gate_pk[52]) :: "memory");
                asm volatile("" : "+r"(gate_pk[53]) :: "memory");
                asm volatile("" : "+r"(gate_pk[54]) :: "memory");
                asm volatile("" : "+r"(gate_pk[55]) :: "memory");
                asm volatile("" : "+r"(up_pk[0]) :: "memory");
                asm volatile("" : "+r"(up_pk[1]) :: "memory");
                asm volatile("" : "+r"(up_pk[2]) :: "memory");
                asm volatile("" : "+r"(up_pk[3]) :: "memory");
                asm volatile("" : "+r"(up_pk[4]) :: "memory");
                asm volatile("" : "+r"(up_pk[5]) :: "memory");
                asm volatile("" : "+r"(up_pk[6]) :: "memory");
                asm volatile("" : "+r"(up_pk[7]) :: "memory");
                asm volatile("" : "+r"(up_pk[8]) :: "memory");
                asm volatile("" : "+r"(up_pk[9]) :: "memory");
                asm volatile("" : "+r"(up_pk[10]) :: "memory");
                asm volatile("" : "+r"(up_pk[11]) :: "memory");
                asm volatile("" : "+r"(up_pk[12]) :: "memory");
                asm volatile("" : "+r"(up_pk[13]) :: "memory");
                asm volatile("" : "+r"(up_pk[14]) :: "memory");
                asm volatile("" : "+r"(up_pk[15]) :: "memory");
                asm volatile("" : "+r"(up_pk[16]) :: "memory");
                asm volatile("" : "+r"(up_pk[17]) :: "memory");
                asm volatile("" : "+r"(up_pk[18]) :: "memory");
                asm volatile("" : "+r"(up_pk[19]) :: "memory");
                asm volatile("" : "+r"(up_pk[20]) :: "memory");
                asm volatile("" : "+r"(up_pk[21]) :: "memory");
                asm volatile("" : "+r"(up_pk[22]) :: "memory");
                asm volatile("" : "+r"(up_pk[23]) :: "memory");
                asm volatile("" : "+r"(up_pk[24]) :: "memory");
                asm volatile("" : "+r"(up_pk[25]) :: "memory");
                asm volatile("" : "+r"(up_pk[26]) :: "memory");
                asm volatile("" : "+r"(up_pk[27]) :: "memory");
                asm volatile("" : "+r"(up_pk[28]) :: "memory");
                asm volatile("" : "+r"(up_pk[29]) :: "memory");
                asm volatile("" : "+r"(up_pk[30]) :: "memory");
                asm volatile("" : "+r"(up_pk[31]) :: "memory");
                asm volatile("" : "+r"(up_pk[32]) :: "memory");
                asm volatile("" : "+r"(up_pk[33]) :: "memory");
                asm volatile("" : "+r"(up_pk[34]) :: "memory");
                asm volatile("" : "+r"(up_pk[35]) :: "memory");
                asm volatile("" : "+r"(up_pk[36]) :: "memory");
                asm volatile("" : "+r"(up_pk[37]) :: "memory");
                asm volatile("" : "+r"(up_pk[38]) :: "memory");
                asm volatile("" : "+r"(up_pk[39]) :: "memory");
                asm volatile("" : "+r"(up_pk[40]) :: "memory");
                asm volatile("" : "+r"(up_pk[41]) :: "memory");
                asm volatile("" : "+r"(up_pk[42]) :: "memory");
                asm volatile("" : "+r"(up_pk[43]) :: "memory");
                asm volatile("" : "+r"(up_pk[44]) :: "memory");
                asm volatile("" : "+r"(up_pk[45]) :: "memory");
                asm volatile("" : "+r"(up_pk[46]) :: "memory");
                asm volatile("" : "+r"(up_pk[47]) :: "memory");
                asm volatile("" : "+r"(up_pk[48]) :: "memory");
                asm volatile("" : "+r"(up_pk[49]) :: "memory");
                asm volatile("" : "+r"(up_pk[50]) :: "memory");
                asm volatile("" : "+r"(up_pk[51]) :: "memory");
                asm volatile("" : "+r"(up_pk[52]) :: "memory");
                asm volatile("" : "+r"(up_pk[53]) :: "memory");
                asm volatile("" : "+r"(up_pk[54]) :: "memory");
                asm volatile("" : "+r"(up_pk[55]) :: "memory");
                _phase_mainloop_done ^= 1;
                #pragma unroll
                for (int n_chunk = 0; n_chunk < 7; n_chunk++) {
                    float out_vals[16];
                    #pragma unroll
                    for (int jp = 0; jp < 8; jp++) {
                        const int w = n_chunk * 8 + jp;
                        float gate_lo = __uint_as_float(gate_pk[w] << 16);
                        float up_lo = __uint_as_float(up_pk[w] << 16);
                        float _exp2_0 = approx_exp2((-gate_lo) * 1.442695f);
                        float sig_lo = approx_rcp(1.0f + _exp2_0);
                        float pair[2];
                        pair[0] = 0.0f;
                        pair[1] = gate_lo * sig_lo;
                        unsigned int word[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                            word[_lp] = *(uint32_t*)&_bf2;
                        }
                        float silu_lo = __uint_as_float(word[0]);
                        float pair_0[2];
                        pair_0[0] = 0.0f;
                        pair_0[1] = silu_lo * up_lo;
                        unsigned int word_1[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_0[_lp*2 + 0], pair_0[_lp*2+1 + 0]));
                            word_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        out_vals[jp * 2] = __uint_as_float(word_1[0]);
                        float gate_hi = __uint_as_float(gate_pk[w] & 4294901760u);
                        float up_hi = __uint_as_float(up_pk[w] & 4294901760u);
                        float _exp2_1 = approx_exp2((-gate_hi) * 1.442695f);
                        float sig_hi = approx_rcp(1.0f + _exp2_1);
                        float pair_2[2];
                        pair_2[0] = 0.0f;
                        pair_2[1] = gate_hi * sig_hi;
                        unsigned int word_3[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_2[_lp*2 + 0], pair_2[_lp*2+1 + 0]));
                            word_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        float silu_hi = __uint_as_float(word_3[0]);
                        float pair_4[2];
                        pair_4[0] = 0.0f;
                        pair_4[1] = silu_hi * up_hi;
                        unsigned int word_5[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_4[_lp*2 + 0], pair_4[_lp*2+1 + 0]));
                            word_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        out_vals[jp * 2 + 1] = __uint_as_float(word_5[0]);
                    }
                    float mags[16];
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        mags[j] = out_vals[j];
                    }
                    mags[0] = fabsf(mags[0]);
                    mags[1] = fabsf(mags[1]);
                    mags[2] = fabsf(mags[2]);
                    mags[3] = fabsf(mags[3]);
                    mags[4] = fabsf(mags[4]);
                    mags[5] = fabsf(mags[5]);
                    mags[6] = fabsf(mags[6]);
                    mags[7] = fabsf(mags[7]);
                    mags[8] = fabsf(mags[8]);
                    mags[9] = fabsf(mags[9]);
                    mags[10] = fabsf(mags[10]);
                    mags[11] = fabsf(mags[11]);
                    mags[12] = fabsf(mags[12]);
                    mags[13] = fabsf(mags[13]);
                    mags[14] = fabsf(mags[14]);
                    mags[15] = fabsf(mags[15]);
                    float absmax = mags[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        absmax = max_noftz(absmax, mags[_lr]);
                    }
                    float sf_value = g * (absmax * sixth);
                    uint16_t sf_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair) : "f"(zero_f32), "f"(sf_value));
                    unsigned int sf_byte = (unsigned int)sf_pair & 255;
                    unsigned int sf_exp = sf_byte >> 3 & 15;
                    unsigned int sf_mant = sf_byte & 7;
                    unsigned int normal_bits = sf_exp + 120 << 23 | sf_mant << 20;
                    float sf_normal = 0.0f;
                    sf_normal = reinterpret_cast<float*>(&normal_bits)[0];
                    float sf_subnormal = (float)sf_mant * 0.001953125f;
                    float sf_f = ((sf_exp == 0) ? sf_subnormal : sf_normal);
                    float out_scale_nonzero = approx_rcp(sf_f * inv_g);
                    float out_scale = ((absmax == 0.0f) ? 0.0f : out_scale_nonzero);
                    const float2 _scale2_2 = {out_scale, out_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(out_vals)[_ls], _scale2_2);
                    unsigned int packed[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(out_vals[0]), "f"(out_vals[1]), "f"(out_vals[2]), "f"(out_vals[3]), "f"(out_vals[4]), "f"(out_vals[5]), "f"(out_vals[6]), "f"(out_vals[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(out_vals[8]), "f"(out_vals[9]), "f"(out_vals[10]), "f"(out_vals[11]), "f"(out_vals[12]), "f"(out_vals[13]), "f"(out_vals[14]), "f"(out_vals[15]));
                    if (global_row < M) {
                        *(reinterpret_cast<unsigned int*>(y_q + (word_base + (unsigned long long)(n_chunk * 2))) + (0)) = packed[0];
                        *(reinterpret_cast<unsigned int*>(y_q + (word_base + (unsigned long long)(n_chunk * 2 + 1))) + (0)) = packed[1];
                        int block = bid_n_1 * 14 + acc_half * 7 + n_chunk;
                        int k_tile = block >> 2;
                        int sf_col = block & 3;
                        *(reinterpret_cast<unsigned char*>(y_sf + (sf_row_base + (unsigned long long)k_tile * 512 + (unsigned long long)sf_col)) + (0)) = (unsigned char)(sf_byte);
                    }
                }
                mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 10000000);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef MINIMAX_H3_MLP_INF
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_SFA_ALL_OFF
#undef SMEM_SMEM_SFA_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFA_ALL_STRIDE
#undef SMEM_SMEM_V10_OFF
#undef SMEM_SMEM_V10_STAGE_BYTES
#undef SMEM_SMEM_V10_STRIDE
#undef SMEM_SMEM_V11_OFF
#undef SMEM_SMEM_V11_STAGE_BYTES
#undef SMEM_SMEM_V11_STRIDE
#undef SMEM_SMEM_V12_OFF
#undef SMEM_SMEM_V12_STAGE_BYTES
#undef SMEM_SMEM_V12_STRIDE
#undef SMEM_SMEM_V13_OFF
#undef SMEM_SMEM_V13_STAGE_BYTES
#undef SMEM_SMEM_V13_STRIDE
#undef SMEM_SMEM_V14_OFF
#undef SMEM_SMEM_V14_STAGE_BYTES
#undef SMEM_SMEM_V14_STRIDE
#undef SMEM_SMEM_V15_OFF
#undef SMEM_SMEM_V15_STAGE_BYTES
#undef SMEM_SMEM_V15_STRIDE
#undef SMEM_SMEM_V16_OFF
#undef SMEM_SMEM_V16_STAGE_BYTES
#undef SMEM_SMEM_V16_STRIDE
#undef SMEM_SMEM_V17_OFF
#undef SMEM_SMEM_V17_STAGE_BYTES
#undef SMEM_SMEM_V17_STRIDE
#undef SMEM_SMEM_V18_OFF
#undef SMEM_SMEM_V18_STAGE_BYTES
#undef SMEM_SMEM_V18_STRIDE
#undef SMEM_SMEM_V2_OFF
#undef SMEM_SMEM_V2_STAGE_BYTES
#undef SMEM_SMEM_V2_STRIDE
#undef SMEM_SMEM_V3_OFF
#undef SMEM_SMEM_V3_STAGE_BYTES
#undef SMEM_SMEM_V3_STRIDE
#undef SMEM_SMEM_V5_OFF
#undef SMEM_SMEM_V5_STAGE_BYTES
#undef SMEM_SMEM_V5_STRIDE
#undef SMEM_SMEM_V6_OFF
#undef SMEM_SMEM_V6_STAGE_BYTES
#undef SMEM_SMEM_V6_STRIDE
#undef SMEM_SMEM_V7_OFF
#undef SMEM_SMEM_V7_STAGE_BYTES
#undef SMEM_SMEM_V7_STRIDE
#undef SMEM_SMEM_V8_OFF
#undef SMEM_SMEM_V8_STAGE_BYTES
#undef SMEM_SMEM_V8_STRIDE
#undef SMEM_SMEM_V9_OFF
#undef SMEM_SMEM_V9_STAGE_BYTES
#undef SMEM_SMEM_V9_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM0_OFFSET
#undef TMEM_ACCUM1_OFFSET
#undef TMEM_NCOLS
#undef TMEM_TMEM_SFA_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef tiles_per_group
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr

#define MINIMAX_H3_MLP_INF CUDART_INF_F
#define TMEM_NCOLS 304
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 272
#define NUM_TMA_PIPE_STAGES 5
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 38912
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 38912
#define SMEM_SMEM_SFA_ALL_OFF 33792
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_ALL_STRIDE 38912
#define SMEM_SMEM_SFB_ALL_OFF 35840
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_ALL_STRIDE 38912
#define SMEM_SMEM_V4_OFF 33792
#define SMEM_SMEM_V4_STAGE_BYTES 512
#define SMEM_SMEM_V4_STRIDE 38912
#define SMEM_SMEM_V5_OFF 34304
#define SMEM_SMEM_V5_STAGE_BYTES 512
#define SMEM_SMEM_V5_STRIDE 38912
#define SMEM_SMEM_V6_OFF 34816
#define SMEM_SMEM_V6_STAGE_BYTES 512
#define SMEM_SMEM_V6_STRIDE 38912
#define SMEM_SMEM_V7_OFF 35328
#define SMEM_SMEM_V7_STAGE_BYTES 512
#define SMEM_SMEM_V7_STRIDE 38912
#define SMEM_SMEM_V8_OFF 35840
#define SMEM_SMEM_V8_STAGE_BYTES 1024
#define SMEM_SMEM_V8_STRIDE 38912
#define SMEM_SMEM_V9_OFF 36864
#define SMEM_SMEM_V9_STAGE_BYTES 1024
#define SMEM_SMEM_V9_STRIDE 38912
#define SMEM_SMEM_V10_OFF 37888
#define SMEM_SMEM_V10_STAGE_BYTES 1024
#define SMEM_SMEM_V10_STRIDE 38912
#define SMEM_SMEM_V11_OFF 38912
#define SMEM_SMEM_V11_STAGE_BYTES 1024
#define SMEM_SMEM_V11_STRIDE 38912
#define SMEM_WORK_RESPONSE_OFF 195584
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 195712

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_mlp_fc2_e2m1(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride, float* __restrict__ partial, unsigned int* __restrict__ flags, int tail_base, int split_count, int group_m)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 40)
    #define mainloop_done_addr (mbar_base + 80)
    #define epilogue_done_addr (mbar_base + 88)
    #define work_full_addr (mbar_base + 96)
    #define work_empty_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_sfa_all_addr = smem + 33792;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_sfb_all_addr = smem + 35840;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v4_addr = smem + 33792;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 34304);
    const int smem_v5_addr = smem + 34304;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_v6_addr = smem + 34816;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 35328);
    const int smem_v7_addr = smem + 35328;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v8_addr = smem + 35840;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v9_addr = smem + 36864;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v10_addr = smem + 37888;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_v11_addr = smem + 38912;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 195584);
    const int work_response_addr = smem + 195584;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // mma_done: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 88, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 128, 546);
            mbarrier_init(smem + 136, 546);
            mbarrier_init(smem + 144, 546);
            mbarrier_init(smem + 152, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 304 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 272;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < m_tiles / 2 * 21 + split_count; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int this_bid_0 = (int)this_bid;
                    int tail_u = this_bid_0 - tail_base;
                    int in_tail = ((tail_u >= 0) ? 1 : 0);
                    int q = tail_u >> 1;
                    int is_h0 = ((q < split_count) ? 1 : 0);
                    int is_h1 = ((q >= split_count && q < 2 * split_count) ? 1 : 0);
                    int half_tail = ((is_h0 == 1) ? 0 : ((is_h1 == 1) ? 1 : -1));
                    int half = ((in_tail == 1) ? half_tail : -1);
                    int j = ((is_h0 == 1) ? q : q - split_count);
                    int slot = ((half >= 0) ? j : 0);
                    int tile_tail = tail_base + 2 * j + (tail_u & 1);
                    int tile = ((in_tail == 1) ? tile_tail : this_bid_0);
                    int k0 = ((half == 1) ? 28 : 0);
                    int k1 = ((half == 0) ? 28 : 56);
                    int group = tile / (group_m * 21);
                    int first_m = group * group_m;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= group_m) ? group_m : remaining);
                    int local = tile % (group_m * 21);
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * 128;
                    int off_n = bid_n * 256;
                    int weight_row = off_n + cta_rank * 128;
                    int sfa_tile_row = bid_m * 224;
                    int sfb_tile_row = bid_n * 224;
                    #pragma unroll 1
                    for (int iter_k = k0; iter_k < k1; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B), 0, weight_row, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        int k_set_base = iter_k * 4;
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB), 0, 0, sfb_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(38912)) : "memory");
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
                #pragma unroll 1
                for (int _ = 0; _ < 5; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int this_bid_m = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < m_tiles / 2 * 21 + split_count; _tile_iter_1++) {
                    int this_bid_1 = (int)this_bid_m;
                    int tail_u_1 = this_bid_1 - tail_base;
                    int in_tail_1 = ((tail_u_1 >= 0) ? 1 : 0);
                    int q_1 = tail_u_1 >> 1;
                    int is_h0_1 = ((q_1 < split_count) ? 1 : 0);
                    int is_h1_1 = ((q_1 >= split_count && q_1 < 2 * split_count) ? 1 : 0);
                    int half_tail_1 = ((is_h0_1 == 1) ? 0 : ((is_h1_1 == 1) ? 1 : -1));
                    int half_1 = ((in_tail_1 == 1) ? half_tail_1 : -1);
                    int j_1 = ((is_h0_1 == 1) ? q_1 : q_1 - split_count);
                    int slot_1 = ((half_1 >= 0) ? j_1 : 0);
                    int tile_tail_1 = tail_base + 2 * j_1 + (tail_u_1 & 1);
                    int tile_1 = ((in_tail_1 == 1) ? tile_tail_1 : this_bid_1);
                    int k0_1 = ((half_1 == 1) ? 28 : 0);
                    int k1_1 = ((half_1 == 0) ? 28 : 56);
                    int group_1 = tile_1 / (group_m * 21);
                    int first_m_1 = group_1 * group_m;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= group_m) ? group_m : remaining_1);
                    int local_1 = tile_1 % (group_m * 21);
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * 128;
                    int off_n_1 = bid_n_1 * 256;
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = k0_1; iter_k_1 < k1_1; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == k0_1) ? 1 : 0);
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
                        }
                        int _smem_desc_lo_3 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_4 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
                        }
                        int _smem_desc_lo_5 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_6 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_0);
                        }
                        int _smem_desc_lo_7 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_8 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_0);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                    this_bid_m = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            float alpha_v = alpha[0];
            int epi_cta_rank = (int)cta_rank;
            unsigned int this_bid_2 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < m_tiles / 2 * 21 + split_count; _tile_iter_2++) {
                int this_bid_0_1 = (int)this_bid_2;
                int tail_u_2 = this_bid_0_1 - tail_base;
                int in_tail_2 = ((tail_u_2 >= 0) ? 1 : 0);
                int q_2 = tail_u_2 >> 1;
                int is_h0_2 = ((q_2 < split_count) ? 1 : 0);
                int is_h1_2 = ((q_2 >= split_count && q_2 < 2 * split_count) ? 1 : 0);
                int half_tail_2 = ((is_h0_2 == 1) ? 0 : ((is_h1_2 == 1) ? 1 : -1));
                int half_2 = ((in_tail_2 == 1) ? half_tail_2 : -1);
                int j_2 = ((is_h0_2 == 1) ? q_2 : q_2 - split_count);
                int slot_2 = ((half_2 >= 0) ? j_2 : 0);
                int tile_tail_2 = tail_base + 2 * j_2 + (tail_u_2 & 1);
                int tile_2 = ((in_tail_2 == 1) ? tile_tail_2 : this_bid_0_1);
                int k0_2 = ((half_2 == 1) ? 28 : 0);
                int k1_2 = ((half_2 == 0) ? 28 : 56);
                int group_2 = tile_2 / (group_m * 21);
                int first_m_2 = group_2 * group_m;
                int remaining_2 = m_tiles - first_m_2;
                int group_size_2 = ((remaining_2 >= group_m) ? group_m : remaining_2);
                int local_2 = tile_2 % (group_m * 21);
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int off_m_2 = bid_m_2 * 128;
                int off_n_2 = bid_n_2 * 256;
                int global_row = off_m_2 + local_row;
                int col0 = col_part * 128;
                int flag_idx = slot_2 * 2 + epi_cta_rank;
                int partial_row = flag_idx * 128 + local_row;
                unsigned long long partial_base = (unsigned long long)partial_row * 256 + (unsigned long long)col0;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_2 + (unsigned long long)col0;
                unsigned long long row_base = (unsigned long long)global_row * 5376 + (unsigned long long)off_n_2 + (unsigned long long)col0;
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                    : "r"(lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                if (half_2 == 0) {
                    #pragma unroll
                    for (int q_1_1 = 0; q_1_1 < 32; q_1_1++) {
                        float part_q[4];
                        #pragma unroll
                        for (int j_1_1 = 0; j_1_1 < 4; j_1_1++) {
                            part_q[j_1_1] = _tmem_load_0[q_1_1 * 4 + j_1_1];
                        }
                        {
                            float4 _v4 = make_float4(part_q[0 + 0], part_q[0 + 1], part_q[0 + 2], part_q[0 + 3]);
                            *reinterpret_cast<float4*>((partial + (partial_base + (unsigned long long)(q_1_1 * 4))) + 0) = _v4;
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
                if (half_2 == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(8)) >= static_cast<unsigned int>(1));
                    }
                    #pragma unroll
                    for (int q_1_2 = 0; q_1_2 < 32; q_1_2++) {
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (partial_base + (unsigned long long)(q_1_2 * 4)) + 0);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int j_1_2 = 0; j_1_2 < 4; j_1_2++) {
                            _tmem_load_0[q_1_2 * 4 + j_1_2] = _tmem_load_0[q_1_2 * 4 + j_1_2] + _vec_load_0[j_1_2];
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (flag_idx))), "r"(static_cast<unsigned int>(4294967295u)) : "memory");
                    }
                }
                if (half_2 != 0) {
                    if (global_row < M) {
                        #pragma unroll
                        for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                            int col = n_chunk * 16;
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                                uint4 _vld_1[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_1[_blk] = _vptr_1[_blk];
                                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_2[8];
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
                                uint4 _vld_2[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_3[8];
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
                                uint4 _vld_3[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float out_vals[16];
                            #pragma unroll
                            for (int j_1_3 = 0; j_1_3 < 8; j_1_3++) {
                                float o_lo = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j_1_3] * alpha_v);
                                float p_lo = (float)(__nv_bfloat16)(_vec_load_1[j_1_3] * gate_mul * o_lo);
                                out_vals[j_1_3] = _vec_load_3[j_1_3] + p_lo;
                                float o_hi = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j_1_3 + 8] * alpha_v);
                                float p_hi = (float)(__nv_bfloat16)(_vec_load_2[j_1_3] * gate_mul * o_hi);
                                out_vals[j_1_3 + 8] = _vec_load_4[j_1_3] + p_hi;
                            }
                            unsigned int _packed_0[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 0], out_vals[_lp*2+1 + 0]));
                                _packed_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 2], out_vals[_lp*2+1 + 2]));
                                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 4], out_vals[_lp*2+1 + 4]));
                                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 6], out_vals[_lp*2+1 + 6]));
                                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 8], out_vals[_lp*2+1 + 8]));
                                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 10], out_vals[_lp*2+1 + 10]));
                                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 12], out_vals[_lp*2+1 + 12]));
                                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals[_lp*2 + 14], out_vals[_lp*2+1 + 14]));
                                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_2 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

#undef MINIMAX_H3_MLP_INF
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_SFA_ALL_OFF
#undef SMEM_SMEM_SFA_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFA_ALL_STRIDE
#undef SMEM_SMEM_SFB_ALL_OFF
#undef SMEM_SMEM_SFB_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFB_ALL_STRIDE
#undef SMEM_SMEM_V10_OFF
#undef SMEM_SMEM_V10_STAGE_BYTES
#undef SMEM_SMEM_V10_STRIDE
#undef SMEM_SMEM_V11_OFF
#undef SMEM_SMEM_V11_STAGE_BYTES
#undef SMEM_SMEM_V11_STRIDE
#undef SMEM_SMEM_V4_OFF
#undef SMEM_SMEM_V4_STAGE_BYTES
#undef SMEM_SMEM_V4_STRIDE
#undef SMEM_SMEM_V5_OFF
#undef SMEM_SMEM_V5_STAGE_BYTES
#undef SMEM_SMEM_V5_STRIDE
#undef SMEM_SMEM_V6_OFF
#undef SMEM_SMEM_V6_STAGE_BYTES
#undef SMEM_SMEM_V6_STRIDE
#undef SMEM_SMEM_V7_OFF
#undef SMEM_SMEM_V7_STAGE_BYTES
#undef SMEM_SMEM_V7_STRIDE
#undef SMEM_SMEM_V8_OFF
#undef SMEM_SMEM_V8_STAGE_BYTES
#undef SMEM_SMEM_V8_STRIDE
#undef SMEM_SMEM_V9_OFF
#undef SMEM_SMEM_V9_STAGE_BYTES
#undef SMEM_SMEM_V9_STRIDE
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef TMEM_TMEM_SFA_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr

#include <cuda_runtime.h>

#include <mutex>
#include <vector>

#include "tvm_ffi_utils.h"

// The one 10.x minor this build serves: 0 for sm_100a, 3 for sm_103a.  The JIT builder passes
// it next to the matching -gencode (the device code above already branches on __CUDA_ARCH__
// wherever 10.0 and 10.3 differ), so one translation unit compiled once per target replaces
// the former per-target copies while each module keeps its exact-capability check.
#ifndef MINIMAX_H3_SM_MINOR
#error "define MINIMAX_H3_SM_MINOR (0 for sm_100a, 3 for sm_103a) to match the -gencode target"
#endif
static_assert(MINIMAX_H3_SM_MINOR == 0 || MINIMAX_H3_SM_MINOR == 3,
              "MINIMAX_H3_SM_MINOR must be 0 (sm_100a) or 3 (sm_103a)");

namespace {

constexpr int64_t kHidden = 5376;       // x / a / out columns, fc2_weight rows
constexpr int64_t kFfn = 14336;             // y columns, fc2_weight columns (the FC2 reduction)
constexpr int64_t kFc1Rows = 28672;       // fc1_weight rows: [gate rows; up rows]
constexpr int64_t kMaxRows = 16777216;
constexpr int64_t kMaxTableRows = 2147483647;  // adaln_rows / gate_rows are int kernel arguments
constexpr int64_t kTableRowAlign = 8;          // elements: 16-byte rows for the 128-bit table loads
constexpr int64_t kBlockM = 128;
constexpr int64_t kCtaGroup = 2;

// Kernel 1 (BF16, MXFP8): one warp per row, kNormRowsPerCta rows per CTA.
constexpr int kNormThreads = 128;
constexpr int kNormRowsPerCta = 4;
constexpr int kNormBf16Smem = 0;
constexpr int kNormMxfp8Smem = 0;
// Kernel 1 (NVFP4): persistent row stream, one CTA per SM (see CheckGrids) of kNormNvfp4RowGroups row
// groups (three warps each, one SMEM row buffer each) plus x_norm_weight widened to f32 in SMEM.
constexpr int kNormNvfp4Threads = 768;
constexpr int kNormNvfp4Smem = 108672;
constexpr int64_t kNormNvfp4RowGroups = 8;

// Kernel 2 (all variants): persistent cta_group::2 FC1 GEMM + SwiGLU, one 128-row x N-half tile per CTA.
constexpr int kFc1Bf16Threads = 192;
constexpr int kFc1Mxfp8Threads = 320;
constexpr int kFc1Mxfp8W448Threads = 320;
constexpr int kFc1Nvfp4Threads = 320;
constexpr int kFc1Bf16Smem = 230528;
constexpr int kFc1Mxfp8Smem = 206976;
constexpr int kFc1Mxfp8W448Smem = 195712;
constexpr int kFc1Nvfp4Smem = 222336;
constexpr int64_t kFc1Bf16NTiles = 112;    // 128 y columns per tile (256x256 pair tile)
constexpr int64_t kFc1Mxfp8NTiles = 112;  // 128 y columns per tile (256x256 pair tile)
constexpr int64_t kFc1Mxfp8W448NTiles = 64;  // 224 y columns per tile (256x448 pair tile)
constexpr int64_t kFc1Nvfp4NTiles = 64;  // 224 y columns per tile (256x448 pair tile)
// Combined 224-row (112 gate + 112 up) weight scale sub-tiles of the 256x448 pair tile.
constexpr int64_t kFc1Mxfp8W448WeightSfTiles = 128;
constexpr int64_t kFc1Nvfp4WeightSfTiles = 128;

// Kernel 3 (all variants): persistent cta_group::2 FC2 GEMM, one 128-row x 256-column tile per CTA pair.
constexpr int kFc2Bf16Threads = 192;
constexpr int kFc2Mxfp8Threads = 320;
constexpr int kFc2Nvfp4Threads = 320;
constexpr int kFc2Bf16Smem = 230528;
constexpr int kFc2Mxfp8Smem = 206976;
constexpr int kFc2Nvfp4Smem = 195712;
constexpr int64_t kFc2NTiles = 21;    // 256 output columns per CTA pair: 21 column tiles
constexpr int64_t kFc2BlockN = 256;
// Row tiles per raster group (runtime kernel argument; the group that keeps the activation block L2-resident).
constexpr int kFc2GroupMBf16 = 16;
constexpr int kFc2GroupMMxfp8 = 16;
constexpr int kFc2GroupMNvfp4 = 8;
constexpr unsigned int kClusterX = 2u;
constexpr unsigned int kClusterY = 1u;
constexpr unsigned int kClusterZ = 1u;
// Programmatic dependent launch (cudaLaunchAttributeProgrammaticStreamSerialization): the FC2 GEMM may
// start while the FC1 launch on the stream is still running; the kernel itself waits
// (griddepcontrol.wait) before touching y / y_sf.  Resolved from the kernel module.
constexpr bool kFc2Bf16Pdl = true;
constexpr bool kFc2Mxfp8Pdl = true;
constexpr bool kFc2Nvfp4Pdl = true;
// The 256x448 MXFP8 FC1 GEMM and the NVFP4 FC1 GEMM are programmatic dependents of the norm launch:
// their prologue (barrier init, TMEM allocation, descriptor prefetch) overlaps the norm and they wait
// (griddepcontrol.wait) before their first activation load.
constexpr bool kFc1Mxfp8W448Pdl = true;
constexpr bool kFc1Nvfp4Pdl = true;

// FC2 tail split-K (the caller decides it, see CheckGrids): the idle clusters of a final partial
// wave take the second K halves of its tiles, which add the first half's FP32 partial before the
// epilogue.
// Per-device workspace: one slot per split tile, [slot][2 CTAs][128 rows][256 cols] FP32 partials and
// [slot][2 CTAs] counters; slots = clusters / 2 + 1 covers split_count <= clusters / 2.
constexpr int64_t kFc2PartialFloatsPerSlot = kCtaGroup * kBlockM * kFc2BlockN;
constexpr int64_t kFc2FlagsPerSlot = kCtaGroup;

// Quantized operand layouts (FlashInfer 128x4 swizzled scale tiles: 512-byte tiles of 128 rows x 4
// K-blocks, byte (row % 32) * 16 + (row / 32) * 4 + kblock).
constexpr int64_t kMxBlock = 32;
constexpr int64_t kNvBlock = 16;
constexpr int64_t kSfTileBytes = 512;
constexpr int64_t kSfbTileBytes = 2 * kSfTileBytes;  // combined 256-row weight tile per K-set (two 128-row tiles)
constexpr int64_t kAMxfp8SfKTiles = 42;  // 512-byte scale tiles per 128-row block of a (K = 5376)
constexpr int64_t kANvfp4SfKTiles = 84;
constexpr int64_t kYMxfp8SfKTiles = 112;  // 512-byte scale tiles per 128-row block of y (K = 14336)
constexpr int64_t kYNvfp4SfKTiles = 224;
constexpr int64_t kNvfp4APackedCols = kHidden / 2;  // E2M1 nibble pairs per row of a / fc1_weight_q
constexpr int64_t kNvfp4YPackedCols = kFfn / 2;     // E2M1 nibble pairs per row of y / fc2_weight_q
constexpr int64_t kMxfp8Fc1ScaleBytes = kFc1Mxfp8NTiles * kAMxfp8SfKTiles * kSfbTileBytes;
constexpr int64_t kMxfp8Fc1W448ScaleBytes = kFc1Mxfp8W448WeightSfTiles * kAMxfp8SfKTiles * kSfbTileBytes;
constexpr int64_t kNvfp4Fc1ScaleBytes = kFc1Nvfp4WeightSfTiles * kANvfp4SfKTiles * kSfbTileBytes;
constexpr int64_t kMxfp8Fc2ScaleBytes = kFc2NTiles * kYMxfp8SfKTiles * kSfbTileBytes;
constexpr int64_t kNvfp4Fc2ScaleBytes = kFc2NTiles * kYNvfp4SfKTiles * kSfbTileBytes;

// TMA boxes (resolved from the kernel descriptors at export time).
constexpr uint32_t kFc1Bf16BoxK = 64;
constexpr uint32_t kFc1Bf16BoxRowsA = 128;
constexpr uint32_t kFc1Bf16BoxRowsB = 128;
constexpr uint32_t kFc1Bf16BoxGroups = 1;
constexpr uint32_t kFc1Mxfp8BoxK = 128;
constexpr uint32_t kFc1Mxfp8BoxRowsA = 128;
constexpr uint32_t kFc1Mxfp8BoxRowsB = 128;
constexpr uint32_t kFc1Mxfp8BoxGroups = 2;
constexpr uint32_t kFc1Mxfp8W448BoxK = 128;
constexpr uint32_t kFc1Mxfp8W448BoxRowsA = 128;
constexpr uint32_t kFc1Mxfp8W448BoxRowsB = 224;
constexpr uint32_t kFc1Mxfp8W448BoxGroups = 1;
constexpr uint32_t kFc1Nvfp4BoxK = 128;
constexpr uint32_t kFc1Nvfp4BoxRowsA = 128;
constexpr uint32_t kFc1Nvfp4BoxRowsB = 224;
constexpr uint32_t kFc1Nvfp4BoxGroups = 1;
constexpr uint32_t kFc2Bf16BoxK = 64;
constexpr uint32_t kFc2Bf16BoxRowsA = 128;
constexpr uint32_t kFc2Bf16BoxRowsB = 128;
constexpr uint32_t kFc2Bf16BoxGroups = 1;
constexpr uint32_t kFc2Mxfp8BoxK = 128;
constexpr uint32_t kFc2Mxfp8BoxRowsA = 128;
constexpr uint32_t kFc2Mxfp8BoxRowsB = 128;
constexpr uint32_t kFc2Mxfp8BoxGroups = 2;
constexpr uint32_t kFc2Nvfp4BoxK = 128;
constexpr uint32_t kFc2Nvfp4BoxRowsA = 128;
constexpr uint32_t kFc2Nvfp4BoxRowsB = 128;
constexpr uint32_t kFc2Nvfp4BoxGroups = 1;
constexpr uint32_t kSfaTileRows = 4;  // one 512-byte tile = 4 rows x 128 bytes
constexpr uint32_t kSfbTileRows = 8;  // one 1024-byte tile = 8 rows x 128 bytes
constexpr uint32_t kSfTileRowBytes = 128;
// Consecutive scale tiles fetched per TMA load (one per K set of a pipeline stage).
constexpr uint32_t kFc1Mxfp8SfaBoxTiles = 2;
constexpr uint32_t kFc1Mxfp8SfbBoxTiles = 2;
constexpr uint32_t kFc1Mxfp8W448SfaBoxTiles = 1;
constexpr uint32_t kFc1Mxfp8W448SfbBoxTiles = 1;
constexpr uint32_t kFc1Nvfp4SfaBoxTiles = 4;
constexpr uint32_t kFc1Nvfp4SfbBoxTiles = 4;
constexpr uint32_t kFc2Mxfp8SfaBoxTiles = 2;
constexpr uint32_t kFc2Mxfp8SfbBoxTiles = 2;
constexpr uint32_t kFc2Nvfp4SfaBoxTiles = 4;
constexpr uint32_t kFc2Nvfp4SfbBoxTiles = 4;

int64_t MTiles(int64_t rows) {
  // Row tiles are consumed in CTA pairs: round the tile count up to an even number.
  int64_t tiles = (rows + kBlockM - 1) / kBlockM;
  return tiles + tiles % kCtaGroup;
}

// Launch grids and FC2 scheduler arguments, computed by the caller from the row count and the SM count.
// The BF16 and MXFP8 norms run one warp per row, ceil(M / kNormRowsPerCta) CTAs.  The NVFP4 norm
// (norm_row_groups > 0) is a persistent row stream: CTA c of a g-CTA grid takes the rows [c * M / g,
// (c + 1) * M / g), so every grid of 1 to ceil(M / norm_row_groups) CTAs writes the same bytes; the caller
// launches min(SMs, ceil(M / norm_row_groups)).  Each GEMM launches one cluster (CTA pair) per output tile
// pair; the hardware launches as many clusters as fit and running clusters claim the remaining tiles in
// order through cluster launch control, so the GEMM grids are exact.  The FC2 grid adds split_count second
// K halves of the final partial wave's tiles, which starts at CTA tail_base; fc2_clusters (the device's CTA
// pairs) sizes the split-K workspace.
struct LaunchGrids {
  int64_t norm;
  int64_t fc1;
  int64_t fc2;
  int64_t fc2_tail_base;
  int64_t fc2_split_count;
  int64_t fc2_clusters;
};

LaunchGrids CheckGrids(int64_t rows, int64_t m_tiles, int64_t fc1_n_tiles, int64_t norm_grid, int64_t fc1_grid,
                       int64_t fc2_grid, int64_t fc2_tail_base, int64_t fc2_split_count, int64_t fc2_clusters,
                       int64_t norm_row_groups = 0) {
  const int64_t pairs = m_tiles / kCtaGroup;
  const int64_t fc2_tiles = pairs * kFc2NTiles;
  if (norm_row_groups > 0) {
    const int64_t norm_max = (rows + norm_row_groups - 1) / norm_row_groups;
    TVM_FFI_CHECK(norm_grid >= 1 && norm_grid <= norm_max, ValueError)
        << "norm_grid must be in [1, ceil(M / " << norm_row_groups << ")] = [1, " << norm_max << "], got "
        << norm_grid;
  } else {
    TVM_FFI_CHECK(norm_grid == (rows + kNormRowsPerCta - 1) / kNormRowsPerCta, ValueError)
        << "norm_grid must be ceil(M / " << kNormRowsPerCta << "), got " << norm_grid;
  }
  TVM_FFI_CHECK(fc1_grid == pairs * fc1_n_tiles * kCtaGroup, ValueError)
      << "fc1_grid must be " << pairs * fc1_n_tiles * kCtaGroup << " (one CTA pair per tile pair), got " << fc1_grid;
  TVM_FFI_CHECK(fc2_clusters >= 1 && fc2_split_count >= 0 && fc2_split_count <= fc2_clusters / 2, ValueError)
      << "fc2_split_count must be in [0, fc2_clusters / 2] with fc2_clusters >= 1, got " << fc2_split_count << " / "
      << fc2_clusters;
  TVM_FFI_CHECK(fc2_grid == (fc2_tiles + fc2_split_count) * kCtaGroup, ValueError)
      << "fc2_grid must be " << (fc2_tiles + fc2_split_count) * kCtaGroup << ", got " << fc2_grid;
  TVM_FFI_CHECK(fc2_tail_base >= 0 && fc2_tail_base <= fc2_tiles * kCtaGroup && fc2_tail_base % kCtaGroup == 0 &&
                    (fc2_split_count == 0 || fc2_tail_base == (fc2_tiles - fc2_split_count) * kCtaGroup),
                ValueError)
      << "fc2_tail_base must be the first CTA of the final partial wave, got " << fc2_tail_base;
  return LaunchGrids{norm_grid, fc1_grid, fc2_grid, fc2_tail_base, fc2_split_count, fc2_clusters};
}

void CheckDeviceTensor(const TensorView& tensor, const char* name, DLDevice device, int64_t alignment) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as x";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, ValueError)
      << name << " must be " << alignment << "-byte aligned";
}

void CheckDtype(const TensorView& tensor, const char* name, DLDataType dtype, const char* dtype_name) {
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " must be " << dtype_name;
}

void CheckMatrix(const TensorView& tensor, const char* name, int64_t rows, int64_t cols, DLDataType dtype,
                 const char* dtype_name, DLDevice device, int64_t alignment) {
  CheckDeviceTensor(tensor, name, device, alignment);
  CheckDtype(tensor, name, dtype, dtype_name);
  TVM_FFI_CHECK(tensor.ndim() == 2 && tensor.size(0) == rows && tensor.size(1) == cols, ValueError)
      << name << " must have shape [" << rows << ", " << cols << "]";
  TVM_FFI_CHECK(tensor.stride(1) == 1 && tensor.stride(0) == cols, ValueError) << name << " must be contiguous";
}

void CheckVector(const TensorView& tensor, const char* name, int64_t length, DLDataType dtype, const char* dtype_name,
                 DLDevice device, int64_t alignment) {
  CheckDeviceTensor(tensor, name, device, alignment);
  CheckDtype(tensor, name, dtype, dtype_name);
  TVM_FFI_CHECK(tensor.ndim() == 1 && tensor.size(0) == length, ValueError)
      << name << " must have shape [" << length << "]";
  TVM_FFI_CHECK(tensor.stride(0) == 1, ValueError) << name << " must be contiguous";
}

void CheckByteBuffer(const TensorView& tensor, const char* name, int64_t min_bytes, DLDevice device) {
  CheckDeviceTensor(tensor, name, device, 16);
  CheckDtype(tensor, name, dl_uint8, "uint8");
  TVM_FFI_CHECK(tensor.ndim() == 1 && tensor.stride(0) == 1, ValueError) << name << " must be a contiguous 1-D tensor";
  TVM_FFI_CHECK(tensor.size(0) >= min_bytes, ValueError) << name << " must hold at least " << min_bytes << " bytes";
}

void CheckScaleTiles(const TensorView& tensor, const char* name, int64_t bytes, DLDevice device) {
  CheckByteBuffer(tensor, name, bytes, device);
  TVM_FFI_CHECK(tensor.size(0) == bytes, ValueError) << name << " must hold exactly " << bytes << " bytes";
}

void CheckScalar(const TensorView& tensor, const char* name, DLDevice device) {
  CheckDeviceTensor(tensor, name, device, 4);
  CheckDtype(tensor, name, dl_float32, "float32");
  TVM_FFI_CHECK(tensor.ndim() == 1 && tensor.size(0) == 1, ValueError) << name << " must have shape [1]";
}

int64_t CheckRows(const TensorView& x) {
  TVM_FFI_CHECK(x.device().device_type == kDLCUDA, ValueError) << "x must be a CUDA tensor";
  TVM_FFI_CHECK(x.ndim() == 2, ValueError) << "x must be a rank-2 tensor [M, " << kHidden << "]";
  const int64_t rows = x.size(0);
  TVM_FFI_CHECK(rows >= 1 && rows <= kMaxRows, ValueError) << "M must satisfy 1 <= M <= " << kMaxRows;
  return rows;
}

// Modulation tables (adaln_scale, adaln_shift, gate): bf16 [rows, kHidden] with unit column stride,
// 1 <= rows <= kMaxTableRows, ONE common row stride that is a multiple of kTableRowAlign elements in
// [kHidden, 2^32) and 16-byte-aligned bases.  Three contiguous [rows, 5376] tables and the shift / scale /
// gate column chunks of one [rows, 6 * 5376] projection both qualify; the kernels read table row r at
// base + r * row_stride and treat an index outside [0, rows) as the zero row / zero gate.
struct ModulationTable {
  int64_t rows;
  int64_t row_stride;  // elements between consecutive rows
};

void CheckTable(const TensorView& tensor, const char* name, DLDevice device) {
  CheckDeviceTensor(tensor, name, device, 16);
  CheckDtype(tensor, name, dl_bfloat16, "bfloat16");
  TVM_FFI_CHECK(tensor.ndim() == 2 && tensor.size(1) == kHidden, ValueError)
      << name << " must have shape [rows, " << kHidden << "]";
  TVM_FFI_CHECK(tensor.size(0) >= 1 && tensor.size(0) <= kMaxTableRows, ValueError)
      << name << " must have 1 <= rows <= " << kMaxTableRows;
  TVM_FFI_CHECK(tensor.stride(1) == 1, ValueError) << name << " must have unit column stride";
  TVM_FFI_CHECK(tensor.stride(0) % kTableRowAlign == 0, ValueError)
      << name << " row stride must be a multiple of " << kTableRowAlign << " elements (16-byte rows)";
  // The FC2 epilogue forms the gate table offset as a 32x32->64-bit multiply of the row index and the
  // stride (the tables share one stride): rows must not overlap and the stride must fit 32 bits.
  TVM_FFI_CHECK(tensor.stride(0) >= kHidden && tensor.stride(0) < (int64_t(1) << 32), ValueError)
      << name << " row stride must satisfy " << kHidden << " <= stride(0) < 2^32 elements, got " << tensor.stride(0);
}

ModulationTable CheckOperands(const TensorView& x, const TensorView& x_norm_weight, const TensorView& adaln_scale,
                              const TensorView& adaln_shift, const TensorView& adaln_index, const TensorView& gate,
                              const TensorView& residual, const TensorView& out, int64_t rows, DLDevice device) {
  CheckMatrix(x, "x", rows, kHidden, dl_bfloat16, "bfloat16", device, 16);
  CheckVector(x_norm_weight, "x_norm_weight", kHidden, dl_bfloat16, "bfloat16", device, 16);
  CheckTable(adaln_scale, "adaln_scale", device);
  CheckTable(adaln_shift, "adaln_shift", device);
  CheckTable(gate, "gate", device);
  TVM_FFI_CHECK(adaln_shift.size(0) == adaln_scale.size(0) && adaln_shift.stride(0) == adaln_scale.stride(0), ValueError)
      << "adaln_scale and adaln_shift must have the same rows and the same row stride";
  TVM_FFI_CHECK(gate.size(0) == adaln_scale.size(0) && gate.stride(0) == adaln_scale.stride(0), ValueError)
      << "gate must have the rows and row stride of the AdaLN tables (one index addresses all three)";
  CheckVector(adaln_index, "adaln_index", rows, dl_int64, "int64", device, 8);
  // The FC2 epilogue reads residual and writes out as 256-bit vectors at 32-byte-aligned column offsets;
  // out may alias residual (each thread reads a 16-column chunk before it writes it).
  CheckMatrix(residual, "residual", rows, kHidden, dl_bfloat16, "bfloat16", device, 32);
  CheckMatrix(out, "out", rows, kHidden, dl_bfloat16, "bfloat16", device, 32);
  return ModulationTable{adaln_scale.size(0), adaln_scale.stride(0)};
}

// FC2 tail split-K workspace of this device: float32 [>= slots * 2 * 128 * 256] partial sums and
// int32 [>= slots * 2] counters (the kernel treats them as unsigned; they must be zero before the first
// launch and are re-armed to zero by every launch that uses them).  Shared by the three variants and
// every stream of the device: the caller serialises FC2 launches of one device.
void CheckFc2Workspace(const TensorView& partial, const TensorView& flags, int64_t clusters, DLDevice device) {
  const int64_t slots = clusters / 2 + 1;
  CheckDeviceTensor(partial, "fc2_partial", device, 16);
  CheckDtype(partial, "fc2_partial", dl_float32, "float32");
  TVM_FFI_CHECK(partial.ndim() == 1 && partial.stride(0) == 1, ValueError)
      << "fc2_partial must be a contiguous 1-D tensor";
  TVM_FFI_CHECK(partial.size(0) >= slots * kFc2PartialFloatsPerSlot, ValueError)
      << "fc2_partial must hold at least " << slots * kFc2PartialFloatsPerSlot << " float32 elements for a device with "
      << clusters << " CTA pairs";
  CheckDeviceTensor(flags, "fc2_flags", device, 16);
  CheckDtype(flags, "fc2_flags", dl_int32, "int32");
  TVM_FFI_CHECK(flags.ndim() == 1 && flags.stride(0) == 1, ValueError) << "fc2_flags must be a contiguous 1-D tensor";
  TVM_FFI_CHECK(flags.size(0) >= slots * kFc2FlagsPerSlot, ValueError)
      << "fc2_flags must hold at least " << slots * kFc2FlagsPerSlot << " int32 counters for a device with " << clusters
      << " CTA pairs";
}

// K-major [rows, cols] operand viewed as the rank-3 tensor (box_k, rows, cols / box_k): coordinate
// (0, row, k / box_k) addresses one box_k-wide K group of one row.  128-byte swizzle; rows beyond
// the tensor are zero-filled by TMA (the activation box may extend past M) and never stored.
CUtensorMap EncodeKMajorRows(const void* base, CUtensorMapDataType type, int64_t elem_bytes, int64_t rows,
                             int64_t cols, uint32_t box_k, uint32_t box_rows, uint32_t box_groups,
                             const char* name) {
  TVM_FFI_CHECK(cols % box_k == 0, RuntimeError) << name << ": K=" << cols << " is not a multiple of the TMA box";
  uint64_t global_dim[3] = {box_k, static_cast<uint64_t>(rows), static_cast<uint64_t>(cols / box_k)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(cols * elem_bytes), static_cast<uint64_t>(box_k * elem_bytes)};
  uint32_t box_dim[3] = {box_k, box_rows, box_groups};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, type, 3, const_cast<void*>(base), global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

// Swizzled scale tiles viewed as the rank-3 uint8 tensor (128 bytes, tile_rows, num_tiles) with
// tile_rows = 4 (512-byte activation tile) or 8 (1024-byte combined weight tile): a box
// (128, tile_rows, box_tiles) is box_tiles consecutive whole tiles fetched as 128-byte TMA rows.
// No swizzle; the tiles are consumed verbatim by tcgen05.cp into TMEM.
CUtensorMap EncodeScaleTiles(const void* base, uint32_t tile_rows, int64_t num_tiles, uint32_t box_tiles,
                             const char* name) {
  TVM_FFI_CHECK(num_tiles % box_tiles == 0, RuntimeError) << name << ": tile count is not a multiple of the TMA box";
  uint64_t global_dim[3] = {kSfTileRowBytes, tile_rows, static_cast<uint64_t>(num_tiles)};
  uint64_t global_strides[2] = {kSfTileRowBytes, static_cast<uint64_t>(kSfTileRowBytes) * tile_rows};
  uint32_t box_dim[3] = {kSfTileRowBytes, tile_rows, box_tiles};
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), global_dim, global_strides, box_dim,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

template <typename Kernel>
void OptInSmem(Kernel kernel, int bytes, const char* name) {
  if (bytes <= 0) return;
  const cudaError_t status = cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to " << bytes << " bytes of dynamic shared memory (" << name << "): "
      << cudaGetErrorString(status);
}

// Per-device one-time configuration: capability check and dynamic shared memory opt-in.
void ConfigureKernels() {
  static std::mutex mutex;
  static std::vector<int> configured_devices;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);
  std::lock_guard<std::mutex> lock(mutex);
  for (int configured : configured_devices) {
    if (configured == device) return;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 10 && properties.minor == MINIMAX_H3_SM_MINOR, RuntimeError)
      << "this MiniMax-H3 MLP build targets compute capability 10." << MINIMAX_H3_SM_MINOR
      << " exactly (tcgen05 + TMEM + 2-CTA MMA)";
  TVM_FFI_CHECK(properties.multiProcessorCount >= 2 * kCtaGroup, RuntimeError)
      << "MiniMax-H3 MLP requires at least " << 2 * kCtaGroup << " SMs";
  OptInSmem(kernel_minimax_h3_norm_adaln_bf16, kNormBf16Smem, "norm bf16");
  OptInSmem(kernel_minimax_h3_norm_adaln_mxfp8, kNormMxfp8Smem, "norm mxfp8");
  OptInSmem(kernel_minimax_h3_mlp_norm_adaln_nvfp4, kNormNvfp4Smem, "norm nvfp4");
  OptInSmem(kernel_minimax_h3_mlp_fc1_bf16, kFc1Bf16Smem, "fc1 bf16");
  OptInSmem(kernel_minimax_h3_mlp_fc1_e4m3, kFc1Mxfp8Smem, "fc1 mxfp8");
  OptInSmem(kernel_minimax_h3_mlp_fc1_e4m3_w448, kFc1Mxfp8W448Smem, "fc1 mxfp8 w448");
  OptInSmem(kernel_minimax_h3_mlp_fc1_e2m1, kFc1Nvfp4Smem, "fc1 nvfp4");
  OptInSmem(kernel_minimax_h3_mlp_fc2_bf16, kFc2Bf16Smem, "fc2 bf16");
  OptInSmem(kernel_minimax_h3_mlp_fc2_e4m3, kFc2Mxfp8Smem, "fc2 mxfp8");
  OptInSmem(kernel_minimax_h3_mlp_fc2_e2m1, kFc2Nvfp4Smem, "fc2 nvfp4");
  configured_devices.push_back(device);
}

void CheckLaunch(const char* what) {
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " launch failed: " << cudaGetErrorString(status);
}

// Cluster launch; with programmatic_dependent the programmatic-stream-serialization attribute is
// added so the launch may overlap the tail of the preceding launch on the same stream.
template <typename Kernel, typename... Args>
void LaunchCluster(Kernel kernel, int64_t grid, int threads, int smem_bytes, cudaStream_t stream,
                   bool programmatic_dependent, const char* what, Args... args) {
  cudaLaunchAttribute attrs[2]{};
  int n = 0;
  attrs[n].id = cudaLaunchAttributeClusterDimension;
  attrs[n].val.clusterDim.x = kClusterX;
  attrs[n].val.clusterDim.y = kClusterY;
  attrs[n].val.clusterDim.z = kClusterZ;
  ++n;
  if (programmatic_dependent) {
    attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned int>(grid), 1, 1);
  config.blockDim = dim3(static_cast<unsigned int>(threads), 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(smem_bytes);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = static_cast<unsigned int>(n);
  const cudaError_t status = cudaLaunchKernelEx(&config, kernel, args...);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError) << what << " launch failed: " << cudaGetErrorString(status);
}

}  // namespace

// BF16 operator: three launches.  x: bf16 [M, 5376]; x_norm_weight: bf16 [5376]; adaln_scale /
// adaln_shift / gate: bf16 [rows, 5376] tables with one row stride (see CheckTable); adaln_index: int64 [M]
// (one table row per activation row; a value outside [0, rows) gives a = 0 and out = residual);
// fc1_weight: bf16 [28672, 5376] (gate rows then up rows); fc2_weight: bf16 [5376, 14336]; residual / out:
// bf16 [M, 5376] (out may alias residual); workspace_a: caller-owned bf16 [M, 5376] (the modulated
// activation a); workspace_y: caller-owned bf16 [M, 14336] (the SwiGLU output y); fc2_partial / fc2_flags:
// the device's tail split-K workspace (see CheckFc2Workspace); eps: the RMSNorm epsilon; norm_grid ...
// fc2_clusters: the launch grids (see CheckGrids).
void minimax_h3_mlp(TensorView x, TensorView x_norm_weight, TensorView adaln_scale, TensorView adaln_shift,
                    TensorView adaln_index, TensorView fc1_weight, TensorView fc2_weight, TensorView gate,
                    TensorView residual, TensorView out, TensorView workspace_a, TensorView workspace_y,
                    TensorView fc2_partial, TensorView fc2_flags, double eps, int64_t norm_grid, int64_t fc1_grid,
                    int64_t fc2_grid, int64_t fc2_tail_base, int64_t fc2_split_count, int64_t fc2_clusters) {
  const int64_t rows = CheckRows(x);
  const DLDevice device = x.device();
  const ModulationTable table =
      CheckOperands(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, out, rows, device);
  CheckMatrix(fc1_weight, "fc1_weight", kFc1Rows, kHidden, dl_bfloat16, "bfloat16", device, 16);
  CheckMatrix(fc2_weight, "fc2_weight", kHidden, kFfn, dl_bfloat16, "bfloat16", device, 16);
  CheckMatrix(workspace_a, "workspace_a", rows, kHidden, dl_bfloat16, "bfloat16", device, 16);
  // The FC1 epilogue writes y as 256-bit vectors at 32-byte-aligned column offsets.
  CheckMatrix(workspace_y, "workspace_y", rows, kFfn, dl_bfloat16, "bfloat16", device, 32);
  const int64_t m_tiles = MTiles(rows);
  const LaunchGrids grids = CheckGrids(rows, m_tiles, kFc1Bf16NTiles, norm_grid, fc1_grid, fc2_grid, fc2_tail_base,
                                       fc2_split_count, fc2_clusters);
  CheckFc2Workspace(fc2_partial, fc2_flags, grids.fc2_clusters, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  ConfigureKernels();

  // Resolve every tensor map before the first launch so the three launches are back to back.
  const CUtensorMap a1_map = EncodeKMajorRows(workspace_a.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, rows, kHidden,
                                              kFc1Bf16BoxK, kFc1Bf16BoxRowsA, kFc1Bf16BoxGroups, "workspace_a");
  const CUtensorMap b1_map = EncodeKMajorRows(fc1_weight.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kFc1Rows,
                                              kHidden, kFc1Bf16BoxK, kFc1Bf16BoxRowsB, kFc1Bf16BoxGroups, "fc1_weight");
  const CUtensorMap a2_map = EncodeKMajorRows(workspace_y.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, rows, kFfn,
                                              kFc2Bf16BoxK, kFc2Bf16BoxRowsA, kFc2Bf16BoxGroups, "workspace_y");
  const CUtensorMap b2_map = EncodeKMajorRows(fc2_weight.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kHidden, kFfn,
                                              kFc2Bf16BoxK, kFc2Bf16BoxRowsB, kFc2Bf16BoxGroups, "fc2_weight");

  kernel_minimax_h3_norm_adaln_bf16<<<static_cast<unsigned int>(grids.norm), kNormThreads, kNormBf16Smem, stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<long long*>(adaln_index.data_ptr()), static_cast<__nv_bfloat16*>(workspace_a.data_ptr()),
      static_cast<int>(rows), static_cast<int>(table.rows), static_cast<long long>(table.row_stride),
      static_cast<float>(eps));
  CheckLaunch("MiniMax-H3 MLP norm+AdaLN (bf16)");
  LaunchCluster(kernel_minimax_h3_mlp_fc1_bf16, grids.fc1, kFc1Bf16Threads, kFc1Bf16Smem, stream, false,
                "MiniMax-H3 MLP FC1+SwiGLU (bf16)", a1_map, b1_map, static_cast<__nv_bfloat16*>(workspace_y.data_ptr()),
                static_cast<int>(rows), static_cast<int>(m_tiles));
  LaunchCluster(kernel_minimax_h3_mlp_fc2_bf16, grids.fc2, kFc2Bf16Threads, kFc2Bf16Smem, stream, kFc2Bf16Pdl,
                "MiniMax-H3 MLP FC2+gate+residual (bf16)", a2_map, b2_map, static_cast<__nv_bfloat16*>(gate.data_ptr()),
                static_cast<long long*>(adaln_index.data_ptr()), static_cast<__nv_bfloat16*>(residual.data_ptr()),
                static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(rows), static_cast<int>(m_tiles),
                static_cast<int>(table.rows), static_cast<long long>(table.row_stride),
                static_cast<float*>(fc2_partial.data_ptr()), static_cast<unsigned int*>(fc2_flags.data_ptr()),
                static_cast<int>(grids.fc2_tail_base), static_cast<int>(grids.fc2_split_count), kFc2GroupMBf16);
}

namespace {

// The FC1 GEMM of an MXFP8 operator: the 256x256 pair tile streaming 128-row weight scale tiles, or the
// 256x448 pair tile streaming 112-row weight scale sub-tiles (the FC1 + SwiGLU operator's layout).
struct Mxfp8Fc1 {
  int threads;
  int smem;
  bool pdl;  // launched as a programmatic dependent of the norm
  int64_t n_tiles;
  int64_t weight_sf_tiles;  // combined 256-row weight scale tiles per K set
  int64_t scale_bytes;
  uint32_t box_k;
  uint32_t box_rows_a;
  uint32_t box_rows_b;
  uint32_t box_groups;
  uint32_t sfa_box_tiles;
  uint32_t sfb_box_tiles;
  const char* what;
};

constexpr Mxfp8Fc1 kMxfp8Fc1{kFc1Mxfp8Threads,       kFc1Mxfp8Smem,          false,
                             kFc1Mxfp8NTiles,        kFc1Mxfp8NTiles,        kMxfp8Fc1ScaleBytes,
                             kFc1Mxfp8BoxK,          kFc1Mxfp8BoxRowsA,      kFc1Mxfp8BoxRowsB,
                             kFc1Mxfp8BoxGroups,     kFc1Mxfp8SfaBoxTiles,   kFc1Mxfp8SfbBoxTiles,
                             "MiniMax-H3 MLP FC1+SwiGLU+quantize (mxfp8)"};
constexpr Mxfp8Fc1 kMxfp8Fc1W448{kFc1Mxfp8W448Threads,     kFc1Mxfp8W448Smem,         kFc1Mxfp8W448Pdl,
                                 kFc1Mxfp8W448NTiles,      kFc1Mxfp8W448WeightSfTiles, kMxfp8Fc1W448ScaleBytes,
                                 kFc1Mxfp8W448BoxK,        kFc1Mxfp8W448BoxRowsA,     kFc1Mxfp8W448BoxRowsB,
                                 kFc1Mxfp8W448BoxGroups,   kFc1Mxfp8W448SfaBoxTiles,  kFc1Mxfp8W448SfbBoxTiles,
                                 "MiniMax-H3 MLP FC1+SwiGLU+quantize (mxfp8, 256x448)"};

template <typename Fc1Kernel>
void MlpMxfp8(Fc1Kernel fc1_kernel, const Mxfp8Fc1& fc1, TensorView x, TensorView x_norm_weight,
              TensorView adaln_scale, TensorView adaln_shift, TensorView adaln_index, TensorView fc1_weight_q,
              TensorView fc1_scale_tiles, TensorView fc2_weight_q, TensorView fc2_scale_tiles, TensorView gate,
              TensorView residual, TensorView out, TensorView workspace_a_q, TensorView workspace_a_sf,
              TensorView workspace_y_q, TensorView workspace_y_sf, TensorView fc2_partial, TensorView fc2_flags,
              double eps, int64_t norm_grid, int64_t fc1_grid, int64_t fc2_grid, int64_t fc2_tail_base,
              int64_t fc2_split_count, int64_t fc2_clusters) {
  const int64_t rows = CheckRows(x);
  const DLDevice device = x.device();
  const int64_t m_tiles = MTiles(rows);
  const int64_t a_sf_bytes = m_tiles * kAMxfp8SfKTiles * kSfTileBytes;
  const int64_t y_sf_bytes = m_tiles * kYMxfp8SfKTiles * kSfTileBytes;
  const ModulationTable table =
      CheckOperands(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, out, rows, device);
  CheckMatrix(fc1_weight_q, "fc1_weight_q", kFc1Rows, kHidden, dl_float8_e4m3fn, "float8_e4m3fn", device, 16);
  CheckScaleTiles(fc1_scale_tiles, "fc1_scale_tiles", fc1.scale_bytes, device);
  CheckMatrix(fc2_weight_q, "fc2_weight_q", kHidden, kFfn, dl_float8_e4m3fn, "float8_e4m3fn", device, 16);
  CheckScaleTiles(fc2_scale_tiles, "fc2_scale_tiles", kMxfp8Fc2ScaleBytes, device);
  CheckMatrix(workspace_a_q, "workspace_a_q", rows, kHidden, dl_float8_e4m3fn, "float8_e4m3fn", device, 16);
  CheckByteBuffer(workspace_a_sf, "workspace_a_sf", a_sf_bytes, device);
  // The FC1 epilogue writes y_q as 32-byte vectors at 32-byte-aligned column offsets.
  CheckMatrix(workspace_y_q, "workspace_y_q", rows, kFfn, dl_float8_e4m3fn, "float8_e4m3fn", device, 32);
  CheckByteBuffer(workspace_y_sf, "workspace_y_sf", y_sf_bytes, device);
  const LaunchGrids grids = CheckGrids(rows, m_tiles, fc1.n_tiles, norm_grid, fc1_grid, fc2_grid, fc2_tail_base,
                                       fc2_split_count, fc2_clusters);
  CheckFc2Workspace(fc2_partial, fc2_flags, grids.fc2_clusters, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  ConfigureKernels();

  const CUtensorMap a1_map = EncodeKMajorRows(workspace_a_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows, kHidden,
                                              fc1.box_k, fc1.box_rows_a, fc1.box_groups, "workspace_a_q");
  const CUtensorMap b1_map = EncodeKMajorRows(fc1_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kFc1Rows,
                                              kHidden, fc1.box_k, fc1.box_rows_b, fc1.box_groups, "fc1_weight_q");
  const CUtensorMap sfa1_map = EncodeScaleTiles(workspace_a_sf.data_ptr(), kSfaTileRows, m_tiles * kAMxfp8SfKTiles,
                                                fc1.sfa_box_tiles, "workspace_a_sf");
  const CUtensorMap sfb1_map = EncodeScaleTiles(fc1_scale_tiles.data_ptr(), kSfbTileRows,
                                                fc1.weight_sf_tiles * kAMxfp8SfKTiles, fc1.sfb_box_tiles,
                                                "fc1_scale_tiles");
  const CUtensorMap a2_map = EncodeKMajorRows(workspace_y_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows, kFfn,
                                              kFc2Mxfp8BoxK, kFc2Mxfp8BoxRowsA, kFc2Mxfp8BoxGroups, "workspace_y_q");
  const CUtensorMap b2_map = EncodeKMajorRows(fc2_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHidden, kFfn,
                                              kFc2Mxfp8BoxK, kFc2Mxfp8BoxRowsB, kFc2Mxfp8BoxGroups, "fc2_weight_q");
  const CUtensorMap sfa2_map = EncodeScaleTiles(workspace_y_sf.data_ptr(), kSfaTileRows, m_tiles * kYMxfp8SfKTiles,
                                                kFc2Mxfp8SfaBoxTiles, "workspace_y_sf");
  const CUtensorMap sfb2_map = EncodeScaleTiles(fc2_scale_tiles.data_ptr(), kSfbTileRows, kFc2NTiles * kYMxfp8SfKTiles,
                                                kFc2Mxfp8SfbBoxTiles, "fc2_scale_tiles");

  kernel_minimax_h3_norm_adaln_mxfp8<<<static_cast<unsigned int>(grids.norm), kNormThreads, kNormMxfp8Smem, stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<long long*>(adaln_index.data_ptr()), static_cast<uint8_t*>(workspace_a_q.data_ptr()),
      static_cast<uint8_t*>(workspace_a_sf.data_ptr()), static_cast<int>(rows), static_cast<int>(table.rows),
      static_cast<long long>(table.row_stride), static_cast<float>(eps));
  CheckLaunch("MiniMax-H3 MLP norm+AdaLN+quantize (mxfp8)");
  LaunchCluster(fc1_kernel, grids.fc1, fc1.threads, fc1.smem, stream, fc1.pdl, fc1.what, a1_map, b1_map, sfa1_map,
                sfb1_map, static_cast<uint8_t*>(workspace_y_q.data_ptr()),
                static_cast<uint8_t*>(workspace_y_sf.data_ptr()), static_cast<int>(rows), static_cast<int>(m_tiles));
  LaunchCluster(kernel_minimax_h3_mlp_fc2_e4m3, grids.fc2, kFc2Mxfp8Threads, kFc2Mxfp8Smem, stream, kFc2Mxfp8Pdl,
                "MiniMax-H3 MLP FC2+gate+residual (mxfp8)", a2_map, b2_map, sfa2_map, sfb2_map,
                static_cast<__nv_bfloat16*>(gate.data_ptr()), static_cast<long long*>(adaln_index.data_ptr()),
                static_cast<__nv_bfloat16*>(residual.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
                static_cast<int>(rows), static_cast<int>(m_tiles), static_cast<int>(table.rows),
                static_cast<long long>(table.row_stride), static_cast<float*>(fc2_partial.data_ptr()),
                static_cast<unsigned int*>(fc2_flags.data_ptr()), static_cast<int>(grids.fc2_tail_base),
                static_cast<int>(grids.fc2_split_count), kFc2GroupMMxfp8);
}

}  // namespace

// MXFP8 operator: three launches.  fc1_weight_q: float8_e4m3fn [28672, 5376] (gate rows then up rows);
// fc1_scale_tiles: uint8 [112 * 42 * 1024] combined 256-row FC1 weight scale tiles ([tile][k_set][half][512]);
// fc2_weight_q: float8_e4m3fn [5376, 14336]; fc2_scale_tiles: uint8 [21 * 112 * 1024] combined 256-row FC2 weight
// scale tiles ([n_tile][k_set][half][512]); workspace_a_q: caller-owned float8_e4m3fn [M, 5376] and
// workspace_a_sf: caller-owned uint8 of at least m_tiles(M) * 42 * 512 bytes (the FlashInfer-exact MXFP8
// quantization of a, swizzled 128x4 scales); workspace_y_q: caller-owned float8_e4m3fn [M, 14336] and
// workspace_y_sf: caller-owned uint8 of at least m_tiles(M) * 112 * 512 bytes (the MXFP8 quantization of y
// written by the FC1 epilogue); norm_grid ... fc2_clusters: the launch grids (see CheckGrids).  The remaining
// operands are as in minimax_h3_mlp.
void minimax_h3_mlp_mxfp8(TensorView x, TensorView x_norm_weight, TensorView adaln_scale, TensorView adaln_shift,
                          TensorView adaln_index, TensorView fc1_weight_q, TensorView fc1_scale_tiles,
                          TensorView fc2_weight_q, TensorView fc2_scale_tiles, TensorView gate, TensorView residual,
                          TensorView out, TensorView workspace_a_q, TensorView workspace_a_sf, TensorView workspace_y_q,
                          TensorView workspace_y_sf, TensorView fc2_partial, TensorView fc2_flags, double eps,
                          int64_t norm_grid, int64_t fc1_grid, int64_t fc2_grid, int64_t fc2_tail_base,
                          int64_t fc2_split_count, int64_t fc2_clusters) {
  MlpMxfp8(kernel_minimax_h3_mlp_fc1_e4m3, kMxfp8Fc1, x, x_norm_weight, adaln_scale, adaln_shift, adaln_index,
           fc1_weight_q, fc1_scale_tiles, fc2_weight_q, fc2_scale_tiles, gate, residual, out, workspace_a_q,
           workspace_a_sf, workspace_y_q, workspace_y_sf, fc2_partial, fc2_flags, eps, norm_grid, fc1_grid, fc2_grid,
           fc2_tail_base, fc2_split_count, fc2_clusters);
}

// MXFP8 operator on the 256x448 FC1 pair tile: as minimax_h3_mlp_mxfp8 with fc1_scale_tiles: uint8
// [128 * 42 * 1024] combined 224-row (112 gate + 112 up, padded to 256) FC1 weight scale sub-tiles
// ([sub_tile][k_set][half][512]; output tile t reads sub-tiles 2t and 2t + 1).
void minimax_h3_mlp_mxfp8_w448(TensorView x, TensorView x_norm_weight, TensorView adaln_scale,
                               TensorView adaln_shift, TensorView adaln_index, TensorView fc1_weight_q,
                               TensorView fc1_scale_tiles, TensorView fc2_weight_q, TensorView fc2_scale_tiles,
                               TensorView gate, TensorView residual, TensorView out, TensorView workspace_a_q,
                               TensorView workspace_a_sf, TensorView workspace_y_q, TensorView workspace_y_sf,
                               TensorView fc2_partial, TensorView fc2_flags, double eps, int64_t norm_grid,
                               int64_t fc1_grid, int64_t fc2_grid, int64_t fc2_tail_base, int64_t fc2_split_count,
                               int64_t fc2_clusters) {
  MlpMxfp8(kernel_minimax_h3_mlp_fc1_e4m3_w448, kMxfp8Fc1W448, x, x_norm_weight, adaln_scale, adaln_shift,
           adaln_index, fc1_weight_q, fc1_scale_tiles, fc2_weight_q, fc2_scale_tiles, gate, residual, out,
           workspace_a_q, workspace_a_sf, workspace_y_q, workspace_y_sf, fc2_partial, fc2_flags, eps, norm_grid,
           fc1_grid, fc2_grid, fc2_tail_base, fc2_split_count, fc2_clusters);
}

// NVFP4 operator: three launches.  a_global_scale / y_global_scale: float32 [1] static activation global scales
// (FlashInfer nvfp4_quantize convention, 448 * 6 / absmax); fc1_weight_q: uint8 [28672, 2688] packed E2M1 (gate rows
// then up rows); fc1_scale_tiles: uint8 [128 * 84 * 1024] combined 224-row (112 gate + 112 up, padded to 256) FC1
// weight scale sub-tiles ([sub_tile][k_set][half][512]; output tile t reads sub-tiles 2t and 2t + 1); alpha1:
// float32 [1] = 1 / (a_global_scale * w1_global_scale); fc2_weight_q: uint8 [5376, 7168] packed E2M1;
// fc2_scale_tiles: uint8 [21 * 224 * 1024] combined 256-row FC2 weight scale tiles; alpha2: float32 [1] =
// 1 / (y_global_scale * w2_global_scale); workspace_a_q: caller-owned uint8 [M, 2688] and workspace_a_sf: at least
// m_tiles(M) * 84 * 512 bytes; workspace_y_q: caller-owned uint8 [M, 7168] and workspace_y_sf: at least
// m_tiles(M) * 224 * 512 bytes; norm_grid ... fc2_clusters: the launch grids (see CheckGrids).  The remaining
// operands are as in minimax_h3_mlp.
void minimax_h3_mlp_nvfp4(TensorView x, TensorView x_norm_weight, TensorView adaln_scale, TensorView adaln_shift,
                          TensorView adaln_index, TensorView a_global_scale, TensorView fc1_weight_q,
                          TensorView fc1_scale_tiles, TensorView alpha1, TensorView y_global_scale,
                          TensorView fc2_weight_q, TensorView fc2_scale_tiles, TensorView alpha2, TensorView gate,
                          TensorView residual, TensorView out, TensorView workspace_a_q, TensorView workspace_a_sf,
                          TensorView workspace_y_q, TensorView workspace_y_sf, TensorView fc2_partial,
                          TensorView fc2_flags, double eps, int64_t norm_grid, int64_t fc1_grid, int64_t fc2_grid,
                          int64_t fc2_tail_base, int64_t fc2_split_count, int64_t fc2_clusters) {
  const int64_t rows = CheckRows(x);
  const DLDevice device = x.device();
  const int64_t m_tiles = MTiles(rows);
  const int64_t a_sf_bytes = m_tiles * kANvfp4SfKTiles * kSfTileBytes;
  const int64_t y_sf_bytes = m_tiles * kYNvfp4SfKTiles * kSfTileBytes;
  const ModulationTable table =
      CheckOperands(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, gate, residual, out, rows, device);
  CheckScalar(a_global_scale, "a_global_scale", device);
  CheckScalar(alpha1, "alpha1", device);
  CheckScalar(y_global_scale, "y_global_scale", device);
  CheckScalar(alpha2, "alpha2", device);
  CheckMatrix(fc1_weight_q, "fc1_weight_q", kFc1Rows, kNvfp4APackedCols, dl_uint8, "uint8", device, 16);
  CheckScaleTiles(fc1_scale_tiles, "fc1_scale_tiles", kNvfp4Fc1ScaleBytes, device);
  CheckMatrix(fc2_weight_q, "fc2_weight_q", kHidden, kNvfp4YPackedCols, dl_uint8, "uint8", device, 16);
  CheckScaleTiles(fc2_scale_tiles, "fc2_scale_tiles", kNvfp4Fc2ScaleBytes, device);
  CheckMatrix(workspace_a_q, "workspace_a_q", rows, kNvfp4APackedCols, dl_uint8, "uint8", device, 16);
  CheckByteBuffer(workspace_a_sf, "workspace_a_sf", a_sf_bytes, device);
  CheckMatrix(workspace_y_q, "workspace_y_q", rows, kNvfp4YPackedCols, dl_uint8, "uint8", device, 16);
  CheckByteBuffer(workspace_y_sf, "workspace_y_sf", y_sf_bytes, device);
  const LaunchGrids grids = CheckGrids(rows, m_tiles, kFc1Nvfp4NTiles, norm_grid, fc1_grid, fc2_grid, fc2_tail_base,
                                       fc2_split_count, fc2_clusters, kNormNvfp4RowGroups);
  CheckFc2Workspace(fc2_partial, fc2_flags, grids.fc2_clusters, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  ConfigureKernels();

  const CUtensorMap a1_map = EncodeKMajorRows(workspace_a_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows,
                                              kNvfp4APackedCols, kFc1Nvfp4BoxK, kFc1Nvfp4BoxRowsA, kFc1Nvfp4BoxGroups,
                                              "workspace_a_q");
  const CUtensorMap b1_map = EncodeKMajorRows(fc1_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kFc1Rows,
                                              kNvfp4APackedCols, kFc1Nvfp4BoxK, kFc1Nvfp4BoxRowsB, kFc1Nvfp4BoxGroups,
                                              "fc1_weight_q");
  const CUtensorMap sfa1_map = EncodeScaleTiles(workspace_a_sf.data_ptr(), kSfaTileRows, m_tiles * kANvfp4SfKTiles,
                                                kFc1Nvfp4SfaBoxTiles, "workspace_a_sf");
  const CUtensorMap sfb1_map = EncodeScaleTiles(fc1_scale_tiles.data_ptr(), kSfbTileRows,
                                                kFc1Nvfp4WeightSfTiles * kANvfp4SfKTiles, kFc1Nvfp4SfbBoxTiles,
                                                "fc1_scale_tiles");
  const CUtensorMap a2_map = EncodeKMajorRows(workspace_y_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows,
                                              kNvfp4YPackedCols, kFc2Nvfp4BoxK, kFc2Nvfp4BoxRowsA, kFc2Nvfp4BoxGroups,
                                              "workspace_y_q");
  const CUtensorMap b2_map = EncodeKMajorRows(fc2_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHidden,
                                              kNvfp4YPackedCols, kFc2Nvfp4BoxK, kFc2Nvfp4BoxRowsB, kFc2Nvfp4BoxGroups,
                                              "fc2_weight_q");
  const CUtensorMap sfa2_map = EncodeScaleTiles(workspace_y_sf.data_ptr(), kSfaTileRows, m_tiles * kYNvfp4SfKTiles,
                                                kFc2Nvfp4SfaBoxTiles, "workspace_y_sf");
  const CUtensorMap sfb2_map = EncodeScaleTiles(fc2_scale_tiles.data_ptr(), kSfbTileRows, kFc2NTiles * kYNvfp4SfKTiles,
                                                kFc2Nvfp4SfbBoxTiles, "fc2_scale_tiles");

  kernel_minimax_h3_mlp_norm_adaln_nvfp4<<<static_cast<unsigned int>(grids.norm), kNormNvfp4Threads, kNormNvfp4Smem,
                                           stream>>>(
      static_cast<__nv_bfloat16*>(x.data_ptr()), static_cast<__nv_bfloat16*>(x_norm_weight.data_ptr()),
      static_cast<__nv_bfloat16*>(adaln_scale.data_ptr()), static_cast<__nv_bfloat16*>(adaln_shift.data_ptr()),
      static_cast<long long*>(adaln_index.data_ptr()), static_cast<float*>(a_global_scale.data_ptr()),
      static_cast<unsigned int*>(workspace_a_q.data_ptr()), static_cast<uint8_t*>(workspace_a_sf.data_ptr()),
      static_cast<int>(rows), static_cast<int>(table.rows), static_cast<long long>(table.row_stride),
      static_cast<float>(eps));
  CheckLaunch("MiniMax-H3 MLP norm+AdaLN+quantize (nvfp4)");
  LaunchCluster(kernel_minimax_h3_mlp_fc1_e2m1, grids.fc1, kFc1Nvfp4Threads, kFc1Nvfp4Smem, stream, kFc1Nvfp4Pdl,
                "MiniMax-H3 MLP FC1+SwiGLU+quantize (nvfp4)", a1_map, b1_map, sfa1_map, sfb1_map,
                static_cast<float*>(alpha1.data_ptr()), static_cast<float*>(y_global_scale.data_ptr()),
                static_cast<unsigned int*>(workspace_y_q.data_ptr()), static_cast<uint8_t*>(workspace_y_sf.data_ptr()),
                static_cast<int>(rows), static_cast<int>(m_tiles));
  LaunchCluster(kernel_minimax_h3_mlp_fc2_e2m1, grids.fc2, kFc2Nvfp4Threads, kFc2Nvfp4Smem, stream, kFc2Nvfp4Pdl,
                "MiniMax-H3 MLP FC2+gate+residual (nvfp4)", a2_map, b2_map, sfa2_map, sfb2_map,
                static_cast<float*>(alpha2.data_ptr()), static_cast<__nv_bfloat16*>(gate.data_ptr()),
                static_cast<long long*>(adaln_index.data_ptr()), static_cast<__nv_bfloat16*>(residual.data_ptr()),
                static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(rows), static_cast<int>(m_tiles),
                static_cast<int>(table.rows), static_cast<long long>(table.row_stride),
                static_cast<float*>(fc2_partial.data_ptr()), static_cast<unsigned int*>(fc2_flags.data_ptr()),
                static_cast<int>(grids.fc2_tail_base), static_cast<int>(grids.fc2_split_count), kFc2GroupMNvfp4);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_mlp, minimax_h3_mlp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_mlp_mxfp8, minimax_h3_mlp_mxfp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_mlp_mxfp8_w448, minimax_h3_mlp_mxfp8_w448);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_mlp_nvfp4, minimax_h3_mlp_nvfp4);
// clang-format on
