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
// MiniMax-H3 attention output projection + indexed gate + residual for sm_100a and sm_103a
// (Blackwell, compute capability 10.0 and 10.3; one translation unit compiled once per target,
// MINIMAX_H3_SM_MINOR naming the minor).  Generated device code; three operator variants share
// one translation unit.  The activation arrives in the Ulysses (sequence-parallel) receive layout
// attn_out[P, M, 56 / P, 128] (block p holds heads [p * 56 / P, (p + 1) * 56 / P)) and is consumed
// in place: the logical activation is A[m, h * 128 + d] = attn_out[h / (56 / P), m, h % (56 / P), d].
//
//   BF16 : o = BF16(A @ o_weight^T)                       FP32 accumulation
//   MXFP8: a_q, a_sf = mxfp8_quantize(A)                  (E4M3 + UE8M0 per 32, FlashInfer recipe)
//          o = BF16(dequant(a_q, a_sf) @ dequant(w_q, w_sf)^T)
//   NVFP4: a_q, a_sf = nvfp4_quantize(A, a_global_scale)  (E2M1 + UE4M3 per 16, FlashInfer recipe)
//          o = BF16(alpha * ((a_q * a_sf) @ (w_q * w_sf)^T)),
//          alpha = 1 / (a_global_scale * w_global_scale)
//   p   = BF16(gate[gate_index[m]] * o)                    gate: bf16 [rows, 5376] table read as
//                                                          base + idx * gate_row_stride (contiguous or a
//                                                          column chunk of a wider projection); gate = 0
//                                                          for an int64 index outside [0, gate_rows)
//   out = BF16(residual + p)
//
// The BF16 variant is a single persistent 2-CTA (cta_group::2) tcgen05 GEMM whose activation
// tensor map spans the [P * M, 7168 / P] view of attn_out (K block kb of row m is descriptor row
// (kb / (112 / P)) * M + m, column block kb % (112 / P)).  The quantized variants first run a plain
// 128-thread launch (one warp per row) that reads the same layout and writes the quantized dense
// [M, 7168] activation plus its 128x4 swizzled block scales, then a persistent 2-CTA block-scaled
// GEMM.  The quantization pass executes griddepcontrol.launch_dependents on entry and the
// block-scaled GEMM is launched with programmatic stream serialization: its prologue (barrier /
// TMEM setup, weight prefetch) overlaps the quantization pass and its load warp executes
// griddepcontrol.wait before the first activation / scale fetch.  Every GEMM requires a cluster
// launch of (2, 1, 1) and does not compile for SM90 or SM120.
// On sm_103a the NVFP4 GEMM issues the K = 96 block16 MMA: three K256 panels form a K768 step of
// eight K = 96 windows (two of them span a panel boundary) and the last K256 panel keeps four
// K = 64 MMAs; the output is bitwise equal to the K = 64 program.
// Each GEMM has a second program, kernel_minimax_h3_out_proj_{bf16,e4m3,e2m1}_ts (N-split tail): when
// the tile pairs leave a partial last wave of R tiles on the co-resident CTA pairs and 2 R fit in one
// wave, those tiles run as 2 R half units of 256 x 128 output columns after the whole tiles, so the
// partial wave costs about half a tile.  A half unit runs the whole tile's K loop with N = 128 and
// the same per-element epilogue, so the output is bitwise equal to the whole-tile program; the host
// picks the program per variant and shape (TailSplitSchedule).
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

__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

#if __CUDA_ARCH__ == 1030
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
}

#endif
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

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
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
#define CTA_GROUP 2
#define NUM_STAGES 7
#define GROUP_M 16
#define WORK_STAGES 4
#define NUM_K_ITERS 112
#define N_TILES 21
#define HIDDEN 5376
#define num_cluster_tiles ((m_tiles / CTA_GROUP) * N_TILES)
#define tiles_per_group (GROUP_M * N_TILES)


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

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_bf16(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int k_blocks_per_seg, int gate_rows, long long gate_row_stride)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_response_addr = smem + 230400;

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
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 128, 16);
            mbarrier_init(smem + 136, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 176, 546);
            mbarrier_init(smem + 184, 546);
            mbarrier_init(smem + 192, 546);
            mbarrier_init(smem + 200, 546);
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
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int seg = iter_k / k_blocks_per_seg;
                        int kb = iter_k - seg * k_blocks_per_seg;
                        int a_row = seg * M + off_m;
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, a_row, kb, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
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
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_half = (warp - 2) / 4;
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
                int off_n_1 = bid_n_1 * BLOCK_N;
                int global_row = off_m_1 + local_row;
                int col0 = col_half * B_HALF_N;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N + (unsigned int)col0;
                int _min_0 = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                int load_row = _min_0;
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_1 + (unsigned long long)col0;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_1 + (unsigned long long)col0;
                #pragma unroll 1
                for (int n_chunk = 0; n_chunk < B_HALF_N / 16; n_chunk++) {
                    int col = n_chunk * 16;
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], lane_addr + col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (global_row < M) {
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
                            uint4 _vld_1[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                        float _vec_load_2[8];
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
                                }
                            }
                        }
                        float _vec_load_3[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_3[_pair]));
                                }
                            }
                        }
                        float out_vals[16];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[j]);
                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                            float o_lo = _cvt_f32_0;
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_vec_load_0[j] * gate_mul * o_lo);
                            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                            float p_lo = _cvt_f32_1;
                            out_vals[j] = _vec_load_2[j] + p_lo;
                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_tmem_load_0[j + 8]);
                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                            float o_hi = _cvt_f32_2;
                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_vec_load_1[j] * gate_mul * o_hi);
                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                            float p_hi = _cvt_f32_3;
                            out_vals[j + 8] = _vec_load_3[j] + p_hi;
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
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
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
#define GROUP_M 16
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 112
#define N_TILES 21
#define HIDDEN 5376
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_bf16_ts(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int k_blocks_per_seg, int gate_rows, long long gate_row_stride, const __grid_constant__ CUtensorMap B_half, int full_units)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_response_addr = smem + 230400;

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
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 128, 16);
            mbarrier_init(smem + 136, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 176, 546);
            mbarrier_init(smem + 184, 546);
            mbarrier_init(smem + 192, 546);
            mbarrier_init(smem + 200, 546);
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
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter++) {
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
                    int unit = (int)(this_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid = this_bid;
                    int is_half = 0;
                    int half_h = 0;
                    if (unit >= full_units) {
                        int half_q = unit - full_units;
                        tile_bid = (unsigned int)((full_units + half_q / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h = half_q % 2;
                        is_half = 1;
                    }
                    int group = tile_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = tile_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    if (is_half == 0) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int seg = iter_k / k_blocks_per_seg;
                            int kb = iter_k - seg * k_blocks_per_seg;
                            int a_row = seg * M + off_m;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, a_row, kb, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), 0, weight_row, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    } else {
                        int half_row = off_n + half_h * 128 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int seg_1 = iter_k_1 / k_blocks_per_seg;
                            int kb_1 = iter_k_1 - seg_1 * k_blocks_per_seg;
                            int a_row_1 = seg_1 * M + off_m;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, a_row_1, kb_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B_half), 0, half_row, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(24576)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
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
                unsigned int mma_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    int unit_1 = (int)(mma_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_1 = mma_bid;
                    int is_half_1 = 0;
                    int half_h_1 = 0;
                    if (unit_1 >= full_units) {
                        int half_q_1 = unit_1 - full_units;
                        tile_bid_1 = (unsigned int)((full_units + half_q_1 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_1 = half_q_1 % 2;
                        is_half_1 = 1;
                    }
                    int unit_n = 256 - is_half_1 * 128;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < NUM_K_ITERS; iter_k_2++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_2 == 0) ? 1 : 0);
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
                    "mov.b32 id, %4;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)), "r"(((0x10400490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17))) : "memory");
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
                    mma_bid = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int unit_2 = (int)(this_bid_1 / (unsigned int)CTA_GROUP);
                unsigned int tile_bid_2 = this_bid_1;
                int is_half_2 = 0;
                int half_h_2 = 0;
                if (unit_2 >= full_units) {
                    int half_q_2 = unit_2 - full_units;
                    tile_bid_2 = (unsigned int)((full_units + half_q_2 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h_2 = half_q_2 % 2;
                    is_half_2 = 1;
                }
                int group_1 = tile_bid_2 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = tile_bid_2 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                int unit_cols = B_HALF_N - is_half_2 * 64;
                int unit_off_n = off_n_1 + half_h_2 * 128;
                int global_row = off_m_1 + local_row;
                int col0 = col_half * unit_cols;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N + (unsigned int)col0;
                int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)unit_off_n + (unsigned long long)col0;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)unit_off_n + (unsigned long long)col0;
                #pragma unroll 1
                for (int n_chunk = 0; n_chunk < unit_cols / 16; n_chunk++) {
                    int col = n_chunk * 16;
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], lane_addr + col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (global_row < M) {
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
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
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
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
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
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
                        float out_vals[16];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            float o_lo = (float)(__nv_bfloat16)_tmem_load_0[j];
                            float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
                            out_vals[j] = _vec_load_2[j] + p_lo;
                            float o_hi = (float)(__nv_bfloat16)_tmem_load_0[j + 8];
                            float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
                            out_vals[j + 8] = _vec_load_3[j] + p_hi;
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
                            const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
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

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#undef tiles_per_group

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
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
#define GROUP_M 16
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 112
#define N_TILES 21
#define HIDDEN 5376
#define tiles_per_group (GROUP_M * N_TILES)

#if MINIMAX_H3_SM_MINOR == 3  // sm_103a only
extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_bf16_ts_wr(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int k_blocks_per_seg, int gate_rows, long long gate_row_stride, const __grid_constant__ CUtensorMap B_half, int full_units)
{
#if __CUDA_ARCH__ == 1030
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_response_addr = smem + 230400;

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
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 128, 16);
            mbarrier_init(smem + 136, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 176, 546);
            mbarrier_init(smem + 184, 546);
            mbarrier_init(smem + 192, 546);
            mbarrier_init(smem + 200, 546);
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
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter++) {
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
                    int unit = (int)(this_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid = this_bid;
                    int is_half = 0;
                    int half_h = 0;
                    if (unit >= full_units) {
                        int half_q = unit - full_units;
                        tile_bid = (unsigned int)((full_units + half_q / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h = half_q % 2;
                        is_half = 1;
                    }
                    int group = tile_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = tile_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    if (is_half == 0) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int seg = iter_k / k_blocks_per_seg;
                            int kb = iter_k - seg * k_blocks_per_seg;
                            int a_row = seg * M + off_m;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, a_row, kb, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), 0, weight_row, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    } else {
                        int half_row = off_n + half_h * 128 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int seg_1 = iter_k_1 / k_blocks_per_seg;
                            int kb_1 = iter_k_1 - seg_1 * k_blocks_per_seg;
                            int a_row_1 = seg_1 * M + off_m;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, a_row_1, kb_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B_half), 0, half_row, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(24576)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
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
                unsigned int mma_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    int unit_1 = (int)(mma_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_1 = mma_bid;
                    int is_half_1 = 0;
                    int half_h_1 = 0;
                    if (unit_1 >= full_units) {
                        int half_q_1 = unit_1 - full_units;
                        tile_bid_1 = (unsigned int)((full_units + half_q_1 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_1 = half_q_1 % 2;
                        is_half_1 = 1;
                    }
                    int unit_n = 256 - is_half_1 * 128;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < NUM_K_ITERS; iter_k_2++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_2 == 0) ? 1 : 0);
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
                    "mov.b32 id, %4;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)), "r"(((0x10400490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17))) : "memory");
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
                    mma_bid = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int unit_2 = (int)(this_bid_1 / (unsigned int)CTA_GROUP);
                unsigned int tile_bid_2 = this_bid_1;
                int is_half_2 = 0;
                int half_h_2 = 0;
                if (unit_2 >= full_units) {
                    int half_q_2 = unit_2 - full_units;
                    tile_bid_2 = (unsigned int)((full_units + half_q_2 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h_2 = half_q_2 % 2;
                    is_half_2 = 1;
                }
                int group_1 = tile_bid_2 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = tile_bid_2 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                int unit_cols = B_HALF_N - is_half_2 * 64;
                int unit_off_n = off_n_1 + half_h_2 * 128;
                int global_row = off_m_1 + local_row;
                int col0 = col_half * unit_cols;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N + (unsigned int)col0;
                int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)unit_off_n + (unsigned long long)col0;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)unit_off_n + (unsigned long long)col0;
                #pragma unroll 1
                for (int n_chunk = 0; n_chunk < unit_cols / 16; n_chunk++) {
                    int col = n_chunk * 16;
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], lane_addr + col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (global_row < M) {
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
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
                        float _vec_load_2[16];
                        {
                            const void* _v8p_2 = (const void*)(residual + (row_base + (unsigned long long)col) + (0));
                            uint32_t _v8_2_0[8];
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
                            (&_vec_load_2[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) << 16);
                            (&_vec_load_2[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) & 0xffff0000u);
                            (&_vec_load_2[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) << 16);
                            (&_vec_load_2[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) & 0xffff0000u);
                            (&_vec_load_2[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) << 16);
                            (&_vec_load_2[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) & 0xffff0000u);
                            (&_vec_load_2[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) << 16);
                            (&_vec_load_2[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) & 0xffff0000u);
                            (&_vec_load_2[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) << 16);
                            (&_vec_load_2[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) & 0xffff0000u);
                            (&_vec_load_2[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) << 16);
                            (&_vec_load_2[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) & 0xffff0000u);
                            (&_vec_load_2[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) << 16);
                            (&_vec_load_2[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) & 0xffff0000u);
                            (&_vec_load_2[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) << 16);
                            (&_vec_load_2[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) & 0xffff0000u);
                        }
                        float out_vals[16];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            float o_lo = (float)(__nv_bfloat16)_tmem_load_0[j];
                            float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
                            out_vals[j] = _vec_load_2[j] + p_lo;
                            float o_hi = (float)(__nv_bfloat16)_tmem_load_0[j + 8];
                            float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
                            out_vals[j + 8] = _vec_load_2[j + 8] + p_hi;
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
                            const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
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
#endif
}

} // extern "C"
#endif  // MINIMAX_H3_SM_MINOR == 3

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#undef tiles_per_group

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
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
#define GROUP_M 16
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 112
#define N_TILES 21
#define HIDDEN 5376
#define tiles_per_group (GROUP_M * N_TILES)

#if MINIMAX_H3_SM_MINOR == 0  // sm_100a only
extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_bf16_ts_ef(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int k_blocks_per_seg, int gate_rows, long long gate_row_stride, const __grid_constant__ CUtensorMap B_half, int full_units)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_response_addr = smem + 230400;

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
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 128, 16);
            mbarrier_init(smem + 136, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 176, 546);
            mbarrier_init(smem + 184, 546);
            mbarrier_init(smem + 192, 546);
            mbarrier_init(smem + 200, 546);
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
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter++) {
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
                    int unit = (int)(this_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid = this_bid;
                    int is_half = 0;
                    int half_h = 0;
                    if (unit >= full_units) {
                        int half_q = unit - full_units;
                        tile_bid = (unsigned int)((full_units + half_q / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h = half_q % 2;
                        is_half = 1;
                    }
                    int group = tile_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = tile_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    if (is_half == 0) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int seg = iter_k / k_blocks_per_seg;
                            int kb = iter_k - seg * k_blocks_per_seg;
                            int a_row = seg * M + off_m;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, a_row, kb, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), 0, weight_row, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    } else {
                        int half_row = off_n + half_h * 128 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int seg_1 = iter_k_1 / k_blocks_per_seg;
                            int kb_1 = iter_k_1 - seg_1 * k_blocks_per_seg;
                            int a_row_1 = seg_1 * M + off_m;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, a_row_1, kb_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B_half), 0, half_row, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(24576)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
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
                unsigned int mma_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    int unit_1 = (int)(mma_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_1 = mma_bid;
                    int is_half_1 = 0;
                    int half_h_1 = 0;
                    if (unit_1 >= full_units) {
                        int half_q_1 = unit_1 - full_units;
                        tile_bid_1 = (unsigned int)((full_units + half_q_1 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_1 = half_q_1 % 2;
                        is_half_1 = 1;
                    }
                    int unit_n = 256 - is_half_1 * 128;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < NUM_K_ITERS; iter_k_2++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_2 == 0) ? 1 : 0);
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
                    "mov.b32 id, %4;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)), "r"(((0x10400490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17))) : "memory");
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
                    mma_bid = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int unit_2 = (int)(this_bid_1 / (unsigned int)CTA_GROUP);
                unsigned int tile_bid_2 = this_bid_1;
                int is_half_2 = 0;
                int half_h_2 = 0;
                if (unit_2 >= full_units) {
                    int half_q_2 = unit_2 - full_units;
                    tile_bid_2 = (unsigned int)((full_units + half_q_2 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h_2 = half_q_2 % 2;
                    is_half_2 = 1;
                }
                int group_1 = tile_bid_2 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = tile_bid_2 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                if (is_half_2 == 0) {
                    int global_row = off_m_1 + local_row;
                    int col0 = col_half * B_HALF_N;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N + (unsigned int)col0;
                    int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                    long long table_row = gate_index[load_row];
                    int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul = (float)in_range;
                    unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                    unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_1 + (unsigned long long)col0;
                    unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_1 + (unsigned long long)col0;
                    #pragma unroll 1
                    for (int n_chunk = 0; n_chunk < B_HALF_N / 16; n_chunk++) {
                        int col = n_chunk * 16;
                        float _tmem_load_0[16];
                        tmem_ld_x16(&_tmem_load_0[0], lane_addr + col);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (global_row < M) {
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                                uint4 _vld_0[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_0[_blk] = _vptr_0[_blk];
                                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
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
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
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
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
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
                            float out_vals[16];
                            #pragma unroll
                            for (int j = 0; j < 8; j++) {
                                float o_lo = (float)(__nv_bfloat16)_tmem_load_0[j];
                                float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
                                out_vals[j] = _vec_load_2[j] + p_lo;
                                float o_hi = (float)(__nv_bfloat16)_tmem_load_0[j + 8];
                                float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
                                out_vals[j + 8] = _vec_load_3[j] + p_hi;
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
                                const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                            }
                        }
                    }
                } else {
                    int half_off_n = off_n_1 + half_h_2 * 128;
                    int global_row_1 = off_m_1 + local_row;
                    int col0_1 = col_half * 64;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N + (unsigned int)col0_1;
                    int load_row_1 = ((global_row_1) < (M - 1) ? (global_row_1) : (M - 1));
                    long long table_row_1 = gate_index[load_row_1];
                    int in_range_1 = (((unsigned long long)table_row_1 < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul_1 = (float)in_range_1;
                    unsigned int safe_row_1 = (unsigned int)table_row_1 * (unsigned int)in_range_1;
                    unsigned long long gate_base_1 = (unsigned long long)safe_row_1 * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)half_off_n + (unsigned long long)col0_1;
                    unsigned long long row_base_1 = (unsigned long long)global_row_1 * (unsigned long long)HIDDEN + (unsigned long long)half_off_n + (unsigned long long)col0_1;
                    #pragma unroll 1
                    for (int n_chunk_1 = 0; n_chunk_1 < 64 / 16; n_chunk_1++) {
                        int col_1 = n_chunk_1 * 16;
                        float _tmem_load_0_1[16];
                        tmem_ld_x16(&_tmem_load_0_1[0], lane_addr_1 + col_1);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (global_row_1 < M) {
                            float _vec_load_0_1[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&_vec_load_0_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1_1[8];
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&_vec_load_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_2_1[8];
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(residual + (row_base_1 + (unsigned long long)col_1) + 0);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_2_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&_vec_load_2_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_3_1[8];
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(residual + (row_base_1 + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&_vec_load_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float out_vals_1[16];
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 8; j_1++) {
                                float o_lo_1 = (float)(__nv_bfloat16)_tmem_load_0_1[j_1];
                                float p_lo_1 = (float)(__nv_bfloat16)(_vec_load_0_1[j_1] * gate_mul_1 * o_lo_1);
                                out_vals_1[j_1] = _vec_load_2_1[j_1] + p_lo_1;
                                float o_hi_1 = (float)(__nv_bfloat16)_tmem_load_0_1[j_1 + 8];
                                float p_hi_1 = (float)(__nv_bfloat16)(_vec_load_1_1[j_1] * gate_mul_1 * o_hi_1);
                                out_vals_1[j_1 + 8] = _vec_load_3_1[j_1] + p_hi_1;
                            }
                            unsigned int _packed_0_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 0], out_vals_1[_lp*2+1 + 0]));
                                _packed_0_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 2], out_vals_1[_lp*2+1 + 2]));
                                _packed_0_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 4], out_vals_1[_lp*2+1 + 4]));
                                _packed_0_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 6], out_vals_1[_lp*2+1 + 6]));
                                _packed_0_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 8], out_vals_1[_lp*2+1 + 8]));
                                _packed_0_1[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 10], out_vals_1[_lp*2+1 + 10]));
                                _packed_0_1[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 12], out_vals_1[_lp*2+1 + 12]));
                                _packed_0_1[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 14], out_vals_1[_lp*2+1 + 14]));
                                _packed_0_1[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(_packed_0_1);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_1))), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
                            }
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
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
#endif  // MINIMAX_H3_SM_MINOR == 0

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#undef tiles_per_group

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define ATTN_DIM 7168
#define ROWS_PER_CTA 4
#define VECS_PER_LANE 28
#define SF_K_TILES 56
#define SF_TILE_BYTES 512
#define LOAD_BATCH 7
#define ROW_SPLIT 1

extern "C" {

__global__ __launch_bounds__(128) void
kernel_minimax_h3_quant_mxfp8(__nv_bfloat16* __restrict__ attn_out, uint8_t* __restrict__ a_q, uint8_t* __restrict__ a_sf, int M, int seg_cols)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int slot = bid * ROWS_PER_CTA + warp;
    int row = slot / ROW_SPLIT;
    int part = slot - row * ROW_SPLIT;
    if (row < M) {
        unsigned long long row_base = (unsigned long long)row * (unsigned long long)ATTN_DIM;
        int m_tile = row / 128;
        int row_in_tile = row - m_tile * 128;
        unsigned long long sf_row_base = (unsigned long long)m_tile * (unsigned long long)SF_K_TILES * (unsigned long long)SF_TILE_BYTES + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4;
        int quad_lane = lane & 3;
        int block_in_vec = lane >> 2;
        int vec0 = part * (VECS_PER_LANE / ROW_SPLIT);
        #pragma unroll
        for (int batch = 0; batch < VECS_PER_LANE / ROW_SPLIT / LOAD_BATCH; batch++) {
            float xs[LOAD_BATCH * 8];
            #pragma unroll
            for (int u = 0; u < LOAD_BATCH; u++) {
                int i = vec0 + batch * LOAD_BATCH + u;
                int k = (lane + i * 32) * 8;
                int seg = k / seg_cols;
                int kk = k - seg * seg_cols;
                int src_row = seg * M + row;
                unsigned long long src = (unsigned long long)src_row * (unsigned long long)seg_cols + (unsigned long long)kk;
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(attn_out + src + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    xs[u * 8 + j] = _vec_load_0[j];
                }
            }
            #pragma unroll
            for (int u_1 = 0; u_1 < LOAD_BATCH; u_1++) {
                int i_1 = vec0 + batch * LOAD_BATCH + u_1;
                int k_1 = (lane + i_1 * 32) * 8;
                float vals[8];
                float mags[8];
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    vals[j_1] = xs[u_1 * 8 + j_1];
                    mags[j_1] = xs[u_1 * 8 + j_1];
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
                float mags_max = mags[0];
                #pragma unroll
                for (int _lr = 1; _lr < 8; _lr++) {
                    mags_max = max_noftz(mags_max, mags[_lr]);
                }
                float absmax = mags_max;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, absmax, 1);
                float o1 = _shfl_xor_0;
                float _max_0 = max_noftz(absmax, o1);
                absmax = _max_0;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, absmax, 2);
                float o2 = _shfl_xor_1;
                float _max_1 = max_noftz(absmax, o2);
                absmax = _max_1;
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
                for (int _ls = 0; _ls < 4; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(vals)[_ls], _scale2_1);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++) {
                    vals[_ls] = vals[_ls] * inverse;
                }
                #endif
                {
                    unsigned int _fp8_pk[2];
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
                    *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(a_q + (row_base + (unsigned long long)k_1)) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                }
                if (quad_lane == 0) {
                    int block = i_1 * 8 + block_in_vec;
                    int k_tile = block >> 2;
                    int sf_col = block & 3;
                    *(reinterpret_cast<unsigned char*>(a_sf + (sf_row_base + (unsigned long long)k_tile * (unsigned long long)SF_TILE_BYTES + (unsigned long long)sf_col)) + (0)) = (unsigned char)(scale_byte);
                }
            }
        }
    }
}

} // extern "C"

#undef ATTN_DIM
#undef LOAD_BATCH
#undef MINIMAX_H3_OUT_PROJ_INF
#undef NUM_MAIN_STAGES
#undef ROWS_PER_CTA
#undef ROW_SPLIT
#undef SF_K_TILES
#undef SF_TILE_BYTES
#undef THREADS
#undef VECS_PER_LANE

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
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
#define GROUP_M 32
#define WORK_STAGES 4
#define NUM_K_ITERS 28
#define N_TILES 21
#define SF_K_TILES 56
#define HIDDEN 5376
#define num_cluster_tiles ((m_tiles / CTA_GROUP) * N_TILES)
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_e4m3(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride)
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
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa_all_addr = smem + 66560;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb_all_addr = smem + 67584;
    uint8_t* smem_sfa0 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa0_addr = smem + 66560;
    uint8_t* smem_sfa1 = reinterpret_cast<uint8_t*>(smem_raw + 67072);
    const int smem_sfa1_addr = smem + 67072;
    uint8_t* smem_sfb0 = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb0_addr = smem + 67584;
    uint8_t* smem_sfb1 = reinterpret_cast<uint8_t*>(smem_raw + 68608);
    const int smem_sfb1_addr = smem + 68608;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 206848);
    const int work_response_addr = smem + 206848;

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
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
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
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                int global_row = off_m_1 + local_row;
                int col0 = col_part * 128;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                int _min_0 = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                int load_row = _min_0;
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_1 + (unsigned long long)col0;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_1 + (unsigned long long)col0;
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
                if (global_row < M) {
                    #pragma unroll
                    for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                        int col = n_chunk * 16;
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
                            uint4 _vld_1[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                        float _vec_load_2[8];
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
                                }
                            }
                        }
                        float _vec_load_3[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_3[_pair]));
                                }
                            }
                        }
                        float out_vals[16];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[n_chunk * 16 + j]);
                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                            float o_lo = _cvt_f32_0;
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_vec_load_0[j] * gate_mul * o_lo);
                            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                            float p_lo = _cvt_f32_1;
                            out_vals[j] = _vec_load_2[j] + p_lo;
                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_tmem_load_0[n_chunk * 16 + j + 8]);
                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                            float o_hi = _cvt_f32_2;
                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_vec_load_1[j] * gate_mul * o_hi);
                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                            float p_hi = _cvt_f32_3;
                            out_vals[j + 8] = _vec_load_3[j] + p_hi;
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
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#undef SMEM_SMEM_SFA_ALL_OFF
#undef SMEM_SMEM_SFA_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFA_ALL_STRIDE
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

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
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
#define GROUP_M 32
#define WORK_STAGES 4
#define NUM_K_ITERS 28
#define N_TILES 21
#define SF_K_TILES 56
#define HIDDEN 5376
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_e4m3_ts(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride, const __grid_constant__ CUtensorMap B_half, const __grid_constant__ CUtensorMap SFB_half, int full_units)
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
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa_all_addr = smem + 66560;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb_all_addr = smem + 67584;
    uint8_t* smem_sfa0 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa0_addr = smem + 66560;
    uint8_t* smem_sfa1 = reinterpret_cast<uint8_t*>(smem_raw + 67072);
    const int smem_sfa1_addr = smem + 67072;
    uint8_t* smem_sfb0 = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb0_addr = smem + 67584;
    uint8_t* smem_sfb1 = reinterpret_cast<uint8_t*>(smem_raw + 68608);
    const int smem_sfb1_addr = smem + 68608;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 206848);
    const int work_response_addr = smem + 206848;

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
                for (unsigned int _tile_iter = 0; _tile_iter < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter++) {
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
                    int unit = (int)(this_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid = this_bid;
                    int is_half = 0;
                    int half_h = 0;
                    if (unit >= full_units) {
                        int half_q = unit - full_units;
                        tile_bid = (unsigned int)((full_units + half_q / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h = half_q % 2;
                        is_half = 1;
                    }
                    int group = tile_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = tile_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    if (is_half == 0) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
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
                    } else {
                        int half_row = off_n + half_h * 128 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int k_group_1 = iter_k_1 * 2;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 68608, (&A), 0, off_m, k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 68608, (&B_half), 0, half_row, k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 68608 + 16384, (&B_half), 0, half_row, k_group_1 + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 68608, (&SFA), 0, 0, sfa_tile_row + k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 68608, (&SFB_half), 0, half_h * 4, sfb_tile_row + k_group_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(52224)) : "memory");
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
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
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int mma_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    int unit_1 = (int)(mma_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_1 = mma_bid;
                    int is_half_1 = 0;
                    int half_h_1 = 0;
                    if (unit_1 >= full_units) {
                        int half_q_1 = unit_1 - full_units;
                        tile_bid_1 = (unsigned int)((full_units + half_q_1 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_1 = half_q_1 % 2;
                        is_half_1 = 1;
                    }
                    int unit_n = 256 - is_half_1 * 128;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < NUM_K_ITERS; iter_k_2++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_2 == 0) ? 1 : 0);
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa0_addr) >> 4) + (mma_tma_stage) * 4288)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb0_addr) >> 4) + (mma_tma_stage) * 4288 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_sfa1_addr) >> 4) + (mma_tma_stage) * 4288)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb1_addr) >> 4) + (mma_tma_stage) * 4288 + 32)), _elected_0);
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                        int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                ((0x10c00000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1), _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                                ((0x30c00010U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                                ((0x50c00020U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                                ((0x70c00030U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa, tmem_tmem_sfb, 1, _elected_0);
                        }
                        int _smem_desc_lo_3 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                        int _smem_desc_lo_4 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 4288;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                ((0x10c00000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 2, b_desc + 2,
                                ((0x30c00010U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 4, b_desc + 4,
                                ((0x50c00020U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader(tmem_accum, a_desc + 6, b_desc + 6,
                                ((0x70c00030U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_0);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 3) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
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
                    mma_bid = _clc_ctaid_1 + (unsigned int)cta_rank;
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
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_2++) {
                int unit_2 = (int)(this_bid_1 / (unsigned int)CTA_GROUP);
                unsigned int tile_bid_2 = this_bid_1;
                int is_half_2 = 0;
                int half_h_2 = 0;
                if (unit_2 >= full_units) {
                    int half_q_2 = unit_2 - full_units;
                    tile_bid_2 = (unsigned int)((full_units + half_q_2 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h_2 = half_q_2 % 2;
                    is_half_2 = 1;
                }
                int group_1 = tile_bid_2 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = tile_bid_2 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                if (is_half_2 == 0) {
                    int global_row = off_m_1 + local_row;
                    int col0 = col_part * 128;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                    int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                    long long table_row = gate_index[load_row];
                    int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul = (float)in_range;
                    unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                    unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_1 + (unsigned long long)col0;
                    unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_1 + (unsigned long long)col0;
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
                    if (global_row < M) {
                        #pragma unroll
                        for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                            int col = n_chunk * 16;
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                                uint4 _vld_0[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_0[_blk] = _vptr_0[_blk];
                                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
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
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
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
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
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
                            float out_vals[16];
                            #pragma unroll
                            for (int j = 0; j < 8; j++) {
                                float o_lo = (float)(__nv_bfloat16)_tmem_load_0[n_chunk * 16 + j];
                                float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
                                out_vals[j] = _vec_load_2[j] + p_lo;
                                float o_hi = (float)(__nv_bfloat16)_tmem_load_0[n_chunk * 16 + j + 8];
                                float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
                                out_vals[j + 8] = _vec_load_3[j] + p_hi;
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
                                const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                            }
                        }
                    }
                } else {
                    int global_row_1 = off_m_1 + local_row;
                    int col0_1 = col_part * 64;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0_1;
                    int load_row_1 = ((global_row_1) < (M - 1) ? (global_row_1) : (M - 1));
                    long long table_row_1 = gate_index[load_row_1];
                    int in_range_1 = (((unsigned long long)table_row_1 < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul_1 = (float)in_range_1;
                    unsigned int safe_row_1 = (unsigned int)table_row_1 * (unsigned int)in_range_1;
                    unsigned long long gate_base_1 = (unsigned long long)safe_row_1 * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)(off_n_1 + half_h_2 * 128) + (unsigned long long)col0_1;
                    unsigned long long row_base_1 = (unsigned long long)global_row_1 * (unsigned long long)HIDDEN + (unsigned long long)(off_n_1 + half_h_2 * 128) + (unsigned long long)col0_1;
                    mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(lane_addr_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                    }
                    _phase_mainloop_done ^= 1;
                    if (global_row_1 < M) {
                        #pragma unroll
                        for (int n_chunk_1 = 0; n_chunk_1 < 4; n_chunk_1++) {
                            int col_1 = n_chunk_1 * 16;
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_5[8];
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_6[8];
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(residual + (row_base_1 + (unsigned long long)col_1) + 0);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_7[8];
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(residual + (row_base_1 + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float out_vals_1[16];
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 8; j_1++) {
                                float o_lo_1 = (float)(__nv_bfloat16)_tmem_load_1[n_chunk_1 * 16 + j_1];
                                float p_lo_1 = (float)(__nv_bfloat16)(_vec_load_4[j_1] * gate_mul_1 * o_lo_1);
                                out_vals_1[j_1] = _vec_load_6[j_1] + p_lo_1;
                                float o_hi_1 = (float)(__nv_bfloat16)_tmem_load_1[n_chunk_1 * 16 + j_1 + 8];
                                float p_hi_1 = (float)(__nv_bfloat16)(_vec_load_5[j_1] * gate_mul_1 * o_hi_1);
                                out_vals_1[j_1 + 8] = _vec_load_7[j_1] + p_hi_1;
                            }
                            unsigned int _packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 0], out_vals_1[_lp*2+1 + 0]));
                                _packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 2], out_vals_1[_lp*2+1 + 2]));
                                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 4], out_vals_1[_lp*2+1 + 4]));
                                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 6], out_vals_1[_lp*2+1 + 6]));
                                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 8], out_vals_1[_lp*2+1 + 8]));
                                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 10], out_vals_1[_lp*2+1 + 10]));
                                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 12], out_vals_1[_lp*2+1 + 12]));
                                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 14], out_vals_1[_lp*2+1 + 14]));
                                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(_packed_1);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_1))), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
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

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#undef tiles_per_group

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define ATTN_DIM 7168
#define ROWS_PER_CTA 4
#define VECS_PER_LANE 28
#define SF_K_TILES 112
#define SF_TILE_BYTES 512
#define LOAD_BATCH 14
#define ROW_SPLIT 2

extern "C" {

__global__ __launch_bounds__(128) void
kernel_minimax_h3_quant_nvfp4(__nv_bfloat16* __restrict__ attn_out, float* __restrict__ a_global_scale, unsigned int* __restrict__ a_q, uint8_t* __restrict__ a_sf, int M, int seg_cols)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int slot = bid * ROWS_PER_CTA + warp;
    int row = slot / ROW_SPLIT;
    int part = slot - row * ROW_SPLIT;
    if (row < M) {
        unsigned long long row_base = (unsigned long long)row * (unsigned long long)ATTN_DIM;
        unsigned long long word_base = row_base >> 3;
        int m_tile = row / 128;
        int row_in_tile = row - m_tile * 128;
        unsigned long long sf_row_base = (unsigned long long)m_tile * (unsigned long long)SF_K_TILES * (unsigned long long)SF_TILE_BYTES + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4;
        int pair_lane = lane & 1;
        int block_in_vec = lane >> 1;
        float g = a_global_scale[0];
        float _rcp_0 = approx_rcp(g);
        float inv_g = _rcp_0;
        float _rcp_1 = approx_rcp(6.0f);
        float sixth = _rcp_1;
        float zero_f32 = 0.0f;
        int vec0 = part * (VECS_PER_LANE / ROW_SPLIT);
        #pragma unroll
        for (int batch = 0; batch < VECS_PER_LANE / ROW_SPLIT / LOAD_BATCH; batch++) {
            float xs[LOAD_BATCH * 8];
            #pragma unroll
            for (int u = 0; u < LOAD_BATCH; u++) {
                int i = vec0 + batch * LOAD_BATCH + u;
                int k = (lane + i * 32) * 8;
                int seg = k / seg_cols;
                int kk = k - seg * seg_cols;
                int src_row = seg * M + row;
                unsigned long long src = (unsigned long long)src_row * (unsigned long long)seg_cols + (unsigned long long)kk;
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(attn_out + src + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    xs[u * 8 + j] = _vec_load_0[j];
                }
            }
            #pragma unroll
            for (int u_1 = 0; u_1 < LOAD_BATCH; u_1++) {
                int i_1 = vec0 + batch * LOAD_BATCH + u_1;
                int k_1 = (lane + i_1 * 32) * 8;
                float vals[8];
                float mags[8];
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    vals[j_1] = xs[u_1 * 8 + j_1];
                    mags[j_1] = xs[u_1 * 8 + j_1];
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
                float mags_max = mags[0];
                #pragma unroll
                for (int _lr = 1; _lr < 8; _lr++) {
                    mags_max = max_noftz(mags_max, mags[_lr]);
                }
                float absmax = mags_max;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, absmax, 1);
                float o1 = _shfl_xor_0;
                float _max_0 = max_noftz(absmax, o1);
                absmax = _max_0;
                float sf_value = g * (absmax * sixth);
                uint16_t _e4m3x2_f32_0;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(zero_f32), "f"(sf_value));
                uint16_t sf_pair = _e4m3x2_f32_0;
                unsigned int sf_byte = (unsigned int)sf_pair & 255;
                unsigned int sf_exp = sf_byte >> 3 & 15;
                unsigned int sf_mant = sf_byte & 7;
                unsigned int normal_bits = sf_exp + 120 << 23 | sf_mant << 20;
                float sf_normal = 0.0f;
                sf_normal = reinterpret_cast<float*>(&normal_bits)[0];
                float sf_subnormal = (float)sf_mant * 0.001953125f;
                float sf_f = ((sf_exp == 0) ? sf_subnormal : sf_normal);
                float _rcp_2 = approx_rcp(sf_f * inv_g);
                float out_scale_nonzero = _rcp_2;
                float out_scale = ((absmax == 0.0f) ? 0.0f : out_scale_nonzero);
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_1 = {out_scale, out_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 4; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(vals)[_ls], _scale2_1);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++) {
                    vals[_ls] = vals[_ls] * out_scale;
                }
                #endif
                unsigned int packed[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(vals[0]), "f"(vals[1]), "f"(vals[2]), "f"(vals[3]), "f"(vals[4]), "f"(vals[5]), "f"(vals[6]), "f"(vals[7]));
                *(reinterpret_cast<unsigned int*>(a_q + (word_base + (unsigned long long)(k_1 >> 3))) + (0)) = packed[0];
                if (pair_lane == 0) {
                    int block = i_1 * 16 + block_in_vec;
                    int k_tile = block >> 2;
                    int sf_col = block & 3;
                    *(reinterpret_cast<unsigned char*>(a_sf + (sf_row_base + (unsigned long long)k_tile * (unsigned long long)SF_TILE_BYTES + (unsigned long long)sf_col)) + (0)) = (unsigned char)(sf_byte);
                }
            }
        }
    }
}

} // extern "C"

#undef ATTN_DIM
#undef LOAD_BATCH
#undef MINIMAX_H3_OUT_PROJ_INF
#undef NUM_MAIN_STAGES
#undef ROWS_PER_CTA
#undef ROW_SPLIT
#undef SF_K_TILES
#undef SF_TILE_BYTES
#undef THREADS
#undef VECS_PER_LANE

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
#if __CUDA_ARCH__ == 1030
#define TMEM_NCOLS 400
#else
#define TMEM_NCOLS 304
#endif
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#if __CUDA_ARCH__ == 1030
#define TMEM_TMEM_SFB_OFFSET 304
#else
#define TMEM_TMEM_SFB_OFFSET 272
#endif
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
#if __CUDA_ARCH__ == 1030
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
#endif
#define SMEM_WORK_RESPONSE_OFF 195584
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#if __CUDA_ARCH__ == 1030
#define SMEM_SMEM_V13_OFF 1120
#define SMEM_SMEM_V13_STAGE_BYTES 6144
#define SMEM_SMEM_V13_STRIDE 38912
#define SMEM_SMEM_V14_OFF 1136
#define SMEM_SMEM_V14_STAGE_BYTES 6144
#define SMEM_SMEM_V14_STRIDE 38912
#define SMEM_SMEM_V15_OFF 17504
#define SMEM_SMEM_V15_STAGE_BYTES 6144
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 17520
#define SMEM_SMEM_V16_STAGE_BYTES 6144
#define SMEM_SMEM_V16_STRIDE 38912
#endif
#define SMEM_TOTAL 195712
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 8
#define WORK_STAGES 4
#define NUM_K_ITERS 28
#define N_TILES 21
#define SF_K_TILES 112
#define HIDDEN 5376
#if __CUDA_ARCH__ == 1030
#else
#define num_cluster_tiles ((m_tiles / CTA_GROUP) * N_TILES)
#endif
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_e2m1(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride)
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
#if __CUDA_ARCH__ == 1030
    uint8_t* smem_v13 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v13_addr = smem + 1120;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v14_addr = smem + 1136;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v15_addr = smem + 17504;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v16_addr = smem + 17520;
#endif

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
#if __CUDA_ARCH__ == 1030
#else

#endif
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

#if __CUDA_ARCH__ == 1030
    // TMEM alloc (512 columns, 400 used)
#else
    // TMEM alloc (512 columns, 304 used)
#endif
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
#if __CUDA_ARCH__ == 1030
#else
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
#if __CUDA_ARCH__ == 1030
    const int tmem_tmem_sfb = taddr + 304;
#else
    const int tmem_tmem_sfb = taddr + 272;
#endif

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
#if __CUDA_ARCH__ == 1030
                for (unsigned int _tile_iter = 0; _tile_iter < m_tiles / CTA_GROUP * N_TILES; _tile_iter++) {
#else
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
#endif
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
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
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
#if __CUDA_ARCH__ == 1030
                        : "+r"(_clc_ctaid_0)
#else
                        : "=r"(_clc_ctaid_0)
#endif
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
#if __CUDA_ARCH__ == 1030
#else
            unsigned int acc_stage = 0;
#endif
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
#else
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
#endif
                    #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                    for (int super_k = 0; super_k < 9; super_k++) {
#else
                    for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
#endif
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1030
                        unsigned int stage_p0 = mma_tma_stage;
                        int init_flag = ((super_k == 0) ? 1 : 0);
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                        int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                        {
                            int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1 | ((uint64_t)((uint32_t)_smem_desc_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_4 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2 | ((uint64_t)((uint32_t)_smem_desc_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
#else
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);
#endif

#if __CUDA_ARCH__ == 1030
                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
                        }
                        int _smem_desc_lo_5 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                        int _smem_desc_lo_6 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                        {
                            int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5 | ((uint64_t)((uint32_t)_smem_desc_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_8 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6 | ((uint64_t)((uint32_t)_smem_desc_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
#else
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1));
                            }
                            int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);
#endif

#if __CUDA_ARCH__ == 1030
                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p1 = mma_tma_stage;
                        uint32_t _elected_9 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                        int _smem_desc_lo_10 = (((smem_v13_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                        int _smem_desc_lo_11 = (((smem_v15_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                        {
                            int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10 | ((uint64_t)((uint32_t)_smem_desc_lo_12 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_13 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11 | ((uint64_t)((uint32_t)_smem_desc_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
#else
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1);
                            }
                            int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);
#endif

#if __CUDA_ARCH__ == 1030
                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_9);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (stage_p0) * 8, (uint16_t)(3));
                        uint32_t _elected_14 = elect_sync();
                        int _smem_desc_lo_15 = (((smem_a_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                        int _smem_desc_lo_16 = (((smem_b_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                        {
                            int _smem_desc_lo_17 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15 | ((uint64_t)((uint32_t)_smem_desc_lo_17 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_18 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16 | ((uint64_t)((uint32_t)_smem_desc_lo_18 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
#else
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1);
                            }
                            int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);
#endif

#if __CUDA_ARCH__ == 1030
                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1, _elected_14);
#else
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1);
                            }
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        int _smem_desc_lo_19 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                        int _smem_desc_lo_20 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                        {
                            int _smem_desc_lo_21 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19 | ((uint64_t)((uint32_t)_smem_desc_lo_21 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_22 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20 | ((uint64_t)((uint32_t)_smem_desc_lo_22 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1, _elected_14);
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p2 = mma_tma_stage;
                        uint32_t _elected_23 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                        int _smem_desc_lo_24 = (((smem_v14_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                        int _smem_desc_lo_25 = (((smem_v16_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                        {
                            int _smem_desc_lo_26 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_24 | ((uint64_t)((uint32_t)_smem_desc_lo_26 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_27 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25 | ((uint64_t)((uint32_t)_smem_desc_lo_27 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1, _elected_23);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (stage_p1) * 8, (uint16_t)(3));
                        uint32_t _elected_28 = elect_sync();
                        int _smem_desc_lo_29 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                        int _smem_desc_lo_30 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                        {
                            int _smem_desc_lo_31 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_29 | ((uint64_t)((uint32_t)_smem_desc_lo_31 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_32 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_30 | ((uint64_t)((uint32_t)_smem_desc_lo_32 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1, _elected_28);
                        }
                        int _smem_desc_lo_33 = (((smem_a_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                        int _smem_desc_lo_34 = (((smem_b_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                        {
                            int _smem_desc_lo_35 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_33 | ((uint64_t)((uint32_t)_smem_desc_lo_35 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_36 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_34 | ((uint64_t)((uint32_t)_smem_desc_lo_36 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1, _elected_28);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (stage_p2) * 8, (uint16_t)(3));
#else
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
#endif
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
#if __CUDA_ARCH__ == 1030
                    mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_37 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                    tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                    int _smem_desc_lo_38 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    int _smem_desc_lo_39 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_38) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_39) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_37);
                    }
                    int _smem_desc_lo_40 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    int _smem_desc_lo_41 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_40) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_41) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10400480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_37);
                    }
                    int _smem_desc_lo_42 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    int _smem_desc_lo_43 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_42) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_43) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10400480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_37);
                    }
                    int _smem_desc_lo_44 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    int _smem_desc_lo_45 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_44) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_45) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                            0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_37);
                    }
                    elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
#else
                    elect_commit_cg2_multicast(mainloop_done_addr + (acc_stage) * 8, (uint16_t)(3));
#endif
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
#if __CUDA_ARCH__ == 1030
                        : "+r"(_clc_ctaid_1)
#else
                        : "=r"(_clc_ctaid_1)
#endif
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#if __CUDA_ARCH__ == 1030
                    asm volatile("tcgen05.fence::before_thread_sync;");
#endif
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
#if __CUDA_ARCH__ == 1030
#else
            unsigned int acc_stage_1 = 0;
#endif
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            float alpha_v = alpha[0];
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
#if __CUDA_ARCH__ == 1030
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_2++) {
#else
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
#endif
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                int global_row = off_m_1 + local_row;
                int col0 = col_part * 128;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
#if __CUDA_ARCH__ == 1030
                int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
#else
                int _min_0 = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                int load_row = _min_0;
#endif
                long long table_row = gate_index[load_row];
                int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                float gate_mul = (float)in_range;
                unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_1 + (unsigned long long)col0;
                unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_1 + (unsigned long long)col0;
#if __CUDA_ARCH__ == 1030
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
#else
                mbarrier_wait(mainloop_done_addr + (acc_stage_1) * 8, _phase_mainloop_done);
#endif
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
#if __CUDA_ARCH__ == 1030
                    asm volatile("tcgen05.fence::before_thread_sync;");
#endif
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
#if __CUDA_ARCH__ == 1030
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
#else
                        :: "r"((epilogue_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
#endif
                }
                _phase_mainloop_done ^= 1;
                if (global_row < M) {
                    #pragma unroll
                    for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                        int col = n_chunk * 16;
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1030
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
#else
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
#endif
                                }
                            }
                        }
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
                            uint4 _vld_1[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1030
                                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
#else
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
#endif
                                }
                            }
                        }
                        float _vec_load_2[8];
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1030
                                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
#else
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
#endif
                                }
                            }
                        }
                        float _vec_load_3[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
#if __CUDA_ARCH__ == 1030
                                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
#else
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_3[_pair]));
#endif
                                }
                            }
                        }
                        float out_vals[16];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
#if __CUDA_ARCH__ == 1030
                            float o_lo = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j] * alpha_v);
                            float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
#else
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[n_chunk * 16 + j] * alpha_v);
                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                            float o_lo = _cvt_f32_0;
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_vec_load_0[j] * gate_mul * o_lo);
                            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                            float p_lo = _cvt_f32_1;
#endif
                            out_vals[j] = _vec_load_2[j] + p_lo;
#if __CUDA_ARCH__ == 1030
                            float o_hi = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j + 8] * alpha_v);
                            float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
#else
                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_tmem_load_0[n_chunk * 16 + j + 8] * alpha_v);
                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                            float o_hi = _cvt_f32_2;
                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_vec_load_1[j] * gate_mul * o_hi);
                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                            float p_hi = _cvt_f32_3;
#endif
                            out_vals[j + 8] = _vec_load_3[j] + p_hi;
                        }
#if __CUDA_ARCH__ == 1030
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
#endif
                        {
#if __CUDA_ARCH__ == 1030
                            const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
#else
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
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "+r"(_clc_ctaid_2)
#else
                    : "=r"(_clc_ctaid_2)
#endif
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#if __CUDA_ARCH__ == 1030
                asm volatile("tcgen05.fence::before_thread_sync;");
#endif
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
#if __CUDA_ARCH__ == 1030
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
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
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#undef SMEM_SMEM_SFA_ALL_OFF
#undef SMEM_SMEM_SFA_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFA_ALL_STRIDE
#undef SMEM_SMEM_SFB_ALL_OFF
#undef SMEM_SMEM_SFB_ALL_STAGE_BYTES
#undef SMEM_SMEM_SFB_ALL_STRIDE
#if __CUDA_ARCH__ == 1030
#undef SMEM_SMEM_V10_OFF
#undef SMEM_SMEM_V10_STAGE_BYTES
#undef SMEM_SMEM_V10_STRIDE
#undef SMEM_SMEM_V11_OFF
#undef SMEM_SMEM_V11_STAGE_BYTES
#undef SMEM_SMEM_V11_STRIDE
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
#endif
#undef SMEM_TOTAL
#undef SMEM_WORK_RESPONSE_OFF
#undef SMEM_WORK_RESPONSE_STAGE_BYTES
#undef SMEM_WORK_RESPONSE_STRIDE
#undef TMEM_ACCUM_OFFSET
#undef TMEM_NCOLS
#undef TMEM_TMEM_SFA_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef WORK_STAGES
#if __CUDA_ARCH__ == 1030
#else
#undef epilogue_done_addr
#undef mainloop_done_addr
#undef mma_done_addr
#undef num_cluster_tiles
#undef smem_a_addr
#undef smem_b_addr
#undef smem_sfa_all_addr
#undef smem_sfb_all_addr
#undef smem_v10_addr
#undef smem_v11_addr
#undef smem_v4_addr
#undef smem_v5_addr
#undef smem_v6_addr
#undef smem_v7_addr
#undef smem_v8_addr
#undef smem_v9_addr
#endif
#undef tiles_per_group
#if __CUDA_ARCH__ == 1030
#else
#undef tma_full_addr
#undef work_empty_addr
#undef work_full_addr
#undef work_response_addr
#endif

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
#if __CUDA_ARCH__ == 1030
#define TMEM_NCOLS 448
#else
#define TMEM_NCOLS 304
#endif
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#if __CUDA_ARCH__ == 1030
#define TMEM_TMEM_SFB_OFFSET 304
#define TMEM_TMEM_SFB_H_OFFSET 400
#else
#define TMEM_TMEM_SFB_OFFSET 272
#endif
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
#if __CUDA_ARCH__ == 1030
#define SMEM_SMEM_V13_OFF 35840
#define SMEM_SMEM_V13_STAGE_BYTES 512
#define SMEM_SMEM_V13_STRIDE 38912
#define SMEM_SMEM_V14_OFF 36352
#define SMEM_SMEM_V14_STAGE_BYTES 512
#define SMEM_SMEM_V14_STRIDE 38912
#define SMEM_SMEM_V15_OFF 36864
#define SMEM_SMEM_V15_STAGE_BYTES 512
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 37376
#define SMEM_SMEM_V16_STAGE_BYTES 512
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_V17_OFF 1120
#define SMEM_SMEM_V17_STAGE_BYTES 6144
#define SMEM_SMEM_V17_STRIDE 38912
#define SMEM_SMEM_V18_OFF 1136
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 38912
#define SMEM_SMEM_V19_OFF 17504
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 38912
#define SMEM_SMEM_V20_OFF 17520
#define SMEM_SMEM_V20_STAGE_BYTES 6144
#define SMEM_SMEM_V20_STRIDE 38912
#endif
#define SMEM_TOTAL 195712
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 8
#define WORK_STAGES 4
#define NUM_K_ITERS 28
#define N_TILES 21
#define SF_K_TILES 112
#define HIDDEN 5376
#define tiles_per_group (GROUP_M * N_TILES)

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_e2m1_ts(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride, const __grid_constant__ CUtensorMap B_half, const __grid_constant__ CUtensorMap SFB_half, int full_units)
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
#if __CUDA_ARCH__ == 1030
    uint8_t* smem_v13 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v13_addr = smem + 35840;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 36352);
    const int smem_v14_addr = smem + 36352;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v15_addr = smem + 36864;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 37376);
    const int smem_v16_addr = smem + 37376;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v17_addr = smem + 1120;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v18_addr = smem + 1136;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v19_addr = smem + 17504;
    uint8_t* smem_v20 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v20_addr = smem + 17520;
#endif

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

#if __CUDA_ARCH__ == 1030
    // TMEM alloc (512 columns, 448 used)
#else
    // TMEM alloc (512 columns, 304 used)
#endif
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
#if __CUDA_ARCH__ == 1030
    const int tmem_tmem_sfb = taddr + 304;
    const int tmem_tmem_sfb_h = taddr + 400;
#else
    const int tmem_tmem_sfb = taddr + 272;
#endif

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
                for (unsigned int _tile_iter = 0; _tile_iter < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter++) {
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
                    int unit = (int)(this_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid = this_bid;
                    int is_half = 0;
                    int half_h = 0;
                    if (unit >= full_units) {
                        int half_q = unit - full_units;
                        tile_bid = (unsigned int)((full_units + half_q / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h = half_q % 2;
                        is_half = 1;
                    }
                    int group = tile_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = tile_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int weight_row = off_n + cta_rank * B_HALF_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    if (is_half == 0) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
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
                    } else {
                        int half_row = off_n + half_h * 128 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B_half), 0, half_row, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            int k_set_base_1 = iter_k_1 * 4;
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB_half), 0, half_h * 4, sfb_tile_row + k_set_base_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
#if __CUDA_ARCH__ == 1030
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(28672)) : "memory");
#else
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(30720)) : "memory");
#endif
                            load_stage += 1;
                            if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
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
                unsigned int mma_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    int unit_1 = (int)(mma_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_1 = mma_bid;
                    int is_half_1 = 0;
                    int half_h_1 = 0;
                    if (unit_1 >= full_units) {
                        int half_q_1 = unit_1 - full_units;
                        tile_bid_1 = (unsigned int)((full_units + half_q_1 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_1 = half_q_1 % 2;
                        is_half_1 = 1;
                    }
#if __CUDA_ARCH__ == 1030
                    if (is_half_1 == 0) {
                        #pragma unroll 1
                        for (int super_k = 0; super_k < 9; super_k++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0 = mma_tma_stage;
                            int init_flag = ((super_k == 0) ? 1 : 0);
                            uint32_t _elected_0 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1 | ((uint64_t)((uint32_t)_smem_desc_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_4 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2 | ((uint64_t)((uint32_t)_smem_desc_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
                            }
                            int _smem_desc_lo_5 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_6 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5 | ((uint64_t)((uint32_t)_smem_desc_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_8 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6 | ((uint64_t)((uint32_t)_smem_desc_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1 = mma_tma_stage;
                            uint32_t _elected_9 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            int _smem_desc_lo_10 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_11 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10 | ((uint64_t)((uint32_t)_smem_desc_lo_12 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_13 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11 | ((uint64_t)((uint32_t)_smem_desc_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_9);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0) * 8, (uint16_t)(3));
                            uint32_t _elected_14 = elect_sync();
                            int _smem_desc_lo_15 = (((smem_a_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_16 = (((smem_b_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_17 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15 | ((uint64_t)((uint32_t)_smem_desc_lo_17 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_18 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16 | ((uint64_t)((uint32_t)_smem_desc_lo_18 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1, _elected_14);
                            }
                            int _smem_desc_lo_19 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_20 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_21 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19 | ((uint64_t)((uint32_t)_smem_desc_lo_21 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_22 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20 | ((uint64_t)((uint32_t)_smem_desc_lo_22 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1, _elected_14);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2 = mma_tma_stage;
                            uint32_t _elected_23 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            int _smem_desc_lo_24 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_25 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_26 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_24 | ((uint64_t)((uint32_t)_smem_desc_lo_26 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_27 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25 | ((uint64_t)((uint32_t)_smem_desc_lo_27 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1, _elected_23);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1) * 8, (uint16_t)(3));
                            uint32_t _elected_28 = elect_sync();
                            int _smem_desc_lo_29 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_30 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_31 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_29 | ((uint64_t)((uint32_t)_smem_desc_lo_31 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_32 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_30 | ((uint64_t)((uint32_t)_smem_desc_lo_32 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1, _elected_28);
                            }
                            int _smem_desc_lo_33 = (((smem_a_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_34 = (((smem_b_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_35 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_33 | ((uint64_t)((uint32_t)_smem_desc_lo_35 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_36 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_34 | ((uint64_t)((uint32_t)_smem_desc_lo_36 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1, _elected_28);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p2) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
#else
                    int unit_n = 256 - is_half_1 * 128;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < NUM_K_ITERS; iter_k_2++) {
#endif
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1030
                        uint32_t _elected_37 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        int _smem_desc_lo_38 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_39 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
#else
                        int init_flag = ((iter_k_2 == 0) ? 1 : 0);
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
#endif
                        {
#if __CUDA_ARCH__ == 1030
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_38) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_39) | ((uint64_t)0x40004040 << 32);
#else
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);
#endif

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
#if __CUDA_ARCH__ == 1030
                                0x10400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_37);
#else
                                ((0x10400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        int _smem_desc_lo_40 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_41 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
#else
                        int _smem_desc_lo_3 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_4 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
#endif
                        {
#if __CUDA_ARCH__ == 1030
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_40) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_41) | ((uint64_t)0x40004040 << 32);
#else
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);
#endif

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
#if __CUDA_ARCH__ == 1030
                                0x10400480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_37);
#else
                                ((0x10400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        int _smem_desc_lo_42 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_43 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
#else
                        int _smem_desc_lo_5 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_6 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
#endif
                        {
#if __CUDA_ARCH__ == 1030
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_42) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_43) | ((uint64_t)0x40004040 << 32);
#else
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x40004040 << 32);
#endif

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
#if __CUDA_ARCH__ == 1030
                                0x10400480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_37);
#else
                                ((0x10400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_0);
#endif
                        }
#if __CUDA_ARCH__ == 1030
                        int _smem_desc_lo_44 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_45 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
#else
                        int _smem_desc_lo_7 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_8 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
#endif
                        {
#if __CUDA_ARCH__ == 1030
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_44) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_45) | ((uint64_t)0x40004040 << 32);
#else
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x40004040 << 32);
#endif

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
#if __CUDA_ARCH__ == 1030
                                0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_37);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    } else {
                        #pragma unroll 1
                        for (int super_k_1 = 0; super_k_1 < 9; super_k_1++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0_1 = mma_tma_stage;
                            int init_flag_1 = ((super_k_1 == 0) ? 1 : 0);
                            uint32_t _elected_46 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 4, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 8, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 12, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            int _smem_desc_lo_47 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_48 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_49 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_47 | ((uint64_t)((uint32_t)_smem_desc_lo_49 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_50 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_48 | ((uint64_t)((uint32_t)_smem_desc_lo_50 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 0, tmem_tmem_sfb_h + 0, ((init_flag_1) ? 0 : 1), _elected_46);
                            }
                            int _smem_desc_lo_51 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_52 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_53 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_51 | ((uint64_t)((uint32_t)_smem_desc_lo_53 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_54 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_52 | ((uint64_t)((uint32_t)_smem_desc_lo_54 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb_h + 4 + 0, 1, _elected_46);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1_1 = mma_tma_stage;
                            uint32_t _elected_55 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 16, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 20, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 24, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 28, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            int _smem_desc_lo_56 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_57 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_58 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_56 | ((uint64_t)((uint32_t)_smem_desc_lo_58 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_59 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_57 | ((uint64_t)((uint32_t)_smem_desc_lo_59 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb_h + 12 + 0, 1, _elected_55);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0_1) * 8, (uint16_t)(3));
                            uint32_t _elected_60 = elect_sync();
                            int _smem_desc_lo_61 = (((smem_a_addr + 16) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_62 = (((smem_b_addr + 16) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_63 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_61 | ((uint64_t)((uint32_t)_smem_desc_lo_63 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_64 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_62 | ((uint64_t)((uint32_t)_smem_desc_lo_64 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb_h + 16 + 0, 1, _elected_60);
                            }
                            int _smem_desc_lo_65 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_66 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_67 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_65 | ((uint64_t)((uint32_t)_smem_desc_lo_67 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_68 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_66 | ((uint64_t)((uint32_t)_smem_desc_lo_68 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb_h + 24 + 0, 1, _elected_60);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2_1 = mma_tma_stage;
                            uint32_t _elected_69 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 32, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 36, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 40, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 44, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            int _smem_desc_lo_70 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_71 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_72 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_70 | ((uint64_t)((uint32_t)_smem_desc_lo_72 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_73 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_71 | ((uint64_t)((uint32_t)_smem_desc_lo_73 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb_h + 28 + 0, 1, _elected_69);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1_1) * 8, (uint16_t)(3));
                            uint32_t _elected_74 = elect_sync();
                            int _smem_desc_lo_75 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            int _smem_desc_lo_76 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            {
                                int _smem_desc_lo_77 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_75 | ((uint64_t)((uint32_t)_smem_desc_lo_77 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_78 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_76 | ((uint64_t)((uint32_t)_smem_desc_lo_78 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb_h + 36 + 0, 1, _elected_74);
                            }
                            int _smem_desc_lo_79 = (((smem_a_addr + 80) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            int _smem_desc_lo_80 = (((smem_b_addr + 80) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            {
                                int _smem_desc_lo_81 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_79 | ((uint64_t)((uint32_t)_smem_desc_lo_81 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_82 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_80 | ((uint64_t)((uint32_t)_smem_desc_lo_82 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb_h + 40 + 0, 1, _elected_74);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p2_1) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_83 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 4, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 8, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 12, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        int _smem_desc_lo_84 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_85 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_84) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_85) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb_h + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_86 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_87 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_86) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_87) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb_h + 4 + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_88 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_89 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_88) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_89) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb_h + 8 + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_90 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_91 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_90) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_91) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb_h + 12 + 0, 1, _elected_83);
#else
                                ((0x10400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(unit_n) >> 3) << 17)), tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_0);
#endif
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
                    mma_bid = _clc_ctaid_1 + (unsigned int)cta_rank;
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
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_2++) {
                int unit_2 = (int)(this_bid_1 / (unsigned int)CTA_GROUP);
                unsigned int tile_bid_2 = this_bid_1;
                int is_half_2 = 0;
                int half_h_2 = 0;
                if (unit_2 >= full_units) {
                    int half_q_2 = unit_2 - full_units;
                    tile_bid_2 = (unsigned int)((full_units + half_q_2 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h_2 = half_q_2 % 2;
                    is_half_2 = 1;
                }
                int group_1 = tile_bid_2 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = tile_bid_2 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                if (is_half_2 == 0) {
                    int global_row = off_m_1 + local_row;
                    int col0 = col_part * 128;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                    int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                    long long table_row = gate_index[load_row];
                    int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul = (float)in_range;
                    unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                    unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_1 + (unsigned long long)col0;
                    unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_1 + (unsigned long long)col0;
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
                    if (global_row < M) {
                        #pragma unroll
                        for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                            int col = n_chunk * 16;
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                                uint4 _vld_0[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_0[_blk] = _vptr_0[_blk];
                                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
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
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col) + 0);
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
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + (row_base + (unsigned long long)col + 8) + 0);
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
                            float out_vals[16];
                            #pragma unroll
                            for (int j = 0; j < 8; j++) {
                                float o_lo = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j] * alpha_v);
                                float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
                                out_vals[j] = _vec_load_2[j] + p_lo;
                                float o_hi = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j + 8] * alpha_v);
                                float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
                                out_vals[j + 8] = _vec_load_3[j] + p_hi;
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
                                const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                            }
                        }
                    }
                } else {
                    int global_row_1 = off_m_1 + local_row;
                    int col0_1 = col_part * 64;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0_1;
                    int load_row_1 = ((global_row_1) < (M - 1) ? (global_row_1) : (M - 1));
                    long long table_row_1 = gate_index[load_row_1];
                    int in_range_1 = (((unsigned long long)table_row_1 < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul_1 = (float)in_range_1;
                    unsigned int safe_row_1 = (unsigned int)table_row_1 * (unsigned int)in_range_1;
                    unsigned long long gate_base_1 = (unsigned long long)safe_row_1 * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)(off_n_1 + half_h_2 * 128) + (unsigned long long)col0_1;
                    unsigned long long row_base_1 = (unsigned long long)global_row_1 * (unsigned long long)HIDDEN + (unsigned long long)(off_n_1 + half_h_2 * 128) + (unsigned long long)col0_1;
                    mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(lane_addr_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                    }
                    _phase_mainloop_done ^= 1;
                    if (global_row_1 < M) {
                        #pragma unroll
                        for (int n_chunk_1 = 0; n_chunk_1 < 4; n_chunk_1++) {
                            int col_1 = n_chunk_1 * 16;
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_5[8];
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_6[8];
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(residual + (row_base_1 + (unsigned long long)col_1) + 0);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_7[8];
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(residual + (row_base_1 + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float out_vals_1[16];
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 8; j_1++) {
                                float o_lo_1 = (float)(__nv_bfloat16)(_tmem_load_1[n_chunk_1 * 16 + j_1] * alpha_v);
                                float p_lo_1 = (float)(__nv_bfloat16)(_vec_load_4[j_1] * gate_mul_1 * o_lo_1);
                                out_vals_1[j_1] = _vec_load_6[j_1] + p_lo_1;
                                float o_hi_1 = (float)(__nv_bfloat16)(_tmem_load_1[n_chunk_1 * 16 + j_1 + 8] * alpha_v);
                                float p_hi_1 = (float)(__nv_bfloat16)(_vec_load_5[j_1] * gate_mul_1 * o_hi_1);
                                out_vals_1[j_1 + 8] = _vec_load_7[j_1] + p_hi_1;
                            }
                            unsigned int _packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 0], out_vals_1[_lp*2+1 + 0]));
                                _packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 2], out_vals_1[_lp*2+1 + 2]));
                                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 4], out_vals_1[_lp*2+1 + 4]));
                                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 6], out_vals_1[_lp*2+1 + 6]));
                                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 8], out_vals_1[_lp*2+1 + 8]));
                                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 10], out_vals_1[_lp*2+1 + 10]));
                                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 12], out_vals_1[_lp*2+1 + 12]));
                                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 14], out_vals_1[_lp*2+1 + 14]));
                                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(_packed_1);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_1))), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
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

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#if __CUDA_ARCH__ == 1030
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
#undef SMEM_SMEM_V19_OFF
#undef SMEM_SMEM_V19_STAGE_BYTES
#undef SMEM_SMEM_V19_STRIDE
#undef SMEM_SMEM_V20_OFF
#undef SMEM_SMEM_V20_STAGE_BYTES
#undef SMEM_SMEM_V20_STRIDE
#endif
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
#if __CUDA_ARCH__ == 1030
#undef TMEM_TMEM_SFB_H_OFFSET
#endif
#undef TMEM_TMEM_SFB_OFFSET
#undef WORK_STAGES
#undef tiles_per_group

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
#define TMEM_NCOLS 448
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 304
#define TMEM_TMEM_SFB_H_OFFSET 400
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
#define SMEM_SMEM_V13_OFF 35840
#define SMEM_SMEM_V13_STAGE_BYTES 512
#define SMEM_SMEM_V13_STRIDE 38912
#define SMEM_SMEM_V14_OFF 36352
#define SMEM_SMEM_V14_STAGE_BYTES 512
#define SMEM_SMEM_V14_STRIDE 38912
#define SMEM_SMEM_V15_OFF 36864
#define SMEM_SMEM_V15_STAGE_BYTES 512
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 37376
#define SMEM_SMEM_V16_STAGE_BYTES 512
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_V17_OFF 1120
#define SMEM_SMEM_V17_STAGE_BYTES 6144
#define SMEM_SMEM_V17_STRIDE 38912
#define SMEM_SMEM_V18_OFF 1136
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 38912
#define SMEM_SMEM_V19_OFF 17504
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 38912
#define SMEM_SMEM_V20_OFF 17520
#define SMEM_SMEM_V20_STAGE_BYTES 6144
#define SMEM_SMEM_V20_STRIDE 38912
#define SMEM_TOTAL 195712
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 8
#define WORK_STAGES 4
#define NUM_K_ITERS 28
#define N_TILES 21
#define SF_K_TILES 112
#define HIDDEN 5376
#define tiles_per_group (GROUP_M * N_TILES)

#if MINIMAX_H3_SM_MINOR == 3  // sm_103a only
extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_e2m1_ts_wr_pw(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride, const __grid_constant__ CUtensorMap B_half, const __grid_constant__ CUtensorMap SFB_half, int full_units)
{
#if __CUDA_ARCH__ == 1030
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
    uint8_t* smem_v13 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v13_addr = smem + 35840;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 36352);
    const int smem_v14_addr = smem + 36352;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v15_addr = smem + 36864;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 37376);
    const int smem_v16_addr = smem + 37376;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v17_addr = smem + 1120;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v18_addr = smem + 1136;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v19_addr = smem + 17504;
    uint8_t* smem_v20 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v20_addr = smem + 17520;

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

    // TMEM alloc (512 columns, 448 used)
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
    const int tmem_tmem_sfb = taddr + 304;
    const int tmem_tmem_sfb_h = taddr + 400;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int pre_b = 0;
            if (elect_sync()) {
                unsigned int first_bid = bid;
                int unit = (int)(first_bid / (unsigned int)CTA_GROUP);
                unsigned int tile_bid = first_bid;
                int is_half = 0;
                int half_h = 0;
                if (unit >= full_units) {
                    int half_q = unit - full_units;
                    tile_bid = (unsigned int)((full_units + half_q / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h = half_q % 2;
                    is_half = 1;
                }
                int group = tile_bid / (unsigned int)tiles_per_group;
                int first_m = group * GROUP_M;
                int remaining = m_tiles - first_m;
                int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                int local = tile_bid % (unsigned int)tiles_per_group;
                int bid_m = first_m + local % group_size;
                int bid_n = local / group_size;
                int off_m = bid_m * BLOCK_M;
                int off_n = bid_n * BLOCK_N;
                if (is_half == 0) {
                    pre_b = 1;
                    int first_weight_row = off_n + cta_rank * B_HALF_N;
                    int first_sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll
                    for (int stage = 0; stage < NUM_STAGES; stage++) {
                        tma_3d_gmem2smem_cta2(smem_b_addr + (unsigned int)(stage * 38912), (&B), 0, first_weight_row, stage, ((tma_full_addr + (stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + (unsigned int)(stage * 38912), (&SFB), 0, 0, first_sfb_tile_row + stage * 4, ((tma_full_addr + (stage) * 8) & 0xFEFFFFFF));
                    }
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter++) {
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
                    int unit_1 = (int)(this_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_1 = this_bid;
                    int is_half_1 = 0;
                    int half_h_1 = 0;
                    if (unit_1 >= full_units) {
                        int half_q_1 = unit_1 - full_units;
                        tile_bid_1 = (unsigned int)((full_units + half_q_1 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_1 = half_q_1 % 2;
                        is_half_1 = 1;
                    }
                    int group_1 = tile_bid_1 / (unsigned int)tiles_per_group;
                    int first_m_1 = group_1 * GROUP_M;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                    int local_1 = tile_bid_1 % (unsigned int)tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * BLOCK_M;
                    int off_n_1 = bid_n_1 * BLOCK_N;
                    int weight_row = off_n_1 + cta_rank * B_HALF_N;
                    int sfa_tile_row = bid_m_1 * SF_K_TILES;
                    int sfb_tile_row = bid_n_1 * SF_K_TILES;
                    if (is_half_1 == 0) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m_1, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            int weights_pending = pre_b * ((iter_k < NUM_STAGES) ? 1 : 0);
                            if (weights_pending == 0) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B), 0, weight_row, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            int k_set_base = iter_k * 4;
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            if (weights_pending == 0) {
                                tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB), 0, 0, sfb_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(38912)) : "memory");
                            load_stage += 1;
                            if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    } else {
                        int half_row = off_n_1 + half_h_1 * 128 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m_1, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            int weights_pending_1 = pre_b * ((iter_k_1 < NUM_STAGES) ? 1 : 0);
                            if (weights_pending_1 == 0) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B_half), 0, half_row, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            int k_set_base_1 = iter_k_1 * 4;
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            if (weights_pending_1 == 0) {
                                tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB_half), 0, half_h_1 * 4, sfb_tile_row + k_set_base_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(28672)) : "memory");
                            load_stage += 1;
                            if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    }
                    pre_b = 0;
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
                unsigned int mma_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    int unit_2 = (int)(mma_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_2 = mma_bid;
                    int is_half_2 = 0;
                    int half_h_2 = 0;
                    if (unit_2 >= full_units) {
                        int half_q_2 = unit_2 - full_units;
                        tile_bid_2 = (unsigned int)((full_units + half_q_2 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_2 = half_q_2 % 2;
                        is_half_2 = 1;
                    }
                    if (is_half_2 == 0) {
                        #pragma unroll 1
                        for (int super_k = 0; super_k < 9; super_k++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0 = mma_tma_stage;
                            int init_flag = ((super_k == 0) ? 1 : 0);
                            uint32_t _elected_0 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1 | ((uint64_t)((uint32_t)_smem_desc_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_4 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2 | ((uint64_t)((uint32_t)_smem_desc_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
                            }
                            int _smem_desc_lo_5 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_6 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5 | ((uint64_t)((uint32_t)_smem_desc_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_8 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6 | ((uint64_t)((uint32_t)_smem_desc_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1 = mma_tma_stage;
                            uint32_t _elected_9 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            int _smem_desc_lo_10 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_11 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10 | ((uint64_t)((uint32_t)_smem_desc_lo_12 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_13 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11 | ((uint64_t)((uint32_t)_smem_desc_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_9);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0) * 8, (uint16_t)(3));
                            uint32_t _elected_14 = elect_sync();
                            int _smem_desc_lo_15 = (((smem_a_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_16 = (((smem_b_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_17 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15 | ((uint64_t)((uint32_t)_smem_desc_lo_17 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_18 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16 | ((uint64_t)((uint32_t)_smem_desc_lo_18 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1, _elected_14);
                            }
                            int _smem_desc_lo_19 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_20 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_21 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19 | ((uint64_t)((uint32_t)_smem_desc_lo_21 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_22 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20 | ((uint64_t)((uint32_t)_smem_desc_lo_22 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1, _elected_14);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2 = mma_tma_stage;
                            uint32_t _elected_23 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            int _smem_desc_lo_24 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_25 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_26 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_24 | ((uint64_t)((uint32_t)_smem_desc_lo_26 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_27 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25 | ((uint64_t)((uint32_t)_smem_desc_lo_27 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1, _elected_23);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1) * 8, (uint16_t)(3));
                            uint32_t _elected_28 = elect_sync();
                            int _smem_desc_lo_29 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_30 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_31 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_29 | ((uint64_t)((uint32_t)_smem_desc_lo_31 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_32 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_30 | ((uint64_t)((uint32_t)_smem_desc_lo_32 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1, _elected_28);
                            }
                            int _smem_desc_lo_33 = (((smem_a_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_34 = (((smem_b_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_35 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_33 | ((uint64_t)((uint32_t)_smem_desc_lo_35 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_36 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_34 | ((uint64_t)((uint32_t)_smem_desc_lo_36 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1, _elected_28);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p2) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_37 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        int _smem_desc_lo_38 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_39 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_38) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_39) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_37);
                        }
                        int _smem_desc_lo_40 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_41 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_40) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_41) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_37);
                        }
                        int _smem_desc_lo_42 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_43 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_42) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_43) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_37);
                        }
                        int _smem_desc_lo_44 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_45 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_44) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_45) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_37);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    } else {
                        #pragma unroll 1
                        for (int super_k_1 = 0; super_k_1 < 9; super_k_1++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0_1 = mma_tma_stage;
                            int init_flag_1 = ((super_k_1 == 0) ? 1 : 0);
                            uint32_t _elected_46 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 4, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 8, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 12, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            int _smem_desc_lo_47 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_48 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_49 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_47 | ((uint64_t)((uint32_t)_smem_desc_lo_49 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_50 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_48 | ((uint64_t)((uint32_t)_smem_desc_lo_50 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 0, tmem_tmem_sfb_h + 0, ((init_flag_1) ? 0 : 1), _elected_46);
                            }
                            int _smem_desc_lo_51 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_52 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_53 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_51 | ((uint64_t)((uint32_t)_smem_desc_lo_53 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_54 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_52 | ((uint64_t)((uint32_t)_smem_desc_lo_54 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb_h + 4 + 0, 1, _elected_46);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1_1 = mma_tma_stage;
                            uint32_t _elected_55 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 16, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 20, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 24, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 28, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            int _smem_desc_lo_56 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_57 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_58 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_56 | ((uint64_t)((uint32_t)_smem_desc_lo_58 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_59 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_57 | ((uint64_t)((uint32_t)_smem_desc_lo_59 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb_h + 12 + 0, 1, _elected_55);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0_1) * 8, (uint16_t)(3));
                            uint32_t _elected_60 = elect_sync();
                            int _smem_desc_lo_61 = (((smem_a_addr + 16) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_62 = (((smem_b_addr + 16) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_63 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_61 | ((uint64_t)((uint32_t)_smem_desc_lo_63 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_64 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_62 | ((uint64_t)((uint32_t)_smem_desc_lo_64 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb_h + 16 + 0, 1, _elected_60);
                            }
                            int _smem_desc_lo_65 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_66 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_67 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_65 | ((uint64_t)((uint32_t)_smem_desc_lo_67 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_68 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_66 | ((uint64_t)((uint32_t)_smem_desc_lo_68 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb_h + 24 + 0, 1, _elected_60);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2_1 = mma_tma_stage;
                            uint32_t _elected_69 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 32, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 36, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 40, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 44, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            int _smem_desc_lo_70 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_71 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_72 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_70 | ((uint64_t)((uint32_t)_smem_desc_lo_72 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_73 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_71 | ((uint64_t)((uint32_t)_smem_desc_lo_73 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb_h + 28 + 0, 1, _elected_69);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1_1) * 8, (uint16_t)(3));
                            uint32_t _elected_74 = elect_sync();
                            int _smem_desc_lo_75 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            int _smem_desc_lo_76 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            {
                                int _smem_desc_lo_77 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_75 | ((uint64_t)((uint32_t)_smem_desc_lo_77 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_78 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_76 | ((uint64_t)((uint32_t)_smem_desc_lo_78 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb_h + 36 + 0, 1, _elected_74);
                            }
                            int _smem_desc_lo_79 = (((smem_a_addr + 80) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            int _smem_desc_lo_80 = (((smem_b_addr + 80) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            {
                                int _smem_desc_lo_81 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_79 | ((uint64_t)((uint32_t)_smem_desc_lo_81 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_82 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_80 | ((uint64_t)((uint32_t)_smem_desc_lo_82 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb_h + 40 + 0, 1, _elected_74);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p2_1) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_83 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 4, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 8, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 12, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        int _smem_desc_lo_84 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_85 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_84) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_85) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb_h + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_86 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_87 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_86) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_87) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb_h + 4 + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_88 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_89 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_88) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_89) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb_h + 8 + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_90 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_91 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_90) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_91) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb_h + 12 + 0, 1, _elected_83);
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
                    mma_bid = _clc_ctaid_1 + (unsigned int)cta_rank;
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
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_2++) {
                int unit_3 = (int)(this_bid_1 / (unsigned int)CTA_GROUP);
                unsigned int tile_bid_3 = this_bid_1;
                int is_half_3 = 0;
                int half_h_3 = 0;
                if (unit_3 >= full_units) {
                    int half_q_3 = unit_3 - full_units;
                    tile_bid_3 = (unsigned int)((full_units + half_q_3 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h_3 = half_q_3 % 2;
                    is_half_3 = 1;
                }
                int group_2 = tile_bid_3 / (unsigned int)tiles_per_group;
                int first_m_2 = group_2 * GROUP_M;
                int remaining_2 = m_tiles - first_m_2;
                int group_size_2 = ((remaining_2 >= GROUP_M) ? GROUP_M : remaining_2);
                int local_2 = tile_bid_3 % (unsigned int)tiles_per_group;
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int off_m_2 = bid_m_2 * BLOCK_M;
                int off_n_2 = bid_n_2 * BLOCK_N;
                if (is_half_3 == 0) {
                    int global_row = off_m_2 + local_row;
                    int col0 = col_part * 128;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                    int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                    long long table_row = gate_index[load_row];
                    int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul = (float)in_range;
                    unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                    unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_2 + (unsigned long long)col0;
                    unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_2 + (unsigned long long)col0;
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
                    if (global_row < M) {
                        #pragma unroll
                        for (int n_chunk = 0; n_chunk < 8; n_chunk++) {
                            int col = n_chunk * 16;
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                                uint4 _vld_0[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_0[_blk] = _vptr_0[_blk];
                                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
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
                            float _vec_load_2[16];
                            {
                                const void* _v8p_2 = (const void*)(residual + (row_base + (unsigned long long)col) + (0));
                                uint32_t _v8_2_0[8];
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_v8_2_0[0]), "=r"(_v8_2_0[1]), "=r"(_v8_2_0[2]), "=r"(_v8_2_0[3]), "=r"(_v8_2_0[4]), "=r"(_v8_2_0[5]), "=r"(_v8_2_0[6]), "=r"(_v8_2_0[7]) : "l"((const char*)_v8p_2 + 0) : "memory");
                                (&_vec_load_2[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) << 16);
                                (&_vec_load_2[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[0]) & 0xffff0000u);
                                (&_vec_load_2[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) << 16);
                                (&_vec_load_2[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[1]) & 0xffff0000u);
                                (&_vec_load_2[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) << 16);
                                (&_vec_load_2[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[2]) & 0xffff0000u);
                                (&_vec_load_2[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) << 16);
                                (&_vec_load_2[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[3]) & 0xffff0000u);
                                (&_vec_load_2[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) << 16);
                                (&_vec_load_2[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[4]) & 0xffff0000u);
                                (&_vec_load_2[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) << 16);
                                (&_vec_load_2[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[5]) & 0xffff0000u);
                                (&_vec_load_2[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) << 16);
                                (&_vec_load_2[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[6]) & 0xffff0000u);
                                (&_vec_load_2[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) << 16);
                                (&_vec_load_2[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_2_0[7]) & 0xffff0000u);
                            }
                            float out_vals[16];
                            #pragma unroll
                            for (int j = 0; j < 8; j++) {
                                float o_lo = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j] * alpha_v);
                                float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
                                out_vals[j] = _vec_load_2[j] + p_lo;
                                float o_hi = (float)(__nv_bfloat16)(_tmem_load_0[n_chunk * 16 + j + 8] * alpha_v);
                                float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
                                out_vals[j + 8] = _vec_load_2[j + 8] + p_hi;
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
                                const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                            }
                        }
                    }
                } else {
                    int global_row_1 = off_m_2 + local_row;
                    int col0_1 = col_part * 64;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0_1;
                    int load_row_1 = ((global_row_1) < (M - 1) ? (global_row_1) : (M - 1));
                    long long table_row_1 = gate_index[load_row_1];
                    int in_range_1 = (((unsigned long long)table_row_1 < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul_1 = (float)in_range_1;
                    unsigned int safe_row_1 = (unsigned int)table_row_1 * (unsigned int)in_range_1;
                    unsigned long long gate_base_1 = (unsigned long long)safe_row_1 * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)(off_n_2 + half_h_3 * 128) + (unsigned long long)col0_1;
                    unsigned long long row_base_1 = (unsigned long long)global_row_1 * (unsigned long long)HIDDEN + (unsigned long long)(off_n_2 + half_h_3 * 128) + (unsigned long long)col0_1;
                    mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(lane_addr_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                    }
                    _phase_mainloop_done ^= 1;
                    if (global_row_1 < M) {
                        #pragma unroll
                        for (int n_chunk_1 = 0; n_chunk_1 < 4; n_chunk_1++) {
                            int col_1 = n_chunk_1 * 16;
                            float _vec_load_3[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_1 + 8) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_5[16];
                            {
                                const void* _v8p_6 = (const void*)(residual + (row_base_1 + (unsigned long long)col_1) + (0));
                                uint32_t _v8_6_0[8];
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_v8_6_0[0]), "=r"(_v8_6_0[1]), "=r"(_v8_6_0[2]), "=r"(_v8_6_0[3]), "=r"(_v8_6_0[4]), "=r"(_v8_6_0[5]), "=r"(_v8_6_0[6]), "=r"(_v8_6_0[7]) : "l"((const char*)_v8p_6 + 0) : "memory");
                                (&_vec_load_5[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[0]) << 16);
                                (&_vec_load_5[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[0]) & 0xffff0000u);
                                (&_vec_load_5[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[1]) << 16);
                                (&_vec_load_5[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[1]) & 0xffff0000u);
                                (&_vec_load_5[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[2]) << 16);
                                (&_vec_load_5[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[2]) & 0xffff0000u);
                                (&_vec_load_5[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[3]) << 16);
                                (&_vec_load_5[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[3]) & 0xffff0000u);
                                (&_vec_load_5[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[4]) << 16);
                                (&_vec_load_5[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[4]) & 0xffff0000u);
                                (&_vec_load_5[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[5]) << 16);
                                (&_vec_load_5[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[5]) & 0xffff0000u);
                                (&_vec_load_5[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[6]) << 16);
                                (&_vec_load_5[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[6]) & 0xffff0000u);
                                (&_vec_load_5[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[7]) << 16);
                                (&_vec_load_5[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_6_0[7]) & 0xffff0000u);
                            }
                            float out_vals_1[16];
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 8; j_1++) {
                                float o_lo_1 = (float)(__nv_bfloat16)(_tmem_load_1[n_chunk_1 * 16 + j_1] * alpha_v);
                                float p_lo_1 = (float)(__nv_bfloat16)(_vec_load_3[j_1] * gate_mul_1 * o_lo_1);
                                out_vals_1[j_1] = _vec_load_5[j_1] + p_lo_1;
                                float o_hi_1 = (float)(__nv_bfloat16)(_tmem_load_1[n_chunk_1 * 16 + j_1 + 8] * alpha_v);
                                float p_hi_1 = (float)(__nv_bfloat16)(_vec_load_4[j_1] * gate_mul_1 * o_hi_1);
                                out_vals_1[j_1 + 8] = _vec_load_5[j_1 + 8] + p_hi_1;
                            }
                            unsigned int _packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 0], out_vals_1[_lp*2+1 + 0]));
                                _packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 2], out_vals_1[_lp*2+1 + 2]));
                                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 4], out_vals_1[_lp*2+1 + 4]));
                                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 6], out_vals_1[_lp*2+1 + 6]));
                                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 8], out_vals_1[_lp*2+1 + 8]));
                                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 10], out_vals_1[_lp*2+1 + 10]));
                                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 12], out_vals_1[_lp*2+1 + 12]));
                                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 14], out_vals_1[_lp*2+1 + 14]));
                                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_7 = reinterpret_cast<const unsigned*>(_packed_1);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_1))), "r"(_raw_stv8_7[0]), "r"(_raw_stv8_7[1]), "r"(_raw_stv8_7[2]), "r"(_raw_stv8_7[3]), "r"(_raw_stv8_7[4]), "r"(_raw_stv8_7[5]), "r"(_raw_stv8_7[6]), "r"(_raw_stv8_7[7]) : "memory");
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
#endif
}

} // extern "C"
#endif  // MINIMAX_H3_SM_MINOR == 3

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
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
#undef SMEM_SMEM_V19_OFF
#undef SMEM_SMEM_V19_STAGE_BYTES
#undef SMEM_SMEM_V19_STRIDE
#undef SMEM_SMEM_V20_OFF
#undef SMEM_SMEM_V20_STAGE_BYTES
#undef SMEM_SMEM_V20_STRIDE
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
#undef TMEM_TMEM_SFB_H_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef WORK_STAGES
#undef tiles_per_group

#define MINIMAX_H3_OUT_PROJ_INF CUDART_INF_F
#define TMEM_NCOLS 448
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 304
#define TMEM_TMEM_SFB_H_OFFSET 400
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
#define SMEM_WORK_RESPONSE_OFF 228352
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_RES_STAGE_OFF 195584
#define SMEM_RES_STAGE_STAGE_BYTES 32768
#define SMEM_RES_STAGE_STRIDE 32768
#define SMEM_SMEM_V14_OFF 35840
#define SMEM_SMEM_V14_STAGE_BYTES 512
#define SMEM_SMEM_V14_STRIDE 38912
#define SMEM_SMEM_V15_OFF 36352
#define SMEM_SMEM_V15_STAGE_BYTES 512
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 36864
#define SMEM_SMEM_V16_STAGE_BYTES 512
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_V17_OFF 37376
#define SMEM_SMEM_V17_STAGE_BYTES 512
#define SMEM_SMEM_V17_STRIDE 38912
#define SMEM_SMEM_V18_OFF 1120
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 38912
#define SMEM_SMEM_V19_OFF 1136
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 38912
#define SMEM_SMEM_V20_OFF 17504
#define SMEM_SMEM_V20_STAGE_BYTES 6144
#define SMEM_SMEM_V20_STRIDE 38912
#define SMEM_SMEM_V21_OFF 17520
#define SMEM_SMEM_V21_STAGE_BYTES 6144
#define SMEM_SMEM_V21_STRIDE 38912
#define SMEM_TOTAL 228480
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 8
#define WORK_STAGES 4
#define NUM_K_ITERS 28
#define N_TILES 21
#define SF_K_TILES 112
#define HIDDEN 5376
#define tiles_per_group (GROUP_M * N_TILES)

#if MINIMAX_H3_SM_MINOR == 3  // sm_103a only
extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_minimax_h3_out_proj_e2m1_ts_wr_pw_sr(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ gate, long long* __restrict__ gate_index, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ out, int M, int m_tiles, int gate_rows, long long gate_row_stride, const __grid_constant__ CUtensorMap B_half, const __grid_constant__ CUtensorMap SFB_half, int full_units)
{
#if __CUDA_ARCH__ == 1030
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
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int work_response_addr = smem + 228352;
    __nv_bfloat16* res_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 195584);
    const int res_stage_addr = smem + 195584;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v14_addr = smem + 35840;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 36352);
    const int smem_v15_addr = smem + 36352;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v16_addr = smem + 36864;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 37376);
    const int smem_v17_addr = smem + 37376;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v18_addr = smem + 1120;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v19_addr = smem + 1136;
    uint8_t* smem_v20 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v20_addr = smem + 17504;
    uint8_t* smem_v21 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v21_addr = smem + 17520;

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

    // TMEM alloc (512 columns, 448 used)
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
    const int tmem_tmem_sfb = taddr + 304;
    const int tmem_tmem_sfb_h = taddr + 400;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int pre_b = 0;
            if (elect_sync()) {
                unsigned int first_bid = bid;
                int unit = (int)(first_bid / (unsigned int)CTA_GROUP);
                unsigned int tile_bid = first_bid;
                int is_half = 0;
                int half_h = 0;
                if (unit >= full_units) {
                    int half_q = unit - full_units;
                    tile_bid = (unsigned int)((full_units + half_q / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h = half_q % 2;
                    is_half = 1;
                }
                int group = tile_bid / (unsigned int)tiles_per_group;
                int first_m = group * GROUP_M;
                int remaining = m_tiles - first_m;
                int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                int local = tile_bid % (unsigned int)tiles_per_group;
                int bid_m = first_m + local % group_size;
                int bid_n = local / group_size;
                int off_m = bid_m * BLOCK_M;
                int off_n = bid_n * BLOCK_N;
                if (is_half == 0) {
                    pre_b = 1;
                    int first_weight_row = off_n + cta_rank * B_HALF_N;
                    int first_sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll
                    for (int stage = 0; stage < NUM_STAGES; stage++) {
                        tma_3d_gmem2smem_cta2(smem_b_addr + (unsigned int)(stage * 38912), (&B), 0, first_weight_row, stage, ((tma_full_addr + (stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + (unsigned int)(stage * 38912), (&SFB), 0, 0, first_sfb_tile_row + stage * 4, ((tma_full_addr + (stage) * 8) & 0xFEFFFFFF));
                    }
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter++) {
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
                    int unit_1 = (int)(this_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_1 = this_bid;
                    int is_half_1 = 0;
                    int half_h_1 = 0;
                    if (unit_1 >= full_units) {
                        int half_q_1 = unit_1 - full_units;
                        tile_bid_1 = (unsigned int)((full_units + half_q_1 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_1 = half_q_1 % 2;
                        is_half_1 = 1;
                    }
                    int group_1 = tile_bid_1 / (unsigned int)tiles_per_group;
                    int first_m_1 = group_1 * GROUP_M;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                    int local_1 = tile_bid_1 % (unsigned int)tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * BLOCK_M;
                    int off_n_1 = bid_n_1 * BLOCK_N;
                    int weight_row = off_n_1 + cta_rank * B_HALF_N;
                    int sfa_tile_row = bid_m_1 * SF_K_TILES;
                    int sfb_tile_row = bid_n_1 * SF_K_TILES;
                    if (is_half_1 == 0) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m_1, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            int weights_pending = pre_b * ((iter_k < NUM_STAGES) ? 1 : 0);
                            if (weights_pending == 0) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B), 0, weight_row, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            int k_set_base = iter_k * 4;
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            if (weights_pending == 0) {
                                tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB), 0, 0, sfb_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(38912)) : "memory");
                            load_stage += 1;
                            if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    } else {
                        int half_row = off_n_1 + half_h_1 * 128 + cta_rank * 64;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m_1, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            int weights_pending_1 = pre_b * ((iter_k_1 < NUM_STAGES) ? 1 : 0);
                            if (weights_pending_1 == 0) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B_half), 0, half_row, iter_k_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            int k_set_base_1 = iter_k_1 * 4;
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            if (weights_pending_1 == 0) {
                                tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB_half), 0, half_h_1 * 4, sfb_tile_row + k_set_base_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(28672)) : "memory");
                            load_stage += 1;
                            if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                    }
                    pre_b = 0;
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
                unsigned int mma_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    int unit_2 = (int)(mma_bid / (unsigned int)CTA_GROUP);
                    unsigned int tile_bid_2 = mma_bid;
                    int is_half_2 = 0;
                    int half_h_2 = 0;
                    if (unit_2 >= full_units) {
                        int half_q_2 = unit_2 - full_units;
                        tile_bid_2 = (unsigned int)((full_units + half_q_2 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                        half_h_2 = half_q_2 % 2;
                        is_half_2 = 1;
                    }
                    if (is_half_2 == 0) {
                        #pragma unroll 1
                        for (int super_k = 0; super_k < 9; super_k++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0 = mma_tma_stage;
                            int init_flag = ((super_k == 0) ? 1 : 0);
                            uint32_t _elected_0 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432)), _elected_0);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432 + 32)), _elected_0);
                            int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1 | ((uint64_t)((uint32_t)_smem_desc_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_4 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2 | ((uint64_t)((uint32_t)_smem_desc_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_0);
                            }
                            int _smem_desc_lo_5 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_6 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5 | ((uint64_t)((uint32_t)_smem_desc_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_8 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6 | ((uint64_t)((uint32_t)_smem_desc_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_0);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1 = mma_tma_stage;
                            uint32_t _elected_9 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            int _smem_desc_lo_10 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_11 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10 | ((uint64_t)((uint32_t)_smem_desc_lo_12 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_13 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11 | ((uint64_t)((uint32_t)_smem_desc_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_9);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0) * 8, (uint16_t)(3));
                            uint32_t _elected_14 = elect_sync();
                            int _smem_desc_lo_15 = (((smem_a_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_16 = (((smem_b_addr + 16) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_17 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15 | ((uint64_t)((uint32_t)_smem_desc_lo_17 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_18 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16 | ((uint64_t)((uint32_t)_smem_desc_lo_18 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1, _elected_14);
                            }
                            int _smem_desc_lo_19 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_20 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_21 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19 | ((uint64_t)((uint32_t)_smem_desc_lo_21 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_22 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20 | ((uint64_t)((uint32_t)_smem_desc_lo_22 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1, _elected_14);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2 = mma_tma_stage;
                            uint32_t _elected_23 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            int _smem_desc_lo_24 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_25 = (((smem_v21_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_26 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_24 | ((uint64_t)((uint32_t)_smem_desc_lo_26 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_27 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25 | ((uint64_t)((uint32_t)_smem_desc_lo_27 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1, _elected_23);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1) * 8, (uint16_t)(3));
                            uint32_t _elected_28 = elect_sync();
                            int _smem_desc_lo_29 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_30 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_31 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_29 | ((uint64_t)((uint32_t)_smem_desc_lo_31 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_32 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_30 | ((uint64_t)((uint32_t)_smem_desc_lo_32 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1, _elected_28);
                            }
                            int _smem_desc_lo_33 = (((smem_a_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_34 = (((smem_b_addr + 80) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_35 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_33 | ((uint64_t)((uint32_t)_smem_desc_lo_35 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_36 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_34 | ((uint64_t)((uint32_t)_smem_desc_lo_36 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1, _elected_28);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p2) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_37 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_37);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_37);
                        int _smem_desc_lo_38 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_39 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_38) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_39) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_37);
                        }
                        int _smem_desc_lo_40 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_41 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_40) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_41) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_37);
                        }
                        int _smem_desc_lo_42 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_43 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_42) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_43) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 16 + 0, 1, _elected_37);
                        }
                        int _smem_desc_lo_44 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_45 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_44) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_45) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_37);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    } else {
                        #pragma unroll 1
                        for (int super_k_1 = 0; super_k_1 < 9; super_k_1++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0_1 = mma_tma_stage;
                            int init_flag_1 = ((super_k_1 == 0) ? 1 : 0);
                            uint32_t _elected_46 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 4, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 8, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 12, make_sf_cp_desc_lo_sbo128((((smem_v17_addr) >> 4) + (stage_p0_1) * 2432)), _elected_46);
                            int _smem_desc_lo_47 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_48 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_49 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_47 | ((uint64_t)((uint32_t)_smem_desc_lo_49 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_50 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_48 | ((uint64_t)((uint32_t)_smem_desc_lo_50 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 0, tmem_tmem_sfb_h + 0, ((init_flag_1) ? 0 : 1), _elected_46);
                            }
                            int _smem_desc_lo_51 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_52 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_53 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_51 | ((uint64_t)((uint32_t)_smem_desc_lo_53 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_54 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_52 | ((uint64_t)((uint32_t)_smem_desc_lo_54 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb_h + 4 + 0, 1, _elected_46);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1_1 = mma_tma_stage;
                            uint32_t _elected_55 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 16, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 20, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 24, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 28, make_sf_cp_desc_lo_sbo128((((smem_v17_addr) >> 4) + (stage_p1_1) * 2432)), _elected_55);
                            int _smem_desc_lo_56 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            int _smem_desc_lo_57 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p0_1) * 2432;
                            {
                                int _smem_desc_lo_58 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_56 | ((uint64_t)((uint32_t)_smem_desc_lo_58 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_59 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_57 | ((uint64_t)((uint32_t)_smem_desc_lo_59 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb_h + 12 + 0, 1, _elected_55);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0_1) * 8, (uint16_t)(3));
                            uint32_t _elected_60 = elect_sync();
                            int _smem_desc_lo_61 = (((smem_a_addr + 16) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_62 = (((smem_b_addr + 16) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_63 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_61 | ((uint64_t)((uint32_t)_smem_desc_lo_63 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_64 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_62 | ((uint64_t)((uint32_t)_smem_desc_lo_64 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb_h + 16 + 0, 1, _elected_60);
                            }
                            int _smem_desc_lo_65 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_66 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_67 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_65 | ((uint64_t)((uint32_t)_smem_desc_lo_67 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_68 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_66 | ((uint64_t)((uint32_t)_smem_desc_lo_68 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb_h + 24 + 0, 1, _elected_60);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2_1 = mma_tma_stage;
                            uint32_t _elected_69 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 32, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 36, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 40, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 44, make_sf_cp_desc_lo_sbo128((((smem_v17_addr) >> 4) + (stage_p2_1) * 2432)), _elected_69);
                            int _smem_desc_lo_70 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            int _smem_desc_lo_71 = (((smem_v21_addr) >> 4) & 0x3FFF) + (stage_p1_1) * 2432;
                            {
                                int _smem_desc_lo_72 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_70 | ((uint64_t)((uint32_t)_smem_desc_lo_72 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_73 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_71 | ((uint64_t)((uint32_t)_smem_desc_lo_73 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb_h + 28 + 0, 1, _elected_69);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1_1) * 8, (uint16_t)(3));
                            uint32_t _elected_74 = elect_sync();
                            int _smem_desc_lo_75 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            int _smem_desc_lo_76 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            {
                                int _smem_desc_lo_77 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_75 | ((uint64_t)((uint32_t)_smem_desc_lo_77 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_78 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_76 | ((uint64_t)((uint32_t)_smem_desc_lo_78 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0x90400480U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb_h + 36 + 0, 1, _elected_74);
                            }
                            int _smem_desc_lo_79 = (((smem_a_addr + 80) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            int _smem_desc_lo_80 = (((smem_b_addr + 80) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                            {
                                int _smem_desc_lo_81 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_79 | ((uint64_t)((uint32_t)_smem_desc_lo_81 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_82 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_p2_1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_80 | ((uint64_t)((uint32_t)_smem_desc_lo_82 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    ((0xd04004a0U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(128) >> 3) << 17)), tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb_h + 40 + 0, 1, _elected_74);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p2_1) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_83 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 4, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 8, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb_h + 12, make_sf_cp_desc_lo_sbo128((((smem_v17_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_83);
                        int _smem_desc_lo_84 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_85 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_84) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_85) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb_h + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_86 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_87 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_86) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_87) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb_h + 4 + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_88 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_89 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_88) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_89) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb_h + 8 + 0, 1, _elected_83);
                        }
                        int _smem_desc_lo_90 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        int _smem_desc_lo_91 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_tma_stage) * 2432;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_90) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_91) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb_h + 12 + 0, 1, _elected_83);
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
                    mma_bid = _clc_ctaid_1 + (unsigned int)cta_rank;
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
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < 2 * (m_tiles / CTA_GROUP * N_TILES) - full_units; _tile_iter_2++) {
                int unit_3 = (int)(this_bid_1 / (unsigned int)CTA_GROUP);
                unsigned int tile_bid_3 = this_bid_1;
                int is_half_3 = 0;
                int half_h_3 = 0;
                if (unit_3 >= full_units) {
                    int half_q_3 = unit_3 - full_units;
                    tile_bid_3 = (unsigned int)((full_units + half_q_3 / 2) * CTA_GROUP) + (unsigned int)cta_rank;
                    half_h_3 = half_q_3 % 2;
                    is_half_3 = 1;
                }
                int group_2 = tile_bid_3 / (unsigned int)tiles_per_group;
                int first_m_2 = group_2 * GROUP_M;
                int remaining_2 = m_tiles - first_m_2;
                int group_size_2 = ((remaining_2 >= GROUP_M) ? GROUP_M : remaining_2);
                int local_2 = tile_bid_3 % (unsigned int)tiles_per_group;
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int off_m_2 = bid_m_2 * BLOCK_M;
                int off_n_2 = bid_n_2 * BLOCK_N;
                if (is_half_3 == 0) {
                    int global_row = off_m_2 + local_row;
                    int col0 = col_part * 128;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0;
                    int load_row = ((global_row) < (M - 1) ? (global_row) : (M - 1));
                    long long table_row = gate_index[load_row];
                    int in_range = (((unsigned long long)table_row < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul = (float)in_range;
                    unsigned int safe_row = (unsigned int)table_row * (unsigned int)in_range;
                    unsigned long long gate_base = (unsigned long long)safe_row * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)off_n_2 + (unsigned long long)col0;
                    unsigned long long row_base = (unsigned long long)global_row * (unsigned long long)HIDDEN + (unsigned long long)off_n_2 + (unsigned long long)col0;
                    int copy_row = global_row + lane / 4 - lane;
                    unsigned long long copy_src = row_base + (unsigned long long)(long long)((lane / 4 - lane) * HIDDEN + lane % 4 * 8);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src + (unsigned long long)0)), "r"((copy_row < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 8 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 8 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src + (unsigned long long)(8 * HIDDEN))), "r"((copy_row + 8 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 16 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 16 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src + (unsigned long long)(16 * HIDDEN))), "r"((copy_row + 16 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 24 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 24 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src + (unsigned long long)(24 * HIDDEN))), "r"((copy_row + 24 < M) ? 16 : 0));
                    asm volatile("cp.async.commit_group;");
                    int copy_row_0 = global_row + lane / 4 - lane;
                    unsigned long long copy_src_1 = row_base + (unsigned long long)(long long)((lane / 4 - lane) * HIDDEN + lane % 4 * 8 + 32);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1 + (unsigned long long)0)), "r"((copy_row_0 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 8 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 8 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1 + (unsigned long long)(8 * HIDDEN))), "r"((copy_row_0 + 8 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 16 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 16 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1 + (unsigned long long)(16 * HIDDEN))), "r"((copy_row_0 + 16 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 24 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 24 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1 + (unsigned long long)(24 * HIDDEN))), "r"((copy_row_0 + 24 < M) ? 16 : 0));
                    asm volatile("cp.async.commit_group;");
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
                    int srow = (col_part * 4 + epi_warp) * 64 + lane;
                    asm volatile("cp.async.wait_group 1;");
                    __syncwarp();
                    if (global_row < M) {
                        int col = 0;
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col + 8) + 0);
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
                        unsigned int r_words[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 ^ (srow * 64 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 + 16 ^ (srow * 64 + 16 >> 7 & 3) << 4))));
                        float _unpack_bf16_0[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words[_pair]) << 16);
                            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words[_pair]) & 0xffff0000u);
                        }
                        float out_vals[16];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            float o_lo = (float)(__nv_bfloat16)(_tmem_load_0[j] * alpha_v);
                            float p_lo = (float)(__nv_bfloat16)(_vec_load_0[j] * gate_mul * o_lo);
                            out_vals[j] = _unpack_bf16_0[j] + p_lo;
                            float o_hi = (float)(__nv_bfloat16)(_tmem_load_0[j + 8] * alpha_v);
                            float p_hi = (float)(__nv_bfloat16)(_vec_load_1[j] * gate_mul * o_hi);
                            out_vals[j + 8] = _unpack_bf16_0[j + 8] + p_hi;
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
                            const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                        }
                        int col_0 = 16;
                        float _vec_load_2[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0) + 0);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_3[8];
                        {
                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0 + 8) + 0);
                            uint4 _vld_4[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_4[_blk] = _vptr_4[_blk];
                                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_1[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 + 32 ^ (srow * 64 + 32 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 + 48 ^ (srow * 64 + 48 >> 7 & 3) << 4))));
                        float _unpack_bf16_1[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_1[_pair]) << 16);
                            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_1[_pair]) & 0xffff0000u);
                        }
                        float out_vals_2[16];
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            float o_lo_1 = (float)(__nv_bfloat16)(_tmem_load_0[16 + j_1] * alpha_v);
                            float p_lo_1 = (float)(__nv_bfloat16)(_vec_load_2[j_1] * gate_mul * o_lo_1);
                            out_vals_2[j_1] = _unpack_bf16_1[j_1] + p_lo_1;
                            float o_hi_1 = (float)(__nv_bfloat16)(_tmem_load_0[16 + j_1 + 8] * alpha_v);
                            float p_hi_1 = (float)(__nv_bfloat16)(_vec_load_3[j_1] * gate_mul * o_hi_1);
                            out_vals_2[j_1 + 8] = _unpack_bf16_1[j_1 + 8] + p_hi_1;
                        }
                        unsigned int _packed_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 0], out_vals_2[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 2], out_vals_2[_lp*2+1 + 2]));
                            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 4], out_vals_2[_lp*2+1 + 4]));
                            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 6], out_vals_2[_lp*2+1 + 6]));
                            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 8], out_vals_2[_lp*2+1 + 8]));
                            _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 10], out_vals_2[_lp*2+1 + 10]));
                            _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 12], out_vals_2[_lp*2+1 + 12]));
                            _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2[_lp*2 + 14], out_vals_2[_lp*2+1 + 14]));
                            _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(_packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col_0))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                        }
                    }
                    __syncwarp();
                    int copy_row_2 = global_row + lane % 8 - lane;
                    unsigned long long copy_src_3 = row_base + (unsigned long long)(long long)((lane % 8 - lane) * HIDDEN + lane / 8 * 8 + 64);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + lane % 8) * 64 + lane / 8 * 16 ^ (((col_part * 4 + epi_warp) * 64 + lane % 8) * 64 + lane / 8 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_3 + (unsigned long long)0)), "r"((copy_row_2 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 8 + lane % 8) * 64 + lane / 8 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 8 + lane % 8) * 64 + lane / 8 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_3 + (unsigned long long)(8 * HIDDEN))), "r"((copy_row_2 + 8 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 16 + lane % 8) * 64 + lane / 8 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 16 + lane % 8) * 64 + lane / 8 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_3 + (unsigned long long)(16 * HIDDEN))), "r"((copy_row_2 + 16 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 24 + lane % 8) * 64 + lane / 8 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 24 + lane % 8) * 64 + lane / 8 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_3 + (unsigned long long)(24 * HIDDEN))), "r"((copy_row_2 + 24 < M) ? 16 : 0));
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 1;");
                    __syncwarp();
                    if (global_row < M) {
                        int col_1 = 32;
                        float _vec_load_4[8];
                        {
                            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_1) + 0);
                            uint4 _vld_6[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_6[_blk] = _vptr_6[_blk];
                                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_5[8];
                        {
                            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_1 + 8) + 0);
                            uint4 _vld_7[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_7[_blk] = _vptr_7[_blk];
                                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_2[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 ^ ((srow + 32) * 64 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_2[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 + 16 ^ ((srow + 32) * 64 + 16 >> 7 & 3) << 4))));
                        float _unpack_bf16_2[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_2[_pair]) << 16);
                            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_2[_pair]) & 0xffff0000u);
                        }
                        float out_vals_1[16];
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 8; j_2++) {
                            float o_lo_2 = (float)(__nv_bfloat16)(_tmem_load_0[32 + j_2] * alpha_v);
                            float p_lo_2 = (float)(__nv_bfloat16)(_vec_load_4[j_2] * gate_mul * o_lo_2);
                            out_vals_1[j_2] = _unpack_bf16_2[j_2] + p_lo_2;
                            float o_hi_2 = (float)(__nv_bfloat16)(_tmem_load_0[32 + j_2 + 8] * alpha_v);
                            float p_hi_2 = (float)(__nv_bfloat16)(_vec_load_5[j_2] * gate_mul * o_hi_2);
                            out_vals_1[j_2 + 8] = _unpack_bf16_2[j_2 + 8] + p_hi_2;
                        }
                        unsigned int _packed_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 0], out_vals_1[_lp*2+1 + 0]));
                            _packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 2], out_vals_1[_lp*2+1 + 2]));
                            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 4], out_vals_1[_lp*2+1 + 4]));
                            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 6], out_vals_1[_lp*2+1 + 6]));
                            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 8], out_vals_1[_lp*2+1 + 8]));
                            _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 10], out_vals_1[_lp*2+1 + 10]));
                            _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 12], out_vals_1[_lp*2+1 + 12]));
                            _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_1[_lp*2 + 14], out_vals_1[_lp*2+1 + 14]));
                            _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_8 = reinterpret_cast<const unsigned*>(_packed_2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col_1))), "r"(_raw_stv8_8[0]), "r"(_raw_stv8_8[1]), "r"(_raw_stv8_8[2]), "r"(_raw_stv8_8[3]), "r"(_raw_stv8_8[4]), "r"(_raw_stv8_8[5]), "r"(_raw_stv8_8[6]), "r"(_raw_stv8_8[7]) : "memory");
                        }
                        int col_0_1 = 48;
                        float _vec_load_6[8];
                        {
                            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0_1) + 0);
                            uint4 _vld_9[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_9[_blk] = _vptr_9[_blk];
                                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_7[8];
                        {
                            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0_1 + 8) + 0);
                            uint4 _vld_10[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_10[_blk] = _vptr_10[_blk];
                                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_1_1[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 + 32 ^ ((srow + 32) * 64 + 32 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_1[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 + 48 ^ ((srow + 32) * 64 + 48 >> 7 & 3) << 4))));
                        float _unpack_bf16_3[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_1_1[_pair]) << 16);
                            (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_1_1[_pair]) & 0xffff0000u);
                        }
                        float out_vals_2_1[16];
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 8; j_3++) {
                            float o_lo_3 = (float)(__nv_bfloat16)(_tmem_load_0[48 + j_3] * alpha_v);
                            float p_lo_3 = (float)(__nv_bfloat16)(_vec_load_6[j_3] * gate_mul * o_lo_3);
                            out_vals_2_1[j_3] = _unpack_bf16_3[j_3] + p_lo_3;
                            float o_hi_3 = (float)(__nv_bfloat16)(_tmem_load_0[48 + j_3 + 8] * alpha_v);
                            float p_hi_3 = (float)(__nv_bfloat16)(_vec_load_7[j_3] * gate_mul * o_hi_3);
                            out_vals_2_1[j_3 + 8] = _unpack_bf16_3[j_3 + 8] + p_hi_3;
                        }
                        unsigned int _packed_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 0], out_vals_2_1[_lp*2+1 + 0]));
                            _packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 2], out_vals_2_1[_lp*2+1 + 2]));
                            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 4], out_vals_2_1[_lp*2+1 + 4]));
                            _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 6], out_vals_2_1[_lp*2+1 + 6]));
                            _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 8], out_vals_2_1[_lp*2+1 + 8]));
                            _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 10], out_vals_2_1[_lp*2+1 + 10]));
                            _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 12], out_vals_2_1[_lp*2+1 + 12]));
                            _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_1[_lp*2 + 14], out_vals_2_1[_lp*2+1 + 14]));
                            _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_11 = reinterpret_cast<const unsigned*>(_packed_3);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col_0_1))), "r"(_raw_stv8_11[0]), "r"(_raw_stv8_11[1]), "r"(_raw_stv8_11[2]), "r"(_raw_stv8_11[3]), "r"(_raw_stv8_11[4]), "r"(_raw_stv8_11[5]), "r"(_raw_stv8_11[6]), "r"(_raw_stv8_11[7]) : "memory");
                        }
                    }
                    __syncwarp();
                    int copy_row_4 = global_row + (lane / 4 ^ 4) - lane;
                    unsigned long long copy_src_5 = row_base + (unsigned long long)(long long)(((lane / 4 ^ 4) - lane) * HIDDEN + lane % 4 * 8 + 96);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_5 + (unsigned long long)0)), "r"((copy_row_4 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 8 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 8 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_5 + (unsigned long long)(8 * HIDDEN))), "r"((copy_row_4 + 8 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 16 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 16 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_5 + (unsigned long long)(16 * HIDDEN))), "r"((copy_row_4 + 16 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 24 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 24 + (lane / 4 ^ 4)) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_5 + (unsigned long long)(24 * HIDDEN))), "r"((copy_row_4 + 24 < M) ? 16 : 0));
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 1;");
                    __syncwarp();
                    if (global_row < M) {
                        int col_2 = 64;
                        float _vec_load_8[8];
                        {
                            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_2) + 0);
                            uint4 _vld_12[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_12[_blk] = _vptr_12[_blk];
                                uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_9[8];
                        {
                            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_2 + 8) + 0);
                            uint4 _vld_13[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_13[_blk] = _vptr_13[_blk];
                                uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_3[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 ^ (srow * 64 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_3[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 + 16 ^ (srow * 64 + 16 >> 7 & 3) << 4))));
                        float _unpack_bf16_4[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_3[_pair]) << 16);
                            (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_3[_pair]) & 0xffff0000u);
                        }
                        float out_vals_3[16];
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            float o_lo_4 = (float)(__nv_bfloat16)(_tmem_load_0[64 + j_4] * alpha_v);
                            float p_lo_4 = (float)(__nv_bfloat16)(_vec_load_8[j_4] * gate_mul * o_lo_4);
                            out_vals_3[j_4] = _unpack_bf16_4[j_4] + p_lo_4;
                            float o_hi_4 = (float)(__nv_bfloat16)(_tmem_load_0[64 + j_4 + 8] * alpha_v);
                            float p_hi_4 = (float)(__nv_bfloat16)(_vec_load_9[j_4] * gate_mul * o_hi_4);
                            out_vals_3[j_4 + 8] = _unpack_bf16_4[j_4 + 8] + p_hi_4;
                        }
                        unsigned int _packed_4[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 0], out_vals_3[_lp*2+1 + 0]));
                            _packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 2], out_vals_3[_lp*2+1 + 2]));
                            _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 4], out_vals_3[_lp*2+1 + 4]));
                            _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 6], out_vals_3[_lp*2+1 + 6]));
                            _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 8], out_vals_3[_lp*2+1 + 8]));
                            _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 10], out_vals_3[_lp*2+1 + 10]));
                            _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 12], out_vals_3[_lp*2+1 + 12]));
                            _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_3[_lp*2 + 14], out_vals_3[_lp*2+1 + 14]));
                            _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_14 = reinterpret_cast<const unsigned*>(_packed_4);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col_2))), "r"(_raw_stv8_14[0]), "r"(_raw_stv8_14[1]), "r"(_raw_stv8_14[2]), "r"(_raw_stv8_14[3]), "r"(_raw_stv8_14[4]), "r"(_raw_stv8_14[5]), "r"(_raw_stv8_14[6]), "r"(_raw_stv8_14[7]) : "memory");
                        }
                        int col_0_2 = 80;
                        float _vec_load_10[8];
                        {
                            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0_2) + 0);
                            uint4 _vld_15[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_15[_blk] = _vptr_15[_blk];
                                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_11[8];
                        {
                            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0_2 + 8) + 0);
                            uint4 _vld_16[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_16[_blk] = _vptr_16[_blk];
                                uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_1_2[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 + 32 ^ (srow * 64 + 32 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_2[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow * 64 + 48 ^ (srow * 64 + 48 >> 7 & 3) << 4))));
                        float _unpack_bf16_5[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_1_2[_pair]) << 16);
                            (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_1_2[_pair]) & 0xffff0000u);
                        }
                        float out_vals_2_2[16];
                        #pragma unroll
                        for (int j_5 = 0; j_5 < 8; j_5++) {
                            float o_lo_5 = (float)(__nv_bfloat16)(_tmem_load_0[80 + j_5] * alpha_v);
                            float p_lo_5 = (float)(__nv_bfloat16)(_vec_load_10[j_5] * gate_mul * o_lo_5);
                            out_vals_2_2[j_5] = _unpack_bf16_5[j_5] + p_lo_5;
                            float o_hi_5 = (float)(__nv_bfloat16)(_tmem_load_0[80 + j_5 + 8] * alpha_v);
                            float p_hi_5 = (float)(__nv_bfloat16)(_vec_load_11[j_5] * gate_mul * o_hi_5);
                            out_vals_2_2[j_5 + 8] = _unpack_bf16_5[j_5 + 8] + p_hi_5;
                        }
                        unsigned int _packed_5[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 0], out_vals_2_2[_lp*2+1 + 0]));
                            _packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 2], out_vals_2_2[_lp*2+1 + 2]));
                            _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 4], out_vals_2_2[_lp*2+1 + 4]));
                            _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 6], out_vals_2_2[_lp*2+1 + 6]));
                            _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 8], out_vals_2_2[_lp*2+1 + 8]));
                            _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 10], out_vals_2_2[_lp*2+1 + 10]));
                            _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 12], out_vals_2_2[_lp*2+1 + 12]));
                            _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_2[_lp*2 + 14], out_vals_2_2[_lp*2+1 + 14]));
                            _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_17 = reinterpret_cast<const unsigned*>(_packed_5);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col_0_2))), "r"(_raw_stv8_17[0]), "r"(_raw_stv8_17[1]), "r"(_raw_stv8_17[2]), "r"(_raw_stv8_17[3]), "r"(_raw_stv8_17[4]), "r"(_raw_stv8_17[5]), "r"(_raw_stv8_17[6]), "r"(_raw_stv8_17[7]) : "memory");
                        }
                    }
                    __syncwarp();
                    asm volatile("cp.async.wait_group 0;");
                    __syncwarp();
                    if (global_row < M) {
                        int col_3 = 96;
                        float _vec_load_12[8];
                        {
                            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_3) + 0);
                            uint4 _vld_18[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_18[_blk] = _vptr_18[_blk];
                                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_13[8];
                        {
                            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_3 + 8) + 0);
                            uint4 _vld_19[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_19[_blk] = _vptr_19[_blk];
                                uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_4[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 ^ ((srow + 32) * 64 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_4[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 + 16 ^ ((srow + 32) * 64 + 16 >> 7 & 3) << 4))));
                        float _unpack_bf16_6[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_4[_pair]) << 16);
                            (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_4[_pair]) & 0xffff0000u);
                        }
                        float out_vals_4[16];
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 8; j_6++) {
                            float o_lo_6 = (float)(__nv_bfloat16)(_tmem_load_0[96 + j_6] * alpha_v);
                            float p_lo_6 = (float)(__nv_bfloat16)(_vec_load_12[j_6] * gate_mul * o_lo_6);
                            out_vals_4[j_6] = _unpack_bf16_6[j_6] + p_lo_6;
                            float o_hi_6 = (float)(__nv_bfloat16)(_tmem_load_0[96 + j_6 + 8] * alpha_v);
                            float p_hi_6 = (float)(__nv_bfloat16)(_vec_load_13[j_6] * gate_mul * o_hi_6);
                            out_vals_4[j_6 + 8] = _unpack_bf16_6[j_6 + 8] + p_hi_6;
                        }
                        unsigned int _packed_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 0], out_vals_4[_lp*2+1 + 0]));
                            _packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 2], out_vals_4[_lp*2+1 + 2]));
                            _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 4], out_vals_4[_lp*2+1 + 4]));
                            _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 6], out_vals_4[_lp*2+1 + 6]));
                            _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 8], out_vals_4[_lp*2+1 + 8]));
                            _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 10], out_vals_4[_lp*2+1 + 10]));
                            _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 12], out_vals_4[_lp*2+1 + 12]));
                            _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_4[_lp*2 + 14], out_vals_4[_lp*2+1 + 14]));
                            _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_20 = reinterpret_cast<const unsigned*>(_packed_6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col_3))), "r"(_raw_stv8_20[0]), "r"(_raw_stv8_20[1]), "r"(_raw_stv8_20[2]), "r"(_raw_stv8_20[3]), "r"(_raw_stv8_20[4]), "r"(_raw_stv8_20[5]), "r"(_raw_stv8_20[6]), "r"(_raw_stv8_20[7]) : "memory");
                        }
                        int col_0_3 = 112;
                        float _vec_load_14[8];
                        {
                            const uint4* _vptr_21 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0_3) + 0);
                            uint4 _vld_21[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_21[_blk] = _vptr_21[_blk];
                                uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
                                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_15[8];
                        {
                            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(gate + (gate_base + (unsigned long long)col_0_3 + 8) + 0);
                            uint4 _vld_22[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_22[_blk] = _vptr_22[_blk];
                                uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
                                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_1_3[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 + 32 ^ ((srow + 32) * 64 + 32 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_3[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow + 32) * 64 + 48 ^ ((srow + 32) * 64 + 48 >> 7 & 3) << 4))));
                        float _unpack_bf16_7[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_1_3[_pair]) << 16);
                            (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_1_3[_pair]) & 0xffff0000u);
                        }
                        float out_vals_2_3[16];
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 8; j_7++) {
                            float o_lo_7 = (float)(__nv_bfloat16)(_tmem_load_0[112 + j_7] * alpha_v);
                            float p_lo_7 = (float)(__nv_bfloat16)(_vec_load_14[j_7] * gate_mul * o_lo_7);
                            out_vals_2_3[j_7] = _unpack_bf16_7[j_7] + p_lo_7;
                            float o_hi_7 = (float)(__nv_bfloat16)(_tmem_load_0[112 + j_7 + 8] * alpha_v);
                            float p_hi_7 = (float)(__nv_bfloat16)(_vec_load_15[j_7] * gate_mul * o_hi_7);
                            out_vals_2_3[j_7 + 8] = _unpack_bf16_7[j_7 + 8] + p_hi_7;
                        }
                        unsigned int _packed_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 0], out_vals_2_3[_lp*2+1 + 0]));
                            _packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 2], out_vals_2_3[_lp*2+1 + 2]));
                            _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 4], out_vals_2_3[_lp*2+1 + 4]));
                            _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 6], out_vals_2_3[_lp*2+1 + 6]));
                            _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 8], out_vals_2_3[_lp*2+1 + 8]));
                            _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 10], out_vals_2_3[_lp*2+1 + 10]));
                            _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 12], out_vals_2_3[_lp*2+1 + 12]));
                            _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_3[_lp*2 + 14], out_vals_2_3[_lp*2+1 + 14]));
                            _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_23 = reinterpret_cast<const unsigned*>(_packed_7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base + (unsigned long long)col_0_3))), "r"(_raw_stv8_23[0]), "r"(_raw_stv8_23[1]), "r"(_raw_stv8_23[2]), "r"(_raw_stv8_23[3]), "r"(_raw_stv8_23[4]), "r"(_raw_stv8_23[5]), "r"(_raw_stv8_23[6]), "r"(_raw_stv8_23[7]) : "memory");
                        }
                    }
                    __syncwarp();
                } else {
                    int global_row_1 = off_m_2 + local_row;
                    int col0_1 = col_part * 64;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)col0_1;
                    int load_row_1 = ((global_row_1) < (M - 1) ? (global_row_1) : (M - 1));
                    long long table_row_1 = gate_index[load_row_1];
                    int in_range_1 = (((unsigned long long)table_row_1 < (unsigned long long)(unsigned int)gate_rows) ? 1 : 0);
                    float gate_mul_1 = (float)in_range_1;
                    unsigned int safe_row_1 = (unsigned int)table_row_1 * (unsigned int)in_range_1;
                    unsigned long long gate_base_1 = (unsigned long long)safe_row_1 * (unsigned long long)(unsigned int)gate_row_stride + (unsigned long long)(off_n_2 + half_h_3 * 128) + (unsigned long long)col0_1;
                    unsigned long long row_base_1 = (unsigned long long)global_row_1 * (unsigned long long)HIDDEN + (unsigned long long)(off_n_2 + half_h_3 * 128) + (unsigned long long)col0_1;
                    int copy_row_1 = global_row_1 + lane / 4 - lane;
                    unsigned long long copy_src_2 = row_base_1 + (unsigned long long)(long long)((lane / 4 - lane) * HIDDEN + lane % 4 * 8);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_2 + (unsigned long long)0)), "r"((copy_row_1 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 8 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 8 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_2 + (unsigned long long)(8 * HIDDEN))), "r"((copy_row_1 + 8 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 16 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 16 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_2 + (unsigned long long)(16 * HIDDEN))), "r"((copy_row_1 + 16 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 24 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 24 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_2 + (unsigned long long)(24 * HIDDEN))), "r"((copy_row_1 + 24 < M) ? 16 : 0));
                    asm volatile("cp.async.commit_group;");
                    int copy_row_0_1 = global_row_1 + lane / 4 - lane;
                    unsigned long long copy_src_1_1 = row_base_1 + (unsigned long long)(long long)((lane / 4 - lane) * HIDDEN + lane % 4 * 8 + 32);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1_1 + (unsigned long long)0)), "r"((copy_row_0_1 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 8 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 8 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1_1 + (unsigned long long)(8 * HIDDEN))), "r"((copy_row_0_1 + 8 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 16 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 16 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1_1 + (unsigned long long)(16 * HIDDEN))), "r"((copy_row_0_1 + 16 < M) ? 16 : 0));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"((res_stage_addr + (unsigned int)(((col_part * 4 + epi_warp) * 64 + 32 + 24 + lane / 4) * 64 + lane % 4 * 16 ^ (((col_part * 4 + epi_warp) * 64 + 32 + 24 + lane / 4) * 64 + lane % 4 * 16 >> 7 & 3) << 4))), "l"(residual + (copy_src_1_1 + (unsigned long long)(24 * HIDDEN))), "r"((copy_row_0_1 + 24 < M) ? 16 : 0));
                    asm volatile("cp.async.commit_group;");
                    mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(lane_addr_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                    }
                    _phase_mainloop_done ^= 1;
                    int srow_1 = (col_part * 4 + epi_warp) * 64 + lane;
                    asm volatile("cp.async.wait_group 1;");
                    __syncwarp();
                    if (global_row_1 < M) {
                        int col_4 = 0;
                        float _vec_load_16[8];
                        {
                            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_4) + 0);
                            uint4 _vld_24[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_24[_blk] = _vptr_24[_blk];
                                uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_17[8];
                        {
                            const uint4* _vptr_25 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_4 + 8) + 0);
                            uint4 _vld_25[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_25[_blk] = _vptr_25[_blk];
                                uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
                                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_5[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow_1 * 64 ^ (srow_1 * 64 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_5[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow_1 * 64 + 16 ^ (srow_1 * 64 + 16 >> 7 & 3) << 4))));
                        float _unpack_bf16_8[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_5[_pair]) << 16);
                            (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_5[_pair]) & 0xffff0000u);
                        }
                        float out_vals_5[16];
                        #pragma unroll
                        for (int j_8 = 0; j_8 < 8; j_8++) {
                            float o_lo_8 = (float)(__nv_bfloat16)(_tmem_load_1[j_8] * alpha_v);
                            float p_lo_8 = (float)(__nv_bfloat16)(_vec_load_16[j_8] * gate_mul_1 * o_lo_8);
                            out_vals_5[j_8] = _unpack_bf16_8[j_8] + p_lo_8;
                            float o_hi_8 = (float)(__nv_bfloat16)(_tmem_load_1[j_8 + 8] * alpha_v);
                            float p_hi_8 = (float)(__nv_bfloat16)(_vec_load_17[j_8] * gate_mul_1 * o_hi_8);
                            out_vals_5[j_8 + 8] = _unpack_bf16_8[j_8 + 8] + p_hi_8;
                        }
                        unsigned int _packed_8[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 0], out_vals_5[_lp*2+1 + 0]));
                            _packed_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 2], out_vals_5[_lp*2+1 + 2]));
                            _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 4], out_vals_5[_lp*2+1 + 4]));
                            _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 6], out_vals_5[_lp*2+1 + 6]));
                            _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 8], out_vals_5[_lp*2+1 + 8]));
                            _packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 10], out_vals_5[_lp*2+1 + 10]));
                            _packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 12], out_vals_5[_lp*2+1 + 12]));
                            _packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_5[_lp*2 + 14], out_vals_5[_lp*2+1 + 14]));
                            _packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_26 = reinterpret_cast<const unsigned*>(_packed_8);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_4))), "r"(_raw_stv8_26[0]), "r"(_raw_stv8_26[1]), "r"(_raw_stv8_26[2]), "r"(_raw_stv8_26[3]), "r"(_raw_stv8_26[4]), "r"(_raw_stv8_26[5]), "r"(_raw_stv8_26[6]), "r"(_raw_stv8_26[7]) : "memory");
                        }
                        int col_0_4 = 16;
                        float _vec_load_18[8];
                        {
                            const uint4* _vptr_27 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_0_4) + 0);
                            uint4 _vld_27[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_27[_blk] = _vptr_27[_blk];
                                uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
                                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_19[8];
                        {
                            const uint4* _vptr_28 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_0_4 + 8) + 0);
                            uint4 _vld_28[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_28[_blk] = _vptr_28[_blk];
                                uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) << 16);
                                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_1_4[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow_1 * 64 + 32 ^ (srow_1 * 64 + 32 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_4[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)(srow_1 * 64 + 48 ^ (srow_1 * 64 + 48 >> 7 & 3) << 4))));
                        float _unpack_bf16_9[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_1_4[_pair]) << 16);
                            (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_1_4[_pair]) & 0xffff0000u);
                        }
                        float out_vals_2_4[16];
                        #pragma unroll
                        for (int j_9 = 0; j_9 < 8; j_9++) {
                            float o_lo_9 = (float)(__nv_bfloat16)(_tmem_load_1[16 + j_9] * alpha_v);
                            float p_lo_9 = (float)(__nv_bfloat16)(_vec_load_18[j_9] * gate_mul_1 * o_lo_9);
                            out_vals_2_4[j_9] = _unpack_bf16_9[j_9] + p_lo_9;
                            float o_hi_9 = (float)(__nv_bfloat16)(_tmem_load_1[16 + j_9 + 8] * alpha_v);
                            float p_hi_9 = (float)(__nv_bfloat16)(_vec_load_19[j_9] * gate_mul_1 * o_hi_9);
                            out_vals_2_4[j_9 + 8] = _unpack_bf16_9[j_9 + 8] + p_hi_9;
                        }
                        unsigned int _packed_9[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 0], out_vals_2_4[_lp*2+1 + 0]));
                            _packed_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 2], out_vals_2_4[_lp*2+1 + 2]));
                            _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 4], out_vals_2_4[_lp*2+1 + 4]));
                            _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 6], out_vals_2_4[_lp*2+1 + 6]));
                            _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 8], out_vals_2_4[_lp*2+1 + 8]));
                            _packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 10], out_vals_2_4[_lp*2+1 + 10]));
                            _packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 12], out_vals_2_4[_lp*2+1 + 12]));
                            _packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_4[_lp*2 + 14], out_vals_2_4[_lp*2+1 + 14]));
                            _packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_29 = reinterpret_cast<const unsigned*>(_packed_9);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_0_4))), "r"(_raw_stv8_29[0]), "r"(_raw_stv8_29[1]), "r"(_raw_stv8_29[2]), "r"(_raw_stv8_29[3]), "r"(_raw_stv8_29[4]), "r"(_raw_stv8_29[5]), "r"(_raw_stv8_29[6]), "r"(_raw_stv8_29[7]) : "memory");
                        }
                    }
                    __syncwarp();
                    asm volatile("cp.async.wait_group 0;");
                    __syncwarp();
                    if (global_row_1 < M) {
                        int col_5 = 32;
                        float _vec_load_20[8];
                        {
                            const uint4* _vptr_30 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_5) + 0);
                            uint4 _vld_30[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_30[_blk] = _vptr_30[_blk];
                                uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
                                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_21[8];
                        {
                            const uint4* _vptr_31 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_5 + 8) + 0);
                            uint4 _vld_31[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_31[_blk] = _vptr_31[_blk];
                                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
                                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_6[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow_1 + 32) * 64 ^ ((srow_1 + 32) * 64 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_6[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow_1 + 32) * 64 + 16 ^ ((srow_1 + 32) * 64 + 16 >> 7 & 3) << 4))));
                        float _unpack_bf16_10[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_6[_pair]) << 16);
                            (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_6[_pair]) & 0xffff0000u);
                        }
                        float out_vals_6[16];
                        #pragma unroll
                        for (int j_10 = 0; j_10 < 8; j_10++) {
                            float o_lo_10 = (float)(__nv_bfloat16)(_tmem_load_1[32 + j_10] * alpha_v);
                            float p_lo_10 = (float)(__nv_bfloat16)(_vec_load_20[j_10] * gate_mul_1 * o_lo_10);
                            out_vals_6[j_10] = _unpack_bf16_10[j_10] + p_lo_10;
                            float o_hi_10 = (float)(__nv_bfloat16)(_tmem_load_1[32 + j_10 + 8] * alpha_v);
                            float p_hi_10 = (float)(__nv_bfloat16)(_vec_load_21[j_10] * gate_mul_1 * o_hi_10);
                            out_vals_6[j_10 + 8] = _unpack_bf16_10[j_10 + 8] + p_hi_10;
                        }
                        unsigned int _packed_10[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 0], out_vals_6[_lp*2+1 + 0]));
                            _packed_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 2], out_vals_6[_lp*2+1 + 2]));
                            _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 4], out_vals_6[_lp*2+1 + 4]));
                            _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 6], out_vals_6[_lp*2+1 + 6]));
                            _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 8], out_vals_6[_lp*2+1 + 8]));
                            _packed_10[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 10], out_vals_6[_lp*2+1 + 10]));
                            _packed_10[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 12], out_vals_6[_lp*2+1 + 12]));
                            _packed_10[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_6[_lp*2 + 14], out_vals_6[_lp*2+1 + 14]));
                            _packed_10[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_32 = reinterpret_cast<const unsigned*>(_packed_10);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_5))), "r"(_raw_stv8_32[0]), "r"(_raw_stv8_32[1]), "r"(_raw_stv8_32[2]), "r"(_raw_stv8_32[3]), "r"(_raw_stv8_32[4]), "r"(_raw_stv8_32[5]), "r"(_raw_stv8_32[6]), "r"(_raw_stv8_32[7]) : "memory");
                        }
                        int col_0_5 = 48;
                        float _vec_load_22[8];
                        {
                            const uint4* _vptr_33 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_0_5) + 0);
                            uint4 _vld_33[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_33[_blk] = _vptr_33[_blk];
                                uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_22[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                                    (&_vec_load_22[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        float _vec_load_23[8];
                        {
                            const uint4* _vptr_34 = reinterpret_cast<const uint4*>(gate + (gate_base_1 + (unsigned long long)col_0_5 + 8) + 0);
                            uint4 _vld_34[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_34[_blk] = _vptr_34[_blk];
                                uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_23[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                                    (&_vec_load_23[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        unsigned int r_words_1_5[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[(0) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow_1 + 32) * 64 + 32 ^ ((srow_1 + 32) * 64 + 32 >> 7 & 3) << 4))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_words_1_5[(4) + 3]))
                            : "r"((res_stage_addr + (unsigned int)((srow_1 + 32) * 64 + 48 ^ ((srow_1 + 32) * 64 + 48 >> 7 & 3) << 4))));
                        float _unpack_bf16_11[16];
                        #pragma unroll
                        for (int _pair = 0; _pair < 8; _pair++) {
                            (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(r_words_1_5[_pair]) << 16);
                            (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(r_words_1_5[_pair]) & 0xffff0000u);
                        }
                        float out_vals_2_5[16];
                        #pragma unroll
                        for (int j_11 = 0; j_11 < 8; j_11++) {
                            float o_lo_11 = (float)(__nv_bfloat16)(_tmem_load_1[48 + j_11] * alpha_v);
                            float p_lo_11 = (float)(__nv_bfloat16)(_vec_load_22[j_11] * gate_mul_1 * o_lo_11);
                            out_vals_2_5[j_11] = _unpack_bf16_11[j_11] + p_lo_11;
                            float o_hi_11 = (float)(__nv_bfloat16)(_tmem_load_1[48 + j_11 + 8] * alpha_v);
                            float p_hi_11 = (float)(__nv_bfloat16)(_vec_load_23[j_11] * gate_mul_1 * o_hi_11);
                            out_vals_2_5[j_11 + 8] = _unpack_bf16_11[j_11 + 8] + p_hi_11;
                        }
                        unsigned int _packed_11[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 0], out_vals_2_5[_lp*2+1 + 0]));
                            _packed_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 2], out_vals_2_5[_lp*2+1 + 2]));
                            _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 4], out_vals_2_5[_lp*2+1 + 4]));
                            _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 6], out_vals_2_5[_lp*2+1 + 6]));
                            _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 8], out_vals_2_5[_lp*2+1 + 8]));
                            _packed_11[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 10], out_vals_2_5[_lp*2+1 + 10]));
                            _packed_11[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 12], out_vals_2_5[_lp*2+1 + 12]));
                            _packed_11[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_vals_2_5[_lp*2 + 14], out_vals_2_5[_lp*2+1 + 14]));
                            _packed_11[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_35 = reinterpret_cast<const unsigned*>(_packed_11);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(out + (row_base_1 + (unsigned long long)col_0_5))), "r"(_raw_stv8_35[0]), "r"(_raw_stv8_35[1]), "r"(_raw_stv8_35[2]), "r"(_raw_stv8_35[3]), "r"(_raw_stv8_35[4]), "r"(_raw_stv8_35[5]), "r"(_raw_stv8_35[6]), "r"(_raw_stv8_35[7]) : "memory");
                        }
                    }
                    __syncwarp();
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
#endif
}

} // extern "C"
#endif  // MINIMAX_H3_SM_MINOR == 3

#undef BLOCK_K
#undef BLOCK_M
#undef BLOCK_N
#undef B_HALF_N
#undef CTA_GROUP
#undef GROUP_M
#undef HIDDEN
#undef MINIMAX_H3_OUT_PROJ_INF
#undef NUM_K_ITERS
#undef NUM_MAINLOOP_PIPE_STAGES
#undef NUM_STAGES
#undef NUM_TMA_PIPE_STAGES
#undef NUM_WORK_PIPE_STAGES
#undef N_TILES
#undef SF_K_TILES
#undef SMEM_RES_STAGE_OFF
#undef SMEM_RES_STAGE_STAGE_BYTES
#undef SMEM_RES_STAGE_STRIDE
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
#undef SMEM_SMEM_V19_OFF
#undef SMEM_SMEM_V19_STAGE_BYTES
#undef SMEM_SMEM_V19_STRIDE
#undef SMEM_SMEM_V20_OFF
#undef SMEM_SMEM_V20_STAGE_BYTES
#undef SMEM_SMEM_V20_STRIDE
#undef SMEM_SMEM_V21_OFF
#undef SMEM_SMEM_V21_STAGE_BYTES
#undef SMEM_SMEM_V21_STRIDE
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
#undef TMEM_TMEM_SFB_H_OFFSET
#undef TMEM_TMEM_SFB_OFFSET
#undef WORK_STAGES
#undef tiles_per_group

#include <cuda_runtime.h>

#include <mutex>
#include <utility>
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

constexpr int64_t kHidden = 5376;        // N: o_weight rows, out columns
constexpr int64_t kAttnDim = 7168;     // K: 56 heads x 128
constexpr int64_t kNumHeads = 56;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kMaxRows = 16777216;
constexpr int64_t kMaxGateRows = 2147483647;  // gate_rows is an int kernel argument
constexpr int64_t kGateRowAlign = 8;          // elements: 16-byte rows for the 128-bit gate loads
constexpr int64_t kBlockM = 128;
constexpr int64_t kCtaGroup = 2;
constexpr int64_t kNTiles = 21;       // 256 output columns per CTA pair, all variants
constexpr int64_t kBf16KBlocks = 112;  // 64-element activation K blocks of the BF16 GEMM

// Quantization pass (MXFP8 / NVFP4): one warp per row, kQuantRowsPerCta rows per CTA.
constexpr int kQuantThreads = 128;
constexpr int kQuantRowsPerCta = 4;
constexpr int kQuantRowSplitMxfp8 = 1;  // warps per activation row
constexpr int kQuantRowSplitNvfp4 = 2;
constexpr int kQuantMxfp8Smem = 0;
constexpr int kQuantNvfp4Smem = 0;

// GEMMs (all variants): persistent cta_group::2, one 128-row x 256-column tile per CTA pair.
constexpr int kGemmBf16Threads = 320;
constexpr int kGemmQuantThreads = 320;
constexpr int kGemmBf16Smem = 230528;
constexpr int kGemmMxfp8Smem = 206976;
constexpr int kGemmNvfp4Smem = 195712;
constexpr int kGemmNvfp4SrSmem = 228480;  // kernel_minimax_h3_out_proj_e2m1_ts_wr_pw_sr: + 32 KiB of residual slots
constexpr unsigned int kClusterX = 2u;
constexpr unsigned int kClusterY = 1u;
constexpr unsigned int kClusterZ = 1u;
// Programmatic dependent launch (cudaLaunchAttributeProgrammaticStreamSerialization): the GEMM may
// start while the preceding launch on the stream is still running; the kernel itself waits
// (griddepcontrol.wait) before touching that launch's outputs.  Resolved from the kernel module.
constexpr bool kGemmBf16Pdl = false;
constexpr bool kGemmMxfp8Pdl = true;
constexpr bool kGemmNvfp4Pdl = true;
// The N-split tail program of each GEMM (kernel_minimax_h3_out_proj_{bf16,e4m3,e2m1}_ts) takes the same launch
// shape and dynamic shared memory as its whole-tile program.  A tail-split shape is routed to it up to max_waves
// waves of tiles over the co-resident CTA pairs, since the half-tile saving on the last wave shrinks with the wave
// count, and from min_rows rows on, since below them the one M tile holds a handful of rows; every other shape
// keeps the whole-tile program.  Both limits are measured per variant and target:
//   BF16:  faster up to 10 waves on sm_103a and 9 waves on sm_100a (not at about 10.2 and 9.1 waves) and at every
//          measured M from 16 on; below about 10 rows some shapes run up to 30 % slower.  On sm_103a the _ts_wr
//          program and on sm_100a the _ts_ef program run every BF16 tail-split launch (kBf16TailSplitKernel).
//   MXFP8: faster up to 26 waves on sm_103a and 17 waves on sm_100a (not at about 32 and 18 waves) and at every
//          measured M from 129 on; below 129 rows some shapes run up to 15 % slower.
//   NVFP4: no longer repays the program beyond about 10 waves on sm_100a (where it costs about 1 % on whole tiles)
//          and about 50 waves on sm_103a; slower at M = 1 and M = 16 and faster at every measured M from 32 on.
//          On sm_103a the _ts_wr_pw_sr program replaces it from kNvfp4WrPwMinWaves waves on (PickNvfp4Schedule).
struct TailSplitRule {
  int64_t max_waves;
  int64_t min_rows;
};
constexpr TailSplitRule kBf16TailSplit = MINIMAX_H3_SM_MINOR == 3 ? TailSplitRule{10, 16} : TailSplitRule{9, 16};
constexpr TailSplitRule kMxfp8TailSplit = MINIMAX_H3_SM_MINOR == 3 ? TailSplitRule{26, 129} : TailSplitRule{17, 129};
constexpr TailSplitRule kNvfp4TailSplit = MINIMAX_H3_SM_MINOR == 3 ? TailSplitRule{48, 32} : TailSplitRule{10, 32};

// Quantized operand layouts (FlashInfer 128x4 swizzled scale tiles: 512-byte tiles of 128 rows x 4
// K-blocks, byte (row % 32) * 16 + (row / 32) * 4 + kblock).
constexpr int64_t kMxBlock = 32;
constexpr int64_t kMxfp8SfKTiles = 56;    // 512-byte activation scale tiles per 128-row block
constexpr int64_t kNvBlock = 16;
constexpr int64_t kNvfp4PackedCols = kAttnDim / 2;         // E2M1 nibble pairs per row
constexpr int64_t kNvfp4SfKTiles = 112;    // 512-byte activation scale tiles per 128-row block
constexpr int64_t kSfTileBytes = 512;
constexpr int64_t kSfbTileBytes = 2 * kSfTileBytes;        // combined 256-row weight tile (rows 0-127, 128-255)
constexpr int64_t kMxfp8OScaleBytes = kNTiles * kMxfp8SfKTiles * kSfbTileBytes;
constexpr int64_t kNvfp4OScaleBytes = kNTiles * kNvfp4SfKTiles * kSfbTileBytes;

// TMA boxes (resolved from the kernel descriptors at export time).
constexpr uint32_t kBf16BoxK = 64;
constexpr uint32_t kBf16BoxRowsA = 128;
constexpr uint32_t kBf16BoxRowsB = 128;
constexpr uint32_t kBf16BoxGroups = 1;
constexpr uint32_t kMxfp8BoxK = 128;
constexpr uint32_t kMxfp8BoxRowsA = 128;
constexpr uint32_t kMxfp8BoxRowsB = 128;
constexpr uint32_t kMxfp8BoxGroups = 2;
constexpr uint32_t kNvfp4BoxK = 128;
constexpr uint32_t kNvfp4BoxRowsA = 128;
constexpr uint32_t kNvfp4BoxRowsB = 128;
constexpr uint32_t kNvfp4BoxGroups = 1;
constexpr uint32_t kSfaTileRows = 4;  // one 512-byte tile = 4 rows x 128 bytes
constexpr uint32_t kSfbTileRows = 8;  // one 1024-byte tile = 8 rows x 128 bytes
constexpr uint32_t kSfTileRowBytes = 128;
// Consecutive scale tiles fetched per TMA load (one per K set of a pipeline stage).
constexpr uint32_t kMxfp8SfaBoxTiles = 2;
constexpr uint32_t kMxfp8SfbBoxTiles = 2;
constexpr uint32_t kNvfp4SfaBoxTiles = 4;
constexpr uint32_t kNvfp4SfbBoxTiles = 4;
// A tail half unit's operands: this CTA's 64 rows of the half's 128 weight rows (MXFP8: one K group per box, at
// that K group's offset in the whole tile's B stage), and per scale tile the half's 128-row atom (NVFP4 sm_103a: 4
// of the tile's 8 rows of 128 bytes, from row 4 h) or the whole tile's box from row 4 h (MXFP8, NVFP4 sm_100a: the
// half's atom lands in the slot the N = 128 MMA reads; rows past the tile are zero-filled).
constexpr uint32_t kBf16BoxRowsBHalf = 64;
constexpr uint32_t kMxfp8BoxRowsBHalf = 64;
constexpr uint32_t kMxfp8BoxGroupsBHalf = 1;
constexpr uint32_t kMxfp8SfbHalfBoxRows = kSfbTileRows;
constexpr uint32_t kNvfp4BoxRowsBHalf = 64;
constexpr uint32_t kNvfp4SfbHalfBoxRows = MINIMAX_H3_SM_MINOR == 3 ? 4 : 8;

// The activation as received: [P, M, 56 / P, 128] with P in {1, 2, 4, 8}.
struct ReceiveLayout {
  int64_t degree;  // P
  int64_t rows;    // M
};

int64_t MTiles(int64_t rows) {
  // Row tiles are consumed in CTA pairs: round the tile count up to an even number.
  int64_t tiles = (rows + kBlockM - 1) / kBlockM;
  return tiles + tiles % kCtaGroup;
}

// One cluster (CTA pair) per output tile pair; the hardware launches as many clusters as fit and
// running clusters claim the remaining tiles in order through cluster launch control.
int64_t GemmGrid(int64_t m_tiles) { return (m_tiles / kCtaGroup) * kNTiles * kCtaGroup; }

void CheckDeviceTensor(const TensorView& tensor, const char* name, DLDevice device, int64_t alignment) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as attn_out";
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

void CheckScalar(const TensorView& tensor, const char* name, DLDevice device) {
  CheckDeviceTensor(tensor, name, device, 4);
  CheckDtype(tensor, name, dl_float32, "float32");
  TVM_FFI_CHECK(tensor.ndim() == 1 && tensor.size(0) == 1, ValueError) << name << " must have shape [1]";
}

// attn_out must be a contiguous bfloat16 [P, M, 56 / P, 128] tensor; returns (P, M).
ReceiveLayout CheckAttnOut(const TensorView& attn_out) {
  TVM_FFI_CHECK(attn_out.device().device_type == kDLCUDA, ValueError) << "attn_out must be a CUDA tensor";
  CheckDtype(attn_out, "attn_out", dl_bfloat16, "bfloat16");
  TVM_FFI_CHECK(attn_out.ndim() == 4, ValueError)
      << "attn_out must be a rank-4 tensor [P, M, " << kNumHeads << " / P, " << kHeadDim << "]";
  const int64_t degree = attn_out.size(0);
  const int64_t rows = attn_out.size(1);
  TVM_FFI_CHECK(degree == 1 || degree == 2 || degree == 4 || degree == 8, ValueError)
      << "attn_out.shape[0] (the sequence-parallel degree P) must be 1, 2, 4 or 8; got " << degree;
  TVM_FFI_CHECK(rows >= 1 && rows <= kMaxRows, ValueError) << "M must satisfy 1 <= M <= " << kMaxRows;
  const int64_t heads_local = kNumHeads / degree;
  TVM_FFI_CHECK(attn_out.size(2) == heads_local && attn_out.size(3) == kHeadDim, ValueError)
      << "attn_out must have shape [" << degree << ", " << rows << ", " << heads_local << ", " << kHeadDim << "]";
  TVM_FFI_CHECK(attn_out.stride(3) == 1 && attn_out.stride(2) == kHeadDim &&
                    attn_out.stride(1) == heads_local * kHeadDim && attn_out.stride(0) == rows * heads_local * kHeadDim,
                ValueError)
      << "attn_out must be contiguous";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(attn_out.data_ptr()) % 16 == 0, ValueError)
      << "attn_out must be 16-byte aligned";
  return ReceiveLayout{degree, rows};
}

// Gate table: bf16 [rows, kHidden] with unit column stride, 1 <= rows <= kMaxGateRows, a row stride that
// is a multiple of kGateRowAlign elements in [kHidden, 2^32) and a 16-byte-aligned base.  A contiguous [rows, 5376] table and
// the gate_msa column chunk of a [rows, 6 * 5376] projection both qualify; the epilogue reads table row
// r at base + r * row_stride and multiplies by zero for an index outside [0, rows).
struct GateTable {
  int64_t rows;
  int64_t row_stride;  // elements between consecutive rows
};

GateTable CheckEpilogueTensors(const TensorView& gate, const TensorView& gate_index, const TensorView& residual,
                               const TensorView& out, int64_t rows, DLDevice device) {
  CheckDeviceTensor(gate, "gate", device, 16);
  CheckDtype(gate, "gate", dl_bfloat16, "bfloat16");
  TVM_FFI_CHECK(gate.ndim() == 2 && gate.size(1) == kHidden, ValueError)
      << "gate must have shape [rows, " << kHidden << "]";
  TVM_FFI_CHECK(gate.size(0) >= 1 && gate.size(0) <= kMaxGateRows, ValueError)
      << "gate must have 1 <= rows <= " << kMaxGateRows;
  TVM_FFI_CHECK(gate.stride(1) == 1, ValueError) << "gate must have unit column stride";
  TVM_FFI_CHECK(gate.stride(0) % kGateRowAlign == 0, ValueError)
      << "gate row stride must be a multiple of " << kGateRowAlign << " elements (16-byte rows)";
  // The epilogue forms the table offset as a 32x32->64-bit multiply of the row index and the stride:
  // rows must not overlap and the stride must fit 32 bits.
  TVM_FFI_CHECK(gate.stride(0) >= kHidden && gate.stride(0) < (int64_t(1) << 32), ValueError)
      << "gate row stride must satisfy " << kHidden << " <= stride(0) < 2^32 elements, got " << gate.stride(0);
  CheckVector(gate_index, "gate_index", rows, dl_int64, "int64", device, 8);
  // The epilogue reads residual and writes out as 256-bit vectors at 32-byte-aligned column offsets.
  CheckMatrix(residual, "residual", rows, kHidden, dl_bfloat16, "bfloat16", device, 32);
  CheckMatrix(out, "out", rows, kHidden, dl_bfloat16, "bfloat16", device, 32);
  return GateTable{gate.size(0), gate.stride(0)};
}

// K-major [rows, cols] operand viewed as the rank-3 tensor (box_k, rows, cols / box_k): coordinate
// (0, row, k / box_k) addresses one box_k-wide K group of one row.  128-byte swizzle; rows beyond
// the tensor are zero-filled by TMA (the activation box may extend past the last row) and never
// stored.
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
// No swizzle; the tiles are consumed verbatim by tcgen05.cp into TMEM.  box_rows (default: tile_rows) narrows
// the box to the first box_rows rows of each tile from the load's row coordinate (a tail half unit's scales).
CUtensorMap EncodeScaleTiles(const void* base, uint32_t tile_rows, int64_t num_tiles, uint32_t box_tiles,
                             const char* name, uint32_t box_rows = 0) {
  TVM_FFI_CHECK(num_tiles % box_tiles == 0, RuntimeError) << name << ": tile count is not a multiple of the TMA box";
  uint64_t global_dim[3] = {kSfTileRowBytes, tile_rows, static_cast<uint64_t>(num_tiles)};
  uint64_t global_strides[2] = {kSfTileRowBytes, static_cast<uint64_t>(kSfTileRowBytes) * tile_rows};
  uint32_t box_dim[3] = {kSfTileRowBytes, box_rows == 0 ? tile_rows : box_rows, box_tiles};
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

// Clusters of `kernel` (kClusterX CTAs of `threads` threads and `smem_bytes` of dynamic shared memory each) the
// active device holds at once.
template <typename Kernel>
int64_t MaxActiveClusters(Kernel kernel, int threads, int smem_bytes, const char* name) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = kClusterX;
  attr.val.clusterDim.y = kClusterY;
  attr.val.clusterDim.z = kClusterZ;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kClusterX, kClusterY, kClusterZ);
  config.blockDim = dim3(static_cast<unsigned int>(threads), 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(smem_bytes);
  config.attrs = &attr;
  config.numAttrs = 1;
  int clusters = 0;
  const cudaError_t status =
      cudaOccupancyMaxActiveClusters(&clusters, reinterpret_cast<const void*>(kernel), &config);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query the co-resident clusters (" << name << "): " << cudaGetErrorString(status);
  return clusters;
}

// Co-resident CTA pairs of each tail-split program on the active device: the wave of its tail schedule.
struct TailSplitPairs {
  int64_t bf16;
  int64_t mxfp8;
  int64_t nvfp4;
};

// Per-device one-time configuration: capability check, dynamic shared memory opt-in and the CTA pairs
// of the tail-split programs the device holds at once, returned.
TailSplitPairs ConfigureKernels() {
  static std::mutex mutex;
  static std::vector<std::pair<int, TailSplitPairs>> configured_devices;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to get the active CUDA device: " << cudaGetErrorString(status);
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto& configured : configured_devices) {
    if (configured.first == device) return configured.second;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to query CUDA device properties: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(properties.major == 10 && properties.minor == MINIMAX_H3_SM_MINOR, RuntimeError)
      << "this MiniMax-H3 out-proj build targets compute capability 10." << MINIMAX_H3_SM_MINOR
      << " exactly (tcgen05 + TMEM + 2-CTA MMA)";
  TVM_FFI_CHECK(properties.multiProcessorCount >= kCtaGroup, RuntimeError)
      << "MiniMax-H3 out-proj requires at least " << kCtaGroup << " SMs";
  OptInSmem(kernel_minimax_h3_quant_mxfp8, kQuantMxfp8Smem, "quantize mxfp8");
  OptInSmem(kernel_minimax_h3_quant_nvfp4, kQuantNvfp4Smem, "quantize nvfp4");
  OptInSmem(kernel_minimax_h3_out_proj_bf16, kGemmBf16Smem, "gemm bf16");
  OptInSmem(kernel_minimax_h3_out_proj_bf16_ts, kGemmBf16Smem, "gemm bf16 tail split");
#if MINIMAX_H3_SM_MINOR == 3
  OptInSmem(kernel_minimax_h3_out_proj_bf16_ts_wr, kGemmBf16Smem, "gemm bf16 tail split (wr)");
#else
  OptInSmem(kernel_minimax_h3_out_proj_bf16_ts_ef, kGemmBf16Smem, "gemm bf16 tail split (ef)");
#endif
  OptInSmem(kernel_minimax_h3_out_proj_e4m3, kGemmMxfp8Smem, "gemm mxfp8");
  OptInSmem(kernel_minimax_h3_out_proj_e4m3_ts, kGemmMxfp8Smem, "gemm mxfp8 tail split");
  OptInSmem(kernel_minimax_h3_out_proj_e2m1, kGemmNvfp4Smem, "gemm nvfp4");
  OptInSmem(kernel_minimax_h3_out_proj_e2m1_ts, kGemmNvfp4Smem, "gemm nvfp4 tail split");
#if MINIMAX_H3_SM_MINOR == 3
  OptInSmem(kernel_minimax_h3_out_proj_e2m1_ts_wr_pw, kGemmNvfp4Smem, "gemm nvfp4 tail split (wr_pw)");
  OptInSmem(kernel_minimax_h3_out_proj_e2m1_ts_wr_pw_sr, kGemmNvfp4SrSmem, "gemm nvfp4 tail split (wr_pw_sr)");
#endif
  const TailSplitPairs pairs{
      MaxActiveClusters(kernel_minimax_h3_out_proj_bf16_ts, kGemmBf16Threads, kGemmBf16Smem, "gemm bf16 tail split"),
      MaxActiveClusters(kernel_minimax_h3_out_proj_e4m3_ts, kGemmQuantThreads, kGemmMxfp8Smem, "gemm mxfp8 tail split"),
      MaxActiveClusters(kernel_minimax_h3_out_proj_e2m1_ts, kGemmQuantThreads, kGemmNvfp4Smem, "gemm nvfp4 tail split")};
  configured_devices.emplace_back(device, pairs);
  return pairs;
}

// A GEMM's schedule.  num_tiles tile pairs on `pairs` co-resident CTA pairs run as W = num_tiles / pairs full
// waves and R = num_tiles % pairs tail tiles.  When 0 < 2 R <= pairs, num_tiles <= rule.max_waves * pairs and
// rows >= rule.min_rows the tail-split program runs the W * pairs whole tiles as units [0, full_units) and each
// tail tile as two half units after them: num_units = num_tiles + R units on a grid of 2 * num_units CTAs.
// Otherwise the whole-tile program runs (one CTA pair per tile pair, claimed in order through cluster launch
// control).
struct GemmSchedule {
  bool tail_split;
  int64_t num_units;
  int64_t full_units;
};

GemmSchedule TailSplitSchedule(TailSplitRule rule, int64_t rows, int64_t num_tiles, int64_t pairs) {
  if (pairs >= 1 && rows >= rule.min_rows) {
    const int64_t tail = num_tiles % pairs;
    const bool within_waves = num_tiles <= rule.max_waves * pairs;
    if (tail > 0 && 2 * tail <= pairs && within_waves) {
      return GemmSchedule{true, num_tiles + tail, num_tiles - tail};
    }
  }
  return GemmSchedule{false, num_tiles, num_tiles};
}

// The schedule a GEMM launches: TailSplitSchedule's choice, or forced_full_units (tests only; kScheduleRule
// otherwise): -1 runs the whole-tile program, 0 <= n <= num_tiles the tail-split program with n whole-tile units
// and every later tile as two half units.
constexpr int64_t kScheduleRule = -2;

GemmSchedule PickSchedule(TailSplitRule rule, int64_t rows, int64_t num_tiles, int64_t pairs,
                          int64_t forced_full_units) {
  if (forced_full_units == kScheduleRule) return TailSplitSchedule(rule, rows, num_tiles, pairs);
  TVM_FFI_CHECK(forced_full_units >= -1 && forced_full_units <= num_tiles, ValueError)
      << "full_units must be -1 (the whole-tile program) or in [0, " << num_tiles << "], got " << forced_full_units;
  return forced_full_units < 0 ? GemmSchedule{false, num_tiles, num_tiles}
                               : GemmSchedule{true, 2 * num_tiles - forced_full_units, forced_full_units};
}

// BF16's tail-split program: on sm_103a kernel_minimax_h3_out_proj_bf16_ts_wr, the _ts program with 256-bit
// residual loads (the same output bits, launch shape and shared memory; measured faster on GB300 at every shape
// kBf16TailSplit routes there); on sm_100a kernel_minimax_h3_out_proj_bf16_ts_ef, the _ts program with one
// epilogue drain per unit form (whole tile, half unit), each with its literal column count (the same output bits,
// launch shape and shared memory; measured faster on GB200 at every shape kBf16TailSplit routes there).
#if MINIMAX_H3_SM_MINOR == 3
constexpr auto kBf16TailSplitKernel = kernel_minimax_h3_out_proj_bf16_ts_wr;
#else
constexpr auto kBf16TailSplitKernel = kernel_minimax_h3_out_proj_bf16_ts_ef;
#endif

// NVFP4 on sm_103a: kernel_minimax_h3_out_proj_e2m1_ts_wr_pw, the _ts program with 256-bit residual loads and the
// first unit's weight loads issued before the dependency wait (the same output bits, launch shape and shared
// memory), runs, as whole-tile units only (full_units = num_tiles), the shapes the tail split leaves whole within
// kNvfp4TailSplit's row and wave limits; kernel_minimax_h3_out_proj_e2m1_ts_wr_pw_sr, the _ts_wr_pw program with
// the epilogue's residual staged in shared memory (two cp.async slots per epilogue warp; the same output bits and
// launch shape, kGemmNvfp4SrSmem dynamic shared memory), runs the tail-split shapes from kNvfp4WrPwMinWaves full
// waves on (fewer keep _ts); every other shape keeps its program.  Measured on GB300 against the program each
// shape ran before: _ts_wr_pw faster at every measured whole shape of 4.7 to 41.7 waves, not at 5 waves and
// fewer; _ts_wr_pw_sr faster than _ts_wr_pw at every measured tail-split shape of 8 to 40 waves.  A forced
// schedule (tests only) runs the program the rule picks for the shape: _ts_wr_pw_sr, _ts_wr_pw or _ts.
constexpr int64_t kNvfp4WrPwMinWaves = 8;

struct Nvfp4Schedule {
  GemmSchedule gemm;
  bool wr_pw;  // the tail-split program is kernel_minimax_h3_out_proj_e2m1_ts_wr_pw (sm_103a only)
  bool sr;     // the tail-split program is kernel_minimax_h3_out_proj_e2m1_ts_wr_pw_sr (sm_103a only)
};

Nvfp4Schedule PickNvfp4Schedule(int64_t rows, int64_t num_tiles, int64_t pairs, int64_t forced_full_units) {
  GemmSchedule rule = TailSplitSchedule(kNvfp4TailSplit, rows, num_tiles, pairs);
  bool wr_pw = false;
  bool sr = false;
  if (MINIMAX_H3_SM_MINOR == 3 && pairs >= 1 && rows >= kNvfp4TailSplit.min_rows &&
      num_tiles <= kNvfp4TailSplit.max_waves * pairs) {
    if (rule.tail_split) {
      wr_pw = num_tiles >= kNvfp4WrPwMinWaves * pairs;
      sr = wr_pw;
    } else {
      wr_pw = true;
      rule = GemmSchedule{true, num_tiles, num_tiles};
    }
  }
  if (forced_full_units == kScheduleRule) return Nvfp4Schedule{rule, wr_pw, sr};
  return Nvfp4Schedule{PickSchedule(kNvfp4TailSplit, rows, num_tiles, pairs, forced_full_units), wr_pw, sr};
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

// BF16 operator: one launch.  attn_out: bfloat16 [P, M, 56 / P, 128] receive layout (P in {1, 2, 4, 8});
// o_weight: bfloat16 [5376, 7168]; gate: bfloat16 [rows, 5376] table (see CheckEpilogueTensors);
// gate_index: int64 [M], values outside [0, rows) give gate = 0; residual / out: bfloat16 [M, 5376].
// out = BF16(residual + BF16(gate[gate_index] * BF16(A @ o_weight^T))).  TailSplitSchedule (kBf16TailSplit)
// picks the GEMM's program; forced_full_units as in PickSchedule.
void Bf16Operator(TensorView attn_out, TensorView o_weight, TensorView gate, TensorView gate_index,
                  TensorView residual, TensorView out, int64_t forced_full_units) {
  const ReceiveLayout layout = CheckAttnOut(attn_out);
  const DLDevice device = attn_out.device();
  CheckMatrix(o_weight, "o_weight", kHidden, kAttnDim, dl_bfloat16, "bfloat16", device, 16);
  const GateTable table = CheckEpilogueTensors(gate, gate_index, residual, out, layout.rows, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const TailSplitPairs pairs = ConfigureKernels();

  const int64_t m_tiles = MTiles(layout.rows);
  const int64_t num_tiles = (m_tiles / kCtaGroup) * kNTiles;
  const GemmSchedule schedule = PickSchedule(kBf16TailSplit, layout.rows, num_tiles, pairs.bf16, forced_full_units);
  // The activation descriptor spans the [P * M, 7168 / P] view of attn_out.
  const CUtensorMap a_map = EncodeKMajorRows(attn_out.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                                             layout.degree * layout.rows, kAttnDim / layout.degree, kBf16BoxK,
                                             kBf16BoxRowsA, kBf16BoxGroups, "attn_out");
  const CUtensorMap b_map = EncodeKMajorRows(o_weight.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kHidden,
                                             kAttnDim, kBf16BoxK, kBf16BoxRowsB, kBf16BoxGroups, "o_weight");
  if (!schedule.tail_split) {
    LaunchCluster(kernel_minimax_h3_out_proj_bf16, GemmGrid(m_tiles), kGemmBf16Threads, kGemmBf16Smem, stream, kGemmBf16Pdl,
                  "MiniMax-H3 out-proj (bf16)", a_map, b_map, static_cast<__nv_bfloat16*>(gate.data_ptr()),
                  static_cast<long long*>(gate_index.data_ptr()), static_cast<__nv_bfloat16*>(residual.data_ptr()),
                  static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(layout.rows), static_cast<int>(m_tiles),
                  static_cast<int>(kBf16KBlocks / layout.degree), static_cast<int>(table.rows),
                  static_cast<long long>(table.row_stride));
    return;
  }
  const CUtensorMap b_half_map = EncodeKMajorRows(o_weight.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kHidden,
                                                  kAttnDim, kBf16BoxK, kBf16BoxRowsBHalf, kBf16BoxGroups,
                                                  "o_weight (half unit)");
  LaunchCluster(kBf16TailSplitKernel, kCtaGroup * schedule.num_units, kGemmBf16Threads, kGemmBf16Smem,
                stream, kGemmBf16Pdl, "MiniMax-H3 out-proj (bf16, tail split)", a_map, b_map,
                static_cast<__nv_bfloat16*>(gate.data_ptr()), static_cast<long long*>(gate_index.data_ptr()),
                static_cast<__nv_bfloat16*>(residual.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
                static_cast<int>(layout.rows), static_cast<int>(m_tiles), static_cast<int>(kBf16KBlocks / layout.degree),
                static_cast<int>(table.rows), static_cast<long long>(table.row_stride), b_half_map,
                static_cast<int>(schedule.full_units));
}

// MXFP8 operator: two launches.  o_weight_q: float8_e4m3fn [5376, 7168]; o_scale_tiles: uint8
// [21 * 56 * 1024] combined 256-row weight scale tiles ([n_tile][k_set][half][512]); workspace_q:
// caller-owned float8_e4m3fn [M, 7168]; workspace_sf: caller-owned uint8 of at least
// m_tiles(M) * 56 * 512 bytes (swizzled 128x4 activation scales).  Both workspaces receive the
// FlashInfer-exact MXFP8 quantization of the logical activation A.  The GEMM is launched
// programmatically dependent on the quantization pass (see kGemmMxfp8Pdl); TailSplitSchedule (kMxfp8TailSplit)
// picks its program; forced_full_units as in PickSchedule.
void Mxfp8Operator(TensorView attn_out, TensorView o_weight_q, TensorView o_scale_tiles, TensorView gate,
                   TensorView gate_index, TensorView residual, TensorView out, TensorView workspace_q,
                   TensorView workspace_sf, int64_t forced_full_units) {
  const ReceiveLayout layout = CheckAttnOut(attn_out);
  const DLDevice device = attn_out.device();
  const int64_t rows = layout.rows;
  const int64_t m_tiles = MTiles(rows);
  const int64_t sf_bytes = m_tiles * kMxfp8SfKTiles * kSfTileBytes;
  CheckMatrix(o_weight_q, "o_weight_q", kHidden, kAttnDim, dl_float8_e4m3fn, "float8_e4m3fn", device, 16);
  CheckByteBuffer(o_scale_tiles, "o_scale_tiles", kMxfp8OScaleBytes, device);
  TVM_FFI_CHECK(o_scale_tiles.size(0) == kMxfp8OScaleBytes, ValueError)
      << "o_scale_tiles must hold exactly " << kMxfp8OScaleBytes << " bytes";
  const GateTable table = CheckEpilogueTensors(gate, gate_index, residual, out, rows, device);
  CheckMatrix(workspace_q, "workspace_q", rows, kAttnDim, dl_float8_e4m3fn, "float8_e4m3fn", device, 16);
  CheckByteBuffer(workspace_sf, "workspace_sf", sf_bytes, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const TailSplitPairs pairs = ConfigureKernels();
  const int64_t num_tiles = (m_tiles / kCtaGroup) * kNTiles;
  const GemmSchedule schedule = PickSchedule(kMxfp8TailSplit, rows, num_tiles, pairs.mxfp8, forced_full_units);

  const int64_t quant_grid = (rows * kQuantRowSplitMxfp8 + kQuantRowsPerCta - 1) / kQuantRowsPerCta;
  kernel_minimax_h3_quant_mxfp8<<<static_cast<unsigned int>(quant_grid), kQuantThreads, kQuantMxfp8Smem, stream>>>(
      static_cast<__nv_bfloat16*>(attn_out.data_ptr()), static_cast<uint8_t*>(workspace_q.data_ptr()),
      static_cast<uint8_t*>(workspace_sf.data_ptr()), static_cast<int>(rows),
      static_cast<int>(kAttnDim / layout.degree));
  CheckLaunch("MiniMax-H3 out-proj quantize (mxfp8)");

  const CUtensorMap a_map = EncodeKMajorRows(workspace_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows, kAttnDim,
                                             kMxfp8BoxK, kMxfp8BoxRowsA, kMxfp8BoxGroups, "workspace_q");
  const CUtensorMap b_map = EncodeKMajorRows(o_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHidden,
                                             kAttnDim, kMxfp8BoxK, kMxfp8BoxRowsB, kMxfp8BoxGroups, "o_weight_q");
  const CUtensorMap sfa_map = EncodeScaleTiles(workspace_sf.data_ptr(), kSfaTileRows, m_tiles * kMxfp8SfKTiles,
                                               kMxfp8SfaBoxTiles, "workspace_sf");
  const CUtensorMap sfb_map = EncodeScaleTiles(o_scale_tiles.data_ptr(), kSfbTileRows, kNTiles * kMxfp8SfKTiles,
                                               kMxfp8SfbBoxTiles, "o_scale_tiles");
  if (!schedule.tail_split) {
    LaunchCluster(kernel_minimax_h3_out_proj_e4m3, GemmGrid(m_tiles), kGemmQuantThreads, kGemmMxfp8Smem, stream, kGemmMxfp8Pdl,
                  "MiniMax-H3 out-proj (mxfp8)", a_map, b_map, sfa_map, sfb_map,
                  static_cast<__nv_bfloat16*>(gate.data_ptr()), static_cast<long long*>(gate_index.data_ptr()),
                  static_cast<__nv_bfloat16*>(residual.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
                  static_cast<int>(rows), static_cast<int>(m_tiles), static_cast<int>(table.rows),
                  static_cast<long long>(table.row_stride));
    return;
  }
  const CUtensorMap b_half_map = EncodeKMajorRows(o_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHidden,
                                                  kAttnDim, kMxfp8BoxK, kMxfp8BoxRowsBHalf, kMxfp8BoxGroupsBHalf,
                                                  "o_weight_q (half unit)");
  const CUtensorMap sfb_half_map = EncodeScaleTiles(o_scale_tiles.data_ptr(), kSfbTileRows, kNTiles * kMxfp8SfKTiles,
                                                    kMxfp8SfbBoxTiles, "o_scale_tiles (half unit)", kMxfp8SfbHalfBoxRows);
  LaunchCluster(kernel_minimax_h3_out_proj_e4m3_ts, kCtaGroup * schedule.num_units, kGemmQuantThreads, kGemmMxfp8Smem,
                stream, kGemmMxfp8Pdl, "MiniMax-H3 out-proj (mxfp8, tail split)", a_map, b_map, sfa_map, sfb_map,
                static_cast<__nv_bfloat16*>(gate.data_ptr()), static_cast<long long*>(gate_index.data_ptr()),
                static_cast<__nv_bfloat16*>(residual.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
                static_cast<int>(rows), static_cast<int>(m_tiles), static_cast<int>(table.rows),
                static_cast<long long>(table.row_stride), b_half_map, sfb_half_map,
                static_cast<int>(schedule.full_units));
}

// NVFP4 operator: two launches.  a_global_scale: float32 [1] activation global scale (FlashInfer
// nvfp4_quantize convention, 448 * 6 / absmax); o_weight_q: uint8 [5376, 3584] packed E2M1;
// o_scale_tiles: uint8 [21 * 112 * 1024] combined 256-row weight scale tiles; alpha: float32 [1] =
// 1 / (a_global_scale * w_global_scale); workspace_q: caller-owned uint8 [M, 3584]; workspace_sf:
// caller-owned uint8 of at least m_tiles(M) * 112 * 512 bytes.  The GEMM is launched programmatically
// dependent on the quantization pass (see kGemmNvfp4Pdl); PickNvfp4Schedule picks its program; forced_full_units as
// in PickSchedule.
void Nvfp4Operator(TensorView attn_out, TensorView a_global_scale, TensorView o_weight_q, TensorView o_scale_tiles,
                   TensorView alpha, TensorView gate, TensorView gate_index, TensorView residual, TensorView out,
                   TensorView workspace_q, TensorView workspace_sf, int64_t forced_full_units) {
  const ReceiveLayout layout = CheckAttnOut(attn_out);
  const DLDevice device = attn_out.device();
  const int64_t rows = layout.rows;
  const int64_t m_tiles = MTiles(rows);
  const int64_t sf_bytes = m_tiles * kNvfp4SfKTiles * kSfTileBytes;
  CheckScalar(a_global_scale, "a_global_scale", device);
  CheckScalar(alpha, "alpha", device);
  CheckMatrix(o_weight_q, "o_weight_q", kHidden, kNvfp4PackedCols, dl_uint8, "uint8", device, 16);
  CheckByteBuffer(o_scale_tiles, "o_scale_tiles", kNvfp4OScaleBytes, device);
  TVM_FFI_CHECK(o_scale_tiles.size(0) == kNvfp4OScaleBytes, ValueError)
      << "o_scale_tiles must hold exactly " << kNvfp4OScaleBytes << " bytes";
  const GateTable table = CheckEpilogueTensors(gate, gate_index, residual, out, rows, device);
  CheckMatrix(workspace_q, "workspace_q", rows, kNvfp4PackedCols, dl_uint8, "uint8", device, 16);
  CheckByteBuffer(workspace_sf, "workspace_sf", sf_bytes, device);

  ffi::CUDADeviceGuard device_guard(device.device_id);
  const cudaStream_t stream = get_stream(device);
  const TailSplitPairs pairs = ConfigureKernels();
  const int64_t num_tiles = (m_tiles / kCtaGroup) * kNTiles;
  const Nvfp4Schedule nvfp4 = PickNvfp4Schedule(rows, num_tiles, pairs.nvfp4, forced_full_units);
  const GemmSchedule schedule = nvfp4.gemm;

  const int64_t quant_grid = (rows * kQuantRowSplitNvfp4 + kQuantRowsPerCta - 1) / kQuantRowsPerCta;
  kernel_minimax_h3_quant_nvfp4<<<static_cast<unsigned int>(quant_grid), kQuantThreads, kQuantNvfp4Smem, stream>>>(
      static_cast<__nv_bfloat16*>(attn_out.data_ptr()), static_cast<float*>(a_global_scale.data_ptr()),
      static_cast<unsigned int*>(workspace_q.data_ptr()), static_cast<uint8_t*>(workspace_sf.data_ptr()),
      static_cast<int>(rows), static_cast<int>(kAttnDim / layout.degree));
  CheckLaunch("MiniMax-H3 out-proj quantize (nvfp4)");

  const CUtensorMap a_map = EncodeKMajorRows(workspace_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, rows,
                                             kNvfp4PackedCols, kNvfp4BoxK, kNvfp4BoxRowsA, kNvfp4BoxGroups,
                                             "workspace_q");
  const CUtensorMap b_map = EncodeKMajorRows(o_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHidden,
                                             kNvfp4PackedCols, kNvfp4BoxK, kNvfp4BoxRowsB, kNvfp4BoxGroups,
                                             "o_weight_q");
  const CUtensorMap sfa_map = EncodeScaleTiles(workspace_sf.data_ptr(), kSfaTileRows, m_tiles * kNvfp4SfKTiles,
                                               kNvfp4SfaBoxTiles, "workspace_sf");
  const CUtensorMap sfb_map = EncodeScaleTiles(o_scale_tiles.data_ptr(), kSfbTileRows, kNTiles * kNvfp4SfKTiles,
                                               kNvfp4SfbBoxTiles, "o_scale_tiles");
  if (!schedule.tail_split) {
    LaunchCluster(kernel_minimax_h3_out_proj_e2m1, GemmGrid(m_tiles), kGemmQuantThreads, kGemmNvfp4Smem, stream, kGemmNvfp4Pdl,
                  "MiniMax-H3 out-proj (nvfp4)", a_map, b_map, sfa_map, sfb_map,
                  static_cast<float*>(alpha.data_ptr()), static_cast<__nv_bfloat16*>(gate.data_ptr()),
                  static_cast<long long*>(gate_index.data_ptr()), static_cast<__nv_bfloat16*>(residual.data_ptr()),
                  static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(rows), static_cast<int>(m_tiles),
                  static_cast<int>(table.rows), static_cast<long long>(table.row_stride));
    return;
  }
  const CUtensorMap b_half_map = EncodeKMajorRows(o_weight_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHidden,
                                                  kNvfp4PackedCols, kNvfp4BoxK, kNvfp4BoxRowsBHalf, kNvfp4BoxGroups,
                                                  "o_weight_q (half unit)");
  const CUtensorMap sfb_half_map = EncodeScaleTiles(o_scale_tiles.data_ptr(), kSfbTileRows, kNTiles * kNvfp4SfKTiles,
                                                    kNvfp4SfbBoxTiles, "o_scale_tiles (half unit)", kNvfp4SfbHalfBoxRows);
#if MINIMAX_H3_SM_MINOR == 3
  const auto tail_kernel = nvfp4.sr      ? kernel_minimax_h3_out_proj_e2m1_ts_wr_pw_sr
                           : nvfp4.wr_pw ? kernel_minimax_h3_out_proj_e2m1_ts_wr_pw
                                         : kernel_minimax_h3_out_proj_e2m1_ts;
  const int tail_smem = nvfp4.sr ? kGemmNvfp4SrSmem : kGemmNvfp4Smem;
#else
  const auto tail_kernel = kernel_minimax_h3_out_proj_e2m1_ts;
  const int tail_smem = kGemmNvfp4Smem;
#endif
  LaunchCluster(tail_kernel, kCtaGroup * schedule.num_units, kGemmQuantThreads, tail_smem,
                stream, kGemmNvfp4Pdl, "MiniMax-H3 out-proj (nvfp4, tail split)", a_map, b_map, sfa_map, sfb_map,
                static_cast<float*>(alpha.data_ptr()), static_cast<__nv_bfloat16*>(gate.data_ptr()),
                static_cast<long long*>(gate_index.data_ptr()), static_cast<__nv_bfloat16*>(residual.data_ptr()),
                static_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(rows), static_cast<int>(m_tiles),
                static_cast<int>(table.rows), static_cast<long long>(table.row_stride), b_half_map, sfb_half_map,
                static_cast<int>(schedule.full_units));
}

void CheckForcedFullUnits(int64_t full_units) {
  TVM_FFI_CHECK(full_units >= -1, ValueError)
      << "full_units must be -1 (the whole-tile program) or a whole-tile unit count, got " << full_units;
}

void minimax_h3_out_proj(TensorView attn_out, TensorView o_weight, TensorView gate, TensorView gate_index,
                         TensorView residual, TensorView out) {
  Bf16Operator(attn_out, o_weight, gate, gate_index, residual, out, kScheduleRule);
}

void minimax_h3_out_proj_mxfp8(TensorView attn_out, TensorView o_weight_q, TensorView o_scale_tiles, TensorView gate,
                               TensorView gate_index, TensorView residual, TensorView out, TensorView workspace_q,
                               TensorView workspace_sf) {
  Mxfp8Operator(attn_out, o_weight_q, o_scale_tiles, gate, gate_index, residual, out, workspace_q, workspace_sf,
                kScheduleRule);
}

void minimax_h3_out_proj_nvfp4(TensorView attn_out, TensorView a_global_scale, TensorView o_weight_q,
                               TensorView o_scale_tiles, TensorView alpha, TensorView gate, TensorView gate_index,
                               TensorView residual, TensorView out, TensorView workspace_q, TensorView workspace_sf) {
  Nvfp4Operator(attn_out, a_global_scale, o_weight_q, o_scale_tiles, alpha, gate, gate_index, residual, out,
                workspace_q, workspace_sf, kScheduleRule);
}

// Test entry points: an operator with its GEMM program forced (full_units as forced_full_units of PickSchedule).
// Every schedule writes the same bits.
void minimax_h3_out_proj_schedule(TensorView attn_out, TensorView o_weight, TensorView gate, TensorView gate_index,
                                  TensorView residual, TensorView out, int64_t full_units) {
  CheckForcedFullUnits(full_units);
  Bf16Operator(attn_out, o_weight, gate, gate_index, residual, out, full_units);
}

void minimax_h3_out_proj_mxfp8_schedule(TensorView attn_out, TensorView o_weight_q, TensorView o_scale_tiles,
                                        TensorView gate, TensorView gate_index, TensorView residual, TensorView out,
                                        TensorView workspace_q, TensorView workspace_sf, int64_t full_units) {
  CheckForcedFullUnits(full_units);
  Mxfp8Operator(attn_out, o_weight_q, o_scale_tiles, gate, gate_index, residual, out, workspace_q, workspace_sf,
                full_units);
}

void minimax_h3_out_proj_nvfp4_schedule(TensorView attn_out, TensorView a_global_scale, TensorView o_weight_q,
                                        TensorView o_scale_tiles, TensorView alpha, TensorView gate,
                                        TensorView gate_index, TensorView residual, TensorView out,
                                        TensorView workspace_q, TensorView workspace_sf, int64_t full_units) {
  CheckForcedFullUnits(full_units);
  Nvfp4Operator(attn_out, a_global_scale, o_weight_q, o_scale_tiles, alpha, gate, gate_index, residual, out,
                workspace_q, workspace_sf, full_units);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_out_proj, minimax_h3_out_proj);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_out_proj_mxfp8, minimax_h3_out_proj_mxfp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_out_proj_nvfp4, minimax_h3_out_proj_nvfp4);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_out_proj_schedule, minimax_h3_out_proj_schedule);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_out_proj_mxfp8_schedule, minimax_h3_out_proj_mxfp8_schedule);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_out_proj_nvfp4_schedule, minimax_h3_out_proj_nvfp4_schedule);
// clang-format on
