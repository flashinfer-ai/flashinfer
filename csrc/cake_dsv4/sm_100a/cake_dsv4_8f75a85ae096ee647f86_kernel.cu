/*
 * Copyright (c) 2023 by FlashInfer team.
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
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 64
#define TMEM_TMEM_SFB_OFFSET 96
#define NUM_K_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 1
#define NUM_INDEX_PIPE_STAGES 6
#define NUM_SOURCE_WORK_PIPE_STAGES 2
#define NUM_SOURCE_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 27648
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 27648
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_Q_A_OFF 1024
#define SMEM_SMEM_Q_A_STAGE_BYTES 8192
#define SMEM_SMEM_Q_A_STRIDE 8192
#define SMEM_SMEM_Q_B_OFF 9216
#define SMEM_SMEM_Q_B_STAGE_BYTES 8192
#define SMEM_SMEM_Q_B_STRIDE 8192
#define SMEM_SMEM_Q_ROPE_OFF 17408
#define SMEM_SMEM_Q_ROPE_STAGE_BYTES 8192
#define SMEM_SMEM_Q_ROPE_STRIDE 8192
#define SMEM_SMEM_Q_SF_OFF 25600
#define SMEM_SMEM_Q_SF_STAGE_BYTES 2048
#define SMEM_SMEM_Q_SF_STRIDE 2048
#define SMEM_SMEM_K_A_OFF 27648
#define SMEM_SMEM_K_A_STAGE_BYTES 8192
#define SMEM_SMEM_K_A_STRIDE 28672
#define SMEM_SMEM_K_B_OFF 35840
#define SMEM_SMEM_K_B_STAGE_BYTES 8192
#define SMEM_SMEM_K_B_STRIDE 28672
#define SMEM_SMEM_K_C_OFF 44032
#define SMEM_SMEM_K_C_STAGE_BYTES 8192
#define SMEM_SMEM_K_C_STRIDE 28672
#define SMEM_SMEM_K_SF_OFF 52224
#define SMEM_SMEM_K_SF_STAGE_BYTES 4096
#define SMEM_SMEM_K_SF_STRIDE 28672
#define SMEM_SMEM_V_FULL_OFF 84992
#define SMEM_SMEM_V_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_V_FULL_STRIDE 32768
#define SMEM_SMEM_STATS_MAX_OFF 150528
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_MAX_STRIDE 1024
#define SMEM_SMEM_STATS_SUM_OFF 151552
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 512
#define SMEM_SMEM_STATS_SUM_STRIDE 512
#define SMEM_SMEM_STATS_FINAL_MAX_OFF 152064
#define SMEM_SMEM_STATS_FINAL_MAX_STAGE_BYTES 512
#define SMEM_SMEM_STATS_FINAL_MAX_STRIDE 512
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_OFF 152576
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STAGE_BYTES 1024
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STRIDE 1024
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_OFF 153600
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STRIDE 512
#define SMEM_SMEM_Q_RING_PAD_OFF 176736
#define SMEM_SMEM_Q_RING_PAD_STAGE_BYTES 416
#define SMEM_SMEM_Q_RING_PAD_STRIDE 416
#define SMEM_SMEM_Q_RING_OFF 177152
#define SMEM_SMEM_Q_RING_STAGE_BYTES 8192
#define SMEM_SMEM_Q_RING_STRIDE 8192
#define SMEM_SMEM_Q_LUT6_OFF 193536
#define SMEM_SMEM_Q_LUT6_STAGE_BYTES 1024
#define SMEM_SMEM_Q_LUT6_STRIDE 1024
#define SMEM_SMEM_Q_LUTINV_OFF 194560
#define SMEM_SMEM_Q_LUTINV_STAGE_BYTES 512
#define SMEM_SMEM_Q_LUTINV_STRIDE 512
#define SMEM_SMEM_SPARSE_INDICES_OFF 154112
#define SMEM_SMEM_SPARSE_INDICES_STAGE_BYTES 1024
#define SMEM_SMEM_SPARSE_INDICES_STRIDE 1024
#define SMEM_SMEM_P_FP8_OFF 160256
#define SMEM_SMEM_P_FP8_STAGE_BYTES 8192
#define SMEM_SMEM_P_FP8_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 176640
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_SMEM_KEXP_OFF 176672
#define SMEM_SMEM_KEXP_STAGE_BYTES 32
#define SMEM_SMEM_KEXP_STRIDE 32
#define SMEM_SMEM_KMAX_OFF 176704
#define SMEM_SMEM_KMAX_STAGE_BYTES 32
#define SMEM_SMEM_KMAX_STRIDE 32
#define SMEM_TOTAL 195072
#define THREADS 512

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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}



__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}





__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)






__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem_mc(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast variant: the PTX grammar ties the .shared::cluster
    // destination to .multicast::cluster + ctaMask (cf. cuda_ptx /
    // SM100_TMA_LOAD_MULTICAST_2D_GATHER4 in CUTLASS).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
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

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_8f75a85ae096ee647f86(const __grid_constant__ CUtensorMap tmap_q, unsigned int* __restrict__ Q, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, const __grid_constant__ CUtensorMap tmap_swa_sf, const __grid_constant__ CUtensorMap tmap_compressed_sf, const __grid_constant__ CUtensorMap tmap_o, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, float* __restrict__ partial_lse, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items, int max_q_len, int batch_size, int swa_page_log2, int swa_pitch_units, int swa_footer_units, int compressed_page_log2, int compressed_pitch_units, int compressed_footer_units, int swa_width, int compressed_width, int* __restrict__ extra_topk_lens)
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
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 48)
    #define v_empty_addr (mbar_base + 64)
    #define nvfp4_sf_full_addr (mbar_base + 80)
    #define nvfp4_kexp_full_addr (mbar_base + 96)
    #define nvfp4_v_ready_addr (mbar_base + 112)
    #define nvfp4_q_rope_full_addr (mbar_base + 128)
    #define nvfp4_q_lut_ready_addr (mbar_base + 136)
    #define nvfp4_q_sf_ready_addr (mbar_base + 144)
    #define nvfp4_q_ring_full_addr (mbar_base + 152)
    #define nvfp4_q_ring_empty_addr (mbar_base + 168)
    #define nvfp4_q_kslot_full_addr (mbar_base + 184)
    #define index_full_addr (mbar_base + 208)
    #define index_empty_addr (mbar_base + 256)
    #define s_full_addr (mbar_base + 304)
    #define p_full_addr (mbar_base + 320)
    #define split_pv_p_empty_addr (mbar_base + 336)
    #define s_empty_addr (mbar_base + 352)
    #define stats_addr (mbar_base + 368)
    #define sum_ready_addr (mbar_base + 384)
    #define source_sum_empty_addr (mbar_base + 392)
    #define o_empty_addr (mbar_base + 400)
    #define o_full_addr (mbar_base + 408)
    #define s_seeded_addr (mbar_base + 416)
    #define q_pair_ready_addr (mbar_base + 424)
    #define kv_pair_ready_addr (mbar_base + 432)
    #define pv_pair_ready_addr (mbar_base + 440)
    #define tmem_dealloc_addr (mbar_base + 456)
    #define tmem_dealloc_peer_addr (mbar_base + 464)
    #define source_work_full_addr (mbar_base + 472)
    #define source_work_empty_addr (mbar_base + 488)
    #define source_throttle_full_addr (mbar_base + 504)
    #define source_throttle_empty_addr (mbar_base + 520)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_kv_addr = smem + 27648;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_v_addr = smem + 27648;
    uint8_t* smem_q_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_a_addr = smem + 1024;
    uint8_t* smem_q_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_q_b_addr = smem + 9216;
    __nv_bfloat16* smem_q_rope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_rope_addr = smem + 17408;
    unsigned int* smem_q_sf = reinterpret_cast<unsigned int*>(smem_raw + 25600);
    const int smem_q_sf_addr = smem + 25600;
    uint8_t* smem_k_a = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_k_a_addr = smem + 27648;
    uint8_t* smem_k_b = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_k_b_addr = smem + 35840;
    __nv_bfloat16* smem_k_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 44032);
    const int smem_k_c_addr = smem + 44032;
    unsigned int* smem_k_sf = reinterpret_cast<unsigned int*>(smem_raw + 52224);
    const int smem_k_sf_addr = smem + 52224;
    uint8_t* smem_v_full = reinterpret_cast<uint8_t*>(smem_raw + 84992);
    const int smem_v_full_addr = smem + 84992;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 150528);
    const int smem_stats_max_addr = smem + 150528;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 151552);
    const int smem_stats_sum_addr = smem + 151552;
    float* smem_stats_final_max = reinterpret_cast<float*>(smem_raw + 152064);
    const int smem_stats_final_max_addr = smem + 152064;
    float* smem_softmax_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 152576);
    const int smem_softmax_warp_pair_exchange_addr = smem + 152576;
    float* smem_corr_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 153600);
    const int smem_corr_warp_pair_exchange_addr = smem + 153600;
    unsigned int* smem_q_ring_pad = reinterpret_cast<unsigned int*>(smem_raw + 176736);
    const int smem_q_ring_pad_addr = smem + 176736;
    __nv_bfloat16* smem_q_ring = reinterpret_cast<__nv_bfloat16*>(smem_raw + 177152);
    const int smem_q_ring_addr = smem + 177152;
    float* smem_q_lut6 = reinterpret_cast<float*>(smem_raw + 193536);
    const int smem_q_lut6_addr = smem + 193536;
    float* smem_q_lutinv = reinterpret_cast<float*>(smem_raw + 194560);
    const int smem_q_lutinv_addr = smem + 194560;
    int* smem_sparse_indices = reinterpret_cast<int*>(smem_raw + 154112);
    const int smem_sparse_indices_addr = smem + 154112;
    uint8_t* smem_p_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 160256);
    const int smem_p_fp8_addr = smem + 160256;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 176640);
    const int work_response_addr = smem + 176640;
    unsigned int* smem_kexp = reinterpret_cast<unsigned int*>(smem_raw + 176672);
    const int smem_kexp_addr = smem + 176672;
    float* smem_kmax = reinterpret_cast<float*>(smem_raw + 176704);
    const int smem_kmax_addr = smem + 176704;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (36 pipeline groups, 0 ordered-sequence groups, 67 barriers)
    // Mbarriers at smem_raw[0..536)

    if (warp == 12) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'k_pipe' ---
            // k_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'k_pipe' ---
            // nvfp4_sf_full: 2 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // nvfp4_kexp_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // nvfp4_v_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // nvfp4_q_rope_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // nvfp4_q_lut_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            // nvfp4_q_sf_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            // nvfp4_q_ring_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // nvfp4_q_ring_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // nvfp4_q_kslot_full: 3 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'index_pipe' ---
            // index_full: 6 barriers, init_count=32
            mbarrier_init(smem + 208, 32);
            mbarrier_init(smem + 216, 32);
            mbarrier_init(smem + 224, 32);
            mbarrier_init(smem + 232, 32);
            mbarrier_init(smem + 240, 32);
            mbarrier_init(smem + 248, 32);
            // index_empty: 6 barriers, init_count=256
            mbarrier_init(smem + 256, 256);
            mbarrier_init(smem + 264, 256);
            mbarrier_init(smem + 272, 256);
            mbarrier_init(smem + 280, 256);
            mbarrier_init(smem + 288, 256);
            mbarrier_init(smem + 296, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 320, 256);
            mbarrier_init(smem + 328, 256);
            // split_pv_p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 352, 256);
            mbarrier_init(smem + 360, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 5) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 368, 128);
            mbarrier_init(smem + 376, 128);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 384, 128);
            // source_sum_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 392, 128);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 464, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 4) {
        uint32_t leader = elect_sync();
        if (leader) {
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 400, 256);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 408, 1);
            // pv_pair_ready: stages (1,), init_count=64
            mbarrier_init(smem + 448, 64);
            // tmem_dealloc: 1 barriers, init_count=448
            mbarrier_init(smem + 456, 448);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // s_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 416, 256);
            // q_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 424, 64);
            // --- pipeline 'kv_pipe' ---
            // kv_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 432, 64);
            // pv_pair_ready: stages (0,), init_count=64
            mbarrier_init(smem + 440, 64);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 10) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'source_work_pipe' ---
            // source_work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 472, 1);
            mbarrier_init(smem + 480, 1);
            // source_work_empty: 2 barriers, init_count=960
            mbarrier_init(smem + 488, 960);
            mbarrier_init(smem + 496, 960);
            // --- pipeline 'source_throttle_pipe' ---
            // source_throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 504, 128);
            mbarrier_init(smem + 512, 128);
            // source_throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 520, 32);
            mbarrier_init(smem + 528, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 536);
    if (warp == 0) {
        int _tmem_hold = smem + 536;
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
    const int tmem_tmem_scratch = taddr;
    const int tmem_tmem_sfa = taddr + 64;
    const int tmem_tmem_sfb = taddr + 96;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: index_warp ----
    if (warp == 9) {
        { // index_warp_main
            const int index_dummy = 0;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_source_work_full = 0;
            unsigned int _phase_q_full_0 = 0;
            {
                int all_num_kv_tiles = (sparse_topk + 128 - 1) / 128;
                unsigned int index_prod_stage = 0;
                unsigned int source_work_stage = 0;
                unsigned int source_work_x = blockIdx.x;
                unsigned int source_work_z = blockIdx.z;
                #pragma unroll 1
                for (unsigned int static_work = 0; static_work < 514; static_work++) {
                    int split_idx = 0;
                    int query_idx = source_work_x >> 1;
                    int source_work_valid = source_work_z * (unsigned int)max_q_len + (source_work_x >> 1) < (unsigned int)num_query_tokens;
                    int mapped_query_idx = source_work_z * (unsigned int)max_q_len + (source_work_x >> 1);
                    if (source_work_valid != 0) {
                        int _max_0 = ((sparse_topk_lens[mapped_query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx] + sparse_topk_lens_offset) : (0));
                        int _min_0 = ((_max_0) < (swa_width) ? (_max_0) : (swa_width));
                        int main_active = _min_0;
                        int extra_active = 0;
                        if (compressed_width > 0) {
                            int _max_1 = ((extra_topk_lens[mapped_query_idx]) > (0) ? (extra_topk_lens[mapped_query_idx]) : (0));
                            int _min_1 = ((_max_1) < (compressed_width) ? (_max_1) : (compressed_width));
                            extra_active = _min_1;
                        }
                        int n_main_tiles = (main_active + 128 - 1) / 128;
                        int n_extra_tiles = (extra_active + 128 - 1) / 128;
                        int _max_2 = ((n_main_tiles + n_extra_tiles) > (1) ? (n_main_tiles + n_extra_tiles) : (1));
                        int active_topk = _max_2 * 128;
                        all_num_kv_tiles = (active_topk + 128 - 1) / 128;
                        int tiles_per_split = all_num_kv_tiles + 1 - 1;
                        int first_tile = split_idx * tiles_per_split;
                        int num_index_passes = (tiles_per_split + 1) / 2;
                        #pragma unroll 1
                        for (int index_pass = 0; index_pass < num_index_passes; index_pass++) {
                            mbarrier_wait(index_empty_addr + (index_prod_stage) * 8, _phase_index_empty);
                            int index_stage_base = smem_sparse_indices_addr + index_prod_stage * 1024;
                            int stage_row_base = index_pass * 256;
                            #pragma unroll
                            for (int index_half = 0; index_half < 2; index_half++) {
                                int index_offset = index_half * 128 + lane * 4;
                                int stage_tile = index_pass * 2 + index_half;
                                int seg_in_main = stage_tile < n_main_tiles || n_extra_tiles == 0;
                                int seg_tile = ((seg_in_main != 0) ? stage_tile : stage_tile - n_main_tiles);
                                int seg_width = ((seg_in_main != 0) ? swa_width : compressed_width);
                                int tile_col_base = seg_tile * 128;
                                int* row_tile_ptr = ((seg_in_main != 0) ? (swa_indices + (mapped_query_idx * swa_index_stride + tile_col_base)) : (compressed_indices + (mapped_query_idx * compressed_index_stride + tile_col_base)));
                                int lane_col = tile_col_base + lane * 4;
                                if (seg_width >= lane_col + 4) {
                                    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;"
                                        :: "r"(index_stage_base + index_offset * 4), "l"(row_tile_ptr + (lane * 4)));
                                } else {
                                    int tail_rows[4];
                                    int tail_fill = -1;
                                    if (tile_col_base < seg_width) {
                                        tail_fill = row_tile_ptr[0];
                                    }
                                    #pragma unroll
                                    for (int row_i = 0; row_i < 4; row_i++) {
                                        tail_rows[row_i] = tail_fill;
                                    }
                                    #pragma unroll
                                    for (int row_i_1 = 0; row_i_1 < 4; row_i_1++) {
                                        if (seg_width > lane_col + row_i_1) {
                                            tail_rows[row_i_1] = row_tile_ptr[lane * 4 + row_i_1];
                                        }
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sparse_indices_addr + ((unsigned int)index_stage_base - smem_sparse_indices_addr + (unsigned int)(index_offset * 4))), "r"(tail_rows[0]), "r"(tail_rows[1]), "r"(tail_rows[2]), "r"(tail_rows[3]) : "memory");
                                    {
                                        asm volatile("fence.release.cta;" ::: "memory");
                                    }
                                }
                            }
                            asm volatile(
                                "{\n\t"
                                "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                                "}"
                                :: "r"(index_full_addr + (index_prod_stage) * 8) : "memory");
                            index_prod_stage += 1;
                            if (index_prod_stage == 6) { index_prod_stage = 0; _phase_index_empty ^= 1; }
                        }
                    }
                    {
                        mbarrier_wait(source_work_full_addr + (source_work_stage) * 8, _phase_source_work_full);
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
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
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
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_1 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_1)
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_2 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_2)
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(source_work_empty_addr + source_work_stage * 8), "r"(0) : "memory");
                        source_work_stage += 1;
                        if (source_work_stage == 2) { source_work_stage = 0; _phase_source_work_full ^= 1; }
                        if (_clc_valid_0 == 0) {
                            break;
                        }
                        source_work_x = _clc_ctaid_0 + (unsigned int)cta_rank;
                        source_work_z = _clc_ctaid_2;
                    }
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 120;");
        { // load_warp_main
            const int wg2_dummy = 0;
            const int load_warp_rank = warp - 12;
            const int tmem_row_base = load_warp_rank * 32;
            int peer_rank = bid % 2 ^ 1;
            int k_cta_offset = bid % 2 * 64;
            const int v_half = load_warp_rank >> 1;
            const int v_token = (load_warp_rank & 1) * 32 + lane;
            int v_row = k_cta_offset + v_token;
            const int q_lane128 = load_warp_rank * 32 + lane;
            const int q_row = q_lane128 >> 1;
            const int q_pair = q_lane128 & 1;
            const int q_swz = q_row & 7;
            unsigned int q_ring_seq = 0;
            float _fdiv_rn_0 = __fdiv_rn((float)q_lane128, 6.0f);
            smem_q_lut6[q_lane128] = _fdiv_rn_0;
            float _fdiv_rn_1 = __fdiv_rn((float)(q_lane128 + 128), 6.0f);
            smem_q_lut6[q_lane128 + 128] = _fdiv_rn_1;
            uint16_t _e4m3x2_decode_0 = (uint16_t)((unsigned int)q_lane128 & 0xFFu);
            uint32_t _f16x2_decode_0;
            float _fp8_decode_0;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_0) : "h"(_e4m3x2_decode_0));
            uint16_t _f16_decode_0 = (uint16_t)_f16x2_decode_0;
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_0) : "h"(_f16_decode_0));
            float q_lut_s = _fp8_decode_0;
            float q_lut_inv = 0.0f;
            if (q_lane128 != 0) {
                float _fdiv_rn_2 = __fdiv_rn(1.0f, q_lut_s);
                q_lut_inv = _fdiv_rn_2;
            }
            smem_q_lutinv[q_lane128] = q_lut_inv;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_arrive(nvfp4_q_lut_ready_addr);
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            const int sf_tok_a = (load_warp_rank >> 1) * 64 + lane;
            const int sf_tok_b = sf_tok_a + 32;
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(smem_v_full_addr), "r"(peer_rank));
            unsigned int peer_v_base = _mapa_0;
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(nvfp4_v_ready_addr + 0 * 8), "r"(peer_rank));
            unsigned int peer_v_ready_0 = _mapa_1;
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(nvfp4_v_ready_addr + 1 * 8), "r"(peer_rank));
            unsigned int peer_v_ready_1 = _mapa_2;
            int tile_cursor = 0;
            int item_cursor = 0;
            unsigned int load_k_index_stage = 0;
            unsigned int source_work_stage_1 = 0;
            unsigned int source_throttle_stage = 0;
            unsigned int source_work_x_1 = blockIdx.x;
            unsigned int source_work_z_1 = blockIdx.z;
            int source_work_valid_1 = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1) < (unsigned int)num_query_tokens;
            int mapped_query_idx_1 = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
            int main_active_1 = 0;
            int extra_active_1 = 0;
            int n_main_tiles_1 = 0;
            int n_extra_tiles_1 = 0;
            int num_kv_tiles = 1;
            unsigned int _phase_index_full = 0;
            if (source_work_valid_1 != 0) {
                int _max_3 = ((sparse_topk_lens[mapped_query_idx_1] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_1] + sparse_topk_lens_offset) : (0));
                int _min_2 = ((_max_3) < (swa_width) ? (_max_3) : (swa_width));
                main_active_1 = _min_2;
                if (compressed_width > 0) {
                    int _max_4 = ((extra_topk_lens[mapped_query_idx_1]) > (0) ? (extra_topk_lens[mapped_query_idx_1]) : (0));
                    int _min_3 = ((_max_4) < (compressed_width) ? (_max_4) : (compressed_width));
                    extra_active_1 = _min_3;
                }
                n_main_tiles_1 = (main_active_1 + 128 - 1) / 128;
                n_extra_tiles_1 = (extra_active_1 + 128 - 1) / 128;
                int _max_5 = ((n_main_tiles_1 + n_extra_tiles_1) > (1) ? (n_main_tiles_1 + n_extra_tiles_1) : (1));
                num_kv_tiles = _max_5;
                int g_stage = 0;
                {
                    mbarrier_wait(index_full_addr + (load_k_index_stage) * 8, _phase_index_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                mbarrier_wait(k_empty_addr + (g_stage) * 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (load_warp_rank == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (g_stage) * 8, 28672);
                    }
                }
                int g_group = lane;
                if (g_group < 16) {
                    int g_rows[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&g_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows[(0) + 3]))
                        : "r"(smem_sparse_indices_addr + load_k_index_stage * 1024 + (unsigned int)((k_cta_offset + g_group * 4) * 4)));
                    int g_use_swa = n_main_tiles_1 > 0 || n_extra_tiles_1 == 0;
                    int g_page_log2 = ((g_use_swa != 0) ? swa_page_log2 : compressed_page_log2);
                    int g_pitch_units = ((g_use_swa != 0) ? swa_pitch_units : compressed_pitch_units);
                    int g_footer_units = ((g_use_swa != 0) ? swa_footer_units : compressed_footer_units);
                    int g_data_rows[4];
                    int g_foot_rows[4];
                    #pragma unroll
                    for (int g_j = 0; g_j < 4; g_j++) {
                        int g_tok = ((g_rows[g_j] >= 0) ? g_rows[g_j] : 0);
                        int g_page = g_tok >> g_page_log2;
                        int g_slot = g_tok - (g_page << g_page_log2);
                        g_data_rows[g_j] = g_page * g_pitch_units + g_slot * 11;
                        g_foot_rows[g_j] = g_page * g_pitch_units + g_footer_units + g_slot;
                    }
                    int g_stage_base = smem_k_a_addr + (unsigned int)(g_stage * 28672);
                    if (load_warp_rank < 3) {
                        int g_col = ((load_warp_rank == 2) ? 224 : load_warp_rank * 128);
                        int g_dst = g_stage_base + load_warp_rank * 8192 + g_group * 512;
                        if (g_use_swa != 0) {
                            tma_gather4_gmem2smem(g_dst, (&tmap_swa_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + (g_stage) * 8);
                        } else {
                            tma_gather4_gmem2smem(g_dst, (&tmap_compressed_kv), g_col, g_data_rows[0], g_data_rows[1], g_data_rows[2], g_data_rows[3], k_full_addr + (g_stage) * 8);
                        }
                    } else {
                        int g_fdst = g_stage_base + 24576 + (k_cta_offset + g_group * 4) * 32;
                        if (g_use_swa != 0) {
                            tma_gather4_gmem2smem_mc(g_fdst, (&tmap_swa_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + (g_stage) * 8, 3);
                        } else {
                            tma_gather4_gmem2smem_mc(g_fdst, (&tmap_compressed_sf), 0, g_foot_rows[0], g_foot_rows[1], g_foot_rows[2], g_foot_rows[3], k_full_addr + (g_stage) * 8, 3);
                        }
                    }
                }
                if (1 == num_kv_tiles) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(index_empty_addr + (load_k_index_stage) * 8);
                    load_k_index_stage += 1;
                    if (load_k_index_stage == 6) { load_k_index_stage = 0; _phase_index_full ^= 1; }
                }
            }
            int has_next = 1;
            int next_valid = 0;
            int next_query_idx = 0;
            int next_main_active = 0;
            int next_extra_active = 0;
            int next_n_main_tiles = 0;
            int next_n_extra_tiles = 0;
            int next_num_kv_tiles = 1;
            unsigned int _phase_source_throttle_empty = 1;
            unsigned int _phase_source_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = 0; work_idx < 514; work_idx++) {
                if (bid % 2 == 0) {
                    mbarrier_wait(source_throttle_empty_addr + (source_throttle_stage) * 8, _phase_source_throttle_empty);
                    mbarrier_arrive(source_throttle_full_addr + (source_throttle_stage) * 8);
                    source_throttle_stage += 1;
                    if (source_throttle_stage == 2) { source_throttle_stage = 0; _phase_source_throttle_empty ^= 1; }
                }
                if (source_work_valid_1 != 0) {
                    int item_tile_base = tile_cursor;
                    int ld_rec = item_cursor;
                    int ld_on = item_cursor < 48;
                    mbarrier_wait(q_empty_addr, item_cursor & 1 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int q_kstage = item_tile_base + 1 & 1;
                    mbarrier_wait(k_empty_addr + (q_kstage) * 8, item_tile_base + 1 >> 1 & 1 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int q_kbase = smem_k_a_addr + (unsigned int)(q_kstage * 28672);
                    if (load_warp_rank == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(nvfp4_q_rope_full_addr, 8192);
                            tma_4d_gmem2smem(smem_q_rope_addr, (&tmap_q), 0, bid % 2 * 64, 7, mapped_query_idx_1, nvfp4_q_rope_full_addr);
                            #pragma unroll
                            for (int q_k = 0; q_k < 3; q_k++) {
                                mbarrier_arrive_expect_tx(nvfp4_q_kslot_full_addr + (q_k) * 8, 8192);
                                tma_4d_gmem2smem(q_kbase + q_k * 8192, (&tmap_q), 0, bid % 2 * 64, 2 + q_k, mapped_query_idx_1, nvfp4_q_kslot_full_addr + (q_k) * 8);
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int q_c = 2; q_c < 5; q_c++) {
                        mbarrier_wait(nvfp4_q_kslot_full_addr + (q_c - 2) * 8, item_cursor & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int q_in[16];
                        #pragma unroll
                        for (int q_u = 0; q_u < 4; q_u++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_in[4 * q_u])), "=r"(*reinterpret_cast<uint32_t*>(&q_in[(4 * q_u) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_in[(4 * q_u) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_in[(4 * q_u) + 3]))
                                : "r"(q_kbase + (q_c - 2) * 8192 + q_row * 128 + ((q_pair * 4 + q_u ^ q_swz) << 4)));
                        }
                        unsigned int q_codes[4];
                        unsigned int q_sf_pair = 0;
                        #pragma unroll
                        for (int q_g = 0; q_g < 2; q_g++) {
                            float q_vals[16];
                            #pragma unroll
                            for (int q_w = 0; q_w < 8; q_w++) {
                                unsigned int q_word = q_in[8 * q_g + q_w];
                                float _cvt_f32_bf16_0;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(q_word & 65535)));
                                q_vals[2 * q_w] = _cvt_f32_bf16_0;
                                float _cvt_f32_bf16_1;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(q_word >> 16)));
                                q_vals[2 * q_w + 1] = _cvt_f32_bf16_1;
                            }
                            float _fabs_0 = fabsf(q_vals[0]);
                            float q_amax = _fabs_0;
                            #pragma unroll
                            for (int q_i = 1; q_i < 16; q_i++) {
                                float _fabs_1 = fabsf(q_vals[q_i]);
                                float _max_6 = max_noftz(q_amax, _fabs_1);
                                q_amax = _max_6;
                            }
                            unsigned int q_amax_bits = __as_u32(q_amax);
                            int q_m = (int)(q_amax_bits >> 16 & 127 | 128);
                            int q_e = (int)(q_amax_bits >> 23);
                            int _max_7 = ((q_e) > (7) ? (q_e) : (7));
                            unsigned int q_pow_bits = (unsigned int)(_max_7 - 7 << 23);
                            float q_div6 = smem_q_lut6[q_m] * __uint_as_float(q_pow_bits);
                            uint8_t _fp8_encode_raw_0;
                            float _fp8_encode_decoded_0;
                            uint16_t _e4m3x2_encode_1;
                            uint32_t _f16x2_encode_1;
                            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_encode_1) : "f"(0.0f), "f"(q_div6));
                            _fp8_encode_raw_0 = (uint8_t)(_e4m3x2_encode_1 & 0x00ffu);
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_encode_1) : "h"(_e4m3x2_encode_1));
                            uint16_t _fp8_encode_h0_1 = (uint16_t)(_f16x2_encode_1 & 0xffffu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_encode_decoded_0) : "h"(_fp8_encode_h0_1));
                            float q_sf_inv = smem_q_lutinv[(int)_fp8_encode_raw_0];
                            #pragma unroll
                            for (int q_w_1 = 0; q_w_1 < 2; q_w_1++) {
                                uint32_t _fp4_pair_0;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_0) : "f"(q_vals[8 * q_w_1] * q_sf_inv), "f"(q_vals[8 * q_w_1 + 1] * q_sf_inv));
                                unsigned int q_b0 = _fp4_pair_0;
                                uint32_t _fp4_pair_1;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_1) : "f"(q_vals[8 * q_w_1 + 2] * q_sf_inv), "f"(q_vals[8 * q_w_1 + 3] * q_sf_inv));
                                unsigned int q_b1 = _fp4_pair_1;
                                uint32_t _fp4_pair_2;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_2) : "f"(q_vals[8 * q_w_1 + 4] * q_sf_inv), "f"(q_vals[8 * q_w_1 + 5] * q_sf_inv));
                                unsigned int q_b2 = _fp4_pair_2;
                                uint32_t _fp4_pair_3;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_3) : "f"(q_vals[8 * q_w_1 + 6] * q_sf_inv), "f"(q_vals[8 * q_w_1 + 7] * q_sf_inv));
                                unsigned int q_b3 = _fp4_pair_3;
                                q_codes[2 * q_g + q_w_1] = q_b0 | q_b1 << 8 | q_b2 << 16 | q_b3 << 24;
                            }
                            q_sf_pair = q_sf_pair | (unsigned int)_fp8_encode_raw_0 << (unsigned int)(q_g * 8);
                        }
                        int q_dst = (q_c >> 2) * 8192 + q_row * 128 + ((2 * (q_c & 3) + q_pair ^ q_swz) << 4);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_q_a_addr + (unsigned int)q_dst), "r"(q_codes[0]), "r"(q_codes[1]), "r"(q_codes[2]), "r"(q_codes[3]) : "memory");
                        unsigned int q_sf_mine = q_sf_pair << (unsigned int)(q_pair * 16);
                        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sf_mine, 1);
                        unsigned int q_sf_other = _shfl_xor_0;
                        if (q_pair == 0) {
                            smem_q_sf[q_row * 8 + q_c] = q_sf_mine | q_sf_other;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                    mbarrier_arrive(nvfp4_q_sf_ready_addr);
                    #pragma unroll 1
                    for (int tile = 0; tile < num_kv_tiles; tile++) {
                        if (num_kv_tiles > tile + 1) {
                            int g_stage_1 = item_tile_base + tile + 1 & 1;
                            if ((tile + 1 & 1) == 0) {
                                mbarrier_wait(index_full_addr + (load_k_index_stage) * 8, _phase_index_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            mbarrier_wait(k_empty_addr + (g_stage_1) * 8, item_tile_base + tile + 1 >> 1 & 1 ^ 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            if (load_warp_rank == 0) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(k_full_addr + (g_stage_1) * 8, 28672);
                                }
                            }
                            int g_group_1 = lane;
                            if (g_group_1 < 16) {
                                int g_rows_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_1[(0) + 3]))
                                    : "r"(smem_sparse_indices_addr + load_k_index_stage * 1024 + (unsigned int)(((tile + 1 & 1) * 128 + k_cta_offset + g_group_1 * 4) * 4)));
                                int g_use_swa_1 = n_main_tiles_1 > tile + 1 || n_extra_tiles_1 == 0;
                                int g_page_log2_1 = ((g_use_swa_1 != 0) ? swa_page_log2 : compressed_page_log2);
                                int g_pitch_units_1 = ((g_use_swa_1 != 0) ? swa_pitch_units : compressed_pitch_units);
                                int g_footer_units_1 = ((g_use_swa_1 != 0) ? swa_footer_units : compressed_footer_units);
                                int g_data_rows_1[4];
                                int g_foot_rows_1[4];
                                #pragma unroll
                                for (int g_j_1 = 0; g_j_1 < 4; g_j_1++) {
                                    int g_tok_1 = ((g_rows_1[g_j_1] >= 0) ? g_rows_1[g_j_1] : 0);
                                    int g_page_1 = g_tok_1 >> g_page_log2_1;
                                    int g_slot_1 = g_tok_1 - (g_page_1 << g_page_log2_1);
                                    g_data_rows_1[g_j_1] = g_page_1 * g_pitch_units_1 + g_slot_1 * 11;
                                    g_foot_rows_1[g_j_1] = g_page_1 * g_pitch_units_1 + g_footer_units_1 + g_slot_1;
                                }
                                int g_stage_base_1 = smem_k_a_addr + (unsigned int)(g_stage_1 * 28672);
                                if (load_warp_rank < 3) {
                                    int g_col_1 = ((load_warp_rank == 2) ? 224 : load_warp_rank * 128);
                                    int g_dst_1 = g_stage_base_1 + load_warp_rank * 8192 + g_group_1 * 512;
                                    if (g_use_swa_1 != 0) {
                                        tma_gather4_gmem2smem(g_dst_1, (&tmap_swa_kv), g_col_1, g_data_rows_1[0], g_data_rows_1[1], g_data_rows_1[2], g_data_rows_1[3], k_full_addr + (g_stage_1) * 8);
                                    } else {
                                        tma_gather4_gmem2smem(g_dst_1, (&tmap_compressed_kv), g_col_1, g_data_rows_1[0], g_data_rows_1[1], g_data_rows_1[2], g_data_rows_1[3], k_full_addr + (g_stage_1) * 8);
                                    }
                                } else {
                                    int g_fdst_1 = g_stage_base_1 + 24576 + (k_cta_offset + g_group_1 * 4) * 32;
                                    if (g_use_swa_1 != 0) {
                                        tma_gather4_gmem2smem_mc(g_fdst_1, (&tmap_swa_sf), 0, g_foot_rows_1[0], g_foot_rows_1[1], g_foot_rows_1[2], g_foot_rows_1[3], k_full_addr + (g_stage_1) * 8, 3);
                                    } else {
                                        tma_gather4_gmem2smem_mc(g_fdst_1, (&tmap_compressed_sf), 0, g_foot_rows_1[0], g_foot_rows_1[1], g_foot_rows_1[2], g_foot_rows_1[3], k_full_addr + (g_stage_1) * 8, 3);
                                    }
                                }
                            }
                            if ((tile + 1 & 1) != 0 || tile + 1 + 1 == num_kv_tiles) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(index_empty_addr + (load_k_index_stage) * 8);
                                load_k_index_stage += 1;
                                if (load_k_index_stage == 6) { load_k_index_stage = 0; _phase_index_full ^= 1; }
                            }
                        } else {
                            mbarrier_wait(source_work_full_addr + (source_work_stage_1) * 8, _phase_source_work_full_1);
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
                                : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                                : "memory");
                            uint32_t _clc_ctaid_3 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_3)
                                : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                                : "memory");
                            uint32_t _clc_ctaid_4 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_4)
                                : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                                : "memory");
                            uint32_t _clc_ctaid_5 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_5)
                                : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                                : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(source_work_empty_addr + source_work_stage_1 * 8), "r"(0) : "memory");
                            source_work_stage_1 += 1;
                            if (source_work_stage_1 == 2) { source_work_stage_1 = 0; _phase_source_work_full_1 ^= 1; }
                            has_next = 0;
                            next_valid = 0;
                            if (_clc_valid_1 != 0) {
                                has_next = 1;
                                source_work_x_1 = _clc_ctaid_3 + (unsigned int)cta_rank;
                                source_work_z_1 = _clc_ctaid_5;
                                next_query_idx = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
                                next_valid = next_query_idx < num_query_tokens;
                                if (next_valid != 0) {
                                    int _max_8 = ((sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) : (0));
                                    int _min_4 = ((_max_8) < (swa_width) ? (_max_8) : (swa_width));
                                    next_main_active = _min_4;
                                    next_extra_active = 0;
                                    if (compressed_width > 0) {
                                        int _max_9 = ((extra_topk_lens[next_query_idx]) > (0) ? (extra_topk_lens[next_query_idx]) : (0));
                                        int _min_5 = ((_max_9) < (compressed_width) ? (_max_9) : (compressed_width));
                                        next_extra_active = _min_5;
                                    }
                                    next_n_main_tiles = (next_main_active + 128 - 1) / 128;
                                    next_n_extra_tiles = (next_extra_active + 128 - 1) / 128;
                                    int _max_10 = ((next_n_main_tiles + next_n_extra_tiles) > (1) ? (next_n_main_tiles + next_n_extra_tiles) : (1));
                                    next_num_kv_tiles = _max_10;
                                    int g_stage_2 = item_tile_base + num_kv_tiles & 1;
                                    {
                                        mbarrier_wait(index_full_addr + (load_k_index_stage) * 8, _phase_index_full);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                    }
                                    mbarrier_wait(k_empty_addr + (g_stage_2) * 8, item_tile_base + num_kv_tiles >> 1 & 1 ^ 1);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    if (load_warp_rank == 0) {
                                        if (elect_sync()) {
                                            mbarrier_arrive_expect_tx(k_full_addr + (g_stage_2) * 8, 28672);
                                        }
                                    }
                                    int g_group_2 = lane;
                                    if (g_group_2 < 16) {
                                        int g_rows_2[4];
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&g_rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_2[(0) + 3]))
                                            : "r"(smem_sparse_indices_addr + load_k_index_stage * 1024 + (unsigned int)((k_cta_offset + g_group_2 * 4) * 4)));
                                        int g_use_swa_2 = next_n_main_tiles > 0 || next_n_extra_tiles == 0;
                                        int g_page_log2_2 = ((g_use_swa_2 != 0) ? swa_page_log2 : compressed_page_log2);
                                        int g_pitch_units_2 = ((g_use_swa_2 != 0) ? swa_pitch_units : compressed_pitch_units);
                                        int g_footer_units_2 = ((g_use_swa_2 != 0) ? swa_footer_units : compressed_footer_units);
                                        int g_data_rows_2[4];
                                        int g_foot_rows_2[4];
                                        #pragma unroll
                                        for (int g_j_2 = 0; g_j_2 < 4; g_j_2++) {
                                            int g_tok_2 = ((g_rows_2[g_j_2] >= 0) ? g_rows_2[g_j_2] : 0);
                                            int g_page_2 = g_tok_2 >> g_page_log2_2;
                                            int g_slot_2 = g_tok_2 - (g_page_2 << g_page_log2_2);
                                            g_data_rows_2[g_j_2] = g_page_2 * g_pitch_units_2 + g_slot_2 * 11;
                                            g_foot_rows_2[g_j_2] = g_page_2 * g_pitch_units_2 + g_footer_units_2 + g_slot_2;
                                        }
                                        int g_stage_base_2 = smem_k_a_addr + (unsigned int)(g_stage_2 * 28672);
                                        if (load_warp_rank < 3) {
                                            int g_col_2 = ((load_warp_rank == 2) ? 224 : load_warp_rank * 128);
                                            int g_dst_2 = g_stage_base_2 + load_warp_rank * 8192 + g_group_2 * 512;
                                            if (g_use_swa_2 != 0) {
                                                tma_gather4_gmem2smem(g_dst_2, (&tmap_swa_kv), g_col_2, g_data_rows_2[0], g_data_rows_2[1], g_data_rows_2[2], g_data_rows_2[3], k_full_addr + (g_stage_2) * 8);
                                            } else {
                                                tma_gather4_gmem2smem(g_dst_2, (&tmap_compressed_kv), g_col_2, g_data_rows_2[0], g_data_rows_2[1], g_data_rows_2[2], g_data_rows_2[3], k_full_addr + (g_stage_2) * 8);
                                            }
                                        } else {
                                            int g_fdst_2 = g_stage_base_2 + 24576 + (k_cta_offset + g_group_2 * 4) * 32;
                                            if (g_use_swa_2 != 0) {
                                                tma_gather4_gmem2smem_mc(g_fdst_2, (&tmap_swa_sf), 0, g_foot_rows_2[0], g_foot_rows_2[1], g_foot_rows_2[2], g_foot_rows_2[3], k_full_addr + (g_stage_2) * 8, 3);
                                            } else {
                                                tma_gather4_gmem2smem_mc(g_fdst_2, (&tmap_compressed_sf), 0, g_foot_rows_2[0], g_foot_rows_2[1], g_foot_rows_2[2], g_foot_rows_2[3], k_full_addr + (g_stage_2) * 8, 3);
                                            }
                                        }
                                    }
                                    if (1 == next_num_kv_tiles) {
                                        asm volatile("tcgen05.fence::before_thread_sync;");
                                        mbarrier_arrive(index_empty_addr + (load_k_index_stage) * 8);
                                        load_k_index_stage += 1;
                                        if (load_k_index_stage == 6) { load_k_index_stage = 0; _phase_index_full ^= 1; }
                                    }
                                }
                            }
                        }
                        int pt = item_tile_base + tile;
                        int p_stage = pt & 1;
                        int p_phase = pt >> 1 & 1;
                        mbarrier_wait(k_full_addr + (p_stage) * 8, p_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int ld_t0 = ld_on & (int)(tile == 0);
                        int sf_base = smem_k_a_addr + (unsigned int)(p_stage * 28672) + 24576;
                        unsigned int sf_a[8];
                        unsigned int sf_b[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(0) + 3]))
                            : "r"(sf_base + sf_tok_a * 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_a[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_a[(4) + 3]))
                            : "r"(sf_base + sf_tok_a * 32 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[0])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(0) + 3]))
                            : "r"(sf_base + sf_tok_b * 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sf_b[4])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sf_b[(4) + 3]))
                            : "r"(sf_base + sf_tok_b * 32 + 16));
                        unsigned int e_pair = 0;
                        #pragma unroll
                        for (int sf_s = 0; sf_s < 7; sf_s++) {
                            uint32_t _e4m3x2_to_f16x2_0;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(sf_a[sf_s] & 65535)));
                            uint32_t _f16x2_max_0;
                            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_0) : "r"(e_pair), "r"(_e4m3x2_to_f16x2_0));
                            e_pair = _f16x2_max_0;
                            uint32_t _e4m3x2_to_f16x2_1;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(sf_a[sf_s] >> 16)));
                            uint32_t _f16x2_max_1;
                            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_1) : "r"(e_pair), "r"(_e4m3x2_to_f16x2_1));
                            e_pair = _f16x2_max_1;
                            uint32_t _e4m3x2_to_f16x2_2;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(sf_b[sf_s] & 65535)));
                            uint32_t _f16x2_max_2;
                            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_2) : "r"(e_pair), "r"(_e4m3x2_to_f16x2_2));
                            e_pair = _f16x2_max_2;
                            uint32_t _e4m3x2_to_f16x2_3;
                            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(sf_b[sf_s] >> 16)));
                            uint32_t _f16x2_max_3;
                            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_3) : "r"(e_pair), "r"(_e4m3x2_to_f16x2_3));
                            e_pair = _f16x2_max_3;
                        }
                        uint16_t _f16_max_0;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_0) : "h"((uint16_t)(e_pair & 65535)), "h"((uint16_t)(e_pair >> 16)));
                        float _cvt_f32_f16_0;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_f16_max_0)));
                        float e_max = _cvt_f32_f16_0;
                        float _warp_reduce_0 = e_max;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                        e_max = _warp_reduce_0;
                        if (elect_sync()) {
                            smem_kmax[p_stage * 4 + load_warp_rank] = e_max;
                        }
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        float _max_11 = max_noftz(smem_kmax[p_stage * 4], smem_kmax[p_stage * 4 + 1]);
                        float _max_12 = max_noftz(smem_kmax[p_stage * 4 + 2], smem_kmax[p_stage * 4 + 3]);
                        float _max_13 = max_noftz(_max_11, _max_12);
                        float k_max = _max_13;
                        int kexp = 0;
                        if (k_max >= 64.0f) {
                            kexp = 1;
                        }
                        if (k_max >= 128.0f) {
                            kexp = 2;
                        }
                        if (k_max >= 256.0f) {
                            kexp = 3;
                        }
                        unsigned int kexp_u = (unsigned int)kexp;
                        unsigned int kexp_f16x2 = (15 - kexp_u << 10) * 65537;
                        unsigned int kexp_f32_bits = 127 - kexp_u << 23;
                        float kexp_scale = __uint_as_float(kexp_f32_bits);
                        mbarrier_wait(v_empty_addr + (p_stage) * 8, p_phase ^ 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (load_warp_rank == 0) {
                            if (elect_sync()) {
                                smem_kexp[p_stage * 4] = kexp_u;
                                mbarrier_arrive(nvfp4_kexp_full_addr + (p_stage) * 8);
                            }
                        }
                        int v_swz = v_row & 7;
                        int v_src_row = smem_k_a_addr + (unsigned int)(p_stage * 28672) + (unsigned int)(v_half * 8192) + (unsigned int)(v_token * 128);
                        unsigned int v_sf_words[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&v_sf_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_sf_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_sf_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_sf_words[(0) + 3]))
                            : "r"(sf_base + v_row * 32 + v_half * 16));
                        int v_dst_row = p_stage * 32768 + (v_row ^ 64) * 128;
                        int v_dst_off = v_dst_row + 16384;
                        int v_local = v_half == bid % 2;
                        unsigned int v_dst_peer = peer_v_base + (unsigned int)v_dst_off;
                        unsigned int v_peer_bar = ((p_stage != 0) ? peer_v_ready_1 : peer_v_ready_0);
                        if (v_half == 0) {
                            #pragma unroll
                            for (int v_u = 0; v_u < 8; v_u++) {
                                unsigned int v_in[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&v_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_in[(0) + 3]))
                                    : "r"(v_src_row + ((v_u ^ v_swz) << 4)));
                                unsigned int v_sc_word = v_sf_words[v_u >> 1];
                                uint32_t _e4m3x2_to_f16x2_4;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(v_sc_word >> (unsigned int)((v_u & 1) * 16))));
                                unsigned int v_sc_pair = _e4m3x2_to_f16x2_4;
                                uint32_t _f16x2_mul_0;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(v_sc_pair), "r"(kexp_f16x2));
                                v_sc_pair = _f16x2_mul_0;
                                uint32_t _prmt_b32_0;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_0) : "r"(v_sc_pair), "r"(v_sc_pair));
                                unsigned int v_sc0 = _prmt_b32_0;
                                uint32_t _prmt_b32_1;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_1) : "r"(v_sc_pair), "r"(v_sc_pair));
                                unsigned int v_sc1 = _prmt_b32_1;
                                unsigned int v_out[8];
                                #pragma unroll
                                for (int v_w = 0; v_w < 4; v_w++) {
                                    unsigned int v_sc = ((v_w < 2) ? v_sc0 : v_sc1);
                                    unsigned int v_word = v_in[v_w];
                                    uint32_t _e2m1_to_f16x2_0;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_0) : "h"((uint16_t)(v_word)));
                                    uint32_t _f16x2_scaled_0;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_0) : "r"(_e2m1_to_f16x2_0), "r"(v_sc));
                                    uint16_t _e4m3x2_0;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_0) : "r"(_f16x2_scaled_0));
                                    uint32_t _e2m1_to_f16x2_1;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_1) : "h"((uint16_t)(v_word >> 8)));
                                    uint32_t _f16x2_scaled_1;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_1) : "r"(_e2m1_to_f16x2_1), "r"(v_sc));
                                    uint16_t _e4m3x2_1;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_1) : "r"(_f16x2_scaled_1));
                                    uint32_t _e2m1_to_f16x2_2;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_2) : "h"((uint16_t)(v_word >> 16)));
                                    uint32_t _f16x2_scaled_2;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_2) : "r"(_e2m1_to_f16x2_2), "r"(v_sc));
                                    uint16_t _e4m3x2_2;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_2) : "r"(_f16x2_scaled_2));
                                    uint32_t _e2m1_to_f16x2_3;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_3) : "h"((uint16_t)(v_word >> 24)));
                                    uint32_t _f16x2_scaled_3;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_3) : "r"(_e2m1_to_f16x2_3), "r"(v_sc));
                                    uint16_t _e4m3x2_3;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_3) : "r"(_f16x2_scaled_3));
                                    uint32_t _pack_u16x2_0;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_0) : "h"(_e4m3x2_0), "h"(_e4m3x2_1));
                                    v_out[2 * v_w] = _pack_u16x2_0;
                                    uint32_t _pack_u16x2_1;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_1) : "h"(_e4m3x2_2), "h"(_e4m3x2_3));
                                    v_out[2 * v_w + 1] = _pack_u16x2_1;
                                }
                                int v_o0 = (v_u & 3) * 2;
                                int v_o1 = v_o0 + 1;
                                int v_dst_u = v_dst_row + (v_u >> 2) * 16384;
                                if (v_local != 0) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_u + ((v_o0 ^ v_swz) << 4))), "r"(v_out[0]), "r"(v_out[1]), "r"(v_out[2]), "r"(v_out[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_u + ((v_o1 ^ v_swz) << 4))), "r"(v_out[4]), "r"(v_out[5]), "r"(v_out[6]), "r"(v_out[7]) : "memory");
                                } else {
                                    asm volatile(
                                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                        :: "r"(peer_v_base + (unsigned int)v_dst_u + (unsigned int)((v_o0 ^ v_swz) << 4)), "r"(v_out[0]), "r"(v_out[1]), "r"(v_out[2]), "r"(v_out[3]), "r"(v_peer_bar) : "memory");
                                    asm volatile(
                                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                        :: "r"(peer_v_base + (unsigned int)v_dst_u + (unsigned int)((v_o1 ^ v_swz) << 4)), "r"(v_out[4]), "r"(v_out[5]), "r"(v_out[6]), "r"(v_out[7]), "r"(v_peer_bar) : "memory");
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int v_u_1 = 0; v_u_1 < 6; v_u_1++) {
                                unsigned int v_in_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_in_1[(0) + 3]))
                                    : "r"(v_src_row + ((v_u_1 ^ v_swz) << 4)));
                                unsigned int v_sc_word_1 = v_sf_words[v_u_1 >> 1];
                                uint32_t _e4m3x2_to_f16x2_5;
                                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(v_sc_word_1 >> (unsigned int)((v_u_1 & 1) * 16))));
                                unsigned int v_sc_pair_1 = _e4m3x2_to_f16x2_5;
                                uint32_t _f16x2_mul_1;
                                asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_1) : "r"(v_sc_pair_1), "r"(kexp_f16x2));
                                v_sc_pair_1 = _f16x2_mul_1;
                                uint32_t _prmt_b32_2;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_2) : "r"(v_sc_pair_1), "r"(v_sc_pair_1));
                                unsigned int v_sc0_1 = _prmt_b32_2;
                                uint32_t _prmt_b32_3;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_3) : "r"(v_sc_pair_1), "r"(v_sc_pair_1));
                                unsigned int v_sc1_1 = _prmt_b32_3;
                                unsigned int v_out_1[8];
                                #pragma unroll
                                for (int v_w_1 = 0; v_w_1 < 4; v_w_1++) {
                                    unsigned int v_sc_1 = ((v_w_1 < 2) ? v_sc0_1 : v_sc1_1);
                                    unsigned int v_word_1 = v_in_1[v_w_1];
                                    uint32_t _e2m1_to_f16x2_4;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_4) : "h"((uint16_t)(v_word_1)));
                                    uint32_t _f16x2_scaled_4;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_4) : "r"(_e2m1_to_f16x2_4), "r"(v_sc_1));
                                    uint16_t _e4m3x2_4;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_4) : "r"(_f16x2_scaled_4));
                                    uint32_t _e2m1_to_f16x2_5;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_5) : "h"((uint16_t)(v_word_1 >> 8)));
                                    uint32_t _f16x2_scaled_5;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_5) : "r"(_e2m1_to_f16x2_5), "r"(v_sc_1));
                                    uint16_t _e4m3x2_5;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_5) : "r"(_f16x2_scaled_5));
                                    uint32_t _e2m1_to_f16x2_6;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_6) : "h"((uint16_t)(v_word_1 >> 16)));
                                    uint32_t _f16x2_scaled_6;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_6) : "r"(_e2m1_to_f16x2_6), "r"(v_sc_1));
                                    uint16_t _e4m3x2_6;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_6) : "r"(_f16x2_scaled_6));
                                    uint32_t _e2m1_to_f16x2_7;
                                    asm("{ .reg .b8 _b, _z;                    \n\t"
                "  mov.b16 {_b, _z}, %1;                \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _b;         }"
                : "=r"(_e2m1_to_f16x2_7) : "h"((uint16_t)(v_word_1 >> 24)));
                                    uint32_t _f16x2_scaled_7;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_scaled_7) : "r"(_e2m1_to_f16x2_7), "r"(v_sc_1));
                                    uint16_t _e4m3x2_7;
                                    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(_e4m3x2_7) : "r"(_f16x2_scaled_7));
                                    uint32_t _pack_u16x2_2;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_2) : "h"(_e4m3x2_4), "h"(_e4m3x2_5));
                                    v_out_1[2 * v_w_1] = _pack_u16x2_2;
                                    uint32_t _pack_u16x2_3;
                                    asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_3) : "h"(_e4m3x2_6), "h"(_e4m3x2_7));
                                    v_out_1[2 * v_w_1 + 1] = _pack_u16x2_3;
                                }
                                int v_o0_1 = (v_u_1 & 3) * 2;
                                int v_o1_1 = v_o0_1 + 1;
                                int v_dst_u_1 = v_dst_row + (v_u_1 >> 2) * 16384;
                                if (v_local != 0) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_u_1 + ((v_o0_1 ^ v_swz) << 4))), "r"(v_out_1[0]), "r"(v_out_1[1]), "r"(v_out_1[2]), "r"(v_out_1[3]) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_u_1 + ((v_o1_1 ^ v_swz) << 4))), "r"(v_out_1[4]), "r"(v_out_1[5]), "r"(v_out_1[6]), "r"(v_out_1[7]) : "memory");
                                } else {
                                    asm volatile(
                                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                        :: "r"(peer_v_base + (unsigned int)v_dst_u_1 + (unsigned int)((v_o0_1 ^ v_swz) << 4)), "r"(v_out_1[0]), "r"(v_out_1[1]), "r"(v_out_1[2]), "r"(v_out_1[3]), "r"(v_peer_bar) : "memory");
                                    asm volatile(
                                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                        :: "r"(peer_v_base + (unsigned int)v_dst_u_1 + (unsigned int)((v_o1_1 ^ v_swz) << 4)), "r"(v_out_1[4]), "r"(v_out_1[5]), "r"(v_out_1[6]), "r"(v_out_1[7]), "r"(v_peer_bar) : "memory");
                                }
                            }
                        }
                        if (v_half != 0) {
                            int r_src_row = smem_k_a_addr + (unsigned int)(p_stage * 28672) + 16384 + (unsigned int)(v_token * 128);
                            #pragma unroll
                            for (int r_u = 0; r_u < 4; r_u++) {
                                unsigned int r_out[4];
                                #pragma unroll
                                for (int r_half = 0; r_half < 2; r_half++) {
                                    unsigned int r_in[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&r_in[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_in[(0) + 3]))
                                        : "r"(r_src_row + ((2 * r_u + r_half ^ v_swz) << 4)));
                                    #pragma unroll
                                    for (int r_w = 0; r_w < 2; r_w++) {
                                        float _cvt_f32_bf16_2;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(r_in[2 * r_w] & 65535)));
                                        float r_lo0 = _cvt_f32_bf16_2 * kexp_scale;
                                        float _cvt_f32_bf16_3;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_3) : "h"((uint16_t)(r_in[2 * r_w] >> 16)));
                                        float r_hi0 = _cvt_f32_bf16_3 * kexp_scale;
                                        float _cvt_f32_bf16_4;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_4) : "h"((uint16_t)(r_in[2 * r_w + 1] & 65535)));
                                        float r_lo1 = _cvt_f32_bf16_4 * kexp_scale;
                                        float _cvt_f32_bf16_5;
                                        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_5) : "h"((uint16_t)(r_in[2 * r_w + 1] >> 16)));
                                        float r_hi1 = _cvt_f32_bf16_5 * kexp_scale;
                                        uint16_t _e4m3x2_f32_0;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(r_hi0), "f"(r_lo0));
                                        uint16_t _e4m3x2_f32_1;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(r_hi1), "f"(r_lo1));
                                        uint32_t _pack_u16x2_4;
                                        asm("mov.b32 %0, {%1, %2};" : "=r"(_pack_u16x2_4) : "h"(_e4m3x2_f32_0), "h"(_e4m3x2_f32_1));
                                        r_out[2 * r_half + r_w] = _pack_u16x2_4;
                                    }
                                }
                                int r_o = 4 + r_u;
                                if (bid % 2 == 1) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_full_addr + (unsigned int)(v_dst_off + ((r_o ^ v_swz) << 4))), "r"(r_out[0]), "r"(r_out[1]), "r"(r_out[2]), "r"(r_out[3]) : "memory");
                                } else {
                                    asm volatile(
                                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                        :: "r"(v_dst_peer + (unsigned int)((r_o ^ v_swz) << 4)), "r"(r_out[0]), "r"(r_out[1]), "r"(r_out[2]), "r"(r_out[3]), "r"(v_peer_bar) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (load_warp_rank == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(nvfp4_v_ready_addr + (p_stage) * 8, 16384);
                            }
                            mbarrier_wait(nvfp4_v_ready_addr + (p_stage) * 8, p_phase);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            if (elect_sync()) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(v_full_addr + p_stage * 8), "r"(0) : "memory");
                            }
                        }
                    }
                    tile_cursor = tile_cursor + num_kv_tiles;
                    item_cursor = item_cursor + 1;
                    q_ring_seq = q_ring_seq + 4;
                } else {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_1) * 8, _phase_source_work_full_1);
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
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_6 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_6)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_7 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_7)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_1 * 8), "r"(0) : "memory");
                    source_work_stage_1 += 1;
                    if (source_work_stage_1 == 2) { source_work_stage_1 = 0; _phase_source_work_full_1 ^= 1; }
                    has_next = 0;
                    next_valid = 0;
                    if (_clc_valid_2 != 0) {
                        has_next = 1;
                        source_work_x_1 = _clc_ctaid_6 + (unsigned int)cta_rank;
                        source_work_z_1 = _clc_ctaid_8;
                        next_query_idx = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
                        next_valid = next_query_idx < num_query_tokens;
                        if (next_valid != 0) {
                            int _max_14 = ((sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[next_query_idx] + sparse_topk_lens_offset) : (0));
                            int _min_6 = ((_max_14) < (swa_width) ? (_max_14) : (swa_width));
                            next_main_active = _min_6;
                            next_extra_active = 0;
                            if (compressed_width > 0) {
                                int _max_15 = ((extra_topk_lens[next_query_idx]) > (0) ? (extra_topk_lens[next_query_idx]) : (0));
                                int _min_7 = ((_max_15) < (compressed_width) ? (_max_15) : (compressed_width));
                                next_extra_active = _min_7;
                            }
                            next_n_main_tiles = (next_main_active + 128 - 1) / 128;
                            next_n_extra_tiles = (next_extra_active + 128 - 1) / 128;
                            int _max_16 = ((next_n_main_tiles + next_n_extra_tiles) > (1) ? (next_n_main_tiles + next_n_extra_tiles) : (1));
                            next_num_kv_tiles = _max_16;
                            int g_stage_3 = tile_cursor & 1;
                            {
                                mbarrier_wait(index_full_addr + (load_k_index_stage) * 8, _phase_index_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            mbarrier_wait(k_empty_addr + (g_stage_3) * 8, tile_cursor >> 1 & 1 ^ 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            if (load_warp_rank == 0) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(k_full_addr + (g_stage_3) * 8, 28672);
                                }
                            }
                            int g_group_3 = lane;
                            if (g_group_3 < 16) {
                                int g_rows_3[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&g_rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&g_rows_3[(0) + 3]))
                                    : "r"(smem_sparse_indices_addr + load_k_index_stage * 1024 + (unsigned int)((k_cta_offset + g_group_3 * 4) * 4)));
                                int g_use_swa_3 = next_n_main_tiles > 0 || next_n_extra_tiles == 0;
                                int g_page_log2_3 = ((g_use_swa_3 != 0) ? swa_page_log2 : compressed_page_log2);
                                int g_pitch_units_3 = ((g_use_swa_3 != 0) ? swa_pitch_units : compressed_pitch_units);
                                int g_footer_units_3 = ((g_use_swa_3 != 0) ? swa_footer_units : compressed_footer_units);
                                int g_data_rows_3[4];
                                int g_foot_rows_3[4];
                                #pragma unroll
                                for (int g_j_3 = 0; g_j_3 < 4; g_j_3++) {
                                    int g_tok_3 = ((g_rows_3[g_j_3] >= 0) ? g_rows_3[g_j_3] : 0);
                                    int g_page_3 = g_tok_3 >> g_page_log2_3;
                                    int g_slot_3 = g_tok_3 - (g_page_3 << g_page_log2_3);
                                    g_data_rows_3[g_j_3] = g_page_3 * g_pitch_units_3 + g_slot_3 * 11;
                                    g_foot_rows_3[g_j_3] = g_page_3 * g_pitch_units_3 + g_footer_units_3 + g_slot_3;
                                }
                                int g_stage_base_3 = smem_k_a_addr + (unsigned int)(g_stage_3 * 28672);
                                if (load_warp_rank < 3) {
                                    int g_col_3 = ((load_warp_rank == 2) ? 224 : load_warp_rank * 128);
                                    int g_dst_3 = g_stage_base_3 + load_warp_rank * 8192 + g_group_3 * 512;
                                    if (g_use_swa_3 != 0) {
                                        tma_gather4_gmem2smem(g_dst_3, (&tmap_swa_kv), g_col_3, g_data_rows_3[0], g_data_rows_3[1], g_data_rows_3[2], g_data_rows_3[3], k_full_addr + (g_stage_3) * 8);
                                    } else {
                                        tma_gather4_gmem2smem(g_dst_3, (&tmap_compressed_kv), g_col_3, g_data_rows_3[0], g_data_rows_3[1], g_data_rows_3[2], g_data_rows_3[3], k_full_addr + (g_stage_3) * 8);
                                    }
                                } else {
                                    int g_fdst_3 = g_stage_base_3 + 24576 + (k_cta_offset + g_group_3 * 4) * 32;
                                    if (g_use_swa_3 != 0) {
                                        tma_gather4_gmem2smem_mc(g_fdst_3, (&tmap_swa_sf), 0, g_foot_rows_3[0], g_foot_rows_3[1], g_foot_rows_3[2], g_foot_rows_3[3], k_full_addr + (g_stage_3) * 8, 3);
                                    } else {
                                        tma_gather4_gmem2smem_mc(g_fdst_3, (&tmap_compressed_sf), 0, g_foot_rows_3[0], g_foot_rows_3[1], g_foot_rows_3[2], g_foot_rows_3[3], k_full_addr + (g_stage_3) * 8, 3);
                                    }
                                }
                            }
                            if (1 == next_num_kv_tiles) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(index_empty_addr + (load_k_index_stage) * 8);
                                load_k_index_stage += 1;
                                if (load_k_index_stage == 6) { load_k_index_stage = 0; _phase_index_full ^= 1; }
                            }
                        }
                    }
                }
                if (has_next == 0) {
                    break;
                }
                source_work_valid_1 = next_valid;
                mapped_query_idx_1 = next_query_idx;
                main_active_1 = next_main_active;
                extra_active_1 = next_extra_active;
                n_main_tiles_1 = next_n_main_tiles;
                n_extra_tiles_1 = next_n_extra_tiles;
                num_kv_tiles = next_num_kv_tiles;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            const int wg_dummy_inc = 0;
            int all_num_kv_tiles_1 = (sparse_topk + 128 - 1) / 128;
            const int sq_lane128 = warp % 4 * 32 + lane;
            const int sq_row = sq_lane128 >> 1;
            const int sq_pair = sq_lane128 & 1;
            const int sq_swz = sq_row & 7;
            unsigned int sm_ring_seq = 0;
            int sm_first = 1;
            mbarrier_wait(nvfp4_q_lut_ready_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            const int sfb_tok_a = (warp % 4 >> 1) * 64 + lane;
            const int sfb_tok_b = sfb_tok_a + 32;
            const int tmem_row_base_1 = ((1) ? warp % 2 * 32 : warp % 4 * 32);
            const int tmem_score_row_base = ((1) ? (int)(warp % 4 * 32) : tmem_row_base_1);
            const int n_half = ((1) ? (int)(warp % 4 / 2) : 0);
            const int my_row = tmem_row_base_1 + lane;
            const int stats_row = n_half * 64 + my_row;
            int softmax_tile_cursor = 0;
            int sm_item = 0;
            unsigned int softmax_index_stage = 0;
            unsigned int source_work_stage_2 = 0;
            unsigned int source_work_x_2 = blockIdx.x;
            unsigned int source_work_z_2 = blockIdx.z;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_source_sum_empty_0 = 1;
            unsigned int _phase_source_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = 0; work_idx_1 < 514; work_idx_1++) {
                int split_idx_1 = 0;
                int query_idx_1 = source_work_x_2 >> 1 >> 1;
                {
                    split_idx_1 = 0;
                    query_idx_1 = source_work_x_2 >> 1;
                }
                int source_work_valid_2 = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_2 = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1);
                if (source_work_valid_2 != 0) {
                    int _max_20 = ((sparse_topk_lens[mapped_query_idx_2] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_2] + sparse_topk_lens_offset) : (0));
                    int _min_10 = ((_max_20) < (swa_width) ? (_max_20) : (swa_width));
                    int main_active_2 = _min_10;
                    int extra_active_2 = 0;
                    if (compressed_width > 0) {
                        int _max_21 = ((extra_topk_lens[mapped_query_idx_2]) > (0) ? (extra_topk_lens[mapped_query_idx_2]) : (0));
                        int _min_11 = ((_max_21) < (compressed_width) ? (_max_21) : (compressed_width));
                        extra_active_2 = _min_11;
                    }
                    int n_main_tiles_2 = (main_active_2 + 128 - 1) / 128;
                    int n_extra_tiles_2 = (extra_active_2 + 128 - 1) / 128;
                    int _max_22 = ((n_main_tiles_2 + n_extra_tiles_2) > (1) ? (n_main_tiles_2 + n_extra_tiles_2) : (1));
                    int active_topk_1 = _max_22 * 128;
                    {
                        all_num_kv_tiles_1 = (active_topk_1 + 128 - 1) / 128;
                    }
                    int tiles_per_split_1 = all_num_kv_tiles_1 + 1 - 1;
                    int first_tile_1 = split_idx_1 * tiles_per_split_1;
                    int num_kv_tiles_1 = tiles_per_split_1;
                    mbarrier_wait(q_empty_addr, sm_item & 1 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (sm_first != 0) {
                        unsigned int qi_slot = sm_ring_seq & 1;
                        if (warp % 4 == 0) {
                            mbarrier_wait(nvfp4_q_ring_empty_addr + (qi_slot) * 8, sm_ring_seq >> 1 & 1 ^ 1);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(nvfp4_q_ring_full_addr + (qi_slot) * 8, 8192);
                                tma_4d_gmem2smem(smem_q_ring_addr + qi_slot * 8192, (&tmap_q), 0, bid % 2 * 64, 0, mapped_query_idx_2, nvfp4_q_ring_full_addr + (qi_slot) * 8);
                            }
                        }
                        unsigned int qi_slot_0 = sm_ring_seq + 1 & 1;
                        if (warp % 4 == 0) {
                            mbarrier_wait(nvfp4_q_ring_empty_addr + (qi_slot_0) * 8, sm_ring_seq + 1 >> 1 & 1 ^ 1);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(nvfp4_q_ring_full_addr + (qi_slot_0) * 8, 8192);
                                tma_4d_gmem2smem(smem_q_ring_addr + qi_slot_0 * 8192, (&tmap_q), 0, bid % 2 * 64, 1, mapped_query_idx_2, nvfp4_q_ring_full_addr + (qi_slot_0) * 8);
                            }
                        }
                        sm_first = 0;
                    }
                    #pragma unroll 1
                    for (int sq_c = 0; sq_c < 2; sq_c++) {
                        unsigned int sq_seq = sm_ring_seq + (unsigned int)sq_c;
                        unsigned int sq_slot = sq_seq & 1;
                        mbarrier_wait(nvfp4_q_ring_full_addr + (sq_slot) * 8, sq_seq >> 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int q_in_1[16];
                        #pragma unroll
                        for (int q_u_1 = 0; q_u_1 < 4; q_u_1++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_in_1[4 * q_u_1])), "=r"(*reinterpret_cast<uint32_t*>(&q_in_1[(4 * q_u_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_in_1[(4 * q_u_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_in_1[(4 * q_u_1) + 3]))
                                : "r"(smem_q_ring_addr + sq_slot * 8192 + (unsigned int)(sq_row * 128) + (unsigned int)((sq_pair * 4 + q_u_1 ^ sq_swz) << 4)));
                        }
                        unsigned int q_codes_1[4];
                        unsigned int q_sf_pair_1 = 0;
                        #pragma unroll
                        for (int q_g_1 = 0; q_g_1 < 2; q_g_1++) {
                            float q_vals_1[16];
                            #pragma unroll
                            for (int q_w_2 = 0; q_w_2 < 8; q_w_2++) {
                                unsigned int q_word_1 = q_in_1[8 * q_g_1 + q_w_2];
                                float _cvt_f32_bf16_6;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_6) : "h"((uint16_t)(q_word_1 & 65535)));
                                q_vals_1[2 * q_w_2] = _cvt_f32_bf16_6;
                                float _cvt_f32_bf16_7;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_7) : "h"((uint16_t)(q_word_1 >> 16)));
                                q_vals_1[2 * q_w_2 + 1] = _cvt_f32_bf16_7;
                            }
                            float _fabs_2 = fabsf(q_vals_1[0]);
                            float q_amax_1 = _fabs_2;
                            #pragma unroll
                            for (int q_i_1 = 1; q_i_1 < 16; q_i_1++) {
                                float _fabs_3 = fabsf(q_vals_1[q_i_1]);
                                float _max_23 = max_noftz(q_amax_1, _fabs_3);
                                q_amax_1 = _max_23;
                            }
                            unsigned int q_amax_bits_1 = __as_u32(q_amax_1);
                            int q_m_1 = (int)(q_amax_bits_1 >> 16 & 127 | 128);
                            int q_e_1 = (int)(q_amax_bits_1 >> 23);
                            int _max_24 = ((q_e_1) > (7) ? (q_e_1) : (7));
                            unsigned int q_pow_bits_1 = (unsigned int)(_max_24 - 7 << 23);
                            float q_div6_1 = smem_q_lut6[q_m_1] * __uint_as_float(q_pow_bits_1);
                            uint8_t _fp8_encode_raw_1;
                            float _fp8_encode_decoded_1;
                            uint16_t _e4m3x2_encode_0;
                            uint32_t _f16x2_encode_0;
                            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_encode_0) : "f"(0.0f), "f"(q_div6_1));
                            _fp8_encode_raw_1 = (uint8_t)(_e4m3x2_encode_0 & 0x00ffu);
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_encode_0) : "h"(_e4m3x2_encode_0));
                            uint16_t _fp8_encode_h0_0 = (uint16_t)(_f16x2_encode_0 & 0xffffu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_encode_decoded_1) : "h"(_fp8_encode_h0_0));
                            float q_sf_inv_1 = smem_q_lutinv[(int)_fp8_encode_raw_1];
                            #pragma unroll
                            for (int q_w_3 = 0; q_w_3 < 2; q_w_3++) {
                                uint32_t _fp4_pair_4;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_4) : "f"(q_vals_1[8 * q_w_3] * q_sf_inv_1), "f"(q_vals_1[8 * q_w_3 + 1] * q_sf_inv_1));
                                unsigned int q_b0_1 = _fp4_pair_4;
                                uint32_t _fp4_pair_5;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_5) : "f"(q_vals_1[8 * q_w_3 + 2] * q_sf_inv_1), "f"(q_vals_1[8 * q_w_3 + 3] * q_sf_inv_1));
                                unsigned int q_b1_1 = _fp4_pair_5;
                                uint32_t _fp4_pair_6;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_6) : "f"(q_vals_1[8 * q_w_3 + 4] * q_sf_inv_1), "f"(q_vals_1[8 * q_w_3 + 5] * q_sf_inv_1));
                                unsigned int q_b2_1 = _fp4_pair_6;
                                uint32_t _fp4_pair_7;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_7) : "f"(q_vals_1[8 * q_w_3 + 6] * q_sf_inv_1), "f"(q_vals_1[8 * q_w_3 + 7] * q_sf_inv_1));
                                unsigned int q_b3_1 = _fp4_pair_7;
                                q_codes_1[2 * q_g_1 + q_w_3] = q_b0_1 | q_b1_1 << 8 | q_b2_1 << 16 | q_b3_1 << 24;
                            }
                            q_sf_pair_1 = q_sf_pair_1 | (unsigned int)_fp8_encode_raw_1 << (unsigned int)(q_g_1 * 8);
                        }
                        int q_dst_1 = (sq_c >> 2) * 8192 + sq_row * 128 + ((2 * (sq_c & 3) + sq_pair ^ sq_swz) << 4);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_q_a_addr + (unsigned int)q_dst_1), "r"(q_codes_1[0]), "r"(q_codes_1[1]), "r"(q_codes_1[2]), "r"(q_codes_1[3]) : "memory");
                        unsigned int q_sf_mine_1 = q_sf_pair_1 << (unsigned int)(sq_pair * 16);
                        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, q_sf_mine_1, 1);
                        unsigned int q_sf_other_1 = _shfl_xor_1;
                        if (sq_pair == 0) {
                            smem_q_sf[sq_row * 8 + sq_c] = q_sf_mine_1 | q_sf_other_1;
                        }
                        mbarrier_arrive(nvfp4_q_ring_empty_addr + (sq_slot) * 8);
                        unsigned int qi_slot_1 = sq_seq + 2 & 1;
                        if (warp % 4 == 0) {
                            mbarrier_wait(nvfp4_q_ring_empty_addr + (qi_slot_1) * 8, sq_seq + 2 >> 1 & 1 ^ 1);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(nvfp4_q_ring_full_addr + (qi_slot_1) * 8, 8192);
                                tma_4d_gmem2smem(smem_q_ring_addr + qi_slot_1 * 8192, (&tmap_q), 0, bid % 2 * 64, sq_c + 5, mapped_query_idx_2, nvfp4_q_ring_full_addr + (qi_slot_1) * 8);
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int sq_c_1 = 5; sq_c_1 < 7; sq_c_1++) {
                        unsigned int sq_seq_1 = sm_ring_seq + (unsigned int)(sq_c_1 - 3);
                        unsigned int sq_slot_1 = sq_seq_1 & 1;
                        mbarrier_wait(nvfp4_q_ring_full_addr + (sq_slot_1) * 8, sq_seq_1 >> 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int q_in_2[16];
                        #pragma unroll
                        for (int q_u_2 = 0; q_u_2 < 4; q_u_2++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_in_2[4 * q_u_2])), "=r"(*reinterpret_cast<uint32_t*>(&q_in_2[(4 * q_u_2) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_in_2[(4 * q_u_2) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_in_2[(4 * q_u_2) + 3]))
                                : "r"(smem_q_ring_addr + sq_slot_1 * 8192 + (unsigned int)(sq_row * 128) + (unsigned int)((sq_pair * 4 + q_u_2 ^ sq_swz) << 4)));
                        }
                        unsigned int q_codes_2[4];
                        unsigned int q_sf_pair_2 = 0;
                        #pragma unroll
                        for (int q_g_2 = 0; q_g_2 < 2; q_g_2++) {
                            float q_vals_2[16];
                            #pragma unroll
                            for (int q_w_4 = 0; q_w_4 < 8; q_w_4++) {
                                unsigned int q_word_2 = q_in_2[8 * q_g_2 + q_w_4];
                                float _cvt_f32_bf16_8;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_8) : "h"((uint16_t)(q_word_2 & 65535)));
                                q_vals_2[2 * q_w_4] = _cvt_f32_bf16_8;
                                float _cvt_f32_bf16_9;
                                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_9) : "h"((uint16_t)(q_word_2 >> 16)));
                                q_vals_2[2 * q_w_4 + 1] = _cvt_f32_bf16_9;
                            }
                            float _fabs_4 = fabsf(q_vals_2[0]);
                            float q_amax_2 = _fabs_4;
                            #pragma unroll
                            for (int q_i_2 = 1; q_i_2 < 16; q_i_2++) {
                                float _fabs_5 = fabsf(q_vals_2[q_i_2]);
                                float _max_25 = max_noftz(q_amax_2, _fabs_5);
                                q_amax_2 = _max_25;
                            }
                            unsigned int q_amax_bits_2 = __as_u32(q_amax_2);
                            int q_m_2 = (int)(q_amax_bits_2 >> 16 & 127 | 128);
                            int q_e_2 = (int)(q_amax_bits_2 >> 23);
                            int _max_26 = ((q_e_2) > (7) ? (q_e_2) : (7));
                            unsigned int q_pow_bits_2 = (unsigned int)(_max_26 - 7 << 23);
                            float q_div6_2 = smem_q_lut6[q_m_2] * __uint_as_float(q_pow_bits_2);
                            uint8_t _fp8_encode_raw_2;
                            float _fp8_encode_decoded_2;
                            uint16_t _e4m3x2_encode_1;
                            uint32_t _f16x2_encode_1;
                            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_encode_1) : "f"(0.0f), "f"(q_div6_2));
                            _fp8_encode_raw_2 = (uint8_t)(_e4m3x2_encode_1 & 0x00ffu);
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_encode_1) : "h"(_e4m3x2_encode_1));
                            uint16_t _fp8_encode_h0_1 = (uint16_t)(_f16x2_encode_1 & 0xffffu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_encode_decoded_2) : "h"(_fp8_encode_h0_1));
                            float q_sf_inv_2 = smem_q_lutinv[(int)_fp8_encode_raw_2];
                            #pragma unroll
                            for (int q_w_5 = 0; q_w_5 < 2; q_w_5++) {
                                uint32_t _fp4_pair_8;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_8) : "f"(q_vals_2[8 * q_w_5] * q_sf_inv_2), "f"(q_vals_2[8 * q_w_5 + 1] * q_sf_inv_2));
                                unsigned int q_b0_2 = _fp4_pair_8;
                                uint32_t _fp4_pair_9;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_9) : "f"(q_vals_2[8 * q_w_5 + 2] * q_sf_inv_2), "f"(q_vals_2[8 * q_w_5 + 3] * q_sf_inv_2));
                                unsigned int q_b1_2 = _fp4_pair_9;
                                uint32_t _fp4_pair_10;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_10) : "f"(q_vals_2[8 * q_w_5 + 4] * q_sf_inv_2), "f"(q_vals_2[8 * q_w_5 + 5] * q_sf_inv_2));
                                unsigned int q_b2_2 = _fp4_pair_10;
                                uint32_t _fp4_pair_11;
                                asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_11) : "f"(q_vals_2[8 * q_w_5 + 6] * q_sf_inv_2), "f"(q_vals_2[8 * q_w_5 + 7] * q_sf_inv_2));
                                unsigned int q_b3_2 = _fp4_pair_11;
                                q_codes_2[2 * q_g_2 + q_w_5] = q_b0_2 | q_b1_2 << 8 | q_b2_2 << 16 | q_b3_2 << 24;
                            }
                            q_sf_pair_2 = q_sf_pair_2 | (unsigned int)_fp8_encode_raw_2 << (unsigned int)(q_g_2 * 8);
                        }
                        int q_dst_2 = (sq_c_1 >> 2) * 8192 + sq_row * 128 + ((2 * (sq_c_1 & 3) + sq_pair ^ sq_swz) << 4);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_q_a_addr + (unsigned int)q_dst_2), "r"(q_codes_2[0]), "r"(q_codes_2[1]), "r"(q_codes_2[2]), "r"(q_codes_2[3]) : "memory");
                        unsigned int q_sf_mine_2 = q_sf_pair_2 << (unsigned int)(sq_pair * 16);
                        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sf_mine_2, 1);
                        unsigned int q_sf_other_2 = _shfl_xor_2;
                        if (sq_pair == 0) {
                            smem_q_sf[sq_row * 8 + sq_c_1] = q_sf_mine_2 | q_sf_other_2;
                        }
                        mbarrier_arrive(nvfp4_q_ring_empty_addr + (sq_slot_1) * 8);
                    }
                    sm_ring_seq = sm_ring_seq + 4;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    mbarrier_wait(nvfp4_q_sf_ready_addr, sm_item & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int sfa_regs[32];
                    #pragma unroll
                    for (int sfa_s = 0; sfa_s < 7; sfa_s++) {
                        sfa_regs[4 * sfa_s] = smem_q_sf[lane * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 1] = smem_q_sf[(lane + 32) * 8 + sfa_s];
                        sfa_regs[4 * sfa_s + 2] = 0;
                        sfa_regs[4 * sfa_s + 3] = 0;
                    }
                    #pragma unroll
                    for (int sfa_pad = 28; sfa_pad < 32; sfa_pad++) {
                        sfa_regs[sfa_pad] = 0;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 64 + (unsigned int)(tmem_score_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfa_regs[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 64 + 16 + (unsigned int)(tmem_score_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(sfa_regs + 16)[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp == 0) {
                        mbarrier_wait(nvfp4_q_rope_full_addr, sm_item & 1);
                        if (elect_sync()) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(q_full_addr), "r"(0) : "memory");
                        }
                    }
                    int sfb_pt0 = softmax_tile_cursor;
                    int sfb_stage0 = sfb_pt0 & 1;
                    mbarrier_wait(k_full_addr + (sfb_stage0) * 8, sfb_pt0 >> 1 & 1);
                    mbarrier_wait(k_empty_addr + (sfb_pt0 + 1 & 1) * 8, sfb_pt0 + 1 >> 1 & 1 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfb_base0 = smem_k_a_addr + (unsigned int)(sfb_stage0 * 28672) + 24576;
                    unsigned int sfb_a0[8];
                    unsigned int sfb_b0[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[(0) + 3]))
                        : "r"(sfb_base0 + sfb_tok_a * 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_a0[(4) + 3]))
                        : "r"(sfb_base0 + sfb_tok_a * 32 + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[(0) + 3]))
                        : "r"(sfb_base0 + sfb_tok_b * 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfb_b0[(4) + 3]))
                        : "r"(sfb_base0 + sfb_tok_b * 32 + 16));
                    unsigned int sfb_regs0[32];
                    #pragma unroll
                    for (int sfb_s = 0; sfb_s < 7; sfb_s++) {
                        sfb_regs0[4 * sfb_s] = sfb_a0[sfb_s];
                        sfb_regs0[4 * sfb_s + 1] = sfb_b0[sfb_s];
                        sfb_regs0[4 * sfb_s + 2] = 0;
                        sfb_regs0[4 * sfb_s + 3] = 0;
                    }
                    #pragma unroll
                    for (int sfb_pad = 28; sfb_pad < 32; sfb_pad++) {
                        sfb_regs0[sfb_pad] = 0;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 96 + (unsigned int)(tmem_score_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[0])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[1])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[2])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[3])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[4])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[5])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[6])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[7])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[8])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[9])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[10])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[11])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[12])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[13])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[14])), "r"(*reinterpret_cast<const uint32_t*>(&sfb_regs0[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 96 + 16 + (unsigned int)(tmem_score_row_base << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(sfb_regs0 + 16)[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(nvfp4_sf_full_addr + sfb_stage0 * 8), "r"(0) : "memory");
                        }
                    }
                    float row_max_val = -CAKE_INF;
                    float row_sum_val = 0.0f;
                    int sink_head = ((1) ? bid % 2 * 64 + my_row : my_row);
                    if (has_sinks != 0 && sink_head < num_heads && split_idx_1 == 0) {
                        row_max_val = sinks[sink_head] * 1.4426950408889634f / softmax_scale_log2;
                        row_sum_val = ((n_half == 0) ? 1.0f : 0.0f);
                    }
                    #pragma unroll 1
                    for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                        int pipeline_tile = softmax_tile_cursor + tile_1;
                        int phase = pipeline_tile & 1;
                        int s_wait_phase = pipeline_tile >> 1 & 1;
                        int sm_t0 = sm_item < 48 && tile_1 == 0;
                        int s_off = ((phase != 0) ? 128 : 0);
                        int s_base = taddr + (unsigned int)s_off + (unsigned int)(tmem_score_row_base << 16);
                        float new_max = row_max_val;
                        int seg_in_main_1 = n_main_tiles_2 > tile_1 || n_extra_tiles_2 == 0;
                        int seg_active = ((seg_in_main_1 != 0) ? main_active_2 : extra_active_2);
                        int seg_tile_1 = ((seg_in_main_1 != 0) ? tile_1 : tile_1 - n_main_tiles_2);
                        int valid_sparse_cols = seg_active - seg_tile_1 * 128 - n_half * 64;
                        if (valid_sparse_cols < 0) {
                            valid_sparse_cols = 0;
                        }
                        if (valid_sparse_cols > 64) {
                            valid_sparse_cols = 64;
                        }
                        {
                            uint32_t _mbar_token_4 = mbarrier_try_wait(s_full_addr + (phase) * 8, s_wait_phase);
                            mbarrier_wait_token(s_full_addr + (phase) * 8, s_wait_phase, _mbar_token_4);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        int index_tile_addr = smem_sparse_indices_addr + softmax_index_stage * 1024 + (unsigned int)((tile_1 & 1) * 512);
                        int index_lane_addr = index_tile_addr + (n_half * 64 + lane) * 4;
                        int staged_index[2];
                        unsigned int invalid_cols[2];
                        unsigned int any_invalid = 0;
                        #pragma unroll
                        for (int w = 0; w < 2; w++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&staged_index[w])) : "r"(index_lane_addr + w * 128));
                            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, staged_index[w] < 0);
                            invalid_cols[w] = _vote_0;
                            any_invalid = any_invalid | invalid_cols[w];
                        }
                        if ((tile_1 & 1) != 0 || tile_1 + 1 == num_kv_tiles_1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(index_empty_addr + (softmax_index_stage) * 8);
                            softmax_index_stage = softmax_index_stage + 1;
                            if (softmax_index_stage == 6) {
                                softmax_index_stage = 0;
                            }
                        }
                        float _tmem_load_0[4];
                        tmem_ld_x4(&_tmem_load_0[0], s_base);
                        float score_tile_max = -CAKE_INF;
                        float score_tile_max_hi = -CAKE_INF;
                        float sv_split[64];
                        if (valid_sparse_cols == 64 && any_invalid == 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                                : "=f"(sv_split[0]), "=f"(sv_split[1]), "=f"(sv_split[2]), "=f"(sv_split[3]), "=f"(sv_split[4]), "=f"(sv_split[5]), "=f"(sv_split[6]), "=f"(sv_split[7]), "=f"(sv_split[8]), "=f"(sv_split[9]), "=f"(sv_split[10]), "=f"(sv_split[11]), "=f"(sv_split[12]), "=f"(sv_split[13]), "=f"(sv_split[14]), "=f"(sv_split[15]), "=f"(sv_split[16]), "=f"(sv_split[17]), "=f"(sv_split[18]), "=f"(sv_split[19]), "=f"(sv_split[20]), "=f"(sv_split[21]), "=f"(sv_split[22]), "=f"(sv_split[23]), "=f"(sv_split[24]), "=f"(sv_split[25]), "=f"(sv_split[26]), "=f"(sv_split[27]), "=f"(sv_split[28]), "=f"(sv_split[29]), "=f"(sv_split[30]), "=f"(sv_split[31]), "=f"(sv_split[32]), "=f"(sv_split[33]), "=f"(sv_split[34]), "=f"(sv_split[35]), "=f"(sv_split[36]), "=f"(sv_split[37]), "=f"(sv_split[38]), "=f"(sv_split[39]), "=f"(sv_split[40]), "=f"(sv_split[41]), "=f"(sv_split[42]), "=f"(sv_split[43]), "=f"(sv_split[44]), "=f"(sv_split[45]), "=f"(sv_split[46]), "=f"(sv_split[47]), "=f"(sv_split[48]), "=f"(sv_split[49]), "=f"(sv_split[50]), "=f"(sv_split[51]), "=f"(sv_split[52]), "=f"(sv_split[53]), "=f"(sv_split[54]), "=f"(sv_split[55]), "=f"(sv_split[56]), "=f"(sv_split[57]), "=f"(sv_split[58]), "=f"(sv_split[59]), "=f"(sv_split[60]), "=f"(sv_split[61]), "=f"(sv_split[62]), "=f"(sv_split[63])
                                : "r"(s_base));
                            float _max3_0;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_0) : "f"(sv_split[0]), "f"(sv_split[1]), "f"(sv_split[2]));
                            float _max3_1;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(sv_split[3]), "f"(sv_split[4]), "f"(sv_split[5]));
                            float _max3_2;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_2) : "f"(sv_split[6]), "f"(sv_split[7]), "f"(sv_split[8]));
                            float _max3_3;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_3) : "f"(sv_split[9]), "f"(sv_split[10]), "f"(sv_split[11]));
                            float _max3_4;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_4) : "f"(sv_split[12]), "f"(sv_split[13]), "f"(sv_split[14]));
                            float _max3_5;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(sv_split[15]), "f"(sv_split[16]), "f"(sv_split[17]));
                            float _max3_6;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(sv_split[18]), "f"(sv_split[19]), "f"(sv_split[20]));
                            float _max3_7;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_7) : "f"(sv_split[21]), "f"(sv_split[22]), "f"(sv_split[23]));
                            float _max3_8;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(sv_split[24]), "f"(sv_split[25]), "f"(sv_split[26]));
                            float _max3_9;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(sv_split[27]), "f"(sv_split[28]), "f"(sv_split[29]));
                            float _max3_10;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_10) : "f"(sv_split[30]), "f"(sv_split[31]), "f"(sv_split[32]));
                            float _max3_11;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_11) : "f"(sv_split[33]), "f"(sv_split[34]), "f"(sv_split[35]));
                            float _max3_12;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_12) : "f"(sv_split[36]), "f"(sv_split[37]), "f"(sv_split[38]));
                            float _max3_13;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_13) : "f"(sv_split[39]), "f"(sv_split[40]), "f"(sv_split[41]));
                            float _max3_14;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_14) : "f"(sv_split[42]), "f"(sv_split[43]), "f"(sv_split[44]));
                            float _max3_15;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_15) : "f"(sv_split[45]), "f"(sv_split[46]), "f"(sv_split[47]));
                            float _max3_16;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(sv_split[48]), "f"(sv_split[49]), "f"(sv_split[50]));
                            float _max3_17;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(sv_split[51]), "f"(sv_split[52]), "f"(sv_split[53]));
                            float _max3_18;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_18) : "f"(sv_split[54]), "f"(sv_split[55]), "f"(sv_split[56]));
                            float _max3_19;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_19) : "f"(sv_split[57]), "f"(sv_split[58]), "f"(sv_split[59]));
                            float _max3_20;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_20) : "f"(sv_split[60]), "f"(sv_split[61]), "f"(sv_split[62]));
                            float _max3_21;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(_max3_0), "f"(_max3_1), "f"(_max3_2));
                            float _max3_22;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(_max3_3), "f"(_max3_4), "f"(_max3_5));
                            float _max3_23;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_23) : "f"(_max3_6), "f"(_max3_7), "f"(_max3_8));
                            float _max3_24;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(_max3_9), "f"(_max3_10), "f"(_max3_11));
                            float _max3_25;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(_max3_12), "f"(_max3_13), "f"(_max3_14));
                            float _max3_26;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(_max3_15), "f"(_max3_16), "f"(_max3_17));
                            float _max3_27;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(_max3_18), "f"(_max3_19), "f"(_max3_20));
                            float _max3_28;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_28) : "f"(_max3_21), "f"(_max3_22), "f"(_max3_23));
                            float _max3_29;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(_max3_24), "f"(_max3_25), "f"(_max3_26));
                            float _max_27 = max_noftz(_max3_27, sv_split[63]);
                            float _max3_30;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(_max3_28), "f"(_max3_29), "f"(_max_27));
                            score_tile_max = _max3_30;
                        } else if (valid_sparse_cols != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_split[0]), "=f"(sv_split[1]), "=f"(sv_split[2]), "=f"(sv_split[3]), "=f"(sv_split[4]), "=f"(sv_split[5]), "=f"(sv_split[6]), "=f"(sv_split[7]), "=f"(sv_split[8]), "=f"(sv_split[9]), "=f"(sv_split[10]), "=f"(sv_split[11]), "=f"(sv_split[12]), "=f"(sv_split[13]), "=f"(sv_split[14]), "=f"(sv_split[15]), "=f"(sv_split[16]), "=f"(sv_split[17]), "=f"(sv_split[18]), "=f"(sv_split[19]), "=f"(sv_split[20]), "=f"(sv_split[21]), "=f"(sv_split[22]), "=f"(sv_split[23]), "=f"(sv_split[24]), "=f"(sv_split[25]), "=f"(sv_split[26]), "=f"(sv_split[27]), "=f"(sv_split[28]), "=f"(sv_split[29]), "=f"(sv_split[30]), "=f"(sv_split[31])
                                : "r"(s_base));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_split[32]), "=f"(sv_split[33]), "=f"(sv_split[34]), "=f"(sv_split[35]), "=f"(sv_split[36]), "=f"(sv_split[37]), "=f"(sv_split[38]), "=f"(sv_split[39]), "=f"(sv_split[40]), "=f"(sv_split[41]), "=f"(sv_split[42]), "=f"(sv_split[43]), "=f"(sv_split[44]), "=f"(sv_split[45]), "=f"(sv_split[46]), "=f"(sv_split[47]), "=f"(sv_split[48]), "=f"(sv_split[49]), "=f"(sv_split[50]), "=f"(sv_split[51]), "=f"(sv_split[52]), "=f"(sv_split[53]), "=f"(sv_split[54]), "=f"(sv_split[55]), "=f"(sv_split[56]), "=f"(sv_split[57]), "=f"(sv_split[58]), "=f"(sv_split[59]), "=f"(sv_split[60]), "=f"(sv_split[61]), "=f"(sv_split[62]), "=f"(sv_split[63])
                                : "r"(s_base + 32));
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_2 = valid_sparse_cols;
                                if (_lim_2 <= 0) { _slice_lo_mask_0 = 0u; }
                                else if (_lim_2 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_2));
                                }
                            }
                            if (!(_slice_lo_mask_0 & (1u << 0))) sv_split[0] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 1))) sv_split[1] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 2))) sv_split[2] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 3))) sv_split[3] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 4))) sv_split[4] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 5))) sv_split[5] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 6))) sv_split[6] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 7))) sv_split[7] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 8))) sv_split[8] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 9))) sv_split[9] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 10))) sv_split[10] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 11))) sv_split[11] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 12))) sv_split[12] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 13))) sv_split[13] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 14))) sv_split[14] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 15))) sv_split[15] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 16))) sv_split[16] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 17))) sv_split[17] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 18))) sv_split[18] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 19))) sv_split[19] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 20))) sv_split[20] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 21))) sv_split[21] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 22))) sv_split[22] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 23))) sv_split[23] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 24))) sv_split[24] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 25))) sv_split[25] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 26))) sv_split[26] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 27))) sv_split[27] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 28))) sv_split[28] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 29))) sv_split[29] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 30))) sv_split[30] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 31))) sv_split[31] = -CAKE_INF;
                            uint32_t _slice_lo_mask_1;
                            {
                                int _lim_3 = valid_sparse_cols - 32;
                                if (_lim_3 <= 0) { _slice_lo_mask_1 = 0u; }
                                else if (_lim_3 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_3));
                                }
                            }
                            if (!(_slice_lo_mask_1 & (1u << 0))) sv_split[32] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 1))) sv_split[33] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 2))) sv_split[34] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 3))) sv_split[35] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 4))) sv_split[36] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 5))) sv_split[37] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 6))) sv_split[38] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 7))) sv_split[39] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 8))) sv_split[40] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 9))) sv_split[41] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 10))) sv_split[42] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 11))) sv_split[43] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 12))) sv_split[44] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 13))) sv_split[45] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 14))) sv_split[46] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 15))) sv_split[47] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 16))) sv_split[48] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 17))) sv_split[49] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 18))) sv_split[50] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 19))) sv_split[51] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 20))) sv_split[52] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 21))) sv_split[53] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 22))) sv_split[54] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 23))) sv_split[55] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 24))) sv_split[56] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 25))) sv_split[57] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 26))) sv_split[58] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 27))) sv_split[59] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 28))) sv_split[60] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 29))) sv_split[61] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 30))) sv_split[62] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 31))) sv_split[63] = -CAKE_INF;
                            if (any_invalid != 0) {
                                #pragma unroll
                                for (int w_1 = 0; w_1 < 2; w_1++) {
                                    unsigned int mask_word = invalid_cols[w_1];
                                    sv_split[w_1 * 32] = (((mask_word & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32]);
                                    sv_split[w_1 * 32 + 1] = (((mask_word >> 1 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 1]);
                                    sv_split[w_1 * 32 + 2] = (((mask_word >> 2 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 2]);
                                    sv_split[w_1 * 32 + 3] = (((mask_word >> 3 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 3]);
                                    sv_split[w_1 * 32 + 4] = (((mask_word >> 4 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 4]);
                                    sv_split[w_1 * 32 + 5] = (((mask_word >> 5 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 5]);
                                    sv_split[w_1 * 32 + 6] = (((mask_word >> 6 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 6]);
                                    sv_split[w_1 * 32 + 7] = (((mask_word >> 7 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 7]);
                                    sv_split[w_1 * 32 + 8] = (((mask_word >> 8 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 8]);
                                    sv_split[w_1 * 32 + 9] = (((mask_word >> 9 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 9]);
                                    sv_split[w_1 * 32 + 10] = (((mask_word >> 10 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 10]);
                                    sv_split[w_1 * 32 + 11] = (((mask_word >> 11 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 11]);
                                    sv_split[w_1 * 32 + 12] = (((mask_word >> 12 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 12]);
                                    sv_split[w_1 * 32 + 13] = (((mask_word >> 13 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 13]);
                                    sv_split[w_1 * 32 + 14] = (((mask_word >> 14 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 14]);
                                    sv_split[w_1 * 32 + 15] = (((mask_word >> 15 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 15]);
                                    sv_split[w_1 * 32 + 16] = (((mask_word >> 16 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 16]);
                                    sv_split[w_1 * 32 + 17] = (((mask_word >> 17 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 17]);
                                    sv_split[w_1 * 32 + 18] = (((mask_word >> 18 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 18]);
                                    sv_split[w_1 * 32 + 19] = (((mask_word >> 19 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 19]);
                                    sv_split[w_1 * 32 + 20] = (((mask_word >> 20 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 20]);
                                    sv_split[w_1 * 32 + 21] = (((mask_word >> 21 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 21]);
                                    sv_split[w_1 * 32 + 22] = (((mask_word >> 22 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 22]);
                                    sv_split[w_1 * 32 + 23] = (((mask_word >> 23 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 23]);
                                    sv_split[w_1 * 32 + 24] = (((mask_word >> 24 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 24]);
                                    sv_split[w_1 * 32 + 25] = (((mask_word >> 25 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 25]);
                                    sv_split[w_1 * 32 + 26] = (((mask_word >> 26 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 26]);
                                    sv_split[w_1 * 32 + 27] = (((mask_word >> 27 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 27]);
                                    sv_split[w_1 * 32 + 28] = (((mask_word >> 28 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 28]);
                                    sv_split[w_1 * 32 + 29] = (((mask_word >> 29 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 29]);
                                    sv_split[w_1 * 32 + 30] = (((mask_word >> 30 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 30]);
                                    sv_split[w_1 * 32 + 31] = (((mask_word >> 31 & 1) != 0) ? -CAKE_INF : sv_split[w_1 * 32 + 31]);
                                }
                            }
                            float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv_split[0], _reg_reduce_max2_4);
                            row_max_x32_accum(&sv_split[32], _reg_reduce_max2_4);
                            float sv_split_max = row_max_reduce(_reg_reduce_max2_4);
                            score_tile_max = sv_split_max;
                        }
                        {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(s_empty_addr + phase * 8), "r"(0) : "memory");
                            float _max_28 = max_noftz(new_max, score_tile_max);
                            new_max = _max_28;
                            mbarrier_wait(split_pv_p_empty_addr + (phase) * 8, s_wait_phase ^ 1);
                            smem_softmax_warp_pair_exchange[phase * 128 + stats_row] = new_max;
                            asm volatile("barrier.sync %0, 64;" :: "r"(2 + warp % 2) : "memory");
                            float _max_29 = max_noftz(new_max, smem_softmax_warp_pair_exchange[phase * 128 + (stats_row ^ 64)]);
                            new_max = _max_29;
                            if (row_max_val > -CAKE_INF && (new_max - row_max_val) * softmax_scale_log2 <= 8.0f) {
                                new_max = row_max_val;
                            }
                        }
                        float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, (-new_max) * softmax_scale_log2);
                        float delta = _fma_0;
                        float _exp2_0 = approx_exp2(delta);
                        float exp_delta = _exp2_0;
                        float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta : 1.0f);
                        smem_stats_max[phase * 128 + stats_row] = acc_scale;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(stats_addr + (phase) * 8);
                        row_max_val = new_max;
                        float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                        float max_scaled = safe_max * softmax_scale_log2;
                        float block_sum = 0.0f;
                        float kexp_inv = 1.0f;
                        {
                            mbarrier_wait(nvfp4_kexp_full_addr + (phase) * 8, s_wait_phase);
                            unsigned int kexp_bits = smem_kexp[phase * 4];
                            float kexp_f = (float)kexp_bits;
                            float p_bias = kexp_f - max_scaled;
                            unsigned int kexp_inv_bits = 127 - kexp_bits << 23;
                            kexp_inv = __uint_as_float(kexp_inv_bits);
                            {
                                if (valid_sparse_cols != 0) {
                                    const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                                    const float2 _fma_c2_6 = {p_bias, p_bias};
                                    #pragma unroll
                                    for (int _lf = 0; _lf < 32; _lf++)
                                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv_split)[_lf], _fma_b2_5, _fma_c2_6);
                                    #pragma unroll
                                    for (int _le = 0; _le < 64; _le++) {
                                        sv_split[_le] = approx_exp2(sv_split[_le]);
                                    }
                                    uint32_t _fp8_0[16];
                                    {
                                        uint32_t _packed;
                                        asm volatile("{\n\t"
                                            ".reg .b16 _lo;\n\t"
                                            ".reg .b16 _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}"
                                            : "=r"(_packed) : "f"(sv_split[0]), "f"(sv_split[1]),
                                                               "f"(sv_split[2]), "f"(sv_split[3]));
                                        _fp8_0[0] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[4]), "f"(sv_split[5]),
                                                               "f"(sv_split[6]), "f"(sv_split[7]));
                                        _fp8_0[1] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[8]), "f"(sv_split[9]),
                                                               "f"(sv_split[10]), "f"(sv_split[11]));
                                        _fp8_0[2] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[12]), "f"(sv_split[13]),
                                                               "f"(sv_split[14]), "f"(sv_split[15]));
                                        _fp8_0[3] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[16]), "f"(sv_split[17]),
                                                               "f"(sv_split[18]), "f"(sv_split[19]));
                                        _fp8_0[4] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[20]), "f"(sv_split[21]),
                                                               "f"(sv_split[22]), "f"(sv_split[23]));
                                        _fp8_0[5] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[24]), "f"(sv_split[25]),
                                                               "f"(sv_split[26]), "f"(sv_split[27]));
                                        _fp8_0[6] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[28]), "f"(sv_split[29]),
                                                               "f"(sv_split[30]), "f"(sv_split[31]));
                                        _fp8_0[7] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[32]), "f"(sv_split[33]),
                                                               "f"(sv_split[34]), "f"(sv_split[35]));
                                        _fp8_0[8] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[36]), "f"(sv_split[37]),
                                                               "f"(sv_split[38]), "f"(sv_split[39]));
                                        _fp8_0[9] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[40]), "f"(sv_split[41]),
                                                               "f"(sv_split[42]), "f"(sv_split[43]));
                                        _fp8_0[10] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[44]), "f"(sv_split[45]),
                                                               "f"(sv_split[46]), "f"(sv_split[47]));
                                        _fp8_0[11] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[48]), "f"(sv_split[49]),
                                                               "f"(sv_split[50]), "f"(sv_split[51]));
                                        _fp8_0[12] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[52]), "f"(sv_split[53]),
                                                               "f"(sv_split[54]), "f"(sv_split[55]));
                                        _fp8_0[13] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[56]), "f"(sv_split[57]),
                                                               "f"(sv_split[58]), "f"(sv_split[59]));
                                        _fp8_0[14] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[60]), "f"(sv_split[61]),
                                                               "f"(sv_split[62]), "f"(sv_split[63]));
                                        _fp8_0[15] = _packed;
                                    }
                                    {
                                        int p_stage_base = smem_p_fp8_addr + (unsigned int)(phase * 8192);
                                        #pragma unroll
                                        for (int p_vec = 0; p_vec < 4; p_vec++) {
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_stage_base + (my_row * 128 + (n_half * 4 + p_vec) * 16 ^ (my_row * 128 + (n_half * 4 + p_vec) * 16 >> 7 & 7) << 4))), "r"(_fp8_0[p_vec * 4]), "r"(_fp8_0[p_vec * 4 + 1]), "r"(_fp8_0[p_vec * 4 + 2]), "r"(_fp8_0[p_vec * 4 + 3]) : "memory");
                                        }
                                    }
                                } else {
                                    int zero_p_stage = smem_p_fp8_addr + (unsigned int)(phase * 8192);
                                    #pragma unroll
                                    for (int zero_p_vec = 0; zero_p_vec < 4; zero_p_vec++) {
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((zero_p_stage + (my_row * 128 + (n_half * 4 + zero_p_vec) * 16 ^ (my_row * 128 + (n_half * 4 + zero_p_vec) * 16 >> 7 & 7) << 4))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                                    }
                                }
                            }
                        }
                        {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        {
                            asm volatile("" ::: "memory");
                        }
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(p_full_addr + phase * 8), "r"(0) : "memory");
                        {
                            {
                                asm volatile("" ::: "memory");
                                if (valid_sparse_cols != 0) {
                                    const float2* _reg_reduce_src2_7 = reinterpret_cast<const float2*>(&sv_split[0]);
                                    float2 _reg_reduce_sum2_7_0 = make_float2(0.0f, 0.0f);
                                    float2 _reg_reduce_sum2_7_1 = make_float2(0.0f, 0.0f);
                                    float2 _reg_reduce_sum2_7_2 = make_float2(0.0f, 0.0f);
                                    float2 _reg_reduce_sum2_7_3 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int _rr = 0; _rr < 8; _rr++) {
                                        add_f32x2_inplace(&_reg_reduce_sum2_7_0, _reg_reduce_src2_7[_rr * 4 + 0]);
                                        add_f32x2_inplace(&_reg_reduce_sum2_7_1, _reg_reduce_src2_7[_rr * 4 + 1]);
                                        add_f32x2_inplace(&_reg_reduce_sum2_7_2, _reg_reduce_src2_7[_rr * 4 + 2]);
                                        add_f32x2_inplace(&_reg_reduce_sum2_7_3, _reg_reduce_src2_7[_rr * 4 + 3]);
                                    }
                                    add_f32x2_inplace(&_reg_reduce_sum2_7_0, _reg_reduce_sum2_7_1);
                                    add_f32x2_inplace(&_reg_reduce_sum2_7_2, _reg_reduce_sum2_7_3);
                                    add_f32x2_inplace(&_reg_reduce_sum2_7_0, _reg_reduce_sum2_7_2);
                                    float sv_split_sum = _reg_reduce_sum2_7_0.x + _reg_reduce_sum2_7_0.y;
                                    block_sum = sv_split_sum;
                                }
                                block_sum = block_sum * kexp_inv;
                                float _fma_2 = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                                row_sum_val = _fma_2;
                            }
                        }
                    }
                    {
                        mbarrier_wait(source_sum_empty_addr, _phase_source_sum_empty_0);
                        _phase_source_sum_empty_0 ^= 1;
                    }
                    smem_stats_sum[stats_row] = row_sum_val;
                    smem_stats_final_max[stats_row] = row_max_val;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(sum_ready_addr);
                    softmax_tile_cursor = softmax_tile_cursor + num_kv_tiles_1;
                    sm_item = sm_item + 1;
                }
                {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_2) * 8, _phase_source_work_full_2);
                    uint32_t _clc_valid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_4)
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_12)
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_13)
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_14 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_14)
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_2 * 8), "r"(0) : "memory");
                    source_work_stage_2 += 1;
                    if (source_work_stage_2 == 2) { source_work_stage_2 = 0; _phase_source_work_full_2 ^= 1; }
                    if (_clc_valid_4 == 0) {
                        break;
                    }
                    source_work_x_2 = _clc_ctaid_12 + (unsigned int)cta_rank;
                    source_work_z_2 = _clc_ctaid_14;
                    int sm_next_q = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1);
                    if (sm_next_q < num_query_tokens) {
                        unsigned int qi_slot_2 = sm_ring_seq & 1;
                        if (warp % 4 == 0) {
                            mbarrier_wait(nvfp4_q_ring_empty_addr + (qi_slot_2) * 8, sm_ring_seq >> 1 & 1 ^ 1);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(nvfp4_q_ring_full_addr + (qi_slot_2) * 8, 8192);
                                tma_4d_gmem2smem(smem_q_ring_addr + qi_slot_2 * 8192, (&tmap_q), 0, bid % 2 * 64, 0, sm_next_q, nvfp4_q_ring_full_addr + (qi_slot_2) * 8);
                            }
                        }
                        unsigned int qi_slot_0_1 = sm_ring_seq + 1 & 1;
                        if (warp % 4 == 0) {
                            mbarrier_wait(nvfp4_q_ring_empty_addr + (qi_slot_0_1) * 8, sm_ring_seq + 1 >> 1 & 1 ^ 1);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(nvfp4_q_ring_full_addr + (qi_slot_0_1) * 8, 8192);
                                tma_4d_gmem2smem(smem_q_ring_addr + qi_slot_0_1 * 8192, (&tmap_q), 0, bid % 2 * 64, 1, sm_next_q, nvfp4_q_ring_full_addr + (qi_slot_0_1) * 8);
                            }
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // correction_wg_main
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            const int wg_dummy_inc_1 = 0;
            int all_num_kv_tiles_2 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base_2 = ((1) ? warp % 2 * 32 : warp % 4 * 32);
            const int n_half_1 = ((1) ? (int)(warp % 4 / 2) : 0);
            const int my_row_1 = tmem_row_base_2 + lane;
            const int stats_row_1 = n_half_1 * 64 + my_row_1;
            const int corr_row = tmem_row_base_2 << 16;
            int correction_tile_cursor = 0;
            const int csf_quad_row = (int)(warp % 4 * 32);
            const int csf_tok_a = (warp % 4 >> 1) * 64 + lane;
            const int csf_tok_b = csf_tok_a + 32;
            int ep_item = 0;
            unsigned int source_work_stage_3 = 0;
            unsigned int source_work_x_3 = blockIdx.x;
            unsigned int source_work_z_3 = blockIdx.z;
            unsigned int _phase_q_full_0_2 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_sum_ready_0 = 0;
            unsigned int _phase_source_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = 0; work_idx_2 < 514; work_idx_2++) {
                int split_idx_2 = 0;
                int query_idx_2 = source_work_x_3 >> 1 >> 1;
                int v_chunk = source_work_x_3 >> 1 & 1;
                {
                    split_idx_2 = 0;
                    query_idx_2 = source_work_x_3 >> 1;
                    v_chunk = bid % 2;
                }
                int source_work_valid_3 = source_work_z_3 * (unsigned int)max_q_len + (source_work_x_3 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_3 = source_work_z_3 * (unsigned int)max_q_len + (source_work_x_3 >> 1);
                if (source_work_valid_3 != 0) {
                    {
                        int _max_32 = ((sparse_topk_lens[mapped_query_idx_3] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_3] + sparse_topk_lens_offset) : (0));
                        int _min_12 = ((_max_32) < (swa_width) ? (_max_32) : (swa_width));
                        int main_active_3 = _min_12;
                        int extra_active_3 = 0;
                        if (compressed_width > 0) {
                            int _max_33 = ((extra_topk_lens[mapped_query_idx_3]) > (0) ? (extra_topk_lens[mapped_query_idx_3]) : (0));
                            int _min_13 = ((_max_33) < (compressed_width) ? (_max_33) : (compressed_width));
                            extra_active_3 = _min_13;
                        }
                        int n_main_tiles_3 = (main_active_3 + 128 - 1) / 128;
                        int n_extra_tiles_3 = (extra_active_3 + 128 - 1) / 128;
                        int _max_34 = ((n_main_tiles_3 + n_extra_tiles_3) > (1) ? (n_main_tiles_3 + n_extra_tiles_3) : (1));
                        int active_topk_2 = _max_34 * 128;
                        all_num_kv_tiles_2 = (active_topk_2 + 128 - 1) / 128;
                    }
                    int tiles_per_split_2 = all_num_kv_tiles_2 + 1 - 1;
                    int first_tile_2 = split_idx_2 * tiles_per_split_2;
                    int num_kv_tiles_2 = tiles_per_split_2;
                    if (num_kv_tiles_2 > 1) {
                        int csf_pt0 = correction_tile_cursor + 1;
                        int csf_stage0 = csf_pt0 & 1;
                        mbarrier_wait(k_full_addr + (csf_stage0) * 8, csf_pt0 >> 1 & 1);
                        mbarrier_wait(k_empty_addr + (csf_pt0 + 1 & 1) * 8, csf_pt0 + 1 >> 1 & 1 ^ 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int csf_base0 = smem_k_a_addr + (unsigned int)(csf_stage0 * 28672) + 24576;
                        unsigned int csf_a0[8];
                        unsigned int csf_b0[8];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[(0) + 3]))
                            : "r"(csf_base0 + csf_tok_a * 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a0[(4) + 3]))
                            : "r"(csf_base0 + csf_tok_a * 32 + 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[0])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[(0) + 3]))
                            : "r"(csf_base0 + csf_tok_b * 32));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[4])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b0[(4) + 3]))
                            : "r"(csf_base0 + csf_tok_b * 32 + 16));
                        unsigned int csf_regs0[32];
                        #pragma unroll
                        for (int csf_s = 0; csf_s < 7; csf_s++) {
                            csf_regs0[4 * csf_s] = csf_a0[csf_s];
                            csf_regs0[4 * csf_s + 1] = csf_b0[csf_s];
                            csf_regs0[4 * csf_s + 2] = 0;
                            csf_regs0[4 * csf_s + 3] = 0;
                        }
                        #pragma unroll
                        for (int csf_pad = 28; csf_pad < 32; csf_pad++) {
                            csf_regs0[csf_pad] = 0;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + 96 + (unsigned int)(csf_quad_row << 16)), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[0])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[1])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[2])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[3])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[4])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[5])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[6])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[7])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[8])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[9])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[10])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[11])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[12])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[13])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[14])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs0[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + 96 + 16 + (unsigned int)(csf_quad_row << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs0 + 16)[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(nvfp4_sf_full_addr + csf_stage0 * 8), "r"(0) : "memory");
                            }
                        }
                    }
                    {
                        int first_stats_tile = correction_tile_cursor;
                        int first_stats_phase = first_stats_tile & 1;
                        int first_stats_wait_phase = first_stats_tile >> 1 & 1;
                        mbarrier_wait(stats_addr + (first_stats_phase) * 8, first_stats_wait_phase);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_empty_addr), "r"(0) : "memory");
                    }
                    #pragma unroll 1
                    for (int tile_2 = 1; tile_2 < num_kv_tiles_2; tile_2++) {
                        int pipeline_tile_1 = correction_tile_cursor + tile_2;
                        int phase_1 = pipeline_tile_1 & 1;
                        int stats_wait_phase = pipeline_tile_1 >> 1 & 1;
                        if (num_kv_tiles_2 > tile_2 + 1) {
                            int csf_pt1 = pipeline_tile_1 + 1;
                            int csf_stage1 = csf_pt1 & 1;
                            mbarrier_wait(k_full_addr + (csf_stage1) * 8, csf_pt1 >> 1 & 1);
                            mbarrier_wait(k_empty_addr + (csf_pt1 + 1 & 1) * 8, csf_pt1 + 1 >> 1 & 1 ^ 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int csf_base1 = smem_k_a_addr + (unsigned int)(csf_stage1 * 28672) + 24576;
                            unsigned int csf_a1[8];
                            unsigned int csf_b1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[(0) + 3]))
                                : "r"(csf_base1 + csf_tok_a * 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_a1[(4) + 3]))
                                : "r"(csf_base1 + csf_tok_a * 32 + 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[0])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[(0) + 3]))
                                : "r"(csf_base1 + csf_tok_b * 32));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[4])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csf_b1[(4) + 3]))
                                : "r"(csf_base1 + csf_tok_b * 32 + 16));
                            unsigned int csf_regs1[32];
                            #pragma unroll
                            for (int csf_s_1 = 0; csf_s_1 < 7; csf_s_1++) {
                                csf_regs1[4 * csf_s_1] = csf_a1[csf_s_1];
                                csf_regs1[4 * csf_s_1 + 1] = csf_b1[csf_s_1];
                                csf_regs1[4 * csf_s_1 + 2] = 0;
                                csf_regs1[4 * csf_s_1 + 3] = 0;
                            }
                            #pragma unroll
                            for (int csf_pad_1 = 28; csf_pad_1 < 32; csf_pad_1++) {
                                csf_regs1[csf_pad_1] = 0;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(taddr + 96 + (unsigned int)(csf_quad_row << 16)), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[0])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[1])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[2])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[3])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[4])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[5])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[6])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[7])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[8])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[9])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[10])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[11])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[12])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[13])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[14])), "r"(*reinterpret_cast<const uint32_t*>(&csf_regs1[15])));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(taddr + 96 + 16 + (unsigned int)(csf_quad_row << 16)), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(csf_regs1 + 16)[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
                            if (warp == 4) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b32 remAddr32;\n\t"
                                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                        "}"
                                        :: "r"(nvfp4_sf_full_addr + csf_stage1 * 8), "r"(0) : "memory");
                                }
                            }
                        }
                        mbarrier_wait(stats_addr + (phase_1) * 8, stats_wait_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int ep_t0 = ep_item < 48 && tile_2 == 1;
                        float acc_scale_1 = smem_stats_max[phase_1 * 128 + stats_row_1];
                        {
                            {
                                int prev_output_tile = pipeline_tile_1 - 1;
                                int o_full_phase = prev_output_tile & 1;
                                uint32_t _mbar_token_5 = mbarrier_try_wait(o_full_addr, o_full_phase);
                                mbarrier_wait_token(o_full_addr, o_full_phase, _mbar_token_5);
                            }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        {
                            int _vote_1 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                            int any_rescale = _vote_1;
                            if (any_rescale != 0) {
                                #pragma unroll
                                for (int vs = 0; vs < 2; vs++) {
                                    int o_base = taddr + 256 + (unsigned int)(vs * 128) + (unsigned int)corr_row;
                                    #pragma unroll
                                    for (int c = 0; c < 128; c += 16) {
                                        float _tmem_load_3[16];
                                        tmem_ld_x16(&_tmem_load_3[0], o_base + c);
                                        const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 8; _ls++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                                        tmem_st_x16_f32(o_base + c, _tmem_load_3);
                                    }
                                }
                                {
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                }
                            }
                        }
                        {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(o_empty_addr), "r"(0) : "memory");
                        }
                    }
                    {
                        int last_output_tile = correction_tile_cursor + num_kv_tiles_2 - 1;
                        int o_full_phase_1 = last_output_tile & 1;
                        uint32_t _mbar_token_6 = mbarrier_try_wait(o_full_addr, o_full_phase_1);
                        mbarrier_wait_token(o_full_addr, o_full_phase_1, _mbar_token_6);
                    }
                    mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
                    _phase_sum_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float total_sum = smem_stats_sum[stats_row_1];
                    float final_max = smem_stats_final_max[stats_row_1];
                    {
                        mbarrier_arrive(source_sum_empty_addr);
                    }
                    {
                        smem_corr_warp_pair_exchange[stats_row_1] = total_sum;
                        asm volatile("barrier.sync %0, 64;" :: "r"(4 + warp % 2) : "memory");
                        total_sum = total_sum + smem_corr_warp_pair_exchange[stats_row_1 ^ 64];
                    }
                    float _rcp_0 = approx_rcp(total_sum);
                    float inv_sum = ((total_sum > 0.0f) ? _rcp_0 : 0.0f);
                    int head_idx = my_row_1;
                    {
                        head_idx = bid % 2 * 64 + my_row_1;
                    }
                    int direct_o_offset = (mapped_query_idx_3 * num_heads + head_idx) * 512 + v_chunk * 256;
                    int partial_o_offset = (mapped_query_idx_3 * num_heads + head_idx + split_idx_2) * 512 + ((0) ? v_chunk : n_half_1) * 256;
                    int o_offset = partial_o_offset;
                    if (n_half_1 == 0 && head_idx < num_heads) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(total_sum));
                        LSE[mapped_query_idx_3 * num_heads + head_idx] = ((total_sum > 0.0f) ? final_max * softmax_scale_log2_1 + _log2_1 : -CAKE_INF);
                    }
                    #pragma unroll 1
                    for (int vs_1 = 0; vs_1 < 2; vs_1++) {
                        int o_base_epi = taddr + 256 + (unsigned int)(vs_1 * 128) + (unsigned int)corr_row;
                        #pragma unroll 1
                        for (int c_1 = 0; c_1 < 128; c_1 += 16) {
                            float _tmem_load_4[16];
                            tmem_ld_x16(&_tmem_load_4[0], o_base_epi + c_1);
                            int gmem_base = o_offset + vs_1 * 128 + c_1;
                            if (head_idx < num_heads) {
                                {
                                    const float2 _prescale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_1);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_4[0 + _ps] *= inv_sum * output_scale;
                                    #endif
                                    {
                                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_4[0 + 8], _tmem_load_4[0 + 9]);
                                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_4[0 + 10], _tmem_load_4[0 + 11]);
                                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_4[0 + 12], _tmem_load_4[0 + 13]);
                                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_4[0 + 14], _tmem_load_4[0 + 15]);
                                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(&((__nv_bfloat16*)(O + gmem_base))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                    correction_tile_cursor = correction_tile_cursor + num_kv_tiles_2;
                    ep_item = ep_item + 1;
                }
                {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_3) * 8, _phase_source_work_full_3);
                    uint32_t _clc_valid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_5)
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_15 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_15)
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_16 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_16)
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_17 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_17)
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_3 * 8), "r"(0) : "memory");
                    source_work_stage_3 += 1;
                    if (source_work_stage_3 == 2) { source_work_stage_3 = 0; _phase_source_work_full_3 ^= 1; }
                    if (_clc_valid_5 == 0) {
                        break;
                    }
                    source_work_x_3 = _clc_ctaid_15 + (unsigned int)cta_rank;
                    source_work_z_3 = _clc_ctaid_17;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg2_dummy_1 = 0;
            int all_num_kv_tiles_3 = (sparse_topk + 128 - 1) / 128;
            unsigned int mma_k_stage = 0;
            unsigned int mma_v_stage = 0;
            unsigned int mma_kv_stage = 0;
            int mma_tile_cursor = 0;
            int mma_item = 0;
            unsigned int source_work_stage_4 = 0;
            unsigned int source_work_x_4 = blockIdx.x;
            unsigned int source_work_z_4 = blockIdx.z;
            {
                if (cta_rank == 0) {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(s_empty_addr, 1);
                    mbarrier_wait_token(s_empty_addr, 1, _mbar_token_0);
                    uint32_t _mbar_token_1 = mbarrier_try_wait(s_empty_addr + 8, 1);
                    mbarrier_wait_token(s_empty_addr + 8, 1, _mbar_token_1);
                }
            }
            unsigned int _phase_s_seeded_0 = 0;
            unsigned int _phase_q_full_0_3 = 0;
            unsigned int _phase_q_pair_ready_0 = 0;
            unsigned int _phase_nvfp4_sf_full = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_kv_pair_ready = 0;
            unsigned int _phase_source_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_3 = 0; work_idx_3 < 514; work_idx_3++) {
                int split_idx_3 = 0;
                {
                    split_idx_3 = 0;
                }
                {
                    int query_idx_3 = source_work_x_4 >> 1;
                }
                int source_work_valid_4 = source_work_z_4 * (unsigned int)max_q_len + (source_work_x_4 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_4 = source_work_z_4 * (unsigned int)max_q_len + (source_work_x_4 >> 1);
                if (source_work_valid_4 != 0) {
                    {
                        int _max_17 = ((sparse_topk_lens[mapped_query_idx_4] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[mapped_query_idx_4] + sparse_topk_lens_offset) : (0));
                        int _min_8 = ((_max_17) < (swa_width) ? (_max_17) : (swa_width));
                        int main_active_4 = _min_8;
                        int extra_active_4 = 0;
                        if (compressed_width > 0) {
                            int _max_18 = ((extra_topk_lens[mapped_query_idx_4]) > (0) ? (extra_topk_lens[mapped_query_idx_4]) : (0));
                            int _min_9 = ((_max_18) < (compressed_width) ? (_max_18) : (compressed_width));
                            extra_active_4 = _min_9;
                        }
                        int n_main_tiles_4 = (main_active_4 + 128 - 1) / 128;
                        int n_extra_tiles_4 = (extra_active_4 + 128 - 1) / 128;
                        int _max_19 = ((n_main_tiles_4 + n_extra_tiles_4) > (1) ? (n_main_tiles_4 + n_extra_tiles_4) : (1));
                        int active_topk_3 = _max_19 * 128;
                        all_num_kv_tiles_3 = (active_topk_3 + 128 - 1) / 128;
                    }
                    int tiles_per_split_3 = all_num_kv_tiles_3 + 1 - 1;
                    int first_tile_3 = split_idx_3 * tiles_per_split_3;
                    int num_kv_tiles_3 = tiles_per_split_3;
                    {
                        if (cta_rank == 0) {
                            mbarrier_wait(q_full_addr, _phase_q_full_0_3);
                            _phase_q_full_0_3 ^= 1;
                        }
                    }
                    int first_pv = 1;
                    int mma_on = mma_item < 48;
                    {
                        int first_pipeline_tile = mma_tile_cursor;
                        int first_phase = first_pipeline_tile & 1;
                        int first_score_col = ((first_phase != 0) ? 128 : 0);
                        if (cta_rank == 0) {
                            mbarrier_wait(nvfp4_sf_full_addr + (mma_k_stage) * 8, _phase_nvfp4_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_q_a_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int _mma_b_lo_0 = (((smem_k_a_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 1792;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 0, b_desc + 0,
                                        0x8200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 2, b_desc + 2,
                                        0x8200480U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 4, b_desc + 4,
                                        0x8200480U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 6, b_desc + 6,
                                        0x8200480U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 12, 1);
                                }
                            }
                            int _mma_a_lo_1 = (((smem_q_b_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int _mma_b_lo_1 = (((smem_k_b_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 1792;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 0, b_desc + 0,
                                        0x8200480U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 16 + 0, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 2, b_desc + 2,
                                        0x8200480U, tmem_tmem_sfa + 16 + 4, tmem_tmem_sfb + 16 + 4, 1);
                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (first_score_col)), a_desc + 4, b_desc + 4,
                                        0x8200480U, tmem_tmem_sfa + 16 + 8, tmem_tmem_sfb + 16 + 8, 1);
                                }
                            }
                            int _mma_a_lo_2 = (((smem_q_rope_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int _mma_b_lo_2 = (((smem_k_c_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 1792;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (first_score_col))), "r"(1));
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                            }
                            mma_k_stage += 1;
                            if (mma_k_stage == 2) { mma_k_stage = 0; _phase_nvfp4_sf_full ^= 1; }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(s_full_addr + (first_phase) * 8, (uint16_t)(3));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int tile_3 = 1; tile_3 < num_kv_tiles_3; tile_3++) {
                        int pipeline_tile_2 = mma_tile_cursor + tile_3;
                        int phase_2 = pipeline_tile_2 & 1;
                        int score_col = ((phase_2 != 0) ? 128 : 0);
                        {
                            if (cta_rank == 0) {
                                mbarrier_wait(nvfp4_sf_full_addr + (mma_k_stage) * 8, _phase_nvfp4_sf_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_a_lo_3 = (((smem_q_a_addr) >> 4) & 0x3FFF) + (0) * 512;
                                int _mma_b_lo_3 = (((smem_k_a_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 1792;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 0, b_desc + 0,
                                            0x8200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 0);
                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 2, b_desc + 2,
                                            0x8200480U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 4, b_desc + 4,
                                            0x8200480U, tmem_tmem_sfa + 8, tmem_tmem_sfb + 8, 1);
                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 6, b_desc + 6,
                                            0x8200480U, tmem_tmem_sfa + 12, tmem_tmem_sfb + 12, 1);
                                    }
                                }
                                int _mma_a_lo_4 = (((smem_q_b_addr) >> 4) & 0x3FFF) + (0) * 512;
                                int _mma_b_lo_4 = (((smem_k_b_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 1792;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 0, b_desc + 0,
                                            0x8200480U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 16 + 0, 1);
                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 2, b_desc + 2,
                                            0x8200480U, tmem_tmem_sfa + 16 + 4, tmem_tmem_sfb + 16 + 4, 1);
                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_tmem_scratch + (score_col)), a_desc + 4, b_desc + 4,
                                            0x8200480U, tmem_tmem_sfa + 16 + 8, tmem_tmem_sfb + 16 + 8, 1);
                                    }
                                }
                                int _mma_a_lo_5 = (((smem_q_rope_addr) >> 4) & 0x3FFF) + (0) * 512;
                                int _mma_b_lo_5 = (((smem_k_c_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 1792;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                                if (cta_rank == 0) {
                                    elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                                }
                                mma_k_stage += 1;
                                if (mma_k_stage == 2) { mma_k_stage = 0; _phase_nvfp4_sf_full ^= 1; }
                                if (cta_rank == 0) {
                                    elect_commit_cg2_multicast(s_full_addr + (phase_2) * 8, (uint16_t)(3));
                                }
                                int next_s_tile = pipeline_tile_2 + 1;
                                int next_s_stage = next_s_tile & 1;
                                int next_s_empty_phase = next_s_tile >> 1 & 1 ^ 1;
                                uint32_t _mbar_token_2 = mbarrier_try_wait(s_empty_addr + (next_s_stage) * 8, next_s_empty_phase);
                                mbarrier_wait_token(s_empty_addr + (next_s_stage) * 8, next_s_empty_phase, _mbar_token_2);
                            }
                        }
                    }
                    {
                        if (cta_rank == 0) {
                            elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        }
                    }
                    int last_pipeline_tile = mma_tile_cursor + num_kv_tiles_3 - 1;
                    int last_phase = last_pipeline_tile & 1;
                    int drain_wait_phase = last_pipeline_tile >> 1 & 1;
                    {
                        int final_s_tile = last_pipeline_tile + 2;
                        int final_s_stage = final_s_tile & 1;
                        int final_s_empty_phase = final_s_tile >> 1 & 1 ^ 1;
                        if (cta_rank == 0) {
                            uint32_t _mbar_token_3 = mbarrier_try_wait(s_empty_addr + (final_s_stage) * 8, final_s_empty_phase);
                            mbarrier_wait_token(s_empty_addr + (final_s_stage) * 8, final_s_empty_phase, _mbar_token_3);
                        }
                    }
                    mma_tile_cursor = mma_tile_cursor + num_kv_tiles_3;
                    mma_item = mma_item + 1;
                }
                {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_4) * 8, _phase_source_work_full_4);
                    uint32_t _clc_valid_3 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_3)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_9)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_10 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_10)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_11 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_11)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_4 * 8), "r"(0) : "memory");
                    source_work_stage_4 += 1;
                    if (source_work_stage_4 == 2) { source_work_stage_4 = 0; _phase_source_work_full_4 ^= 1; }
                    if (_clc_valid_3 == 0) {
                        break;
                    }
                    source_work_x_4 = _clc_ctaid_9 + (unsigned int)cta_rank;
                    source_work_z_4 = _clc_ctaid_11;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            {
                int tmem_dealloc_peer_rank = bid % 2 ^ 1;
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_peer_addr), "r"(tmem_dealloc_peer_rank) : "memory");
                mbarrier_wait(tmem_dealloc_peer_addr, 0);
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            const int wg2_dummy_2 = 0;
            unsigned int _phase_source_throttle_full = 0;
            unsigned int _phase_source_work_empty = 1;
            unsigned int _phase_source_work_full_5 = 0;
            unsigned int _phase_q_full_0_4 = 0;
            {
                unsigned int source_work_stage_5 = 0;
                unsigned int source_throttle_stage_1 = 0;
                if (cta_rank == 0) {
                    #pragma unroll 1
                    for (unsigned int _source_work = 0; _source_work < 514; _source_work++) {
                        mbarrier_wait(source_throttle_full_addr + (source_throttle_stage_1) * 8, _phase_source_throttle_full);
                        mbarrier_arrive(source_throttle_empty_addr + (source_throttle_stage_1) * 8);
                        source_throttle_stage_1 += 1;
                        if (source_throttle_stage_1 == 2) { source_throttle_stage_1 = 0; _phase_source_throttle_full ^= 1; }
                        mbarrier_wait(source_work_empty_addr + (source_work_stage_5) * 8, _phase_source_work_empty);
                        if (lane < 2) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(source_work_full_addr + source_work_stage_5 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                    " [%0], [%1];"
                                :: "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16), "r"(source_work_full_addr + source_work_stage_5 * 8)
                                : "memory");
                        }
                        mbarrier_wait(source_work_full_addr + (source_work_stage_5) * 8, _phase_source_work_full_5);
                        uint32_t _clc_valid_6 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %0, 1, 0, p1;\n\t"
                            "}\n"
                            : "=r"(_clc_valid_6)
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_18 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_18)
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_19 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_19)
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_20 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_20)
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        source_work_stage_5 += 1;
                        if (source_work_stage_5 == 2) { source_work_stage_5 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_5 ^= 1; }
                        if (_clc_valid_6 == 0) {
                            break;
                        }
                    }
                    #pragma unroll
                    for (unsigned int _source_tail = 0; _source_tail < 2; _source_tail++) {
                        mbarrier_wait(source_work_empty_addr + (source_work_stage_5) * 8, _phase_source_work_empty);
                        source_work_stage_5 += 1;
                        if (source_work_stage_5 == 2) { source_work_stage_5 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_5 ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            const int wg3_dummy = 0;
            unsigned int source_work_stage_6 = 0;
            unsigned int source_work_x_5 = blockIdx.x;
            unsigned int source_work_z_5 = blockIdx.z;
            int pv_tile_cursor = 0;
            unsigned int pv_v_stage = 0;
            unsigned int _phase_source_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _pv_work = 0; _pv_work < 514; _pv_work++) {
                int pv_valid = source_work_z_5 * (unsigned int)max_q_len + (source_work_x_5 >> 1) < (unsigned int)num_query_tokens;
                if (pv_valid != 0) {
                    int pv_query = source_work_z_5 * (unsigned int)max_q_len + (source_work_x_5 >> 1);
                    int _max_35 = ((sparse_topk_lens[pv_query] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[pv_query] + sparse_topk_lens_offset) : (0));
                    int _min_14 = ((_max_35) < (swa_width) ? (_max_35) : (swa_width));
                    int pv_main = _min_14;
                    int pv_extra = 0;
                    if (compressed_width > 0) {
                        int _max_36 = ((extra_topk_lens[pv_query]) > (0) ? (extra_topk_lens[pv_query]) : (0));
                        int _min_15 = ((_max_36) < (compressed_width) ? (_max_36) : (compressed_width));
                        pv_extra = _min_15;
                    }
                    int _max_37 = (((pv_main + 128 - 1) / 128 + (pv_extra + 128 - 1) / 128) > (1) ? ((pv_main + 128 - 1) / 128 + (pv_extra + 128 - 1) / 128) : (1));
                    int pv_tiles = _max_37;
                    int pv_first = 1;
                    #pragma unroll 1
                    for (int pv_tile = 0; pv_tile < pv_tiles; pv_tile++) {
                        int pv_cursor = pv_tile_cursor + pv_tile;
                        int pv_phase = pv_cursor & 1;
                        int pv_wait_phase = pv_cursor >> 1 & 1;
                        if (cta_rank == 0) {
                            mbarrier_wait(p_full_addr + (pv_phase) * 8, pv_wait_phase);
                            mbarrier_wait(o_empty_addr, pv_cursor & 1);
                            mbarrier_wait(v_full_addr + (pv_v_stage) * 8, pv_wait_phase);
                            int _mma_a_lo_8 = (((smem_p_fp8_addr) >> 4) & 0x3FFF) + (pv_phase) * 512;
                            int _mma_b_lo_8 = ((((smem_v_full_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_v_stage) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"((tmem_tmem_scratch + (256))), "r"(((pv_first) ? 0 : 1)));
                            int _mma_a_lo_9 = (((smem_p_fp8_addr) >> 4) & 0x3FFF) + (pv_phase) * 512;
                            int _mma_b_lo_9 = ((((smem_v_full_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (pv_v_stage) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem_scratch + (384))), "r"(((pv_first) ? 0 : 1)));
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(v_empty_addr + (pv_v_stage) * 8, (uint16_t)(3));
                            }
                            pv_v_stage += 1;
                            if (pv_v_stage == 2) { pv_v_stage = 0; }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(split_pv_p_empty_addr + (pv_phase) * 8, (uint16_t)(3));
                            }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                            }
                        }
                        pv_first = 0;
                    }
                    pv_tile_cursor = pv_tile_cursor + pv_tiles;
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_6) * 8, _phase_source_work_full_6);
                uint32_t _clc_valid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_7)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_21 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_21)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_22 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_22)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_23 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_23)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + source_work_stage_6 * 8), "r"(0) : "memory");
                source_work_stage_6 += 1;
                if (source_work_stage_6 == 2) { source_work_stage_6 = 0; _phase_source_work_full_6 ^= 1; }
                if (_clc_valid_7 == 0) {
                    break;
                }
                source_work_x_5 = _clc_ctaid_21 + (unsigned int)cta_rank;
                source_work_z_5 = _clc_ctaid_23;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }

    // Cleanup
}

} // extern "C"
