typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) BlackwellMsaTensorMap { uint64_t opaque[16]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define BLACKWELL_MSA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 16
#define TMEM_TMEM_O0_OFFSET 32
#define TMEM_TMEM_O1_OFFSET 48
#define TMEM_TMEM_STATS0_OFFSET 64
#define TMEM_TMEM_STATS1_OFFSET 80
#define TMEM_PREFILL_SCORES_0_OFFSET 0
#define TMEM_PREFILL_SCORES_1_OFFSET 128
#define TMEM_PREFILL_OUTPUT_0_OFFSET 256
#define TMEM_PREFILL_OUTPUT_1_OFFSET 384
#define NUM_DECODE_KV_STAGES 4
#define NUM_PREFILL_KV_PIPELINE_STAGES 2
#define SMEM_SMEM_CORR0_OFF 1024
#define SMEM_SMEM_CORR0_STAGE_BYTES 64
#define SMEM_SMEM_CORR0_STRIDE 64
#define SMEM_SMEM_CORR1_OFF 1088
#define SMEM_SMEM_CORR1_STAGE_BYTES 64
#define SMEM_SMEM_CORR1_STRIDE 64
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 4096
#define SMEM_SMEM_QT_STRIDE 4096
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_KV_FP8_OFF 151168
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_P0_OFF 137216
#define SMEM_SMEM_P0_STAGE_BYTES 4096
#define SMEM_SMEM_P0_STRIDE 4096
#define SMEM_SMEM_P1_OFF 141312
#define SMEM_SMEM_P1_STAGE_BYTES 4096
#define SMEM_SMEM_P1_STRIDE 4096
#define SMEM_SMEM_PAGE_INDICES_OFF 145408
#define SMEM_SMEM_PAGE_INDICES_STAGE_BYTES 2048
#define SMEM_SMEM_PAGE_INDICES_STRIDE 2048
#define SMEM_PREFILL_SCALE_OFF 1024
#define SMEM_PREFILL_SCALE_STAGE_BYTES 3072
#define SMEM_PREFILL_SCALE_STRIDE 3072
#define SMEM_PREFILL_Q0_OFF 4096
#define SMEM_PREFILL_Q0_STAGE_BYTES 32768
#define SMEM_PREFILL_Q0_STRIDE 32768
#define SMEM_PREFILL_PARTIAL_TILE_OFF 4096
#define SMEM_PREFILL_PARTIAL_TILE_STAGE_BYTES 65536
#define SMEM_PREFILL_PARTIAL_TILE_STRIDE 65536
#define SMEM_PREFILL_SPLIT_WEIGHTS_OFF 4096
#define SMEM_PREFILL_SPLIT_WEIGHTS_STAGE_BYTES 4096
#define SMEM_PREFILL_SPLIT_WEIGHTS_STRIDE 4096
#define SMEM_PREFILL_Q1_OFF 36864
#define SMEM_PREFILL_Q1_STAGE_BYTES 32768
#define SMEM_PREFILL_Q1_STRIDE 32768
#define SMEM_PREFILL_KV_OFF 69632
#define SMEM_PREFILL_KV_STAGE_BYTES 32768
#define SMEM_PREFILL_KV_STRIDE 32768
#define SMEM_PREFILL_V_OFF 69632
#define SMEM_PREFILL_V_STAGE_BYTES 32768
#define SMEM_PREFILL_V_STRIDE 32768
#define SMEM_TASK_OFFSETS_OFF 147968
#define SMEM_TASK_OFFSETS_STAGE_BYTES 2052
#define SMEM_TASK_OFFSETS_STRIDE 2052
#define SMEM_WORK_DESC_SLOTS_OFF 150032
#define SMEM_WORK_DESC_SLOTS_STAGE_BYTES 104
#define SMEM_WORK_DESC_SLOTS_STRIDE 104
#define SMEM_DECODE_ROW_MAX_OFF 150144
#define SMEM_DECODE_ROW_MAX_STAGE_BYTES 512
#define SMEM_DECODE_ROW_MAX_STRIDE 512
#define SMEM_DECODE_ROW_SUM_OFF 150656
#define SMEM_DECODE_ROW_SUM_STAGE_BYTES 512
#define SMEM_DECODE_ROW_SUM_STRIDE 512
#define SMEM_SPLIT_REDUCE_FLAG_OFF 145408
#define SMEM_SPLIT_REDUCE_FLAG_STAGE_BYTES 4
#define SMEM_SPLIT_REDUCE_FLAG_STRIDE 4
#define SMEM_TOTAL 216704
#define THREADS 512
#define MAX_REQUESTS_CONST 512

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
        :: "r"(mbar_addr), "r"(count));
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

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_minimax_sparse_decode_m16_paged_fp8_sm100(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap Q_prefill, __nv_bfloat16* __restrict__ Q_prefill_raw, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_prefill_pair, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_prefill_pair, const __grid_constant__ CUtensorMap KV, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ qo_indptr, int* __restrict__ kv_indptr, int* __restrict__ kv_len_arr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int* __restrict__ task_q_tile, int* __restrict__ task_split, int* __restrict__ task_kv_tile_begin, int* __restrict__ task_kv_tile_end, int* __restrict__ task_qo_begin, int* __restrict__ task_qo_end, int* __restrict__ task_page_begin, int* __restrict__ task_page_end, int* __restrict__ status, int num_requests, int num_q_heads, int num_kv_heads, int max_kv_tiles, int max_splits, int max_task_claims, float softmax_scale_log2, int attention_mode, int is_causal, int derive_q_offset, int record_tasks, int msa_max_pages, int msa_split_policy)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* smem_corr0 = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr0_addr = smem + 1024;
    float* smem_corr1 = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_corr1_addr = smem + 1088;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 151168);
    const int smem_kv_fp8_addr = smem + 151168;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_p0_addr = smem + 137216;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 141312);
    const int smem_p1_addr = smem + 141312;
    int* smem_page_indices = reinterpret_cast<int*>(smem_raw + 145408);
    const int smem_page_indices_addr = smem + 145408;
    float* prefill_scale = reinterpret_cast<float*>(smem_raw + 1024);
    const int prefill_scale_addr = smem + 1024;
    __nv_bfloat16* prefill_q0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int prefill_q0_addr = smem + 4096;
    float* prefill_partial_tile = reinterpret_cast<float*>(smem_raw + 4096);
    const int prefill_partial_tile_addr = smem + 4096;
    float* prefill_split_weights = reinterpret_cast<float*>(smem_raw + 4096);
    const int prefill_split_weights_addr = smem + 4096;
    __nv_bfloat16* prefill_q1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 36864);
    const int prefill_q1_addr = smem + 36864;
    __nv_bfloat16* prefill_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int prefill_kv_addr = smem + 69632;
    __nv_bfloat16* prefill_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int prefill_v_addr = smem + 69632;
    int* task_offsets = reinterpret_cast<int*>(smem_raw + 147968);
    const int task_offsets_addr = smem + 147968;
    int* work_desc_slots = reinterpret_cast<int*>(smem_raw + 150032);
    const int work_desc_slots_addr = smem + 150032;
    float* decode_row_max = reinterpret_cast<float*>(smem_raw + 150144);
    const int decode_row_max_addr = smem + 150144;
    float* decode_row_sum = reinterpret_cast<float*>(smem_raw + 150656);
    const int decode_row_sum_addr = smem + 150656;
    int* split_reduce_flag = reinterpret_cast<int*>(smem_raw + 145408);
    const int split_reduce_flag_addr = smem + 145408;

    // Mbarrier init (20 groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // work_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // work_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // work_empty_0: 1 barriers, init_count=15
            mbarrier_init(smem + 16, 15);
            // work_empty_1: 1 barriers, init_count=15
            mbarrier_init(smem + 24, 15);
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // q_tail_full: 2 barriers, init_count=32
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            // --- pipeline 'decode_kv' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // kv_src_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'prefill_kv_pipeline' ---
            // prefill_kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // prefill_kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 208, 256);
            mbarrier_init(smem + 216, 256);
            // p_full_tail: 2 barriers, init_count=256
            mbarrier_init(smem + 224, 256);
            mbarrier_init(smem + 232, 256);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            mbarrier_init(smem + 264, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // decode_done: 1 barriers, init_count=128
            mbarrier_init(smem + 288, 128);
            // decode_inputs_reusable: 1 barriers, init_count=256
            mbarrier_init(smem + 296, 256);
            // prefill_partial_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 312);
    if (warp == 0) {
        int _tmem_hold = smem + 312;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define work_full_0_addr (mbar_base + 0)
    #define work_full_1_addr (mbar_base + 8)
    #define work_empty_0_addr (mbar_base + 16)
    #define work_empty_1_addr (mbar_base + 24)
    #define q_full_addr (mbar_base + 32)
    #define q_tail_full_addr (mbar_base + 48)
    #define kv_full_addr (mbar_base + 64)
    #define kv_src_full_addr (mbar_base + 96)
    #define kv_empty_addr (mbar_base + 128)
    #define prefill_kv_full_addr (mbar_base + 160)
    #define prefill_kv_empty_addr (mbar_base + 176)
    #define s_full_addr (mbar_base + 192)
    #define p_full_addr (mbar_base + 208)
    #define p_full_tail_addr (mbar_base + 224)
    #define corr_sig_addr (mbar_base + 240)
    #define corr_done_addr (mbar_base + 256)
    #define o_full_addr (mbar_base + 272)
    #define decode_done_addr (mbar_base + 288)
    #define decode_inputs_reusable_addr (mbar_base + 296)
    #define prefill_partial_ready_addr (mbar_base + 304)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 16;
    const int tmem_tmem_o0 = taddr + 32;
    const int tmem_tmem_o1 = taddr + 48;
    const int tmem_tmem_stats0 = taddr + 64;
    const int tmem_tmem_stats1 = taddr + 80;
    const int tmem_prefill_scores_0 = taddr;
    const int tmem_prefill_scores_1 = taddr + 128;
    const int tmem_prefill_output_0 = taddr + 256;
    const int tmem_prefill_output_1 = taddr + 384;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            int group_size = num_q_heads / num_kv_heads;
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s = taddr + (unsigned int)(((is_wg1 != 0) ? 16 : 0)) + (unsigned int)(tmem_row_base_v << 16);
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 80 : 64)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            float* my_exch_ptr = ((is_wg1 != 0) ? smem_exch1 : smem_exch0);
            float* my_corr_ptr = ((is_wg1 != 0) ? smem_corr1 : smem_corr0);
            unsigned int* base = ((is_wg1 != 0) ? reinterpret_cast<unsigned int*>(smem_p1) : reinterpret_cast<unsigned int*>(smem_p0));
            int direct_decode = 1;
            unsigned int _phase_work_full_0_0 = 0;
            unsigned int _phase_work_full_1_0 = 0;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (int task_iter = 0; task_iter < max_task_claims + 1; task_iter++) {
                int work_slot = task_iter % 2;
                int* work_desc = work_desc_slots + (work_slot * 13);
                int ticket = -1;
                ticket = blockIdx.x * gridDim.y + blockIdx.y + task_iter * gridDim.x * gridDim.y;
                if (ticket >= num_requests * num_kv_heads * 1) {
                    ticket = -1;
                }
                if (ticket < 0) {
                    break;
                }
                int kind = 1;
                int direct_request = 0;
                direct_request = ticket / 1 / num_kv_heads;
                int kv_tile_begin = 0;
                int direct_batch = direct_request;
                direct_batch = direct_request / record_tasks;
                int direct_kv_len = kv_len_arr[direct_batch];
                int kv_tile_end = (direct_kv_len + 128 - 1) / 128;
                kv_tile_end = 16;
                int direct_kv_pairs = kv_tile_end / 2;
                int direct_split = ticket % 1;
                kv_tile_begin = 2 * (direct_kv_pairs * direct_split / 1);
                kv_tile_end = 2 * (direct_kv_pairs * (direct_split + 1) / 1);
                int num_n_blocks = kv_tile_end - kv_tile_begin;
                int num_pairs = num_n_blocks / 2;
                const int row_state_base = warp * 16;
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    decode_row_max[row_state_base + c] = -BLACKWELL_MSA_INF;
                    decode_row_sum[row_state_base + c] = 0.0f;
                }
                int max_decode_pairs = 8;
                #pragma unroll 1
                for (int pair = 0; pair < max_decode_pairs; pair++) {
                    if (num_pairs <= pair) {
                        break;
                    }
                    if (is_wg1 != 0) {
                        mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                        _phase_s_full_1 ^= 1;
                    } else {
                        mbarrier_wait(s_full_addr, _phase_s_full_0);
                        _phase_s_full_0 ^= 1;
                    }
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], my_tmem_s);
                    int valid_cols = smem_page_indices[pair * 2 + is_wg1];
                    int token_in_block = warp_in_wg * 32 + lane;
                    if (token_in_block >= valid_cols) {
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 16; c_1++) {
                            _tmem_load_0[c_1] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float partial_max[16];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 16; c_2++) {
                        partial_max[c_2] = _tmem_load_0[c_2];
                    }
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 16; c_3++) {
                        float _warp_reduce_0 = partial_max[c_3];
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                        partial_max[c_3] = _warp_reduce_0;
                    }
                    if (lane == 0) {
                        #pragma unroll
                        for (int c_4 = 0; c_4 < 16; c_4++) {
                            my_exch_ptr[warp_in_wg * 16 + c_4] = partial_max[c_4];
                        }
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                    }
                    float tile_max_lane = -BLACKWELL_MSA_INF;
                    if (lane < 16) {
                        float _max_0 = max_noftz(my_exch_ptr[lane], my_exch_ptr[16 + lane]);
                        float _max_1 = max_noftz(my_exch_ptr[32 + lane], my_exch_ptr[48 + lane]);
                        float _max_2 = max_noftz(_max_0, _max_1);
                        tile_max_lane = _max_2;
                    }
                    float acc_scale[16];
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 16; c_5++) {
                        float _shfl_0 = __shfl_sync(0xFFFFFFFF, tile_max_lane, c_5);
                        float tile_max_c = _shfl_0;
                        float old_max = decode_row_max[row_state_base + c_5];
                        float _max_3 = max_noftz(old_max, tile_max_c);
                        float new_max = _max_3;
                        decode_row_max[row_state_base + c_5] = new_max;
                        float delta = softmax_scale_log2 * (old_max - new_max);
                        float _exp2_0 = approx_exp2(delta);
                        acc_scale[c_5] = ((old_max > -BLACKWELL_MSA_INF) ? _exp2_0 : 1.0f);
                    }
                    tmem_st_x16_f32(my_tmem_stats, acc_scale);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(corr_sig_addr + 8);
                    } else {
                        mbarrier_arrive(corr_sig_addr);
                    }
                    float exp_vals[16];
                    #pragma unroll
                    for (int c_6 = 0; c_6 < 16; c_6++) {
                        float new_max_1 = decode_row_max[row_state_base + c_6];
                        float safe_max = ((new_max_1 == -BLACKWELL_MSA_INF) ? 0.0f : new_max_1);
                        float max_scaled = safe_max * softmax_scale_log2;
                        float _exp2_1 = approx_exp2(_tmem_load_0[c_6] * softmax_scale_log2 - max_scaled);
                        exp_vals[c_6] = _exp2_1;
                    }
                    float warp_sum[16];
                    #pragma unroll
                    for (int c_7 = 0; c_7 < 16; c_7++) {
                        warp_sum[c_7] = exp_vals[c_7];
                    }
                    #pragma unroll
                    for (int c_8 = 0; c_8 < 16; c_8++) {
                        float _warp_reduce_1 = warp_sum[c_8];
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                        warp_sum[c_8] = _warp_reduce_1;
                    }
                    #pragma unroll
                    for (int c_9 = 0; c_9 < 16; c_9++) {
                        float old_sum = decode_row_sum[row_state_base + c_9];
                        decode_row_sum[row_state_base + c_9] = old_sum * acc_scale[c_9] + warp_sum[c_9];
                    }
                    #pragma unroll
                    for (int h = 0; h < 16; h++) {
                        {
                            __nv_bfloat16 _bval_0 = __float2bfloat16_rn(exp_vals[h]);
                            uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                            const void* _ptr_0 = reinterpret_cast<const void*>((reinterpret_cast<uint8_t*>(base) + (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 ^ (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 >> 7 & 7) << 4)));
                            uint64_t _addr64_0;
                            asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_addr64_0) : "l"(_ptr_0));
                            uint32_t _addr_0;
                            asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_addr_0) : "l"(_addr64_0));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(p_full_addr + 8);
                    } else {
                        mbarrier_arrive(p_full_addr);
                    }
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                }
                if (lane < 16) {
                    my_exch_ptr[warp_in_wg * 16 + lane] = decode_row_sum[row_state_base + lane];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                }
                float total_sum[16];
                if (lane < 16) {
                    total_sum[lane] = my_exch_ptr[lane] + my_exch_ptr[16 + lane] + my_exch_ptr[32 + lane] + my_exch_ptr[48 + lane];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                }
                if (warp_in_wg == 0 && lane < 16) {
                    my_corr_ptr[lane] = total_sum[lane];
                    my_exch_ptr[lane] = decode_row_max[row_state_base + lane];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                }
                if (is_wg1 != 0) {
                    mbarrier_arrive(corr_sig_addr + 8);
                } else {
                    mbarrier_arrive(corr_sig_addr);
                }
                mbarrier_arrive(decode_inputs_reusable_addr);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            int d_idx = warp % 4 * 32 + lane;
            int direct_decode_1 = 1;
            unsigned int _phase_work_full_0_0_1 = 0;
            unsigned int _phase_work_full_1_0_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (int task_iter_1 = 0; task_iter_1 < max_task_claims + 1; task_iter_1++) {
                int work_slot_1 = task_iter_1 % 2;
                int* work_desc_1 = work_desc_slots + (work_slot_1 * 13);
                int ticket_1 = -1;
                ticket_1 = blockIdx.x * gridDim.y + blockIdx.y + task_iter_1 * gridDim.x * gridDim.y;
                if (ticket_1 >= num_requests * num_kv_heads * 1) {
                    ticket_1 = -1;
                }
                if (ticket_1 < 0) {
                    break;
                }
                int kind_1 = 1;
                int request = 0;
                int kv_head = 0;
                int split = 0;
                int splits = 1;
                int logical_output = ticket_1 / 1;
                request = logical_output / num_kv_heads;
                kv_head = logical_output - request * num_kv_heads;
                split = ticket_1 - logical_output * 1;
                splits = 1;
                int kv_tile_begin_1 = 0;
                int direct_batch_1 = request;
                direct_batch_1 = request / record_tasks;
                int direct_kv_len_1 = kv_len_arr[direct_batch_1];
                int kv_tile_end_1 = (direct_kv_len_1 + 128 - 1) / 128;
                int qo_begin = qo_indptr[request];
                kv_tile_end_1 = 16;
                qo_begin = request;
                int direct_kv_pairs_1 = kv_tile_end_1 / 2;
                kv_tile_begin_1 = 2 * (direct_kv_pairs_1 * split / splits);
                kv_tile_end_1 = 2 * (direct_kv_pairs_1 * (split + 1) / splits);
                int num_n_blocks_1 = kv_tile_end_1 - kv_tile_begin_1;
                int group_size_1 = num_q_heads / num_kv_heads;
                int num_pairs_1 = num_n_blocks_1 / 2;
                int max_decode_pairs_1 = 8;
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < max_decode_pairs_1; pair_1++) {
                    if (num_pairs_1 <= pair_1) {
                        break;
                    }
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 64 + (unsigned int)corr_row);
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 32 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_1 = 0; h_1 < 16; h_1++) {
                        _tmem_load_5[h_1] = _tmem_load_5[h_1] * _tmem_load_4[h_1];
                    }
                    tmem_st_x16_f32(taddr + 32 + (unsigned int)corr_row, _tmem_load_5);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr);
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 80 + (unsigned int)corr_row);
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 48 + (unsigned int)corr_row);
                    #pragma unroll
                    for (int h_2 = 0; h_2 < 16; h_2++) {
                        _tmem_load_7[h_2] = _tmem_load_7[h_2] * _tmem_load_6[h_2];
                    }
                    tmem_st_x16_f32(taddr + 48 + (unsigned int)corr_row, _tmem_load_7);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + 8);
                }
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                float scale0[16];
                float scale1[16];
                float final_sum[16];
                float final_max[16];
                #pragma unroll
                for (int c_10 = 0; c_10 < 16; c_10++) {
                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, smem_exch0[c_10], c_10);
                    float _shfl_2 = __shfl_sync(0xFFFFFFFF, smem_exch1[c_10], c_10);
                    float _shfl_3 = __shfl_sync(0xFFFFFFFF, smem_corr0[c_10], c_10);
                    float _shfl_4 = __shfl_sync(0xFFFFFFFF, smem_corr1[c_10], c_10);
                    float _max_6 = max_noftz(_shfl_1, _shfl_2);
                    float fm = _max_6;
                    final_max[c_10] = fm;
                    float d0 = ((_shfl_1 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_1 - fm));
                    float d1 = ((_shfl_2 == -BLACKWELL_MSA_INF) ? 0.0f : softmax_scale_log2 * (_shfl_2 - fm));
                    float _exp2_3 = approx_exp2(d0);
                    scale0[c_10] = _exp2_3;
                    float _exp2_4 = approx_exp2(d1);
                    scale1[c_10] = _exp2_4;
                    final_sum[c_10] = _shfl_3 * scale0[c_10] + _shfl_4 * scale1[c_10];
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float inv_sum[16];
                #pragma unroll
                for (int h_3 = 0; h_3 < 16; h_3++) {
                    float _rcp_0 = approx_rcp(final_sum[h_3]);
                    inv_sum[h_3] = ((final_sum[h_3] > 0.0f) ? _rcp_0 : 0.0f);
                }
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], taddr + 32 + (unsigned int)corr_row);
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], taddr + 48 + (unsigned int)corr_row);
                #pragma unroll
                for (int h_4 = 0; h_4 < 16; h_4++) {
                    if (group_size_1 > h_4) {
                        float merged = _tmem_load_8[h_4] * scale0[h_4] + _tmem_load_9[h_4] * scale1[h_4];
                        int q_row = qo_begin * num_q_heads + kv_head * group_size_1 + h_4;
                        int out_idx = q_row * 128 + d_idx;
                        if (splits == 1) {
                            if (d_idx == 0) {
                                float natural_lse = -BLACKWELL_MSA_INF;
                                if (final_sum[h_4] > 0.0f) {
                                    float _log2_0;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum[h_4]));
                                    natural_lse = final_max[h_4] * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                                }
                                *(reinterpret_cast<float*>(msa_lse + q_row) + (0)) = natural_lse;
                            }
                            *(reinterpret_cast<__nv_bfloat16*>(O + out_idx) + (0)) = __float2bfloat16_rn(merged * inv_sum[h_4]);
                        } else {
                            int logical_output_1 = request * num_kv_heads + kv_head;
                            int partial_slot = logical_output_1 * 1 + split;
                            int partial_o_idx = (partial_slot * 128 + h_4) * 128 + d_idx;
                            *(reinterpret_cast<float*>(partial_O + partial_o_idx) + (0)) = merged;
                            if (d_idx == 0) {
                                int stat_idx = partial_slot * 128 + h_4;
                                *(reinterpret_cast<float*>(partial_M + stat_idx) + (0)) = final_max[h_4];
                                *(reinterpret_cast<float*>(partial_D + stat_idx) + (0)) = final_sum[h_4];
                            }
                        }
                    }
                }
                mbarrier_arrive(decode_done_addr);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int direct_decode_2 = 1;
            unsigned int _phase_work_full_0_0_2 = 0;
            unsigned int _phase_work_full_1_0_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_decode_done_0 = 0;
            unsigned int _phase_q_tail_full_0 = 0;
            unsigned int _phase_p_full_tail_0 = 0;
            unsigned int _phase_q_tail_full_1 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_p_full_tail_1 = 0;
            #pragma unroll 1
            for (int task_iter_2 = 0; task_iter_2 < max_task_claims + 1; task_iter_2++) {
                int work_slot_2 = task_iter_2 % 2;
                int* work_desc_2 = work_desc_slots + (work_slot_2 * 13);
                int ticket_2 = -1;
                ticket_2 = blockIdx.x * gridDim.y + blockIdx.y + task_iter_2 * gridDim.x * gridDim.y;
                if (ticket_2 >= num_requests * num_kv_heads * 1) {
                    ticket_2 = -1;
                }
                if (ticket_2 < 0) {
                    break;
                }
                int kind_2 = 1;
                int direct_request_1 = 0;
                direct_request_1 = ticket_2 / 1 / num_kv_heads;
                int kv_tile_begin_2 = 0;
                int direct_batch_2 = direct_request_1;
                direct_batch_2 = direct_request_1 / record_tasks;
                int direct_kv_len_2 = kv_len_arr[direct_batch_2];
                int kv_tile_end_2 = (direct_kv_len_2 + 128 - 1) / 128;
                kv_tile_end_2 = 16;
                int direct_kv_pairs_2 = kv_tile_end_2 / 2;
                int direct_split_1 = ticket_2 % 1;
                kv_tile_begin_2 = 2 * (direct_kv_pairs_2 * direct_split_1 / 1);
                kv_tile_end_2 = 2 * (direct_kv_pairs_2 * (direct_split_1 + 1) / 1);
                int num_n_blocks_2 = kv_tile_end_2 - kv_tile_begin_2;
                int num_pairs_2 = num_n_blocks_2 / 2;
                int inst0_stage = 0;
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr, 0);
                int _mma_a_lo_8 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_8 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_tmem_s0), "r"(0));
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr);
                int max_decode_pairs_2 = 8;
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < max_decode_pairs_2 - 1; pair_2++) {
                    if (pair_2 >= num_pairs_2 - 1) {
                        break;
                    }
                    int s0 = inst0_stage;
                    int s1 = (inst0_stage + 1) % 4;
                    int s0_next = (inst0_stage + 2) % 4;
                    mbarrier_wait(kv_full_addr + (s1) * 8, 0);
                    int _mma_a_lo_9 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1) * 2048);
                    int _mma_b_lo_9 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"(tmem_tmem_s1), "r"(0));
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (s1) * 8);
                    mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 2048);
                    int _mma_b_lo_10 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (s0) * 8);
                    mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                    int _mma_a_lo_11 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_9), "r"(tmem_tmem_s0), "r"(0));
                    elect_commit(s_full_addr);
                    elect_commit(kv_empty_addr + (s0_next) * 8);
                    mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_12 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 2048);
                    int _mma_b_lo_12 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_12), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (s1) * 8);
                    inst0_stage = s0_next;
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                int s0_last = inst0_stage;
                int s1_last = (inst0_stage + 1) % 4;
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 0);
                int _mma_a_lo_13 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_last) * 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_8), "r"(tmem_tmem_s1), "r"(0));
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (s1_last) * 8);
                mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_14 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
                int _mma_b_lo_14 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"(tmem_tmem_o0), "r"(((first_pv0) ? 0 : 1)));
                elect_commit(kv_empty_addr + (s0_last) * 8);
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_15 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
                int _mma_b_lo_15 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134513808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_15), "r"(tmem_tmem_o1), "r"(((first_pv1) ? 0 : 1)));
                elect_commit(kv_empty_addr + (s1_last) * 8);
                elect_commit(o_full_addr);
                mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                _phase_decode_done_0 ^= 1;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 13) {
        // Task scheduler_main: no body
    }
    // ---- Role: producer ----
    if (warp == 14) {
        { // producer_main
            int direct_decode_3 = 1;
            unsigned int _phase_work_full_0_0_3 = 0;
            unsigned int _phase_work_full_1_0_3 = 0;
            unsigned int _phase_prefill_partial_ready_0 = 0;
            unsigned int _phase_decode_inputs_reusable_0 = 0;
            #pragma unroll 1
            for (int task_iter_3 = 0; task_iter_3 < max_task_claims + 1; task_iter_3++) {
                int work_slot_3 = task_iter_3 % 2;
                int* work_desc_3 = work_desc_slots + (work_slot_3 * 13);
                int ticket_3 = -1;
                ticket_3 = blockIdx.x * gridDim.y + blockIdx.y + task_iter_3 * gridDim.x * gridDim.y;
                if (ticket_3 >= num_requests * num_kv_heads * 1) {
                    ticket_3 = -1;
                }
                if (ticket_3 < 0) {
                    break;
                }
                int kind_3 = 1;
                int direct_request_2 = 0;
                int kv_head_1 = 0;
                direct_request_2 = ticket_3 / 1 / num_kv_heads;
                kv_head_1 = ticket_3 / 1 % num_kv_heads;
                int q_tile = 0;
                int kv_tile_begin_3 = 0;
                int direct_batch_3 = direct_request_2;
                direct_batch_3 = direct_request_2 / record_tasks;
                int direct_kv_len_3 = kv_len_arr[direct_batch_3];
                int kv_tile_end_3 = (direct_kv_len_3 + 128 - 1) / 128;
                int qo_begin_1 = qo_indptr[direct_request_2];
                int page_begin = kv_indptr[direct_request_2];
                kv_tile_end_3 = 16;
                qo_begin_1 = direct_request_2;
                page_begin = kv_indptr[direct_batch_3];
                int direct_kv_pairs_3 = kv_tile_end_3 / 2;
                int direct_split_2 = ticket_3 % 1;
                kv_tile_begin_3 = 2 * (direct_kv_pairs_3 * direct_split_2 / 1);
                kv_tile_end_3 = 2 * (direct_kv_pairs_3 * (direct_split_2 + 1) / 1);
                int num_n_blocks_3 = kv_tile_end_3 - kv_tile_begin_3;
                int group_size_2 = num_q_heads / num_kv_heads;
                if (elect_sync()) {
                    int q_row_1 = qo_begin_1 * num_q_heads + kv_head_1 * group_size_2 + q_tile * 16;
                    mbarrier_arrive_expect_tx(q_full_addr, 4096);
                    tma_3d_gmem2smem(smem_qt_addr, (&Q), 0, q_row_1, 0, q_full_addr);
                }
                int native_num_n_blocks = num_n_blocks_3;
                if (elect_sync()) {
                    int native_kv_stage = 0;
                    int native_kv_phase = 1;
                    int native_prefill = ((native_num_n_blocks < 4) ? native_num_n_blocks : 4);
                    #pragma unroll
                    for (int native_ni = 0; native_ni < 4; native_ni++) {
                        if (native_prefill <= native_ni) {
                            break;
                        }
                        int native_n_block = kv_tile_end_3 - 1 - native_ni;
                        int native_pg0 = 0;
                        int native_pg1 = 0;
                        int msa_token_base = 0;
                        int msa_page_head = 0;
                        int msa_valid_cols = 128;
                        int batch = direct_request_2 / record_tasks;
                        int query_in_batch = direct_request_2 - batch * record_tasks;
                        int selected_block = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_n_block];
                        int kv_len = task_kv_head[batch];
                        int valid_cols_1 = 0;
                        if (selected_block >= 0) {
                            int block_start = selected_block * 128;
                            valid_cols_1 = kv_len - block_start;
                            if (valid_cols_1 > 128) {
                                valid_cols_1 = 128;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                            if (is_causal != 0) {
                                int query_position = kv_len - record_tasks + query_in_batch;
                                if (derive_q_offset == 0) {
                                    query_position = task_request[batch] + query_in_batch;
                                }
                                int causal_cols = query_position - block_start + 1;
                                if (valid_cols_1 > causal_cols) {
                                    valid_cols_1 = causal_cols;
                                }
                                if (valid_cols_1 < 0) {
                                    valid_cols_1 = 0;
                                }
                            }
                        }
                        int token_base = 0;
                        int page_head = 0;
                        int physical_page = 0;
                        if (selected_block >= 0) {
                            physical_page = kv_indices[batch * msa_max_pages + selected_block];
                            if (physical_page < 0) {
                                valid_cols_1 = 0;
                                physical_page = 0;
                            }
                        }
                        page_head = physical_page * num_kv_heads + kv_head_1;
                        msa_token_base = token_base;
                        msa_page_head = page_head;
                        msa_valid_cols = valid_cols_1;
                        smem_page_indices[native_ni] = msa_valid_cols;
                        smem_page_indices[16 + native_ni] = msa_token_base;
                        smem_page_indices[32 + native_ni] = msa_page_head;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_empty_addr + (native_kv_stage) * 8, native_kv_phase);
                        mbarrier_arrive_expect_tx(kv_src_full_addr + (native_kv_stage) * 8, 16384);
                        int native_dst = smem_kv_addr + (unsigned int)(native_kv_stage * 32768);
                        int token0 = msa_token_base;
                        int token1 = msa_token_base + 64;
                        token0 = 0;
                        token1 = 64;
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_kv_stage * 16384), (&K), 0, token0, msa_page_head, kv_src_full_addr + (native_kv_stage) * 8);
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_kv_stage * 16384) + 8192, (&K), 0, token1, msa_page_head, kv_src_full_addr + (native_kv_stage) * 8);
                        native_kv_stage += 1;
                        if (native_kv_stage == 4) { native_kv_stage = 0; native_kv_phase ^= 1; }
                    }
                    #pragma unroll 1
                    for (int native_ni_1 = 0; native_ni_1 < 16; native_ni_1++) {
                        if (native_num_n_blocks <= native_ni_1) {
                            break;
                        }
                        int native_stage = native_ni_1 % 4;
                        int native_n_block_1 = kv_tile_end_3 - 1 - native_ni_1;
                        int native_pg0_1 = 0;
                        int native_pg1_1 = 0;
                        int msa_token_base_1 = 0;
                        int msa_page_head_1 = 0;
                        int msa_valid_cols_1 = 128;
                        msa_valid_cols_1 = smem_page_indices[native_ni_1];
                        msa_token_base_1 = smem_page_indices[16 + native_ni_1];
                        msa_page_head_1 = smem_page_indices[32 + native_ni_1];
                        mbarrier_wait(kv_empty_addr + (native_stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_src_full_addr + (native_stage) * 8, 16384);
                        int native_dst_1 = smem_kv_addr + (unsigned int)(native_stage * 32768);
                        int token0_1 = msa_token_base_1;
                        int token1_1 = msa_token_base_1 + 64;
                        token0_1 = 0;
                        token1_1 = 64;
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384), (&V), 0, token0_1, msa_page_head_1, kv_src_full_addr + (native_stage) * 8);
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384) + 8192, (&V), 0, token1_1, msa_page_head_1, kv_src_full_addr + (native_stage) * 8);
                        int native_next_ni = native_ni_1 + 4;
                        if (native_next_ni < native_num_n_blocks) {
                            int native_next_n = kv_tile_end_3 - 1 - native_next_ni;
                            int native_npg0 = 0;
                            int native_npg1 = 0;
                            int msa_next_token_base = 0;
                            int msa_next_page_head = 0;
                            int msa_next_valid_cols = 128;
                            int batch_1 = direct_request_2 / record_tasks;
                            int query_in_batch_1 = direct_request_2 - batch_1 * record_tasks;
                            int selected_block_1 = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_next_n];
                            int kv_len_1 = task_kv_head[batch_1];
                            int valid_cols_2 = 0;
                            if (selected_block_1 >= 0) {
                                int block_start_1 = selected_block_1 * 128;
                                valid_cols_2 = kv_len_1 - block_start_1;
                                if (valid_cols_2 > 128) {
                                    valid_cols_2 = 128;
                                }
                                if (valid_cols_2 < 0) {
                                    valid_cols_2 = 0;
                                }
                                if (is_causal != 0) {
                                    int query_position_1 = kv_len_1 - record_tasks + query_in_batch_1;
                                    if (derive_q_offset == 0) {
                                        query_position_1 = task_request[batch_1] + query_in_batch_1;
                                    }
                                    int causal_cols_1 = query_position_1 - block_start_1 + 1;
                                    if (valid_cols_2 > causal_cols_1) {
                                        valid_cols_2 = causal_cols_1;
                                    }
                                    if (valid_cols_2 < 0) {
                                        valid_cols_2 = 0;
                                    }
                                }
                            }
                            int token_base_1 = 0;
                            int page_head_1 = 0;
                            int physical_page_1 = 0;
                            if (selected_block_1 >= 0) {
                                physical_page_1 = kv_indices[batch_1 * msa_max_pages + selected_block_1];
                                if (physical_page_1 < 0) {
                                    valid_cols_2 = 0;
                                    physical_page_1 = 0;
                                }
                            }
                            page_head_1 = physical_page_1 * num_kv_heads + kv_head_1;
                            msa_next_token_base = token_base_1;
                            msa_next_page_head = page_head_1;
                            msa_next_valid_cols = valid_cols_2;
                            smem_page_indices[native_next_ni] = msa_next_valid_cols;
                            smem_page_indices[16 + native_next_ni] = msa_next_token_base;
                            smem_page_indices[32 + native_next_ni] = msa_next_page_head;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_wait(kv_empty_addr + (native_stage) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_src_full_addr + (native_stage) * 8, 16384);
                            int native_kdst = smem_kv_addr + (unsigned int)(native_stage * 32768);
                            int token0_2 = msa_next_token_base;
                            int token1_2 = msa_next_token_base + 64;
                            token0_2 = 0;
                            token1_2 = 64;
                            tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384), (&K), 0, token0_2, msa_next_page_head, kv_src_full_addr + (native_stage) * 8);
                            tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384) + 8192, (&K), 0, token1_2, msa_next_page_head, kv_src_full_addr + (native_stage) * 8);
                        }
                    }
                }
                num_n_blocks_3 = 0;
                int kv_stage = 0;
                int kv_phase = 1;
                int prefill = ((num_n_blocks_3 < 4) ? num_n_blocks_3 : 4);
                #pragma unroll 1
                for (int ni = 0; ni < num_n_blocks_3; ni++) {
                    if (prefill <= ni) {
                        break;
                    }
                    int n_block = kv_tile_end_3 - 1 - ni;
                    #pragma unroll
                    for (int pp = 0; pp < 4; pp++) {
                        int pt_off = pp * 32 + lane;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                            :: "r"(smem_page_indices_addr + (unsigned int)((kv_stage * 128 + pt_off) * 4)), "l"(kv_indices + (page_begin + n_block * 128 + pt_off)));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync 10, 32;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, kv_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                    }
                    int dst = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                    #pragma unroll
                    for (int g = 0; g < 32; g++) {
                        int off_g = g * 4;
                        int k_stage_index = kv_stage * 128 + off_g;
                        int r0 = smem_page_indices[k_stage_index] * (2 * num_kv_heads) + kv_head_1;
                        int r1 = smem_page_indices[k_stage_index + 1] * (2 * num_kv_heads) + kv_head_1;
                        int r2 = smem_page_indices[k_stage_index + 2] * (2 * num_kv_heads) + kv_head_1;
                        int r3 = smem_page_indices[k_stage_index + 3] * (2 * num_kv_heads) + kv_head_1;
                        if (elect_sync()) {
                            tma_gather4_gmem2smem(dst + g * 512, (&KV), 0, r0, r1, r2, r3, kv_full_addr + (kv_stage) * 8);
                            tma_gather4_gmem2smem(dst + 16384 + g * 512, (&KV), 64, r0, r1, r2, r3, kv_full_addr + (kv_stage) * 8);
                        }
                    }
                    kv_stage += 1;
                    if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                }
                #pragma unroll 1
                for (int ni_1 = 0; ni_1 < num_n_blocks_3; ni_1++) {
                    if (num_n_blocks_3 <= ni_1) {
                        break;
                    }
                    int stage = ni_1 % 4;
                    if (elect_sync()) {
                        mbarrier_wait(kv_empty_addr + (stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
                    }
                    int dst_1 = smem_kv_addr + (unsigned int)(stage * 32768);
                    #pragma unroll
                    for (int g_1 = 0; g_1 < 32; g_1++) {
                        int off_gv = g_1 * 4;
                        int v_stage_index = stage * 128 + off_gv;
                        int r0_1 = smem_page_indices[v_stage_index] * (2 * num_kv_heads) + num_kv_heads + kv_head_1;
                        int r1_1 = smem_page_indices[v_stage_index + 1] * (2 * num_kv_heads) + num_kv_heads + kv_head_1;
                        int r2_1 = smem_page_indices[v_stage_index + 2] * (2 * num_kv_heads) + num_kv_heads + kv_head_1;
                        int r3_1 = smem_page_indices[v_stage_index + 3] * (2 * num_kv_heads) + num_kv_heads + kv_head_1;
                        if (elect_sync()) {
                            tma_gather4_gmem2smem(dst_1 + g_1 * 512, (&KV), 0, r0_1, r1_1, r2_1, r3_1, kv_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(dst_1 + 16384 + g_1 * 512, (&KV), 64, r0_1, r1_1, r2_1, r3_1, kv_full_addr + (stage) * 8);
                        }
                    }
                    int next_ni = ni_1 + 4;
                    if (next_ni < num_n_blocks_3) {
                        int next_n = kv_tile_end_3 - 1 - next_ni;
                        #pragma unroll
                        for (int pp_1 = 0; pp_1 < 4; pp_1++) {
                            int pt_off_1 = pp_1 * 32 + lane;
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(smem_page_indices_addr + (unsigned int)((stage * 128 + pt_off_1) * 4)), "l"(kv_indices + (page_begin + next_n * 128 + pt_off_1)));
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("barrier.sync 10, 32;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (stage) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
                        }
                        int kdst = smem_kv_addr + (unsigned int)(stage * 32768);
                        #pragma unroll
                        for (int g_2 = 0; g_2 < 32; g_2++) {
                            int off_gk = g_2 * 4;
                            int next_k_stage_index = stage * 128 + off_gk;
                            int r0_2 = smem_page_indices[next_k_stage_index] * (2 * num_kv_heads) + kv_head_1;
                            int r1_2 = smem_page_indices[next_k_stage_index + 1] * (2 * num_kv_heads) + kv_head_1;
                            int r2_2 = smem_page_indices[next_k_stage_index + 2] * (2 * num_kv_heads) + kv_head_1;
                            int r3_2 = smem_page_indices[next_k_stage_index + 3] * (2 * num_kv_heads) + kv_head_1;
                            if (elect_sync()) {
                                tma_gather4_gmem2smem(kdst + g_2 * 512, (&KV), 0, r0_2, r1_2, r2_2, r3_2, kv_full_addr + (stage) * 8);
                                tma_gather4_gmem2smem(kdst + 16384 + g_2 * 512, (&KV), 64, r0_2, r1_2, r2_2, r3_2, kv_full_addr + (stage) * 8);
                            }
                        }
                    }
                }
                mbarrier_wait(decode_inputs_reusable_addr, _phase_decode_inputs_reusable_0);
                _phase_decode_inputs_reusable_0 ^= 1;
            }
        }
    }
    // ---- Role: producer_aux ----
    if (warp == 15) {
        { // producer_aux_main
            int direct_decode_4 = 1;
            unsigned int _phase_work_full_0_0_4 = 0;
            unsigned int _phase_work_full_1_0_4 = 0;
            #pragma unroll 1
            for (int task_iter_4 = 0; task_iter_4 < max_task_claims + 1; task_iter_4++) {
                int work_slot_4 = task_iter_4 % 2;
                int* work_desc_4 = work_desc_slots + (work_slot_4 * 13);
                int ticket_4 = -1;
                ticket_4 = blockIdx.x * gridDim.y + blockIdx.y + task_iter_4 * gridDim.x * gridDim.y;
                if (ticket_4 >= num_requests * num_kv_heads * 1) {
                    ticket_4 = -1;
                }
                if (ticket_4 < 0) {
                    break;
                }
                int num_n_blocks_4 = 16;
                int direct_kv_pairs_4 = 8;
                int direct_split_3 = ticket_4 % 1;
                int direct_kv_begin = 2 * (direct_kv_pairs_4 * direct_split_3 / 1);
                int direct_kv_end = 2 * (direct_kv_pairs_4 * (direct_split_3 + 1) / 1);
                num_n_blocks_4 = direct_kv_end - direct_kv_begin;
                int prefill_1 = ((num_n_blocks_4 < 4) ? num_n_blocks_4 : 4);
                #pragma unroll
                for (int ni_2 = 0; ni_2 < 4; ni_2++) {
                    if (prefill_1 <= ni_2) {
                        break;
                    }
                    mbarrier_wait(kv_src_full_addr + (ni_2) * 8, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(ni_2 * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(ni_2 * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (15) * 32;
                        #pragma unroll 4
                        for (int _off = _tid; _off < 2048; _off += 32) {
                            uint64_t _src64 = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                                #else
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                #endif
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (ni_2) * 8);
                    }
                }
                #pragma unroll 1
                for (int ni_3 = 0; ni_3 < 16; ni_3++) {
                    if (num_n_blocks_4 <= ni_3) {
                        break;
                    }
                    int stage_1 = ni_3 % 4;
                    mbarrier_wait(kv_src_full_addr + (stage_1) * 8, 1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(stage_1 * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(stage_1 * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (15) * 32;
                        #pragma unroll 4
                        for (int _off = _tid; _off < 2048; _off += 32) {
                            uint64_t _src64 = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                                #else
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                #endif
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (stage_1) * 8);
                    }
                    int next_ni_1 = ni_3 + 4;
                    if (next_ni_1 < num_n_blocks_4) {
                        mbarrier_wait(kv_src_full_addr + (stage_1) * 8, 0);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        {
                            const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(stage_1 * 16384) - smem);
                            char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(stage_1 * 32768) - smem);
                            const int _tid = (int)threadIdx.x - (15) * 32;
                            #pragma unroll 4
                            for (int _off = _tid; _off < 2048; _off += 32) {
                                uint64_t _src64 = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                                    #else
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                    #endif
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        }
                        __syncwarp();
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(kv_full_addr + (stage_1) * 8);
                        }
                    }
                }
            }
            unsigned int _phase_prefill_partial_ready_0_1 = 0;
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

