typedef signed char        int8_t;
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
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_OFFSET 256
#define NUM_Q_PIPE_STAGES 2
#define NUM_S_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define NUM_O_PIPE_STAGES 2
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 32768
#define SMEM_Q_SMEM_STRIDE 32768
#define SMEM_Q_STORE_SMEM_OFF 1024
#define SMEM_Q_STORE_SMEM_STAGE_BYTES 32768
#define SMEM_Q_STORE_SMEM_STRIDE 32768
#define SMEM_K_SMEM_OFF 66560
#define SMEM_K_SMEM_STAGE_BYTES 32768
#define SMEM_K_SMEM_STRIDE 32768
#define SMEM_V_SMEM_OFF 99328
#define SMEM_V_SMEM_STAGE_BYTES 16384
#define SMEM_V_SMEM_STRIDE 16384
#define SMEM_FP8_SMEM_OFF 115712
#define SMEM_FP8_SMEM_STAGE_BYTES 16384
#define SMEM_FP8_SMEM_STRIDE 16384
#define SMEM_TOTAL 132096
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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
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


__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
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


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
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


__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
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
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_minimax_sparse_reverse_prefill_fp8_gqa16_qagg_pdl_sm100(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, int* __restrict__ scheduler_metadata, int* __restrict__ k2q_row_ptr, int* __restrict__ k2q_qsplit_indices, uint8_t* __restrict__ partial_o, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, unsigned int* __restrict__ completion_counts, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int total_q, int num_q_heads, int num_kv_heads, int total_rows, int nnz_per_head, int work_capacity, int num_work_items, int topk, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_temperature_lse)
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
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* q_store_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_store_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int k_smem_addr = smem + 66560;
    uint8_t* v_smem = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int v_smem_addr = smem + 99328;
    uint8_t* fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int fp8_smem_addr = smem + 115712;

    // Mbarrier init (12 groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=4
            mbarrier_init(smem + 0, 4);
            mbarrier_init(smem + 8, 4);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // kv_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // fp8_k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // --- pipeline 's_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            mbarrier_init(smem + 88, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'o_pipe' ---
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 40)
    #define kv_empty_addr (mbar_base + 48)
    #define fp8_k_full_addr (mbar_base + 56)
    #define s_full_addr (mbar_base + 64)
    #define s_empty_addr (mbar_base + 80)
    #define p_full_addr (mbar_base + 96)
    #define p_empty_addr (mbar_base + 112)
    #define o_full_addr (mbar_base + 128)
    #define o_empty_addr (mbar_base + 144)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output = taddr + 256;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax_even ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // softmax_even_main
            int work_idx = bid;
            int metadata_base = work_idx * 6;
            int head_kv = scheduler_metadata[metadata_base];
            int row_linear = scheduler_metadata[metadata_base + 1];
            int q_begin = scheduler_metadata[metadata_base + 2];
            int q_count = scheduler_metadata[metadata_base + 3];
            int batch = scheduler_metadata[metadata_base + 4];
            int kv_block = scheduler_metadata[metadata_base + 5];
            int row_ptr_base = head_kv * (total_rows + 1) + row_linear;
            int row_start = k2q_row_ptr[row_ptr_base] + q_begin;
            int q_batch_offset = cu_seqlens_q[batch];
            int k_batch_offset = cu_seqlens_k[batch];
            int kv_len = 0;
            if (max_pages == 0) {
                kv_len = cu_seqlens_k[batch + 1] - k_batch_offset;
            } else {
                kv_len = kv_lens[batch];
            }
            int query_offset = 0;
            if (derive_q_offset != 0) {
                query_offset = kv_len - (cu_seqlens_q[batch + 1] - q_batch_offset);
            } else {
                query_offset = q_offsets[batch];
            }
            int group_count = (q_count + 8 - 1) / 8;
            int metadata_base_0 = work_idx * 6;
            int head_kv_1 = scheduler_metadata[metadata_base_0];
            int row_linear_2 = scheduler_metadata[metadata_base_0 + 1];
            int q_begin_3 = scheduler_metadata[metadata_base_0 + 2];
            int q_count_4 = scheduler_metadata[metadata_base_0 + 3];
            int batch_5 = scheduler_metadata[metadata_base_0 + 4];
            int kv_block_6 = scheduler_metadata[metadata_base_0 + 5];
            int row_ptr_base_7 = head_kv_1 * (total_rows + 1) + row_linear_2;
            int row_start_8 = k2q_row_ptr[row_ptr_base_7] + q_begin_3;
            int q_batch_offset_9 = cu_seqlens_q[batch_5];
            int k_batch_offset_10 = cu_seqlens_k[batch_5];
            int kv_len_11 = 0;
            if (max_pages == 0) {
                kv_len_11 = cu_seqlens_k[batch_5 + 1] - k_batch_offset_10;
            } else {
                kv_len_11 = kv_lens[batch_5];
            }
            int query_offset_12 = 0;
            if (derive_q_offset != 0) {
                query_offset_12 = kv_len_11 - (cu_seqlens_q[batch_5 + 1] - q_batch_offset_9);
            } else {
                query_offset_12 = q_offsets[batch_5];
            }
            int first_local_group = 0;
            int stage_group_count = (group_count + 1 - first_local_group) / 2;
            int stage_warp = warp;
            int my_row = stage_warp * 32 + lane;
            int tmem_row_base = stage_warp * 32 << 16;
            #pragma unroll 1
            for (int stage_iteration = 0; stage_iteration < stage_group_count; stage_iteration++) {
                int group = stage_iteration * 2 + first_local_group;
                int absolute_group = group;
                int softmax_phase = absolute_group / 2 & 1;
                mbarrier_wait(s_full_addr, softmax_phase);
                int whole_group_valid = 1;
                int token_in_group = 0;
                int edge_in_work = 0;
                int packed_q = 0;
                int q_idx = 0;
                int valid_cols = 0;
                int body_valid = 0;
                int tail_valid = 0;
                float row_max = -BLACKWELL_MSA_INF;
                float score_bias = -BLACKWELL_MSA_INF;
                int score_base = taddr + (unsigned int)tmem_row_base;
                if (whole_group_valid != 0) {
                    token_in_group = my_row / 16;
                    edge_in_work = group * 8 + token_in_group;
                    int row_valid = ((edge_in_work < q_count_4) ? 1 : 0);
                    int owner_lane = lane / 16 * 16;
                    int owned_packed = -1;
                    if (lane == owner_lane && edge_in_work < q_count_4) {
                        owned_packed = k2q_qsplit_indices[head_kv_1 * nnz_per_head + row_start_8 + edge_in_work];
                    }
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, owned_packed, owner_lane);
                    packed_q = _shfl_0;
                    q_idx = packed_q & 16777215;
                    if (row_valid != 0) {
                        valid_cols = kv_len_11 - kv_block_6 * 128;
                        if (valid_cols > 128) {
                            valid_cols = 128;
                        }
                        if (causal != 0) {
                            int query_position = query_offset_12 + q_idx;
                            int causal_cols = query_position - kv_block_6 * 128 + 1;
                            if (valid_cols > causal_cols) {
                                valid_cols = causal_cols;
                            }
                        }
                        if (valid_cols < 0) {
                            valid_cols = 0;
                        }
                    }
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(score_base)
                        : "memory");
                    body_valid = valid_cols;
                    if (body_valid < 0) {
                        body_valid = 0;
                    }
                    if (body_valid > 0 && body_valid < 64) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = body_valid;
                            if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                            }
                        }
                        #pragma unroll
                        for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                            if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_0[0 + _i_1] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_2 = body_valid - 32;
                            if (_lim_2 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_2));
                            }
                        }
                        #pragma unroll
                        for (int _i_3 = 0; _i_3 < 32; _i_3++) {
                            if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_0[32 + _i_3] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_4 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_4);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_4);
                    float body_max = _tmem_load_0_max;
                    if (body_valid <= 0) {
                        body_max = -BLACKWELL_MSA_INF;
                    }
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(score_base + 64)
                        : "memory");
                    tail_valid = valid_cols - 64;
                    if (tail_valid < 0) {
                        tail_valid = 0;
                    }
                    if (valid_cols > 0 && tail_valid < 64) {
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_5 = tail_valid;
                            if (_lim_5 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_5 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_5));
                            }
                        }
                        #pragma unroll
                        for (int _i_6 = 0; _i_6 < 32; _i_6++) {
                            if (!(_slice_lo_mask_2 & (1u << _i_6))) _tmem_load_1[0 + _i_6] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_7 = tail_valid - 32;
                            if (_lim_7 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_7 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_7));
                            }
                        }
                        #pragma unroll
                        for (int _i_8 = 0; _i_8 < 32; _i_8++) {
                            if (!(_slice_lo_mask_3 & (1u << _i_8))) _tmem_load_1[32 + _i_8] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_9 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&_tmem_load_1[0], _reg_reduce_max2_9);
                    row_max_x32_accum(&_tmem_load_1[32], _reg_reduce_max2_9);
                    float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_9);
                    float tail_max = _tmem_load_1_max;
                    if (tail_valid <= 0) {
                        tail_max = -BLACKWELL_MSA_INF;
                    }
                    float _max_0 = max_noftz(body_max, tail_max);
                    row_max = _max_0;
                    float safe_max = ((row_max == -BLACKWELL_MSA_INF) ? 0.0f : row_max);
                    score_bias = ((valid_cols > 0) ? (-safe_max) * softmax_scale_log2 : -BLACKWELL_MSA_INF);
                }
                mbarrier_wait(p_empty_addr, softmax_phase ^ 1);
                float row_sum = 0.0f;
                if (whole_group_valid != 0) {
                    int p_base = taddr + 96 + (unsigned int)tmem_row_base;
                    float _tmem_load_2[32];
                    tmem_ld_x32(&_tmem_load_2[0], score_base);
                    uint32_t _slice_lo_mask_4;
                    {
                        int _lim_10 = valid_cols;
                        if (_lim_10 <= 0) { _slice_lo_mask_4 = 0u; }
                        else if (_lim_10 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_10));
                        }
                    }
                    #pragma unroll
                    for (int _i_11 = 0; _i_11 < 32; _i_11++) {
                        if (!(_slice_lo_mask_4 & (1u << _i_11))) _tmem_load_2[0 + _i_11] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_12 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_13 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_lf], _fma_b2_12, _fma_c2_13);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_2[_le] = approx_exp2(_tmem_load_2[_le]);
                    }
                    float2 _reg_reduce_sum2_14 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_2[0], &_reg_reduce_sum2_14);
                    float _tmem_load_2_sum = _reg_reduce_sum2_14.x + _reg_reduce_sum2_14.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_2[0 + _j * 4]), "f"(_tmem_load_2[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_2[0 + _j * 4 + 2]), "f"(_tmem_load_2[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base, _pv_packed);
                    }
                    row_sum += _tmem_load_2_sum;
                    float _tmem_load_3[32];
                    tmem_ld_x32(&_tmem_load_3[0], score_base + 32);
                    uint32_t _slice_lo_mask_5;
                    {
                        int _lim_15 = valid_cols - 32;
                        if (_lim_15 <= 0) { _slice_lo_mask_5 = 0u; }
                        else if (_lim_15 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_15));
                        }
                    }
                    #pragma unroll
                    for (int _i_16 = 0; _i_16 < 32; _i_16++) {
                        if (!(_slice_lo_mask_5 & (1u << _i_16))) _tmem_load_3[0 + _i_16] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_18 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_lf], _fma_b2_17, _fma_c2_18);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_3[_le] = approx_exp2(_tmem_load_3[_le]);
                    }
                    float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_3[0], &_reg_reduce_sum2_19);
                    float _tmem_load_3_sum = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_3[0 + _j * 4]), "f"(_tmem_load_3[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_3[0 + _j * 4 + 2]), "f"(_tmem_load_3[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base + 8, _pv_packed);
                    }
                    row_sum += _tmem_load_3_sum;
                    float _tmem_load_4[32];
                    tmem_ld_x32(&_tmem_load_4[0], score_base + 64);
                    uint32_t _slice_lo_mask_6;
                    {
                        int _lim_20 = valid_cols - 64;
                        if (_lim_20 <= 0) { _slice_lo_mask_6 = 0u; }
                        else if (_lim_20 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_20));
                        }
                    }
                    #pragma unroll
                    for (int _i_21 = 0; _i_21 < 32; _i_21++) {
                        if (!(_slice_lo_mask_6 & (1u << _i_21))) _tmem_load_4[0 + _i_21] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_22 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_23 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_lf], _fma_b2_22, _fma_c2_23);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_4[_le] = approx_exp2(_tmem_load_4[_le]);
                    }
                    float2 _reg_reduce_sum2_24 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_4[0], &_reg_reduce_sum2_24);
                    float _tmem_load_4_sum = _reg_reduce_sum2_24.x + _reg_reduce_sum2_24.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_4[0 + _j * 4]), "f"(_tmem_load_4[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_4[0 + _j * 4 + 2]), "f"(_tmem_load_4[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base + 16, _pv_packed);
                    }
                    row_sum += _tmem_load_4_sum;
                    float _tmem_load_5[32];
                    tmem_ld_x32(&_tmem_load_5[0], score_base + 96);
                    uint32_t _slice_lo_mask_7;
                    {
                        int _lim_25 = valid_cols - 96;
                        if (_lim_25 <= 0) { _slice_lo_mask_7 = 0u; }
                        else if (_lim_25 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_25));
                        }
                    }
                    #pragma unroll
                    for (int _i_26 = 0; _i_26 < 32; _i_26++) {
                        if (!(_slice_lo_mask_7 & (1u << _i_26))) _tmem_load_5[0 + _i_26] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_27 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_28 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_lf], _fma_b2_27, _fma_c2_28);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_5[_le] = approx_exp2(_tmem_load_5[_le]);
                    }
                    float2 _reg_reduce_sum2_29 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_5[0], &_reg_reduce_sum2_29);
                    float _tmem_load_5_sum = _reg_reduce_sum2_29.x + _reg_reduce_sum2_29.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_5[0 + _j * 4]), "f"(_tmem_load_5[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_5[0 + _j * 4 + 2]), "f"(_tmem_load_5[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base + 24, _pv_packed);
                    }
                    row_sum += _tmem_load_5_sum;
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr);
                mbarrier_wait(o_full_addr, softmax_phase);
                if (whole_group_valid != 0) {
                    int q_head_local = my_row - token_in_group * 16;
                    int output_valid = 0;
                    long long partial_row = 0;
                    float inv_sum = 0.0f;
                    if (edge_in_work < q_count_4) {
                        int split_slot = packed_q >> 24 & 255;
                        if (split_slot >= 0 && split_slot < topk) {
                            output_valid = 1;
                            int q_abs = q_batch_offset_9 + q_idx;
                            int q_head = head_kv_1 * 16 + q_head_local;
                            partial_row = (long long)split_slot * (long long)total_q * (long long)num_q_heads + (long long)q_abs * (long long)num_q_heads + (long long)q_head;
                            float _rcp_0 = approx_rcp(row_sum);
                            inv_sum = ((row_sum > 0.0f && row_sum == row_sum) ? _rcp_0 : 0.0f);
                        }
                    }
                    long long partial_base = partial_row * 128;
                    #pragma unroll 1
                    for (int output_segment = 0; output_segment < 8; output_segment++) {
                        float _tmem_load_6[16];
                        tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base + (unsigned int)(output_segment * 16));
                        if (output_valid != 0) {
                            {
                                const float2 _prescale2_30 = {inv_sum, inv_sum};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_6[0])[_ps], _prescale2_30);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_6[0 + _ps] *= inv_sum;
                                #endif
                                unsigned int _fp8_pk[4];
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_6[0 + 1]), "f"(_tmem_load_6[0 + 0]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_6[0 + 3]), "f"(_tmem_load_6[0 + 2]));
                                    _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_6[0 + 5]), "f"(_tmem_load_6[0 + 4]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_6[0 + 7]), "f"(_tmem_load_6[0 + 6]));
                                    _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_6[0 + 9]), "f"(_tmem_load_6[0 + 8]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_6[0 + 11]), "f"(_tmem_load_6[0 + 10]));
                                    _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_6[0 + 13]), "f"(_tmem_load_6[0 + 12]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_6[0 + 15]), "f"(_tmem_load_6[0 + 14]));
                                    _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + (long long)output_segment * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                            }
                        }
                    }
                    if (output_valid != 0) {
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                        float partial_lse_value = ((row_sum > 0.0f) ? row_max * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
                        partial_lse[partial_row] = partial_lse_value;
                        if (return_temperature_lse != 0) {
                            partial_temperature_lse[partial_row] = partial_lse_value;
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(o_empty_addr);
            }
            asm volatile("barrier.sync 9, 256;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    __threadfence();
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(completion_counts) + (work_idx);
                        unsigned int _gc_old;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: softmax_odd ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // softmax_odd_main
            int work_idx_1 = bid;
            int metadata_base_1 = work_idx_1 * 6;
            int head_kv_2 = scheduler_metadata[metadata_base_1];
            int row_linear_1 = scheduler_metadata[metadata_base_1 + 1];
            int q_begin_1 = scheduler_metadata[metadata_base_1 + 2];
            int q_count_1 = scheduler_metadata[metadata_base_1 + 3];
            int batch_1 = scheduler_metadata[metadata_base_1 + 4];
            int kv_block_1 = scheduler_metadata[metadata_base_1 + 5];
            int row_ptr_base_1 = head_kv_2 * (total_rows + 1) + row_linear_1;
            int row_start_1 = k2q_row_ptr[row_ptr_base_1] + q_begin_1;
            int q_batch_offset_1 = cu_seqlens_q[batch_1];
            int k_batch_offset_1 = cu_seqlens_k[batch_1];
            int kv_len_1 = 0;
            if (max_pages == 0) {
                kv_len_1 = cu_seqlens_k[batch_1 + 1] - k_batch_offset_1;
            } else {
                kv_len_1 = kv_lens[batch_1];
            }
            int query_offset_1 = 0;
            if (derive_q_offset != 0) {
                query_offset_1 = kv_len_1 - (cu_seqlens_q[batch_1 + 1] - q_batch_offset_1);
            } else {
                query_offset_1 = q_offsets[batch_1];
            }
            int group_count_1 = (q_count_1 + 8 - 1) / 8;
            int metadata_base_0_1 = work_idx_1 * 6;
            int head_kv_1_1 = scheduler_metadata[metadata_base_0_1];
            int row_linear_2_1 = scheduler_metadata[metadata_base_0_1 + 1];
            int q_begin_3_1 = scheduler_metadata[metadata_base_0_1 + 2];
            int q_count_4_1 = scheduler_metadata[metadata_base_0_1 + 3];
            int batch_5_1 = scheduler_metadata[metadata_base_0_1 + 4];
            int kv_block_6_1 = scheduler_metadata[metadata_base_0_1 + 5];
            int row_ptr_base_7_1 = head_kv_1_1 * (total_rows + 1) + row_linear_2_1;
            int row_start_8_1 = k2q_row_ptr[row_ptr_base_7_1] + q_begin_3_1;
            int q_batch_offset_9_1 = cu_seqlens_q[batch_5_1];
            int k_batch_offset_10_1 = cu_seqlens_k[batch_5_1];
            int kv_len_11_1 = 0;
            if (max_pages == 0) {
                kv_len_11_1 = cu_seqlens_k[batch_5_1 + 1] - k_batch_offset_10_1;
            } else {
                kv_len_11_1 = kv_lens[batch_5_1];
            }
            int query_offset_12_1 = 0;
            if (derive_q_offset != 0) {
                query_offset_12_1 = kv_len_11_1 - (cu_seqlens_q[batch_5_1 + 1] - q_batch_offset_9_1);
            } else {
                query_offset_12_1 = q_offsets[batch_5_1];
            }
            int first_local_group_1 = 1;
            int stage_group_count_1 = (group_count_1 + 1 - first_local_group_1) / 2;
            int stage_warp_1 = warp - 4;
            int my_row_1 = stage_warp_1 * 32 + lane;
            int tmem_row_base_1 = stage_warp_1 * 32 << 16;
            #pragma unroll 1
            for (int stage_iteration_1 = 0; stage_iteration_1 < stage_group_count_1; stage_iteration_1++) {
                int group_1 = stage_iteration_1 * 2 + first_local_group_1;
                int absolute_group_1 = group_1;
                int softmax_phase_1 = absolute_group_1 / 2 & 1;
                mbarrier_wait(s_full_addr + 8, softmax_phase_1);
                int whole_group_valid_1 = 1;
                int token_in_group_1 = 0;
                int edge_in_work_1 = 0;
                int packed_q_1 = 0;
                int q_idx_1 = 0;
                int valid_cols_1 = 0;
                int body_valid_1 = 0;
                int tail_valid_1 = 0;
                float row_max_1 = -BLACKWELL_MSA_INF;
                float score_bias_1 = -BLACKWELL_MSA_INF;
                int score_base_1 = taddr + 128 + (unsigned int)tmem_row_base_1;
                if (whole_group_valid_1 != 0) {
                    token_in_group_1 = my_row_1 / 16;
                    edge_in_work_1 = group_1 * 8 + token_in_group_1;
                    int row_valid_1 = ((edge_in_work_1 < q_count_4_1) ? 1 : 0);
                    int owner_lane_1 = lane / 16 * 16;
                    int owned_packed_1 = -1;
                    if (lane == owner_lane_1 && edge_in_work_1 < q_count_4_1) {
                        owned_packed_1 = k2q_qsplit_indices[head_kv_1_1 * nnz_per_head + row_start_8_1 + edge_in_work_1];
                    }
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, owned_packed_1, owner_lane_1);
                    packed_q_1 = _shfl_1;
                    q_idx_1 = packed_q_1 & 16777215;
                    if (row_valid_1 != 0) {
                        valid_cols_1 = kv_len_11_1 - kv_block_6_1 * 128;
                        if (valid_cols_1 > 128) {
                            valid_cols_1 = 128;
                        }
                        if (causal != 0) {
                            int query_position_1 = query_offset_12_1 + q_idx_1;
                            int causal_cols_1 = query_position_1 - kv_block_6_1 * 128 + 1;
                            if (valid_cols_1 > causal_cols_1) {
                                valid_cols_1 = causal_cols_1;
                            }
                        }
                        if (valid_cols_1 < 0) {
                            valid_cols_1 = 0;
                        }
                    }
                    float _tmem_load_7[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31]), "=f"(_tmem_load_7[32]), "=f"(_tmem_load_7[33]), "=f"(_tmem_load_7[34]), "=f"(_tmem_load_7[35]), "=f"(_tmem_load_7[36]), "=f"(_tmem_load_7[37]), "=f"(_tmem_load_7[38]), "=f"(_tmem_load_7[39]), "=f"(_tmem_load_7[40]), "=f"(_tmem_load_7[41]), "=f"(_tmem_load_7[42]), "=f"(_tmem_load_7[43]), "=f"(_tmem_load_7[44]), "=f"(_tmem_load_7[45]), "=f"(_tmem_load_7[46]), "=f"(_tmem_load_7[47]), "=f"(_tmem_load_7[48]), "=f"(_tmem_load_7[49]), "=f"(_tmem_load_7[50]), "=f"(_tmem_load_7[51]), "=f"(_tmem_load_7[52]), "=f"(_tmem_load_7[53]), "=f"(_tmem_load_7[54]), "=f"(_tmem_load_7[55]), "=f"(_tmem_load_7[56]), "=f"(_tmem_load_7[57]), "=f"(_tmem_load_7[58]), "=f"(_tmem_load_7[59]), "=f"(_tmem_load_7[60]), "=f"(_tmem_load_7[61]), "=f"(_tmem_load_7[62]), "=f"(_tmem_load_7[63])
                        : "r"(score_base_1)
                        : "memory");
                    body_valid_1 = valid_cols_1;
                    if (body_valid_1 < 0) {
                        body_valid_1 = 0;
                    }
                    if (body_valid_1 > 0 && body_valid_1 < 64) {
                        uint32_t _slice_lo_mask_8;
                        {
                            int _lim_0 = body_valid_1;
                            if (_lim_0 <= 0) { _slice_lo_mask_8 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_8 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_8) : "r"(_lim_0));
                            }
                        }
                        #pragma unroll
                        for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                            if (!(_slice_lo_mask_8 & (1u << _i_1))) _tmem_load_7[0 + _i_1] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_9;
                        {
                            int _lim_2 = body_valid_1 - 32;
                            if (_lim_2 <= 0) { _slice_lo_mask_9 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_9 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_9) : "r"(_lim_2));
                            }
                        }
                        #pragma unroll
                        for (int _i_3 = 0; _i_3 < 32; _i_3++) {
                            if (!(_slice_lo_mask_9 & (1u << _i_3))) _tmem_load_7[32 + _i_3] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_4 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&_tmem_load_7[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_7[32], _reg_reduce_max2_4);
                    float _tmem_load_7_max = row_max_reduce(_reg_reduce_max2_4);
                    float body_max_1 = _tmem_load_7_max;
                    if (body_valid_1 <= 0) {
                        body_max_1 = -BLACKWELL_MSA_INF;
                    }
                    float _tmem_load_8[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31]), "=f"(_tmem_load_8[32]), "=f"(_tmem_load_8[33]), "=f"(_tmem_load_8[34]), "=f"(_tmem_load_8[35]), "=f"(_tmem_load_8[36]), "=f"(_tmem_load_8[37]), "=f"(_tmem_load_8[38]), "=f"(_tmem_load_8[39]), "=f"(_tmem_load_8[40]), "=f"(_tmem_load_8[41]), "=f"(_tmem_load_8[42]), "=f"(_tmem_load_8[43]), "=f"(_tmem_load_8[44]), "=f"(_tmem_load_8[45]), "=f"(_tmem_load_8[46]), "=f"(_tmem_load_8[47]), "=f"(_tmem_load_8[48]), "=f"(_tmem_load_8[49]), "=f"(_tmem_load_8[50]), "=f"(_tmem_load_8[51]), "=f"(_tmem_load_8[52]), "=f"(_tmem_load_8[53]), "=f"(_tmem_load_8[54]), "=f"(_tmem_load_8[55]), "=f"(_tmem_load_8[56]), "=f"(_tmem_load_8[57]), "=f"(_tmem_load_8[58]), "=f"(_tmem_load_8[59]), "=f"(_tmem_load_8[60]), "=f"(_tmem_load_8[61]), "=f"(_tmem_load_8[62]), "=f"(_tmem_load_8[63])
                        : "r"(score_base_1 + 64)
                        : "memory");
                    tail_valid_1 = valid_cols_1 - 64;
                    if (tail_valid_1 < 0) {
                        tail_valid_1 = 0;
                    }
                    if (valid_cols_1 > 0 && tail_valid_1 < 64) {
                        uint32_t _slice_lo_mask_10;
                        {
                            int _lim_5 = tail_valid_1;
                            if (_lim_5 <= 0) { _slice_lo_mask_10 = 0u; }
                            else if (_lim_5 >= 32) { _slice_lo_mask_10 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_10) : "r"(_lim_5));
                            }
                        }
                        #pragma unroll
                        for (int _i_6 = 0; _i_6 < 32; _i_6++) {
                            if (!(_slice_lo_mask_10 & (1u << _i_6))) _tmem_load_8[0 + _i_6] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_11;
                        {
                            int _lim_7 = tail_valid_1 - 32;
                            if (_lim_7 <= 0) { _slice_lo_mask_11 = 0u; }
                            else if (_lim_7 >= 32) { _slice_lo_mask_11 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_11) : "r"(_lim_7));
                            }
                        }
                        #pragma unroll
                        for (int _i_8 = 0; _i_8 < 32; _i_8++) {
                            if (!(_slice_lo_mask_11 & (1u << _i_8))) _tmem_load_8[32 + _i_8] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_9 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&_tmem_load_8[0], _reg_reduce_max2_9);
                    row_max_x32_accum(&_tmem_load_8[32], _reg_reduce_max2_9);
                    float _tmem_load_8_max = row_max_reduce(_reg_reduce_max2_9);
                    float tail_max_1 = _tmem_load_8_max;
                    if (tail_valid_1 <= 0) {
                        tail_max_1 = -BLACKWELL_MSA_INF;
                    }
                    float _max_1 = max_noftz(body_max_1, tail_max_1);
                    row_max_1 = _max_1;
                    float safe_max_1 = ((row_max_1 == -BLACKWELL_MSA_INF) ? 0.0f : row_max_1);
                    score_bias_1 = ((valid_cols_1 > 0) ? (-safe_max_1) * softmax_scale_log2 : -BLACKWELL_MSA_INF);
                }
                mbarrier_wait(p_empty_addr + 8, softmax_phase_1 ^ 1);
                float row_sum_1 = 0.0f;
                if (whole_group_valid_1 != 0) {
                    int p_base_1 = taddr + 128 + 96 + (unsigned int)tmem_row_base_1;
                    float _tmem_load_9[32];
                    tmem_ld_x32(&_tmem_load_9[0], score_base_1);
                    uint32_t _slice_lo_mask_12;
                    {
                        int _lim_10 = valid_cols_1;
                        if (_lim_10 <= 0) { _slice_lo_mask_12 = 0u; }
                        else if (_lim_10 >= 32) { _slice_lo_mask_12 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_12) : "r"(_lim_10));
                        }
                    }
                    #pragma unroll
                    for (int _i_11 = 0; _i_11 < 32; _i_11++) {
                        if (!(_slice_lo_mask_12 & (1u << _i_11))) _tmem_load_9[0 + _i_11] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_12 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_13 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_lf], _fma_b2_12, _fma_c2_13);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_9[_le] = approx_exp2(_tmem_load_9[_le]);
                    }
                    float2 _reg_reduce_sum2_14 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_9[0], &_reg_reduce_sum2_14);
                    float _tmem_load_9_sum = _reg_reduce_sum2_14.x + _reg_reduce_sum2_14.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_9[0 + _j * 4]), "f"(_tmem_load_9[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_9[0 + _j * 4 + 2]), "f"(_tmem_load_9[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base_1, _pv_packed);
                    }
                    row_sum_1 += _tmem_load_9_sum;
                    float _tmem_load_10[32];
                    tmem_ld_x32(&_tmem_load_10[0], score_base_1 + 32);
                    uint32_t _slice_lo_mask_13;
                    {
                        int _lim_15 = valid_cols_1 - 32;
                        if (_lim_15 <= 0) { _slice_lo_mask_13 = 0u; }
                        else if (_lim_15 >= 32) { _slice_lo_mask_13 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_13) : "r"(_lim_15));
                        }
                    }
                    #pragma unroll
                    for (int _i_16 = 0; _i_16 < 32; _i_16++) {
                        if (!(_slice_lo_mask_13 & (1u << _i_16))) _tmem_load_10[0 + _i_16] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_18 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_10)[_lf], _fma_b2_17, _fma_c2_18);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_10[_le] = approx_exp2(_tmem_load_10[_le]);
                    }
                    float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_10[0], &_reg_reduce_sum2_19);
                    float _tmem_load_10_sum = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_10[0 + _j * 4]), "f"(_tmem_load_10[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_10[0 + _j * 4 + 2]), "f"(_tmem_load_10[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base_1 + 8, _pv_packed);
                    }
                    row_sum_1 += _tmem_load_10_sum;
                    float _tmem_load_11[32];
                    tmem_ld_x32(&_tmem_load_11[0], score_base_1 + 64);
                    uint32_t _slice_lo_mask_14;
                    {
                        int _lim_20 = valid_cols_1 - 64;
                        if (_lim_20 <= 0) { _slice_lo_mask_14 = 0u; }
                        else if (_lim_20 >= 32) { _slice_lo_mask_14 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_14) : "r"(_lim_20));
                        }
                    }
                    #pragma unroll
                    for (int _i_21 = 0; _i_21 < 32; _i_21++) {
                        if (!(_slice_lo_mask_14 & (1u << _i_21))) _tmem_load_11[0 + _i_21] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_22 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_23 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_lf], _fma_b2_22, _fma_c2_23);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_11[_le] = approx_exp2(_tmem_load_11[_le]);
                    }
                    float2 _reg_reduce_sum2_24 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_11[0], &_reg_reduce_sum2_24);
                    float _tmem_load_11_sum = _reg_reduce_sum2_24.x + _reg_reduce_sum2_24.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_11[0 + _j * 4]), "f"(_tmem_load_11[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_11[0 + _j * 4 + 2]), "f"(_tmem_load_11[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base_1 + 16, _pv_packed);
                    }
                    row_sum_1 += _tmem_load_11_sum;
                    float _tmem_load_12[32];
                    tmem_ld_x32(&_tmem_load_12[0], score_base_1 + 96);
                    uint32_t _slice_lo_mask_15;
                    {
                        int _lim_25 = valid_cols_1 - 96;
                        if (_lim_25 <= 0) { _slice_lo_mask_15 = 0u; }
                        else if (_lim_25 >= 32) { _slice_lo_mask_15 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_15) : "r"(_lim_25));
                        }
                    }
                    #pragma unroll
                    for (int _i_26 = 0; _i_26 < 32; _i_26++) {
                        if (!(_slice_lo_mask_15 & (1u << _i_26))) _tmem_load_12[0 + _i_26] = -BLACKWELL_MSA_INF;
                    }
                    const float2 _fma_b2_27 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_28 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 16; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_12)[_lf], _fma_b2_27, _fma_c2_28);
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        _tmem_load_12[_le] = approx_exp2(_tmem_load_12[_le]);
                    }
                    float2 _reg_reduce_sum2_29 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_12[0], &_reg_reduce_sum2_29);
                    float _tmem_load_12_sum = _reg_reduce_sum2_29.x + _reg_reduce_sum2_29.y;
                    {
                        uint32_t _pv_packed[8];
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++) {
                            uint32_t _pk;
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_pk) : "f"(_tmem_load_12[0 + _j * 4]), "f"(_tmem_load_12[0 + _j * 4 + 1]),
                                  "f"(_tmem_load_12[0 + _j * 4 + 2]), "f"(_tmem_load_12[0 + _j * 4 + 3]));
                            _pv_packed[_j] = _pk;
                        }
                        tmem_st_x8_u32(p_base_1 + 24, _pv_packed);
                    }
                    row_sum_1 += _tmem_load_12_sum;
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr + 8);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr + 8);
                mbarrier_wait(o_full_addr + 8, softmax_phase_1);
                if (whole_group_valid_1 != 0) {
                    int q_head_local_1 = my_row_1 - token_in_group_1 * 16;
                    int output_valid_1 = 0;
                    long long partial_row_1 = 0;
                    float inv_sum_1 = 0.0f;
                    if (edge_in_work_1 < q_count_4_1) {
                        int split_slot_1 = packed_q_1 >> 24 & 255;
                        if (split_slot_1 >= 0 && split_slot_1 < topk) {
                            output_valid_1 = 1;
                            int q_abs_1 = q_batch_offset_9_1 + q_idx_1;
                            int q_head_1 = head_kv_1_1 * 16 + q_head_local_1;
                            partial_row_1 = (long long)split_slot_1 * (long long)total_q * (long long)num_q_heads + (long long)q_abs_1 * (long long)num_q_heads + (long long)q_head_1;
                            float _rcp_1 = approx_rcp(row_sum_1);
                            inv_sum_1 = ((row_sum_1 > 0.0f && row_sum_1 == row_sum_1) ? _rcp_1 : 0.0f);
                        }
                    }
                    long long partial_base_1 = partial_row_1 * 128;
                    #pragma unroll 1
                    for (int output_segment_1 = 0; output_segment_1 < 8; output_segment_1++) {
                        float _tmem_load_13[16];
                        tmem_ld_x16(&_tmem_load_13[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + 128 + (unsigned int)tmem_row_base_1 + (unsigned int)(output_segment_1 * 16));
                        if (output_valid_1 != 0) {
                            {
                                const float2 _prescale2_30 = {inv_sum_1, inv_sum_1};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_13[0])[_ps], _prescale2_30);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_13[0 + _ps] *= inv_sum_1;
                                #endif
                                unsigned int _fp8_pk[4];
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_13[0 + 1]), "f"(_tmem_load_13[0 + 0]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_13[0 + 3]), "f"(_tmem_load_13[0 + 2]));
                                    _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_13[0 + 5]), "f"(_tmem_load_13[0 + 4]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_13[0 + 7]), "f"(_tmem_load_13[0 + 6]));
                                    _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_13[0 + 9]), "f"(_tmem_load_13[0 + 8]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_13[0 + 11]), "f"(_tmem_load_13[0 + 10]));
                                    _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_13[0 + 13]), "f"(_tmem_load_13[0 + 12]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_13[0 + 15]), "f"(_tmem_load_13[0 + 14]));
                                    _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + (long long)output_segment_1 * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                            }
                        }
                    }
                    if (output_valid_1 != 0) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(row_sum_1));
                        float partial_lse_value_1 = ((row_sum_1 > 0.0f) ? row_max_1 * softmax_scale_log2 * 0.6931471805599453f + _log2_1 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
                        partial_lse[partial_row_1] = partial_lse_value_1;
                        if (return_temperature_lse != 0) {
                            partial_temperature_lse[partial_row_1] = partial_lse_value_1;
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(o_empty_addr + 8);
            }
            asm volatile("barrier.sync 9, 256;" ::: "memory");
        }
    }
    // ---- Role: qload ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // qload_main
            int work_idx_2 = bid;
            int metadata_base_2 = work_idx_2 * 6;
            int head_kv_3 = scheduler_metadata[metadata_base_2];
            int row_linear_3 = scheduler_metadata[metadata_base_2 + 1];
            int q_begin_2 = scheduler_metadata[metadata_base_2 + 2];
            int q_count_2 = scheduler_metadata[metadata_base_2 + 3];
            int batch_2 = scheduler_metadata[metadata_base_2 + 4];
            int kv_block_2 = scheduler_metadata[metadata_base_2 + 5];
            int row_ptr_base_2 = head_kv_3 * (total_rows + 1) + row_linear_3;
            int row_start_2 = k2q_row_ptr[row_ptr_base_2] + q_begin_2;
            int q_batch_offset_2 = cu_seqlens_q[batch_2];
            int k_batch_offset_2 = cu_seqlens_k[batch_2];
            int kv_len_2 = 0;
            if (max_pages == 0) {
                kv_len_2 = cu_seqlens_k[batch_2 + 1] - k_batch_offset_2;
            } else {
                kv_len_2 = kv_lens[batch_2];
            }
            int query_offset_2 = 0;
            if (derive_q_offset != 0) {
                query_offset_2 = kv_len_2 - (cu_seqlens_q[batch_2 + 1] - q_batch_offset_2);
            } else {
                query_offset_2 = q_offsets[batch_2];
            }
            int group_count_2 = (q_count_2 + 8 - 1) / 8;
            #pragma unroll 1
            for (int group_2 = 0; group_2 < group_count_2; group_2++) {
                int q_stage = group_2 & 1;
                int q_phase = group_2 / 2 & 1;
                mbarrier_wait(q_empty_addr + (q_stage) * 8, q_phase ^ 1);
                int q_stage_addr = q_store_smem_addr + (unsigned int)(q_stage * 32768);
                {
                    int qload_warp = warp - 8;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 8192);
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (elect_sync()) {
                        int token_in_group_2 = qload_warp * 2;
                        int edge_in_work_2 = group_2 * 8 + token_in_group_2;
                        int edge_valid = ((edge_in_work_2 < q_count_2) ? 1 : 0);
                        int safe_edge = ((edge_valid != 0) ? edge_in_work_2 : 0);
                        int packed_q_2 = k2q_qsplit_indices[head_kv_3 * nnz_per_head + row_start_2 + safe_edge];
                        int decoded_q_abs = q_batch_offset_2 + (packed_q_2 & 16777215);
                        int q_abs_2 = ((edge_valid != 0) ? decoded_q_abs : 0);
                        int dst_offset = token_in_group_2 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset, (&q), 0, head_kv_3 * 16, 0, q_abs_2, q_full_addr + (q_stage) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset, (&q), 0, head_kv_3 * 16, 1, q_abs_2, q_full_addr + (q_stage) * 8);
                        int token_in_group_0 = qload_warp * 2 + 1;
                        int edge_in_work_1_1 = group_2 * 8 + token_in_group_0;
                        int edge_valid_2 = ((edge_in_work_1_1 < q_count_2) ? 1 : 0);
                        int safe_edge_3 = ((edge_valid_2 != 0) ? edge_in_work_1_1 : 0);
                        int packed_q_4 = k2q_qsplit_indices[head_kv_3 * nnz_per_head + row_start_2 + safe_edge_3];
                        int decoded_q_abs_5 = q_batch_offset_2 + (packed_q_4 & 16777215);
                        int q_abs_6 = ((edge_valid_2 != 0) ? decoded_q_abs_5 : 0);
                        int dst_offset_7 = token_in_group_0 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset_7, (&q), 0, head_kv_3 * 16, 0, q_abs_6, q_full_addr + (q_stage) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset_7, (&q), 0, head_kv_3 * 16, 1, q_abs_6, q_full_addr + (q_stage) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            #pragma unroll
            for (int _one_work = 0; _one_work < 1; _one_work++) {
                int work_idx_3 = bid;
                int metadata_base_3 = work_idx_3 * 6;
                int head_kv_4 = scheduler_metadata[metadata_base_3];
                int row_linear_4 = scheduler_metadata[metadata_base_3 + 1];
                int q_begin_4 = scheduler_metadata[metadata_base_3 + 2];
                int q_count_3 = scheduler_metadata[metadata_base_3 + 3];
                int batch_3 = scheduler_metadata[metadata_base_3 + 4];
                int kv_block_3 = scheduler_metadata[metadata_base_3 + 5];
                int row_ptr_base_3 = head_kv_4 * (total_rows + 1) + row_linear_4;
                int row_start_3 = k2q_row_ptr[row_ptr_base_3] + q_begin_4;
                int q_batch_offset_3 = cu_seqlens_q[batch_3];
                int k_batch_offset_3 = cu_seqlens_k[batch_3];
                int kv_len_3 = 0;
                if (max_pages == 0) {
                    kv_len_3 = cu_seqlens_k[batch_3 + 1] - k_batch_offset_3;
                } else {
                    kv_len_3 = kv_lens[batch_3];
                }
                int query_offset_3 = 0;
                if (derive_q_offset != 0) {
                    query_offset_3 = kv_len_3 - (cu_seqlens_q[batch_3 + 1] - q_batch_offset_3);
                } else {
                    query_offset_3 = q_offsets[batch_3];
                }
                int group_count_3 = (q_count_3 + 8 - 1) / 8;
                int group_base = 0;
                int work_phase = 0;
                mbarrier_wait(k_full_addr, 0);
                int absolute_group_2 = group_base;
                int q_stage_1 = absolute_group_2 & 1;
                int q_phase_1 = absolute_group_2 / 2 & 1;
                mbarrier_wait(q_full_addr + (q_stage_1) * 8, q_phase_1);
                mbarrier_wait(s_empty_addr + (q_stage_1) * 8, q_phase_1 ^ 1);
                int _mma_a_lo_0 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_1) * 2048);
                int _mma_b_lo_0 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_scores + (q_stage_1 * 128))), "r"(0));
                elect_commit(s_full_addr + (q_stage_1) * 8);
                elect_commit(q_empty_addr + (q_stage_1) * 8);
                if (group_count_3 > 1) {
                    int absolute_group_0 = group_base + 1;
                    int q_stage_1_1 = absolute_group_0 & 1;
                    int q_phase_2 = absolute_group_0 / 2 & 1;
                    mbarrier_wait(q_full_addr + (q_stage_1_1) * 8, q_phase_2);
                    mbarrier_wait(s_empty_addr + (q_stage_1_1) * 8, q_phase_2 ^ 1);
                    int _mma_a_lo_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_1_1) * 2048);
                    int _mma_b_lo_1 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (q_stage_1_1 * 128))), "r"(0));
                    elect_commit(s_full_addr + (q_stage_1_1) * 8);
                    elect_commit(q_empty_addr + (q_stage_1_1) * 8);
                }
                mbarrier_wait(v_full_addr, work_phase);
                #pragma unroll 1
                for (int group_3 = 2; group_3 < group_count_3; group_3++) {
                    int pv_group = group_3 - 2;
                    int absolute_group_0_1 = group_base + pv_group;
                    int pv_stage = absolute_group_0_1 & 1;
                    int pv_phase = absolute_group_0_1 / 2 & 1;
                    mbarrier_wait(p_full_addr + (pv_stage) * 8, pv_phase);
                    mbarrier_wait(o_empty_addr + (pv_stage) * 8, pv_phase ^ 1);
                    int _mma_b_lo_2 = make_warp_uniform((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(_mma_b_lo_2), "r"(tmem_scores + (pv_stage * 128 + 96)), "r"(0));
                    elect_commit(o_full_addr + (pv_stage) * 8);
                    elect_commit(p_empty_addr + (pv_stage) * 8);
                    int absolute_group_1_1 = group_base + group_3;
                    int q_stage_2 = absolute_group_1_1 & 1;
                    int q_phase_3 = absolute_group_1_1 / 2 & 1;
                    mbarrier_wait(q_full_addr + (q_stage_2) * 8, q_phase_3);
                    mbarrier_wait(s_empty_addr + (q_stage_2) * 8, q_phase_3 ^ 1);
                    int _mma_a_lo_3 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_2) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_scores + (q_stage_2 * 128))), "r"(0));
                    elect_commit(s_full_addr + (q_stage_2) * 8);
                    elect_commit(q_empty_addr + (q_stage_2) * 8);
                }
                int drain_start = ((group_count_3 == 1) ? 0 : group_count_3 - 2);
                #pragma unroll 1
                for (int pv_group_1 = drain_start; pv_group_1 < group_count_3; pv_group_1++) {
                    int absolute_group_0_2 = group_base + pv_group_1;
                    int pv_stage_1 = absolute_group_0_2 & 1;
                    int pv_phase_1 = absolute_group_0_2 / 2 & 1;
                    mbarrier_wait(p_full_addr + (pv_stage_1) * 8, pv_phase_1);
                    mbarrier_wait(o_empty_addr + (pv_stage_1) * 8, pv_phase_1 ^ 1);
                    int _mma_b_lo_4 = make_warp_uniform((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(_mma_b_lo_4), "r"(tmem_scores + (pv_stage_1 * 128 + 96)), "r"(0));
                    elect_commit(o_full_addr + (pv_stage_1) * 8);
                    elect_commit(p_empty_addr + (pv_stage_1) * 8);
                }
                #pragma unroll 1
                for (int completed_group = drain_start; completed_group < group_count_3; completed_group++) {
                    int absolute_group_0_3 = group_base + completed_group;
                    int completed_stage = absolute_group_0_3 & 1;
                    int completed_phase = absolute_group_0_3 / 2 & 1;
                    mbarrier_wait(o_empty_addr + (completed_stage) * 8, completed_phase);
                }
                if (elect_sync()) {
                    mbarrier_arrive(kv_empty_addr);
                }
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: transform ----
    if (warp >= 13 && warp <= 14) {
        { // transform_main
            unsigned int _phase_fp8_k_full_0 = 0;
            mbarrier_wait(fp8_k_full_addr, _phase_fp8_k_full_0);
            _phase_fp8_k_full_0 ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            {
                const char* _src_ptr = smem_raw + (fp8_smem_addr - smem);
                char* _dst_ptr = smem_raw + (k_smem_addr - smem);
                const int _tid = (int)threadIdx.x - (13) * 32;
                #pragma unroll 4
                for (int _off = _tid; _off < 2048; _off += 64) {
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
            asm volatile("barrier.sync 10, 64;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (warp == 13) {
                if (elect_sync()) {
                    mbarrier_arrive(k_full_addr);
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            int work_idx_4 = bid;
            int metadata_base_4 = work_idx_4 * 6;
            int head_kv_5 = scheduler_metadata[metadata_base_4];
            int row_linear_5 = scheduler_metadata[metadata_base_4 + 1];
            int q_begin_5 = scheduler_metadata[metadata_base_4 + 2];
            int q_count_5 = scheduler_metadata[metadata_base_4 + 3];
            int batch_4 = scheduler_metadata[metadata_base_4 + 4];
            int kv_block_4 = scheduler_metadata[metadata_base_4 + 5];
            int row_ptr_base_4 = head_kv_5 * (total_rows + 1) + row_linear_5;
            int row_start_4 = k2q_row_ptr[row_ptr_base_4] + q_begin_5;
            int q_batch_offset_4 = cu_seqlens_q[batch_4];
            int k_batch_offset_4 = cu_seqlens_k[batch_4];
            int kv_len_4 = 0;
            if (max_pages == 0) {
                kv_len_4 = cu_seqlens_k[batch_4 + 1] - k_batch_offset_4;
            } else {
                kv_len_4 = kv_lens[batch_4];
            }
            int query_offset_4 = 0;
            if (derive_q_offset != 0) {
                query_offset_4 = kv_len_4 - (cu_seqlens_q[batch_4 + 1] - q_batch_offset_4);
            } else {
                query_offset_4 = q_offsets[batch_4];
            }
            int token_base = k_batch_offset_4 + kv_block_4 * 128;
            int page_head = head_kv_5;
            mbarrier_wait(kv_empty_addr, 1);
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(fp8_k_full_addr, 16384);
                int token0 = token_base;
                int token1 = token_base + 64;
                tma_3d_gmem2smem(fp8_smem_addr, (&k), 0, token0, page_head, fp8_k_full_addr);
                tma_3d_gmem2smem(fp8_smem_addr + 8192, (&k), 0, token1, page_head, fp8_k_full_addr);
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(v_full_addr, 16384);
                int token0_1 = token_base;
                int token1_1 = token_base + 64;
                tma_3d_gmem2smem(v_smem_addr, (&v), 0, token0_1, page_head, v_full_addr);
                tma_3d_gmem2smem(v_smem_addr + 8192, (&v), 0, token1_1, page_head, v_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
