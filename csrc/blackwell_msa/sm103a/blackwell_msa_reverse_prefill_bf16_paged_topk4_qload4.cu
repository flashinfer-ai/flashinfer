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
#define SMEM_V_SMEM_STAGE_BYTES 32768
#define SMEM_V_SMEM_STRIDE 32768
#define SMEM_V_CONVERT_SMEM_OFF 99328
#define SMEM_V_CONVERT_SMEM_STAGE_BYTES 32768
#define SMEM_V_CONVERT_SMEM_STRIDE 32768
#define SMEM_FP8_SMEM_OFF 132096
#define SMEM_FP8_SMEM_STAGE_BYTES 16384
#define SMEM_FP8_SMEM_STRIDE 16384
#define SMEM_TOTAL 148480
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
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
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


__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(*x0_ptr, -127.0f), x1 = max_noftz(*x1_ptr, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    *x0_ptr = r0; *x1_ptr = r1;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
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

__global__ __launch_bounds__(512, 1) void
kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_fp8partial_temp1reuse_sm100(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, int* __restrict__ scheduler_metadata, int* __restrict__ k2q_row_ptr, int* __restrict__ k2q_qsplit_indices, uint8_t* __restrict__ partial_o, float* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int q_group_segment_end_21, int q_group_segment_end_20, int q_group_segment_end_19, int q_group_segment_end_18, int q_group_segment_end_17, int q_group_segment_end_16, int q_group_segment_end_15, int q_group_segment_end_14, int q_group_segment_end_13, int q_group_segment_end_12, int q_group_segment_end_11, int q_group_segment_end_10, int q_group_segment_end_9, int q_group_segment_end_8, int q_group_segment_end_7, int q_group_segment_end_6, int q_group_segment_end_5, int q_group_segment_end_4, int q_group_segment_end_3, int q_group_segment_end_2, int total_q, int num_q_heads, int num_kv_heads, int total_rows, int nnz_per_head, int work_capacity, int num_work_items, int topk, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_temperature_lse)
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
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int v_smem_addr = smem + 99328;
    __nv_bfloat16* v_convert_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int v_convert_smem_addr = smem + 99328;
    uint8_t* fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int fp8_smem_addr = smem + 132096;

    // Mbarrier init (13 groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

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
            // fp8_k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // fp8_v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // fp8_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // --- pipeline 's_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            mbarrier_init(smem + 96, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // --- pipeline 'o_pipe' ---
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 0) {
        int _tmem_hold = smem + 168;
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
    #define fp8_k_full_addr (mbar_base + 48)
    #define fp8_v_full_addr (mbar_base + 56)
    #define fp8_empty_addr (mbar_base + 64)
    #define s_full_addr (mbar_base + 72)
    #define s_empty_addr (mbar_base + 88)
    #define p_full_addr (mbar_base + 104)
    #define p_empty_addr (mbar_base + 120)
    #define o_full_addr (mbar_base + 136)
    #define o_empty_addr (mbar_base + 152)
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
            int group_count = 21;
            if (blockIdx.x >= q_group_segment_end_21) {
                group_count = 20;
            }
            if (blockIdx.x >= q_group_segment_end_20) {
                group_count = 19;
            }
            if (blockIdx.x >= q_group_segment_end_19) {
                group_count = 18;
            }
            if (blockIdx.x >= q_group_segment_end_18) {
                group_count = 17;
            }
            if (blockIdx.x >= q_group_segment_end_17) {
                group_count = 16;
            }
            if (blockIdx.x >= q_group_segment_end_16) {
                group_count = 15;
            }
            if (blockIdx.x >= q_group_segment_end_15) {
                group_count = 14;
            }
            if (blockIdx.x >= q_group_segment_end_14) {
                group_count = 13;
            }
            if (blockIdx.x >= q_group_segment_end_13) {
                group_count = 12;
            }
            if (blockIdx.x >= q_group_segment_end_12) {
                group_count = 11;
            }
            if (blockIdx.x >= q_group_segment_end_11) {
                group_count = 10;
            }
            if (blockIdx.x >= q_group_segment_end_10) {
                group_count = 9;
            }
            if (blockIdx.x >= q_group_segment_end_9) {
                group_count = 8;
            }
            if (blockIdx.x >= q_group_segment_end_8) {
                group_count = 7;
            }
            if (blockIdx.x >= q_group_segment_end_7) {
                group_count = 6;
            }
            if (blockIdx.x >= q_group_segment_end_6) {
                group_count = 5;
            }
            if (blockIdx.x >= q_group_segment_end_5) {
                group_count = 4;
            }
            if (blockIdx.x >= q_group_segment_end_4) {
                group_count = 3;
            }
            if (blockIdx.x >= q_group_segment_end_3) {
                group_count = 2;
            }
            if (blockIdx.x >= q_group_segment_end_2) {
                group_count = 1;
            }
            int group_count_0 = group_count;
            int work_idx = blockIdx.x;
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
            int kv_len = kv_lens[batch];
            if (max_pages == 0) {
                kv_len = cu_seqlens_k[batch + 1] - k_batch_offset;
            }
            int query_offset = q_offsets[batch];
            if (derive_q_offset != 0) {
                query_offset = kv_len - (cu_seqlens_q[batch + 1] - q_batch_offset);
            }
            int stage_warp = warp;
            int my_row = stage_warp * 32 + lane;
            int tmem_row_base = stage_warp * 32 << 16;
            #pragma unroll 1
            for (int stage_iteration = 0; stage_iteration < (group_count_0 + 1) / 2; stage_iteration++) {
                int group = stage_iteration * 2;
                int softmax_phase = stage_iteration & 1;
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
                    token_in_group = my_row / 4;
                    edge_in_work = group * 32 + token_in_group;
                    int row_valid = ((edge_in_work < q_count) ? 1 : 0);
                    int owner_lane = lane / 4 * 4;
                    int owned_packed = -1;
                    if (lane == owner_lane && edge_in_work < q_count) {
                        owned_packed = k2q_qsplit_indices[head_kv * nnz_per_head + row_start + edge_in_work];
                    }
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, owned_packed, owner_lane);
                    packed_q = _shfl_0;
                    q_idx = packed_q & 16777215;
                    if (row_valid != 0) {
                        valid_cols = kv_len - kv_block * 128;
                        if (valid_cols > 128) {
                            valid_cols = 128;
                        }
                        if (causal != 0) {
                            int query_position = query_offset + q_idx;
                            int causal_cols = query_position - kv_block * 128 + 1;
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
                    int p_base = taddr + 64 + (unsigned int)tmem_row_base;
                    float _tmem_load_2[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31]), "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                        : "r"(score_base)
                        : "memory");
                    if (body_valid > 0 && body_valid < 64) {
                        uint32_t _slice_lo_mask_4;
                        {
                            int _lim_10 = body_valid;
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
                        uint32_t _slice_lo_mask_5;
                        {
                            int _lim_12 = body_valid - 32;
                            if (_lim_12 <= 0) { _slice_lo_mask_5 = 0u; }
                            else if (_lim_12 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_12));
                            }
                        }
                        #pragma unroll
                        for (int _i_13 = 0; _i_13 < 32; _i_13++) {
                            if (!(_slice_lo_mask_5 & (1u << _i_13))) _tmem_load_2[32 + _i_13] = -BLACKWELL_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_14 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_15 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_lf], _fma_b2_14, _fma_c2_15);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_2[_le] = approx_exp2(_tmem_load_2[_le]);
                    }
                    float2 _reg_reduce_sum2_16 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_2[0], &_reg_reduce_sum2_16);
                    softmax_block_sum(&_tmem_load_2[32], &_reg_reduce_sum2_16);
                    float _tmem_load_2_sum = _reg_reduce_sum2_16.x + _reg_reduce_sum2_16.y;
                    row_sum = _tmem_load_2_sum;
                    uint32_t _tmem_load_2_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                        _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[31]))
                        : "memory");
                    float _tmem_load_3[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31]), "=f"(_tmem_load_3[32]), "=f"(_tmem_load_3[33]), "=f"(_tmem_load_3[34]), "=f"(_tmem_load_3[35]), "=f"(_tmem_load_3[36]), "=f"(_tmem_load_3[37]), "=f"(_tmem_load_3[38]), "=f"(_tmem_load_3[39]), "=f"(_tmem_load_3[40]), "=f"(_tmem_load_3[41]), "=f"(_tmem_load_3[42]), "=f"(_tmem_load_3[43]), "=f"(_tmem_load_3[44]), "=f"(_tmem_load_3[45]), "=f"(_tmem_load_3[46]), "=f"(_tmem_load_3[47]), "=f"(_tmem_load_3[48]), "=f"(_tmem_load_3[49]), "=f"(_tmem_load_3[50]), "=f"(_tmem_load_3[51]), "=f"(_tmem_load_3[52]), "=f"(_tmem_load_3[53]), "=f"(_tmem_load_3[54]), "=f"(_tmem_load_3[55]), "=f"(_tmem_load_3[56]), "=f"(_tmem_load_3[57]), "=f"(_tmem_load_3[58]), "=f"(_tmem_load_3[59]), "=f"(_tmem_load_3[60]), "=f"(_tmem_load_3[61]), "=f"(_tmem_load_3[62]), "=f"(_tmem_load_3[63])
                        : "r"(score_base + 64)
                        : "memory");
                    if (valid_cols > 0 && tail_valid < 64) {
                        uint32_t _slice_lo_mask_6;
                        {
                            int _lim_17 = tail_valid;
                            if (_lim_17 <= 0) { _slice_lo_mask_6 = 0u; }
                            else if (_lim_17 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_17));
                            }
                        }
                        #pragma unroll
                        for (int _i_18 = 0; _i_18 < 32; _i_18++) {
                            if (!(_slice_lo_mask_6 & (1u << _i_18))) _tmem_load_3[0 + _i_18] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_7;
                        {
                            int _lim_19 = tail_valid - 32;
                            if (_lim_19 <= 0) { _slice_lo_mask_7 = 0u; }
                            else if (_lim_19 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_19));
                            }
                        }
                        #pragma unroll
                        for (int _i_20 = 0; _i_20 < 32; _i_20++) {
                            if (!(_slice_lo_mask_7 & (1u << _i_20))) _tmem_load_3[32 + _i_20] = -BLACKWELL_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_21 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_22 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_lf], _fma_b2_21, _fma_c2_22);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_3[_le] = approx_exp2(_tmem_load_3[_le]);
                    }
                    float2 _reg_reduce_sum2_23 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_3[0], &_reg_reduce_sum2_23);
                    softmax_block_sum(&_tmem_load_3[32], &_reg_reduce_sum2_23);
                    float _tmem_load_3_sum = _reg_reduce_sum2_23.x + _reg_reduce_sum2_23.y;
                    row_sum += _tmem_load_3_sum;
                    uint32_t _tmem_load_3_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                        _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[31]))
                        : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr);
                mbarrier_wait(o_full_addr, softmax_phase);
                if (whole_group_valid != 0) {
                    int q_head_local = my_row - token_in_group * 4;
                    int output_valid = 0;
                    long long partial_row = 0;
                    float inv_sum = 0.0f;
                    if (edge_in_work < q_count) {
                        int split_slot = packed_q >> 24 & 255;
                        if (split_slot >= 0 && split_slot < topk) {
                            output_valid = 1;
                            int q_abs = q_batch_offset + q_idx;
                            int q_head = head_kv * 4 + q_head_local;
                            partial_row = (long long)split_slot * (long long)total_q * (long long)num_q_heads + (long long)q_abs * (long long)num_q_heads + (long long)q_head;
                            float _rcp_0 = approx_rcp(row_sum);
                            inv_sum = ((row_sum > 0.0f && row_sum == row_sum) ? _rcp_0 : 0.0f);
                        }
                    }
                    long long partial_base = partial_row * 128;
                    float row_abs_max = 0.0f;
                    #pragma unroll 1
                    for (int output_segment = 0; output_segment < 8; output_segment++) {
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base + (unsigned int)(output_segment * 16));
                        float _tmem_load_4_max = _tmem_load_4[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _tmem_load_4_max = max_noftz(_tmem_load_4_max, _tmem_load_4[_lr]);
                        }
                        float segment_max = _tmem_load_4_max;
                        float _tmem_load_4_min = _tmem_load_4[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _tmem_load_4_min = fminf(_tmem_load_4_min, _tmem_load_4[_lr]);
                        }
                        float segment_min = _tmem_load_4_min;
                        float segment_neg_min = -segment_min;
                        float _max_1 = max_noftz(segment_max, segment_neg_min);
                        float segment_abs_max = _max_1;
                        float _max_2 = max_noftz(row_abs_max, segment_abs_max);
                        row_abs_max = _max_2;
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float dequant_scale = 0.0f;
                    float quant_scale = 0.0f;
                    if (row_abs_max > 0.0f && row_abs_max == row_abs_max) {
                        dequant_scale = row_abs_max * inv_sum * 0.002232142857142857f;
                        quant_scale = 448.0f / row_abs_max;
                    }
                    if (output_valid != 0) {
                        partial_scale[partial_row] = dequant_scale;
                    }
                    #pragma unroll 1
                    for (int output_segment_1 = 0; output_segment_1 < 8; output_segment_1++) {
                        float _tmem_load_5[16];
                        tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base + (unsigned int)(output_segment_1 * 16));
                        if (output_valid != 0) {
                            {
                                const float2 _prescale2_24 = {quant_scale, quant_scale};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_5[0])[_ps], _prescale2_24);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_5[0 + _ps] *= quant_scale;
                                #endif
                                unsigned int _fp8_pk[4];
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_5[0 + 1]), "f"(_tmem_load_5[0 + 0]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_5[0 + 3]), "f"(_tmem_load_5[0 + 2]));
                                    _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_5[0 + 5]), "f"(_tmem_load_5[0 + 4]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_5[0 + 7]), "f"(_tmem_load_5[0 + 6]));
                                    _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_5[0 + 9]), "f"(_tmem_load_5[0 + 8]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_5[0 + 11]), "f"(_tmem_load_5[0 + 10]));
                                    _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_5[0 + 13]), "f"(_tmem_load_5[0 + 12]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_5[0 + 15]), "f"(_tmem_load_5[0 + 14]));
                                    _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + (long long)output_segment_1 * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                            }
                        }
                    }
                    if (output_valid != 0) {
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                        partial_lse[partial_row] = ((row_sum > 0.0f) ? row_max * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(o_empty_addr);
            }
        }
    }
    // ---- Role: softmax_odd ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // softmax_odd_main
            int group_count_1 = 21;
            if (blockIdx.x >= q_group_segment_end_21) {
                group_count_1 = 20;
            }
            if (blockIdx.x >= q_group_segment_end_20) {
                group_count_1 = 19;
            }
            if (blockIdx.x >= q_group_segment_end_19) {
                group_count_1 = 18;
            }
            if (blockIdx.x >= q_group_segment_end_18) {
                group_count_1 = 17;
            }
            if (blockIdx.x >= q_group_segment_end_17) {
                group_count_1 = 16;
            }
            if (blockIdx.x >= q_group_segment_end_16) {
                group_count_1 = 15;
            }
            if (blockIdx.x >= q_group_segment_end_15) {
                group_count_1 = 14;
            }
            if (blockIdx.x >= q_group_segment_end_14) {
                group_count_1 = 13;
            }
            if (blockIdx.x >= q_group_segment_end_13) {
                group_count_1 = 12;
            }
            if (blockIdx.x >= q_group_segment_end_12) {
                group_count_1 = 11;
            }
            if (blockIdx.x >= q_group_segment_end_11) {
                group_count_1 = 10;
            }
            if (blockIdx.x >= q_group_segment_end_10) {
                group_count_1 = 9;
            }
            if (blockIdx.x >= q_group_segment_end_9) {
                group_count_1 = 8;
            }
            if (blockIdx.x >= q_group_segment_end_8) {
                group_count_1 = 7;
            }
            if (blockIdx.x >= q_group_segment_end_7) {
                group_count_1 = 6;
            }
            if (blockIdx.x >= q_group_segment_end_6) {
                group_count_1 = 5;
            }
            if (blockIdx.x >= q_group_segment_end_5) {
                group_count_1 = 4;
            }
            if (blockIdx.x >= q_group_segment_end_4) {
                group_count_1 = 3;
            }
            if (blockIdx.x >= q_group_segment_end_3) {
                group_count_1 = 2;
            }
            if (blockIdx.x >= q_group_segment_end_2) {
                group_count_1 = 1;
            }
            int group_count_0_1 = group_count_1;
            int work_idx_1 = blockIdx.x;
            int metadata_base_1 = work_idx_1 * 6;
            int head_kv_1 = scheduler_metadata[metadata_base_1];
            int row_linear_1 = scheduler_metadata[metadata_base_1 + 1];
            int q_begin_1 = scheduler_metadata[metadata_base_1 + 2];
            int q_count_1 = scheduler_metadata[metadata_base_1 + 3];
            int batch_1 = scheduler_metadata[metadata_base_1 + 4];
            int kv_block_1 = scheduler_metadata[metadata_base_1 + 5];
            int row_ptr_base_1 = head_kv_1 * (total_rows + 1) + row_linear_1;
            int row_start_1 = k2q_row_ptr[row_ptr_base_1] + q_begin_1;
            int q_batch_offset_1 = cu_seqlens_q[batch_1];
            int k_batch_offset_1 = cu_seqlens_k[batch_1];
            int kv_len_1 = kv_lens[batch_1];
            if (max_pages == 0) {
                kv_len_1 = cu_seqlens_k[batch_1 + 1] - k_batch_offset_1;
            }
            int query_offset_1 = q_offsets[batch_1];
            if (derive_q_offset != 0) {
                query_offset_1 = kv_len_1 - (cu_seqlens_q[batch_1 + 1] - q_batch_offset_1);
            }
            int stage_warp_1 = warp - 4;
            int my_row_1 = stage_warp_1 * 32 + lane;
            int tmem_row_base_1 = stage_warp_1 * 32 << 16;
            #pragma unroll 1
            for (int stage_iteration_1 = 0; stage_iteration_1 < group_count_0_1 / 2; stage_iteration_1++) {
                int group_1 = stage_iteration_1 * 2 + 1;
                int softmax_phase_1 = stage_iteration_1 & 1;
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
                    token_in_group_1 = my_row_1 / 4;
                    edge_in_work_1 = group_1 * 32 + token_in_group_1;
                    int row_valid_1 = ((edge_in_work_1 < q_count_1) ? 1 : 0);
                    int owner_lane_1 = lane / 4 * 4;
                    int owned_packed_1 = -1;
                    if (lane == owner_lane_1 && edge_in_work_1 < q_count_1) {
                        owned_packed_1 = k2q_qsplit_indices[head_kv_1 * nnz_per_head + row_start_1 + edge_in_work_1];
                    }
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, owned_packed_1, owner_lane_1);
                    packed_q_1 = _shfl_1;
                    q_idx_1 = packed_q_1 & 16777215;
                    if (row_valid_1 != 0) {
                        valid_cols_1 = kv_len_1 - kv_block_1 * 128;
                        if (valid_cols_1 > 128) {
                            valid_cols_1 = 128;
                        }
                        if (causal != 0) {
                            int query_position_1 = query_offset_1 + q_idx_1;
                            int causal_cols_1 = query_position_1 - kv_block_1 * 128 + 1;
                            if (valid_cols_1 > causal_cols_1) {
                                valid_cols_1 = causal_cols_1;
                            }
                        }
                        if (valid_cols_1 < 0) {
                            valid_cols_1 = 0;
                        }
                    }
                    float _tmem_load_6[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31]), "=f"(_tmem_load_6[32]), "=f"(_tmem_load_6[33]), "=f"(_tmem_load_6[34]), "=f"(_tmem_load_6[35]), "=f"(_tmem_load_6[36]), "=f"(_tmem_load_6[37]), "=f"(_tmem_load_6[38]), "=f"(_tmem_load_6[39]), "=f"(_tmem_load_6[40]), "=f"(_tmem_load_6[41]), "=f"(_tmem_load_6[42]), "=f"(_tmem_load_6[43]), "=f"(_tmem_load_6[44]), "=f"(_tmem_load_6[45]), "=f"(_tmem_load_6[46]), "=f"(_tmem_load_6[47]), "=f"(_tmem_load_6[48]), "=f"(_tmem_load_6[49]), "=f"(_tmem_load_6[50]), "=f"(_tmem_load_6[51]), "=f"(_tmem_load_6[52]), "=f"(_tmem_load_6[53]), "=f"(_tmem_load_6[54]), "=f"(_tmem_load_6[55]), "=f"(_tmem_load_6[56]), "=f"(_tmem_load_6[57]), "=f"(_tmem_load_6[58]), "=f"(_tmem_load_6[59]), "=f"(_tmem_load_6[60]), "=f"(_tmem_load_6[61]), "=f"(_tmem_load_6[62]), "=f"(_tmem_load_6[63])
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
                            if (!(_slice_lo_mask_8 & (1u << _i_1))) _tmem_load_6[0 + _i_1] = -BLACKWELL_MSA_INF;
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
                            if (!(_slice_lo_mask_9 & (1u << _i_3))) _tmem_load_6[32 + _i_3] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_4 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&_tmem_load_6[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_6[32], _reg_reduce_max2_4);
                    float _tmem_load_6_max = row_max_reduce(_reg_reduce_max2_4);
                    float body_max_1 = _tmem_load_6_max;
                    if (body_valid_1 <= 0) {
                        body_max_1 = -BLACKWELL_MSA_INF;
                    }
                    float _tmem_load_7[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31]), "=f"(_tmem_load_7[32]), "=f"(_tmem_load_7[33]), "=f"(_tmem_load_7[34]), "=f"(_tmem_load_7[35]), "=f"(_tmem_load_7[36]), "=f"(_tmem_load_7[37]), "=f"(_tmem_load_7[38]), "=f"(_tmem_load_7[39]), "=f"(_tmem_load_7[40]), "=f"(_tmem_load_7[41]), "=f"(_tmem_load_7[42]), "=f"(_tmem_load_7[43]), "=f"(_tmem_load_7[44]), "=f"(_tmem_load_7[45]), "=f"(_tmem_load_7[46]), "=f"(_tmem_load_7[47]), "=f"(_tmem_load_7[48]), "=f"(_tmem_load_7[49]), "=f"(_tmem_load_7[50]), "=f"(_tmem_load_7[51]), "=f"(_tmem_load_7[52]), "=f"(_tmem_load_7[53]), "=f"(_tmem_load_7[54]), "=f"(_tmem_load_7[55]), "=f"(_tmem_load_7[56]), "=f"(_tmem_load_7[57]), "=f"(_tmem_load_7[58]), "=f"(_tmem_load_7[59]), "=f"(_tmem_load_7[60]), "=f"(_tmem_load_7[61]), "=f"(_tmem_load_7[62]), "=f"(_tmem_load_7[63])
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
                            if (!(_slice_lo_mask_10 & (1u << _i_6))) _tmem_load_7[0 + _i_6] = -BLACKWELL_MSA_INF;
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
                            if (!(_slice_lo_mask_11 & (1u << _i_8))) _tmem_load_7[32 + _i_8] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_9 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&_tmem_load_7[0], _reg_reduce_max2_9);
                    row_max_x32_accum(&_tmem_load_7[32], _reg_reduce_max2_9);
                    float _tmem_load_7_max = row_max_reduce(_reg_reduce_max2_9);
                    float tail_max_1 = _tmem_load_7_max;
                    if (tail_valid_1 <= 0) {
                        tail_max_1 = -BLACKWELL_MSA_INF;
                    }
                    float _max_3 = max_noftz(body_max_1, tail_max_1);
                    row_max_1 = _max_3;
                    float safe_max_1 = ((row_max_1 == -BLACKWELL_MSA_INF) ? 0.0f : row_max_1);
                    score_bias_1 = ((valid_cols_1 > 0) ? (-safe_max_1) * softmax_scale_log2 : -BLACKWELL_MSA_INF);
                }
                mbarrier_wait(p_empty_addr + 8, softmax_phase_1 ^ 1);
                float row_sum_1 = 0.0f;
                if (whole_group_valid_1 != 0) {
                    int p_base_1 = taddr + 128 + 64 + (unsigned int)tmem_row_base_1;
                    float _tmem_load_8[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31]), "=f"(_tmem_load_8[32]), "=f"(_tmem_load_8[33]), "=f"(_tmem_load_8[34]), "=f"(_tmem_load_8[35]), "=f"(_tmem_load_8[36]), "=f"(_tmem_load_8[37]), "=f"(_tmem_load_8[38]), "=f"(_tmem_load_8[39]), "=f"(_tmem_load_8[40]), "=f"(_tmem_load_8[41]), "=f"(_tmem_load_8[42]), "=f"(_tmem_load_8[43]), "=f"(_tmem_load_8[44]), "=f"(_tmem_load_8[45]), "=f"(_tmem_load_8[46]), "=f"(_tmem_load_8[47]), "=f"(_tmem_load_8[48]), "=f"(_tmem_load_8[49]), "=f"(_tmem_load_8[50]), "=f"(_tmem_load_8[51]), "=f"(_tmem_load_8[52]), "=f"(_tmem_load_8[53]), "=f"(_tmem_load_8[54]), "=f"(_tmem_load_8[55]), "=f"(_tmem_load_8[56]), "=f"(_tmem_load_8[57]), "=f"(_tmem_load_8[58]), "=f"(_tmem_load_8[59]), "=f"(_tmem_load_8[60]), "=f"(_tmem_load_8[61]), "=f"(_tmem_load_8[62]), "=f"(_tmem_load_8[63])
                        : "r"(score_base_1)
                        : "memory");
                    if (body_valid_1 > 0 && body_valid_1 < 64) {
                        uint32_t _slice_lo_mask_12;
                        {
                            int _lim_10 = body_valid_1;
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
                            if (!(_slice_lo_mask_12 & (1u << _i_11))) _tmem_load_8[0 + _i_11] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_13;
                        {
                            int _lim_12 = body_valid_1 - 32;
                            if (_lim_12 <= 0) { _slice_lo_mask_13 = 0u; }
                            else if (_lim_12 >= 32) { _slice_lo_mask_13 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_13) : "r"(_lim_12));
                            }
                        }
                        #pragma unroll
                        for (int _i_13 = 0; _i_13 < 32; _i_13++) {
                            if (!(_slice_lo_mask_13 & (1u << _i_13))) _tmem_load_8[32 + _i_13] = -BLACKWELL_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_14 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_15 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_lf], _fma_b2_14, _fma_c2_15);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_8[_le] = approx_exp2(_tmem_load_8[_le]);
                    }
                    float2 _reg_reduce_sum2_16 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_8[0], &_reg_reduce_sum2_16);
                    softmax_block_sum(&_tmem_load_8[32], &_reg_reduce_sum2_16);
                    float _tmem_load_8_sum = _reg_reduce_sum2_16.x + _reg_reduce_sum2_16.y;
                    row_sum_1 = _tmem_load_8_sum;
                    uint32_t _tmem_load_8_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                        _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[31]))
                        : "memory");
                    float _tmem_load_9[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31]), "=f"(_tmem_load_9[32]), "=f"(_tmem_load_9[33]), "=f"(_tmem_load_9[34]), "=f"(_tmem_load_9[35]), "=f"(_tmem_load_9[36]), "=f"(_tmem_load_9[37]), "=f"(_tmem_load_9[38]), "=f"(_tmem_load_9[39]), "=f"(_tmem_load_9[40]), "=f"(_tmem_load_9[41]), "=f"(_tmem_load_9[42]), "=f"(_tmem_load_9[43]), "=f"(_tmem_load_9[44]), "=f"(_tmem_load_9[45]), "=f"(_tmem_load_9[46]), "=f"(_tmem_load_9[47]), "=f"(_tmem_load_9[48]), "=f"(_tmem_load_9[49]), "=f"(_tmem_load_9[50]), "=f"(_tmem_load_9[51]), "=f"(_tmem_load_9[52]), "=f"(_tmem_load_9[53]), "=f"(_tmem_load_9[54]), "=f"(_tmem_load_9[55]), "=f"(_tmem_load_9[56]), "=f"(_tmem_load_9[57]), "=f"(_tmem_load_9[58]), "=f"(_tmem_load_9[59]), "=f"(_tmem_load_9[60]), "=f"(_tmem_load_9[61]), "=f"(_tmem_load_9[62]), "=f"(_tmem_load_9[63])
                        : "r"(score_base_1 + 64)
                        : "memory");
                    if (valid_cols_1 > 0 && tail_valid_1 < 64) {
                        uint32_t _slice_lo_mask_14;
                        {
                            int _lim_17 = tail_valid_1;
                            if (_lim_17 <= 0) { _slice_lo_mask_14 = 0u; }
                            else if (_lim_17 >= 32) { _slice_lo_mask_14 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_14) : "r"(_lim_17));
                            }
                        }
                        #pragma unroll
                        for (int _i_18 = 0; _i_18 < 32; _i_18++) {
                            if (!(_slice_lo_mask_14 & (1u << _i_18))) _tmem_load_9[0 + _i_18] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_15;
                        {
                            int _lim_19 = tail_valid_1 - 32;
                            if (_lim_19 <= 0) { _slice_lo_mask_15 = 0u; }
                            else if (_lim_19 >= 32) { _slice_lo_mask_15 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_15) : "r"(_lim_19));
                            }
                        }
                        #pragma unroll
                        for (int _i_20 = 0; _i_20 < 32; _i_20++) {
                            if (!(_slice_lo_mask_15 & (1u << _i_20))) _tmem_load_9[32 + _i_20] = -BLACKWELL_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_21 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_22 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_lf], _fma_b2_21, _fma_c2_22);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_9[_le] = approx_exp2(_tmem_load_9[_le]);
                    }
                    float2 _reg_reduce_sum2_23 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_9[0], &_reg_reduce_sum2_23);
                    softmax_block_sum(&_tmem_load_9[32], &_reg_reduce_sum2_23);
                    float _tmem_load_9_sum = _reg_reduce_sum2_23.x + _reg_reduce_sum2_23.y;
                    row_sum_1 += _tmem_load_9_sum;
                    uint32_t _tmem_load_9_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                        _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_9_bf16[31]))
                        : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr + 8);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr + 8);
                mbarrier_wait(o_full_addr + 8, softmax_phase_1);
                if (whole_group_valid_1 != 0) {
                    int q_head_local_1 = my_row_1 - token_in_group_1 * 4;
                    int output_valid_1 = 0;
                    long long partial_row_1 = 0;
                    float inv_sum_1 = 0.0f;
                    if (edge_in_work_1 < q_count_1) {
                        int split_slot_1 = packed_q_1 >> 24 & 255;
                        if (split_slot_1 >= 0 && split_slot_1 < topk) {
                            output_valid_1 = 1;
                            int q_abs_1 = q_batch_offset_1 + q_idx_1;
                            int q_head_1 = head_kv_1 * 4 + q_head_local_1;
                            partial_row_1 = (long long)split_slot_1 * (long long)total_q * (long long)num_q_heads + (long long)q_abs_1 * (long long)num_q_heads + (long long)q_head_1;
                            float _rcp_1 = approx_rcp(row_sum_1);
                            inv_sum_1 = ((row_sum_1 > 0.0f && row_sum_1 == row_sum_1) ? _rcp_1 : 0.0f);
                        }
                    }
                    long long partial_base_1 = partial_row_1 * 128;
                    float row_abs_max_1 = 0.0f;
                    #pragma unroll 1
                    for (int output_segment_2 = 0; output_segment_2 < 8; output_segment_2++) {
                        float _tmem_load_10[16];
                        tmem_ld_x16(&_tmem_load_10[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + 128 + (unsigned int)tmem_row_base_1 + (unsigned int)(output_segment_2 * 16));
                        float _tmem_load_10_max = _tmem_load_10[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _tmem_load_10_max = max_noftz(_tmem_load_10_max, _tmem_load_10[_lr]);
                        }
                        float segment_max_1 = _tmem_load_10_max;
                        float _tmem_load_10_min = _tmem_load_10[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            _tmem_load_10_min = fminf(_tmem_load_10_min, _tmem_load_10[_lr]);
                        }
                        float segment_min_1 = _tmem_load_10_min;
                        float segment_neg_min_1 = -segment_min_1;
                        float _max_4 = max_noftz(segment_max_1, segment_neg_min_1);
                        float segment_abs_max_1 = _max_4;
                        float _max_5 = max_noftz(row_abs_max_1, segment_abs_max_1);
                        row_abs_max_1 = _max_5;
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float dequant_scale_1 = 0.0f;
                    float quant_scale_1 = 0.0f;
                    if (row_abs_max_1 > 0.0f && row_abs_max_1 == row_abs_max_1) {
                        dequant_scale_1 = row_abs_max_1 * inv_sum_1 * 0.002232142857142857f;
                        quant_scale_1 = 448.0f / row_abs_max_1;
                    }
                    if (output_valid_1 != 0) {
                        partial_scale[partial_row_1] = dequant_scale_1;
                    }
                    #pragma unroll 1
                    for (int output_segment_3 = 0; output_segment_3 < 8; output_segment_3++) {
                        float _tmem_load_11[16];
                        tmem_ld_x16(&_tmem_load_11[0], taddr + (unsigned int)TMEM_OUTPUT_OFFSET + 128 + (unsigned int)tmem_row_base_1 + (unsigned int)(output_segment_3 * 16));
                        if (output_valid_1 != 0) {
                            {
                                const float2 _prescale2_24 = {quant_scale_1, quant_scale_1};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_11[0])[_ps], _prescale2_24);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_11[0 + _ps] *= quant_scale_1;
                                #endif
                                unsigned int _fp8_pk[4];
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_11[0 + 1]), "f"(_tmem_load_11[0 + 0]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_11[0 + 3]), "f"(_tmem_load_11[0 + 2]));
                                    _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_11[0 + 5]), "f"(_tmem_load_11[0 + 4]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_11[0 + 7]), "f"(_tmem_load_11[0 + 6]));
                                    _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_11[0 + 9]), "f"(_tmem_load_11[0 + 8]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_11[0 + 11]), "f"(_tmem_load_11[0 + 10]));
                                    _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                { unsigned short _lo, _hi;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_11[0 + 13]), "f"(_tmem_load_11[0 + 12]));
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_11[0 + 15]), "f"(_tmem_load_11[0 + 14]));
                                    _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                }
                                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + (long long)output_segment_3 * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                            }
                        }
                    }
                    if (output_valid_1 != 0) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(row_sum_1));
                        partial_lse[partial_row_1] = ((row_sum_1 > 0.0f) ? row_max_1 * softmax_scale_log2 * 0.6931471805599453f + _log2_1 * 0.6931471805599453f : -BLACKWELL_MSA_INF);
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(o_empty_addr + 8);
            }
        }
    }
    // ---- Role: qload ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // qload_main
            int work_idx_2 = blockIdx.x;
            int metadata_base_2 = work_idx_2 * 6;
            int head_kv_2 = scheduler_metadata[metadata_base_2];
            int row_linear_2 = scheduler_metadata[metadata_base_2 + 1];
            int q_begin_2 = scheduler_metadata[metadata_base_2 + 2];
            int q_count_2 = scheduler_metadata[metadata_base_2 + 3];
            int batch_2 = scheduler_metadata[metadata_base_2 + 4];
            int kv_block_2 = scheduler_metadata[metadata_base_2 + 5];
            int row_ptr_base_2 = head_kv_2 * (total_rows + 1) + row_linear_2;
            int row_start_2 = k2q_row_ptr[row_ptr_base_2] + q_begin_2;
            int q_batch_offset_2 = cu_seqlens_q[batch_2];
            int k_batch_offset_2 = cu_seqlens_k[batch_2];
            int kv_len_2 = kv_lens[batch_2];
            if (max_pages == 0) {
                kv_len_2 = cu_seqlens_k[batch_2 + 1] - k_batch_offset_2;
            }
            int query_offset_2 = q_offsets[batch_2];
            if (derive_q_offset != 0) {
                query_offset_2 = kv_len_2 - (cu_seqlens_q[batch_2 + 1] - q_batch_offset_2);
            }
            int group_count_2 = 21;
            if (blockIdx.x >= q_group_segment_end_21) {
                group_count_2 = 20;
            }
            if (blockIdx.x >= q_group_segment_end_20) {
                group_count_2 = 19;
            }
            if (blockIdx.x >= q_group_segment_end_19) {
                group_count_2 = 18;
            }
            if (blockIdx.x >= q_group_segment_end_18) {
                group_count_2 = 17;
            }
            if (blockIdx.x >= q_group_segment_end_17) {
                group_count_2 = 16;
            }
            if (blockIdx.x >= q_group_segment_end_16) {
                group_count_2 = 15;
            }
            if (blockIdx.x >= q_group_segment_end_15) {
                group_count_2 = 14;
            }
            if (blockIdx.x >= q_group_segment_end_14) {
                group_count_2 = 13;
            }
            if (blockIdx.x >= q_group_segment_end_13) {
                group_count_2 = 12;
            }
            if (blockIdx.x >= q_group_segment_end_12) {
                group_count_2 = 11;
            }
            if (blockIdx.x >= q_group_segment_end_11) {
                group_count_2 = 10;
            }
            if (blockIdx.x >= q_group_segment_end_10) {
                group_count_2 = 9;
            }
            if (blockIdx.x >= q_group_segment_end_9) {
                group_count_2 = 8;
            }
            if (blockIdx.x >= q_group_segment_end_8) {
                group_count_2 = 7;
            }
            if (blockIdx.x >= q_group_segment_end_7) {
                group_count_2 = 6;
            }
            if (blockIdx.x >= q_group_segment_end_6) {
                group_count_2 = 5;
            }
            if (blockIdx.x >= q_group_segment_end_5) {
                group_count_2 = 4;
            }
            if (blockIdx.x >= q_group_segment_end_4) {
                group_count_2 = 3;
            }
            if (blockIdx.x >= q_group_segment_end_3) {
                group_count_2 = 2;
            }
            if (blockIdx.x >= q_group_segment_end_2) {
                group_count_2 = 1;
            }
            int group_count_0_2 = group_count_2;
            #pragma unroll 1
            for (int group_2 = 0; group_2 < group_count_0_2; group_2++) {
                int q_stage = group_2 & 1;
                int q_phase = group_2 / 2 & 1;
                mbarrier_wait(q_empty_addr + (q_stage) * 8, q_phase ^ 1);
                int q_stage_addr = q_store_smem_addr + (unsigned int)(q_stage * 32768);
                {
                    int qload_warp = warp - 8;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 8192);
                    }
                    if (elect_sync()) {
                        int token_in_group_2 = qload_warp * 8;
                        int edge_in_work_2 = group_2 * 32 + token_in_group_2;
                        int edge_valid = ((edge_in_work_2 < q_count_2) ? 1 : 0);
                        int safe_edge = ((edge_valid != 0) ? edge_in_work_2 : 0);
                        int packed_q_2 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge];
                        int decoded_q_abs = q_batch_offset_2 + (packed_q_2 & 16777215);
                        int q_abs_2 = ((edge_valid != 0) ? decoded_q_abs : 0);
                        int row_base = q_abs_2 * num_q_heads + head_kv_2 * 4;
                        int dst_offset = token_in_group_2 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset, (&q), 0, row_base, row_base + 1, row_base + 2, row_base + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset, (&q), 64, row_base, row_base + 1, row_base + 2, row_base + 3, q_full_addr + (q_stage) * 8);
                        int token_in_group_0 = qload_warp * 8 + 1;
                        int edge_in_work_1_1 = group_2 * 32 + token_in_group_0;
                        int edge_valid_2 = ((edge_in_work_1_1 < q_count_2) ? 1 : 0);
                        int safe_edge_3 = ((edge_valid_2 != 0) ? edge_in_work_1_1 : 0);
                        int packed_q_4 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_3];
                        int decoded_q_abs_5 = q_batch_offset_2 + (packed_q_4 & 16777215);
                        int q_abs_6 = ((edge_valid_2 != 0) ? decoded_q_abs_5 : 0);
                        int row_base_7 = q_abs_6 * num_q_heads + head_kv_2 * 4;
                        int dst_offset_8 = token_in_group_0 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset_8, (&q), 0, row_base_7, row_base_7 + 1, row_base_7 + 2, row_base_7 + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset_8, (&q), 64, row_base_7, row_base_7 + 1, row_base_7 + 2, row_base_7 + 3, q_full_addr + (q_stage) * 8);
                        int token_in_group_9 = qload_warp * 8 + 2;
                        int edge_in_work_10 = group_2 * 32 + token_in_group_9;
                        int edge_valid_11 = ((edge_in_work_10 < q_count_2) ? 1 : 0);
                        int safe_edge_12 = ((edge_valid_11 != 0) ? edge_in_work_10 : 0);
                        int packed_q_13 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_12];
                        int decoded_q_abs_14 = q_batch_offset_2 + (packed_q_13 & 16777215);
                        int q_abs_15 = ((edge_valid_11 != 0) ? decoded_q_abs_14 : 0);
                        int row_base_16 = q_abs_15 * num_q_heads + head_kv_2 * 4;
                        int dst_offset_17 = token_in_group_9 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset_17, (&q), 0, row_base_16, row_base_16 + 1, row_base_16 + 2, row_base_16 + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset_17, (&q), 64, row_base_16, row_base_16 + 1, row_base_16 + 2, row_base_16 + 3, q_full_addr + (q_stage) * 8);
                        int token_in_group_18 = qload_warp * 8 + 3;
                        int edge_in_work_19 = group_2 * 32 + token_in_group_18;
                        int edge_valid_20 = ((edge_in_work_19 < q_count_2) ? 1 : 0);
                        int safe_edge_21 = ((edge_valid_20 != 0) ? edge_in_work_19 : 0);
                        int packed_q_22 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_21];
                        int decoded_q_abs_23 = q_batch_offset_2 + (packed_q_22 & 16777215);
                        int q_abs_24 = ((edge_valid_20 != 0) ? decoded_q_abs_23 : 0);
                        int row_base_25 = q_abs_24 * num_q_heads + head_kv_2 * 4;
                        int dst_offset_26 = token_in_group_18 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset_26, (&q), 0, row_base_25, row_base_25 + 1, row_base_25 + 2, row_base_25 + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset_26, (&q), 64, row_base_25, row_base_25 + 1, row_base_25 + 2, row_base_25 + 3, q_full_addr + (q_stage) * 8);
                        int token_in_group_27 = qload_warp * 8 + 4;
                        int edge_in_work_28 = group_2 * 32 + token_in_group_27;
                        int edge_valid_29 = ((edge_in_work_28 < q_count_2) ? 1 : 0);
                        int safe_edge_30 = ((edge_valid_29 != 0) ? edge_in_work_28 : 0);
                        int packed_q_31 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_30];
                        int decoded_q_abs_32 = q_batch_offset_2 + (packed_q_31 & 16777215);
                        int q_abs_33 = ((edge_valid_29 != 0) ? decoded_q_abs_32 : 0);
                        int row_base_34 = q_abs_33 * num_q_heads + head_kv_2 * 4;
                        int dst_offset_35 = token_in_group_27 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset_35, (&q), 0, row_base_34, row_base_34 + 1, row_base_34 + 2, row_base_34 + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset_35, (&q), 64, row_base_34, row_base_34 + 1, row_base_34 + 2, row_base_34 + 3, q_full_addr + (q_stage) * 8);
                        int token_in_group_36 = qload_warp * 8 + 5;
                        int edge_in_work_37 = group_2 * 32 + token_in_group_36;
                        int edge_valid_38 = ((edge_in_work_37 < q_count_2) ? 1 : 0);
                        int safe_edge_39 = ((edge_valid_38 != 0) ? edge_in_work_37 : 0);
                        int packed_q_40 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_39];
                        int decoded_q_abs_41 = q_batch_offset_2 + (packed_q_40 & 16777215);
                        int q_abs_42 = ((edge_valid_38 != 0) ? decoded_q_abs_41 : 0);
                        int row_base_43 = q_abs_42 * num_q_heads + head_kv_2 * 4;
                        int dst_offset_44 = token_in_group_36 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset_44, (&q), 0, row_base_43, row_base_43 + 1, row_base_43 + 2, row_base_43 + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset_44, (&q), 64, row_base_43, row_base_43 + 1, row_base_43 + 2, row_base_43 + 3, q_full_addr + (q_stage) * 8);
                        int token_in_group_45 = qload_warp * 8 + 6;
                        int edge_in_work_46 = group_2 * 32 + token_in_group_45;
                        int edge_valid_47 = ((edge_in_work_46 < q_count_2) ? 1 : 0);
                        int safe_edge_48 = ((edge_valid_47 != 0) ? edge_in_work_46 : 0);
                        int packed_q_49 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_48];
                        int decoded_q_abs_50 = q_batch_offset_2 + (packed_q_49 & 16777215);
                        int q_abs_51 = ((edge_valid_47 != 0) ? decoded_q_abs_50 : 0);
                        int row_base_52 = q_abs_51 * num_q_heads + head_kv_2 * 4;
                        int dst_offset_53 = token_in_group_45 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset_53, (&q), 0, row_base_52, row_base_52 + 1, row_base_52 + 2, row_base_52 + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset_53, (&q), 64, row_base_52, row_base_52 + 1, row_base_52 + 2, row_base_52 + 3, q_full_addr + (q_stage) * 8);
                        int token_in_group_54 = qload_warp * 8 + 7;
                        int edge_in_work_55 = group_2 * 32 + token_in_group_54;
                        int edge_valid_56 = ((edge_in_work_55 < q_count_2) ? 1 : 0);
                        int safe_edge_57 = ((edge_valid_56 != 0) ? edge_in_work_55 : 0);
                        int packed_q_58 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_57];
                        int decoded_q_abs_59 = q_batch_offset_2 + (packed_q_58 & 16777215);
                        int q_abs_60 = ((edge_valid_56 != 0) ? decoded_q_abs_59 : 0);
                        int row_base_61 = q_abs_60 * num_q_heads + head_kv_2 * 4;
                        int dst_offset_62 = token_in_group_54 * 512;
                        tma_gather4_gmem2smem(q_stage_addr + dst_offset_62, (&q), 0, row_base_61, row_base_61 + 1, row_base_61 + 2, row_base_61 + 3, q_full_addr + (q_stage) * 8);
                        tma_gather4_gmem2smem(q_stage_addr + 16384 + dst_offset_62, (&q), 64, row_base_61, row_base_61 + 1, row_base_61 + 2, row_base_61 + 3, q_full_addr + (q_stage) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int work_idx_3 = blockIdx.x;
            int metadata_base_3 = work_idx_3 * 6;
            int head_kv_3 = scheduler_metadata[metadata_base_3];
            int row_linear_3 = scheduler_metadata[metadata_base_3 + 1];
            int q_begin_3 = scheduler_metadata[metadata_base_3 + 2];
            int q_count_3 = scheduler_metadata[metadata_base_3 + 3];
            int batch_3 = scheduler_metadata[metadata_base_3 + 4];
            int kv_block_3 = scheduler_metadata[metadata_base_3 + 5];
            int row_ptr_base_3 = head_kv_3 * (total_rows + 1) + row_linear_3;
            int row_start_3 = k2q_row_ptr[row_ptr_base_3] + q_begin_3;
            int q_batch_offset_3 = cu_seqlens_q[batch_3];
            int k_batch_offset_3 = cu_seqlens_k[batch_3];
            int kv_len_3 = kv_lens[batch_3];
            if (max_pages == 0) {
                kv_len_3 = cu_seqlens_k[batch_3 + 1] - k_batch_offset_3;
            }
            int query_offset_3 = q_offsets[batch_3];
            if (derive_q_offset != 0) {
                query_offset_3 = kv_len_3 - (cu_seqlens_q[batch_3 + 1] - q_batch_offset_3);
            }
            int group_count_3 = 21;
            if (blockIdx.x >= q_group_segment_end_21) {
                group_count_3 = 20;
            }
            if (blockIdx.x >= q_group_segment_end_20) {
                group_count_3 = 19;
            }
            if (blockIdx.x >= q_group_segment_end_19) {
                group_count_3 = 18;
            }
            if (blockIdx.x >= q_group_segment_end_18) {
                group_count_3 = 17;
            }
            if (blockIdx.x >= q_group_segment_end_17) {
                group_count_3 = 16;
            }
            if (blockIdx.x >= q_group_segment_end_16) {
                group_count_3 = 15;
            }
            if (blockIdx.x >= q_group_segment_end_15) {
                group_count_3 = 14;
            }
            if (blockIdx.x >= q_group_segment_end_14) {
                group_count_3 = 13;
            }
            if (blockIdx.x >= q_group_segment_end_13) {
                group_count_3 = 12;
            }
            if (blockIdx.x >= q_group_segment_end_12) {
                group_count_3 = 11;
            }
            if (blockIdx.x >= q_group_segment_end_11) {
                group_count_3 = 10;
            }
            if (blockIdx.x >= q_group_segment_end_10) {
                group_count_3 = 9;
            }
            if (blockIdx.x >= q_group_segment_end_9) {
                group_count_3 = 8;
            }
            if (blockIdx.x >= q_group_segment_end_8) {
                group_count_3 = 7;
            }
            if (blockIdx.x >= q_group_segment_end_7) {
                group_count_3 = 6;
            }
            if (blockIdx.x >= q_group_segment_end_6) {
                group_count_3 = 5;
            }
            if (blockIdx.x >= q_group_segment_end_5) {
                group_count_3 = 4;
            }
            if (blockIdx.x >= q_group_segment_end_4) {
                group_count_3 = 3;
            }
            if (blockIdx.x >= q_group_segment_end_3) {
                group_count_3 = 2;
            }
            if (blockIdx.x >= q_group_segment_end_2) {
                group_count_3 = 1;
            }
            int group_count_0_3 = group_count_3;
            unsigned int _phase_k_full_0 = 0;
            mbarrier_wait(k_full_addr, _phase_k_full_0);
            _phase_k_full_0 ^= 1;
            int q_stage_1 = 0;
            int q_phase_1 = 0;
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
            if (group_count_0_3 > 1) {
                int q_stage_0 = 1;
                int q_phase_1_1 = 0;
                mbarrier_wait(q_full_addr + (q_stage_0) * 8, q_phase_1_1);
                mbarrier_wait(s_empty_addr + (q_stage_0) * 8, q_phase_1_1 ^ 1);
                int _mma_a_lo_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_0) * 2048);
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (q_stage_0 * 128))), "r"(0));
                elect_commit(s_full_addr + (q_stage_0) * 8);
                elect_commit(q_empty_addr + (q_stage_0) * 8);
            }
            unsigned int _phase_v_full_0 = 0;
            mbarrier_wait(v_full_addr, _phase_v_full_0);
            _phase_v_full_0 ^= 1;
            #pragma unroll 1
            for (int group_3 = 2; group_3 < group_count_0_3; group_3++) {
                int pv_group = group_3 - 2;
                int pv_stage = pv_group & 1;
                int pv_phase = pv_group / 2 & 1;
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(_mma_b_lo_2), "r"(tmem_scores + (pv_stage * 128 + 64)), "r"(0));
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(_mma_b_lo_2), "r"(tmem_scores + (pv_stage * 128 + 64)), "r"(1));
                elect_commit(o_full_addr + (pv_stage) * 8);
                elect_commit(p_empty_addr + (pv_stage) * 8);
                int q_stage_0_1 = group_3 & 1;
                int q_phase_1_2 = group_3 / 2 & 1;
                mbarrier_wait(q_full_addr + (q_stage_0_1) * 8, q_phase_1_2);
                mbarrier_wait(s_empty_addr + (q_stage_0_1) * 8, q_phase_1_2 ^ 1);
                int _mma_a_lo_4 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_0_1) * 2048);
                int _mma_b_lo_4 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_scores + (q_stage_0_1 * 128))), "r"(0));
                elect_commit(s_full_addr + (q_stage_0_1) * 8);
                elect_commit(q_empty_addr + (q_stage_0_1) * 8);
            }
            int drain_start = ((group_count_0_3 == 1) ? 0 : group_count_0_3 - 2);
            #pragma unroll 1
            for (int pv_group_1 = drain_start; pv_group_1 < group_count_0_3; pv_group_1++) {
                int pv_stage_1 = pv_group_1 & 1;
                int pv_phase_1 = pv_group_1 / 2 & 1;
                mbarrier_wait(p_full_addr + (pv_stage_1) * 8, pv_phase_1);
                mbarrier_wait(o_empty_addr + (pv_stage_1) * 8, pv_phase_1 ^ 1);
                int _mma_b_lo_5 = make_warp_uniform((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000);
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
                    "mov.b32 id, 136381584;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(_mma_b_lo_5), "r"(tmem_scores + (pv_stage_1 * 128 + 64)), "r"(0));
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(_mma_b_lo_5), "r"(tmem_scores + (pv_stage_1 * 128 + 64)), "r"(1));
                elect_commit(o_full_addr + (pv_stage_1) * 8);
                elect_commit(p_empty_addr + (pv_stage_1) * 8);
            }
            #pragma unroll 1
            for (int completed_group = drain_start; completed_group < group_count_0_3; completed_group++) {
                int completed_stage = completed_group & 1;
                int completed_phase = completed_group / 2 & 1;
                mbarrier_wait(o_empty_addr + (completed_stage) * 8, completed_phase);
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: transform ----
    if (warp >= 13 && warp <= 14) {
        { // transform_main
            int work_idx_4 = blockIdx.x;
            int metadata_base_4 = work_idx_4 * 6;
            int head_kv_4 = scheduler_metadata[metadata_base_4];
            int row_linear_4 = scheduler_metadata[metadata_base_4 + 1];
            int q_begin_4 = scheduler_metadata[metadata_base_4 + 2];
            int q_count_4 = scheduler_metadata[metadata_base_4 + 3];
            int batch_4 = scheduler_metadata[metadata_base_4 + 4];
            int kv_block_4 = scheduler_metadata[metadata_base_4 + 5];
            int row_ptr_base_4 = head_kv_4 * (total_rows + 1) + row_linear_4;
            int row_start_4 = k2q_row_ptr[row_ptr_base_4] + q_begin_4;
            int q_batch_offset_4 = cu_seqlens_q[batch_4];
            int k_batch_offset_4 = cu_seqlens_k[batch_4];
            int kv_len_4 = kv_lens[batch_4];
            if (max_pages == 0) {
                kv_len_4 = cu_seqlens_k[batch_4 + 1] - k_batch_offset_4;
            }
            int query_offset_4 = q_offsets[batch_4];
            if (derive_q_offset != 0) {
                query_offset_4 = kv_len_4 - (cu_seqlens_q[batch_4 + 1] - q_batch_offset_4);
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            int work_idx_5 = blockIdx.x;
            int metadata_base_5 = work_idx_5 * 6;
            int head_kv_5 = scheduler_metadata[metadata_base_5];
            int row_linear_5 = scheduler_metadata[metadata_base_5 + 1];
            int q_begin_5 = scheduler_metadata[metadata_base_5 + 2];
            int q_count_5 = scheduler_metadata[metadata_base_5 + 3];
            int batch_5 = scheduler_metadata[metadata_base_5 + 4];
            int kv_block_5 = scheduler_metadata[metadata_base_5 + 5];
            int row_ptr_base_5 = head_kv_5 * (total_rows + 1) + row_linear_5;
            int row_start_5 = k2q_row_ptr[row_ptr_base_5] + q_begin_5;
            int q_batch_offset_5 = cu_seqlens_q[batch_5];
            int k_batch_offset_5 = cu_seqlens_k[batch_5];
            int kv_len_5 = kv_lens[batch_5];
            if (max_pages == 0) {
                kv_len_5 = cu_seqlens_k[batch_5 + 1] - k_batch_offset_5;
            }
            int query_offset_5 = q_offsets[batch_5];
            if (derive_q_offset != 0) {
                query_offset_5 = kv_len_5 - (cu_seqlens_q[batch_5 + 1] - q_batch_offset_5);
            }
            int token_base = k_batch_offset_5 + kv_block_5 * 128;
            int page_head = head_kv_5;
            {
                int physical_page = page_table[batch_5 * max_pages + kv_block_5];
                if (physical_page < 0) {
                    physical_page = 0;
                }
                token_base = 0;
                page_head = physical_page * num_kv_heads + head_kv_5;
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr, 32768);
                int token0 = token_base;
                int token1 = token_base + 64;
                {
                    token0 = 0;
                    token1 = 64;
                }
                tma_4d_gmem2smem(k_smem_addr, (&k), 0, token0, 0, page_head, k_full_addr);
                tma_4d_gmem2smem(k_smem_addr + 8192, (&k), 0, token1, 0, page_head, k_full_addr);
                tma_4d_gmem2smem(k_smem_addr + 16384, (&k), 0, token0, 1, page_head, k_full_addr);
                tma_4d_gmem2smem(k_smem_addr + 24576, (&k), 0, token1, 1, page_head, k_full_addr);
                mbarrier_arrive_expect_tx(v_full_addr, 32768);
                int token0_0 = token_base;
                int token1_1 = token_base + 64;
                {
                    token0_0 = 0;
                    token1_1 = 64;
                }
                tma_4d_gmem2smem(v_smem_addr, (&v), 0, token0_0, 0, page_head, v_full_addr);
                tma_4d_gmem2smem(v_smem_addr + 8192, (&v), 0, token1_1, 0, page_head, v_full_addr);
                tma_4d_gmem2smem(v_smem_addr + 16384, (&v), 0, token0_0, 1, page_head, v_full_addr);
                tma_4d_gmem2smem(v_smem_addr + 24576, (&v), 0, token1_1, 1, page_head, v_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
