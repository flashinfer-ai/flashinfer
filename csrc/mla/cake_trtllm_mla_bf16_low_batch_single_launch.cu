typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_KV_PIPE_STAGES 9
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 74752
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 74752
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_STATS_MAX_OFF 222208
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_MAX_STRIDE 1024
#define SMEM_SMEM_STATS_SUM_OFF 223232
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 512
#define SMEM_SMEM_STATS_SUM_STRIDE 512
#define SMEM_SMEM_PAGETABLE_OFF 223744
#define SMEM_SMEM_PAGETABLE_STAGE_BYTES 4096
#define SMEM_SMEM_PAGETABLE_STRIDE 4096
#define SMEM_TOTAL 227840
#define THREADS 384

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


__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void mma_ts_step_cg2(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], db, %4, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
}


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


__device__ __forceinline__ void tmem_st_x16(int tmem_addr, uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "r"(src[0]),  "r"(src[1]),  "r"(src[2]),  "r"(src[3]),
           "r"(src[4]),  "r"(src[5]),  "r"(src[6]),  "r"(src[7]),
           "r"(src[8]),  "r"(src[9]),  "r"(src[10]), "r"(src[11]),
           "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]));
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


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
}


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}

extern "C" {

__global__ __launch_bounds__(384) __cluster_dims__(2,1,1) void
kernel_cake_trtllm_mla_bf16_low_batch_single_launch(CakeTensorMap const* tmap_q, CakeTensorMap const* tmap_kv, __nv_bfloat16* __restrict__ O, int* __restrict__ page_table, int* __restrict__ seq_lens, int q_len, int source_table_width, int seqlen_kv, float softmax_scale_log2, int total_work_items)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_kv)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_kv_addr = smem + 74752;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_v_addr = smem + 74752;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 222208);
    const int smem_stats_max_addr = smem + 222208;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 223232);
    const int smem_stats_sum_addr = smem + 223232;
    int* smem_pagetable = reinterpret_cast<int*>(smem_raw + 223744);
    const int smem_pagetable_addr = smem + 223744;

    // Mbarrier init (17 groups, 46 barriers)
    // Mbarriers at smem_raw[0..368)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 9 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // kv_empty: 9 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            // s_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 248, 256);
            // q_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 256, 64);
            // kv_pair_ready: 9 barriers, init_count=64
            mbarrier_init(smem + 264, 64);
            mbarrier_init(smem + 272, 64);
            mbarrier_init(smem + 280, 64);
            mbarrier_init(smem + 288, 64);
            mbarrier_init(smem + 296, 64);
            mbarrier_init(smem + 304, 64);
            mbarrier_init(smem + 312, 64);
            mbarrier_init(smem + 320, 64);
            mbarrier_init(smem + 328, 64);
            // pv_pair_ready: 2 barriers, init_count=64
            mbarrier_init(smem + 336, 64);
            mbarrier_init(smem + 344, 64);
            // tmem_dealloc: 1 barriers, init_count=320
            mbarrier_init(smem + 352, 320);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 360, 32);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 368);
    if (warp == 0) {
        int _tmem_hold = smem + 368;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 160)
    #define p_full_addr (mbar_base + 176)
    #define corr_done_addr (mbar_base + 192)
    #define stats_addr (mbar_base + 208)
    #define sum_ready_addr (mbar_base + 224)
    #define o_done_addr (mbar_base + 232)
    #define pv_done_addr (mbar_base + 240)
    #define s_seeded_addr (mbar_base + 248)
    #define q_pair_ready_addr (mbar_base + 256)
    #define kv_pair_ready_addr (mbar_base + 264)
    #define pv_pair_ready_addr (mbar_base + 336)
    #define tmem_dealloc_addr (mbar_base + 352)
    #define tmem_dealloc_peer_addr (mbar_base + 360)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_wg_main
            const int wg_dummy_inc = 0;
            int num_kv_tiles = seqlen_kv / 128;
            const int tmem_row_base = warp % 4 * 32;
            const int my_row = tmem_row_base + lane;
            unsigned int _phase_q_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work_items; work_idx += num_clusters) {
                float seed_zero[4];
                #pragma unroll
                for (int seed_c4 = 0; seed_c4 < 4; seed_c4++) {
                    seed_zero[seed_c4] = 0.0f;
                }
                #pragma unroll
                for (int seed_half = 0; seed_half < 2; seed_half++) {
                    #pragma unroll
                    for (int seed_c = 0; seed_c < 16; seed_c++) {
                        int seed_addr = taddr + (unsigned int)(((seed_half == 0) ? 64 : 192)) + (unsigned int)(seed_c * 4) + (unsigned int)(tmem_row_base << 16);
                        tmem_st_x4_f32(seed_addr, seed_zero);
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_seeded_addr);
                int seed_peer_rank = cta_rank ^ 1;
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(s_seeded_addr), "r"(seed_peer_rank) : "memory");
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                float row_max_val = -CAKE_INF;
                float row_sum_val = 0.0f;
                #pragma unroll 1
                for (int tile = 0; tile < num_kv_tiles; tile++) {
                    int phase = tile & 1;
                    int s_wait_phase = tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (phase) * 8, s_wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_off = ((phase != 0) ? 128 : 0);
                    int s_base = taddr + (unsigned int)s_off + (unsigned int)(tmem_row_base << 16);
                    float _tmem_load_0[128];
                    tmem_ld_x32(&_tmem_load_0[0], s_base);
                    tmem_ld_x32(&_tmem_load_0[32], s_base + 32);
                    tmem_ld_x32(&_tmem_load_0[64], s_base + 64);
                    tmem_ld_x32(&_tmem_load_0[96], s_base + 96);
                    int query_row = work_idx >> 1;
                    int source_batch = query_row / q_len;
                    int query_in_batch = query_row - source_batch * q_len;
                    int causal_len = seq_lens[source_batch] - q_len + query_in_batch;
                    #pragma unroll
                    for (int i = 0; i < 128; i++) {
                        int token = tile * 128 + i;
                        if (token >= causal_len) {
                            _tmem_load_0[i] = -CAKE_INF;
                        }
                    }
                    float new_max = -CAKE_INF;
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 128; i_1++) {
                        float _max_0 = max_noftz(new_max, _tmem_load_0[i_1]);
                        new_max = _max_0;
                    }
                    float _max_1 = max_noftz(new_max, row_max_val);
                    new_max = _max_1;
                    float delta = softmax_scale_log2 * (row_max_val - new_max);
                    float _exp2_0 = approx_exp2(delta);
                    float exp_delta = _exp2_0;
                    float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta : 1.0f);
                    smem_stats_max[phase * 128 + my_row] = acc_scale;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(stats_addr + (phase) * 8);
                    row_max_val = new_max;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    #pragma unroll
                    for (int i_2 = 0; i_2 < 128; i_2++) {
                        float _exp2_1 = approx_exp2(_tmem_load_0[i_2] * softmax_scale_log2 - max_scaled);
                        _tmem_load_0[i_2] = _exp2_1;
                    }
                    row_sum_val = row_sum_val * acc_scale;
                    #pragma unroll
                    for (int i_3 = 0; i_3 < 128; i_3++) {
                        row_sum_val = row_sum_val + _tmem_load_0[i_3];
                    }
                    int p_off = ((phase != 0) ? 192 : 64);
                    int p_base = taddr + (unsigned int)p_off + (unsigned int)(tmem_row_base << 16);
                    {
                        uint32_t _pv_packed[16];
                        #pragma unroll
                        for (int _j = 0; _j < 16; _j++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[0 + _j * 2], _tmem_load_0[0 + _j * 2 + 1]));
                            _pv_packed[_j] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x16(p_base, _pv_packed);
                    }
                    {
                        uint32_t _pv_packed[16];
                        #pragma unroll
                        for (int _j = 0; _j < 16; _j++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[32 + _j * 2], _tmem_load_0[32 + _j * 2 + 1]));
                            _pv_packed[_j] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x16(p_base + 16, _pv_packed);
                    }
                    {
                        uint32_t _pv_packed[16];
                        #pragma unroll
                        for (int _j = 0; _j < 16; _j++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[64 + _j * 2], _tmem_load_0[64 + _j * 2 + 1]));
                            _pv_packed[_j] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x16(p_base + 32, _pv_packed);
                    }
                    {
                        uint32_t _pv_packed[16];
                        #pragma unroll
                        for (int _j = 0; _j < 16; _j++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[96 + _j * 2], _tmem_load_0[96 + _j * 2 + 1]));
                            _pv_packed[_j] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x16(p_base + 48, _pv_packed);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (phase) * 8);
                }
                smem_stats_sum[my_row] = row_sum_val;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(sum_ready_addr);
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_wg_main
            const int wg_dummy_inc_1 = 0;
            int num_kv_tiles_1 = seqlen_kv / 128;
            const int tmem_row_base_1 = warp % 4 * 32;
            const int my_row_1 = tmem_row_base_1 + lane;
            const int corr_row = tmem_row_base_1 << 16;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_pv_done_0 = 0;
            unsigned int _phase_o_done_0 = 0;
            unsigned int _phase_sum_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = cluster_id; work_idx_1 < total_work_items; work_idx_1 += num_clusters) {
                int batch_idx = work_idx_1 >> 1;
                int v_chunk = work_idx_1 & 1;
                mbarrier_wait(q_full_addr, _phase_q_full_0_1);
                _phase_q_full_0_1 ^= 1;
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                    int phase_1 = tile_1 & 1;
                    int stats_wait_phase = tile_1 >> 1 & 1;
                    mbarrier_wait(stats_addr + (phase_1) * 8, stats_wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc_scale_1 = smem_stats_max[phase_1 * 128 + my_row_1];
                    if (tile_1 > 0) {
                        mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                        _phase_pv_done_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    if (tile_1 > 0) {
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        int any_rescale = _vote_0;
                        if (any_rescale != 0) {
                            #pragma unroll
                            for (int vs = 0; vs < 4; vs++) {
                                int o_base = taddr + 256 + (unsigned int)(vs * 64) + (unsigned int)corr_row;
                                #pragma unroll
                                for (int c = 0; c < 64; c += 16) {
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], o_base + c);
                                    #pragma unroll
                                    for (int j = 0; j < 16; j++) {
                                        _tmem_load_1[j] = _tmem_load_1[j] * acc_scale_1;
                                    }
                                    tmem_st_x16_f32(o_base + c, _tmem_load_1);
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                }
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(corr_done_addr + (phase_1) * 8);
                }
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
                _phase_sum_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float total_sum = smem_stats_sum[my_row_1];
                float _rcp_0 = approx_rcp(total_sum);
                float inv_sum = _rcp_0;
                int head_idx = my_row_1;
                int o_offset = (batch_idx * 128 + head_idx) * 512 + v_chunk * 256;
                #pragma unroll
                for (int vs_1 = 0; vs_1 < 4; vs_1++) {
                    int o_base_epi = taddr + 256 + (unsigned int)(vs_1 * 64) + (unsigned int)corr_row;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 64; c_1 += 32) {
                        float _tmem_load_2[32];
                        tmem_ld_x32(&_tmem_load_2[0], o_base_epi + c_1);
                        int gmem_base = o_offset + vs_1 * 64 + c_1;
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 32; j_1 += 8) {
                            {
                                const float2 _prescale2_0 = {inv_sum, inv_sum};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 4; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[j_1])[_ps], _prescale2_0);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    _tmem_load_2[j_1 + _ps] *= inv_sum;
                                #endif
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_2[j_1 + 0], _tmem_load_2[j_1 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_2[j_1 + 2], _tmem_load_2[j_1 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_2[j_1 + 4], _tmem_load_2[j_1 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_2[j_1 + 6], _tmem_load_2[j_1 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (gmem_base + j_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
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
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg2_dummy = 0;
            int num_kv_tiles_2 = seqlen_kv / 128;
            unsigned int mma_kv_stage = 0;
            unsigned int _phase_s_seeded_0 = 0;
            unsigned int _phase_q_full_0_2 = 0;
            unsigned int _phase_q_pair_ready_0 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_kv_pair_ready = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = cluster_id; work_idx_2 < total_work_items; work_idx_2 += num_clusters) {
                mbarrier_wait_cluster(s_seeded_addr, _phase_s_seeded_0);
                _phase_s_seeded_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(q_full_addr, _phase_q_full_0_2);
                _phase_q_full_0_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(q_pair_ready_addr), "r"(0) : "memory");
                if (cta_rank == 0) {
                    mbarrier_wait(q_pair_ready_addr, _phase_q_pair_ready_0);
                    _phase_q_pair_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                int first_pv = 1;
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                    int phase_2 = tile_2 & 1;
                    int score_col = ((phase_2 != 0) ? 128 : 0);
                    #pragma unroll 1
                    for (int n = 0; n < 4; n++) {
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(kv_pair_ready_addr + mma_kv_stage * 8), "r"(0) : "memory");
                        if (cta_rank == 0) {
                            mbarrier_wait(kv_pair_ready_addr + (mma_kv_stage) * 8, _phase_kv_pair_ready);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (n * 2) * 512;
                            int _mma_b_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (score_col))), "r"(((n == 0) ? 0 : 1)));
                            int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (n * 2 + 1) * 512;
                            int _mma_b_lo_1 = (((smem_kv_addr + 8192) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                        }
                        if (cta_rank == 0) {
                            elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                        }
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 9) { mma_kv_stage = 0; _phase_kv_full ^= 1; _phase_kv_pair_ready ^= 1; }
                    }
                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(kv_pair_ready_addr + mma_kv_stage * 8), "r"(0) : "memory");
                    if (cta_rank == 0) {
                        mbarrier_wait(kv_pair_ready_addr + (mma_kv_stage) * 8, _phase_kv_pair_ready);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (8) * 512;
                        int _mma_b_lo_2 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                    }
                    if (cta_rank == 0) {
                        elect_commit_cg2_multicast(s_full_addr + (phase_2) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                    }
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 9) { mma_kv_stage = 0; _phase_kv_full ^= 1; _phase_kv_pair_ready ^= 1; }
                    if (tile_2 > 0) {
                        int prev_phase = tile_2 - 1 & 1;
                        int prev_tile = tile_2 - 1;
                        int pv_wait_phase = prev_tile >> 1 & 1;
                        mbarrier_wait(p_full_addr + (prev_phase) * 8, pv_wait_phase);
                        mbarrier_wait(corr_done_addr + (prev_phase) * 8, pv_wait_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(pv_pair_ready_addr + (unsigned int)(prev_phase * 8)), "r"(0) : "memory");
                        if (cta_rank == 0) {
                            mbarrier_wait(pv_pair_ready_addr + (prev_phase) * 8, pv_wait_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        int p_col = ((prev_phase != 0) ? 192 : 64);
                        #pragma unroll
                        for (int vs_2 = 0; vs_2 < 4; vs_2++) {
                            mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(kv_pair_ready_addr + mma_kv_stage * 8), "r"(0) : "memory");
                            int output_col = 256 + vs_2 * 64;
                            if (cta_rank == 0) {
                                mbarrier_wait(kv_pair_ready_addr + (mma_kv_stage) * 8, _phase_kv_pair_ready);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_b_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 8], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 16], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 24], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 32], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 40], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 48], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 56], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (output_col))), "r"(_mma_b_lo_3), "r"(tmem_tmem_scratch + p_col), "r"(((first_pv) ? 0 : 1)));
                            }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                            }
                            mma_kv_stage += 1;
                            if (mma_kv_stage == 9) { mma_kv_stage = 0; _phase_kv_full ^= 1; _phase_kv_pair_ready ^= 1; }
                        }
                        first_pv = 0;
                        if (cta_rank == 0) {
                            elect_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                        }
                    }
                }
                int last_phase = num_kv_tiles_2 - 1 & 1;
                int last_tile = num_kv_tiles_2 - 1;
                int drain_wait_phase = last_tile >> 1 & 1;
                mbarrier_wait(p_full_addr + (last_phase) * 8, drain_wait_phase);
                mbarrier_wait(corr_done_addr + (last_phase) * 8, drain_wait_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(pv_pair_ready_addr + (unsigned int)(last_phase * 8)), "r"(0) : "memory");
                if (cta_rank == 0) {
                    mbarrier_wait(pv_pair_ready_addr + (last_phase) * 8, drain_wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                int p_col_last = ((last_phase != 0) ? 192 : 64);
                #pragma unroll
                for (int vs_3 = 0; vs_3 < 4; vs_3++) {
                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(kv_pair_ready_addr + mma_kv_stage * 8), "r"(0) : "memory");
                    int output_col_d = 256 + vs_3 * 64;
                    if (cta_rank == 0) {
                        mbarrier_wait(kv_pair_ready_addr + (mma_kv_stage) * 8, _phase_kv_pair_ready);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_kv_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 8], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 16], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 24], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 32], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 40], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 48], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [%2 + 56], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (output_col_d))), "r"(_mma_b_lo_4), "r"(tmem_tmem_scratch + p_col_last), "r"(((first_pv) ? 0 : 1)));
                    }
                    if (cta_rank == 0) {
                        elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                    }
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 9) { mma_kv_stage = 0; _phase_kv_full ^= 1; _phase_kv_pair_ready ^= 1; }
                }
                if (cta_rank == 0) {
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_done_addr, (uint16_t)(3));
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int tmem_dealloc_peer_rank = cta_rank ^ 1;
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
    // ---- Role: load_warp ----
    if (warp == 9) {
        { // load_warp_main
            const int wg2_dummy_1 = 0;
            int num_kv_tiles_3 = seqlen_kv / 128;
            unsigned int load_kv_stage = 0;
            int k_g_offset = cta_rank * 16;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int work_idx_3 = cluster_id; work_idx_3 < total_work_items; work_idx_3 += num_clusters) {
                int query_row_1 = work_idx_3 >> 1;
                int v_chunk_1 = work_idx_3 & 1;
                int source_batch_1 = query_row_1 / q_len;
                int query_in_batch_1 = query_row_1 - source_batch_1 * q_len;
                int causal_len_1 = seq_lens[source_batch_1] - q_len + query_in_batch_1;
                int q_row_global = query_row_1 * 128 + cta_rank * 64;
                int num_pt_passes = seqlen_kv / 32;
                #pragma unroll 1
                for (int pp = 0; pp < num_pt_passes; pp++) {
                    int pt_off = pp * 32 + lane;
                    int safe_token = ((pt_off < causal_len_1) ? pt_off : causal_len_1 - 1);
                    int logical_page = safe_token / 32;
                    int token_in_page = safe_token - logical_page * 32;
                    int physical_page = page_table[source_batch_1 * source_table_width + logical_page];
                    smem_pagetable[pt_off] = physical_page * 32 + token_in_page;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 32;" ::: "memory");
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 73728);
                    #pragma unroll
                    for (int s = 0; s < 9; s++) {
                        int q_dst = smem_q_addr + (unsigned int)(s * 8192);
                        tma_3d_gmem2smem(q_dst, tmap_q, 0, q_row_global, s, q_full_addr);
                    }
                }
                int token_base_0 = k_g_offset * 4;
                int k_lane = lane & 15;
                int local_off_k0 = token_base_0 + k_lane * 4;
                int k0_i0 = smem_pagetable[local_off_k0];
                int k0_i1 = smem_pagetable[local_off_k0 + 1];
                int k0_i2 = smem_pagetable[local_off_k0 + 2];
                int k0_i3 = smem_pagetable[local_off_k0 + 3];
                #pragma unroll
                for (int n_1 = 0; n_1 < 4; n_1++) {
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                    }
                    int dst = smem_kv_addr + load_kv_stage * 16384;
                    int col0 = n_1 * 128;
                    int col1 = n_1 * 128 + 64;
                    #pragma unroll
                    for (int g = 0; g < 16; g++) {
                        int _shfl_0 = __shfl_sync(0xFFFFFFFF, k0_i0, g);
                        int r0 = _shfl_0;
                        int _shfl_1 = __shfl_sync(0xFFFFFFFF, k0_i1, g);
                        int r1 = _shfl_1;
                        int _shfl_2 = __shfl_sync(0xFFFFFFFF, k0_i2, g);
                        int r2 = _shfl_2;
                        int _shfl_3 = __shfl_sync(0xFFFFFFFF, k0_i3, g);
                        int r3 = _shfl_3;
                        if (elect_sync()) {
                            tma_gather4_gmem2smem(dst + g * 512, tmap_kv, col0, r0, r1, r2, r3, kv_full_addr + (load_kv_stage) * 8);
                            tma_gather4_gmem2smem(dst + 8192 + g * 512, tmap_kv, col1, r0, r1, r2, r3, kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 9) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                }
                mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 8192);
                }
                int dst_r = smem_kv_addr + load_kv_stage * 16384;
                #pragma unroll
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    int _shfl_4 = __shfl_sync(0xFFFFFFFF, k0_i0, g_1);
                    int r0_1 = _shfl_4;
                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, k0_i1, g_1);
                    int r1_1 = _shfl_5;
                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, k0_i2, g_1);
                    int r2_1 = _shfl_6;
                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, k0_i3, g_1);
                    int r3_1 = _shfl_7;
                    if (elect_sync()) {
                        tma_gather4_gmem2smem(dst_r + g_1 * 512, tmap_kv, 512, r0_1, r1_1, r2_1, r3_1, kv_full_addr + (load_kv_stage) * 8);
                    }
                }
                load_kv_stage += 1;
                if (load_kv_stage == 9) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                #pragma unroll 1
                for (int tile_3 = 1; tile_3 < num_kv_tiles_3; tile_3++) {
                    int k_token_base = tile_3 * 128 + token_base_0;
                    int local_off_k = k_token_base + k_lane * 4;
                    int k_i0 = smem_pagetable[local_off_k];
                    int k_i1 = smem_pagetable[local_off_k + 1];
                    int k_i2 = smem_pagetable[local_off_k + 2];
                    int k_i3 = smem_pagetable[local_off_k + 3];
                    #pragma unroll
                    for (int n_2 = 0; n_2 < 4; n_2++) {
                        mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                        }
                        int dst_k = smem_kv_addr + load_kv_stage * 16384;
                        int col0_1 = n_2 * 128;
                        int col1_1 = n_2 * 128 + 64;
                        #pragma unroll
                        for (int g_2 = 0; g_2 < 16; g_2++) {
                            int _shfl_8 = __shfl_sync(0xFFFFFFFF, k_i0, g_2);
                            int r0_2 = _shfl_8;
                            int _shfl_9 = __shfl_sync(0xFFFFFFFF, k_i1, g_2);
                            int r1_2 = _shfl_9;
                            int _shfl_10 = __shfl_sync(0xFFFFFFFF, k_i2, g_2);
                            int r2_2 = _shfl_10;
                            int _shfl_11 = __shfl_sync(0xFFFFFFFF, k_i3, g_2);
                            int r3_2 = _shfl_11;
                            if (elect_sync()) {
                                tma_gather4_gmem2smem(dst_k + g_2 * 512, tmap_kv, col0_1, r0_2, r1_2, r2_2, r3_2, kv_full_addr + (load_kv_stage) * 8);
                                tma_gather4_gmem2smem(dst_k + 8192 + g_2 * 512, tmap_kv, col1_1, r0_2, r1_2, r2_2, r3_2, kv_full_addr + (load_kv_stage) * 8);
                            }
                        }
                        load_kv_stage += 1;
                        if (load_kv_stage == 9) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 8192);
                    }
                    int dst_kr = smem_kv_addr + load_kv_stage * 16384;
                    #pragma unroll
                    for (int g_3 = 0; g_3 < 16; g_3++) {
                        int _shfl_12 = __shfl_sync(0xFFFFFFFF, k_i0, g_3);
                        int r0_3 = _shfl_12;
                        int _shfl_13 = __shfl_sync(0xFFFFFFFF, k_i1, g_3);
                        int r1_3 = _shfl_13;
                        int _shfl_14 = __shfl_sync(0xFFFFFFFF, k_i2, g_3);
                        int r2_3 = _shfl_14;
                        int _shfl_15 = __shfl_sync(0xFFFFFFFF, k_i3, g_3);
                        int r3_3 = _shfl_15;
                        if (elect_sync()) {
                            tma_gather4_gmem2smem(dst_kr + g_3 * 512, tmap_kv, 512, r0_3, r1_3, r2_3, r3_3, kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 9) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    int v_token_base = (tile_3 - 1) * 128;
                    int local_off_v = v_token_base + lane * 4;
                    int v_i0 = smem_pagetable[local_off_v];
                    int v_i1 = smem_pagetable[local_off_v + 1];
                    int v_i2 = smem_pagetable[local_off_v + 2];
                    int v_i3 = smem_pagetable[local_off_v + 3];
                    #pragma unroll
                    for (int vs_4 = 0; vs_4 < 4; vs_4++) {
                        int v_col = v_chunk_1 * 256 + vs_4 * 64;
                        mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                        }
                        int dst_v = smem_kv_addr + load_kv_stage * 16384;
                        #pragma unroll
                        for (int g_4 = 0; g_4 < 32; g_4++) {
                            int _shfl_16 = __shfl_sync(0xFFFFFFFF, v_i0, g_4);
                            int r0_4 = _shfl_16;
                            int _shfl_17 = __shfl_sync(0xFFFFFFFF, v_i1, g_4);
                            int r1_4 = _shfl_17;
                            int _shfl_18 = __shfl_sync(0xFFFFFFFF, v_i2, g_4);
                            int r2_4 = _shfl_18;
                            int _shfl_19 = __shfl_sync(0xFFFFFFFF, v_i3, g_4);
                            int r3_4 = _shfl_19;
                            if (elect_sync()) {
                                tma_gather4_gmem2smem(dst_v + g_4 * 512, tmap_kv, v_col, r0_4, r1_4, r2_4, r3_4, kv_full_addr + (load_kv_stage) * 8);
                            }
                        }
                        load_kv_stage += 1;
                        if (load_kv_stage == 9) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
                int vl_token_base = (num_kv_tiles_3 - 1) * 128;
                int local_off_vl = vl_token_base + lane * 4;
                int vl_i0 = smem_pagetable[local_off_vl];
                int vl_i1 = smem_pagetable[local_off_vl + 1];
                int vl_i2 = smem_pagetable[local_off_vl + 2];
                int vl_i3 = smem_pagetable[local_off_vl + 3];
                #pragma unroll
                for (int vs_5 = 0; vs_5 < 4; vs_5++) {
                    int v_col_last = v_chunk_1 * 256 + vs_5 * 64;
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                    }
                    int dst_vl = smem_kv_addr + load_kv_stage * 16384;
                    #pragma unroll
                    for (int g_5 = 0; g_5 < 32; g_5++) {
                        int _shfl_20 = __shfl_sync(0xFFFFFFFF, vl_i0, g_5);
                        int r0_5 = _shfl_20;
                        int _shfl_21 = __shfl_sync(0xFFFFFFFF, vl_i1, g_5);
                        int r1_5 = _shfl_21;
                        int _shfl_22 = __shfl_sync(0xFFFFFFFF, vl_i2, g_5);
                        int r2_5 = _shfl_22;
                        int _shfl_23 = __shfl_sync(0xFFFFFFFF, vl_i3, g_5);
                        int r3_5 = _shfl_23;
                        if (elect_sync()) {
                            tma_gather4_gmem2smem(dst_vl + g_5 * 512, tmap_kv, v_col_last, r0_5, r1_5, r2_5, r3_5, kv_full_addr + (load_kv_stage) * 8);
                        }
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 9) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: empty0 ----
    if (warp == 10) {
        { // empty0_main
            const int wg2_dummy_2 = 0;
            unsigned int _phase_q_full_0_3 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_4 = cluster_id; work_idx_4 < total_work_items; work_idx_4 += num_clusters) {
                mbarrier_wait(q_full_addr, _phase_q_full_0_3);
                _phase_q_full_0_3 ^= 1;
            }
        }
    }
    // ---- Role: empty1 ----
    if (warp == 11) {
        { // empty1_main
            const int wg2_dummy_3 = 0;
            unsigned int _phase_q_full_0_4 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_5 = cluster_id; work_idx_5 < total_work_items; work_idx_5 += num_clusters) {
                mbarrier_wait(q_full_addr, _phase_q_full_0_4);
                _phase_q_full_0_4 ^= 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"
