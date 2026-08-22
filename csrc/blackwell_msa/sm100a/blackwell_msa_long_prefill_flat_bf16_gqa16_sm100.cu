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
kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_nobar_sm100(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, int* __restrict__ scheduler_metadata, int* __restrict__ k2q_row_ptr, int* __restrict__ k2q_qsplit_indices, uint8_t* __restrict__ partial_o, float* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, __nv_bfloat16* __restrict__ out, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int q_group_segment_end_128, int q_group_segment_end_64, int q_group_segment_end_32, int q_group_segment_end_16, int q_group_segment_end_8, int q_group_segment_end_4, int q_group_segment_end_2, int total_q, int num_q_heads, int num_kv_heads, int total_rows, int nnz_per_head, int work_capacity, int num_work_items, int topk, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_temperature_lse)
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

    // Mbarrier init (14 groups, 23 barriers)
    // Mbarriers at smem_raw[0..184)

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
            // p_full_2: 2 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            mbarrier_init(smem + 128, 128);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // --- pipeline 'o_pipe' ---
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184);
    if (warp == 0) {
        int _tmem_hold = smem + 184;
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
    #define p_full_2_addr (mbar_base + 120)
    #define p_empty_addr (mbar_base + 136)
    #define o_full_addr (mbar_base + 152)
    #define o_empty_addr (mbar_base + 168)
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
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_even_main
            int group_count = 1;
            {
                if (blockIdx.x < q_group_segment_end_16) {
                    if (blockIdx.x < q_group_segment_end_64) {
                        group_count = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                    } else {
                        group_count = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                    }
                } else if (blockIdx.x < q_group_segment_end_4) {
                    group_count = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
                } else {
                    group_count = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
                }
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
                float row_max = -BLACKWELL_MSA_INF;
                float score_bias = -BLACKWELL_MSA_INF;
                float score_values[128];
                int score_base = taddr + (unsigned int)tmem_row_base;
                if (whole_group_valid != 0) {
                    token_in_group = my_row / 16;
                    edge_in_work = group * 8 + token_in_group;
                    int row_valid = ((edge_in_work < q_count) ? 1 : 0);
                    int owner_lane = lane / 16 * 16;
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
                    tmem_ld_x32(&score_values[0], score_base);
                    tmem_ld_x32(&score_values[32], score_base + 32);
                    tmem_ld_x32(&score_values[64], score_base + 64);
                    tmem_ld_x32(&score_values[96], score_base + 96);
                    if (valid_cols < 128) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = valid_cols;
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
                            if (!(_slice_lo_mask_0 & (1u << _i_1))) score_values[0 + _i_1] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_2 = valid_cols - 32;
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
                            if (!(_slice_lo_mask_1 & (1u << _i_3))) score_values[32 + _i_3] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_4 = valid_cols - 64;
                            if (_lim_4 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_4 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_4));
                            }
                        }
                        #pragma unroll
                        for (int _i_5 = 0; _i_5 < 32; _i_5++) {
                            if (!(_slice_lo_mask_2 & (1u << _i_5))) score_values[64 + _i_5] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_6 = valid_cols - 96;
                            if (_lim_6 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_6 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_6));
                            }
                        }
                        #pragma unroll
                        for (int _i_7 = 0; _i_7 < 32; _i_7++) {
                            if (!(_slice_lo_mask_3 & (1u << _i_7))) score_values[96 + _i_7] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_8 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&score_values[0], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values[32], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values[64], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values[96], _reg_reduce_max2_8);
                    float score_values_max = row_max_reduce(_reg_reduce_max2_8);
                    row_max = score_values_max;
                    float safe_max = ((row_max == -BLACKWELL_MSA_INF) ? 0.0f : row_max);
                    score_bias = ((valid_cols > 0) ? (-safe_max) * softmax_scale_log2 : -BLACKWELL_MSA_INF);
                }
                mbarrier_wait(p_empty_addr, softmax_phase ^ 1);
                float row_sum = 0.0f;
                if (whole_group_valid != 0) {
                    int p_base = taddr + 64 + (unsigned int)tmem_row_base;
                    const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_10 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(score_values)[_lf], _fma_b2_9, _fma_c2_10);
                    for (int probability_segment = 0; probability_segment < 2; probability_segment++) {
                        uint32_t score_values_bf16[16];
                        softmax_frag_exp2_cast(&(score_values + probability_segment * 32)[0], score_values_bf16, 0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base + probability_segment * 16), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16[15]))
                            : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr);
                    uint32_t score_values_bf16_1[32];
                    softmax_frag_exp2_cast(&score_values[64], score_values_bf16_1, 1);
                    softmax_frag_exp2_cast(&score_values[96], &score_values_bf16_1[16], 1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_1[31]))
                        : "memory");
                    float2 _reg_reduce_sum2_11 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values[0], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values[32], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values[64], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values[96], &_reg_reduce_sum2_11);
                    float score_values_sum = _reg_reduce_sum2_11.x + _reg_reduce_sum2_11.y;
                    row_sum = score_values_sum;
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_2_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr);
                mbarrier_wait(o_full_addr, softmax_phase);
                if (whole_group_valid != 0) {
                    int q_head_local = my_row - token_in_group * 16;
                    int output_valid = 0;
                    long long partial_row = 0;
                    long long final_row = 0;
                    float inv_sum = 0.0f;
                    int single_split = 0;
                    if (edge_in_work < q_count) {
                        int split_slot = packed_q >> 24 & 15;
                        if (split_slot >= 0 && split_slot < topk) {
                            output_valid = 1;
                            int q_abs = q_batch_offset + q_idx;
                            int q_head = head_kv * 16 + q_head_local;
                            final_row = (long long)q_abs * (long long)num_q_heads + (long long)q_head;
                            partial_row = (long long)split_slot * (long long)total_q * (long long)num_q_heads + (long long)q_abs * (long long)num_q_heads + (long long)q_head;
                            float _rcp_0 = approx_rcp(row_sum);
                            inv_sum = ((row_sum > 0.0f && row_sum == row_sum) ? _rcp_0 : 0.0f);
                        }
                    }
                    long long partial_base = partial_row * 128;
                    {
                        int output_row_addr = taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base;
                        long long partial_metadata_rows = (long long)topk * (long long)total_q * (long long)num_q_heads;
                        {
                            float row_min = BLACKWELL_MSA_INF;
                            float row_max_0 = -BLACKWELL_MSA_INF;
                            #pragma unroll 1
                            for (int output_segment = 0; output_segment < 4; output_segment++) {
                                float _tmem_load_2[16];
                                tmem_ld_x16(&_tmem_load_2[0], output_row_addr + output_segment * 16);
                                float _tmem_load_2_min = _tmem_load_2[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_2_min = fminf(_tmem_load_2_min, _tmem_load_2[_lr]);
                                }
                                float _min_0 = fminf(row_min, _tmem_load_2_min);
                                row_min = _min_0;
                                float _tmem_load_2_max = _tmem_load_2[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_2_max = max_noftz(_tmem_load_2_max, _tmem_load_2[_lr]);
                                }
                                float _max_1 = max_noftz(row_max_0, _tmem_load_2_max);
                                row_max_0 = _max_1;
                            }
                            float _tmem_load_3[64];
                            tmem_ld_x32(&_tmem_load_3[0], output_row_addr + 64);
                            tmem_ld_x32(&_tmem_load_3[32], output_row_addr + 64 + 32);
                            float _tmem_load_3_min = _tmem_load_3[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 64; _lr++) {
                                _tmem_load_3_min = fminf(_tmem_load_3_min, _tmem_load_3[_lr]);
                            }
                            float _min_1 = fminf(row_min, _tmem_load_3_min);
                            row_min = _min_1;
                            float2 _reg_reduce_max2_12 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                            row_max_x32_accum(&_tmem_load_3[0], _reg_reduce_max2_12);
                            row_max_x32_accum(&_tmem_load_3[32], _reg_reduce_max2_12);
                            float _tmem_load_3_max = row_max_reduce(_reg_reduce_max2_12);
                            float _max_2 = max_noftz(row_max_0, _tmem_load_3_max);
                            row_max_0 = _max_2;
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float row_center = (row_max_0 + row_min) * 0.5f;
                            float residual_abs_max = (row_max_0 - row_min) * 0.5f;
                            float dequant_scale = 0.0f;
                            float quant_scale = 0.0f;
                            if (residual_abs_max > 0.0f && residual_abs_max == residual_abs_max) {
                                dequant_scale = residual_abs_max * inv_sum * 0.002232142857142857f;
                                quant_scale = 448.0f / residual_abs_max;
                            }
                            if (output_valid != 0) {
                                partial_scale[partial_row] = dequant_scale;
                                partial_scale[partial_metadata_rows + partial_row] = row_center * inv_sum;
                            }
                            #pragma unroll 1
                            for (int output_segment_1 = 0; output_segment_1 < 4; output_segment_1++) {
                                float _tmem_load_4[16];
                                tmem_ld_x16(&_tmem_load_4[0], output_row_addr + output_segment_1 * 16);
                                const float2 _sub2_13 = {row_center, row_center};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _sub2_13);
                                if (output_valid != 0) {
                                    {
                                        const float2 _prescale2_14 = {quant_scale, quant_scale};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_14);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 16; _ps++)
                                            _tmem_load_4[0 + _ps] *= quant_scale;
                                        #endif
                                        unsigned int _fp8_pk[4];
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 1]), "f"(_tmem_load_4[0 + 0]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 3]), "f"(_tmem_load_4[0 + 2]));
                                            _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 5]), "f"(_tmem_load_4[0 + 4]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 7]), "f"(_tmem_load_4[0 + 6]));
                                            _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 9]), "f"(_tmem_load_4[0 + 8]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 11]), "f"(_tmem_load_4[0 + 10]));
                                            _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_4[0 + 13]), "f"(_tmem_load_4[0 + 12]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_4[0 + 15]), "f"(_tmem_load_4[0 + 14]));
                                            _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + (long long)output_segment_1 * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                    }
                                }
                            }
                            const float2 _sub2_15 = {row_center, row_center};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _sub2_15);
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_16 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[0])[_ps], _prescale2_16);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_3[0 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[0 + 1]), "f"(_tmem_load_3[0 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[0 + 3]), "f"(_tmem_load_3[0 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[0 + 5]), "f"(_tmem_load_3[0 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[0 + 7]), "f"(_tmem_load_3[0 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[0 + 9]), "f"(_tmem_load_3[0 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[0 + 11]), "f"(_tmem_load_3[0 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[0 + 13]), "f"(_tmem_load_3[0 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[0 + 15]), "f"(_tmem_load_3[0 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 64)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_17 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[16])[_ps], _prescale2_17);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_3[16 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[16 + 1]), "f"(_tmem_load_3[16 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[16 + 3]), "f"(_tmem_load_3[16 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[16 + 5]), "f"(_tmem_load_3[16 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[16 + 7]), "f"(_tmem_load_3[16 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[16 + 9]), "f"(_tmem_load_3[16 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[16 + 11]), "f"(_tmem_load_3[16 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[16 + 13]), "f"(_tmem_load_3[16 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[16 + 15]), "f"(_tmem_load_3[16 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 80)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_18 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[32])[_ps], _prescale2_18);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_3[32 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[32 + 1]), "f"(_tmem_load_3[32 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[32 + 3]), "f"(_tmem_load_3[32 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[32 + 5]), "f"(_tmem_load_3[32 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[32 + 7]), "f"(_tmem_load_3[32 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[32 + 9]), "f"(_tmem_load_3[32 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[32 + 11]), "f"(_tmem_load_3[32 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[32 + 13]), "f"(_tmem_load_3[32 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[32 + 15]), "f"(_tmem_load_3[32 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 96)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_19 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[48])[_ps], _prescale2_19);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_3[48 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[48 + 1]), "f"(_tmem_load_3[48 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[48 + 3]), "f"(_tmem_load_3[48 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[48 + 5]), "f"(_tmem_load_3[48 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[48 + 7]), "f"(_tmem_load_3[48 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[48 + 9]), "f"(_tmem_load_3[48 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[48 + 11]), "f"(_tmem_load_3[48 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_3[48 + 13]), "f"(_tmem_load_3[48 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_3[48 + 15]), "f"(_tmem_load_3[48 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 112)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
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
        }
    }
    // ---- Role: softmax_odd ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_odd_main
            int group_count_1 = 1;
            {
                if (blockIdx.x < q_group_segment_end_16) {
                    if (blockIdx.x < q_group_segment_end_64) {
                        group_count_1 = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                    } else {
                        group_count_1 = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                    }
                } else if (blockIdx.x < q_group_segment_end_4) {
                    group_count_1 = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
                } else {
                    group_count_1 = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
                }
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
                float row_max_1 = -BLACKWELL_MSA_INF;
                float score_bias_1 = -BLACKWELL_MSA_INF;
                float score_values_1[128];
                int score_base_1 = taddr + 128 + (unsigned int)tmem_row_base_1;
                if (whole_group_valid_1 != 0) {
                    token_in_group_1 = my_row_1 / 16;
                    edge_in_work_1 = group_1 * 8 + token_in_group_1;
                    int row_valid_1 = ((edge_in_work_1 < q_count_1) ? 1 : 0);
                    int owner_lane_1 = lane / 16 * 16;
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
                    tmem_ld_x32(&score_values_1[0], score_base_1);
                    tmem_ld_x32(&score_values_1[32], score_base_1 + 32);
                    tmem_ld_x32(&score_values_1[64], score_base_1 + 64);
                    tmem_ld_x32(&score_values_1[96], score_base_1 + 96);
                    if (valid_cols_1 < 128) {
                        uint32_t _slice_lo_mask_4;
                        {
                            int _lim_0 = valid_cols_1;
                            if (_lim_0 <= 0) { _slice_lo_mask_4 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_0));
                            }
                        }
                        #pragma unroll
                        for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                            if (!(_slice_lo_mask_4 & (1u << _i_1))) score_values_1[0 + _i_1] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_5;
                        {
                            int _lim_2 = valid_cols_1 - 32;
                            if (_lim_2 <= 0) { _slice_lo_mask_5 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_2));
                            }
                        }
                        #pragma unroll
                        for (int _i_3 = 0; _i_3 < 32; _i_3++) {
                            if (!(_slice_lo_mask_5 & (1u << _i_3))) score_values_1[32 + _i_3] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_6;
                        {
                            int _lim_4 = valid_cols_1 - 64;
                            if (_lim_4 <= 0) { _slice_lo_mask_6 = 0u; }
                            else if (_lim_4 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_4));
                            }
                        }
                        #pragma unroll
                        for (int _i_5 = 0; _i_5 < 32; _i_5++) {
                            if (!(_slice_lo_mask_6 & (1u << _i_5))) score_values_1[64 + _i_5] = -BLACKWELL_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_7;
                        {
                            int _lim_6 = valid_cols_1 - 96;
                            if (_lim_6 <= 0) { _slice_lo_mask_7 = 0u; }
                            else if (_lim_6 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_6));
                            }
                        }
                        #pragma unroll
                        for (int _i_7 = 0; _i_7 < 32; _i_7++) {
                            if (!(_slice_lo_mask_7 & (1u << _i_7))) score_values_1[96 + _i_7] = -BLACKWELL_MSA_INF;
                        }
                    }
                    float2 _reg_reduce_max2_8 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                    row_max_x32_accum(&score_values_1[0], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values_1[32], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values_1[64], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values_1[96], _reg_reduce_max2_8);
                    float score_values_max_1 = row_max_reduce(_reg_reduce_max2_8);
                    row_max_1 = score_values_max_1;
                    float safe_max_1 = ((row_max_1 == -BLACKWELL_MSA_INF) ? 0.0f : row_max_1);
                    score_bias_1 = ((valid_cols_1 > 0) ? (-safe_max_1) * softmax_scale_log2 : -BLACKWELL_MSA_INF);
                }
                mbarrier_wait(p_empty_addr + 8, softmax_phase_1 ^ 1);
                float row_sum_1 = 0.0f;
                if (whole_group_valid_1 != 0) {
                    int p_base_1 = taddr + 128 + 64 + (unsigned int)tmem_row_base_1;
                    const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_10 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(score_values_1)[_lf], _fma_b2_9, _fma_c2_10);
                    for (int probability_segment_1 = 0; probability_segment_1 < 2; probability_segment_1++) {
                        uint32_t score_values_bf16_2[16];
                        softmax_frag_exp2_cast(&(score_values_1 + probability_segment_1 * 32)[0], score_values_bf16_2, 0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base_1 + probability_segment_1 * 16), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_2[15]))
                            : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + 8);
                    uint32_t score_values_bf16_3[32];
                    softmax_frag_exp2_cast(&score_values_1[64], score_values_bf16_3, 1);
                    softmax_frag_exp2_cast(&score_values_1[96], &score_values_bf16_3[16], 1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[15])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[16])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[17])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[18])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[19])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[20])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[21])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[22])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[23])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[24])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[25])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[26])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[27])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[28])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[29])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[30])), "r"(*reinterpret_cast<const uint32_t*>(&score_values_bf16_3[31]))
                        : "memory");
                    float2 _reg_reduce_sum2_11 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values_1[0], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values_1[32], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values_1[64], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values_1[96], &_reg_reduce_sum2_11);
                    float score_values_sum_1 = _reg_reduce_sum2_11.x + _reg_reduce_sum2_11.y;
                    row_sum_1 = score_values_sum_1;
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_2_addr + 8);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr + 8);
                mbarrier_wait(o_full_addr + 8, softmax_phase_1);
                if (whole_group_valid_1 != 0) {
                    int q_head_local_1 = my_row_1 - token_in_group_1 * 16;
                    int output_valid_1 = 0;
                    long long partial_row_1 = 0;
                    long long final_row_1 = 0;
                    float inv_sum_1 = 0.0f;
                    int single_split_1 = 0;
                    if (edge_in_work_1 < q_count_1) {
                        int split_slot_1 = packed_q_1 >> 24 & 15;
                        if (split_slot_1 >= 0 && split_slot_1 < topk) {
                            output_valid_1 = 1;
                            int q_abs_1 = q_batch_offset_1 + q_idx_1;
                            int q_head_1 = head_kv_1 * 16 + q_head_local_1;
                            final_row_1 = (long long)q_abs_1 * (long long)num_q_heads + (long long)q_head_1;
                            partial_row_1 = (long long)split_slot_1 * (long long)total_q * (long long)num_q_heads + (long long)q_abs_1 * (long long)num_q_heads + (long long)q_head_1;
                            float _rcp_1 = approx_rcp(row_sum_1);
                            inv_sum_1 = ((row_sum_1 > 0.0f && row_sum_1 == row_sum_1) ? _rcp_1 : 0.0f);
                        }
                    }
                    long long partial_base_1 = partial_row_1 * 128;
                    {
                        int output_row_addr_1 = taddr + (unsigned int)TMEM_OUTPUT_OFFSET + 128 + (unsigned int)tmem_row_base_1;
                        long long partial_metadata_rows_1 = (long long)topk * (long long)total_q * (long long)num_q_heads;
                        {
                            float row_min_1 = BLACKWELL_MSA_INF;
                            float row_max_0_1 = -BLACKWELL_MSA_INF;
                            #pragma unroll 1
                            for (int output_segment_2 = 0; output_segment_2 < 4; output_segment_2++) {
                                float _tmem_load_8[16];
                                tmem_ld_x16(&_tmem_load_8[0], output_row_addr_1 + output_segment_2 * 16);
                                float _tmem_load_8_min = _tmem_load_8[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_8_min = fminf(_tmem_load_8_min, _tmem_load_8[_lr]);
                                }
                                float _min_2 = fminf(row_min_1, _tmem_load_8_min);
                                row_min_1 = _min_2;
                                float _tmem_load_8_max = _tmem_load_8[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_8_max = max_noftz(_tmem_load_8_max, _tmem_load_8[_lr]);
                                }
                                float _max_4 = max_noftz(row_max_0_1, _tmem_load_8_max);
                                row_max_0_1 = _max_4;
                            }
                            float _tmem_load_9[64];
                            tmem_ld_x32(&_tmem_load_9[0], output_row_addr_1 + 64);
                            tmem_ld_x32(&_tmem_load_9[32], output_row_addr_1 + 64 + 32);
                            float _tmem_load_9_min = _tmem_load_9[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 64; _lr++) {
                                _tmem_load_9_min = fminf(_tmem_load_9_min, _tmem_load_9[_lr]);
                            }
                            float _min_3 = fminf(row_min_1, _tmem_load_9_min);
                            row_min_1 = _min_3;
                            float2 _reg_reduce_max2_12 = {-BLACKWELL_MSA_INF, -BLACKWELL_MSA_INF};
                            row_max_x32_accum(&_tmem_load_9[0], _reg_reduce_max2_12);
                            row_max_x32_accum(&_tmem_load_9[32], _reg_reduce_max2_12);
                            float _tmem_load_9_max = row_max_reduce(_reg_reduce_max2_12);
                            float _max_5 = max_noftz(row_max_0_1, _tmem_load_9_max);
                            row_max_0_1 = _max_5;
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float row_center_1 = (row_max_0_1 + row_min_1) * 0.5f;
                            float residual_abs_max_1 = (row_max_0_1 - row_min_1) * 0.5f;
                            float dequant_scale_1 = 0.0f;
                            float quant_scale_1 = 0.0f;
                            if (residual_abs_max_1 > 0.0f && residual_abs_max_1 == residual_abs_max_1) {
                                dequant_scale_1 = residual_abs_max_1 * inv_sum_1 * 0.002232142857142857f;
                                quant_scale_1 = 448.0f / residual_abs_max_1;
                            }
                            if (output_valid_1 != 0) {
                                partial_scale[partial_row_1] = dequant_scale_1;
                                partial_scale[partial_metadata_rows_1 + partial_row_1] = row_center_1 * inv_sum_1;
                            }
                            #pragma unroll 1
                            for (int output_segment_3 = 0; output_segment_3 < 4; output_segment_3++) {
                                float _tmem_load_10[16];
                                tmem_ld_x16(&_tmem_load_10[0], output_row_addr_1 + output_segment_3 * 16);
                                const float2 _sub2_13 = {row_center_1, row_center_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_10)[_ls], _sub2_13);
                                if (output_valid_1 != 0) {
                                    {
                                        const float2 _prescale2_14 = {quant_scale_1, quant_scale_1};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_10[0])[_ps], _prescale2_14);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 16; _ps++)
                                            _tmem_load_10[0 + _ps] *= quant_scale_1;
                                        #endif
                                        unsigned int _fp8_pk[4];
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_10[0 + 1]), "f"(_tmem_load_10[0 + 0]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_10[0 + 3]), "f"(_tmem_load_10[0 + 2]));
                                            _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_10[0 + 5]), "f"(_tmem_load_10[0 + 4]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_10[0 + 7]), "f"(_tmem_load_10[0 + 6]));
                                            _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_10[0 + 9]), "f"(_tmem_load_10[0 + 8]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_10[0 + 11]), "f"(_tmem_load_10[0 + 10]));
                                            _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        { unsigned short _lo, _hi;
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_10[0 + 13]), "f"(_tmem_load_10[0 + 12]));
                                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_10[0 + 15]), "f"(_tmem_load_10[0 + 14]));
                                            _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                        }
                                        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + (long long)output_segment_3 * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                    }
                                }
                            }
                            const float2 _sub2_15 = {row_center_1, row_center_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_ls], _sub2_15);
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_16 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_9[0])[_ps], _prescale2_16);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_9[0 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[0 + 1]), "f"(_tmem_load_9[0 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[0 + 3]), "f"(_tmem_load_9[0 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[0 + 5]), "f"(_tmem_load_9[0 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[0 + 7]), "f"(_tmem_load_9[0 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[0 + 9]), "f"(_tmem_load_9[0 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[0 + 11]), "f"(_tmem_load_9[0 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[0 + 13]), "f"(_tmem_load_9[0 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[0 + 15]), "f"(_tmem_load_9[0 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 64)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_17 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_9[16])[_ps], _prescale2_17);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_9[16 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[16 + 1]), "f"(_tmem_load_9[16 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[16 + 3]), "f"(_tmem_load_9[16 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[16 + 5]), "f"(_tmem_load_9[16 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[16 + 7]), "f"(_tmem_load_9[16 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[16 + 9]), "f"(_tmem_load_9[16 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[16 + 11]), "f"(_tmem_load_9[16 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[16 + 13]), "f"(_tmem_load_9[16 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[16 + 15]), "f"(_tmem_load_9[16 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 80)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_18 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_9[32])[_ps], _prescale2_18);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_9[32 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[32 + 1]), "f"(_tmem_load_9[32 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[32 + 3]), "f"(_tmem_load_9[32 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[32 + 5]), "f"(_tmem_load_9[32 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[32 + 7]), "f"(_tmem_load_9[32 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[32 + 9]), "f"(_tmem_load_9[32 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[32 + 11]), "f"(_tmem_load_9[32 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[32 + 13]), "f"(_tmem_load_9[32 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[32 + 15]), "f"(_tmem_load_9[32 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 96)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_19 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_9[48])[_ps], _prescale2_19);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_9[48 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[48 + 1]), "f"(_tmem_load_9[48 + 0]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[48 + 3]), "f"(_tmem_load_9[48 + 2]));
                                        _fp8_pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[48 + 5]), "f"(_tmem_load_9[48 + 4]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[48 + 7]), "f"(_tmem_load_9[48 + 6]));
                                        _fp8_pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[48 + 9]), "f"(_tmem_load_9[48 + 8]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[48 + 11]), "f"(_tmem_load_9[48 + 10]));
                                        _fp8_pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    { unsigned short _lo, _hi;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(_tmem_load_9[48 + 13]), "f"(_tmem_load_9[48 + 12]));
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(_tmem_load_9[48 + 15]), "f"(_tmem_load_9[48 + 14]));
                                        _fp8_pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
                                    }
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 112)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
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
        }
    }
    // ---- Role: qload ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
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
            int group_count_2 = 1;
            {
                if (blockIdx.x < q_group_segment_end_16) {
                    if (blockIdx.x < q_group_segment_end_64) {
                        group_count_2 = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                    } else {
                        group_count_2 = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                    }
                } else if (blockIdx.x < q_group_segment_end_4) {
                    group_count_2 = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
                } else {
                    group_count_2 = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
                }
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
                        int token_in_group_2 = qload_warp * 2;
                        int edge_in_work_2 = group_2 * 8 + token_in_group_2;
                        int edge_valid = ((edge_in_work_2 < q_count_2) ? 1 : 0);
                        int safe_edge = ((edge_valid != 0) ? edge_in_work_2 : 0);
                        int packed_q_2 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge];
                        int decoded_q_abs = q_batch_offset_2 + (packed_q_2 & 16777215);
                        int q_abs_2 = ((edge_valid != 0) ? decoded_q_abs : 0);
                        int dst_offset = token_in_group_2 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset, (&q), 0, head_kv_2 * 16, 0, q_abs_2, q_full_addr + (q_stage) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset, (&q), 0, head_kv_2 * 16, 1, q_abs_2, q_full_addr + (q_stage) * 8);
                        int token_in_group_0 = qload_warp * 2 + 1;
                        int edge_in_work_1_1 = group_2 * 8 + token_in_group_0;
                        int edge_valid_2 = ((edge_in_work_1_1 < q_count_2) ? 1 : 0);
                        int safe_edge_3 = ((edge_valid_2 != 0) ? edge_in_work_1_1 : 0);
                        int packed_q_4 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_3];
                        int decoded_q_abs_5 = q_batch_offset_2 + (packed_q_4 & 16777215);
                        int q_abs_6 = ((edge_valid_2 != 0) ? decoded_q_abs_5 : 0);
                        int dst_offset_7 = token_in_group_0 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset_7, (&q), 0, head_kv_2 * 16, 0, q_abs_6, q_full_addr + (q_stage) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset_7, (&q), 0, head_kv_2 * 16, 1, q_abs_6, q_full_addr + (q_stage) * 8);
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
            int group_count_3 = 1;
            {
                if (blockIdx.x < q_group_segment_end_16) {
                    if (blockIdx.x < q_group_segment_end_64) {
                        group_count_3 = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                    } else {
                        group_count_3 = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                    }
                } else if (blockIdx.x < q_group_segment_end_4) {
                    group_count_3 = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
                } else {
                    group_count_3 = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
                }
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
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(_mma_b_lo_2), "r"(tmem_scores + (pv_stage * 128 + 64)), "r"(0));
                mbarrier_wait(p_full_2_addr + (pv_stage) * 8, pv_phase);
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
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(_mma_b_lo_5), "r"(tmem_scores + (pv_stage_1 * 128 + 64)), "r"(0));
                mbarrier_wait(p_full_2_addr + (pv_stage_1) * 8, pv_phase_1);
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
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
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
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr, 32768);
                int token0 = token_base;
                int token1 = token_base + 64;
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr), "l"((&k)), "r"(0), "r"(token0), "r"(0), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr + 8192), "l"((&k)), "r"(0), "r"(token1), "r"(0), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr + 16384), "l"((&k)), "r"(0), "r"(token0), "r"(1), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr + 24576), "l"((&k)), "r"(0), "r"(token1), "r"(1), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                mbarrier_arrive_expect_tx(v_full_addr, 32768);
                int token0_0 = token_base;
                int token1_1 = token_base + 64;
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr), "l"((&v)), "r"(0), "r"(token0_0), "r"(0), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr + 8192), "l"((&v)), "r"(0), "r"(token1_1), "r"(0), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr + 16384), "l"((&v)), "r"(0), "r"(token0_0), "r"(1), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr + 24576), "l"((&v)), "r"(0), "r"(token1_1), "r"(1), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
    }

    // Cleanup
}

} // extern "C"
