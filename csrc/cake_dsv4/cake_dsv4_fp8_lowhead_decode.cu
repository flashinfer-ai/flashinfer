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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 8
#define NUM_INDEX_PIPE_STAGES 1
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 33792
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 33792
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_Q_FULL_OFF 1024
#define SMEM_SMEM_Q_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_Q_FULL_STRIDE 32768
#define SMEM_SMEM_K_FULL_OFF 33792
#define SMEM_SMEM_K_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_K_FULL_STRIDE 32768
#define SMEM_SMEM_V_FULL_OFF 99328
#define SMEM_SMEM_V_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_V_FULL_STRIDE 32768
#define SMEM_SMEM_STATS_MAX_OFF 164864
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_MAX_STRIDE 1024
#define SMEM_SMEM_STATS_SUM_OFF 165888
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 512
#define SMEM_SMEM_STATS_SUM_STRIDE 512
#define SMEM_SMEM_STATS_FINAL_MAX_OFF 166400
#define SMEM_SMEM_STATS_FINAL_MAX_STAGE_BYTES 512
#define SMEM_SMEM_STATS_FINAL_MAX_STRIDE 512
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_OFF 166912
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STAGE_BYTES 1024
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STRIDE 1024
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_OFF 167936
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STRIDE 512
#define SMEM_SMEM_SPARSE_INDICES_OFF 168448
#define SMEM_SMEM_SPARSE_INDICES_STAGE_BYTES 4608
#define SMEM_SMEM_SPARSE_INDICES_STRIDE 4608
#define SMEM_SMEM_P_FP8_OFF 1024
#define SMEM_SMEM_P_FP8_STAGE_BYTES 8192
#define SMEM_SMEM_P_FP8_STRIDE 8192
#define SMEM_TOTAL 173056
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


__device__ __forceinline__ void tcgen05_mma_f8f6f4_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
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
        "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, %3, "
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
        "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], db, %4, "
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


__device__ __forceinline__ void tma_gather4_gmem2smem_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form; see tma_gather4_gmem2smem above.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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


__device__ __forceinline__ void tma_gather4_gmem2smem_mc_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast + cta_group::2 variant; see tma_gather4_gmem2smem_mc.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
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

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_fp8_lowhead_decode(CakeTensorMap const* tmap_q, CakeTensorMap const* tmap_swa_kv, CakeTensorMap const* tmap_compressed_kv, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_lse, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_swa_kv)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_compressed_kv)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_kv_addr = smem + 33792;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v_addr = smem + 33792;
    uint8_t* smem_q_full = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_full_addr = smem + 1024;
    uint8_t* smem_k_full = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_k_full_addr = smem + 33792;
    uint8_t* smem_v_full = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_v_full_addr = smem + 99328;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 164864);
    const int smem_stats_max_addr = smem + 164864;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 165888);
    const int smem_stats_sum_addr = smem + 165888;
    float* smem_stats_final_max = reinterpret_cast<float*>(smem_raw + 166400);
    const int smem_stats_final_max_addr = smem + 166400;
    float* smem_softmax_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 166912);
    const int smem_softmax_warp_pair_exchange_addr = smem + 166912;
    float* smem_corr_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 167936);
    const int smem_corr_warp_pair_exchange_addr = smem + 167936;
    int* smem_sparse_indices = reinterpret_cast<int*>(smem_raw + 168448);
    const int smem_sparse_indices_addr = smem + 168448;
    uint8_t* smem_p_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_p_fp8_addr = smem + 1024;

    // Mbarrier init (23 groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // k_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // v_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 8 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // kv_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // --- pipeline 'index_pipe' ---
            // index_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // index_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            // s_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 280, 256);
            // q_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 288, 64);
            // --- pipeline 'kv_pipe' ---
            // kv_pair_ready: 8 barriers, init_count=64
            mbarrier_init(smem + 296, 64);
            mbarrier_init(smem + 304, 64);
            mbarrier_init(smem + 312, 64);
            mbarrier_init(smem + 320, 64);
            mbarrier_init(smem + 328, 64);
            mbarrier_init(smem + 336, 64);
            mbarrier_init(smem + 344, 64);
            mbarrier_init(smem + 352, 64);
            // pv_pair_ready: 2 barriers, init_count=64
            mbarrier_init(smem + 360, 64);
            mbarrier_init(smem + 368, 64);
            // tmem_dealloc: 1 barriers, init_count=416
            mbarrier_init(smem + 376, 416);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 384, 32);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 0) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 24)
    #define v_full_addr (mbar_base + 32)
    #define v_empty_addr (mbar_base + 40)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 112)
    #define index_full_addr (mbar_base + 176)
    #define index_empty_addr (mbar_base + 184)
    #define s_full_addr (mbar_base + 192)
    #define p_full_addr (mbar_base + 208)
    #define corr_done_addr (mbar_base + 224)
    #define stats_addr (mbar_base + 240)
    #define sum_ready_addr (mbar_base + 256)
    #define o_done_addr (mbar_base + 264)
    #define pv_done_addr (mbar_base + 272)
    #define s_seeded_addr (mbar_base + 280)
    #define q_pair_ready_addr (mbar_base + 288)
    #define kv_pair_ready_addr (mbar_base + 296)
    #define pv_pair_ready_addr (mbar_base + 360)
    #define tmem_dealloc_addr (mbar_base + 376)
    #define tmem_dealloc_peer_addr (mbar_base + 384)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: index_warp ----
    if (warp == 13) {
        { // index_warp_main
            const int index_dummy = 0;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_q_full_0 = 0;
            {
                #pragma unroll 1
                for (unsigned int work_idx = bid / 2; work_idx < total_work_items; work_idx += num_bids / 2) {
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 9 && warp <= 12) {
        { // load_warp_main
            const int wg2_dummy = 0;
            const int load_warp_rank = warp - ((0) ? 12 : 9);
            int all_num_kv_tiles = (sparse_topk + 128 - 1) / 128;
            unsigned int load_k_stage = 0;
            unsigned int load_v_stage = 0;
            unsigned int load_kv_stage = 0;
            unsigned int load_k_index_stage = 0;
            unsigned int load_v_index_stage = 0;
            int k_cta_offset = bid % 2 * 64;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_index_full = 0;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int work_idx_1 = bid / 2; work_idx_1 < total_work_items; work_idx_1 += num_bids / 2) {
                int split_idx = 0;
                int query_idx = work_idx_1 >> 1;
                int v_chunk = work_idx_1 & 1;
                {
                }
                int tiles_per_split = all_num_kv_tiles;
                int first_tile = split_idx * tiles_per_split;
                int num_kv_tiles = tiles_per_split;
                int sparse_extent = num_kv_tiles * 128;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (load_warp_rank == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr, 32768);
                        #pragma unroll
                        for (int q_stage = 0; q_stage < 4; q_stage++) {
                            tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 8192), tmap_q, 0, bid % 2 * 64, q_stage, query_idx, q_full_addr);
                        }
                    }
                }
                {
                    int num_index_passes = sparse_extent / 128;
                    int sparse_base = query_idx * sparse_topk;
                    #pragma unroll 1
                    for (int index_pass = load_warp_rank; index_pass < num_index_passes; index_pass += 4) {
                        int index_offset = index_pass * 128 + lane * 4;
                        int global_index_offset = first_tile * 128 + index_offset;
                        int sparse_rows[4];
                        #pragma unroll
                        for (int row_i = 0; row_i < 4; row_i++) {
                            sparse_rows[row_i] = -1;
                        }
                        if (global_index_offset < sparse_topk) {
                            int _vec_load_0[4];
                            {
                                int4 _iv4 = *reinterpret_cast<const int4*>(sparse_indices + (sparse_base + global_index_offset) + 0);
                                _vec_load_0[0 + 0] = _iv4.x;
                                _vec_load_0[0 + 1] = _iv4.y;
                                _vec_load_0[0 + 2] = _iv4.z;
                                _vec_load_0[0 + 3] = _iv4.w;
                            }
                            #pragma unroll
                            for (int row_i_1 = 0; row_i_1 < 4; row_i_1++) {
                                sparse_rows[row_i_1] = _vec_load_0[row_i_1];
                            }
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sparse_indices_addr + (unsigned int)(index_offset * 4)), "r"(sparse_rows[0]), "r"(sparse_rows[1]), "r"(sparse_rows[2]), "r"(sparse_rows[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int k_index_stage_base = smem_sparse_indices_addr;
                {
                    #pragma unroll
                    for (int qk_stage = 0; qk_stage < 4; qk_stage++) {
                        if (qk_stage == load_warp_rank) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 8192);
                            }
                            int k_dst = smem_kv_addr + load_kv_stage * 16384;
                            int group = lane;
                            if (group < 16) {
                                int group_offset = k_cta_offset + group * 4;
                                int raw_rows[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                                    : "r"(k_index_stage_base + group_offset * 4));
                                int row0 = ((raw_rows[0] >= 0) ? raw_rows[0] : 0);
                                int row1 = ((raw_rows[1] >= 0) ? raw_rows[1] : 0);
                                int row2 = ((raw_rows[2] >= 0) ? raw_rows[2] : 0);
                                int row3 = ((raw_rows[3] >= 0) ? raw_rows[3] : 0);
                                if (first_tile == 0) {
                                    tma_gather4_gmem2smem(k_dst + group * 512, tmap_swa_kv, qk_stage * 128, row0, row1, row2, row3, kv_full_addr + (load_kv_stage) * 8);
                                } else {
                                    tma_gather4_gmem2smem(k_dst + group * 512, tmap_compressed_kv, qk_stage * 128, row0, row1, row2, row3, kv_full_addr + (load_kv_stage) * 8);
                                }
                            }
                        }
                        load_kv_stage += 1;
                        if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int tile = 1; tile < num_kv_tiles; tile++) {
                    if ((0 & (int)((tile & 1) == 0)) != 0) {
                        mbarrier_wait(index_full_addr + (load_k_index_stage) * 8, _phase_index_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    k_index_stage_base = smem_sparse_indices_addr;
                    {
                        #pragma unroll
                        for (int qk_stage_1 = 0; qk_stage_1 < 4; qk_stage_1++) {
                            if (qk_stage_1 == load_warp_rank) {
                                mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 8192);
                                }
                                int k_dst_1 = smem_kv_addr + load_kv_stage * 16384;
                                int group_1 = lane;
                                if (group_1 < 16) {
                                    int group_offset_1 = tile * 128 + k_cta_offset + group_1 * 4;
                                    int raw_rows_1[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                                        : "r"(k_index_stage_base + group_offset_1 * 4));
                                    int row0_1 = ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : 0);
                                    int row1_1 = ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : 0);
                                    int row2_1 = ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : 0);
                                    int row3_1 = ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : 0);
                                    tma_gather4_gmem2smem(k_dst_1 + group_1 * 512, tmap_compressed_kv, qk_stage_1 * 128, row0_1, row1_1, row2_1, row3_1, kv_full_addr + (load_kv_stage) * 8);
                                }
                            }
                            load_kv_stage += 1;
                            if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                    int prev_tile = tile - 1;
                    int global_prev_tile = first_tile + prev_tile;
                    int v_index_stage_base = smem_sparse_indices_addr;
                    {
                        #pragma unroll
                        for (int pv_stage = 0; pv_stage < 2; pv_stage++) {
                            if (pv_stage == load_warp_rank) {
                                mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                                }
                                int v_dst = smem_kv_addr + load_kv_stage * 16384;
                                int v_col = v_chunk * 256 + pv_stage * 128;
                                int group_2 = lane;
                                int group_offset_2 = prev_tile * 128 + group_2 * 4;
                                int raw_rows_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 3]))
                                    : "r"(v_index_stage_base + group_offset_2 * 4));
                                int row0_2 = ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : 0);
                                int row1_2 = ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : 0);
                                int row2_2 = ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : 0);
                                int row3_2 = ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : 0);
                                if (global_prev_tile == 0) {
                                    tma_gather4_gmem2smem(v_dst + group_2 * 512, tmap_swa_kv, v_col, row0_2, row1_2, row2_2, row3_2, kv_full_addr + (load_kv_stage) * 8);
                                } else {
                                    tma_gather4_gmem2smem(v_dst + group_2 * 512, tmap_compressed_kv, v_col, row0_2, row1_2, row2_2, row3_2, kv_full_addr + (load_kv_stage) * 8);
                                }
                            }
                            load_kv_stage += 1;
                            if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                int last_tile = num_kv_tiles - 1;
                int global_last_tile = first_tile + last_tile;
                {
                    #pragma unroll
                    for (int pv_stage_1 = 0; pv_stage_1 < 2; pv_stage_1++) {
                        if (pv_stage_1 == load_warp_rank) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 16384);
                            }
                            int v_dst_1 = smem_kv_addr + load_kv_stage * 16384;
                            int v_col_1 = v_chunk * 256 + pv_stage_1 * 128;
                            int group_3 = lane;
                            int group_offset_3 = last_tile * 128 + group_3 * 4;
                            int raw_rows_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 3]))
                                : "r"(smem_sparse_indices_addr + (unsigned int)(group_offset_3 * 4)));
                            int row0_3 = ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : 0);
                            int row1_3 = ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : 0);
                            int row2_3 = ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : 0);
                            int row3_3 = ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : 0);
                            if (global_last_tile == 0) {
                                tma_gather4_gmem2smem(v_dst_1 + group_3 * 512, tmap_swa_kv, v_col_1, row0_3, row1_3, row2_3, row3_3, kv_full_addr + (load_kv_stage) * 8);
                            } else {
                                tma_gather4_gmem2smem(v_dst_1 + group_3 * 512, tmap_compressed_kv, v_col_1, row0_3, row1_3, row2_3, row3_3, kv_full_addr + (load_kv_stage) * 8);
                            }
                        }
                        load_kv_stage += 1;
                        if (load_kv_stage == 8) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            const int wg_dummy_inc = 0;
            int all_num_kv_tiles_1 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base = ((0) ? warp % 2 * 32 : warp % 4 * 32);
            const int tmem_score_row_base = ((0) ? warp % 4 * 32 : tmem_row_base);
            const int n_half = ((0) ? warp % 4 / 2 : 0);
            const int my_row = tmem_row_base + lane;
            const int stats_row = n_half * 64 + my_row;
            int softmax_tile_cursor = 0;
            unsigned int _phase_q_full_0_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = bid / 2; work_idx_2 < total_work_items; work_idx_2 += num_bids / 2) {
                int split_idx_1 = 0;
                int query_idx_1 = work_idx_2 >> 1;
                {
                }
                int tiles_per_split_1 = all_num_kv_tiles_1;
                int first_tile_1 = split_idx_1 * tiles_per_split_1;
                int num_kv_tiles_1 = tiles_per_split_1;
                int active_topk = sparse_topk_lens[query_idx_1];
                {
                    float seed_zero[4];
                    #pragma unroll
                    for (int seed_c4 = 0; seed_c4 < 4; seed_c4++) {
                        seed_zero[seed_c4] = 0.0f;
                    }
                    #pragma unroll
                    for (int seed_half = 0; seed_half < 2; seed_half++) {
                        #pragma unroll
                        for (int seed_c = 0; seed_c < 8; seed_c++) {
                            int seed_addr = taddr + (unsigned int)(((seed_half == 0) ? 96 : 224)) + (unsigned int)(seed_c * 4) + (unsigned int)(tmem_row_base << 16);
                            tmem_st_x4_f32(seed_addr, seed_zero);
                        }
                    }
                    {
                        #pragma unroll
                        for (int seed_half_1 = 0; seed_half_1 < 2; seed_half_1++) {
                            #pragma unroll
                            for (int seed_c_1 = 16; seed_c_1 < 24; seed_c_1++) {
                                int seed_addr_1 = taddr + (unsigned int)(((seed_half_1 == 0) ? 0 : 128)) + (unsigned int)(seed_c_1 * 4) + (unsigned int)(tmem_row_base << 16);
                                tmem_st_x4_f32(seed_addr_1, seed_zero);
                            }
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(s_seeded_addr);
                    {
                        int seed_peer_rank = bid % 2 ^ 1;
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(s_seeded_addr), "r"(seed_peer_rank) : "memory");
                    }
                }
                {
                    mbarrier_wait(q_full_addr, _phase_q_full_0_1);
                    _phase_q_full_0_1 ^= 1;
                }
                float row_max_val = -CAKE_INF;
                float row_sum_val = 0.0f;
                int sink_head = ((0) ? bid % 2 * 64 + my_row : my_row);
                if (has_sinks != 0 && sink_head < num_heads && split_idx_1 == 0) {
                    row_max_val = sinks[sink_head] * 1.4426950408889634f / softmax_scale_log2;
                    row_sum_val = 1.0f;
                }
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                    int pipeline_tile = softmax_tile_cursor + tile_1;
                    int phase = pipeline_tile & 1;
                    int s_wait_phase = pipeline_tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (phase) * 8, s_wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_off = ((phase != 0) ? 128 : 0);
                    int s_base = taddr + (unsigned int)s_off + (unsigned int)(tmem_score_row_base << 16);
                    float new_max = row_max_val;
                    int valid_sparse_cols = ((active_topk < sparse_topk) ? active_topk : sparse_topk);
                    valid_sparse_cols = valid_sparse_cols - (first_tile_1 + tile_1) * 128 - n_half * 64;
                    if (valid_sparse_cols < 0) {
                        valid_sparse_cols = 0;
                    }
                    if (valid_sparse_cols > 128) {
                        valid_sparse_cols = 128;
                    }
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], s_base);
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], s_base);
                    {
                        {
                            #pragma unroll
                            for (int frag = 0; frag < 4; frag++) {
                                float _tmem_load_2[32];
                                tmem_ld_x32(&_tmem_load_2[0], s_base + frag * 32);
                                int frag_valid = valid_sparse_cols - frag * 32;
                                uint32_t _slice_lo_mask_1;
                                {
                                    int _lim_0 = frag_valid;
                                    if (_lim_0 <= 0) { _slice_lo_mask_1 = 0u; }
                                    else if (_lim_0 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_0));
                                    }
                                }
                                #pragma unroll
                                for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                                    if (!(_slice_lo_mask_1 & (1u << _i_1))) _tmem_load_2[0 + _i_1] = -CAKE_INF;
                                }
                                float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                                row_max_x32_accum(&_tmem_load_2[0], _reg_reduce_max2_2);
                                float _tmem_load_2_max = row_max_reduce(_reg_reduce_max2_2);
                                float _max_2 = max_noftz(new_max, _tmem_load_2_max);
                                new_max = _max_2;
                            }
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
                    {
                        {
                            int p_off = ((phase != 0) ? 224 : 96);
                            int p_base = taddr + (unsigned int)p_off + (unsigned int)(tmem_row_base << 16);
                            #pragma unroll
                            for (int frag_1 = 0; frag_1 < 4; frag_1++) {
                                float _tmem_load_3[32];
                                tmem_ld_x32(&_tmem_load_3[0], s_base + frag_1 * 32);
                                int frag_valid_1 = valid_sparse_cols - frag_1 * 32;
                                uint32_t _slice_lo_mask_3;
                                {
                                    int _lim_3 = frag_valid_1;
                                    if (_lim_3 <= 0) { _slice_lo_mask_3 = 0u; }
                                    else if (_lim_3 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_3));
                                    }
                                }
                                #pragma unroll
                                for (int _i_4 = 0; _i_4 < 32; _i_4++) {
                                    if (!(_slice_lo_mask_3 & (1u << _i_4))) _tmem_load_3[0 + _i_4] = -CAKE_INF;
                                }
                                const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                                const float2 _fma_c2_6 = {-max_scaled, -max_scaled};
                                #pragma unroll
                                for (int _lf = 0; _lf < 16; _lf++)
                                    fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_lf], _fma_b2_5, _fma_c2_6);
                                #pragma unroll
                                for (int _le = 0; _le < 32; _le++) {
                                    _tmem_load_3[_le] = approx_exp2(_tmem_load_3[_le]);
                                }
                                float2 _reg_reduce_sum2_7 = make_float2(0.0f, 0.0f);
                                softmax_block_sum(&_tmem_load_3[0], &_reg_reduce_sum2_7);
                                float _tmem_load_3_sum = _reg_reduce_sum2_7.x + _reg_reduce_sum2_7.y;
                                block_sum = block_sum + _tmem_load_3_sum;
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
                                    tmem_st_x8_u32(p_base + frag_1 * 8, _pv_packed);
                                }
                            }
                        }
                    }
                    {
                        float _fma_1 = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                        row_sum_val = _fma_1;
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + (phase) * 8);
                }
                smem_stats_sum[stats_row] = row_sum_val;
                smem_stats_final_max[stats_row] = row_max_val;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(sum_ready_addr);
                softmax_tile_cursor = softmax_tile_cursor + num_kv_tiles_1;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // correction_wg_main
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            const int wg_dummy_inc_1 = 0;
            int all_num_kv_tiles_2 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base_1 = ((0) ? warp % 2 * 32 : warp % 4 * 32);
            const int n_half_1 = ((0) ? warp % 4 / 2 : 0);
            const int my_row_1 = tmem_row_base_1 + lane;
            const int stats_row_1 = n_half_1 * 64 + my_row_1;
            const int corr_row = tmem_row_base_1 << 16;
            int correction_tile_cursor = 0;
            unsigned int _phase_q_full_0_2 = 0;
            unsigned int _phase_pv_done_0 = 0;
            unsigned int _phase_o_done_0 = 0;
            unsigned int _phase_sum_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_3 = bid / 2; work_idx_3 < total_work_items; work_idx_3 += num_bids / 2) {
                int split_idx_2 = 0;
                int query_idx_2 = work_idx_3 >> 1;
                int v_chunk_1 = work_idx_3 & 1;
                {
                }
                int tiles_per_split_2 = all_num_kv_tiles_2;
                int first_tile_2 = split_idx_2 * tiles_per_split_2;
                int num_kv_tiles_2 = tiles_per_split_2;
                {
                    mbarrier_wait(q_full_addr, _phase_q_full_0_2);
                    _phase_q_full_0_2 ^= 1;
                }
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                    int pipeline_tile_1 = correction_tile_cursor + tile_2;
                    int phase_1 = pipeline_tile_1 & 1;
                    int stats_wait_phase = pipeline_tile_1 >> 1 & 1;
                    mbarrier_wait(stats_addr + (phase_1) * 8, stats_wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc_scale_1 = smem_stats_max[phase_1 * 128 + stats_row_1];
                    if (tile_2 > 0) {
                        mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                        _phase_pv_done_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    if (tile_2 > 0) {
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        int any_rescale = _vote_0;
                        if (any_rescale != 0) {
                            #pragma unroll
                            for (int vs = 0; vs < 2; vs++) {
                                int o_base = taddr + 256 + (unsigned int)(vs * 128) + (unsigned int)corr_row;
                                #pragma unroll
                                for (int c = 0; c < 128; c += 16) {
                                    float _tmem_load_4[16];
                                    tmem_ld_x16(&_tmem_load_4[0], o_base + c);
                                    const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_0);
                                    tmem_st_x16_f32(o_base + c, _tmem_load_4);
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
                float total_sum = smem_stats_sum[stats_row_1];
                float final_max = smem_stats_final_max[stats_row_1];
                float _rcp_0 = approx_rcp(total_sum);
                float inv_sum = ((total_sum > 0.0f) ? _rcp_0 : 0.0f);
                int head_idx = my_row_1;
                int direct_o_offset = (query_idx_2 * num_heads + head_idx) * 512 + v_chunk_1 * 256;
                int partial_o_offset = (query_idx_2 * num_heads + head_idx + split_idx_2) * 512 + ((0) ? v_chunk_1 : n_half_1) * 256;
                int o_offset = direct_o_offset;
                if ((0 & (int)(n_half_1 == 0) & (int)(head_idx < num_heads)) != 0) {
                    int stat_offset = query_idx_2 * num_heads + head_idx + split_idx_2;
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(total_sum));
                    partial_lse[stat_offset] = ((total_sum > 0.0f) ? final_max * softmax_scale_log2_1 + _log2_0 : -CAKE_INF);
                }
                #pragma unroll
                for (int vs_1 = 0; vs_1 < 2; vs_1++) {
                    int o_base_epi = taddr + 256 + (unsigned int)(vs_1 * 128) + (unsigned int)corr_row;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 128; c_1 += 32) {
                        float _tmem_load_5[32];
                        tmem_ld_x32(&_tmem_load_5[0], o_base_epi + c_1);
                        int gmem_base = o_offset + vs_1 * 128 + c_1;
                        if (head_idx < num_heads) {
                            #pragma unroll
                            for (int j = 0; j < 32; j += 8) {
                                {
                                    const float2 _prescale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_5[j])[_ps], _prescale2_1);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_5[j + _ps] *= inv_sum * output_scale;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_5[j + 0], _tmem_load_5[j + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_5[j + 2], _tmem_load_5[j + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_5[j + 4], _tmem_load_5[j + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_5[j + 6], _tmem_load_5[j + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (gmem_base + j)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
                correction_tile_cursor = correction_tile_cursor + num_kv_tiles_2;
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
            unsigned int _phase_s_seeded_0 = 0;
            unsigned int _phase_q_full_0_3 = 0;
            unsigned int _phase_q_pair_ready_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_kv_pair_ready = 0;
            unsigned int _phase_v_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx_4 = bid / 2; work_idx_4 < total_work_items; work_idx_4 += num_bids / 2) {
                int split_idx_3 = 0;
                int tiles_per_split_3 = all_num_kv_tiles_3;
                int first_tile_3 = split_idx_3 * tiles_per_split_3;
                int num_kv_tiles_3 = tiles_per_split_3;
                {
                    {
                        mbarrier_wait_cluster(s_seeded_addr, _phase_s_seeded_0);
                        _phase_s_seeded_0 ^= 1;
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                {
                    mbarrier_wait(q_full_addr, _phase_q_full_0_3);
                    _phase_q_full_0_3 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(q_pair_ready_addr), "r"(0) : "memory");
                        if (cta_rank == 0) {
                            mbarrier_wait_cluster(q_pair_ready_addr, _phase_q_pair_ready_0);
                            _phase_q_pair_ready_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                    }
                }
                int first_pv = 1;
                #pragma unroll 1
                for (int tile_3 = 0; tile_3 < num_kv_tiles_3; tile_3++) {
                    int pipeline_tile_2 = mma_tile_cursor + tile_3;
                    int phase_2 = pipeline_tile_2 & 1;
                    int score_col = ((phase_2 != 0) ? 128 : 0);
                    {
                        #pragma unroll 1
                        for (int n = 0; n < 4; n++) {
                            mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(kv_pair_ready_addr + mma_kv_stage * 8), "r"(0) : "memory");
                                if (cta_rank == 0) {
                                    mbarrier_wait_cluster(kv_pair_ready_addr + (mma_kv_stage) * 8, _phase_kv_pair_ready);
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                    int _mma_a_lo_5 = (((smem_q_addr) >> 4) & 0x3FFF) + (n) * 512;
                                    int _mma_b_lo_5 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
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
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_tmem_scratch + (score_col))), "r"(((n == 0) ? 0 : 1)));
                                }
                                if (cta_rank == 0) {
                                    if (n == 3) {
                                        if (cta_rank == 0) {
                                            elect_commit_cg2_multicast(s_full_addr + (phase_2) * 8, (uint16_t)(3));
                                        }
                                    }
                                    if (cta_rank == 0) {
                                        elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                                    }
                                }
                            }
                            mma_kv_stage += 1;
                            if (mma_kv_stage == 8) { mma_kv_stage = 0; _phase_kv_full ^= 1; _phase_kv_pair_ready ^= 1; }
                        }
                    }
                    if (tile_3 > 0) {
                        int prev_pipeline_tile = pipeline_tile_2 - 1;
                        int prev_phase = prev_pipeline_tile & 1;
                        int pv_wait_phase = prev_pipeline_tile >> 1 & 1;
                        mbarrier_wait(p_full_addr + (prev_phase) * 8, pv_wait_phase);
                        mbarrier_wait(corr_done_addr + (prev_phase) * 8, pv_wait_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(pv_pair_ready_addr + (unsigned int)(prev_phase * 8)), "r"(0) : "memory");
                            if (cta_rank == 0) {
                                mbarrier_wait_cluster(pv_pair_ready_addr + (prev_phase) * 8, pv_wait_phase);
                                {
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                }
                            }
                        }
                        int p_col = ((prev_phase != 0) ? 224 : 96);
                        {
                            #pragma unroll
                            for (int vs_2 = 0; vs_2 < 2; vs_2++) {
                                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int output_col = 256 + vs_2 * 128;
                                {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b32 remAddr32;\n\t"
                                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                        "}"
                                        :: "r"(kv_pair_ready_addr + mma_kv_stage * 8), "r"(0) : "memory");
                                    if (cta_rank == 0) {
                                        mbarrier_wait_cluster(kv_pair_ready_addr + (mma_kv_stage) * 8, _phase_kv_pair_ready);
                                        asm volatile("tcgen05.fence::after_thread_sync;");
                                        int _mma_b_lo_9 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x8000000) + (mma_kv_stage) * 1024;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2 + 8], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2 + 16], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2 + 24], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (output_col))), "r"(_mma_b_lo_9), "r"(tmem_tmem_scratch + p_col), "r"(((first_pv) ? 0 : 1)));
                                    }
                                    if (cta_rank == 0) {
                                        if (cta_rank == 0) {
                                            elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                                        }
                                    }
                                }
                                mma_kv_stage += 1;
                                if (mma_kv_stage == 8) { mma_kv_stage = 0; _phase_kv_full ^= 1; _phase_kv_pair_ready ^= 1; }
                            }
                        }
                        first_pv = 0;
                        if (cta_rank == 0) {
                            elect_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                        }
                    }
                }
                int last_pipeline_tile = mma_tile_cursor + num_kv_tiles_3 - 1;
                int last_phase = last_pipeline_tile & 1;
                int drain_wait_phase = last_pipeline_tile >> 1 & 1;
                mbarrier_wait(p_full_addr + (last_phase) * 8, drain_wait_phase);
                mbarrier_wait(corr_done_addr + (last_phase) * 8, drain_wait_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(pv_pair_ready_addr + (unsigned int)(last_phase * 8)), "r"(0) : "memory");
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster(pv_pair_ready_addr + (last_phase) * 8, drain_wait_phase);
                        {
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                    }
                }
                int p_col_last = ((last_phase != 0) ? 224 : 96);
                {
                    #pragma unroll
                    for (int vs_3 = 0; vs_3 < 2; vs_3++) {
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int output_col_d = 256 + vs_3 * 128;
                        {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(kv_pair_ready_addr + mma_kv_stage * 8), "r"(0) : "memory");
                            if (cta_rank == 0) {
                                mbarrier_wait_cluster(kv_pair_ready_addr + (mma_kv_stage) * 8, _phase_kv_pair_ready);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_b_lo_13 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x8000000) + (mma_kv_stage) * 1024;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2 + 8], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2 + 16], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%2 + 24], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (output_col_d))), "r"(_mma_b_lo_13), "r"(tmem_tmem_scratch + p_col_last), "r"(((first_pv) ? 0 : 1)));
                            }
                            if (cta_rank == 0) {
                                if (cta_rank == 0) {
                                    elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                                }
                            }
                        }
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 8) { mma_kv_stage = 0; _phase_kv_full ^= 1; _phase_kv_pair_ready ^= 1; }
                    }
                }
                if (cta_rank == 0) {
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                }
                if (cta_rank == 0) {
                    elect_commit_cg2_multicast(o_done_addr, (uint16_t)(3));
                }
                mma_tile_cursor = mma_tile_cursor + num_kv_tiles_3;
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
    // ---- Role: empty1 ----
    if (warp >= 14 && warp <= 15) {
        { // empty1_main
            const int wg2_dummy_2 = 0;
            unsigned int _phase_q_full_0_4 = 0;
            {
                #pragma unroll 1
                for (unsigned int work_idx_5 = bid / 2; work_idx_5 < total_work_items; work_idx_5 += num_bids / 2) {
                    mbarrier_wait(q_full_addr, _phase_q_full_0_4);
                    _phase_q_full_0_4 ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

