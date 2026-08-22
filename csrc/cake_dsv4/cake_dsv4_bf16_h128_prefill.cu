typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };

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
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_K_OFF 66560
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_V_OFF 115712
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P_OFF 181248
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_INDICES_OFF 214016
#define SMEM_SMEM_INDICES_STAGE_BYTES 4608
#define SMEM_SMEM_INDICES_STRIDE 4608
#define SMEM_SMEM_SOFTMAX_EXCHANGE_OFF 218624
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STRIDE 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_OFF 219136
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STRIDE 512
#define SMEM_TOTAL 219648
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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_bf16_h128_prefill(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_k, const __grid_constant__ CUtensorMap tmap_compressed_k, const __grid_constant__ CUtensorMap tmap_swa_v, const __grid_constant__ CUtensorMap tmap_compressed_v, __nv_bfloat16* __restrict__ O, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items)
{
    // PTX global compiler scheduling controls
    asm volatile(".pragma \"global knob ForceLateCommoning=1\";\n" : : : "memory");
    asm volatile(".pragma \"global knob HoistLate=3\";\n" : : : "memory");
    asm volatile(".pragma \"global knob MbarrierInitRegMapping=1\";\n" : : : "memory");

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

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_k_addr = smem + 66560;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int smem_v_addr = smem + 115712;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int smem_p_addr = smem + 181248;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 214016);
    const int smem_indices_addr = smem + 214016;
    float* smem_softmax_exchange = reinterpret_cast<float*>(smem_raw + 218624);
    const int smem_softmax_exchange_addr = smem + 218624;
    float* smem_epilogue_exchange = reinterpret_cast<float*>(smem_raw + 219136);
    const int smem_epilogue_exchange_addr = smem + 219136;

    // Mbarrier init (16 groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // v_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 216, 128);
            mbarrier_init(smem + 224, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            // o_first_slice_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 240, 256);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 248, 32);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
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
    #define k_empty_addr (mbar_base + 40)
    #define v_full_addr (mbar_base + 64)
    #define v_empty_addr (mbar_base + 96)
    #define s_full_addr (mbar_base + 128)
    #define s_empty_addr (mbar_base + 144)
    #define p_full_addr (mbar_base + 160)
    #define p_empty_addr (mbar_base + 176)
    #define o_empty_addr (mbar_base + 192)
    #define stats_addr (mbar_base + 200)
    #define stats_empty_addr (mbar_base + 216)
    #define o_full_addr (mbar_base + 232)
    #define o_first_slice_seeded_addr (mbar_base + 240)
    #define tmem_dealloc_peer_addr (mbar_base + 248)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_wg_main
            const int softmax_dummy = 0;
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            const int head_row_base = warp % 2 * 32;
            const int score_row_base = warp % 4 * 32;
            const int n_half = warp % 4 / 2;
            const int my_row = head_row_base + lane;
            const int exchange_idx = n_half * 64 + my_row;
            int tile_cursor = 0;
            float seed_zero[4];
            #pragma unroll
            for (int seed_i = 0; seed_i < 4; seed_i++) {
                seed_zero[seed_i] = 0.0f;
            }
            #pragma unroll
            for (int seed_stage = 0; seed_stage < 2; seed_stage++) {
                #pragma unroll
                for (int seed_col = 128 + seed_stage * 128; seed_col < 128 + (seed_stage + 1) * 128; seed_col += 4) {
                    int seed_addr = taddr + (unsigned int)seed_col + (unsigned int)(score_row_base << 16);
                    tmem_st_x4_f32(seed_addr, seed_zero);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(o_first_slice_seeded_addr);
            int peer_rank = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(o_first_slice_seeded_addr), "r"(peer_rank) : "memory");
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work_items; work_idx += num_clusters) {
                int query_idx = work_idx;
                int active_topk = sparse_topk_lens[query_idx];
                int num_kv_tiles = (sparse_topk + 128 - 1) / 128;
                int head_idx = cta_rank * 64 + my_row;
                float row_max_val = -CAKE_INF;
                float row_sum_val = 0.0f;
                if (has_sinks != 0 && head_idx < num_heads) {
                    row_max_val = sinks[head_idx] * 1.4426950408889634f / softmax_scale_log2;
                    if (n_half == 0) {
                        row_sum_val = 1.0f;
                    }
                }
                #pragma unroll 1
                for (int tile = 0; tile < num_kv_tiles; tile++) {
                    int pipeline_tile = tile_cursor + tile;
                    int score_phase = pipeline_tile & 1;
                    int s_full_phase = pipeline_tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (score_phase) * 8, s_full_phase);
                    int score_offset = ((score_phase != 0) ? 64 : 0);
                    int score_base = taddr + (unsigned int)score_offset + (unsigned int)(score_row_base << 16);
                    float _tmem_load_0[64];
                    tmem_ld_x32(&_tmem_load_0[0], score_base);
                    tmem_ld_x32(&_tmem_load_0[32], score_base + 32);
                    int valid_cols = active_topk - tile * 128 - n_half * 64;
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                    if (valid_cols > 64) {
                        valid_cols = 64;
                    }
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
                        if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_0[0 + _i_1] = -CAKE_INF;
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
                        if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_0[32 + _i_3] = -CAKE_INF;
                    }
                    float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_4);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_4);
                    float _max_0 = max_noftz(_tmem_load_0_max, row_max_val);
                    float new_max = _max_0;
                    smem_softmax_exchange[exchange_idx] = new_max;
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    float _max_1 = max_noftz(new_max, smem_softmax_exchange[exchange_idx ^ 64]);
                    new_max = _max_1;
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    float no_correction = (((new_max - row_max_val) * softmax_scale_log2 <= 0.0f) ? 1.0f : 0.0f);
                    float delta = softmax_scale_log2 * (row_max_val - new_max);
                    float _exp2_0 = approx_exp2(delta);
                    float exp_delta = _exp2_0;
                    float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta : 1.0f);
                    row_max_val = new_max;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_6 = {-max_scaled, -max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_5, _fma_c2_6);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_7 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_7);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_7);
                    float _tmem_load_0_sum = _reg_reduce_sum2_7.x + _reg_reduce_sum2_7.y;
                    row_sum_val = row_sum_val * acc_scale + _tmem_load_0_sum;
                    int stats_empty_phase = pipeline_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(stats_empty_addr + (score_phase) * 8, stats_empty_phase);
                    float meta[4];
                    meta[0] = row_sum_val;
                    meta[1] = row_max_val;
                    meta[2] = acc_scale;
                    meta[3] = no_correction;
                    int meta_addr = taddr + 384 + (unsigned int)(score_phase * 8) + (unsigned int)(n_half * 4) + (unsigned int)(head_row_base << 16);
                    tmem_st_x4_f32(meta_addr, meta);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(stats_addr + (score_phase) * 8);
                    int p_stage = pipeline_tile % 2;
                    int p_empty_phase = pipeline_tile / 2 & 1 ^ 1;
                    mbarrier_wait(p_empty_addr + (p_stage) * 8, p_empty_phase);
                    uint32_t _tmem_load_0_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_p_stage = p_stage * 2 + n_half;
                    int p_base_smem = smem_p_addr + (unsigned int)(smem_p_stage * 8192);
                    #pragma unroll
                    for (int vec = 0; vec < 8; vec++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_smem + (my_row * 128 + vec * 16 ^ (my_row * 128 + vec * 16 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[vec * 4]), "r"(_tmem_load_0_bf16[vec * 4 + 1]), "r"(_tmem_load_0_bf16[vec * 4 + 2]), "r"(_tmem_load_0_bf16[vec * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (unsigned int)(p_stage * 8)), "r"(0) : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr + (unsigned int)(score_phase * 8)), "r"(0) : "memory");
                }
                tile_cursor = tile_cursor + num_kv_tiles;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_wg_main
            const int correction_dummy = 0;
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            const int head_row_base_1 = warp % 2 * 32;
            const int output_row_base = warp % 4 * 32;
            const int n_half_1 = warp % 4 / 2;
            const int my_row_1 = head_row_base_1 + lane;
            const int corr_row = head_row_base_1 << 16;
            int tile_cursor_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = cluster_id; work_idx_1 < total_work_items; work_idx_1 += num_clusters) {
                int query_idx_1 = work_idx_1;
                int num_kv_tiles_1 = (sparse_topk + 128 - 1) / 128;
                float final_sum = 0.0f;
                float final_max = -CAKE_INF;
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                    int pipeline_tile_1 = tile_cursor_1 + tile_1;
                    int score_phase_1 = pipeline_tile_1 & 1;
                    int stats_phase = pipeline_tile_1 >> 1 & 1;
                    mbarrier_wait(stats_addr + (score_phase_1) * 8, stats_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int meta_addr_1 = taddr + 384 + (unsigned int)(score_phase_1 * 8) + (unsigned int)(n_half_1 * 4) + (unsigned int)corr_row;
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], meta_addr_1);
                    final_sum = _tmem_load_1[0];
                    final_max = _tmem_load_1[1];
                    float acc_scale_1 = _tmem_load_1[2];
                    float no_correction_1 = _tmem_load_1[3];
                    if (tile_1 > 0) {
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _vote_0 = __all_sync(0xFFFFFFFF, no_correction_1 == 1.0f);
                        int skip_correction = _vote_0;
                        if (skip_correction == 0) {
                            #pragma unroll
                            for (int local_slice = 0; local_slice < 2; local_slice++) {
                                #pragma unroll
                                for (int acc_stage = 0; acc_stage < 2; acc_stage++) {
                                    int o_base = taddr + 128 + (unsigned int)(acc_stage * 128) + (unsigned int)(local_slice * 64) + (unsigned int)(output_row_base << 16);
                                    #pragma unroll
                                    for (int c = 0; c < 64; c += 16) {
                                        float _tmem_load_2[16];
                                        tmem_ld_x16(&_tmem_load_2[0], o_base + c);
                                        const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 8; _ls++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                                        tmem_st_x16_f32(o_base + c, _tmem_load_2);
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_empty_addr), "r"(0) : "memory");
                    }
                    mbarrier_arrive(stats_empty_addr + (score_phase_1) * 8);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                const int exchange_idx_1 = n_half_1 * 64 + my_row_1;
                smem_epilogue_exchange[exchange_idx_1] = final_sum;
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                final_sum = final_sum + smem_epilogue_exchange[exchange_idx_1 ^ 64];
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                float _rcp_0 = approx_rcp(final_sum);
                float inv_sum = ((final_sum > 0.0f) ? _rcp_0 : 0.0f);
                int head_idx_1 = cta_rank * 64 + my_row_1;
                int output_offset = (query_idx_1 * num_heads + head_idx_1) * 512;
                #pragma unroll
                for (int acc_stage_1 = 0; acc_stage_1 < 2; acc_stage_1++) {
                    #pragma unroll
                    for (int local_slice_1 = 0; local_slice_1 < 2; local_slice_1++) {
                        int logical_slice = acc_stage_1 * 2 * 2 + n_half_1 * 2 + local_slice_1;
                        int o_base_1 = taddr + 128 + (unsigned int)(acc_stage_1 * 128) + (unsigned int)(local_slice_1 * 64) + (unsigned int)(output_row_base << 16);
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 64; c_1 += 32) {
                            float _tmem_load_3[32];
                            tmem_ld_x32(&_tmem_load_3[0], o_base_1 + c_1);
                            int gmem_base = output_offset + logical_slice * 64 + c_1;
                            #pragma unroll
                            for (int j = 0; j < 32; j += 8) {
                                {
                                    const float2 _prescale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[j])[_ps], _prescale2_1);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_3[j + _ps] *= inv_sum * output_scale;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_3[j + 0], _tmem_load_3[j + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_3[j + 2], _tmem_load_3[j + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_3[j + 4], _tmem_load_3[j + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_3[j + 6], _tmem_load_3[j + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (gmem_base + j)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_empty_addr), "r"(0) : "memory");
                tile_cursor_1 = tile_cursor_1 + num_kv_tiles_1;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int mma_dummy = 0;
            unsigned int k_stage = 0;
            unsigned int v_stage = 0;
            int tile_cursor_2 = 0;
            unsigned int _phase_o_first_slice_seeded_0 = 0;
            mbarrier_wait(o_first_slice_seeded_addr, _phase_o_first_slice_seeded_0);
            _phase_o_first_slice_seeded_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_v_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = cluster_id; work_idx_2 < total_work_items; work_idx_2 += num_clusters) {
                int num_kv_tiles_2 = (sparse_topk + 128 - 1) / 128;
                if (cta_rank == 0) {
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                        int pipeline_tile_2 = tile_cursor_2 + tile_2;
                        int score_phase_2 = pipeline_tile_2 & 1;
                        int s_empty_phase = pipeline_tile_2 >> 1 & 1 ^ 1;
                        mbarrier_wait(s_empty_addr + (score_phase_2) * 8, s_empty_phase);
                        int score_col = ((score_phase_2 != 0) ? 64 : 0);
                        #pragma unroll
                        for (int qk_stage = 0; qk_stage < 4; qk_stage++) {
                            mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                            int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (qk_stage * 2) * 512;
                            int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (score_col))), "r"(((qk_stage == 0) ? 0 : 1)));
                            int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (qk_stage * 2 + 1) * 512;
                            int _mma_b_lo_1 = (((smem_k_addr + 8192) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 3) { k_stage = 0; _phase_k_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(s_full_addr + (score_phase_2) * 8, (uint16_t)(3));
                        if (tile_2 > 0) {
                            int prev_pipeline_tile = pipeline_tile_2 - 1;
                            int p_stage_1 = prev_pipeline_tile % 2;
                            int p_full_phase = prev_pipeline_tile / 2 & 1;
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            _phase_o_empty_0 ^= 1;
                            mbarrier_wait(p_full_addr + (p_stage_1) * 8, p_full_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            #pragma unroll
                            for (int ps = 0; ps < 2; ps++) {
                                int smem_p_stage_1 = p_stage_1 * 2 + ps;
                                #pragma unroll
                                for (int acc_stage_2 = 0; acc_stage_2 < 2; acc_stage_2++) {
                                    mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                                    int output_col = 128 + acc_stage_2 * 128;
                                    int _mma_a_lo_2 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_1) * 512;
                                    int _mma_b_lo_2 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 1024;
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
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (output_col))), "r"(((((ps == 0) ? first_pv : 0)) ? 0 : 1)));
                                    elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                                    v_stage += 1;
                                    if (v_stage == 4) { v_stage = 0; _phase_v_full ^= 1; }
                                }
                            }
                            first_pv = 0;
                            elect_commit_cg2_multicast(p_empty_addr + (p_stage_1) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        }
                    }
                    int last_pipeline_tile = tile_cursor_2 + num_kv_tiles_2 - 1;
                    int last_p_stage = last_pipeline_tile % 2;
                    int last_p_phase = last_pipeline_tile / 2 & 1;
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    mbarrier_wait(p_full_addr + (last_p_stage) * 8, last_p_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int ps_1 = 0; ps_1 < 2; ps_1++) {
                        int smem_p_stage_2 = last_p_stage * 2 + ps_1;
                        #pragma unroll
                        for (int acc_stage_3 = 0; acc_stage_3 < 2; acc_stage_3++) {
                            mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                            int output_col_1 = 128 + acc_stage_3 * 128;
                            int _mma_a_lo_3 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_2) * 512;
                            int _mma_b_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 1024;
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
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_scratch + (output_col_1))), "r"(((((ps_1 == 0) ? first_pv : 0)) ? 0 : 1)));
                            elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                            v_stage += 1;
                            if (v_stage == 4) { v_stage = 0; _phase_v_full ^= 1; }
                        }
                    }
                    elect_commit_cg2_multicast(p_empty_addr + (last_p_stage) * 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                }
                tile_cursor_2 = tile_cursor_2 + num_kv_tiles_2;
            }
            if (cta_rank == 0) {
                #pragma unroll
                for (int tail_offset = 0; tail_offset < 2; tail_offset++) {
                    int tail_tile = tile_cursor_2 + tail_offset;
                    int tail_stage = tail_tile & 1;
                    int tail_phase = tail_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(s_empty_addr + (tail_stage) * 8, tail_phase);
                }
                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                _phase_o_empty_0 ^= 1;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            int peer_rank_1 = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_dealloc_peer_addr), "r"(peer_rank_1) : "memory");
            mbarrier_wait(tmem_dealloc_peer_addr, 0);
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_wg ----
    if (warp >= 9 && warp <= 12) {
        { // load_wg_main
            const int load_dummy = 0;
            const int load_warp_rank = warp - 9;
            unsigned int k_stage_1 = 0;
            unsigned int v_stage_1 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int work_idx_3 = cluster_id; work_idx_3 < total_work_items; work_idx_3 += num_clusters) {
                int query_idx_2 = work_idx_3;
                int num_kv_tiles_3 = (sparse_topk + 128 - 1) / 128;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (load_warp_rank == 0) {
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(131072)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        #pragma unroll
                        for (int q_stage = 0; q_stage < 8; q_stage++) {
                            tma_4d_gmem2smem_cta2(smem_q_addr + (unsigned int)(q_stage * 8192), (&tmap_q), 0, cta_rank * 64, q_stage, query_idx_2, ((q_full_addr) & 0xFEFFFFFF));
                        }
                    }
                }
                asm volatile("barrier.sync 2, 160;" ::: "memory");
                #pragma unroll
                for (int qk_stage_1 = 0; qk_stage_1 < 4; qk_stage_1++) {
                    mbarrier_wait(k_empty_addr + (k_stage_1) * 8, _phase_k_empty);
                    if (load_warp_rank == 0) {
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            }
                        }
                    }
                    int group = load_warp_rank * 4 + lane;
                    if (lane < 4) {
                        int index_offset = cta_rank * 64 + group * 4;
                        int raw_rows[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                            : "r"(smem_indices_addr + (unsigned int)(index_offset * 4)));
                        tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group * 512), (&tmap_swa_k), qk_stage_1 * 128, ((raw_rows[0] >= 0) ? raw_rows[0] : 0), ((raw_rows[1] >= 0) ? raw_rows[1] : 0), ((raw_rows[2] >= 0) ? raw_rows[2] : 0), ((raw_rows[3] >= 0) ? raw_rows[3] : 0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                        tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group * 512) + 8192, (&tmap_swa_k), qk_stage_1 * 128 + 64, ((raw_rows[0] >= 0) ? raw_rows[0] : 0), ((raw_rows[1] >= 0) ? raw_rows[1] : 0), ((raw_rows[2] >= 0) ? raw_rows[2] : 0), ((raw_rows[3] >= 0) ? raw_rows[3] : 0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                    }
                    k_stage_1 += 1;
                    if (k_stage_1 == 3) { k_stage_1 = 0; _phase_k_empty ^= 1; }
                }
                #pragma unroll 1
                for (int tile_3 = 1; tile_3 < num_kv_tiles_3; tile_3++) {
                    #pragma unroll
                    for (int qk_stage_2 = 0; qk_stage_2 < 4; qk_stage_2++) {
                        mbarrier_wait(k_empty_addr + (k_stage_1) * 8, _phase_k_empty);
                        if (load_warp_rank == 0) {
                            if (cta_rank == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                }
                            }
                        }
                        int group_1 = load_warp_rank * 4 + lane;
                        if (lane < 4) {
                            int index_offset_1 = tile_3 * 128 + cta_rank * 64 + group_1 * 4;
                            int raw_rows_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                                : "r"(smem_indices_addr + (unsigned int)(index_offset_1 * 4)));
                            tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_1 * 512), (&tmap_compressed_k), qk_stage_2 * 128, ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : 0), ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : 0), ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : 0), ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : 0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                            tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_1 * 512) + 8192, (&tmap_compressed_k), qk_stage_2 * 128 + 64, ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : 0), ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : 0), ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : 0), ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : 0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                        }
                        k_stage_1 += 1;
                        if (k_stage_1 == 3) { k_stage_1 = 0; _phase_k_empty ^= 1; }
                    }
                    int prev_tile = tile_3 - 1;
                    #pragma unroll
                    for (int ps_2 = 0; ps_2 < 2; ps_2++) {
                        #pragma unroll
                        for (int acc_stage_4 = 0; acc_stage_4 < 2; acc_stage_4++) {
                            mbarrier_wait(v_empty_addr + (v_stage_1) * 8, _phase_v_empty);
                            if (load_warp_rank == 0) {
                                if (cta_rank == 0) {
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                    }
                                }
                            }
                            int group_2 = load_warp_rank * 4 + lane;
                            if (lane < 4) {
                                int index_offset_2 = prev_tile * 128 + ps_2 * 64 + group_2 * 4;
                                int raw_rows_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 3]))
                                    : "r"(smem_indices_addr + (unsigned int)(index_offset_2 * 4)));
                                if (prev_tile == 0) {
                                    tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_2 * 512), (&tmap_swa_v), acc_stage_4 * 128 * 2 + cta_rank * 128, ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : 0), ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : 0), ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : 0), ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_2 * 512) + 8192, (&tmap_swa_v), acc_stage_4 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : 0), ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : 0), ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : 0), ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                } else {
                                    tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_2 * 512), (&tmap_compressed_v), acc_stage_4 * 128 * 2 + cta_rank * 128, ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : 0), ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : 0), ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : 0), ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_2 * 512) + 8192, (&tmap_compressed_v), acc_stage_4 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : 0), ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : 0), ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : 0), ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                }
                            }
                            v_stage_1 += 1;
                            if (v_stage_1 == 4) { v_stage_1 = 0; _phase_v_empty ^= 1; }
                        }
                    }
                }
                int last_tile = num_kv_tiles_3 - 1;
                #pragma unroll
                for (int ps_3 = 0; ps_3 < 2; ps_3++) {
                    #pragma unroll
                    for (int acc_stage_5 = 0; acc_stage_5 < 2; acc_stage_5++) {
                        mbarrier_wait(v_empty_addr + (v_stage_1) * 8, _phase_v_empty);
                        if (load_warp_rank == 0) {
                            if (cta_rank == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                }
                            }
                        }
                        int group_3 = load_warp_rank * 4 + lane;
                        if (lane < 4) {
                            int index_offset_3 = last_tile * 128 + ps_3 * 64 + group_3 * 4;
                            int raw_rows_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 3]))
                                : "r"(smem_indices_addr + (unsigned int)(index_offset_3 * 4)));
                            if (last_tile == 0) {
                                tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_3 * 512), (&tmap_swa_v), acc_stage_5 * 128 * 2 + cta_rank * 128, ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : 0), ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : 0), ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : 0), ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_3 * 512) + 8192, (&tmap_swa_v), acc_stage_5 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : 0), ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : 0), ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : 0), ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                            } else {
                                tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_3 * 512), (&tmap_compressed_v), acc_stage_5 * 128 * 2 + cta_rank * 128, ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : 0), ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : 0), ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : 0), ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                tma_gather4_gmem2smem_mc_cta2(smem_v_addr + v_stage_1 * 16384 + (unsigned int)(group_3 * 512) + 8192, (&tmap_compressed_v), acc_stage_5 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : 0), ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : 0), ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : 0), ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : 0), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                            }
                        }
                        v_stage_1 += 1;
                        if (v_stage_1 == 4) { v_stage_1 = 0; _phase_v_empty ^= 1; }
                    }
                }
                asm volatile("barrier.sync 2, 160;" ::: "memory");
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
            _phase_q_empty_0 ^= 1;
        }
    }
    // ---- Role: index_warp ----
    if (warp == 13) {
        { // index_warp_main
            const int index_dummy = 0;
            #pragma unroll 1
            for (unsigned int work_idx_4 = cluster_id; work_idx_4 < total_work_items; work_idx_4 += num_clusters) {
                int sparse_base = work_idx_4 * (unsigned int)sparse_topk;
                #pragma unroll
                for (int index_pass = 0; index_pass < 9; index_pass++) {
                    int index_offset_4 = index_pass * 128 + lane * 4;
                    int values[4];
                    int _vec_load_0[4];
                    {
                        int4 _iv4 = *reinterpret_cast<const int4*>(sparse_indices + (sparse_base + index_offset_4) + 0);
                        _vec_load_0[0 + 0] = _iv4.x;
                        _vec_load_0[0 + 1] = _iv4.y;
                        _vec_load_0[0 + 2] = _iv4.z;
                        _vec_load_0[0 + 3] = _iv4.w;
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        values[i] = _vec_load_0[i];
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + (unsigned int)(index_offset_4 * 4)), "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 2, 160;" ::: "memory");
                asm volatile("barrier.sync 2, 160;" ::: "memory");
            }
        }
    }
    // ---- Role: empty_warps ----
    if (warp >= 14 && warp <= 15) {
        { // empty_warps_main
            const int empty_dummy = 0;
            #pragma unroll 1
            for (unsigned int work_idx_5 = cluster_id; work_idx_5 < total_work_items; work_idx_5 += num_clusters) {
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
        }
    }

    // Cleanup
}

} // extern "C"

