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
#define TMEM_SCORES0_OFFSET 0
#define TMEM_SCORES1_OFFSET 128
#define TMEM_OUTPUT0_OFFSET 256
#define TMEM_OUTPUT1_OFFSET 384
#define NUM_KV_PIPE_STAGES 3
#define SMEM_Q0_SMEM_OFF 1024
#define SMEM_Q0_SMEM_STAGE_BYTES 32768
#define SMEM_Q0_SMEM_STRIDE 32768
#define SMEM_KV_SMEM_OFF 33792
#define SMEM_KV_SMEM_STAGE_BYTES 32768
#define SMEM_KV_SMEM_STRIDE 32768
#define SMEM_V_SMEM_OFF 33792
#define SMEM_V_SMEM_STAGE_BYTES 32768
#define SMEM_V_SMEM_STRIDE 32768
#define SMEM_SCALE_SMEM_OFF 132096
#define SMEM_SCALE_SMEM_STAGE_BYTES 3072
#define SMEM_SCALE_SMEM_STRIDE 3072
#define SMEM_UNION_COUNT_SMEM_OFF 135168
#define SMEM_UNION_COUNT_SMEM_STAGE_BYTES 4
#define SMEM_UNION_COUNT_SMEM_STRIDE 4
#define SMEM_UNION_BLOCKS_OFF 135172
#define SMEM_UNION_BLOCKS_STAGE_BYTES 128
#define SMEM_UNION_BLOCKS_STRIDE 128
#define SMEM_TOTAL 135424
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_flashinfer_blackwell_vsa_ultrasparse_bsr_sm100(CakeTensorMap const* q, CakeTensorMap const* k, CakeTensorMap const* v, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int* __restrict__ bsr_indices, int mb, int nb, int selected_blocks, int total_tiles, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float lse_temperature_scale, int return_softmax_lse, int return_temperature_lse)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* q0_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q0_smem_addr = smem + 1024;
    __nv_bfloat16* kv_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kv_smem_addr = smem + 33792;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_smem_addr = smem + 33792;
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 132096);
    const int scale_smem_addr = smem + 132096;
    int* union_count_smem = reinterpret_cast<int*>(smem_raw + 135168);
    const int union_count_smem_addr = smem + 135168;
    int* union_blocks = reinterpret_cast<int*>(smem_raw + 135172);
    const int union_blocks_addr = smem + 135172;

    // Mbarrier init (11 groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 8, 128);
            // union_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 16, 32);
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            mbarrier_init(smem + 128, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // tile_done: 1 barriers, init_count=128
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
    #define q_empty_addr (mbar_base + 8)
    #define union_ready_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 24)
    #define kv_empty_addr (mbar_base + 48)
    #define s_full_addr (mbar_base + 72)
    #define p_full_addr (mbar_base + 88)
    #define corr_sig_addr (mbar_base + 104)
    #define corr_done_addr (mbar_base + 120)
    #define o_full_addr (mbar_base + 136)
    #define tile_done_addr (mbar_base + 152)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores0 = taddr;
    const int tmem_scores1 = taddr + 128;
    const int tmem_output0 = taddr + 256;
    const int tmem_output1 = taddr + 384;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            unsigned int _phase_union_ready_0 = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_corr_done_0 = 0;
            unsigned int _phase_corr_done_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < total_tiles; tile_idx += num_bids) {
                int batch = 0;
                int q_tile = tile_idx % (unsigned int)mb;
                int q_head = tile_idx / (unsigned int)mb;
                int kv_head = q_head;
                int q_local_base = q_tile * 128;
                int q_valid = 128;
                if (q_tile >= mb) {
                    q_valid = 0;
                }
                int query_base = q_local_base;
                int k_start = 0;
                int kv_len = nb * 128;
                int query_offset = 0;
                int num_n_blocks = nb;
                mbarrier_wait(union_ready_addr, _phase_union_ready_0);
                _phase_union_ready_0 ^= 1;
                int n_stage = make_warp_uniform(warp / 4);
                const int warp_in_stage = warp % 4;
                int stage_tmem_offset = make_warp_uniform(n_stage * 128);
                int stage_row_offset = make_warp_uniform(n_stage * 128);
                const int tmem_row_origin = warp_in_stage * 32;
                int my_row = tmem_row_origin + lane;
                int row_addr = tmem_row_origin << 16;
                int row_valid = ((my_row < q_valid) ? 1 : 0);
                float row_max = -CAKE_INF;
                float row_sum = 0.0f;
                #pragma unroll
                for (int pair_index = 0; pair_index < 3; pair_index++) {
                    int union_index = 5 - n_stage - 2 * pair_index;
                    int n_block = union_blocks[union_index];
                    if (n_stage == 0) {
                        mbarrier_wait(s_full_addr, _phase_s_full_0);
                        _phase_s_full_0 ^= 1;
                    } else {
                        mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                        _phase_s_full_1 ^= 1;
                    }
                    int valid_cols = 0;
                    if (row_valid != 0) {
                        valid_cols = kv_len - n_block * 128;
                        if (valid_cols > 128) {
                            valid_cols = 128;
                        }
                        if (valid_cols < 0) {
                            valid_cols = 0;
                        }
                    }
                    int score_addr = taddr + (unsigned int)stage_tmem_offset + (unsigned int)row_addr;
                    float _tmem_load_0[128];
                    tmem_ld_x32(&_tmem_load_0[0], score_addr);
                    tmem_ld_x32(&_tmem_load_0[32], score_addr + 32);
                    tmem_ld_x32(&_tmem_load_0[64], score_addr + 64);
                    tmem_ld_x32(&_tmem_load_0[96], score_addr + 96);
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
                            if (!(_slice_lo_mask_2 & (1u << _i_5))) _tmem_load_0[64 + _i_5] = -CAKE_INF;
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
                            if (!(_slice_lo_mask_3 & (1u << _i_7))) _tmem_load_0[96 + _i_7] = -CAKE_INF;
                        }
                    }
                    float2 _reg_reduce_max2_8 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_8);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_8);
                    row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_8);
                    row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_8);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_8);
                    float tile_max = _tmem_load_0_max;
                    float _max_0 = max_noftz(tile_max, row_max);
                    float new_max = _max_0;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float new_max_scaled = safe_max * softmax_scale_log2;
                    float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                    float acc_scale_log2 = _fma_0;
                    float acc_scale;
                    float selected_max;
                    if (acc_scale_log2 >= -8.0f) {
                        selected_max = row_max;
                        safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                        acc_scale = 1.0f;
                        new_max_scaled = safe_max * softmax_scale_log2;
                    } else {
                        selected_max = new_max;
                        float _exp2_0 = approx_exp2(acc_scale_log2);
                        acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                    }
                    row_max = selected_max;
                    scale_smem[stage_row_offset + my_row] = acc_scale;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (n_stage == 0) {
                        mbarrier_arrive(corr_sig_addr);
                    } else {
                        mbarrier_arrive(corr_sig_addr + 8);
                    }
                    const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_10 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_9, _fma_c2_10);
                    int p_addr = taddr + (unsigned int)stage_tmem_offset + 64 + (unsigned int)row_addr;
                    uint32_t _tmem_load_0_bf16[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[0], _tmem_load_0_bf16, 0);
                    tmem_st_x16(p_addr, _tmem_load_0_bf16);
                    uint32_t _tmem_load_0_bf16_0[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[32], _tmem_load_0_bf16_0, 0);
                    tmem_st_x16(p_addr + 16, _tmem_load_0_bf16_0);
                    uint32_t _tmem_load_0_bf16_1[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[64], _tmem_load_0_bf16_1, 0);
                    tmem_st_x16(p_addr + 32, _tmem_load_0_bf16_1);
                    uint32_t _tmem_load_0_bf16_2[16];
                    softmax_frag_exp2_cast(&_tmem_load_0[96], _tmem_load_0_bf16_2, 0);
                    tmem_st_x16(p_addr + 48, _tmem_load_0_bf16_2);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (n_stage == 0) {
                        mbarrier_arrive(p_full_addr);
                        mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                        _phase_corr_done_0 ^= 1;
                    } else {
                        mbarrier_arrive(p_full_addr + 8);
                        mbarrier_wait(corr_done_addr + 8, _phase_corr_done_1);
                        _phase_corr_done_1 ^= 1;
                    }
                    float2 _reg_reduce_sum2_11 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_11);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_11);
                    softmax_block_sum(&_tmem_load_0[64], &_reg_reduce_sum2_11);
                    softmax_block_sum(&_tmem_load_0[96], &_reg_reduce_sum2_11);
                    float _tmem_load_0_sum = _reg_reduce_sum2_11.x + _reg_reduce_sum2_11.y;
                    row_sum = row_sum * acc_scale + _tmem_load_0_sum;
                }
                scale_smem[256 + stage_row_offset + my_row] = row_sum;
                scale_smem[512 + stage_row_offset + my_row] = row_max;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (n_stage == 0) {
                    mbarrier_arrive(corr_sig_addr);
                } else {
                    mbarrier_arrive(corr_sig_addr + 8);
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // correction_main
            unsigned int _phase_union_ready_0_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = bid; tile_idx_1 < total_tiles; tile_idx_1 += num_bids) {
                int batch_1 = 0;
                int q_tile_1 = tile_idx_1 % (unsigned int)mb;
                int q_head_1 = tile_idx_1 / (unsigned int)mb;
                int kv_head_1 = q_head_1;
                int q_local_base_1 = q_tile_1 * 128;
                int q_valid_1 = 128;
                if (q_tile_1 >= mb) {
                    q_valid_1 = 0;
                }
                int query_base_1 = q_local_base_1;
                int k_start_1 = 0;
                int kv_len_1 = nb * 128;
                int query_offset_1 = 0;
                int num_n_blocks_1 = nb;
                mbarrier_wait(union_ready_addr, _phase_union_ready_0_1);
                _phase_union_ready_0_1 ^= 1;
                const int warp_in_role = warp - 8;
                const int tmem_row_origin_1 = warp_in_role * 32;
                int my_row_1 = tmem_row_origin_1 + lane;
                int row_addr_1 = tmem_row_origin_1 << 16;
                mbarrier_arrive(p_full_addr);
                mbarrier_arrive(p_full_addr + 8);
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                mbarrier_arrive(corr_done_addr + 8);
                #pragma unroll
                for (int _pair_index = 1; _pair_index < 3; _pair_index++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float acc_scale0 = scale_smem[my_row_1];
                    int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale0 < 1.0f);
                    if (_vote_0 != 0) {
                        #pragma unroll
                        for (int cr_col = 0; cr_col < 8; cr_col++) {
                            int cr_addr0 = taddr + 256 + (unsigned int)row_addr_1 + (unsigned int)(cr_col * 16);
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], cr_addr0);
                            const float2 _scale2_0 = {acc_scale0, acc_scale0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                            tmem_st_x16_f32(cr_addr0, _tmem_load_1);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    float acc_scale1 = scale_smem[128 + my_row_1];
                    int _vote_1 = __any_sync(0xFFFFFFFF, acc_scale1 < 1.0f);
                    if (_vote_1 != 0) {
                        #pragma unroll
                        for (int cr_col_1 = 0; cr_col_1 < 8; cr_col_1++) {
                            int cr_addr1 = taddr + 256 + 128 + (unsigned int)row_addr_1 + (unsigned int)(cr_col_1 * 16);
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], cr_addr1);
                            const float2 _scale2_1 = {acc_scale1, acc_scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_1);
                            tmem_st_x16_f32(cr_addr1, _tmem_load_2);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr + 8);
                    mbarrier_arrive(corr_done_addr + 8);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                _phase_o_full_1 ^= 1;
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_sum0 = scale_smem[256 + my_row_1];
                float final_sum1 = scale_smem[384 + my_row_1];
                float final_max0 = scale_smem[512 + my_row_1];
                float final_max1 = scale_smem[640 + my_row_1];
                int valid0 = ((final_sum0 > 0.0f && final_sum0 == final_sum0) ? 1 : 0);
                int valid1 = ((final_sum1 > 0.0f && final_sum1 == final_sum1) ? 1 : 0);
                float max0 = ((valid0 != 0) ? final_max0 : -CAKE_INF);
                float max1 = ((valid1 != 0) ? final_max1 : -CAKE_INF);
                float _max_1 = max_noftz(max0, max1);
                float final_max = _max_1;
                float safe_max_1 = ((final_max == -CAKE_INF) ? 0.0f : final_max);
                float _exp2_1 = approx_exp2((max0 - safe_max_1) * softmax_scale_log2);
                float combine_scale0 = ((valid0 != 0) ? _exp2_1 : 0.0f);
                float _exp2_2 = approx_exp2((max1 - safe_max_1) * softmax_scale_log2);
                float combine_scale1 = ((valid1 != 0) ? _exp2_2 : 0.0f);
                float final_sum = final_sum0 * combine_scale0 + final_sum1 * combine_scale1;
                float _rcp_0 = approx_rcp(final_sum);
                float inv_sum = ((final_sum > 0.0f && final_sum == final_sum) ? _rcp_0 : 0.0f);
                float output_scale0 = combine_scale0 * inv_sum;
                float output_scale1 = combine_scale1 * inv_sum;
                int query = query_base_1 + my_row_1;
                int output_row = (query * num_q_heads + q_head_1) * 128;
                if (my_row_1 < q_valid_1) {
                    #pragma unroll
                    for (int out_col = 0; out_col < 16; out_col++) {
                        int out_addr0 = taddr + 256 + (unsigned int)row_addr_1 + (unsigned int)(out_col * 8);
                        int out_addr1 = out_addr0 + 128;
                        float _tmem_load_3[8];
                        tmem_ld_x8(&_tmem_load_3[0], out_addr0);
                        float _tmem_load_4[8];
                        tmem_ld_x8(&_tmem_load_4[0], out_addr1);
                        const float2 _scale2_2 = {output_scale0, output_scale0};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_2);
                        const float2 _scale2_3 = {output_scale1, output_scale1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_3);
                        #pragma unroll
                        for (int _la = 0; _la < 8; _la++)
                            _tmem_load_3[_la] = _tmem_load_3[_la] + _tmem_load_4[_la];
                        {
                            const float2 _prescale2_4 = {1.0f, 1.0f};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[0])[_ps], _prescale2_4);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                _tmem_load_3[0 + _ps] *= 1.0f;
                            #endif
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_3[0 + 0], _tmem_load_3[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_3[0 + 2], _tmem_load_3[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_3[0 + 4], _tmem_load_3[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_3[0 + 6], _tmem_load_3[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (output_row + out_col * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    int stat_idx = query * num_q_heads + q_head_1;
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum));
                    float final_lse = ((final_sum > 0.0f) ? final_max * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -CAKE_INF);
                    if (return_softmax_lse != 0) {
                        lse[stat_idx] = final_lse;
                    }
                    if (return_temperature_lse != 0) {
                        temperature_lse[stat_idx] = final_lse;
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(q_empty_addr);
                mbarrier_arrive(tile_done_addr);
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int _phase_union_ready_0_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_tile_done_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = bid; tile_idx_2 < total_tiles; tile_idx_2 += num_bids) {
                mbarrier_wait(union_ready_addr, _phase_union_ready_0_2);
                _phase_union_ready_0_2 ^= 1;
                unsigned int kv_stage = 0;
                unsigned int kv_phase = 0;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                int first_pv0 = 1;
                int first_pv1 = 1;
                #pragma unroll
                for (int n_stage_1 = 0; n_stage_1 < 2; n_stage_1++) {
                    unsigned int k_stage = kv_stage;
                    unsigned int k_phase = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                    if (n_stage_1 == 0) {
                        int _mma_a_lo_0 = make_warp_uniform(((q0_smem_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_0 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores0), "r"(0));
                        elect_commit(s_full_addr);
                    } else {
                        int _mma_a_lo_1 = make_warp_uniform(((q0_smem_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores1), "r"(0));
                        elect_commit(s_full_addr + 8);
                    }
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                }
                #pragma unroll
                for (int pair_index_1 = 0; pair_index_1 < 3; pair_index_1++) {
                    #pragma unroll
                    for (int n_stage_2 = 0; n_stage_2 < 2; n_stage_2++) {
                        unsigned int v_stage = kv_stage;
                        unsigned int v_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                        if (n_stage_2 == 0) {
                            mbarrier_wait(p_full_addr, _phase_p_full_0);
                            _phase_p_full_0 ^= 1;
                            int _mma_b_lo_2 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output0), "r"(_mma_b_lo_2), "r"(tmem_scores0 + 64), "r"(((first_pv0) ? 0 : 1)));
                            first_pv0 = 0;
                            if (pair_index_1 + 1 == 3) {
                                elect_commit(o_full_addr);
                            }
                        } else {
                            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                            _phase_p_full_1 ^= 1;
                            int _mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 2048);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output1), "r"(_mma_b_lo_3), "r"(tmem_scores1 + 64), "r"(((first_pv1) ? 0 : 1)));
                            first_pv1 = 0;
                            if (pair_index_1 + 1 == 3) {
                                elect_commit(o_full_addr + 8);
                            }
                        }
                        elect_commit(kv_empty_addr + (v_stage) * 8);
                        if (pair_index_1 + 1 < 3) {
                            unsigned int next_k_stage = kv_stage;
                            unsigned int next_k_phase = kv_phase;
                            kv_stage += 1;
                            if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                            mbarrier_wait(kv_full_addr + (next_k_stage) * 8, next_k_phase);
                            if (n_stage_2 == 0) {
                                int _mma_a_lo_4 = make_warp_uniform(((q0_smem_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_4 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (next_k_stage) * 2048);
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_scores0), "r"(0));
                                elect_commit(s_full_addr);
                            } else {
                                int _mma_a_lo_5 = make_warp_uniform(((q0_smem_addr) >> 4) & 0x3FFF);
                                int _mma_b_lo_5 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (next_k_stage) * 2048);
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_scores1), "r"(0));
                                elect_commit(s_full_addr + 8);
                            }
                            elect_commit(kv_empty_addr + (next_k_stage) * 8);
                        }
                    }
                }
                mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                _phase_tile_done_0 ^= 1;
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: empty ----
    if (warp >= 13 && warp <= 14) {
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = bid; tile_idx_3 < total_tiles; tile_idx_3 += num_bids) {
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                int batch_2 = 0;
                int q_tile_2 = tile_idx_3 % (unsigned int)mb;
                int q_head_2 = tile_idx_3 / (unsigned int)mb;
                int kv_head_2 = q_head_2;
                int q_local_base_2 = q_tile_2 * 128;
                int q_valid_2 = 128;
                if (q_tile_2 >= mb) {
                    q_valid_2 = 0;
                }
                int query_base_2 = q_local_base_2;
                int k_start_2 = 0;
                int kv_len_2 = nb * 128;
                int query_offset_2 = 0;
                int num_n_blocks_2 = nb;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 32768);
                    tma_4d_gmem2smem(q0_smem_addr, q, 0, query_base_2, q_head_2, 0, q_full_addr);
                }
                int q_block = query_base_2 / 128;
                if (lane < 6) {
                    int n_block_1 = bsr_indices[q_block * 6 + lane];
                    union_blocks[lane] = n_block_1;
                }
                asm volatile("barrier.sync 8, 32;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(union_ready_addr);
                unsigned int kv_stage_1 = 0;
                #pragma unroll
                for (int n_stage_3 = 0; n_stage_3 < 2; n_stage_3++) {
                    int logical_index = 5 - n_stage_3;
                    int first_block = union_blocks[logical_index];
                    int first_token = k_start_2 + first_block * 128;
                    mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                        int token0 = first_token;
                        int token1 = first_token + 64;
                        tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768, k, 0, token0, 0, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768 + 8192, k, 0, token1, 0, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768 + 16384, k, 0, token0, 1, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768 + 24576, k, 0, token1, 1, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                    }
                    kv_stage_1 += 1;
                    if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty ^= 1; }
                }
                #pragma unroll
                for (int pair_index_2 = 0; pair_index_2 < 3; pair_index_2++) {
                    #pragma unroll
                    for (int n_stage_4 = 0; n_stage_4 < 2; n_stage_4++) {
                        int logical_index_1 = 5 - n_stage_4 - 2 * pair_index_2;
                        int n_block_2 = union_blocks[logical_index_1];
                        int token_base = k_start_2 + n_block_2 * 128;
                        mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                            int token0_1 = token_base;
                            int token1_1 = token_base + 64;
                            tma_4d_gmem2smem(v_smem_addr + kv_stage_1 * 32768, v, 0, token0_1, 0, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                            tma_4d_gmem2smem(v_smem_addr + kv_stage_1 * 32768 + 8192, v, 0, token1_1, 0, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                            tma_4d_gmem2smem(v_smem_addr + kv_stage_1 * 32768 + 16384, v, 0, token0_1, 1, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                            tma_4d_gmem2smem(v_smem_addr + kv_stage_1 * 32768 + 24576, v, 0, token1_1, 1, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                        }
                        kv_stage_1 += 1;
                        if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty ^= 1; }
                        int next_logical_index = logical_index_1 - 2;
                        if (next_logical_index >= 0) {
                            int next_block = union_blocks[next_logical_index];
                            int next_token = k_start_2 + next_block * 128;
                            mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                                int token0_2 = next_token;
                                int token1_2 = next_token + 64;
                                tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768, k, 0, token0_2, 0, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768 + 8192, k, 0, token1_2, 0, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768 + 16384, k, 0, token0_2, 1, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + kv_stage_1 * 32768 + 24576, k, 0, token1_2, 1, kv_head_2, kv_full_addr + (kv_stage_1) * 8);
                            }
                            kv_stage_1 += 1;
                            if (kv_stage_1 == 3) { kv_stage_1 = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
