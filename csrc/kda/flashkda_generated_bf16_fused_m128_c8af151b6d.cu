typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) FlashKDATensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) FlashKDATensorMapPack { FlashKDATensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define FLASHKDA_INF CUDART_INF_F
#define TMEM_NCOLS 288
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_U_ACC_OFFSET 224
#define TMEM_TMEM_U2_INP_OFFSET 224
#define TMEM_TMEM_U2_ACC_OFFSET 256
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define NUM_CHUNK_PIPE_STAGES 5
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 41984
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_G_RAW_STRIDE 41984
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_G_RAW_ALL_STRIDE 176128
#define SMEM_SMEM_KD_OFF 9216
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STRIDE 41984
#define SMEM_SMEM_Q_RAW_PREFETCH_OFF 17408
#define SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES 8192
#define SMEM_SMEM_Q_RAW_PREFETCH_STRIDE 41984
#define SMEM_SMEM_FINAL_TRANS_OFF 17408
#define SMEM_SMEM_FINAL_TRANS_STAGE_BYTES 12288
#define SMEM_SMEM_FINAL_TRANS_STRIDE 41984
#define SMEM_SMEM_KR_TRANS_OFF 17408
#define SMEM_SMEM_KR_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_KR_TRANS_STRIDE 41984
#define SMEM_SMEM_MQK_TRANS_OFF 25600
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 2048
#define SMEM_SMEM_MQK_TRANS_STRIDE 41984
#define SMEM_SMEM_INV_OFF 29696
#define SMEM_SMEM_INV_STAGE_BYTES 2048
#define SMEM_SMEM_INV_STRIDE 41984
#define SMEM_SMEM_STATE_DIAG0_OFF 227328
#define SMEM_SMEM_STATE_DIAG0_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG0_STRIDE 512
#define SMEM_SMEM_STATE_DIAG1_OFF 227840
#define SMEM_SMEM_STATE_DIAG1_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG1_STRIDE 512
#define SMEM_SMEM_STATE_DIAG2_OFF 228352
#define SMEM_SMEM_STATE_DIAG2_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG2_STRIDE 512
#define SMEM_SMEM_STATE_DIAG3_OFF 228864
#define SMEM_SMEM_STATE_DIAG3_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG3_STRIDE 512
#define SMEM_SMEM_STATE_DIAG4_OFF 229376
#define SMEM_SMEM_STATE_DIAG4_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG4_STRIDE 512
#define SMEM_SMEM_STATE_DIAG5_OFF 229888
#define SMEM_SMEM_STATE_DIAG5_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG5_STRIDE 512
#define SMEM_SMEM_STATE_DIAG6_OFF 230400
#define SMEM_SMEM_STATE_DIAG6_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG6_STRIDE 512
#define SMEM_SMEM_STATE_DIAG7_OFF 230912
#define SMEM_SMEM_STATE_DIAG7_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DIAG7_STRIDE 512
#define SMEM_SMEM_V_OFF 219136
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 8192
#define SMEM_SMEM_KI_OFF 17408
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 41984
#define SMEM_SMEM_GATE_OFF 25600
#define SMEM_SMEM_GATE_STAGE_BYTES 16384
#define SMEM_SMEM_GATE_STRIDE 41984
#define SMEM_SMEM_BETA_RAW_OFF 41984
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 512
#define SMEM_SMEM_BETA_RAW_STRIDE 41984
#define SMEM_SMEM_INV_WORK_OFF 32384
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 41984
#define SMEM_SMEM_OUT_OFF 210944
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 41984
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 168452
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 168452
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 41472
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 168448
#define SMEM_SMEM_GT_ALL_OFF 31744
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_ALL_STRIDE 168448
#define SMEM_SMEM_PREP_BETA_ALL_OFF 42512
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 168064
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 168064
#define SMEM_SMEM_GATE_RATE_ALL_OFF 42640
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 167940
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 167940
#define SMEM_SMEM_V_ALL_OFF 32384
#define SMEM_SMEM_V_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_V_ALL_STRIDE 176128
#define SMEM_SMEM_GATE_ALL_OFF 25600
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 184320
#define SMEM_SMEM_GATE_ALL_STRIDE 184320
#define SMEM_TOTAL 231424
#define THREADS 1024
#define NUM_HEADS 64
#define USE_INITIAL_STATE 1
#define STORE_FINAL_STATE 1
#define SCALE_VALUE 0.08838834764831845
#define LOWER_BOUND_VALUE -5.0
#define PERSISTENT_SCHEDULE 1

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


__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
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


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
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


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(1024) void
kernel_flashkda_bf16_fused_m128(__nv_bfloat16* __restrict__ q, FlashKDATensorMap const* q_tma, __nv_bfloat16* __restrict__ k, FlashKDATensorMap const* k_tma, __nv_bfloat16* __restrict__ v, FlashKDATensorMap const* v_tma, __nv_bfloat16* __restrict__ g, FlashKDATensorMap const* g_tma, __nv_bfloat16* __restrict__ beta, FlashKDATensorMap const* beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, int* __restrict__ tile_schedule, int* __restrict__ tile_schedule_counts, int schedule_stride, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, FlashKDATensorMap const* out_tma, __nv_bfloat16* __restrict__ final_state, unsigned long long state_indices_addr, long long state_slot_stride, int use_state_indices, float* __restrict__ initial_state_f32, float* __restrict__ final_state_f32, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int mbar_base = smem;
    #define qk_full_addr (mbar_base + 0)
    #define gate_raw_full_addr (mbar_base + 40)
    #define qk_raw_full_addr (mbar_base + 80)
    #define v_full_addr (mbar_base + 120)
    #define v_free_addr (mbar_base + 128)
    #define smem_free_addr (mbar_base + 136)
    #define raw_inputs_free_addr (mbar_base + 176)
    #define state_inp_ready_addr (mbar_base + 216)
    #define state_diag_ready_addr (mbar_base + 256)
    #define old_out_ready_addr (mbar_base + 296)
    #define u_inp_ready_addr (mbar_base + 336)
    #define u2_acc_ready_addr (mbar_base + 376)
    #define u2_inp_ready_addr (mbar_base + 416)
    #define final_ready_addr (mbar_base + 456)
    #define out_empty_addr (mbar_base + 496)
    #define tmem_dealloc_ready_addr (mbar_base + 504)
    #define prep_diag_ready_addr (mbar_base + 512)
    #define prep_inv16_ready_addr (mbar_base + 552)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(g_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(beta_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_all_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_addr = smem + 9216;
    __nv_bfloat16* smem_q_raw_prefetch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_raw_prefetch_addr = smem + 17408;
    __nv_bfloat16* smem_final_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_final_trans_addr = smem + 17408;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_kr_trans_addr = smem + 17408;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_mqk_trans_addr = smem + 25600;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 29696);
    const int smem_inv_addr = smem + 29696;
    __nv_bfloat16* smem_state_diag0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 227328);
    const int smem_state_diag0_addr = smem + 227328;
    __nv_bfloat16* smem_state_diag1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 227840);
    const int smem_state_diag1_addr = smem + 227840;
    __nv_bfloat16* smem_state_diag2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 228352);
    const int smem_state_diag2_addr = smem + 228352;
    __nv_bfloat16* smem_state_diag3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 228864);
    const int smem_state_diag3_addr = smem + 228864;
    __nv_bfloat16* smem_state_diag4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 229376);
    const int smem_state_diag4_addr = smem + 229376;
    __nv_bfloat16* smem_state_diag5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 229888);
    const int smem_state_diag5_addr = smem + 229888;
    __nv_bfloat16* smem_state_diag6 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 230400);
    const int smem_state_diag6_addr = smem + 230400;
    __nv_bfloat16* smem_state_diag7 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 230912);
    const int smem_state_diag7_addr = smem + 230912;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 219136);
    const int smem_v_addr = smem + 219136;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_ki_addr = smem + 17408;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_addr = smem + 25600;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_beta_raw_addr = smem + 41984;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_inv_work_addr = smem + 32384;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_out_addr = smem + 210944;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 41984);
    const int smem_restore_factor_all_addr = smem + 41984;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_all_addr = smem + 41472;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 31744);
    const int smem_gt_all_addr = smem + 31744;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 42512);
    const int smem_prep_beta_all_addr = smem + 42512;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 42640);
    const int smem_gate_rate_all_addr = smem + 42640;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_v_all_addr = smem + 32384;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_all_addr = smem + 25600;

    // Mbarrier init (18 groups, 74 barriers)
    // Mbarriers at smem_raw[0..592)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'chunk_pipe' ---
            // qk_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // gate_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // qk_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // v_free: 1 barriers, init_count=4
            mbarrier_init(smem + 128, 4);
            // smem_free: 5 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // raw_inputs_free: 5 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // state_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 216, 4);
            mbarrier_init(smem + 224, 4);
            mbarrier_init(smem + 232, 4);
            mbarrier_init(smem + 240, 4);
            mbarrier_init(smem + 248, 4);
            // state_diag_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 256, 4);
            mbarrier_init(smem + 264, 4);
            mbarrier_init(smem + 272, 4);
            mbarrier_init(smem + 280, 4);
            mbarrier_init(smem + 288, 4);
            // old_out_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // u_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 336, 4);
            mbarrier_init(smem + 344, 4);
            mbarrier_init(smem + 352, 4);
            mbarrier_init(smem + 360, 4);
            mbarrier_init(smem + 368, 4);
            // u2_acc_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 376, 1);
            mbarrier_init(smem + 384, 1);
            mbarrier_init(smem + 392, 1);
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            // u2_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 416, 4);
            mbarrier_init(smem + 424, 4);
            mbarrier_init(smem + 432, 4);
            mbarrier_init(smem + 440, 4);
            mbarrier_init(smem + 448, 4);
            // final_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 456, 1);
            mbarrier_init(smem + 464, 1);
            mbarrier_init(smem + 472, 1);
            mbarrier_init(smem + 480, 1);
            mbarrier_init(smem + 488, 1);
            // out_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 496, 1);
            // tmem_dealloc_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 504, 2);
            // prep_diag_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 512, 2);
            mbarrier_init(smem + 520, 2);
            mbarrier_init(smem + 528, 2);
            mbarrier_init(smem + 536, 2);
            mbarrier_init(smem + 544, 2);
            // prep_inv16_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 552, 2);
            mbarrier_init(smem + 560, 2);
            mbarrier_init(smem + 568, 2);
            mbarrier_init(smem + 576, 2);
            mbarrier_init(smem + 584, 2);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 288 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 592);
    if (warp == 0) {
        int _tmem_hold = smem + 592;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_u_acc = taddr + 224;
    const int tmem_tmem_u2_inp = taddr + 224;
    const int tmem_tmem_u2_acc = taddr + 256;
    const int tmem_tmem_out = taddr + 192;
    const int tmem_tmem_state_out = taddr + 64;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_main
            int task_idx = blockIdx.x;
            {
                int first_work = tile_schedule[blockIdx.x * schedule_stride];
                task_idx = first_work & 1023;
            }
            int seq_idx = task_idx / NUM_HEADS;
            int head_idx = task_idx % NUM_HEADS;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int seq_len = (int)(eos - bos);
            int num_chunks = (seq_len + 32 - 1) / 32;
            int total_chunks = num_chunks;
            {
                total_chunks = tile_schedule_counts[blockIdx.x];
            }
            int warp_in_wg = warp % 4;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            int state_row = warp_in_wg * 32 + lane;
            int warp_id_in_role = (warp - 0);
            int compute_local_warp = warp_id_in_role;
            int lane_quad = lane & 3;
            int state_slot = seq_idx;
            if (use_state_indices != 0) {
                state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx];
            }
            long long state_base = (long long)state_slot * state_slot_stride + ((long long)head_idx * 128 + (long long)state_row) * 128;
            unsigned int zero_diag[4];
            zero_diag[0] = 0;
            zero_diag[1] = 0;
            zero_diag[2] = 0;
            zero_diag[3] = 0;
            int diag_lane_row = lane % 16;
            int diag_lane_col = lane / 16 * 8;
            int diag_block_0 = compute_local_warp;
            int diag_block_1 = compute_local_warp + 4;
            int diag_base_0 = smem_state_diag0_addr + (unsigned int)(diag_block_0 * 16 * 16 * 2);
            int diag_base_1 = smem_state_diag0_addr + (unsigned int)(diag_block_1 * 16 * 16 * 2);
            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(diag_base_0 + (diag_lane_col / 16 * 512 + diag_lane_row * 32 + diag_lane_col % 16 * 2 ^ (diag_lane_col / 16 * 512 + diag_lane_row * 32 + diag_lane_col % 16 * 2 >> 7 & 1) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[3]))
                : "memory");
            uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(diag_base_1 + (diag_lane_col / 16 * 512 + diag_lane_row * 32 + diag_lane_col % 16 * 2 ^ (diag_lane_col / 16 * 512 + diag_lane_row * 32 + diag_lane_col % 16 * 2 >> 7 & 1) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero_diag[3]))
                : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int state_col_block = 0; state_col_block < 4; state_col_block++) {
                float state_frag[32];
                state_frag[0] = 0.0f;
                state_frag[1] = 0.0f;
                state_frag[2] = 0.0f;
                state_frag[3] = 0.0f;
                state_frag[4] = 0.0f;
                state_frag[5] = 0.0f;
                state_frag[6] = 0.0f;
                state_frag[7] = 0.0f;
                state_frag[8] = 0.0f;
                state_frag[9] = 0.0f;
                state_frag[10] = 0.0f;
                state_frag[11] = 0.0f;
                state_frag[12] = 0.0f;
                state_frag[13] = 0.0f;
                state_frag[14] = 0.0f;
                state_frag[15] = 0.0f;
                state_frag[16] = 0.0f;
                state_frag[17] = 0.0f;
                state_frag[18] = 0.0f;
                state_frag[19] = 0.0f;
                state_frag[20] = 0.0f;
                state_frag[21] = 0.0f;
                state_frag[22] = 0.0f;
                state_frag[23] = 0.0f;
                state_frag[24] = 0.0f;
                state_frag[25] = 0.0f;
                state_frag[26] = 0.0f;
                state_frag[27] = 0.0f;
                state_frag[28] = 0.0f;
                state_frag[29] = 0.0f;
                state_frag[30] = 0.0f;
                state_frag[31] = 0.0f;
                {
                    {
                        #pragma unroll
                        for (int state_quarter = 0; state_quarter < 4; state_quarter++) {
                            {
                                unsigned _ldv8_2_0;
                                unsigned _ldv8_2_1;
                                unsigned _ldv8_2_2;
                                unsigned _ldv8_2_3;
                                unsigned _ldv8_2_4;
                                unsigned _ldv8_2_5;
                                unsigned _ldv8_2_6;
                                unsigned _ldv8_2_7;
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(initial_state_f32 + (state_base + (long long)(state_col_block * 32) + (long long)(state_quarter * 8)))) : "memory");
                                state_frag[state_quarter * 8 + 0] = __uint_as_float(_ldv8_2_0);
                                state_frag[state_quarter * 8 + 1] = __uint_as_float(_ldv8_2_1);
                                state_frag[state_quarter * 8 + 2] = __uint_as_float(_ldv8_2_2);
                                state_frag[state_quarter * 8 + 3] = __uint_as_float(_ldv8_2_3);
                                state_frag[state_quarter * 8 + 4] = __uint_as_float(_ldv8_2_4);
                                state_frag[state_quarter * 8 + 5] = __uint_as_float(_ldv8_2_5);
                                state_frag[state_quarter * 8 + 6] = __uint_as_float(_ldv8_2_6);
                                state_frag[state_quarter * 8 + 7] = __uint_as_float(_ldv8_2_7);
                            }
                        }
                    }
                }
                tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32), state_frag);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int compute_stage = 0;
            unsigned int _phase_qk_full = 0;
            unsigned int _phase_v_full_0 = 0;
            unsigned int _phase_old_out_ready = 0;
            unsigned int _phase_u2_acc_ready = 0;
            unsigned int _phase_final_ready = 0;
            #pragma unroll 1
            for (int global_chunk_idx = 0; global_chunk_idx < total_chunks; global_chunk_idx++) {
                int chunk_idx = global_chunk_idx;
                long long current_state_base = state_base;
                {
                    int work = tile_schedule[blockIdx.x * schedule_stride + global_chunk_idx];
                    task_idx = work & 1023;
                    chunk_idx = work >> 10 & 255;
                    num_chunks = work >> 18;
                    seq_idx = task_idx / NUM_HEADS;
                    head_idx = task_idx % NUM_HEADS;
                    state_slot = seq_idx;
                    if (use_state_indices != 0) {
                        state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx];
                    }
                    current_state_base = (long long)state_slot * state_slot_stride + ((long long)head_idx * 128 + (long long)state_row) * 128;
                    if (global_chunk_idx != 0 && chunk_idx == 0) {
                        #pragma unroll
                        for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                            float state_frag_1[32];
                            state_frag_1[0] = 0.0f;
                            state_frag_1[1] = 0.0f;
                            state_frag_1[2] = 0.0f;
                            state_frag_1[3] = 0.0f;
                            state_frag_1[4] = 0.0f;
                            state_frag_1[5] = 0.0f;
                            state_frag_1[6] = 0.0f;
                            state_frag_1[7] = 0.0f;
                            state_frag_1[8] = 0.0f;
                            state_frag_1[9] = 0.0f;
                            state_frag_1[10] = 0.0f;
                            state_frag_1[11] = 0.0f;
                            state_frag_1[12] = 0.0f;
                            state_frag_1[13] = 0.0f;
                            state_frag_1[14] = 0.0f;
                            state_frag_1[15] = 0.0f;
                            state_frag_1[16] = 0.0f;
                            state_frag_1[17] = 0.0f;
                            state_frag_1[18] = 0.0f;
                            state_frag_1[19] = 0.0f;
                            state_frag_1[20] = 0.0f;
                            state_frag_1[21] = 0.0f;
                            state_frag_1[22] = 0.0f;
                            state_frag_1[23] = 0.0f;
                            state_frag_1[24] = 0.0f;
                            state_frag_1[25] = 0.0f;
                            state_frag_1[26] = 0.0f;
                            state_frag_1[27] = 0.0f;
                            state_frag_1[28] = 0.0f;
                            state_frag_1[29] = 0.0f;
                            state_frag_1[30] = 0.0f;
                            state_frag_1[31] = 0.0f;
                            {
                                {
                                    #pragma unroll
                                    for (int state_quarter_1 = 0; state_quarter_1 < 4; state_quarter_1++) {
                                        {
                                            unsigned _ldv8_3_0;
                                            unsigned _ldv8_3_1;
                                            unsigned _ldv8_3_2;
                                            unsigned _ldv8_3_3;
                                            unsigned _ldv8_3_4;
                                            unsigned _ldv8_3_5;
                                            unsigned _ldv8_3_6;
                                            unsigned _ldv8_3_7;
                                            asm volatile(
                                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(initial_state_f32 + (current_state_base + (long long)(state_col_block_1 * 32) + (long long)(state_quarter_1 * 8)))) : "memory");
                                            state_frag_1[state_quarter_1 * 8 + 0] = __uint_as_float(_ldv8_3_0);
                                            state_frag_1[state_quarter_1 * 8 + 1] = __uint_as_float(_ldv8_3_1);
                                            state_frag_1[state_quarter_1 * 8 + 2] = __uint_as_float(_ldv8_3_2);
                                            state_frag_1[state_quarter_1 * 8 + 3] = __uint_as_float(_ldv8_3_3);
                                            state_frag_1[state_quarter_1 * 8 + 4] = __uint_as_float(_ldv8_3_4);
                                            state_frag_1[state_quarter_1 * 8 + 5] = __uint_as_float(_ldv8_3_5);
                                            state_frag_1[state_quarter_1 * 8 + 6] = __uint_as_float(_ldv8_3_6);
                                            state_frag_1[state_quarter_1 * 8 + 7] = __uint_as_float(_ldv8_3_7);
                                        }
                                    }
                                }
                            }
                            tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 32), state_frag_1);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                #pragma unroll 1
                for (int state_col_block_2 = 0; state_col_block_2 < 4; state_col_block_2++) {
                    int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32);
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(state_addr));
                    uint32_t _tmem_load_0_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 16)), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(qk_full_addr + (compute_stage) * 8, _phase_qk_full);
                if (lane < 16) {
                    int diag_stage_f32 = compute_stage * 10496;
                    float diag_scale_0 = smem_gt_all[diag_stage_f32 + diag_block_0 * 16 + lane];
                    float diag_scale_1 = smem_gt_all[diag_stage_f32 + diag_block_1 * 16 + lane];
                    {
                        __nv_bfloat16 _bval_4 = __float2bfloat16_rn(diag_scale_0);
                        uint16_t _bits_4 = *(uint16_t*)&_bval_4;
                        uint32_t _addr_4 = static_cast<uint32_t>((diag_base_0 + (lane / 16 * 512 + lane * 32 + lane % 16 * 2 ^ (lane / 16 * 512 + lane * 32 + lane % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4), "h"(_bits_4) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_5 = __float2bfloat16_rn(diag_scale_1);
                        uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                        uint32_t _addr_5 = static_cast<uint32_t>((diag_base_1 + (lane / 16 * 512 + lane * 32 + lane % 16 * 2 ^ (lane / 16 * 512 + lane * 32 + lane % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_diag_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(v_full_addr, _phase_v_full_0);
                _phase_v_full_0 ^= 1;
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, _phase_old_out_ready);
                int v_stage_addr = smem_v_addr + (unsigned int)(warp_in_wg / 2 * 32 * 64 * 2);
                float _tmem_load_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                    : "r"(taddr + 224 + (unsigned int)tmem_row_base));
                float _tmem_load_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                    : "r"(taddr + 224 + (unsigned int)tmem_row_base + 1048576));
                float residual_values_0[16];
                float residual_values_1[16];
                unsigned int v_ld_bits_0[2];
                unsigned int v_ld_bits_1[2];
                #pragma unroll
                for (int token_group = 0; token_group < 4; token_group++) {
                    int token_pair = token_group * 8 + lane_quad * 2;
                    const int residual_reg_base = token_group * 4;
                    float beta_0 = smem_prep_beta_all[compute_stage * 10496 + (unsigned int)token_pair];
                    float beta_1 = smem_prep_beta_all[compute_stage * 10496 + (unsigned int)token_pair + 1];
                    int v_ld_matrix = lane / 8 & 1;
                    int v_ld_token = token_group * 8 + (lane & 7);
                    int v_ld_row_0 = warp_in_wg % 2 * 32 + v_ld_matrix * 8;
                    int v_ld_row_1 = v_ld_row_0 + 16;
                    int v_ld_row_addr = v_stage_addr + v_ld_token * 64 * 2;
                    int v_ld_addr_0 = (v_ld_row_addr + (v_ld_row_0 * 2 ^ (v_ld_row_addr >> 7 & 7) << 4));
                    int v_ld_addr_1 = (v_ld_row_addr + (v_ld_row_1 * 2 ^ (v_ld_row_addr >> 7 & 7) << 4));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(v_ld_bits_0[0]), "=r"(v_ld_bits_0[1])
                        : "r"(v_ld_addr_0)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(v_ld_bits_1[0]), "=r"(v_ld_bits_1[1])
                        : "r"(v_ld_addr_1)
                        : "memory");
                    float v_ld_bits_0_f32[4];
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&v_ld_bits_0_f32[_pair * 2])[0]), "=f"((&v_ld_bits_0_f32[_pair * 2])[1])
                            : "r"(v_ld_bits_0[_pair]));
                    }
                    float v_ld_bits_1_f32[4];
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&v_ld_bits_1_f32[_pair * 2])[0]), "=f"((&v_ld_bits_1_f32[_pair * 2])[1])
                            : "r"(v_ld_bits_1[_pair]));
                    }
                    residual_values_0[residual_reg_base] = (v_ld_bits_0_f32[0] - _tmem_load_1[residual_reg_base]) * beta_0;
                    residual_values_0[residual_reg_base + 1] = (v_ld_bits_0_f32[1] - _tmem_load_1[residual_reg_base + 1]) * beta_1;
                    residual_values_0[residual_reg_base + 2] = (v_ld_bits_0_f32[2] - _tmem_load_1[residual_reg_base + 2]) * beta_0;
                    residual_values_0[residual_reg_base + 3] = (v_ld_bits_0_f32[3] - _tmem_load_1[residual_reg_base + 3]) * beta_1;
                    residual_values_1[residual_reg_base] = (v_ld_bits_1_f32[0] - _tmem_load_2[residual_reg_base]) * beta_0;
                    residual_values_1[residual_reg_base + 1] = (v_ld_bits_1_f32[1] - _tmem_load_2[residual_reg_base + 1]) * beta_1;
                    residual_values_1[residual_reg_base + 2] = (v_ld_bits_1_f32[2] - _tmem_load_2[residual_reg_base + 2]) * beta_0;
                    residual_values_1[residual_reg_base + 3] = (v_ld_bits_1_f32[3] - _tmem_load_2[residual_reg_base + 3]) * beta_1;
                }
                uint32_t residual_values_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_values_0[_lp*2 + 0], residual_values_0[_lp*2+1 + 0]));
                    residual_values_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t residual_values_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_values_1[_lp*2 + 0], residual_values_1[_lp*2+1 + 0]));
                    residual_values_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_0_bf16[7])));
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&residual_values_1_bf16[7])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(v_free_addr);
                    mbarrier_arrive(u_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(u2_acc_ready_addr + (compute_stage) * 8, _phase_u2_acc_ready);
                float _tmem_load_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                    : "r"(taddr + 256 + (unsigned int)tmem_row_base));
                float _tmem_load_4[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                    : "r"(taddr + 256 + (unsigned int)tmem_row_base + 1048576));
                uint32_t _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[7])));
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_4_bf16[7])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(final_ready_addr + (compute_stage) * 8, _phase_final_ready);
                if (PERSISTENT_SCHEDULE && STORE_FINAL_STATE && chunk_idx + 1 == num_chunks) {
                    #pragma unroll
                    for (int state_col_block_3 = 0; state_col_block_3 < 4; state_col_block_3++) {
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                            : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_3 * 32)));
                        {
                            #pragma unroll
                            for (int state_quarter_2 = 0; state_quarter_2 < 4; state_quarter_2++) {
                                {
                                    unsigned _stv8_6_0 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 0]);
                                    unsigned _stv8_6_1 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 1]);
                                    unsigned _stv8_6_2 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 2]);
                                    unsigned _stv8_6_3 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 3]);
                                    unsigned _stv8_6_4 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 4]);
                                    unsigned _stv8_6_5 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 5]);
                                    unsigned _stv8_6_6 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 6]);
                                    unsigned _stv8_6_7 = __float_as_uint(_tmem_load_5[state_quarter_2 * 8 + 7]);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(final_state_f32 + (current_state_base + (long long)(state_col_block_3 * 32) + (long long)(state_quarter_2 * 8)) + (0))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
                                }
                            }
                        }
                    }
                }
                compute_stage += 1;
                if (compute_stage == 5) { compute_stage = 0; _phase_qk_full ^= 1; _phase_old_out_ready ^= 1; _phase_u2_acc_ready ^= 1; _phase_final_ready ^= 1; }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (compute_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: epilogue ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 80;");
        { // epilogue_main
            int task_idx_1 = blockIdx.x;
            {
                int first_work_1 = tile_schedule[blockIdx.x * schedule_stride];
                task_idx_1 = first_work_1 & 1023;
            }
            int seq_idx_1 = task_idx_1 / NUM_HEADS;
            int head_idx_1 = task_idx_1 % NUM_HEADS;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_1 = (seq_len_1 + 32 - 1) / 32;
            int total_chunks_1 = num_chunks_1;
            {
                total_chunks_1 = tile_schedule_counts[blockIdx.x];
            }
            int warp_id_in_role_1 = (warp - 4);
            int epilogue_local_warp = warp_id_in_role_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            unsigned int epilogue_stage = 0;
            unsigned int output_stage = 0;
            unsigned int _phase_final_ready_1 = 0;
            #pragma unroll 1
            for (int global_chunk_idx_1 = 0; global_chunk_idx_1 < total_chunks_1; global_chunk_idx_1++) {
                int chunk_idx_1 = global_chunk_idx_1;
                {
                    int work_1 = tile_schedule[blockIdx.x * schedule_stride + global_chunk_idx_1];
                    task_idx_1 = work_1 & 1023;
                    chunk_idx_1 = work_1 >> 10 & 255;
                    num_chunks_1 = work_1 >> 18;
                    seq_idx_1 = task_idx_1 / NUM_HEADS;
                    head_idx_1 = task_idx_1 % NUM_HEADS;
                    bos_1 = cu_seqlens[seq_idx_1];
                    eos_1 = cu_seqlens[seq_idx_1 + 1];
                    seq_len_1 = (int)(eos_1 - bos_1);
                }
                mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 32) ? 1 : 0);
                if (chunk_is_full != 0) {
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                    float _tmem_load_8[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1 + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    unsigned int out_packed[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                        out_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (epilogue_local_warp == 0) {
                        if (global_chunk_idx_1 >= 1) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + output_stage * 8192;
                    #pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        if (dim_half != 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int token_group_1 = 0; token_group_1 < 2; token_group_1++) {
                            int mtx_idx = lane / 8;
                            int row_addr = lane & 7;
                            int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                            int token_base = token_group_1 * 16 + mtx_idx / 2 * 8;
                            int token_addr = token_base + row_addr;
                            int token_pair_1 = token_addr / 2;
                            int token_parity = token_addr & 1;
                            int raw_row = token_pair_1 + dim_base / 64 * 16;
                            int raw_col = (dim_base & 63 ^ (token_pair_1 & 3) << 4 ^ token_parity << 3) + token_parity * 64;
                            int stsm_offset = (raw_row * 128 + raw_col) * 2;
                            const int pack_base = token_group_1 * 4;
                            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(out_stage_addr + stsm_offset));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(out_tma, 0, (int)(bos_1 + (long long)(chunk_idx_1 * 32)), head_idx_1, 0, smem_out_addr + output_stage * 8192);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                } else {
                    float _tmem_load_9[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31])
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    #pragma unroll
                    for (int token_col = 0; token_col < 32; token_col++) {
                        long long out_token = bos_1 + (long long)(chunk_idx_1 * 32 + token_col);
                        if (out_token < eos_1) {
                            long long out_idx = (out_token * (long long)NUM_HEADS + (long long)head_idx_1) * 128 + (long long)state_row_1;
                            out[out_idx] = _tmem_load_9[token_col];
                        }
                    }
                }
                epilogue_stage += 1;
                if (epilogue_stage == 5) { epilogue_stage = 0; _phase_final_ready_1 ^= 1; }
            }
            if (epilogue_local_warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (epilogue_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: idle ----
    } else if (warp == 8 || warp == 11) {
        // idle — no tasks assigned
    // ---- Role: mma ----
    } else if (warp == 9) {
        { // mma_main
            int task_idx_2 = blockIdx.x;
            int seq_idx_2 = seq_order[task_idx_2 / NUM_HEADS];
            long long bos_2 = cu_seqlens[seq_idx_2];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_2 = (seq_len_2 + 32 - 1) / 32;
            int total_chunks_2 = num_chunks_2;
            {
                total_chunks_2 = tile_schedule_counts[blockIdx.x];
            }
            unsigned int mma_stage = 0;
            unsigned int _phase_qk_full_1 = 0;
            unsigned int _phase_state_inp_ready = 0;
            unsigned int _phase_out_empty_0 = 1;
            unsigned int _phase_state_diag_ready = 0;
            unsigned int _phase_u_inp_ready = 0;
            unsigned int _phase_u2_inp_ready = 0;
            #pragma unroll 1
            for (int _global_chunk_idx = 0; _global_chunk_idx < total_chunks_2; _global_chunk_idx++) {
                mbarrier_wait(qk_full_addr + (mma_stage) * 8, _phase_qk_full_1);
                mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, _phase_state_inp_ready);
                {
                    int _mma_b_lo_0 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0));
                    elect_commit(old_out_ready_addr + (mma_stage) * 8);
                    mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                    _phase_out_empty_0 ^= 1;
                    int _mma_b_lo_1 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(0));
                    elect_commit(raw_inputs_free_addr + (mma_stage) * 8);
                }
                mbarrier_wait(state_diag_ready_addr + (mma_stage) * 8, _phase_state_diag_ready);
                int _mma_b_lo_6 = make_warp_uniform(((((smem_state_diag0_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step(tmem_tmem_state, tmem_tmem_state_inp, _mma_b_lo_6, 0xC0004010, 134546576, 0);
                int _mma_b_lo_7 = make_warp_uniform(((((smem_state_diag1_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_tmem_state + (16)), tmem_tmem_state_inp + 8, _mma_b_lo_7, 0xC0004010, 134546576, 0);
                int _mma_b_lo_8 = make_warp_uniform(((((smem_state_diag2_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_tmem_state + (32)), tmem_tmem_state_inp + 16, _mma_b_lo_8, 0xC0004010, 134546576, 0);
                int _mma_b_lo_9 = make_warp_uniform(((((smem_state_diag3_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_tmem_state + (48)), tmem_tmem_state_inp + 24, _mma_b_lo_9, 0xC0004010, 134546576, 0);
                int _mma_b_lo_10 = make_warp_uniform(((((smem_state_diag4_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_tmem_state + (64)), tmem_tmem_state_inp + 32, _mma_b_lo_10, 0xC0004010, 134546576, 0);
                int _mma_b_lo_11 = make_warp_uniform(((((smem_state_diag5_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_tmem_state + (80)), tmem_tmem_state_inp + 40, _mma_b_lo_11, 0xC0004010, 134546576, 0);
                int _mma_b_lo_12 = make_warp_uniform(((((smem_state_diag6_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_tmem_state + (96)), tmem_tmem_state_inp + 48, _mma_b_lo_12, 0xC0004010, 134546576, 0);
                int _mma_b_lo_13 = make_warp_uniform(((((smem_state_diag7_addr) >> 4) & 0x3FFF) | 0x200000) + (0) * 32);
                mma_ts_step((tmem_tmem_state + (112)), tmem_tmem_state_inp + 56, _mma_b_lo_13, 0xC0004010, 134546576, 0);
                mbarrier_wait(u_inp_ready_addr + (mma_stage) * 8, _phase_u_inp_ready);
                int _mma_b_lo_14 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u2_acc), "r"(_mma_b_lo_14), "r"(tmem_tmem_u2_inp), "r"(0));
                elect_commit(u2_acc_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, _phase_u2_inp_ready);
                int _mma_b_lo_15 = make_warp_uniform(((((smem_final_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_stage) * 2624);
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
                    "mov.b32 id, 136905872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state_out), "r"(_mma_b_lo_15), "r"(tmem_tmem_u2_inp), "r"(1));
                elect_commit2(final_ready_addr + (mma_stage) * 8, smem_free_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 5) { mma_stage = 0; _phase_qk_full_1 ^= 1; _phase_state_inp_ready ^= 1; _phase_state_diag_ready ^= 1; _phase_u_inp_ready ^= 1; _phase_u2_inp_ready ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    // ---- Role: load ----
    } else if (warp == 10) {
        { // load_main
            int task_idx_3 = blockIdx.x;
            {
                int first_work_2 = tile_schedule[blockIdx.x * schedule_stride];
                task_idx_3 = first_work_2 & 1023;
            }
            int seq_idx_3 = task_idx_3 / NUM_HEADS;
            int head_idx_2 = task_idx_3 % NUM_HEADS;
            long long bos_3 = cu_seqlens[seq_idx_3];
            long long eos_3 = cu_seqlens[seq_idx_3 + 1];
            int seq_len_3 = (int)(eos_3 - bos_3);
            int num_chunks_3 = (seq_len_3 + 32 - 1) / 32;
            int total_chunks_3 = num_chunks_3;
            {
                total_chunks_3 = tile_schedule_counts[blockIdx.x];
            }
            unsigned int _phase_v_free_0 = 1;
            #pragma unroll 1
            for (int global_chunk_idx_2 = 0; global_chunk_idx_2 < total_chunks_3; global_chunk_idx_2++) {
                int chunk_idx_2 = global_chunk_idx_2;
                {
                    int work_2 = tile_schedule[blockIdx.x * schedule_stride + global_chunk_idx_2];
                    task_idx_3 = work_2 & 1023;
                    chunk_idx_2 = work_2 >> 10 & 255;
                    num_chunks_3 = work_2 >> 18;
                    seq_idx_3 = task_idx_3 / NUM_HEADS;
                    head_idx_2 = task_idx_3 % NUM_HEADS;
                    bos_3 = cu_seqlens[seq_idx_3];
                    eos_3 = cu_seqlens[seq_idx_3 + 1];
                    seq_len_3 = (int)(eos_3 - bos_3);
                }
                mbarrier_wait(v_free_addr, _phase_v_free_0);
                _phase_v_free_0 ^= 1;
                int chunk_is_full_1 = ((seq_len_3 >= (chunk_idx_2 + 1) * 32) ? 1 : 0);
                if (elect_sync()) {
                    if (chunk_is_full_1 != 0) {
                        mbarrier_arrive_expect_tx(v_full_addr, 8192);
                        tma_4d_gmem2smem(smem_v_addr, v_tma, 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_2, 0, v_full_addr);
                    }
                }
                if (chunk_is_full_1 == 0) {
                    #pragma unroll
                    for (int v_load_iter = 0; v_load_iter < 16; v_load_iter++) {
                        int v_item = v_load_iter * 32 + lane;
                        int row = v_item / 16;
                        int segment = v_item % 16;
                        long long token = bos_3 + (long long)(chunk_idx_2 * 32 + row);
                        int token_valid = ((token < eos_3) ? 1 : 0);
                        long long v_src = (token * (long long)NUM_HEADS + (long long)head_idx_2) * 128 + (long long)(segment * 8);
                        int v_half = segment / 8;
                        int v_half_segment = segment % 8;
                        int v_dst_row_addr = smem_v_addr + (unsigned int)(v_half * 32 * 64 * 2) + (unsigned int)(row * 64 * 2);
                        int v_dst_addr = (v_dst_row_addr + (v_half_segment * 8 * 2 ^ (v_dst_row_addr >> 7 & 7) << 4));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(v_dst_addr), "l"(v + v_src), "r"((token_valid != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                }
                asm volatile("barrier.sync 8, 32;" ::: "memory");
                if (elect_sync()) {
                    if (chunk_is_full_1 == 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(v_full_addr);
                    }
                }
            }
        }
    // ---- Role: prep ----
    } else if (warp >= 12 && warp <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int task_idx_4 = blockIdx.x;
            {
                int first_work_3 = tile_schedule[blockIdx.x * schedule_stride];
                task_idx_4 = first_work_3 & 1023;
            }
            int seq_idx_4 = task_idx_4 / NUM_HEADS;
            int head_idx_3 = task_idx_4 % NUM_HEADS;
            long long bos_4 = cu_seqlens[seq_idx_4];
            long long eos_4 = cu_seqlens[seq_idx_4 + 1];
            int seq_len_4 = (int)(eos_4 - bos_4);
            int num_chunks_4 = (seq_len_4 + 32 - 1) / 32;
            int total_chunks_4 = num_chunks_4;
            {
                total_chunks_4 = tile_schedule_counts[blockIdx.x];
            }
            int instance_id = (warp - 12) / 4;
            int prep_instance = instance_id;
            int warp_id_in_role_2 = (warp - 12);
            int prep_local_warp = warp_id_in_role_2 - prep_instance * 4;
            int prep_tid = prep_local_warp * 32 + lane;
            float specialized_scale = SCALE_VALUE;
            float specialized_lower_bound = LOWER_BOUND_VALUE;
            int num_prep_iters = (total_chunks_4 + 4 - prep_instance) / 5;
            unsigned int prep_stage = (unsigned int)prep_instance;
            unsigned int prep_reuse_phase = 1;
            unsigned int prep_ready_phase = 0;
            int gate_rate_stage_f32 = prep_instance * 10496;
            float gate_bias_cached = 0.0f;
            if (!PERSISTENT_SCHEDULE && prep_tid == 0) {
                float _expf_0 = __expf(A_log[head_idx_3]);
                smem_gate_rate_all[gate_rate_stage_f32] = _expf_0;
            }
            if (!PERSISTENT_SCHEDULE && prep_tid < 128) {
                gate_bias_cached = dt_bias[head_idx_3 * 128 + prep_tid];
            }
            #pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int global_chunk_idx_3 = prep_iter * 5 + prep_instance;
                int chunk_idx_3 = global_chunk_idx_3;
                float gate_rate_cached = 0.0f;
                {
                    int work_3 = tile_schedule[blockIdx.x * schedule_stride + global_chunk_idx_3];
                    task_idx_4 = work_3 & 1023;
                    chunk_idx_3 = work_3 >> 10 & 255;
                    num_chunks_4 = work_3 >> 18;
                    seq_idx_4 = task_idx_4 / NUM_HEADS;
                    head_idx_3 = task_idx_4 % NUM_HEADS;
                    bos_4 = cu_seqlens[seq_idx_4];
                    eos_4 = cu_seqlens[seq_idx_4 + 1];
                    seq_len_4 = (int)(eos_4 - bos_4);
                    float gate_rate_lane = 0.0f;
                    if (lane == 0) {
                        float _expf_1 = __expf(A_log[head_idx_3]);
                        gate_rate_lane = _expf_1;
                    }
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, gate_rate_lane, 0);
                    gate_rate_cached = _shfl_0;
                    if (prep_tid < 128) {
                        gate_bias_cached = dt_bias[head_idx_3 * 128 + prep_tid];
                    }
                }
                int stage_f32 = prep_stage * 10496;
                int stage_bf16 = prep_stage * 20992;
                int chunk_is_full_2 = ((seq_len_4 >= (chunk_idx_3 + 1) * 32) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0 = 0.0f;
                float gate_half_log2 = specialized_lower_bound * 1.4426950408889634f * 0.5f;
                mbarrier_wait(raw_inputs_free_addr + (prep_stage) * 8, prep_reuse_phase);
                if (chunk_is_full_2 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + (prep_stage) * 8, 8704);
                            tma_3d_gmem2smem(smem_g_raw_addr + prep_stage * 41984, g_tma, 0, head_idx_3, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), gate_raw_full_addr + (prep_stage) * 8);
                            tma_2d_gmem2smem(smem_beta_raw_addr + prep_stage * 41984, beta_tma, head_idx_3 / 8 * 8, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), gate_raw_full_addr + (prep_stage) * 8);
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + (prep_stage) * 8, 16384);
                            tma_4d_gmem2smem(smem_kd_addr + prep_stage * 41984, k_tma, 0, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, prep_ready_phase);
                    if (prep_local_warp == 2 && lane < 32) {
                        unsigned int beta_raw_pair[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&beta_raw_pair[0])) : "r"(smem_beta_raw_addr + prep_stage * 41984 + (unsigned int)(lane * 16) + (unsigned int)(head_idx_3 % 8 / 2 * 4)));
                        float beta_raw_pair_f32[2];
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&beta_raw_pair_f32[_pair * 2])[0]), "=f"((&beta_raw_pair_f32[_pair * 2])[1])
                                : "r"(beta_raw_pair[_pair]));
                        }
                        float beta_logit = beta_raw_pair_f32[0];
                        if (head_idx_3 % 2 != 0) {
                            beta_logit = beta_raw_pair_f32[1];
                        }
                        float _tanh_approx_0;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(beta_logit * 0.5f));
                        early_beta_value = _tanh_approx_0 * 0.5f + 0.5f;
                    }
                    if (prep_tid < 128) {
                        float early_gate_rate = gate_rate_cached;
                        float early_gate_bias = gate_bias_cached;
                        __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                        float _cvt_f32_0 = __bfloat162float(early_gate_raw);
                        float early_gate_arg = early_gate_rate * (_cvt_f32_0 + early_gate_bias);
                        {
                            float _tanh_approx_1;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(early_gate_arg * 0.5f));
                            float early_gate_tanh = _tanh_approx_1;
                            early_gate0 = gate_half_log2 * early_gate_tanh + gate_half_log2;
                        }
                    }
                } else if (PERSISTENT_SCHEDULE) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(gate_raw_full_addr + (prep_stage) * 8);
                            mbarrier_arrive(qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, prep_ready_phase);
                }
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, prep_reuse_phase);
                if (chunk_is_full_2 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + prep_stage * 41984, q_tma, 0, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                if (chunk_is_full_2 == 0) {
                    #pragma unroll
                    for (int gate_load_pass = 0; gate_load_pass < 4; gate_load_pass++) {
                        int gate_load_item = gate_load_pass * 128 + prep_tid;
                        int gate_load_row = gate_load_item / 16;
                        int gate_load_segment = gate_load_item % 16;
                        long long gate_load_token = bos_4 + (long long)(chunk_idx_3 * 32 + gate_load_row);
                        long long gate_load_base = (gate_load_token * (long long)NUM_HEADS + (long long)head_idx_3) * 128 + (long long)(gate_load_segment * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_g_raw_addr + prep_stage * 41984 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                    }
                }
                if (chunk_is_full_2 == 0) {
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(11 + prep_instance) : "memory");
                }
                if (prep_local_warp == 2 && lane < 32) {
                    float beta_value = early_beta_value;
                    if (chunk_is_full_2 == 0) {
                        long long beta_token = bos_4 + (long long)(chunk_idx_3 * 32 + lane);
                        if (beta_token < eos_4) {
                            float beta_logit_1 = (float)beta[beta_token * (long long)NUM_HEADS + (long long)head_idx_3];
                            float _tanh_approx_3;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(beta_logit_1 * 0.5f));
                            beta_value = _tanh_approx_3 * 0.5f + 0.5f;
                        }
                    }
                    smem_prep_beta_all[stage_f32 + lane] = beta_value;
                }
                if (prep_tid < 128) {
                    int gate_col = prep_tid;
                    float gate_rate = gate_rate_cached;
                    float gate_bias = gate_bias_cached;
                    float prefix_log2 = 0.0f;
                    if (chunk_is_full_2 != 0) {
                        prefix_log2 = early_gate0;
                        int segment_1 = gate_col / 8;
                        int elem = gate_col & 7;
                        int half = elem / 4;
                        int vector_idx = segment_1 * 2 + half;
                        int cycle = vector_idx / 8;
                        int group = (segment_1 & 3) + (segment_1 / 4 + half & 1) * 4;
                        smem_gate_all[stage_f32 + (cycle * 32 + group * 4 + (elem & 3))] = prefix_log2;
                        for (int gate_block_idx = 0; gate_block_idx < 4; gate_block_idx++) {
                            float gate_block[8];
                            for (int gate_row_in_block = 0; gate_row_in_block < 8; gate_row_in_block++) {
                                gate_block[gate_row_in_block] = 0.0f;
                                if (1 + gate_block_idx * 8 + gate_row_in_block < 32) {
                                    __nv_bfloat16 gate_raw = smem_g_raw_all[stage_bf16 + (1 + gate_block_idx * 8 + gate_row_in_block) * 128 + gate_col];
                                    float _cvt_f32_1 = __bfloat162float(gate_raw);
                                    float gate_arg = gate_rate * (_cvt_f32_1 + gate_bias);
                                    float _tanh_approx_4;
                                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(gate_arg * 0.5f));
                                    float gate_tanh = _tanh_approx_4;
                                    {
                                        gate_block[gate_row_in_block] = gate_tanh;
                                    }
                                }
                            }
                            for (int gate_row_in_block_1 = 0; gate_row_in_block_1 < 8; gate_row_in_block_1++) {
                                if (1 + gate_block_idx * 8 + gate_row_in_block_1 < 32) {
                                    {
                                        prefix_log2 += gate_half_log2;
                                        prefix_log2 += gate_half_log2 * gate_block[gate_row_in_block_1];
                                    }
                                    int segment_0 = gate_col / 8;
                                    int elem_1 = gate_col & 7;
                                    int half_2 = elem_1 / 4;
                                    int vector_idx_3 = segment_0 * 2 + half_2;
                                    int cycle_4 = vector_idx_3 / 8;
                                    int group_5 = (segment_0 & 3) + (segment_0 / 4 + half_2 & 1) * 4;
                                    smem_gate_all[stage_f32 + (1 + gate_block_idx * 8 + gate_row_in_block_1) * 128 + (cycle_4 * 32 + group_5 * 4 + (elem_1 & 3))] = prefix_log2;
                                }
                            }
                        }
                    } else {
                        for (int gate_row = 0; gate_row < 32; gate_row++) {
                            long long gate_token = bos_4 + (long long)(chunk_idx_3 * 32 + gate_row);
                            float gate_log2 = 0.0f;
                            if (gate_token < eos_4) {
                                __nv_bfloat16 gate_raw_1 = smem_g_raw_all[stage_bf16 + gate_row * 128 + gate_col];
                                float _cvt_f32_2 = __bfloat162float(gate_raw_1);
                                float gate_arg_1 = gate_rate * (_cvt_f32_2 + gate_bias);
                                float _tanh_approx_5;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_5) : "f"(gate_arg_1 * 0.5f));
                                float gate_sigmoid = _tanh_approx_5 * 0.5f + 0.5f;
                                gate_log2 = specialized_lower_bound * 1.4426950408889634f * gate_sigmoid;
                            }
                            prefix_log2 += gate_log2;
                            int segment_2 = gate_col / 8;
                            int elem_2 = gate_col & 7;
                            int half_1 = elem_2 / 4;
                            int vector_idx_1 = segment_2 * 2 + half_1;
                            int cycle_1 = vector_idx_1 / 8;
                            int group_1 = (segment_2 & 3) + (segment_2 / 4 + half_1 & 1) * 4;
                            smem_gate_all[stage_f32 + gate_row * 128 + (cycle_1 * 32 + group_1 * 4 + (elem_2 & 3))] = prefix_log2;
                        }
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(11 + prep_instance) : "memory");
                if (chunk_is_full_2 != 0 || PERSISTENT_SCHEDULE) {
                    mbarrier_wait(qk_raw_full_addr + (prep_stage) * 8, prep_ready_phase);
                }
                #pragma unroll 1
                for (int work_pass = 0; work_pass < 4; work_pass++) {
                    int work_item = work_pass * 128 + prep_tid;
                    int row_1 = work_item / 16;
                    int segment_3 = work_item % 16;
                    long long token_1 = bos_4 + (long long)(chunk_idx_3 * 32 + row_1);
                    int token_valid_1 = ((token_1 < eos_4) ? 1 : 0);
                    long long gmem_base = (token_1 * (long long)NUM_HEADS + (long long)head_idx_3) * 128 + (long long)(segment_3 * 8);
                    float q_raw_vec[8];
                    float k_raw_vec[8];
                    q_raw_vec[0] = 0.0f;
                    q_raw_vec[1] = 0.0f;
                    q_raw_vec[2] = 0.0f;
                    q_raw_vec[3] = 0.0f;
                    q_raw_vec[4] = 0.0f;
                    q_raw_vec[5] = 0.0f;
                    q_raw_vec[6] = 0.0f;
                    q_raw_vec[7] = 0.0f;
                    k_raw_vec[0] = 0.0f;
                    k_raw_vec[1] = 0.0f;
                    k_raw_vec[2] = 0.0f;
                    k_raw_vec[3] = 0.0f;
                    k_raw_vec[4] = 0.0f;
                    k_raw_vec[5] = 0.0f;
                    k_raw_vec[6] = 0.0f;
                    k_raw_vec[7] = 0.0f;
                    if (chunk_is_full_2 != 0) {
                        unsigned int packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                            : "r"((smem_q_raw_prefetch_addr + prep_stage * 41984 + (unsigned int)(segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 ^ (segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32[_pair * 2])[0]), "=f"((&packed_f32[_pair * 2])[1])
                                : "r"(packed[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx = 0; value_idx < 8; value_idx++) {
                            q_raw_vec[value_idx] = packed_f32[value_idx];
                        }
                        unsigned int packed_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                            : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 ^ (segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_0_f32[_pair * 2])[0]), "=f"((&packed_0_f32[_pair * 2])[1])
                                : "r"(packed_0[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                            k_raw_vec[value_idx_1] = packed_0_f32[value_idx_1];
                        }
                    } else if (token_valid_1 != 0) {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + gmem_base);
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
                                        : "=f"((&q_raw_vec[0 + _blk * 8 + _pair * 2])[0]), "=f"((&q_raw_vec[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k + gmem_base);
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
                                        : "=f"((&k_raw_vec[0 + _blk * 8 + _pair * 2])[0]), "=f"((&k_raw_vec[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                    }
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                        float q_raw = q_raw_vec[elem_in_segment];
                        float k_raw = k_raw_vec[elem_in_segment];
                        float _fma_0 = __fmaf_rn(q_raw, q_raw, q_sum);
                        q_sum = _fma_0;
                        float _fma_1 = __fmaf_rn(k_raw, k_raw, k_sum);
                        k_sum = _fma_1;
                    }
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 8);
                    q_sum += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 8);
                    k_sum += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                    q_sum += _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                    k_sum += _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                    q_sum += _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                    k_sum += _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                    q_sum += _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                    k_sum += _shfl_xor_7;
                    float _rsqrt_0 = rsqrtf(q_sum + 1e-06f);
                    float q_inv = _rsqrt_0;
                    float _rsqrt_1 = rsqrtf(k_sum + 1e-06f);
                    float k_inv = _rsqrt_1;
                    const float2 _scale2_2 = {q_inv, q_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_raw_vec)[_ls], _scale2_2);
                    const float2 _scale2_3 = {k_inv, k_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(k_raw_vec)[_ls], _scale2_3);
                    float qd_vec[8];
                    float kd_vec[8];
                    float ki_vec[8];
                    float gate_prefix_lo[4];
                    float gate_prefix_hi[4];
                    int segment_0_1 = segment_3 * 8 / 8;
                    int elem_3 = segment_3 * 8 & 7;
                    int half_3 = elem_3 / 4;
                    int vector_idx_2 = segment_0_1 * 2 + half_3;
                    int cycle_2 = vector_idx_2 / 8;
                    int group_2 = (segment_0_1 & 3) + (segment_0_1 / 4 + half_3 & 1) * 4;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_lo[(0) + 3]))
                        : "r"(smem_gate_all_addr + (unsigned int)((stage_f32 + row_1 * 128 + (cycle_2 * 32 + group_2 * 4 + (elem_3 & 3))) * 4)));
                    int segment_1_1 = (segment_3 * 8 + 4) / 8;
                    int elem_2_1 = segment_3 * 8 + 4 & 7;
                    int half_3_1 = elem_2_1 / 4;
                    int vector_idx_4 = segment_1_1 * 2 + half_3_1;
                    int cycle_5 = vector_idx_4 / 8;
                    int group_6 = (segment_1_1 & 3) + (segment_1_1 / 4 + half_3_1 & 1) * 4;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gate_prefix_hi[(0) + 3]))
                        : "r"(smem_gate_all_addr + (unsigned int)((stage_f32 + row_1 * 128 + (cycle_5 * 32 + group_6 * 4 + (elem_2_1 & 3))) * 4)));
                    for (int elem_in_segment_1 = 0; elem_in_segment_1 < 8; elem_in_segment_1++) {
                        float prefix = ((elem_in_segment_1 < 4) ? gate_prefix_lo[elem_in_segment_1] : gate_prefix_hi[elem_in_segment_1 - 4]);
                        float common_log2 = specialized_lower_bound * 1.4426950408889634f * 16.0f;
                        float _exp2_0 = approx_exp2(prefix - common_log2);
                        float decay = _exp2_0;
                        qd_vec[elem_in_segment_1] = decay;
                        kd_vec[elem_in_segment_1] = decay;
                        ki_vec[elem_in_segment_1] = k_raw_vec[elem_in_segment_1] / decay;
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                    const float2 _scale2_4 = {specialized_scale, specialized_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_4);
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 ^ (segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3])));
                    unsigned int packed_7[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        packed_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 ^ (segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_7[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 3])));
                    unsigned int packed_8[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        packed_8[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 ^ (segment_3 * 8 / 64 * 4096 + row_1 * 128 + segment_3 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_8[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_8[(0) + 3])));
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(11 + prep_instance) : "memory");
                int pair_row_base = prep_local_warp / 2 * 16;
                int pair_col_base = prep_local_warp % 2 * 16;
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float k_acc[8];
                float q_acc[8];
                if (pair_row_base >= pair_col_base) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(k_acc[0]), "=f"(k_acc[1]), "=f"(k_acc[2]), "=f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(k_acc[4]), "=f"(k_acc[(4) + 1]), "=f"(k_acc[(4) + 2]), "=f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(q_acc[0]), "=f"(q_acc[1]), "=f"(q_acc[2]), "=f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(q_acc[4]), "=f"(q_acc[(4) + 1]), "=f"(q_acc[(4) + 2]), "=f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]), "+f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]), "+f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]), "+f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]), "+f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]), "+f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]), "+f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]), "+f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]), "+f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]), "+f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]), "+f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]), "+f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]), "+f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[0]), "+f"(k_acc[1]), "+f"(k_acc[2]), "+f"(k_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(k_acc[4]), "+f"(k_acc[(4) + 1]), "+f"(k_acc[(4) + 2]), "+f"(k_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    __syncwarp();
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[0]), "+f"(q_acc[1]), "+f"(q_acc[2]), "+f"(q_acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(q_acc[4]), "+f"(q_acc[(4) + 1]), "+f"(q_acc[(4) + 2]), "+f"(q_acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    int row0 = pair_row_base + lane / 4;
                    int row1 = row0 + 8;
                    int col0 = pair_col_base + lane % 4 * 2;
                    float beta0 = smem_prep_beta_all[stage_f32 + row0];
                    float beta1 = smem_prep_beta_all[stage_f32 + row1];
                    float seed[8];
                    seed[0] = 0.0f;
                    seed[1] = 0.0f;
                    seed[2] = 0.0f;
                    seed[3] = 0.0f;
                    seed[4] = 0.0f;
                    seed[5] = 0.0f;
                    seed[6] = 0.0f;
                    seed[7] = 0.0f;
                    if (row0 > col0) {
                        seed[0] = k_acc[0] * beta0;
                    }
                    if (row0 > col0 + 1) {
                        seed[1] = k_acc[1] * beta0;
                    }
                    if (row1 > col0) {
                        seed[2] = k_acc[2] * beta1;
                    }
                    if (row1 > col0 + 1) {
                        seed[3] = k_acc[3] * beta1;
                    }
                    if (row0 > col0 + 8) {
                        seed[4] = k_acc[4] * beta0;
                    }
                    if (row0 > col0 + 9) {
                        seed[5] = k_acc[5] * beta0;
                    }
                    if (row1 > col0 + 8) {
                        seed[6] = k_acc[6] * beta1;
                    }
                    if (row1 > col0 + 9) {
                        seed[7] = k_acc[7] * beta1;
                    }
                    unsigned int seed_packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed[_lp*2 + 0], seed[_lp*2+1 + 0]));
                        seed_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int seed_lane_row = lane % 16;
                    int seed_lane_col = lane / 16 * 8;
                    int byte_off = (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2;
                    int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                    int seed_addr = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off;
                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)seed_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                        : "memory");
                } else {
                    q_acc[0] = 0.0f;
                    q_acc[1] = 0.0f;
                    q_acc[2] = 0.0f;
                    q_acc[3] = 0.0f;
                    q_acc[4] = 0.0f;
                    q_acc[5] = 0.0f;
                    q_acc[6] = 0.0f;
                    q_acc[7] = 0.0f;
                }
                int row0_1 = pair_row_base + lane / 4;
                int row1_1 = row0_1 + 8;
                int col0_1 = pair_col_base + lane % 4 * 2;
                float mqk[8];
                mqk[0] = 0.0f;
                mqk[1] = 0.0f;
                mqk[2] = 0.0f;
                mqk[3] = 0.0f;
                mqk[4] = 0.0f;
                mqk[5] = 0.0f;
                mqk[6] = 0.0f;
                mqk[7] = 0.0f;
                if (row0_1 >= col0_1) {
                    mqk[0] = q_acc[0];
                }
                if (row0_1 >= col0_1 + 1) {
                    mqk[1] = q_acc[1];
                }
                if (row1_1 >= col0_1) {
                    mqk[2] = q_acc[2];
                }
                if (row1_1 >= col0_1 + 1) {
                    mqk[3] = q_acc[3];
                }
                if (row0_1 >= col0_1 + 8) {
                    mqk[4] = q_acc[4];
                }
                if (row0_1 >= col0_1 + 9) {
                    mqk[5] = q_acc[5];
                }
                if (row1_1 >= col0_1 + 8) {
                    mqk[6] = q_acc[6];
                }
                if (row1_1 >= col0_1 + 9) {
                    mqk[7] = q_acc[7];
                }
                unsigned int mqk_packed[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk[_lp*2 + 0], mqk[_lp*2+1 + 0]));
                    mqk_packed[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int publish_pair = 0; publish_pair < 2; publish_pair++) {
                    int publish_row = pair_col_base + publish_pair * 8 + (lane & 7);
                    int publish_col = 128 + pair_row_base + lane / 8 * 8;
                    uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + prep_stage * 41984 + (unsigned int)(publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 ^ (publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                        :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                        : "memory");
                }
                if (prep_local_warp == 1) {
                    #pragma unroll
                    for (int restore_pass = 0; restore_pass < 4; restore_pass++) {
                        int restore_col = restore_pass * 32 + lane;
                        int segment_4 = restore_col / 8;
                        int elem_4 = restore_col & 7;
                        int half_4 = elem_4 / 4;
                        int vector_idx_5 = segment_4 * 2 + half_4;
                        int cycle_3 = vector_idx_5 / 8;
                        int group_3 = (segment_4 & 3) + (segment_4 / 4 + half_4 & 1) * 4;
                        float total_log2 = smem_gate_all[stage_f32 + 3968 + (cycle_3 * 32 + group_3 * 4 + (elem_4 & 3))];
                        int restore_publish_col = restore_col;
                        float _exp2_1 = approx_exp2(total_log2 - specialized_lower_bound * 1.4426950408889634f * 16.0f);
                        smem_restore_factor_all[stage_f32 + restore_publish_col] = _exp2_1;
                    }
                    if (lane == 0) {
                        float _exp2_2 = approx_exp2(specialized_lower_bound * 1.4426950408889634f * 16.0f);
                        smem_restore_factor_all[stage_f32 + 128] = _exp2_2;
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(11 + prep_instance) : "memory");
                if (prep_tid < 128) {
                    int segment_5 = prep_tid / 8;
                    int elem_5 = prep_tid & 7;
                    int half_5 = elem_5 / 4;
                    int vector_idx_6 = segment_5 * 2 + half_5;
                    int cycle_6 = vector_idx_6 / 8;
                    int group_4 = (segment_5 & 3) + (segment_5 / 4 + half_5 & 1) * 4;
                    float total_log2_1 = smem_gate_all[stage_f32 + 3968 + (cycle_6 * 32 + group_4 * 4 + (elem_5 & 3))];
                    float _exp2_3 = approx_exp2(total_log2_1);
                    smem_gt_all[stage_f32 + prep_tid] = _exp2_3;
                }
                if (prep_local_warp >= 2) {
                    int stage_f32_0 = prep_stage * 10496;
                    float restore_scale = smem_restore_factor_all[stage_f32_0 + 128];
                    float restore_factor[8];
                    int restore_segment = lane & 15;
                    {
                        #pragma unroll
                        for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                            int restore_col_1 = restore_segment * 8 + restore_elem;
                            restore_factor[restore_elem] = smem_restore_factor_all[stage_f32_0 + restore_col_1];
                        }
                    }
                    #pragma unroll 1
                    for (int restore_pass_1 = 0; restore_pass_1 < 6; restore_pass_1++) {
                        int restore_row = 8 + (prep_local_warp - 2) * 12 + restore_pass_1 * 2 + (lane >> 4);
                        float restore_qd_values[8];
                        float restore_kd_values[8];
                        float restore_ki_values[8];
                        unsigned int packed_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                            : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32_1[_pair * 2])[0]), "=f"((&packed_f32_1[_pair * 2])[1])
                                : "r"(packed_2[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                            restore_qd_values[value_idx_2] = packed_f32_1[value_idx_2];
                        }
                        unsigned int packed_0_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 3]))
                            : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_0_f32_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_0_f32_1[_pair * 2])[0]), "=f"((&packed_0_f32_1[_pair * 2])[1])
                                : "r"(packed_0_1[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                            restore_kd_values[value_idx_3] = packed_0_f32_1[value_idx_3];
                        }
                        unsigned int packed_1_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 3]))
                            : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_1_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_1_f32[_pair * 2])[0]), "=f"((&packed_1_f32[_pair * 2])[1])
                                : "r"(packed_1_1[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                            restore_ki_values[value_idx_4] = packed_1_f32[value_idx_4];
                        }
                        float restore_kr_values[8];
                        #pragma unroll
                        for (int restore_elem_1 = 0; restore_elem_1 < 8; restore_elem_1++) {
                            restore_kr_values[restore_elem_1] = restore_ki_values[restore_elem_1] * restore_factor[restore_elem_1];
                        }
                        const float2 _scale2_7 = {restore_scale, restore_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values)[_ls], _scale2_7);
                        const float2 _scale2_8 = {restore_scale, restore_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values)[_ls], _scale2_8);
                        unsigned int packed_2_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 0], restore_qd_values[_lp*2+1 + 0]));
                            packed_2_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_1[(0) + 3])));
                        unsigned int packed_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 0], restore_kd_values[_lp*2+1 + 0]));
                            packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_3[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_3[(0) + 3])));
                        unsigned int packed_4[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                            packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3])));
                    }
                } else if (prep_local_warp == 1 && !PERSISTENT_SCHEDULE) {
                    int stage_f32_0_1 = prep_stage * 10496;
                    float restore_scale_1 = smem_restore_factor_all[stage_f32_0_1 + 128];
                    float restore_factor_1[8];
                    int restore_segment_1 = lane & 15;
                    {
                        #pragma unroll
                        for (int restore_elem_2 = 0; restore_elem_2 < 8; restore_elem_2++) {
                            int restore_col_2 = restore_segment_1 * 8 + restore_elem_2;
                            restore_factor_1[restore_elem_2] = smem_restore_factor_all[stage_f32_0_1 + restore_col_2];
                        }
                    }
                    #pragma unroll 1
                    for (int restore_pass_2 = 0; restore_pass_2 < 4; restore_pass_2++) {
                        int restore_row_1 = restore_pass_2 * 2 + (lane >> 4);
                        float restore_qd_values_1[8];
                        float restore_kd_values_1[8];
                        float restore_ki_values_1[8];
                        unsigned int packed_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
                            : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32_2[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32_2[_pair * 2])[0]), "=f"((&packed_f32_2[_pair * 2])[1])
                                : "r"(packed_5[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                            restore_qd_values_1[value_idx_5] = packed_f32_2[value_idx_5];
                        }
                        unsigned int packed_0_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 3]))
                            : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_0_f32_2[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_0_f32_2[_pair * 2])[0]), "=f"((&packed_0_f32_2[_pair * 2])[1])
                                : "r"(packed_0_2[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_6 = 0; value_idx_6 < 8; value_idx_6++) {
                            restore_kd_values_1[value_idx_6] = packed_0_f32_2[value_idx_6];
                        }
                        unsigned int packed_1_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 3]))
                            : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_1_f32_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_1_f32_1[_pair * 2])[0]), "=f"((&packed_1_f32_1[_pair * 2])[1])
                                : "r"(packed_1_2[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_7 = 0; value_idx_7 < 8; value_idx_7++) {
                            restore_ki_values_1[value_idx_7] = packed_1_f32_1[value_idx_7];
                        }
                        float restore_kr_values_1[8];
                        #pragma unroll
                        for (int restore_elem_3 = 0; restore_elem_3 < 8; restore_elem_3++) {
                            restore_kr_values_1[restore_elem_3] = restore_ki_values_1[restore_elem_3] * restore_factor_1[restore_elem_3];
                        }
                        const float2 _scale2_9 = {restore_scale_1, restore_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values_1)[_ls], _scale2_9);
                        const float2 _scale2_10 = {restore_scale_1, restore_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values_1)[_ls], _scale2_10);
                        unsigned int packed_2_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 0], restore_qd_values_1[_lp*2+1 + 0]));
                            packed_2_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_2[(0) + 3])));
                        unsigned int packed_3_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 0], restore_kd_values_1[_lp*2+1 + 0]));
                            packed_3_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_3_1[(0) + 3])));
                        unsigned int packed_4_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values_1[_lp*2 + 0], restore_kr_values_1[_lp*2+1 + 0]));
                            packed_4_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_4_1[(0) + 3])));
                    }
                }
                if (prep_local_warp == 0) {
                    int inverse_row = lane;
                    int diag_block = inverse_row / 8;
                    int lane_in_diag = lane & 7;
                    float inv_row[8];
                    unsigned int packed_6[4];
                    int byte_off_1 = inverse_row * 128 + diag_block * 8 * 2;
                    int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 3]))
                        : "r"(smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_1));
                    float packed_f32_3[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&packed_f32_3[_pair * 2])[0]), "=f"((&packed_f32_3[_pair * 2])[1])
                            : "r"(packed_6[_pair]));
                    }
                    #pragma unroll
                    for (int value_idx_8 = 0; value_idx_8 < 8; value_idx_8++) {
                        inv_row[value_idx_8] = packed_f32_3[value_idx_8];
                    }
                    #pragma unroll
                    for (int diag_elem = 0; diag_elem < 8; diag_elem++) {
                        if (lane_in_diag == diag_elem) {
                            inv_row[diag_elem] = 1.0f;
                        }
                    }
                    int diag_group_base = lane - lane_in_diag;
                    #pragma unroll
                    for (int src_row = 0; src_row < 7; src_row++) {
                        float row_scale = -inv_row[src_row];
                        #pragma unroll
                        for (int prev_col = 0; prev_col < src_row; prev_col++) {
                            int pivot_lane = diag_group_base + src_row;
                            float _shfl_1 = __shfl_sync(0xFFFFFFFF, inv_row[prev_col], pivot_lane);
                            float pivot = _shfl_1;
                            if (lane_in_diag > src_row) {
                                float _fma_2 = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                                inv_row[prev_col] = _fma_2;
                            }
                        }
                        if (lane_in_diag > src_row) {
                            inv_row[src_row] = row_scale;
                        }
                    }
                    unsigned int packed_0_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                        packed_0_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    int byte_off_1_1 = inverse_row * 128 + diag_block * 8 * 2;
                    int swizzled_off_2 = byte_off_1_1 ^ (byte_off_1_1 >> 7 & 7) << 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_2), "r"(*reinterpret_cast<uint32_t*>(&packed_0_3[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 3])));
                }
                if (prep_local_warp < 2) {
                    if (elect_sync()) {
                        mbarrier_arrive(prep_diag_ready_addr + (prep_stage) * 8);
                    }
                    mbarrier_wait(prep_diag_ready_addr + (prep_stage) * 8, prep_ready_phase);
                }
                if (prep_local_warp < 2) {
                    int lane_row = lane & 7;
                    int byte_off_2 = (prep_local_warp * 16 + 8 + lane_row) * 128 + (prep_local_warp * 16 + 8) * 2;
                    int swizzled_off_3 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                    int d_addr = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_3;
                    int byte_off_0 = (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_1_1 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                    int c_addr = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_1_1;
                    int byte_off_2_1 = (prep_local_warp * 16 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_3_1 = byte_off_2_1 ^ (byte_off_2_1 >> 7 & 7) << 4;
                    int a_addr = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_3_1;
                    unsigned int d_frag[2];
                    unsigned int c_frag[1];
                    float dc_acc[4];
                    unsigned int dc_bf16[2];
                    unsigned int inv_a_frag[1];
                    float o_acc[4];
                    unsigned int o_bf16[2];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(d_frag[0])
                        : "r"(d_addr)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(d_frag[1])
                        : "r"(d_addr)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(c_frag[0])
                        : "r"(c_addr)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
                        : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    const float2 _scale2_11 = {-1.0f, -1.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc)[_ls], _scale2_11);
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc_acc[_lp*2 + 0], dc_acc[_lp*2+1 + 0]));
                        dc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv_a_frag[0])
                        : "r"(a_addr)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(o_acc[0]), "=f"(o_acc[1]), "=f"(o_acc[2]), "=f"(o_acc[3])
                        : "r"(dc_bf16[0]), "r"(dc_bf16[1]), "r"(inv_a_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_acc[_lp*2 + 0], o_acc[_lp*2+1 + 0]));
                        o_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int byte_off_4 = (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_5 = byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4;
                    int o_addr = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_5;
                    uint32_t _stmatrix_addr_12 = static_cast<uint32_t>((unsigned long long)o_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                        :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&o_bf16[0]))
                        : "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(prep_inv16_ready_addr + (prep_stage) * 8);
                    }
                    mbarrier_wait(prep_inv16_ready_addr + (prep_stage) * 8, prep_ready_phase);
                }
                if (prep_local_warp == 0) {
                    int lane_row_1 = lane % 16;
                    int lane_col = lane / 16 * 8;
                    int byte_off_3 = (16 + lane_row_1) * 128 + (16 + lane_col) * 2;
                    int swizzled_off_4 = byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4;
                    int d_addr_1 = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_4;
                    int byte_off_0_1 = (16 + lane_row_1) * 128 + lane_col * 2;
                    int swizzled_off_1_2 = byte_off_0_1 ^ (byte_off_0_1 >> 7 & 7) << 4;
                    int c_addr_1 = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_1_2;
                    int byte_off_2_2 = lane_row_1 * 128 + lane_col * 2;
                    int swizzled_off_3_2 = byte_off_2_2 ^ (byte_off_2_2 >> 7 & 7) << 4;
                    int a_addr_1 = smem_inv_work_addr + prep_stage * 41984 + (unsigned int)swizzled_off_3_2;
                    unsigned int d32_frag[4];
                    unsigned int c32_frag[4];
                    float dc32_acc[8];
                    unsigned int dc32_bf16[4];
                    unsigned int a32_frag[4];
                    float o32_acc[8];
                    unsigned int o32_bf16[4];
                    unsigned int zero32_bf16[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(d32_frag[0]), "=r"(d32_frag[1]), "=r"(d32_frag[2]), "=r"(d32_frag[3])
                        : "r"(d_addr_1)
                        : "memory");
                    int d_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)d_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[3]))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(c32_frag[0]), "=r"(c32_frag[1]), "=r"(c32_frag[2]), "=r"(c32_frag[3])
                        : "r"(c_addr_1)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(dc32_acc[0]), "=f"(dc32_acc[1]), "=f"(dc32_acc[2]), "=f"(dc32_acc[3])
                        : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[0]), "r"(c32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(dc32_acc[4]), "=f"(dc32_acc[(4) + 1]), "=f"(dc32_acc[(4) + 2]), "=f"(dc32_acc[(4) + 3])
                        : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[2]), "r"(c32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    const float2 _scale2_14 = {-1.0f, -1.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(dc32_acc)[_ls], _scale2_14);
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc32_acc[_lp*2 + 0], dc32_acc[_lp*2+1 + 0]));
                        dc32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a32_frag[0]), "=r"(a32_frag[1]), "=r"(a32_frag[2]), "=r"(a32_frag[3])
                        : "r"(a_addr_1)
                        : "memory");
                    int a_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_15 = static_cast<uint32_t>((unsigned long long)a_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[3]))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(o32_acc[0]), "=f"(o32_acc[1]), "=f"(o32_acc[2]), "=f"(o32_acc[3])
                        : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[0]), "r"(a32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(o32_acc[4]), "=f"(o32_acc[(4) + 1]), "=f"(o32_acc[(4) + 2]), "=f"(o32_acc[(4) + 3])
                        : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[2]), "r"(a32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o32_acc[_lp*2 + 0], o32_acc[_lp*2+1 + 0]));
                        o32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int o_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_16 = static_cast<uint32_t>((unsigned long long)o_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[3]))
                        : "memory");
                    #pragma unroll
                    for (int zero_word = 0; zero_word < 4; zero_word++) {
                        zero32_bf16[zero_word] = 0;
                    }
                    int zero_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_17 = static_cast<uint32_t>((unsigned long long)zero_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[3]))
                        : "memory");
                } else if (prep_local_warp == 1 && PERSISTENT_SCHEDULE) {
                    int stage_f32_0_2 = prep_stage * 10496;
                    float restore_scale_2 = smem_restore_factor_all[stage_f32_0_2 + 128];
                    float restore_factor_2[8];
                    int restore_segment_2 = lane & 15;
                    {
                        #pragma unroll
                        for (int restore_elem_4 = 0; restore_elem_4 < 8; restore_elem_4++) {
                            int restore_col_3 = restore_segment_2 * 8 + restore_elem_4;
                            restore_factor_2[restore_elem_4] = smem_restore_factor_all[stage_f32_0_2 + restore_col_3];
                        }
                    }
                    #pragma unroll 1
                    for (int restore_pass_3 = 0; restore_pass_3 < 4; restore_pass_3++) {
                        int restore_row_2 = restore_pass_3 * 2 + (lane >> 4);
                        float restore_qd_values_2[8];
                        float restore_kd_values_2[8];
                        float restore_ki_values_2[8];
                        unsigned int packed_9[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_9[(0) + 3]))
                            : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32_4[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32_4[_pair * 2])[0]), "=f"((&packed_f32_4[_pair * 2])[1])
                                : "r"(packed_9[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_9 = 0; value_idx_9 < 8; value_idx_9++) {
                            restore_qd_values_2[value_idx_9] = packed_f32_4[value_idx_9];
                        }
                        unsigned int packed_0_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_4[(0) + 3]))
                            : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_0_f32_3[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_0_f32_3[_pair * 2])[0]), "=f"((&packed_0_f32_3[_pair * 2])[1])
                                : "r"(packed_0_4[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_10 = 0; value_idx_10 < 8; value_idx_10++) {
                            restore_kd_values_2[value_idx_10] = packed_0_f32_3[value_idx_10];
                        }
                        unsigned int packed_1_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 3]))
                            : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_1_f32_2[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_1_f32_2[_pair * 2])[0]), "=f"((&packed_1_f32_2[_pair * 2])[1])
                                : "r"(packed_1_3[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_11 = 0; value_idx_11 < 8; value_idx_11++) {
                            restore_ki_values_2[value_idx_11] = packed_1_f32_2[value_idx_11];
                        }
                        float restore_kr_values_2[8];
                        #pragma unroll
                        for (int restore_elem_5 = 0; restore_elem_5 < 8; restore_elem_5++) {
                            restore_kr_values_2[restore_elem_5] = restore_ki_values_2[restore_elem_5] * restore_factor_2[restore_elem_5];
                        }
                        const float2 _scale2_18 = {restore_scale_2, restore_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values_2)[_ls], _scale2_18);
                        const float2 _scale2_19 = {restore_scale_2, restore_scale_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values_2)[_ls], _scale2_19);
                        unsigned int packed_2_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_2[_lp*2 + 0], restore_qd_values_2[_lp*2+1 + 0]));
                            packed_2_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_2_3[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_2_3[(0) + 3])));
                        unsigned int packed_3_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_2[_lp*2 + 0], restore_kd_values_2[_lp*2+1 + 0]));
                            packed_3_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_3_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_3_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_3_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_3_2[(0) + 3])));
                        unsigned int packed_4_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values_2[_lp*2 + 0], restore_kr_values_2[_lp*2+1 + 0]));
                            packed_4_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&packed_4_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_4_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_4_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_4_2[(0) + 3])));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(11 + prep_instance) : "memory");
                if (prep_local_warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(qk_full_addr + (prep_stage) * 8);
                    }
                }
                prep_reuse_phase ^= 1;
                prep_ready_phase ^= 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"

