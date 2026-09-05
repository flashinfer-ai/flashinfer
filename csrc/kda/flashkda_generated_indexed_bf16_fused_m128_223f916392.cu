typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CudaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CudaTensorMapPack { CudaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define CUDA_INF CUDART_INF_F
#define TMEM_NCOLS 240
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_DECAY0_OFFSET 64
#define TMEM_TMEM_STATE_DECAY1_OFFSET 80
#define TMEM_TMEM_STATE_DECAY2_OFFSET 96
#define TMEM_TMEM_STATE_DECAY3_OFFSET 112
#define TMEM_TMEM_STATE_DECAY4_OFFSET 128
#define TMEM_TMEM_STATE_DECAY5_OFFSET 144
#define TMEM_TMEM_STATE_DECAY6_OFFSET 160
#define TMEM_TMEM_STATE_DECAY7_OFFSET 176
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_STATE_INP_DECAY0_OFFSET 0
#define TMEM_TMEM_STATE_INP_DECAY1_OFFSET 8
#define TMEM_TMEM_STATE_INP_DECAY2_OFFSET 16
#define TMEM_TMEM_STATE_INP_DECAY3_OFFSET 24
#define TMEM_TMEM_STATE_INP_DECAY4_OFFSET 32
#define TMEM_TMEM_STATE_INP_DECAY5_OFFSET 40
#define TMEM_TMEM_STATE_INP_DECAY6_OFFSET 48
#define TMEM_TMEM_STATE_INP_DECAY7_OFFSET 56
#define TMEM_TMEM_U_ACC_OFFSET 224
#define TMEM_TMEM_U2_INP_OFFSET 224
#define TMEM_TMEM_U2_ACC_OFFSET 208
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define NUM_CHUNK_PIPE_STAGES 5
#define NUM_CHECKPOINT_PIPE_STAGES 2
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 4096
#define SMEM_SMEM_QD_STRIDE 21504
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_G_RAW_STRIDE 21504
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 90112
#define SMEM_SMEM_G_RAW_ALL_STRIDE 90112
#define SMEM_SMEM_KD_OFF 5120
#define SMEM_SMEM_KD_STAGE_BYTES 4096
#define SMEM_SMEM_KD_STRIDE 21504
#define SMEM_SMEM_Q_RAW_PREFETCH_OFF 9216
#define SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES 4096
#define SMEM_SMEM_Q_RAW_PREFETCH_STRIDE 21504
#define SMEM_SMEM_FINAL_TRANS_OFF 9216
#define SMEM_SMEM_FINAL_TRANS_STAGE_BYTES 6144
#define SMEM_SMEM_FINAL_TRANS_STRIDE 21504
#define SMEM_SMEM_KR_TRANS_OFF 9216
#define SMEM_SMEM_KR_TRANS_STAGE_BYTES 4096
#define SMEM_SMEM_KR_TRANS_STRIDE 21504
#define SMEM_SMEM_MQK_TRANS_OFF 13312
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 512
#define SMEM_SMEM_MQK_TRANS_STRIDE 21504
#define SMEM_SMEM_FINAL_MQK_SLAB_OFF 13312
#define SMEM_SMEM_FINAL_MQK_SLAB_STAGE_BYTES 2048
#define SMEM_SMEM_FINAL_MQK_SLAB_STRIDE 21504
#define SMEM_SMEM_INV_OFF 15360
#define SMEM_SMEM_INV_STAGE_BYTES 512
#define SMEM_SMEM_INV_STRIDE 21504
#define SMEM_SMEM_V_OFF 16512
#define SMEM_SMEM_V_STAGE_BYTES 4096
#define SMEM_SMEM_V_STRIDE 21504
#define SMEM_SMEM_SHORT_N32_V_OFF 87040
#define SMEM_SMEM_SHORT_N32_V_STAGE_BYTES 16384
#define SMEM_SMEM_SHORT_N32_V_STRIDE 16384
#define SMEM_SMEM_KI_OFF 9216
#define SMEM_SMEM_KI_STAGE_BYTES 4096
#define SMEM_SMEM_KI_STRIDE 21504
#define SMEM_SMEM_GATE_OFF 13312
#define SMEM_SMEM_GATE_STAGE_BYTES 8192
#define SMEM_SMEM_GATE_STRIDE 21504
#define SMEM_SMEM_BETA_RAW_OFF 21504
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 256
#define SMEM_SMEM_BETA_RAW_STRIDE 21504
#define SMEM_SMEM_BETA_RAW_ALL_OFF 21504
#define SMEM_SMEM_BETA_RAW_ALL_STAGE_BYTES 86272
#define SMEM_SMEM_BETA_RAW_ALL_STRIDE 86272
#define SMEM_SMEM_INV_WORK_OFF 16512
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 21504
#define SMEM_SMEM_OUT_OFF 108544
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_SMEM_CHECKPOINT_OFF 117760
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 21504
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 86532
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 86532
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 20992
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 86528
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 86528
#define SMEM_SMEM_GT_ALL_OFF 15872
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 86528
#define SMEM_SMEM_GT_ALL_STRIDE 86528
#define SMEM_SMEM_PREP_BETA_ALL_OFF 22020
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 86080
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 86080
#define SMEM_SMEM_PREP_BETA_BF16_ALL_OFF 22020
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STAGE_BYTES 86048
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STRIDE 86048
#define SMEM_SMEM_PREP_BETA_U32_ALL_OFF 22020
#define SMEM_SMEM_PREP_BETA_U32_ALL_STAGE_BYTES 86048
#define SMEM_SMEM_PREP_BETA_U32_ALL_STRIDE 86048
#define SMEM_SMEM_GATE_RATE_ALL_OFF 22084
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 86020
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 86020
#define SMEM_SMEM_GATE_BIAS_ALL_OFF 116816
#define SMEM_SMEM_GATE_BIAS_ALL_STAGE_BYTES 512
#define SMEM_SMEM_GATE_BIAS_ALL_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_OFF 1024
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_STRIDE 4096
#define SMEM_SMEM_STATE_DECAY_DIAG0_OFF 1024
#define SMEM_SMEM_STATE_DECAY_DIAG0_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG0_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG1_OFF 1536
#define SMEM_SMEM_STATE_DECAY_DIAG1_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG1_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG2_OFF 2048
#define SMEM_SMEM_STATE_DECAY_DIAG2_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG2_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG3_OFF 2560
#define SMEM_SMEM_STATE_DECAY_DIAG3_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG3_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG4_OFF 3072
#define SMEM_SMEM_STATE_DECAY_DIAG4_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG4_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG5_OFF 3584
#define SMEM_SMEM_STATE_DECAY_DIAG5_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG5_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG6_OFF 4096
#define SMEM_SMEM_STATE_DECAY_DIAG6_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG6_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG7_OFF 4608
#define SMEM_SMEM_STATE_DECAY_DIAG7_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG7_STRIDE 512
#define SMEM_SMEM_V_ALL_OFF 16512
#define SMEM_SMEM_V_ALL_STAGE_BYTES 90112
#define SMEM_SMEM_V_ALL_STRIDE 90112
#define SMEM_SMEM_GATE_ALL_OFF 13312
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 94208
#define SMEM_SMEM_GATE_ALL_STRIDE 94208
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_OFF 116736
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STAGE_BYTES 80
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STRIDE 80
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_OFF 116736
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STAGE_BYTES 16
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STRIDE 16
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_OFF 116752
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STRIDE 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_OFF 116756
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STRIDE 4
#define SMEM_TOTAL 117376
#define STORE_BACKWARD_TAPE 0
#define STORE_E_TAPE 1
#define SPLIT_WORK_ITEMS 0

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


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
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
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
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


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
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
kernel_flashkda_bf16_fused_m128(__nv_bfloat16* __restrict__ q, CudaTensorMap const* q_tma, __nv_bfloat16* __restrict__ k, CudaTensorMap const* k_tma, __nv_bfloat16* __restrict__ v, CudaTensorMap const* v_tma, __nv_bfloat16* __restrict__ g, CudaTensorMap const* g_tma, __nv_bfloat16* __restrict__ beta, CudaTensorMap const* beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, CudaTensorMap const* out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, long long* __restrict__ cu_chunk_offsets, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, float* __restrict__ norm_inv_out, __nv_bfloat16* __restrict__ decay_out, float* __restrict__ beta_active_out, float* __restrict__ initial_state_f32, unsigned int* __restrict__ zero_workspace, int zero_words, int num_sequences, CudaTensorMap const* state_checkpoints_tma, float* __restrict__ final_state_f32)
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
    #define v_free_addr (mbar_base + 160)
    #define smem_free_addr (mbar_base + 200)
    #define raw_inputs_free_addr (mbar_base + 240)
    #define state_inp_ready_addr (mbar_base + 280)
    #define state_inp_left_ready_addr (mbar_base + 320)
    #define old_out_ready_addr (mbar_base + 360)
    #define u_inp_ready_addr (mbar_base + 400)
    #define u2_acc_ready_addr (mbar_base + 440)
    #define u2_inp_ready_addr (mbar_base + 480)
    #define final_ready_addr (mbar_base + 520)
    #define out_empty_addr (mbar_base + 560)
    #define tmem_dealloc_ready_addr (mbar_base + 568)
    #define checkpoint_ready_addr (mbar_base + 576)
    #define checkpoint_free_addr (mbar_base + 592)
    #define work_item_ready_addr (mbar_base + 608)
    #define aux_pairwise_inputs_ready_addr (mbar_base + 616)
    #define aux_pairwise_consumed_addr (mbar_base + 656)
    #define aux_inverse_ready_addr (mbar_base + 696)
    #define short_beta_ready_addr (mbar_base + 736)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (warp == 8 && lane == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(g_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(beta_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(state_checkpoints_tma)) : "memory");
    }


    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_all_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_kd_addr = smem + 5120;
    __nv_bfloat16* smem_q_raw_prefetch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_q_raw_prefetch_addr = smem + 9216;
    __nv_bfloat16* smem_final_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_final_trans_addr = smem + 9216;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kr_trans_addr = smem + 9216;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_mqk_trans_addr = smem + 13312;
    __nv_bfloat16* smem_final_mqk_slab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 13312);
    const int smem_final_mqk_slab_addr = smem + 13312;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 15360);
    const int smem_inv_addr = smem + 15360;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16512);
    const int smem_v_addr = smem + 16512;
    __nv_bfloat16* smem_short_n32_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 87040);
    const int smem_short_n32_v_addr = smem + 87040;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_ki_addr = smem + 9216;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 13312);
    const int smem_gate_addr = smem + 13312;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_beta_raw_addr = smem + 21504;
    __nv_bfloat16* smem_beta_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int smem_beta_raw_all_addr = smem + 21504;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16512);
    const int smem_inv_work_addr = smem + 16512;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 108544);
    const int smem_out_addr = smem + 108544;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 117760);
    const int smem_checkpoint_addr = smem + 117760;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 21504);
    const int smem_restore_factor_all_addr = smem + 21504;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 20992);
    const int smem_gt_prefix_all_addr = smem + 20992;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 15872);
    const int smem_gt_all_addr = smem + 15872;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 22020);
    const int smem_prep_beta_all_addr = smem + 22020;
    __nv_bfloat16* smem_prep_beta_bf16_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 22020);
    const int smem_prep_beta_bf16_all_addr = smem + 22020;
    unsigned int* smem_prep_beta_u32_all = reinterpret_cast<unsigned int*>(smem_raw + 22020);
    const int smem_prep_beta_u32_all_addr = smem + 22020;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 22084);
    const int smem_gate_rate_all_addr = smem + 22084;
    float* smem_gate_bias_all = reinterpret_cast<float*>(smem_raw + 116816);
    const int smem_gate_bias_all_addr = smem + 116816;
    unsigned int* smem_state_decay_diag_raw = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_state_decay_diag_raw_addr = smem + 1024;
    __nv_bfloat16* smem_state_decay_diag0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_state_decay_diag0_addr = smem + 1024;
    __nv_bfloat16* smem_state_decay_diag1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_state_decay_diag1_addr = smem + 1536;
    __nv_bfloat16* smem_state_decay_diag2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_state_decay_diag2_addr = smem + 2048;
    __nv_bfloat16* smem_state_decay_diag3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2560);
    const int smem_state_decay_diag3_addr = smem + 2560;
    __nv_bfloat16* smem_state_decay_diag4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_state_decay_diag4_addr = smem + 3072;
    __nv_bfloat16* smem_state_decay_diag5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3584);
    const int smem_state_decay_diag5_addr = smem + 3584;
    __nv_bfloat16* smem_state_decay_diag6 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int smem_state_decay_diag6_addr = smem + 4096;
    __nv_bfloat16* smem_state_decay_diag7 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4608);
    const int smem_state_decay_diag7_addr = smem + 4608;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16512);
    const int smem_v_all_addr = smem + 16512;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 13312);
    const int smem_gate_all_addr = smem + 13312;
    unsigned int* smem_state_checkpoint_needed = reinterpret_cast<unsigned int*>(smem_raw + 116736);
    const int smem_state_checkpoint_needed_addr = smem + 116736;
    float* smem_work_item_warp_max = reinterpret_cast<float*>(smem_raw + 116736);
    const int smem_work_item_warp_max_addr = smem + 116736;
    int* smem_work_item_compute_start = reinterpret_cast<int*>(smem_raw + 116752);
    const int smem_work_item_compute_start_addr = smem + 116752;
    unsigned int* smem_work_item_resolved = reinterpret_cast<unsigned int*>(smem_raw + 116756);
    const int smem_work_item_resolved_addr = smem + 116756;

    // Mbarrier init (23 groups, 97 barriers)
    // Mbarriers at smem_raw[0..776)

    if (warp == 10) {
        // --- pipeline 'chunk_pipe' ---
        // qk_full: 5 barriers, init_count=1
        // gate_raw_full: 5 barriers, init_count=1
        // qk_raw_full: 5 barriers, init_count=1
        // v_full: 5 barriers, init_count=1
        // v_free: 5 barriers, init_count=4
        // smem_free: 5 barriers, init_count=4
        // raw_inputs_free: 5 barriers, init_count=1
        // state_inp_ready: 5 barriers, init_count=4
        // state_inp_left_ready: 5 barriers, init_count=4
        // old_out_ready: 5 barriers, init_count=1
        // u_inp_ready: 5 barriers, init_count=4
        // u2_acc_ready: 5 barriers, init_count=1
        // u2_inp_ready: 5 barriers, init_count=4
        // final_ready: 5 barriers, init_count=1
        // out_empty: 1 barriers, init_count=1
        // tmem_dealloc_ready: 1 barriers, init_count=2
        // --- pipeline 'checkpoint_pipe' ---
        // checkpoint_ready: 2 barriers, init_count=4
        // checkpoint_free: 2 barriers, init_count=1
        // work_item_ready: 1 barriers, init_count=1
        // --- pipeline 'chunk_pipe' ---
        // aux_pairwise_inputs_ready: 5 barriers, init_count=1
        // aux_pairwise_consumed: 5 barriers, init_count=1
        // aux_inverse_ready: 5 barriers, init_count=1
        // short_beta_ready: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 20; _bar += 32) {
            mbarrier_init(smem + 0 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 10; _bar += 32) {
            mbarrier_init(smem + 160 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 240 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 10; _bar += 32) {
            mbarrier_init(smem + 280 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 360 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 400 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 440 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 480 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 6; _bar += 32) {
            mbarrier_init(smem + 520 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 1; _bar += 32) {
            mbarrier_init(smem + 568 + _bar * 8, 2);
        }
        for (int _bar = lane; _bar < 2; _bar += 32) {
            mbarrier_init(smem + 576 + _bar * 8, 4);
        }
        for (int _bar = lane; _bar < 23; _bar += 32) {
            mbarrier_init(smem + 592 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 240 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 776);
    if (warp == 0) {
        int _tmem_hold = smem + 776;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_decay0 = taddr + 64;
    const int tmem_tmem_state_decay1 = taddr + 80;
    const int tmem_tmem_state_decay2 = taddr + 96;
    const int tmem_tmem_state_decay3 = taddr + 112;
    const int tmem_tmem_state_decay4 = taddr + 128;
    const int tmem_tmem_state_decay5 = taddr + 144;
    const int tmem_tmem_state_decay6 = taddr + 160;
    const int tmem_tmem_state_decay7 = taddr + 176;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_state_inp_decay0 = taddr;
    const int tmem_tmem_state_inp_decay1 = taddr + 8;
    const int tmem_tmem_state_inp_decay2 = taddr + 16;
    const int tmem_tmem_state_inp_decay3 = taddr + 24;
    const int tmem_tmem_state_inp_decay4 = taddr + 32;
    const int tmem_tmem_state_inp_decay5 = taddr + 40;
    const int tmem_tmem_state_inp_decay6 = taddr + 48;
    const int tmem_tmem_state_inp_decay7 = taddr + 56;
    const int tmem_tmem_u_acc = taddr + 224;
    const int tmem_tmem_u2_inp = taddr + 224;
    const int tmem_tmem_u2_acc = taddr + 208;
    const int tmem_tmem_out = taddr + 192;
    const int tmem_tmem_state_out = taddr + 64;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_main
            int task_idx = blockIdx.x;
            int warp_id_in_role = (warp - 0);
            int compute_local_warp = warp_id_in_role;
            int warp_in_wg = warp % 4;
            int state_row = warp_in_wg * 32 + lane;
            int split_compute_start = 0;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int num_chunks = ((int)(eos - bos) + 16 - 1) / 16;
            int seq_len = (int)(eos - bos);
            int num_chunks_0 = (seq_len + 16 - 1) / 16;
            long long total_chunks = cu_chunk_offsets[num_sequences];
            long long fallback_head = total_chunks * (long long)num_heads + (long long)seq_idx * (long long)num_heads + (long long)head_idx;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            long long state_base = (((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
            int state_slot = seq_idx;
            if (use_state_indices != 0) {
                state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx];
            }
            state_base = (long long)state_slot * state_slot_stride + ((long long)head_idx * 128 + (long long)state_row) * 128;
            long long initial_state_base = state_base;
            int initial_state_enabled = (int)(use_initial_state != 0);
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
                if (initial_state_enabled != 0) {
                    {
                        float initial_values[8];
                        #pragma unroll
                        for (int initial_quarter = 0; initial_quarter < 4; initial_quarter++) {
                            {
                                unsigned _ldv8_0_0;
                                unsigned _ldv8_0_1;
                                unsigned _ldv8_0_2;
                                unsigned _ldv8_0_3;
                                unsigned _ldv8_0_4;
                                unsigned _ldv8_0_5;
                                unsigned _ldv8_0_6;
                                unsigned _ldv8_0_7;
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(initial_state_f32 + (initial_state_base + (long long)(state_col_block * 32) + (long long)(initial_quarter * 8)))) : "memory");
                                initial_values[0 + 0] = __uint_as_float(_ldv8_0_0);
                                initial_values[0 + 1] = __uint_as_float(_ldv8_0_1);
                                initial_values[0 + 2] = __uint_as_float(_ldv8_0_2);
                                initial_values[0 + 3] = __uint_as_float(_ldv8_0_3);
                                initial_values[0 + 4] = __uint_as_float(_ldv8_0_4);
                                initial_values[0 + 5] = __uint_as_float(_ldv8_0_5);
                                initial_values[0 + 6] = __uint_as_float(_ldv8_0_6);
                                initial_values[0 + 7] = __uint_as_float(_ldv8_0_7);
                            }
                            #pragma unroll
                            for (int initial_item = 0; initial_item < 8; initial_item++) {
                                __nv_bfloat16 _cvt_bf16_22 = __float2bfloat16(initial_values[initial_item]);
                                float _cvt_f32_34 = __bfloat162float(_cvt_bf16_22);
                                state_frag[initial_quarter * 8 + initial_item] = _cvt_f32_34;
                            }
                        }
                    }
                }
                tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32), state_frag);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            if (checkpoint_every_n_tokens != 0) {
                long long checkpoint_base = ((reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx] * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
                #pragma unroll
                for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_0[0 + 4], _tmem_load_0[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_0[0 + 6], _tmem_load_0[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_0[0 + 8], _tmem_load_0[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_0[0 + 10], _tmem_load_0[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_0[0 + 12], _tmem_load_0[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_0[0 + 14], _tmem_load_0[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_0[16 + 0], _tmem_load_0[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_0[16 + 2], _tmem_load_0[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_0[16 + 4], _tmem_load_0[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_0[16 + 6], _tmem_load_0[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_0[16 + 8], _tmem_load_0[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_0[16 + 10], _tmem_load_0[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_0[16 + 12], _tmem_load_0[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_0[16 + 14], _tmem_load_0[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
            }
            unsigned int compute_stage = 0;
            unsigned int checkpoint_stage_compute = 0;
            unsigned int _phase_checkpoint_free = 1;
            unsigned int _phase_qk_full = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_old_out_ready = 0;
            unsigned int _phase_u2_acc_ready = 0;
            unsigned int _phase_final_ready = 0;
            #pragma unroll 1
            for (int chunk_idx = 0; chunk_idx < num_chunks_0; chunk_idx++) {
                int chunk_global_local = chunk_idx;
                int owned_chunk = chunk_global_local >= 0 && chunk_global_local < num_chunks;
                int checkpoint_token_entering = chunk_idx * 16;
                int checkpoint_entering = checkpoint_every_n_tokens != 0 && checkpoint_token_entering % checkpoint_every_n_tokens == 0;
                float state_panel0[32];
                float state_panel1[32];
                float state_panel2[32];
                float state_panel3[32];
                unsigned int state_packed0[16];
                unsigned int state_packed1[16];
                unsigned int state_packed2[16];
                unsigned int state_packed3[16];
                mbarrier_wait(qk_full_addr + (compute_stage) * 8, _phase_qk_full);
                #pragma unroll 1
                for (int state_col_block_2 = 0; state_col_block_2 < ((1) ? 4 : 3); state_col_block_2++) {
                    int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32);
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(state_addr));
                    uint32_t _tmem_load_1_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 16)), "r"(_tmem_load_1_bf16[0]), "r"(_tmem_load_1_bf16[1]), "r"(_tmem_load_1_bf16[2]), "r"(_tmem_load_1_bf16[3]), "r"(_tmem_load_1_bf16[4]), "r"(_tmem_load_1_bf16[5]), "r"(_tmem_load_1_bf16[6]), "r"(_tmem_load_1_bf16[7]), "r"(_tmem_load_1_bf16[8]), "r"(_tmem_load_1_bf16[9]), "r"(_tmem_load_1_bf16[10]), "r"(_tmem_load_1_bf16[11]), "r"(_tmem_load_1_bf16[12]), "r"(_tmem_load_1_bf16[13]), "r"(_tmem_load_1_bf16[14]), "r"(_tmem_load_1_bf16[15]));
                    {
                        float state_scale[16];
                        #pragma unroll
                        for (int state_half = 0; state_half < 2; state_half++) {
                            #pragma unroll
                            for (int state_col = 0; state_col < 16; state_col++) {
                                state_scale[state_col] = smem_restore_factor_all[compute_stage * 5376 + (unsigned int)(state_col_block_2 * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                        }
                        tmem_st_x32_f32(state_addr, _tmem_load_1);
                    }
                    {
                        if (state_col_block_2 == 1) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            if (elect_sync()) {
                                mbarrier_arrive(state_inp_left_ready_addr + (compute_stage) * 8);
                            }
                        }
                    }
                }
                int state_tail_addr = taddr + 64 + (unsigned int)tmem_row_base + 96;
                float state_tail_frag[32];
                unsigned int state_tail_packed[16];
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(v_full_addr + (compute_stage) * 8, _phase_v_full);
                unsigned int v_prefetch_bits[8];
                {
                    int v_stage_addr = smem_v_addr + compute_stage * 21504;
                    #pragma unroll
                    for (int residual_row_half = 0; residual_row_half < 2; residual_row_half++) {
                        #pragma unroll
                        for (int token_group = 0; token_group < 2; token_group++) {
                            const int v_prefetch_reg_base = residual_row_half * 4 + token_group * 2;
                            int v_ld_matrix = lane / 8 & 1;
                            int v_ld_token = token_group * 8 + (lane & 7);
                            int v_ld_panel = warp_in_wg / 2;
                            int v_ld_row = warp_in_wg % 2 * 32 + residual_row_half * 16 + v_ld_matrix * 8;
                            int v_ld_row_addr = v_stage_addr + v_ld_panel * 16 * 64 * 2 + v_ld_token * 64 * 2;
                            int v_ld_addr = (v_ld_row_addr + (v_ld_row * 2 ^ (v_ld_row_addr >> 7 & 7) << 4));
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                : "=r"(v_prefetch_bits[v_prefetch_reg_base]), "=r"(v_prefetch_bits[v_prefetch_reg_base + 1])
                                : "r"(v_ld_addr)
                                : "memory");
                        }
                    }
                }
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, _phase_old_out_ready);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + 224 + (unsigned int)tmem_row_base);
                long long chunk_global_e = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                long long tape_ex_base = ((chunk_global_e * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 16;
                #pragma unroll
                for (int residual_half = 0; residual_half < 1; residual_half++) {
                    float residual_v[16];
                    float residual_beta[16];
                    #pragma unroll
                    for (int residual_col = 0; residual_col < 16; residual_col++) {
                        int token_col = residual_half * 16 + residual_col;
                    }
                    {
                        #pragma unroll
                        for (int residual_row_half_1 = 0; residual_row_half_1 < 2; residual_row_half_1++) {
                            float _tmem_load_3[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                                : "r"(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_row_half_1 * 1048576)));
                            uint32_t _tmem_load_3_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                                _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int residual_packed[4];
                            #pragma unroll
                            for (int token_group_1 = 0; token_group_1 < 2; token_group_1++) {
                                const int residual_pack_base = token_group_1 * 2;
                                unsigned int beta_pair = smem_prep_beta_u32_all[compute_stage * 5376 + (unsigned int)(token_group_1 * 4) + (unsigned int)(lane & 3)];
                                #pragma unroll
                                for (int residual_matrix = 0; residual_matrix < 2; residual_matrix++) {
                                    const int residual_pair = residual_pack_base + residual_matrix;
                                    const int v_prefetch_pair = residual_row_half_1 * 4 + residual_pair;
                                    uint32_t _bf16x2_sub_0;
                                    asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_sub_0) : "r"(v_prefetch_bits[v_prefetch_pair]), "r"(_tmem_load_3_bf16[residual_pair]));
                                    unsigned int residual_pair_delta = _bf16x2_sub_0;
                                    uint32_t _bf16x2_mul_0;
                                    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(residual_pair_delta), "r"(beta_pair));
                                    residual_packed[residual_pair] = _bf16x2_mul_0;
                                }
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x2.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_row_half_1 * 1048576)), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&residual_packed[3])));
                        }
                    }
                    if (STORE_BACKWARD_TAPE != 0 && STORE_E_TAPE != 0 && owned_chunk != 0) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[0 + 0], residual_v[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[0 + 2], residual_v[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[0 + 4], residual_v[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[0 + 6], residual_v[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_e + (tape_ex_base + (long long)(residual_half * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[8 + 0], residual_v[8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[8 + 2], residual_v[8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[8 + 4], residual_v[8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[8 + 6], residual_v[8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_e + (tape_ex_base + (long long)(residual_half * 16) + 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(v_free_addr + (compute_stage) * 8);
                    mbarrier_arrive(u_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(u2_acc_ready_addr + (compute_stage) * 8, _phase_u2_acc_ready);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + 208 + (unsigned int)tmem_row_base);
                if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                    long long chunk_global_r = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                    long long tape_r_base = ((chunk_global_r * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 16;
                    #pragma unroll
                    for (int tape_r_vec = 0; tape_r_vec < 4; tape_r_vec++) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_4[tape_r_vec * 8 + 0], _tmem_load_4[tape_r_vec * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_4[tape_r_vec * 8 + 2], _tmem_load_4[tape_r_vec * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_4[tape_r_vec * 8 + 4], _tmem_load_4[tape_r_vec * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_4[tape_r_vec * 8 + 6], _tmem_load_4[tape_r_vec * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_r + (tape_r_base + (long long)(tape_r_vec * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                uint32_t _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base, (const uint32_t*)_tmem_load_4_bf16);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(final_ready_addr + (compute_stage) * 8, _phase_final_ready);
                if (elect_sync()) {
                    mbarrier_arrive(smem_free_addr + (compute_stage) * 8);
                }
                int checkpoint_token = (chunk_idx + 1) * 16;
                if (checkpoint_every_n_tokens != 0 && checkpoint_token < seq_len && checkpoint_token % checkpoint_every_n_tokens == 0) {
                    long long checkpoint_idx = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx] + (long long)(checkpoint_token / checkpoint_every_n_tokens);
                    long long checkpoint_base_1 = ((checkpoint_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
                    #pragma unroll
                    for (int state_col_block_3 = 0; state_col_block_3 < 4; state_col_block_3++) {
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                            : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_3 * 32)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_5[0 + 4], _tmem_load_5[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_5[0 + 6], _tmem_load_5[0 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_5[0 + 8], _tmem_load_5[0 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_5[0 + 10], _tmem_load_5[0 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_5[0 + 12], _tmem_load_5[0 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_5[0 + 14], _tmem_load_5[0 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base_1 + (long long)(state_col_block_3 * 32))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base_1 + (long long)(state_col_block_3 * 32))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                        {
                            __nv_bfloat162 _pk[8];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_5[16 + 0], _tmem_load_5[16 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_5[16 + 2], _tmem_load_5[16 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_5[16 + 4], _tmem_load_5[16 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_5[16 + 6], _tmem_load_5[16 + 7]);
                            _pk[4] = __floats2bfloat162_rn(_tmem_load_5[16 + 8], _tmem_load_5[16 + 9]);
                            _pk[5] = __floats2bfloat162_rn(_tmem_load_5[16 + 10], _tmem_load_5[16 + 11]);
                            _pk[6] = __floats2bfloat162_rn(_tmem_load_5[16 + 12], _tmem_load_5[16 + 13]);
                            _pk[7] = __floats2bfloat162_rn(_tmem_load_5[16 + 14], _tmem_load_5[16 + 15]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base_1 + (long long)(state_col_block_3 * 32) + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(reinterpret_cast<__nv_bfloat16*>(state_checkpoints_addr) + (checkpoint_base_1 + (long long)(state_col_block_3 * 32) + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                        }
                    }
                }
                compute_stage += 1;
                if (compute_stage == 5) { compute_stage = 0; _phase_qk_full ^= 1; _phase_v_full ^= 1; _phase_old_out_ready ^= 1; _phase_u2_acc_ready ^= 1; _phase_final_ready ^= 1; }
            }
            if (store_final_state != 0) {
                #pragma unroll
                for (int state_col_block_4 = 0; state_col_block_4 < 4; state_col_block_4++) {
                    float _tmem_load_6[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_4 * 32)));
                    {
                        #pragma unroll
                        for (int state_vec = 0; state_vec < 4; state_vec++) {
                            {
                                unsigned _stv8_1_0 = __float_as_uint(_tmem_load_6[state_vec * 8 + 0]);
                                unsigned _stv8_1_1 = __float_as_uint(_tmem_load_6[state_vec * 8 + 1]);
                                unsigned _stv8_1_2 = __float_as_uint(_tmem_load_6[state_vec * 8 + 2]);
                                unsigned _stv8_1_3 = __float_as_uint(_tmem_load_6[state_vec * 8 + 3]);
                                unsigned _stv8_1_4 = __float_as_uint(_tmem_load_6[state_vec * 8 + 4]);
                                unsigned _stv8_1_5 = __float_as_uint(_tmem_load_6[state_vec * 8 + 5]);
                                unsigned _stv8_1_6 = __float_as_uint(_tmem_load_6[state_vec * 8 + 6]);
                                unsigned _stv8_1_7 = __float_as_uint(_tmem_load_6[state_vec * 8 + 7]);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(final_state_f32 + (state_base + (long long)(state_col_block_4 * 32) + (long long)(state_vec * 8)) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                            }
                        }
                    }
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (compute_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: epilogue ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // epilogue_main
            int task_idx_1 = blockIdx.x;
            int split_compute_start_1 = 0;
            unsigned int _phase_work_item_ready_0 = 0;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int num_chunks_1 = ((int)(eos_1 - bos_1) + 16 - 1) / 16;
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_0_1 = (seq_len_1 + 16 - 1) / 16;
            int warp_id_in_role_1 = (warp - 4);
            int epilogue_local_warp = warp_id_in_role_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            unsigned int epilogue_stage = 0;
            unsigned int output_stage = 0;
            unsigned int checkpoint_stage_epilogue = 0;
            int epilogue_chunks = num_chunks_0_1;
            unsigned int _phase_checkpoint_ready = 0;
            unsigned int _phase_final_ready_1 = 0;
            #pragma unroll 1
            for (int chunk_idx_1 = 0; chunk_idx_1 < epilogue_chunks; chunk_idx_1++) {
                int checkpoint_token_epilogue = chunk_idx_1 * 16;
                int checkpoint_entering_epilogue = checkpoint_every_n_tokens != 0 && checkpoint_token_epilogue % checkpoint_every_n_tokens == 0;
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 16) ? 1 : 0);
                if (chunk_is_full != 0) {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_7[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                    float _tmem_load_8[8];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1 + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    if (epilogue_local_warp == 0) {
                        if (chunk_idx_1 >= 2) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + output_stage * 4096;
                    #pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        unsigned int out_packed[8];
                        if (dim_half == 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        } else {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int token_group_2 = 0; token_group_2 < 1; token_group_2++) {
                            int mtx_idx = lane / 8;
                            int row_addr = lane & 7;
                            int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                            int token_base = token_group_2 * 16 + mtx_idx / 2 * 8;
                            int token_addr = token_base + row_addr;
                            int token_pair = token_addr / 2;
                            int token_parity = token_addr & 1;
                            int raw_row = token_pair + dim_base / 64 * 8;
                            int raw_col = (dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3) + token_parity * 64;
                            int stsm_offset = (raw_row * 128 + raw_col) * 2;
                            const int pack_base = token_group_2 * 4;
                            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(out_stage_addr + stsm_offset));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(out_tma, 0, (int)(bos_1 + (long long)(chunk_idx_1 * 16)), head_idx_1, 0, smem_out_addr + output_stage * 4096);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    output_stage = output_stage ^ 1;
                } else {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_9[16];
                    tmem_ld_x16(&_tmem_load_9[0], taddr + 192 + (unsigned int)tmem_row_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    #pragma unroll
                    for (int token_col_1 = 0; token_col_1 < 16; token_col_1++) {
                        long long out_token = bos_1 + (long long)(chunk_idx_1 * 16 + token_col_1);
                        if (out_token < eos_1) {
                            long long out_idx = (out_token * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)state_row_1;
                            out[out_idx] = _tmem_load_9[token_col_1];
                        }
                    }
                }
                epilogue_stage += 1;
                if (epilogue_stage == 5) { epilogue_stage = 0; _phase_final_ready_1 ^= 1; }
            }
            {
                if (epilogue_local_warp == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (epilogue_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: beta_prefetch ----
    } else if (warp == 8) {
        { // beta_prefetch_main

        }
    // ---- Role: aux_mma ----
    } else if (warp >= 10 && warp <= 11) {
        { // aux_mma_main
            unsigned int _phase_aux_pairwise_inputs_ready = 0;
            {
                int task_idx_2 = blockIdx.x;
                int seq_idx_2 = seq_order[task_idx_2 / num_heads];
                long long bos_2 = cu_seqlens[seq_idx_2];
                long long eos_2 = cu_seqlens[seq_idx_2 + 1];
                int seq_len_2 = (int)(eos_2 - bos_2);
                int num_chunks_2 = (seq_len_2 + 16 - 1) / 16;
                int instance_id = (warp - 10) / 1;
                int aux_instance = instance_id;
                int num_aux_iters = (num_chunks_2 + 1 - aux_instance) / 2;
                unsigned int aux_stage = (unsigned int)aux_instance;
                #pragma unroll 1
                for (int aux_iter = 0; aux_iter < num_aux_iters; aux_iter++) {
                    int chunk_idx_2 = aux_iter * 2 + aux_instance;
                    int stage_f32 = aux_stage * 5376;
                    mbarrier_wait(aux_pairwise_inputs_ready_addr + (aux_stage) * 8, _phase_aux_pairwise_inputs_ready);
                    unsigned int a_frag[4];
                    unsigned int b_frag[4];
                    float acc[8];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_kd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    float inverse_values[8];
                    int row0 = lane / 4;
                    int row1 = row0 + 8;
                    int col0 = lane % 4 * 2;
                    float beta0 = 0.0f;
                    float beta1 = 0.0f;
                    {
                        __nv_bfloat16 beta0_bf16 = smem_prep_beta_bf16_all[stage_f32 * 2 + row0];
                        __nv_bfloat16 beta1_bf16 = smem_prep_beta_bf16_all[stage_f32 * 2 + row1];
                        float _cvt_f32_24 = __bfloat162float(beta0_bf16);
                        beta0 = _cvt_f32_24;
                        float _cvt_f32_25 = __bfloat162float(beta1_bf16);
                        beta1 = _cvt_f32_25;
                    }
                    float l_values[8];
                    l_values[0] = 0.0f;
                    l_values[1] = 0.0f;
                    l_values[2] = 0.0f;
                    l_values[3] = 0.0f;
                    l_values[4] = 0.0f;
                    l_values[5] = 0.0f;
                    l_values[6] = 0.0f;
                    l_values[7] = 0.0f;
                    if (row0 > col0) {
                        __nv_bfloat16 _cvt_bf16_14 = __float2bfloat16(acc[0] * beta0);
                        float _cvt_f32_26 = __bfloat162float(_cvt_bf16_14);
                        l_values[0] = _cvt_f32_26;
                    }
                    if (row0 > col0 + 1) {
                        __nv_bfloat16 _cvt_bf16_15 = __float2bfloat16(acc[1] * beta0);
                        float _cvt_f32_27 = __bfloat162float(_cvt_bf16_15);
                        l_values[1] = _cvt_f32_27;
                    }
                    if (row1 > col0) {
                        __nv_bfloat16 _cvt_bf16_16 = __float2bfloat16(acc[2] * beta1);
                        float _cvt_f32_28 = __bfloat162float(_cvt_bf16_16);
                        l_values[2] = _cvt_f32_28;
                    }
                    if (row1 > col0 + 1) {
                        __nv_bfloat16 _cvt_bf16_17 = __float2bfloat16(acc[3] * beta1);
                        float _cvt_f32_29 = __bfloat162float(_cvt_bf16_17);
                        l_values[3] = _cvt_f32_29;
                    }
                    if (row0 > col0 + 8) {
                        __nv_bfloat16 _cvt_bf16_18 = __float2bfloat16(acc[4] * beta0);
                        float _cvt_f32_30 = __bfloat162float(_cvt_bf16_18);
                        l_values[4] = _cvt_f32_30;
                    }
                    if (row0 > col0 + 9) {
                        __nv_bfloat16 _cvt_bf16_19 = __float2bfloat16(acc[5] * beta0);
                        float _cvt_f32_31 = __bfloat162float(_cvt_bf16_19);
                        l_values[5] = _cvt_f32_31;
                    }
                    if (row1 > col0 + 8) {
                        __nv_bfloat16 _cvt_bf16_20 = __float2bfloat16(acc[6] * beta1);
                        float _cvt_f32_32 = __bfloat162float(_cvt_bf16_20);
                        l_values[6] = _cvt_f32_32;
                    }
                    if (row1 > col0 + 9) {
                        __nv_bfloat16 _cvt_bf16_21 = __float2bfloat16(acc[7] * beta1);
                        float _cvt_f32_33 = __bfloat162float(_cvt_bf16_21);
                        l_values[7] = _cvt_f32_33;
                    }
                    unsigned int l_frag[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(l_values[_lp*2 + 0], l_values[_lp*2+1 + 0]));
                        l_frag[_lp] = *(uint32_t*)&_h2;
                    }
                    unsigned int allmma_d_frag[4];
                    allmma_d_frag[0] = 0;
                    allmma_d_frag[1] = 0;
                    allmma_d_frag[2] = 0;
                    allmma_d_frag[3] = 0;
                    allmma_d_frag[0] = l_frag[0];
                    allmma_d_frag[3] = l_frag[3];
                    float n_values[8];
                    n_values[0] = 0.0f;
                    n_values[1] = 0.0f;
                    n_values[2] = 0.0f;
                    n_values[3] = 0.0f;
                    n_values[4] = 0.0f;
                    n_values[5] = 0.0f;
                    n_values[6] = 0.0f;
                    n_values[7] = 0.0f;
                    n_values[0] = -l_values[0];
                    n_values[1] = -l_values[1];
                    n_values[6] = -l_values[6];
                    n_values[7] = -l_values[7];
                    if (row0 == col0) {
                        n_values[0] = n_values[0] + 1.0f;
                    }
                    if (row0 == col0 + 1) {
                        n_values[1] = n_values[1] + 1.0f;
                    }
                    if (row1 == col0) {
                        n_values[2] = n_values[2] + 1.0f;
                    }
                    if (row1 == col0 + 1) {
                        n_values[3] = n_values[3] + 1.0f;
                    }
                    if (row0 == col0 + 8) {
                        n_values[4] = n_values[4] + 1.0f;
                    }
                    if (row0 == col0 + 9) {
                        n_values[5] = n_values[5] + 1.0f;
                    }
                    if (row1 == col0 + 8) {
                        n_values[6] = n_values[6] + 1.0f;
                    }
                    if (row1 == col0 + 9) {
                        n_values[7] = n_values[7] + 1.0f;
                    }
                    unsigned int n_frag[4];
                    unsigned int low_word[1];
                    unsigned int high_word[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 0], n_values[_lp*2+1 + 0]));
                        low_word[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 6], n_values[_lp*2+1 + 6]));
                        high_word[_lp] = *(uint32_t*)&_h2;
                    }
                    n_frag[0] = 0;
                    n_frag[1] = 0;
                    n_frag[2] = 0;
                    n_frag[3] = 0;
                    n_frag[0] = low_word[0];
                    n_frag[3] = high_word[0];
                    float product[8];
                    unsigned int rhs_trans_frag[4];
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag[word])
                            : "r"(allmma_d_frag[word]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(allmma_d_frag[0]), "r"(allmma_d_frag[1]), "r"(allmma_d_frag[2]), "r"(allmma_d_frag[3]), "r"(rhs_trans_frag[0]), "r"(rhs_trans_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(allmma_d_frag[0]), "r"(allmma_d_frag[1]), "r"(allmma_d_frag[2]), "r"(allmma_d_frag[3]), "r"(rhs_trans_frag[2]), "r"(rhs_trans_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    unsigned int d2_frag[4];
                    unsigned int low_word_0[1];
                    unsigned int high_word_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 0], product[_lp*2+1 + 0]));
                        low_word_0[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 6], product[_lp*2+1 + 6]));
                        high_word_1[_lp] = *(uint32_t*)&_h2;
                    }
                    d2_frag[0] = 0;
                    d2_frag[1] = 0;
                    d2_frag[2] = 0;
                    d2_frag[3] = 0;
                    d2_frag[0] = low_word_0[0];
                    d2_frag[3] = high_word_1[0];
                    unsigned int rhs_trans_frag_2[4];
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_2[word_1])
                            : "r"(d2_frag[word_1]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_2[0]), "r"(rhs_trans_frag_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_2[2]), "r"(rhs_trans_frag_2[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int value_idx = 0; value_idx < 8; value_idx++) {
                        n_values[value_idx] = n_values[value_idx] + product[value_idx];
                    }
                    unsigned int low_word_3[1];
                    unsigned int high_word_4[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 0], n_values[_lp*2+1 + 0]));
                        low_word_3[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(n_values[_lp*2 + 6], n_values[_lp*2+1 + 6]));
                        high_word_4[_lp] = *(uint32_t*)&_h2;
                    }
                    n_frag[0] = 0;
                    n_frag[1] = 0;
                    n_frag[2] = 0;
                    n_frag[3] = 0;
                    n_frag[0] = low_word_3[0];
                    n_frag[3] = high_word_4[0];
                    unsigned int rhs_trans_frag_5[4];
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_5[word_2])
                            : "r"(d2_frag[word_2]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(d2_frag[0]), "r"(d2_frag[1]), "r"(d2_frag[2]), "r"(d2_frag[3]), "r"(rhs_trans_frag_5[0]), "r"(rhs_trans_frag_5[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(d2_frag[0]), "r"(d2_frag[1]), "r"(d2_frag[2]), "r"(d2_frag[3]), "r"(rhs_trans_frag_5[2]), "r"(rhs_trans_frag_5[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    unsigned int d4_frag[4];
                    unsigned int low_word_6[1];
                    unsigned int high_word_7[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 0], product[_lp*2+1 + 0]));
                        low_word_6[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(product[_lp*2 + 6], product[_lp*2+1 + 6]));
                        high_word_7[_lp] = *(uint32_t*)&_h2;
                    }
                    d4_frag[0] = 0;
                    d4_frag[1] = 0;
                    d4_frag[2] = 0;
                    d4_frag[3] = 0;
                    d4_frag[0] = low_word_6[0];
                    d4_frag[3] = high_word_7[0];
                    unsigned int rhs_trans_frag_8[4];
                    #pragma unroll
                    for (int word_3 = 0; word_3 < 4; word_3++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_8[word_3])
                            : "r"(d4_frag[word_3]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_8[0]), "r"(rhs_trans_frag_8[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(n_frag[0]), "r"(n_frag[1]), "r"(n_frag[2]), "r"(n_frag[3]), "r"(rhs_trans_frag_8[2]), "r"(rhs_trans_frag_8[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                        n_values[value_idx_1] = n_values[value_idx_1] + product[value_idx_1];
                    }
                    unsigned int binv_frag[4];
                    binv_frag[0] = 0;
                    binv_frag[1] = 0;
                    binv_frag[2] = 0;
                    binv_frag[3] = 0;
                    binv_frag[0] = n_frag[0];
                    binv_frag[3] = n_frag[3];
                    unsigned int a21_frag[4];
                    a21_frag[0] = 0;
                    a21_frag[1] = 0;
                    a21_frag[2] = 0;
                    a21_frag[3] = 0;
                    a21_frag[1] = l_frag[1];
                    unsigned int rhs_trans_frag_9[4];
                    #pragma unroll
                    for (int word_4 = 0; word_4 < 4; word_4++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_9[word_4])
                            : "r"(a21_frag[word_4]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(binv_frag[0]), "r"(binv_frag[1]), "r"(binv_frag[2]), "r"(binv_frag[3]), "r"(rhs_trans_frag_9[0]), "r"(rhs_trans_frag_9[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(binv_frag[0]), "r"(binv_frag[1]), "r"(binv_frag[2]), "r"(binv_frag[3]), "r"(rhs_trans_frag_9[2]), "r"(rhs_trans_frag_9[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    float correction_values[2];
                    correction_values[0] = -product[2];
                    correction_values[1] = -product[3];
                    unsigned int correction_word[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(correction_values[_lp*2 + 0], correction_values[_lp*2+1 + 0]));
                        correction_word[_lp] = *(uint32_t*)&_h2;
                    }
                    unsigned int correction_frag[4];
                    correction_frag[0] = 0;
                    correction_frag[1] = 0;
                    correction_frag[2] = 0;
                    correction_frag[3] = 0;
                    correction_frag[1] = correction_word[0];
                    unsigned int rhs_trans_frag_10[4];
                    #pragma unroll
                    for (int word_5 = 0; word_5 < 4; word_5++) {
                        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
                            : "=r"(rhs_trans_frag_10[word_5])
                            : "r"(binv_frag[word_5]));
                    }
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[0]), "=f"(product[1]), "=f"(product[2]), "=f"(product[3])
                        : "r"(correction_frag[0]), "r"(correction_frag[1]), "r"(correction_frag[2]), "r"(correction_frag[3]), "r"(rhs_trans_frag_10[0]), "r"(rhs_trans_frag_10[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(product[4]), "=f"(product[(4) + 1]), "=f"(product[(4) + 2]), "=f"(product[(4) + 3])
                        : "r"(correction_frag[0]), "r"(correction_frag[1]), "r"(correction_frag[2]), "r"(correction_frag[3]), "r"(rhs_trans_frag_10[2]), "r"(rhs_trans_frag_10[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    n_values[2] = product[2];
                    n_values[3] = product[3];
                    #pragma unroll
                    for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                        inverse_values[value_idx_2] = n_values[value_idx_2];
                    }
                    unsigned int inverse_packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inverse_values[_lp*2 + 0], inverse_values[_lp*2+1 + 0]));
                        inverse_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int inverse_lane_row = lane % 16;
                    int inverse_lane_col = lane / 16 * 8;
                    int byte_off = (int)aux_stage * 21504 + inverse_lane_row * 128 + inverse_lane_col * 2;
                    int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                    int inverse_work_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)inverse_work_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&inverse_packed[3]))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_qd_addr + aux_stage * 21504 + (unsigned int)(((lane / 16 / 8 * 128 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 128 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_ki_addr + aux_stage * 21504 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 128 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 128 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    int row0_11 = lane / 4;
                    int row1_12 = row0_11 + 8;
                    int col0_13 = lane % 4 * 2;
                    float mqk[8];
                    mqk[0] = 0.0f;
                    mqk[1] = 0.0f;
                    mqk[2] = 0.0f;
                    mqk[3] = 0.0f;
                    mqk[4] = 0.0f;
                    mqk[5] = 0.0f;
                    mqk[6] = 0.0f;
                    mqk[7] = 0.0f;
                    if (row0_11 >= col0_13) {
                        mqk[0] = acc[0];
                    }
                    if (row0_11 >= col0_13 + 1) {
                        mqk[1] = acc[1];
                    }
                    if (row1_12 >= col0_13) {
                        mqk[2] = acc[2];
                    }
                    if (row1_12 >= col0_13 + 1) {
                        mqk[3] = acc[3];
                    }
                    if (row0_11 >= col0_13 + 8) {
                        mqk[4] = acc[4];
                    }
                    if (row0_11 >= col0_13 + 9) {
                        mqk[5] = acc[5];
                    }
                    if (row1_12 >= col0_13 + 8) {
                        mqk[6] = acc[6];
                    }
                    if (row1_12 >= col0_13 + 9) {
                        mqk[7] = acc[7];
                    }
                    unsigned int mqk_packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk[_lp*2 + 0], mqk[_lp*2+1 + 0]));
                        mqk_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int publish_pair = 0; publish_pair < 2; publish_pair++) {
                        int publish_row = publish_pair * 8 + (lane & 7);
                        int publish_col = lane / 8 * 8;
                        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(smem_mqk_trans_addr + aux_stage * 21504 + (unsigned int)(publish_col / 16 * 512 + publish_row * 32 + publish_col % 16 * 2 ^ (publish_col / 16 * 512 + publish_row * 32 + publish_col % 16 * 2 >> 7 & 1) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                            : "memory");
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(aux_pairwise_consumed_addr + (aux_stage) * 8);
                    }
                    int lane_row = lane % 16;
                    int lane_col = lane / 16 * 8;
                    int byte_off_14 = (int)aux_stage * 21504 + lane_row * 128 + lane_col * 2;
                    int swizzled_off_15 = byte_off_14 ^ (byte_off_14 >> 7 & 7) << 4;
                    int inv16_addr = smem_inv_work_addr + (unsigned int)swizzled_off_15;
                    unsigned int inv16_frag[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(inv16_frag[0]), "=r"(inv16_frag[1]), "=r"(inv16_frag[2]), "=r"(inv16_frag[3])
                        : "r"(inv16_addr)
                        : "memory");
                    int inv16_publish_addr = (smem_inv_addr + aux_stage * 21504 + (unsigned int)(lane_col / 16 * 512 + lane_row * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 512 + lane_row * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)inv16_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv16_frag[3]))
                        : "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(qk_full_addr + (aux_stage) * 8);
                        mbarrier_arrive(aux_inverse_ready_addr + (aux_stage) * 8);
                    }
                    aux_stage += 1;
                    if (aux_stage == 5) { aux_stage = 0; _phase_aux_pairwise_inputs_ready ^= 1; }
                    aux_stage += 1;
                    if (aux_stage == 5) { aux_stage = 0; _phase_aux_pairwise_inputs_ready ^= 1; }
                }
            }
            unsigned int _phase_work_item_ready_0_1 = 0;
        }
    // ---- Role: mma ----
    } else if (warp == 9) {
        { // mma_main
            int task_idx_3 = blockIdx.x;
            int split_compute_start_2 = 0;
            unsigned int _phase_work_item_ready_0_2 = 0;
            int seq_idx_3 = seq_order[task_idx_3 / num_heads];
            int head_idx_2 = task_idx_3 % num_heads;
            long long bos_3 = cu_seqlens[seq_idx_3];
            long long eos_3 = cu_seqlens[seq_idx_3 + 1];
            int num_chunks_3 = ((int)(eos_3 - bos_3) + 16 - 1) / 16;
            int seq_len_3 = (int)(eos_3 - bos_3);
            int num_chunks_0_2 = (seq_len_3 + 16 - 1) / 16;
            unsigned int mma_stage = 0;
            unsigned int _phase_qk_full_1 = 0;
            unsigned int _phase_out_empty_0 = 1;
            unsigned int _phase_state_inp_left_ready = 0;
            unsigned int _phase_state_inp_ready = 0;
            unsigned int _phase_u_inp_ready = 0;
            unsigned int _phase_u2_inp_ready = 0;
            unsigned int _phase_final_ready_2 = 0;
            #pragma unroll 1
            for (int _chunk_idx = 0; _chunk_idx < num_chunks_0_2; _chunk_idx++) {
                mbarrier_wait(qk_full_addr + (mma_stage) * 8, _phase_qk_full_1);
                {
                    mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                    _phase_out_empty_0 ^= 1;
                }
                {
                    mbarrier_wait(state_inp_left_ready_addr + (mma_stage) * 8, _phase_state_inp_left_ready);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 1344);
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
                    "mov.b32 id, 134481040;\n\t"
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
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_0), "r"(tmem_tmem_state_inp), "r"(0));
                    mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, _phase_state_inp_ready);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 1344);
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
                    "mov.b32 id, 134481040;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
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
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(1));
                    {
                        int _mma_b_lo_2 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + (mma_stage) * 1344);
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
                    "mov.b32 id, 134481040;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_2), "r"(tmem_tmem_state_inp), "r"(0));
                    }
                }
                {
                    elect_commit2(old_out_ready_addr + (mma_stage) * 8, raw_inputs_free_addr + (mma_stage) * 8);
                }
                mbarrier_wait(u_inp_ready_addr + (mma_stage) * 8, _phase_u_inp_ready);
                int _mma_b_lo_7 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + (mma_stage) * 1344);
                mma_ts_step(tmem_tmem_u2_acc, tmem_tmem_u2_inp, _mma_b_lo_7, 0xC0004010, 134481040, 0);
                elect_commit(u2_acc_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, _phase_u2_inp_ready);
                int _mma_b_lo_8 = make_warp_uniform(((((smem_kr_trans_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_stage) * 1344);
                mma_ts_step(tmem_tmem_state, tmem_tmem_u2_inp, _mma_b_lo_8, 0x40004040, 136381584, 1);
                int _mma_b_lo_9 = make_warp_uniform(((((smem_mqk_trans_addr) >> 4) & 0x3FFF) | 0x200000) + (mma_stage) * 1344);
                mma_ts_step(tmem_tmem_out, tmem_tmem_u2_inp, _mma_b_lo_9, 0xC0004010, 134546576, 1);
                elect_commit(final_ready_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 5) { mma_stage = 0; _phase_qk_full_1 ^= 1; _phase_state_inp_left_ready ^= 1; _phase_state_inp_ready ^= 1; _phase_u_inp_ready ^= 1; _phase_u2_inp_ready ^= 1; _phase_final_ready_2 ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    // ---- Role: prep ----
    } else if (warp >= 12 && warp <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int task_idx_4 = blockIdx.x;
            int split_compute_start_3 = 0;
            unsigned int _phase_work_item_ready_0_3 = 0;
            int seq_idx_4 = seq_order[task_idx_4 / num_heads];
            int head_idx_3 = task_idx_4 % num_heads;
            long long bos_4 = cu_seqlens[seq_idx_4];
            long long eos_4 = cu_seqlens[seq_idx_4 + 1];
            int num_chunks_4 = ((int)(eos_4 - bos_4) + 16 - 1) / 16;
            int seq_len_4 = (int)(eos_4 - bos_4);
            int num_chunks_0_3 = (seq_len_4 + 16 - 1) / 16;
            int instance_id_1 = (warp - 12) / 4;
            int prep_instance = instance_id_1;
            int warp_id_in_role_2 = (warp - 12);
            int prep_local_warp = warp_id_in_role_2 - prep_instance * 4;
            int prep_tid = prep_local_warp * 32 + lane;
            int num_prep_iters = (num_chunks_0_3 + 5 - 1 - prep_instance) / 5;
            unsigned int prep_stage = (unsigned int)prep_instance;
            int gate_rate_stage_f32 = prep_instance * 5376;
            int prep_global_tid = warp_id_in_role_2 * 32 + lane;
            if (prep_tid == 0) {
                float _expf_0 = __expf(A_log[head_idx_3]);
                smem_gate_rate_all[gate_rate_stage_f32] = _expf_0;
            }
            if (prep_global_tid < 128) {
                smem_gate_bias_all[prep_global_tid] = dt_bias[head_idx_3 * 128 + prep_global_tid];
            }
            asm volatile("barrier.sync 15, 640;" ::: "memory");
            unsigned int _phase_raw_inputs_free = 1;
            unsigned int _phase_gate_raw_full = 0;
            unsigned int _phase_smem_free = 1;
            unsigned int _phase_v_free = 1;
            unsigned int _phase_qk_raw_full = 0;
            unsigned int _phase_short_beta_ready = 0;
            unsigned int _phase_aux_pairwise_consumed = 0;
            unsigned int _phase_aux_inverse_ready = 0;
            #pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int chunk_idx_3 = prep_iter * 5 + prep_instance;
                int chunk_global_local_1 = chunk_idx_3;
                int owned_chunk_1 = chunk_global_local_1 >= 0 && chunk_global_local_1 < num_chunks_4;
                int stage_f32_1 = prep_stage * 5376;
                int stage_bf16 = prep_stage * 10752;
                int chunk_is_full_1 = ((seq_len_4 >= (chunk_idx_3 + 1) * 16) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0 = 0.0f;
                if (chunk_is_full_1 != 0 || prep_iter != 0) {
                    mbarrier_wait(raw_inputs_free_addr + (prep_stage) * 8, _phase_raw_inputs_free);
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + (prep_stage) * 8, 4096);
                            tma_3d_gmem2smem(smem_g_raw_addr + prep_stage * 21504, g_tma, 0, head_idx_3, (int)(bos_4 + (long long)(chunk_idx_3 * 16)), gate_raw_full_addr + (prep_stage) * 8);
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + (prep_stage) * 8, 8192);
                            tma_4d_gmem2smem(smem_kd_addr + prep_stage * 21504, k_tma, 0, (int)(bos_4 + (long long)(chunk_idx_3 * 16)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, _phase_gate_raw_full);
                    if (prep_local_warp == 2 && lane < 16) {
                        float beta_logit = 0.0f;
                        {
                            {
                                long long beta_token = bos_4 + (long long)(chunk_idx_3 * 16 + lane);
                                beta_logit = (float)beta[beta_token * beta_token_stride + (long long)head_idx_3];
                            }
                        }
                        float _tanh_approx_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(beta_logit * 0.5f));
                        early_beta_value = _tanh_approx_1 * 0.5f + 0.5f;
                    }
                    {
                        if (prep_tid < 128) {
                            float early_gate_rate = smem_gate_rate_all[stage_f32_1];
                            float early_gate_bias = smem_gate_bias_all[prep_tid];
                            __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                            float _cvt_f32_1 = __bfloat162float(early_gate_raw);
                            float early_gate_arg = early_gate_rate * (_cvt_f32_1 + early_gate_bias);
                            float _tanh_approx_2;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(early_gate_arg * 0.5f));
                            float early_gate_sigmoid = _tanh_approx_2 * 0.5f + 0.5f;
                            early_gate0 = lower_bound * 1.4426950408889634f * early_gate_sigmoid;
                        }
                    }
                }
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, _phase_smem_free);
                mbarrier_wait(v_free_addr + (prep_stage) * 8, _phase_v_free);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + prep_stage * 21504, q_tma, 0, (int)(bos_4 + (long long)(chunk_idx_3 * 16)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                if (chunk_is_full_1 == 0) {
                    #pragma unroll
                    for (int gate_load_pass = 0; gate_load_pass < 2; gate_load_pass++) {
                        int gate_load_item = gate_load_pass * 128 + prep_tid;
                        int gate_load_row = gate_load_item / 16;
                        int gate_load_segment = gate_load_item % 16;
                        long long gate_load_token = bos_4 + (long long)(chunk_idx_3 * 16 + gate_load_row);
                        long long gate_load_base = (gate_load_token * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(gate_load_segment * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_g_raw_addr + prep_stage * 21504 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                        int q_tail_addr = (smem_q_raw_prefetch_addr + prep_stage * 21504 + (unsigned int)(gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(q_tail_addr), "l"(q + gate_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                        int k_tail_addr = (smem_kd_addr + prep_stage * 21504 + (unsigned int)(gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 2048 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(k_tail_addr), "l"(k + gate_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                    }
                }
                if (chunk_is_full_1 == 0) {
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                if (prep_local_warp == 2 && lane < 16) {
                    long long beta_token_1 = bos_4 + (long long)(chunk_idx_3 * 16 + lane);
                    float beta_value = early_beta_value;
                    if (chunk_is_full_1 == 0) {
                        if (beta_token_1 < eos_4) {
                            float beta_logit_1 = (float)beta[beta_token_1 * (long long)num_heads + (long long)head_idx_3];
                            beta_logit_1 = (float)beta[beta_token_1 * beta_token_stride + (long long)head_idx_3];
                            float _tanh_approx_3;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(beta_logit_1 * 0.5f));
                            beta_value = _tanh_approx_3 * 0.5f + 0.5f;
                        }
                    }
                    {
                        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(beta_value);
                        smem_prep_beta_bf16_all[stage_bf16 + lane] = _cvt_bf16_0;
                    }
                }
                if (prep_tid < 128) {
                    int gate_col = prep_tid;
                    float gate_rate = smem_gate_rate_all[stage_f32_1];
                    float gate_bias = smem_gate_bias_all[gate_col];
                    float prefix_log2 = 0.0f;
                    {
                        for (int gate_row = 0; gate_row < 16; gate_row++) {
                            long long gate_token = bos_4 + (long long)(chunk_idx_3 * 16 + gate_row);
                            float gate_log2 = 0.0f;
                            int gate_needs_compute = 1;
                            if (gate_row == 0) {
                                if (chunk_is_full_1 != 0) {
                                    gate_log2 = early_gate0;
                                    gate_needs_compute = 0;
                                }
                            }
                            if (gate_needs_compute != 0) {
                                if (gate_token < eos_4) {
                                    __nv_bfloat16 gate_raw = smem_g_raw_all[stage_bf16 + gate_row * 128 + gate_col];
                                    float _cvt_f32_7 = __bfloat162float(gate_raw);
                                    float gate_arg = gate_rate * (_cvt_f32_7 + gate_bias);
                                    float _tanh_approx_8;
                                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_8) : "f"(gate_arg * 0.5f));
                                    float gate_sigmoid = _tanh_approx_8 * 0.5f + 0.5f;
                                    gate_log2 = lower_bound * 1.4426950408889634f * gate_sigmoid;
                                }
                            }
                            prefix_log2 += gate_log2;
                            smem_gate_all[stage_f32_1 + gate_row * 128 + gate_col] = prefix_log2;
                        }
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (chunk_is_full_1 != 0) {
                    mbarrier_wait(qk_raw_full_addr + (prep_stage) * 8, _phase_qk_raw_full);
                }
                if (prep_tid < 128) {
                    float total_log2 = smem_gt_prefix_all[stage_f32_1 + prep_tid];
                    float _exp2_0 = approx_exp2(total_log2);
                    float restore_factor_value = _exp2_0;
                    smem_restore_factor_all[stage_f32_1 + prep_tid] = restore_factor_value;
                }
                if (prep_tid == 0) {
                    float _exp2_1 = approx_exp2(lower_bound * 1.4426950408889634f * 8.0f);
                    smem_restore_factor_all[stage_f32_1 + 128] = _exp2_1;
                }
                #pragma unroll 1
                for (int work_pass = 0; work_pass < 2; work_pass++) {
                    int work_item = work_pass * 128 + prep_tid;
                    int row = work_item / 16;
                    int segment = work_item % 16;
                    long long token = bos_4 + (long long)(chunk_idx_3 * 16 + row);
                    int token_valid = ((token < eos_4) ? 1 : 0);
                    long long gmem_base = (token * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(segment * 8);
                    float q_raw_vec[8];
                    float k_raw_vec[8];
                    unsigned int packed[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                        : "r"((smem_q_raw_prefetch_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
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
                    for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                        q_raw_vec[value_idx_3] = packed_f32[value_idx_3];
                    }
                    unsigned int packed_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                        : "r"((smem_kd_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
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
                    for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                        k_raw_vec[value_idx_4] = packed_0_f32[value_idx_4];
                    }
                    float2 _f2_7 = make_float2(0.0f, 0.0f);
                    float2 q_sum_pair = _f2_7;
                    float2 _f2_8 = make_float2(0.0f, 0.0f);
                    float2 k_sum_pair = _f2_8;
                    for (int elem_pair = 0; elem_pair < 4; elem_pair++) {
                        float2 _f2_9 = make_float2(q_raw_vec[elem_pair * 2], q_raw_vec[elem_pair * 2 + 1]);
                        float2 q_raw_pair = _f2_9;
                        float2 _f2_10 = make_float2(k_raw_vec[elem_pair * 2], k_raw_vec[elem_pair * 2 + 1]);
                        float2 k_raw_pair = _f2_10;
                        q_sum_pair = fma_f32x2(q_raw_pair, q_raw_pair, q_sum_pair);
                        k_sum_pair = fma_f32x2(k_raw_pair, k_raw_pair, k_sum_pair);
                    }
                    float q_sum = q_sum_pair.x + q_sum_pair.y;
                    float k_sum = k_sum_pair.x + k_sum_pair.y;
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
                    const float2 _scale2_0 = {q_inv, q_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_raw_vec)[_ls], _scale2_0);
                    const float2 _scale2_1 = {k_inv, k_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(k_raw_vec)[_ls], _scale2_1);
                    float qd_vec[8];
                    float kd_vec[8];
                    float ki_vec[8];
                    for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                        int col = segment * 8 + elem_in_segment;
                        float prefix = smem_gate_all[stage_f32_1 + row * 128 + col];
                        float common_log2 = smem_gate_all[stage_f32_1 + 1024 + col];
                        float _exp2_2 = approx_exp2(prefix - common_log2);
                        float decay = _exp2_2;
                        qd_vec[elem_in_segment] = decay;
                        kd_vec[elem_in_segment] = decay;
                        ki_vec[elem_in_segment] = k_raw_vec[elem_in_segment] / decay;
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                    {
                        const float2 _scale2_2 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_2);
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(scale);
                    float _cvt_f32_8 = __bfloat162float(_cvt_bf16_2);
                    float n16_decay_values[8];
                    float n16_inv_decay_values[8];
                    n16_decay_values[0] = smem_gate_all[stage_f32_1 + row * 128 + segment * 8];
                    n16_decay_values[1] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 1)];
                    n16_decay_values[2] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 2)];
                    n16_decay_values[3] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 3)];
                    n16_decay_values[4] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 4)];
                    n16_decay_values[5] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 5)];
                    n16_decay_values[6] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 6)];
                    n16_decay_values[7] = smem_gate_all[stage_f32_1 + row * 128 + (segment * 8 + 7)];
                    uint32_t n16_decay_values_bf16[4];
                    #pragma unroll
                    for (int _fe = 0; _fe < 4; _fe++) {
                        n16_decay_values[0 + _fe*2] = approx_exp2(n16_decay_values[0 + _fe*2]);
                        n16_decay_values[0 + _fe*2 + 1] = approx_exp2(n16_decay_values[0 + _fe*2 + 1]);
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(n16_decay_values[0 + _fe*2], n16_decay_values[0 + _fe*2 + 1]));
                        n16_decay_values_bf16[0 + _fe] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&n16_decay_values[_pair * 2])[0]), "=f"((&n16_decay_values[_pair * 2])[1])
                            : "r"(n16_decay_values_bf16[_pair]));
                    }
                    float _rcp_0 = approx_rcp(n16_decay_values[0]);
                    n16_inv_decay_values[0] = _rcp_0;
                    float _rcp_1 = approx_rcp(n16_decay_values[1]);
                    n16_inv_decay_values[1] = _rcp_1;
                    float _rcp_2 = approx_rcp(n16_decay_values[2]);
                    n16_inv_decay_values[2] = _rcp_2;
                    float _rcp_3 = approx_rcp(n16_decay_values[3]);
                    n16_inv_decay_values[3] = _rcp_3;
                    float _rcp_4 = approx_rcp(n16_decay_values[4]);
                    n16_inv_decay_values[4] = _rcp_4;
                    float _rcp_5 = approx_rcp(n16_decay_values[5]);
                    n16_inv_decay_values[5] = _rcp_5;
                    float _rcp_6 = approx_rcp(n16_decay_values[6]);
                    n16_inv_decay_values[6] = _rcp_6;
                    float _rcp_7 = approx_rcp(n16_decay_values[7]);
                    n16_inv_decay_values[7] = _rcp_7;
                    uint32_t q_raw_vec_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_raw_vec[_lp*2 + 0], q_raw_vec[_lp*2+1 + 0]));
                        q_raw_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&qd_vec[_pair * 2])[0]), "=f"((&qd_vec[_pair * 2])[1])
                            : "r"(q_raw_vec_bf16[_pair]));
                    }
                    uint32_t k_raw_vec_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(k_raw_vec[_lp*2 + 0], k_raw_vec[_lp*2+1 + 0]));
                        k_raw_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&kd_vec[_pair * 2])[0]), "=f"((&kd_vec[_pair * 2])[1])
                            : "r"(k_raw_vec_bf16[_pair]));
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&ki_vec[_pair * 2])[0]), "=f"((&ki_vec[_pair * 2])[1])
                            : "r"(k_raw_vec_bf16[_pair]));
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(n16_decay_values)[_ls]);
                    uint32_t qd_vec_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        qd_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&qd_vec[_pair * 2])[0]), "=f"((&qd_vec[_pair * 2])[1])
                            : "r"(qd_vec_bf16[_pair]));
                    }
                    const float2 _scale2_3 = {_cvt_f32_8, _cvt_f32_8};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_3);
                    uint32_t qd_vec_bf16_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        qd_vec_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&qd_vec[_pair * 2])[0]), "=f"((&qd_vec[_pair * 2])[1])
                            : "r"(qd_vec_bf16_1[_pair]));
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(n16_decay_values)[_ls]);
                    uint32_t kd_vec_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        kd_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&kd_vec[_pair * 2])[0]), "=f"((&kd_vec[_pair * 2])[1])
                            : "r"(kd_vec_bf16[_pair]));
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(ki_vec)[_ls], reinterpret_cast<const float2*>(n16_inv_decay_values)[_ls]);
                    uint32_t ki_vec_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        ki_vec_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&ki_vec[_pair * 2])[0]), "=f"((&ki_vec[_pair * 2])[1])
                            : "r"(ki_vec_bf16[_pair]));
                    }
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_6 = 0; word_6 < 4; word_6++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_6 * 4)), "r"((packed_2[word_6])));
                    }
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_7 = 0; word_7 < 4; word_7++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_7 * 4)), "r"((packed_3[word_7])));
                    }
                    unsigned int packed_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        packed_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_8 = 0; word_8 < 4; word_8++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + prep_stage * 21504 + (unsigned int)(segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 2048 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_8 * 4)), "r"((packed_4[word_8])));
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(aux_pairwise_inputs_ready_addr + (prep_stage) * 8);
                        }
                    }
                }
                unsigned int a_frag_1[4];
                unsigned int b_frag_1[4];
                float acc_1[8];
                long long tape_scaled_base = 0;
                {
                    mbarrier_wait(aux_pairwise_consumed_addr + (prep_stage) * 8, _phase_aux_pairwise_consumed);
                }
                {
                    int stage_f32_0 = prep_stage * 5376;
                    int restore_segment = lane & 15;
                    #pragma unroll 1
                    for (int restore_k_pass = 0; restore_k_pass < 2; restore_k_pass++) {
                        int restore_row = prep_local_warp * 4 + restore_k_pass * 2 + (lane >> 4);
                        float restore_ki_values[8];
                        float restore_kr_values[8];
                        unsigned int packed_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3]))
                            : "r"((smem_ki_addr + prep_stage * 21504 + (unsigned int)(restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32_1[_pair * 2])[0]), "=f"((&packed_f32_1[_pair * 2])[1])
                                : "r"(packed_1[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                            restore_ki_values[value_idx_5] = packed_f32_1[value_idx_5];
                        }
                        #pragma unroll
                        for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                            int restore_col = restore_segment * 8 + restore_elem;
                            __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(restore_ki_values[restore_elem]);
                            float _cvt_f32_21 = __bfloat162float(_cvt_bf16_11);
                            float restore_ki_carrier = _cvt_f32_21;
                            float restore_total = smem_restore_factor_all[stage_f32_0 + restore_col];
                            __nv_bfloat16 _cvt_bf16_12 = __float2bfloat16(restore_total);
                            float _cvt_f32_22 = __bfloat162float(_cvt_bf16_12);
                            float restore_total_carrier = _cvt_f32_22;
                            __nv_bfloat16 _cvt_bf16_13 = __float2bfloat16(restore_ki_carrier * restore_total_carrier);
                            float _cvt_f32_23 = __bfloat162float(_cvt_bf16_13);
                            restore_kr_values[restore_elem] = _cvt_f32_23;
                        }
                        unsigned int packed_0_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                            packed_0_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int word_9 = 0; word_9 < 4; word_9++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 21504 + (unsigned int)(restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 2048 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_9 * 4)), "r"((packed_0_1[word_9])));
                        }
                    }
                }
                {
                    mbarrier_wait(aux_inverse_ready_addr + (prep_stage) * 8, _phase_aux_inverse_ready);
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (prep_local_warp == 0) {
                    if (elect_sync()) {
                    }
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + (prep_stage) * 8, 4096);
                            {
                                int v_token = (int)(bos_4 + (long long)(chunk_idx_3 * 16));
                                tma_4d_gmem2smem(smem_v_addr + prep_stage * 21504, v_tma, 0, head_idx_3, v_token, 0, v_full_addr + (prep_stage) * 8);
                                tma_4d_gmem2smem(smem_v_addr + prep_stage * 21504 + 2048, v_tma, 0, head_idx_3, v_token, 1, v_full_addr + (prep_stage) * 8);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int v_load_iter = 0; v_load_iter < 2; v_load_iter++) {
                        int v_item = v_load_iter * 128 + prep_tid;
                        int row_1 = v_item / 16;
                        int segment_1 = v_item % 16;
                        long long token_1 = bos_4 + (long long)(chunk_idx_3 * 16 + row_1);
                        int token_valid_1 = ((token_1 < eos_4) ? 1 : 0);
                        long long v_src = (token_1 * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(segment_1 * 8);
                        int v_dst = smem_v_addr + prep_stage * 21504 + (unsigned int)((row_1 * 128 + segment_1 * 8) * 2);
                        {
                            int v_panel = segment_1 / 8;
                            int v_panel_segment = segment_1 % 8;
                            int v_row_addr = smem_v_addr + prep_stage * 21504 + (unsigned int)(v_panel * 16 * 64 * 2) + (unsigned int)(row_1 * 64 * 2);
                            v_dst = (v_row_addr + (v_panel_segment * 8 * 2 ^ (v_row_addr >> 7 & 7) << 4));
                        }
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(v_dst), "l"(v + v_src), "r"((token_valid_1 != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(v_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                for (int _advance = 0; _advance < 5; _advance++) {
                    prep_stage += 1;
                    if (prep_stage == 5) { prep_stage = 0; _phase_raw_inputs_free ^= 1; _phase_gate_raw_full ^= 1; _phase_smem_free ^= 1; _phase_v_free ^= 1; _phase_qk_raw_full ^= 1; _phase_short_beta_ready ^= 1; _phase_aux_pairwise_consumed ^= 1; _phase_aux_inverse_ready ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

