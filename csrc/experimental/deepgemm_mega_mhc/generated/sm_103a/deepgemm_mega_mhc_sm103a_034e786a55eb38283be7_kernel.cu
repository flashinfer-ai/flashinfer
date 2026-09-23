// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see ../DEEPGEMM_NOTICE.txt.

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Deepgemm requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) DeepgemmTensorMap { uint64_t opaque[16]; };
struct __align__(64) DeepgemmTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(DeepgemmTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(DeepgemmTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(DeepgemmTensorMap) >= alignof(CUtensorMap), "DeepgemmTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DEEPGEMM_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_A_TMEM_OFFSET 0
#define TMEM_ACCUM_OFFSET 128
#define NUM_MAIN_STAGES 1
#define SMEM_RESIDUAL_OFF 0
#define SMEM_RESIDUAL_STAGE_BYTES 32768
#define SMEM_RESIDUAL_STRIDE 32768
#define SMEM_X_OFF 131072
#define SMEM_X_STAGE_BYTES 8192
#define SMEM_X_STRIDE 8192
#define SMEM_RESIDUAL_PAIRS_OFF 0
#define SMEM_RESIDUAL_PAIRS_STAGE_BYTES 131072
#define SMEM_RESIDUAL_PAIRS_STRIDE 131072
#define SMEM_X_PAIRS_OFF 131072
#define SMEM_X_PAIRS_STAGE_BYTES 32768
#define SMEM_X_PAIRS_STRIDE 32768
#define SMEM_FN_OFF 163840
#define SMEM_FN_STAGE_BYTES 24576
#define SMEM_FN_STRIDE 24576
#define SMEM_FN_ATOMS_OFF 163840
#define SMEM_FN_ATOMS_STAGE_BYTES 3072
#define SMEM_FN_ATOMS_STRIDE 3072
#define SMEM_PRE_OFF 212992
#define SMEM_PRE_STAGE_BYTES 1024
#define SMEM_PRE_STRIDE 1024
#define SMEM_POST_OFF 215040
#define SMEM_POST_STAGE_BYTES 1024
#define SMEM_POST_STRIDE 1024
#define SMEM_COMB_OFF 217088
#define SMEM_COMB_STAGE_BYTES 4096
#define SMEM_COMB_STRIDE 4096
#define SMEM_HC_SUMS_OFF 225280
#define SMEM_HC_SUMS_STAGE_BYTES 512
#define SMEM_HC_SUMS_STRIDE 512
#define SMEM_X1_SUMS_OFF 225792
#define SMEM_X1_SUMS_STAGE_BYTES 512
#define SMEM_X1_SUMS_STRIDE 512
#define SMEM_NORMAL_SCRATCH_OFF 226304
#define SMEM_NORMAL_SCRATCH_STAGE_BYTES 256
#define SMEM_NORMAL_SCRATCH_STRIDE 256
#define SMEM_NORMAL_CONTROL_OFF 226304
#define SMEM_NORMAL_CONTROL_STAGE_BYTES 256
#define SMEM_NORMAL_CONTROL_STRIDE 256
#define SMEM_QUEUE_OFF 226560
#define SMEM_QUEUE_STAGE_BYTES 4
#define SMEM_QUEUE_STRIDE 4
#define SMEM_LAUNCH_EPOCH_OFF 226568
#define SMEM_LAUNCH_EPOCH_STAGE_BYTES 8
#define SMEM_LAUNCH_EPOCH_STRIDE 8
#define SMEM_TOTAL 227328
#define THREADS 768

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

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void tcgen05_mma_tf32(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n\t"
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
        "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], db, %4, p;\n\t"
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

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("add.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("sub.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.f32x2 %0, %1, %2;"
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

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
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
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(768, 1) void
kernel_deepgemm_mega_mhc_sm103a_034e786a55eb38283be7(const __grid_constant__ CUtensorMap residual_map, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap fn_map, const __grid_constant__ CUtensorMap post_map, const __grid_constant__ CUtensorMap comb_map, const __grid_constant__ CUtensorMap prev_map, const __grid_constant__ CUtensorMap new_residual_map, const __grid_constant__ CUtensorMap y_map, float* __restrict__ mix_scales, float* __restrict__ mix_bases, float* __restrict__ new_prev_mix, float* __restrict__ new_post_mix, float* __restrict__ new_comb_res_mix, __nv_bfloat16* __restrict__ rmsnorm_weight, __nv_bfloat16* __restrict__ new_residual, __nv_bfloat16* __restrict__ y_bf16, uint8_t* __restrict__ y_fp8, unsigned int* __restrict__ y_primary_sf, unsigned int* __restrict__ y_shared_sf, float* __restrict__ scratch, unsigned long long* __restrict__ split_barriers, unsigned long long* __restrict__ launch_epochs, unsigned int num_tokens, float hc_norm_eps, float hc_pre_eps, float hc_post_scale, float sinkhorn_eps, unsigned int num_sinkhorn_iters, float rmsnorm_eps, float rmsnorm_scale, unsigned long long primary_sf_stride_token, unsigned long long primary_sf_stride_word, unsigned long long shared_sf_stride_word)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 226688;
    #define full_io_addr (mbar_base + 0)
    #define empty_io_addr (mbar_base + 32)
    #define store_ready_addr (mbar_base + 64)
    #define full_fn_addr (mbar_base + 96)
    #define empty_fn_addr (mbar_base + 112)
    #define full_a_addr (mbar_base + 128)
    #define empty_a_addr (mbar_base + 160)
    #define full_accum_addr (mbar_base + 192)
    #define empty_accum_addr (mbar_base + 208)
    #define full_coeff_addr (mbar_base + 224)
    #define empty_coeff_addr (mbar_base + 240)
    #define full_stats_addr (mbar_base + 256)
    #define empty_stats_addr (mbar_base + 264)
    #define mix_arrival_addr (mbar_base + 272)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* residual = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int residual_addr = smem + 0;
    __nv_bfloat16* x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int x_addr = smem + 131072;
    unsigned int* residual_pairs = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int residual_pairs_addr = smem + 0;
    unsigned int* x_pairs = reinterpret_cast<unsigned int*>(smem_raw + 131072);
    const int x_pairs_addr = smem + 131072;
    float* fn = reinterpret_cast<float*>(smem_raw + 163840);
    const int fn_addr = smem + 163840;
    float* fn_atoms = reinterpret_cast<float*>(smem_raw + 163840);
    const int fn_atoms_addr = smem + 163840;
    float* pre = reinterpret_cast<float*>(smem_raw + 212992);
    const int pre_addr = smem + 212992;
    float* post = reinterpret_cast<float*>(smem_raw + 215040);
    const int post_addr = smem + 215040;
    float* comb = reinterpret_cast<float*>(smem_raw + 217088);
    const int comb_addr = smem + 217088;
    float* hc_sums = reinterpret_cast<float*>(smem_raw + 225280);
    const int hc_sums_addr = smem + 225280;
    float* x1_sums = reinterpret_cast<float*>(smem_raw + 225792);
    const int x1_sums_addr = smem + 225792;
    float* normal_scratch = reinterpret_cast<float*>(smem_raw + 226304);
    const int normal_scratch_addr = smem + 226304;
    unsigned int* normal_control = reinterpret_cast<unsigned int*>(smem_raw + 226304);
    const int normal_control_addr = smem + 226304;
    unsigned int* queue = reinterpret_cast<unsigned int*>(smem_raw + 226560);
    const int queue_addr = smem + 226560;
    unsigned long long* launch_epoch = reinterpret_cast<unsigned long long*>(smem_raw + 226568);
    const int launch_epoch_addr = smem + 226568;
    unsigned int lane_in_warp = threadIdx.x % 32;
    if (warp == 0) {
        if (elect_sync()) {
            unsigned long long _atomic_old_0 = atomicAdd(&launch_epochs[bid], 1);
            launch_epoch[0] = _atomic_old_0;
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            queue[0] = 0;
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&residual_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&x_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&fn_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&post_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&comb_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&prev_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&new_residual_map))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&y_map))) : "memory");
        }
    }

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 35 barriers)
    // Mbarriers at smem_raw[226688..226968)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_io: 4 barriers, init_count=1
            mbarrier_init(smem + 226688, 1);
            mbarrier_init(smem + 226696, 1);
            mbarrier_init(smem + 226704, 1);
            mbarrier_init(smem + 226712, 1);
            // empty_io: 4 barriers, init_count=1
            mbarrier_init(smem + 226720, 1);
            mbarrier_init(smem + 226728, 1);
            mbarrier_init(smem + 226736, 1);
            mbarrier_init(smem + 226744, 1);
            // store_ready: 4 barriers, init_count=8
            mbarrier_init(smem + 226752, 8);
            mbarrier_init(smem + 226760, 8);
            mbarrier_init(smem + 226768, 8);
            mbarrier_init(smem + 226776, 8);
            // full_coeff: 2 barriers, init_count=1
            mbarrier_init(smem + 226912, 1);
            mbarrier_init(smem + 226920, 1);
            // empty_coeff: 2 barriers, init_count=8
            mbarrier_init(smem + 226928, 8);
            mbarrier_init(smem + 226936, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 226784, 1);
            mbarrier_init(smem + 226792, 1);
            // empty_fn: 2 barriers, init_count=1
            mbarrier_init(smem + 226800, 1);
            mbarrier_init(smem + 226808, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full_a: 4 barriers, init_count=128
            mbarrier_init(smem + 226816, 128);
            mbarrier_init(smem + 226824, 128);
            mbarrier_init(smem + 226832, 128);
            mbarrier_init(smem + 226840, 128);
            // empty_a: 4 barriers, init_count=1
            mbarrier_init(smem + 226848, 1);
            mbarrier_init(smem + 226856, 1);
            mbarrier_init(smem + 226864, 1);
            mbarrier_init(smem + 226872, 1);
            // full_accum: 2 barriers, init_count=1
            mbarrier_init(smem + 226880, 1);
            mbarrier_init(smem + 226888, 1);
            // empty_accum: 2 barriers, init_count=4
            mbarrier_init(smem + 226896, 4);
            mbarrier_init(smem + 226904, 4);
            // full_stats: 1 barriers, init_count=256
            mbarrier_init(smem + 226944, 256);
            // empty_stats: 1 barriers, init_count=160
            mbarrier_init(smem + 226952, 160);
            // mix_arrival: 1 barriers, init_count=4
            mbarrier_init(smem + 226960, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 176 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 226968);
    if (warp == 3) {
        int _tmem_hold = smem + 226968;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_a_tmem = taddr;
    const int tmem_accum = taddr + 128;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == 5) {
        if (elect_sync()) {
            for (int mb = bid; mb < (num_tokens + 63) / 64; mb += 152) {
                asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb) * 16))), "l"(static_cast<unsigned long long>((launch_epoch[0] + 1) * 128)) : "memory");
            }
        }
    }
    if (warp == 4) {
        if (elect_sync()) {
            for (int mb_1 = bid; mb_1 < (num_tokens + 63) / 64; mb_1 += 152) {
                asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + (mb_1 * 16))), "l"(static_cast<unsigned long long>((launch_epoch[0] + 1) * 128)) : "memory");
            }
        }
    }
    asm volatile("barrier.sync 0, 768;" ::: "memory");

    // ---- Role: control ----
    if (warp <= 3) {
        { // control_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
            if (warp < 2) {
                unsigned int seq = 0;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int task = bid; task < (num_tokens + 63) / 64 * 16; task += 152) {
                        #pragma unroll 1
                        for (int kb = 0; kb < 5 + ((task % 16 < 0) ? 1 : 0); kb++) {
                            unsigned int extra = ((task % 16 < 0) ? (unsigned int)(task % 16) : (unsigned int)0);
                            unsigned int hidden = ((unsigned int)(task % 16 * 5) + extra) * 64 + (unsigned int)(kb * 64);
                            if (warp == 0) {
                                unsigned int stage = seq % 4;
                                mbarrier_wait(empty_io_addr + (stage) * 8, seq / 4 % 2 ^ 1);
                                tma_3d_gmem2smem(residual_addr + stage * 32768, (&residual_map), hidden, task / 16 * 64, 0, full_io_addr + (stage) * 8);
                                tma_2d_gmem2smem(x_addr + stage * 8192, (&x_map), hidden, task / 16 * 64, full_io_addr + (stage) * 8);
                                mbarrier_arrive_expect_tx(full_io_addr + (stage) * 8, 40960);
                            } else {
                                unsigned int stage_1 = seq % 2;
                                mbarrier_wait(empty_fn_addr + (stage_1) * 8, seq / 2 % 2 ^ 1);
                                tma_3d_gmem2smem(fn_addr + stage_1 * 24576, (&fn_map), hidden, 0, 0, full_fn_addr + (stage_1) * 8);
                                tma_3d_gmem2smem(fn_addr + stage_1 * 24576 + 12288, (&fn_map), hidden + 32, 0, 0, full_fn_addr + (stage_1) * 8);
                                mbarrier_arrive_expect_tx(full_fn_addr + (stage_1) * 8, 24576);
                            }
                            seq += 1;
                        }
                    }
                }
                __syncwarp();
            } else if (warp == 2) {
                unsigned int fseq = 0;
                unsigned int aseq = 0;
                unsigned int cseq = 0;
                #pragma unroll 1
                for (int task_1 = bid; task_1 < (num_tokens + 63) / 64 * 16; task_1 += 152) {
                    unsigned int cstage = cseq % 2;
                    mbarrier_wait(empty_accum_addr + (cstage) * 8, cseq / 2 % 2 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kb_1 = 0; kb_1 < 5 + ((task_1 % 16 < 0) ? 1 : 0); kb_1++) {
                        unsigned int fstage = fseq % 2;
                        mbarrier_wait(full_fn_addr + (fstage) * 8, fseq / 2 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int astage = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_0 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage * 32, _mma_b_lo_0, 0x40004040, 67504400, ((kb_1 == 0) ? 0 : 1));
                        int _mma_b_lo_1 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 8), _mma_b_lo_1, 0x40004040, 67504400, 1);
                        int _mma_b_lo_2 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 16), _mma_b_lo_2, 0x40004040, 67504400, 1);
                        int _mma_b_lo_3 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage * 32 + 24), _mma_b_lo_3, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage) * 8);
                        aseq += 1;
                        unsigned int astage_0 = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage_0) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_4 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_0 * 32, _mma_b_lo_4, 0x40004040, 67504400, 1);
                        int _mma_b_lo_5 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 8), _mma_b_lo_5, 0x40004040, 67504400, 1);
                        int _mma_b_lo_6 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 16), _mma_b_lo_6, 0x40004040, 67504400, 1);
                        int _mma_b_lo_7 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_0 * 32 + 24), _mma_b_lo_7, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage_0) * 8);
                        aseq += 1;
                        unsigned int astage_1 = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage_1) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_8 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_1 * 32, _mma_b_lo_8, 0x40004040, 67504400, 1);
                        int _mma_b_lo_9 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_1 * 32 + 8), _mma_b_lo_9, 0x40004040, 67504400, 1);
                        int _mma_b_lo_10 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_1 * 32 + 16), _mma_b_lo_10, 0x40004040, 67504400, 1);
                        int _mma_b_lo_11 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_1 * 32 + 24), _mma_b_lo_11, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage_1) * 8);
                        aseq += 1;
                        unsigned int astage_2 = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage_2) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_12 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_2 * 32, _mma_b_lo_12, 0x40004040, 67504400, 1);
                        int _mma_b_lo_13 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 8), _mma_b_lo_13, 0x40004040, 67504400, 1);
                        int _mma_b_lo_14 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 16), _mma_b_lo_14, 0x40004040, 67504400, 1);
                        int _mma_b_lo_15 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_2 * 32 + 24), _mma_b_lo_15, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage_2) * 8);
                        aseq += 1;
                        unsigned int astage_3 = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage_3) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_16 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8 + 4) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_3 * 32, _mma_b_lo_16, 0x40004040, 67504400, 1);
                        int _mma_b_lo_17 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 8), _mma_b_lo_17, 0x40004040, 67504400, 1);
                        int _mma_b_lo_18 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 16), _mma_b_lo_18, 0x40004040, 67504400, 1);
                        int _mma_b_lo_19 = make_warp_uniform((((fn_atoms_addr) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_3 * 32 + 24), _mma_b_lo_19, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage_3) * 8);
                        aseq += 1;
                        unsigned int astage_4 = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage_4) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_20 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8 + 4) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_4 * 32, _mma_b_lo_20, 0x40004040, 67504400, 1);
                        int _mma_b_lo_21 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 8), _mma_b_lo_21, 0x40004040, 67504400, 1);
                        int _mma_b_lo_22 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 16), _mma_b_lo_22, 0x40004040, 67504400, 1);
                        int _mma_b_lo_23 = make_warp_uniform((((fn_atoms_addr + 32) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_4 * 32 + 24), _mma_b_lo_23, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage_4) * 8);
                        aseq += 1;
                        unsigned int astage_5 = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage_5) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_24 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8 + 4) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_5 * 32, _mma_b_lo_24, 0x40004040, 67504400, 1);
                        int _mma_b_lo_25 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_5 * 32 + 8), _mma_b_lo_25, 0x40004040, 67504400, 1);
                        int _mma_b_lo_26 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_5 * 32 + 16), _mma_b_lo_26, 0x40004040, 67504400, 1);
                        int _mma_b_lo_27 = make_warp_uniform((((fn_atoms_addr + 64) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_5 * 32 + 24), _mma_b_lo_27, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage_5) * 8);
                        aseq += 1;
                        unsigned int astage_6 = aseq % 4;
                        mbarrier_wait(full_a_addr + (astage_6) * 8, aseq / 4 % 2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_28 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8 + 4) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + astage_6 * 32, _mma_b_lo_28, 0x40004040, 67504400, 1);
                        int _mma_b_lo_29 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 1) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 8), _mma_b_lo_29, 0x40004040, 67504400, 1);
                        int _mma_b_lo_30 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 2) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 16), _mma_b_lo_30, 0x40004040, 67504400, 1);
                        int _mma_b_lo_31 = make_warp_uniform((((fn_atoms_addr + 96) >> 4) & 0x3FFF) + (fstage * 8 + 4 + 3) * 192);
                        mma_ts_step((tmem_accum + (cstage * 24)), (unsigned int)tmem_a_tmem + (astage_6 * 32 + 24), _mma_b_lo_31, 0x40004040, 67504400, 1);
                        elect_commit(empty_a_addr + (astage_6) * 8);
                        aseq += 1;
                        elect_commit(empty_fn_addr + (fstage) * 8);
                        fseq += 1;
                    }
                    elect_commit(full_accum_addr + (cstage) * 8);
                    cseq += 1;
                }
            } else {
                unsigned int io_seq = 0;
                unsigned int coeff_seq = 0;
                if ((unsigned int)bid < (num_tokens + 63) / 64 * 16) {
                    mbarrier_wait(empty_coeff_addr, 1);
                    if (elect_sync()) {
                        tma_2d_gmem2smem(post_addr, (&post_map), 0, bid / 16 * 64, full_coeff_addr);
                        tma_2d_gmem2smem(comb_addr, (&comb_map), 0, bid / 16 * 64, full_coeff_addr);
                        tma_2d_gmem2smem(pre_addr, (&prev_map), 0, bid / 16 * 64, full_coeff_addr);
                        mbarrier_arrive_expect_tx(full_coeff_addr, 6144);
                    }
                    __syncwarp();
                    coeff_seq = 1;
                }
                #pragma unroll 1
                for (int task_2 = bid; task_2 < (num_tokens + 63) / 64 * 16; task_2 += 152) {
                    unsigned int mb_2 = task_2 / 16;
                    unsigned long long base = (launch_epoch[0] + 1) * 128;
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + (mb_2 * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(base)) >= static_cast<unsigned long long>(16));
                    }
                    #pragma unroll 1
                    for (int kb_2 = 0; kb_2 < 5 + ((task_2 % 16 < 0) ? 1 : 0); kb_2++) {
                        unsigned int stage_2 = io_seq % 4;
                        mbarrier_wait(store_ready_addr + (stage_2) * 8, io_seq / 4 % 2);
                        unsigned int prefetch = kb_2 + 1 == 5 + ((task_2 % 16 < 0) ? 1 : 0) && (unsigned int)(task_2 + 152) < (num_tokens + 63) / 64 * 16;
                        if (prefetch != 0) {
                            mbarrier_wait(empty_coeff_addr + (coeff_seq % 2) * 8, coeff_seq / 2 % 2 ^ 1);
                        }
                        if (elect_sync()) {
                            unsigned int extra_1 = ((task_2 % 16 < 0) ? (unsigned int)(task_2 % 16) : (unsigned int)0);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2, %3}], [%4], %5;"
                                :: "l"((&new_residual_map)), "r"(((unsigned int)(task_2 % 16 * 5) + extra_1) * 64 + (unsigned int)(kb_2 * 64)), "r"(mb_2 * 64), "r"(0), "r"(residual_addr + stage_2 * 32768), "l"(0x1000000000000000ULL) : "memory");
                            unsigned int extra_0 = ((task_2 % 16 < 0) ? (unsigned int)(task_2 % 16) : (unsigned int)0);
                            asm volatile(
                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                " [%0, {%1, %2}], [%3], %4;"
                                :: "l"((&y_map)), "r"(((unsigned int)(task_2 % 16 * 5) + extra_0) * 64 + (unsigned int)(kb_2 * 64)), "r"(mb_2 * 64), "r"(x_addr + stage_2 * 8192), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile("cp.async.bulk.commit_group;");
                            if (prefetch != 0) {
                                tma_2d_gmem2smem(post_addr + coeff_seq % 2 * 1024, (&post_map), 0, (task_2 + 152) / 16 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                tma_2d_gmem2smem(comb_addr + coeff_seq % 2 * 4096, (&comb_map), 0, (task_2 + 152) / 16 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                tma_2d_gmem2smem(pre_addr + coeff_seq % 2 * 1024, (&prev_map), 0, (task_2 + 152) / 16 * 64, full_coeff_addr + (coeff_seq % 2) * 8);
                                mbarrier_arrive_expect_tx(full_coeff_addr + (coeff_seq % 2) * 8, 6144);
                            }
                            asm volatile("cp.async.bulk.wait_group 0;");
                            mbarrier_arrive(empty_io_addr + (stage_2) * 8);
                        }
                        if (prefetch != 0) {
                            coeff_seq += 1;
                        }
                        io_seq += 1;
                    }
                    unsigned int phase = (task_2 - bid) / 152 % 2;
                    mbarrier_wait(full_stats_addr, phase);
                    scratch[(unsigned long long)((num_tokens + 63) / 64 * 16) * 1536 + ((1) ? (unsigned long long)((num_tokens + 63) / 64 * 16) * 64 : (unsigned long long)0) + (unsigned long long)task_2 * 64 + (unsigned long long)lane_in_warp] = x1_sums[lane_in_warp] + x1_sums[64 + lane_in_warp];
                    scratch[(unsigned long long)((num_tokens + 63) / 64 * 16) * 1536 + ((1) ? (unsigned long long)((num_tokens + 63) / 64 * 16) * 64 : (unsigned long long)0) + (unsigned long long)task_2 * 64 + (unsigned long long)lane_in_warp + 32] = x1_sums[lane_in_warp + 32] + x1_sums[96 + lane_in_warp];
                    __syncwarp();
                    mbarrier_arrive(empty_stats_addr);
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                        asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + (mb_2 * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
                        #elif defined(__CUDA_ARCH__)
                        #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
                        #endif
                    }
                }
            }
        }
    }
    // ---- Role: post0 ----
    if (warp >= 4 && warp <= 7) {
        { // post0_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
            unsigned int io_seq_1 = 0;
            unsigned int a_seq = 0;
            unsigned int coeff_seq_1 = 0;
            unsigned int first_row = warp % 4 * 16 + lane_in_warp / 4;
            #pragma unroll 1
            for (int task_3 = bid; task_3 < (num_tokens + 63) / 64 * 16; task_3 += 152) {
                unsigned int coeff_stage = coeff_seq_1 % 2;
                mbarrier_wait(full_coeff_addr + (coeff_stage) * 8, coeff_seq_1 / 2 % 2);
                float post_coeff[8];
                float pre_coeff[8];
                float comb_coeff[32];
                #pragma unroll
                for (int ri = 0; ri < 2; ri++) {
                    unsigned int row = first_row + (unsigned int)(ri * 8);
                    #pragma unroll
                    for (int route = 0; route < 4; route++) {
                        post_coeff[ri * 4 + route] = post[coeff_stage * 256 + row * 4 + (unsigned int)route];
                        pre_coeff[ri * 4 + route] = pre[coeff_stage * 256 + row * 4 + (unsigned int)route];
                    }
                    #pragma unroll
                    for (int ci = 0; ci < 16; ci++) {
                        comb_coeff[ri * 16 + ci] = comb[coeff_stage * 1024 + row * 16 + ((unsigned int)(ci / 4) ^ row / 2 & 3) * 4 + (unsigned int)(ci % 4)];
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(empty_coeff_addr + (coeff_stage) * 8);
                }
                coeff_seq_1 += 1;
                float hc_sq[4];
                float x1_sq[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    hc_sq[j] = 0.0f;
                    x1_sq[j] = 0.0f;
                }
                #pragma unroll 1
                for (int kb_3 = 0; kb_3 < 5 + ((task_3 % 16 < 0) ? 1 : 0); kb_3++) {
                    unsigned int io_stage = io_seq_1 % 4;
                    mbarrier_wait(full_io_addr + (io_stage) * 8, io_seq_1 / 4 % 2);
                    #pragma unroll
                    for (int atom = 0; atom < 8; atom += 2) {
                        unsigned int pair = (unsigned int)(atom * 4) + lane_in_warp % 4;
                        float xv[4];
                        float rv[16];
                        #pragma unroll
                        for (int ri_1 = 0; ri_1 < 2; ri_1++) {
                            unsigned int row_1 = first_row + (unsigned int)(ri_1 * 8);
                            unsigned int offset = row_1 * 32 + (pair / 4 ^ row_1 & 7) * 4 + pair % 4;
                            unsigned int x_word = x_pairs[io_stage * 2048 + offset];
                            xv[ri_1 * 2] = __uint_as_float(x_word << 16);
                            xv[ri_1 * 2 + 1] = __uint_as_float(x_word & 4294901760);
                            #pragma unroll
                            for (int route_1 = 0; route_1 < 4; route_1++) {
                                unsigned int word = residual_pairs[io_stage * 8192 + (unsigned int)(route_1 * 2048) + offset];
                                rv[ri_1 * 8 + route_1 * 2] = __uint_as_float(word << 16);
                                rv[ri_1 * 8 + route_1 * 2 + 1] = __uint_as_float(word & 4294901760);
                            }
                        }
                        unsigned int ast = a_seq % 4;
                        mbarrier_wait(empty_a_addr + (ast) * 8, a_seq / 4 % 2 ^ 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float tmem_values[16];
                        #pragma unroll
                        for (int ri_2 = 0; ri_2 < 2; ri_2++) {
                            unsigned int row_2 = first_row + (unsigned int)(ri_2 * 8);
                            unsigned int offset_1 = row_2 * 32 + (pair / 4 ^ row_2 & 7) * 4 + pair % 4;
                            float2 _f2_0 = make_float2(0.0f, 0.0f);
                            float2 x1v = _f2_0;
                            #pragma unroll
                            for (int route_2 = 0; route_2 < 4; route_2++) {
                                float coeff = post_coeff[ri_2 * 4 + route_2];
                                float2 _f2_1 = make_float2(xv[ri_2 * 2], xv[ri_2 * 2 + 1]);
                                float2 _f2_2 = make_float2(coeff, coeff);
                                float2 _mul_f32x2_0;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&_f2_2));
                                float2 value = _mul_f32x2_0;
                                #pragma unroll
                                for (int input_route = 0; input_route < 4; input_route++) {
                                    float cc = comb_coeff[ri_2 * 16 + input_route * 4 + route_2];
                                    float2 _f2_3 = make_float2(rv[ri_2 * 8 + input_route * 2], rv[ri_2 * 8 + input_route * 2 + 1]);
                                    float2 _f2_4 = make_float2(cc, cc);
                                    value = fma_f32x2_rn_ftz(_f2_3, _f2_4, value);
                                }
                                __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(value.x, value.y));
                                unsigned int word_1 = __as_u32(_bf16x2_0);
                                residual_pairs[io_stage * 8192 + (unsigned int)(route_2 * 2048) + offset_1] = word_1;
                                float lo = __uint_as_float(word_1 << 16);
                                float hi = __uint_as_float(word_1 & 4294901760);
                                tmem_values[route_2 * 4 + ri_2 * 2] = lo;
                                tmem_values[route_2 * 4 + ri_2 * 2 + 1] = hi;
                                float2 _f2_5 = make_float2(lo, hi);
                                float2 rounded_pair = _f2_5;
                                float2 _f2_6 = make_float2(hc_sq[ri_2 * 2], hc_sq[ri_2 * 2 + 1]);
                                float2 squares = fma_f32x2_rn_ftz(rounded_pair, rounded_pair, _f2_6);
                                hc_sq[ri_2 * 2] = squares.x;
                                hc_sq[ri_2 * 2 + 1] = squares.y;
                                float pc = pre_coeff[ri_2 * 4 + route_2];
                                float2 _f2_7 = make_float2(pc, pc);
                                x1v = fma_f32x2_rn_ftz(rounded_pair, _f2_7, x1v);
                            }
                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(x1v.x, x1v.y));
                            unsigned int x1_word = __as_u32(_bf16x2_1);
                            x_pairs[io_stage * 2048 + offset_1] = x1_word;
                            float _bf16x2_fma_f32_0[2] = {x1_sq[ri_2 * 2], x1_sq[ri_2 * 2 + 1]};
                            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(_bf16x2_fma_f32_0[0]) : "h"(static_cast<uint16_t>((x1_word))), "h"(static_cast<uint16_t>((x1_word))));
                            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(_bf16x2_fma_f32_0[1]) : "h"(static_cast<uint16_t>((x1_word) >> 16)), "h"(static_cast<uint16_t>((x1_word) >> 16)));
                            x1_sq[ri_2 * 2] = _bf16x2_fma_f32_0[0];
                            x1_sq[ri_2 * 2 + 1] = _bf16x2_fma_f32_0[1];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"((unsigned int)tmem_a_tmem + ast * 32), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[0])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[1])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[2])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[3])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[4])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[5])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[6])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[7])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[8])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[9])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[10])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[11])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[12])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[13])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[14])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(full_a_addr + (ast) * 8);
                        a_seq += 2;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(store_ready_addr + (io_stage) * 8);
                    }
                    io_seq_1 += 1;
                }
                unsigned int phase_1 = (task_3 - bid) / 152 % 2;
                mbarrier_wait(empty_stats_addr, phase_1 ^ 1);
                #pragma unroll
                for (int ri_3 = 0; ri_3 < 2; ri_3++) {
                    unsigned int row_3 = first_row + (unsigned int)(ri_3 * 8);
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, hc_sq[ri_3 * 2] + hc_sq[ri_3 * 2 + 1], 2);
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, hc_sq[ri_3 * 2] + hc_sq[ri_3 * 2 + 1] + _shfl_xor_0, 1);
                    if (lane_in_warp % 4 == 0) {
                        hc_sums[row_3] = hc_sq[ri_3 * 2] + hc_sq[ri_3 * 2 + 1] + _shfl_xor_0 + _shfl_xor_1;
                    }
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, x1_sq[ri_3 * 2] + x1_sq[ri_3 * 2 + 1], 2);
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, x1_sq[ri_3 * 2] + x1_sq[ri_3 * 2 + 1] + _shfl_xor_2, 1);
                    if (lane_in_warp % 4 == 0) {
                        x1_sums[row_3] = x1_sq[ri_3 * 2] + x1_sq[ri_3 * 2 + 1] + _shfl_xor_2 + _shfl_xor_3;
                    }
                }
                __syncwarp();
                mbarrier_arrive(full_stats_addr);
                __syncwarp();
            }
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        }
    }
    // ---- Role: post1 ----
    if (warp >= 8 && warp <= 11) {
        { // post1_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
            unsigned int io_seq_2 = 0;
            unsigned int a_seq_1 = 1;
            unsigned int coeff_seq_2 = 0;
            unsigned int first_row_1 = warp % 4 * 16 + lane_in_warp / 4;
            #pragma unroll 1
            for (int task_4 = bid; task_4 < (num_tokens + 63) / 64 * 16; task_4 += 152) {
                unsigned int coeff_stage_1 = coeff_seq_2 % 2;
                mbarrier_wait(full_coeff_addr + (coeff_stage_1) * 8, coeff_seq_2 / 2 % 2);
                float post_coeff_1[8];
                float pre_coeff_1[8];
                float comb_coeff_1[32];
                #pragma unroll
                for (int ri_4 = 0; ri_4 < 2; ri_4++) {
                    unsigned int row_4 = first_row_1 + (unsigned int)(ri_4 * 8);
                    #pragma unroll
                    for (int route_3 = 0; route_3 < 4; route_3++) {
                        post_coeff_1[ri_4 * 4 + route_3] = post[coeff_stage_1 * 256 + row_4 * 4 + (unsigned int)route_3];
                        pre_coeff_1[ri_4 * 4 + route_3] = pre[coeff_stage_1 * 256 + row_4 * 4 + (unsigned int)route_3];
                    }
                    #pragma unroll
                    for (int ci_1 = 0; ci_1 < 16; ci_1++) {
                        comb_coeff_1[ri_4 * 16 + ci_1] = comb[coeff_stage_1 * 1024 + row_4 * 16 + ((unsigned int)(ci_1 / 4) ^ row_4 / 2 & 3) * 4 + (unsigned int)(ci_1 % 4)];
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(empty_coeff_addr + (coeff_stage_1) * 8);
                }
                coeff_seq_2 += 1;
                float hc_sq_1[4];
                float x1_sq_1[4];
                #pragma unroll
                for (int j_1 = 0; j_1 < 4; j_1++) {
                    hc_sq_1[j_1] = 0.0f;
                    x1_sq_1[j_1] = 0.0f;
                }
                #pragma unroll 1
                for (int kb_4 = 0; kb_4 < 5 + ((task_4 % 16 < 0) ? 1 : 0); kb_4++) {
                    unsigned int io_stage_1 = io_seq_2 % 4;
                    mbarrier_wait(full_io_addr + (io_stage_1) * 8, io_seq_2 / 4 % 2);
                    #pragma unroll
                    for (int atom_1 = 1; atom_1 < 8; atom_1 += 2) {
                        unsigned int pair_1 = (unsigned int)(atom_1 * 4) + lane_in_warp % 4;
                        float xv_1[4];
                        float rv_1[16];
                        #pragma unroll
                        for (int ri_5 = 0; ri_5 < 2; ri_5++) {
                            unsigned int row_5 = first_row_1 + (unsigned int)(ri_5 * 8);
                            unsigned int offset_2 = row_5 * 32 + (pair_1 / 4 ^ row_5 & 7) * 4 + pair_1 % 4;
                            unsigned int x_word_1 = x_pairs[io_stage_1 * 2048 + offset_2];
                            xv_1[ri_5 * 2] = __uint_as_float(x_word_1 << 16);
                            xv_1[ri_5 * 2 + 1] = __uint_as_float(x_word_1 & 4294901760);
                            #pragma unroll
                            for (int route_4 = 0; route_4 < 4; route_4++) {
                                unsigned int word_2 = residual_pairs[io_stage_1 * 8192 + (unsigned int)(route_4 * 2048) + offset_2];
                                rv_1[ri_5 * 8 + route_4 * 2] = __uint_as_float(word_2 << 16);
                                rv_1[ri_5 * 8 + route_4 * 2 + 1] = __uint_as_float(word_2 & 4294901760);
                            }
                        }
                        unsigned int ast_1 = a_seq_1 % 4;
                        mbarrier_wait(empty_a_addr + (ast_1) * 8, a_seq_1 / 4 % 2 ^ 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float tmem_values_1[16];
                        #pragma unroll
                        for (int ri_6 = 0; ri_6 < 2; ri_6++) {
                            unsigned int row_6 = first_row_1 + (unsigned int)(ri_6 * 8);
                            unsigned int offset_3 = row_6 * 32 + (pair_1 / 4 ^ row_6 & 7) * 4 + pair_1 % 4;
                            float2 _f2_8 = make_float2(0.0f, 0.0f);
                            float2 x1v_1 = _f2_8;
                            #pragma unroll
                            for (int route_5 = 0; route_5 < 4; route_5++) {
                                float coeff_1 = post_coeff_1[ri_6 * 4 + route_5];
                                float2 _f2_9 = make_float2(xv_1[ri_6 * 2], xv_1[ri_6 * 2 + 1]);
                                float2 _f2_10 = make_float2(coeff_1, coeff_1);
                                float2 _mul_f32x2_1;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_9), "l"(*(const unsigned long long*)&_f2_10));
                                float2 value_1 = _mul_f32x2_1;
                                #pragma unroll
                                for (int input_route_1 = 0; input_route_1 < 4; input_route_1++) {
                                    float cc_1 = comb_coeff_1[ri_6 * 16 + input_route_1 * 4 + route_5];
                                    float2 _f2_11 = make_float2(rv_1[ri_6 * 8 + input_route_1 * 2], rv_1[ri_6 * 8 + input_route_1 * 2 + 1]);
                                    float2 _f2_12 = make_float2(cc_1, cc_1);
                                    value_1 = fma_f32x2_rn_ftz(_f2_11, _f2_12, value_1);
                                }
                                __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(value_1.x, value_1.y));
                                unsigned int word_3 = __as_u32(_bf16x2_2);
                                residual_pairs[io_stage_1 * 8192 + (unsigned int)(route_5 * 2048) + offset_3] = word_3;
                                float lo_1 = __uint_as_float(word_3 << 16);
                                float hi_1 = __uint_as_float(word_3 & 4294901760);
                                tmem_values_1[route_5 * 4 + ri_6 * 2] = lo_1;
                                tmem_values_1[route_5 * 4 + ri_6 * 2 + 1] = hi_1;
                                float2 _f2_13 = make_float2(lo_1, hi_1);
                                float2 rounded_pair_1 = _f2_13;
                                float2 _f2_14 = make_float2(hc_sq_1[ri_6 * 2], hc_sq_1[ri_6 * 2 + 1]);
                                float2 squares_1 = fma_f32x2_rn_ftz(rounded_pair_1, rounded_pair_1, _f2_14);
                                hc_sq_1[ri_6 * 2] = squares_1.x;
                                hc_sq_1[ri_6 * 2 + 1] = squares_1.y;
                                float pc_1 = pre_coeff_1[ri_6 * 4 + route_5];
                                float2 _f2_15 = make_float2(pc_1, pc_1);
                                x1v_1 = fma_f32x2_rn_ftz(rounded_pair_1, _f2_15, x1v_1);
                            }
                            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(x1v_1.x, x1v_1.y));
                            unsigned int x1_word_1 = __as_u32(_bf16x2_3);
                            x_pairs[io_stage_1 * 2048 + offset_3] = x1_word_1;
                            float _bf16x2_fma_f32_1[2] = {x1_sq_1[ri_6 * 2], x1_sq_1[ri_6 * 2 + 1]};
                            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(_bf16x2_fma_f32_1[0]) : "h"(static_cast<uint16_t>((x1_word_1))), "h"(static_cast<uint16_t>((x1_word_1))));
                            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(_bf16x2_fma_f32_1[1]) : "h"(static_cast<uint16_t>((x1_word_1) >> 16)), "h"(static_cast<uint16_t>((x1_word_1) >> 16)));
                            x1_sq_1[ri_6 * 2] = _bf16x2_fma_f32_1[0];
                            x1_sq_1[ri_6 * 2 + 1] = _bf16x2_fma_f32_1[1];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"((unsigned int)tmem_a_tmem + ast_1 * 32), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&tmem_values_1[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(full_a_addr + (ast_1) * 8);
                        a_seq_1 += 2;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(store_ready_addr + (io_stage_1) * 8);
                    }
                    io_seq_2 += 1;
                }
                unsigned int phase_2 = (task_4 - bid) / 152 % 2;
                mbarrier_wait(empty_stats_addr, phase_2 ^ 1);
                #pragma unroll
                for (int ri_7 = 0; ri_7 < 2; ri_7++) {
                    unsigned int row_7 = first_row_1 + (unsigned int)(ri_7 * 8);
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, hc_sq_1[ri_7 * 2] + hc_sq_1[ri_7 * 2 + 1], 2);
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, hc_sq_1[ri_7 * 2] + hc_sq_1[ri_7 * 2 + 1] + _shfl_xor_4, 1);
                    if (lane_in_warp % 4 == 0) {
                        hc_sums[64 + row_7] = hc_sq_1[ri_7 * 2] + hc_sq_1[ri_7 * 2 + 1] + _shfl_xor_4 + _shfl_xor_5;
                    }
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, x1_sq_1[ri_7 * 2] + x1_sq_1[ri_7 * 2 + 1], 2);
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, x1_sq_1[ri_7 * 2] + x1_sq_1[ri_7 * 2 + 1] + _shfl_xor_6, 1);
                    if (lane_in_warp % 4 == 0) {
                        x1_sums[64 + row_7] = x1_sq_1[ri_7 * 2] + x1_sq_1[ri_7 * 2 + 1] + _shfl_xor_6 + _shfl_xor_7;
                    }
                }
                __syncwarp();
                mbarrier_arrive(full_stats_addr);
                __syncwarp();
            }
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        }
    }
    // ---- Role: workspace ----
    if (warp >= 12 && warp <= 15) {
        { // workspace_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
            unsigned int seq_1 = 0;
            unsigned int first_row_2 = warp % 4 * 16 + lane_in_warp / 4;
            #pragma unroll 1
            for (int task_5 = bid; task_5 < (num_tokens + 63) / 64 * 16; task_5 += 152) {
                unsigned int mb_3 = task_5 / 16;
                if (warp % 4 == 0) {
                    unsigned long long base_1 = (launch_epoch[0] + 1) * 128;
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_3) * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(base_1)) >= static_cast<unsigned long long>(16));
                    }
                }
                unsigned int stage_3 = seq_1 % 2;
                mbarrier_wait(full_accum_addr + (stage_3) * 8, seq_1 / 2 % 2);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                    : "r"((unsigned int)tmem_accum + stage_3 * 24));
                float _tmem_load_1[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                    : "r"((unsigned int)tmem_accum + (stage_3 * 24 + 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int group = 0; group < 3; group++) {
                    float values[2];
                    #pragma unroll
                    for (int ri_8 = 0; ri_8 < 2; ri_8++) {
                        unsigned int row_8 = first_row_2 + (unsigned int)(ri_8 * 8);
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 2; j_2++) {
                            if (group < 2) {
                                values[j_2] = ((mb_3 * 64 + row_8 < num_tokens) ? _tmem_load_0[group * 4 + ri_8 * 2 + j_2] : 0.0f);
                            } else {
                                values[j_2] = ((mb_3 * 64 + row_8 < num_tokens) ? _tmem_load_1[ri_8 * 2 + j_2] : 0.0f);
                            }
                        }
                        {
                            float2 _v2 = make_float2(values[0 + 0], values[0 + 1]);
                            *reinterpret_cast<float2*>(scratch + ((unsigned long long)task_5 * 1536 + (unsigned long long)(row_8 * 24) + (unsigned long long)(group * 8) + (unsigned long long)(lane_in_warp % 4 * 2)) + 0) = _v2;
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (elect_sync()) {
                    mbarrier_arrive(empty_accum_addr + (stage_3) * 8);
                }
                mbarrier_wait(full_stats_addr, seq_1 % 2);
                if (lane_in_warp < 16) {
                    unsigned int row_9 = warp % 4 * 16 + lane_in_warp;
                    scratch[(unsigned long long)((num_tokens + 63) / 64 * 16) * 1536 + ((0) ? (unsigned long long)((num_tokens + 63) / 64 * 16) * 64 : (unsigned long long)0) + (unsigned long long)task_5 * 64 + (unsigned long long)row_9] = hc_sums[row_9] + hc_sums[64 + row_9];
                }
                __syncwarp();
                mbarrier_arrive(empty_stats_addr);
                if (elect_sync()) {
                    mbarrier_arrive(mix_arrival_addr);
                }
                if (warp % 4 == 0) {
                    mbarrier_wait(mix_arrival_addr, seq_1 % 2);
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                        asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + mb_3) * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
                        #elif defined(__CUDA_ARCH__)
                        #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
                        #endif
                    }
                }
                seq_1 += 1;
            }
        }
    }
    // ---- Role: mix ----
    if (warp >= 16 && warp <= 19) {
        { // mix_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
            #pragma unroll 1
            for (int token = (unsigned int)(bid * 4) + warp % 4; token < num_tokens; token += 608) {
                unsigned long long target = (launch_epoch[0] + 1) * 128 + 16;
                {
                unsigned long long _acquire_observed;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + ((16384 + token / 64) * 16))) : "memory");
                } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target)) >= static_cast<unsigned long long>(1));
                }
                unsigned int row_10 = token % 64;
                unsigned int first_task = token / 64 * 16;
                float norm_sum = 0.0f;
                #pragma unroll 1
                for (int split = lane_in_warp; split < 16; split += 32) {
                    norm_sum += scratch[(unsigned long long)((num_tokens + 63) / 64 * 16) * 1536 + ((0) ? (unsigned long long)((num_tokens + 63) / 64 * 16) * 64 : (unsigned long long)0) + (unsigned long long)first_task * 64 + (unsigned long long)(split * 64) + (unsigned long long)row_10];
                }
                float _warp_reduce_0 = norm_sum;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                norm_sum = _warp_reduce_0;
                unsigned int safe_col = ((lane_in_warp < 24) ? (unsigned int)lane_in_warp : (unsigned int)23);
                float hc_output = 0.0f;
                #pragma unroll
                for (int split_1 = 0; split_1 < 16; split_1++) {
                    hc_output += scratch[((unsigned long long)first_task + (unsigned long long)split_1) * 1536 + (unsigned long long)(row_10 * 24) + (unsigned long long)safe_col];
                }
                float _rsqrt_0 = rsqrtf(norm_sum * 4.8828125e-05f + hc_norm_eps);
                float hc_scale = _rsqrt_0;
                float mix_value = 0.0f;
                if (lane_in_warp < 24) {
                    unsigned int scale_idx = ((lane_in_warp < 4) ? (unsigned int)0 : ((lane_in_warp < 8) ? (unsigned int)1 : (unsigned int)2));
                    float affine = hc_output * hc_scale * mix_scales[scale_idx] + mix_bases[lane_in_warp];
                    if (lane_in_warp < 8) {
                        float _exp_0 = expf(-affine);
                        float _rcp_0 = approx_rcp(1.0f + _exp_0);
                        float sigmoid = _rcp_0;
                        mix_value = ((lane_in_warp < 4) ? sigmoid + hc_pre_eps : sigmoid * hc_post_scale);
                    } else {
                        mix_value = affine;
                    }
                }
                __syncwarp();
                float value_2 = ((lane_in_warp >= 8 && lane_in_warp < 24) ? mix_value : 0.0f);
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, value_2, 2);
                float _max_0 = max_noftz(value_2, _shfl_xor_8);
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, _max_0, 1);
                float _max_1 = max_noftz(_max_0, _shfl_xor_9);
                float _exp_1 = expf(value_2 - _max_1);
                value_2 = _exp_1;
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, value_2, 2);
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, value_2 + _shfl_xor_10, 1);
                float _rcp_1 = approx_rcp(value_2 + _shfl_xor_10 + _shfl_xor_11);
                value_2 = value_2 * _rcp_1 + sinkhorn_eps;
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, value_2, 4);
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, value_2 + _shfl_xor_12, 24);
                float _rcp_2 = approx_rcp(value_2 + _shfl_xor_12 + _shfl_xor_13 + sinkhorn_eps);
                value_2 *= _rcp_2;
                #pragma unroll 1
                for (int iteration = 1; iteration < num_sinkhorn_iters; iteration++) {
                    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, value_2, 2);
                    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, value_2 + _shfl_xor_14, 1);
                    float _rcp_3 = approx_rcp(value_2 + _shfl_xor_14 + _shfl_xor_15 + sinkhorn_eps);
                    value_2 *= _rcp_3;
                    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, value_2, 4);
                    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, value_2 + _shfl_xor_16, 24);
                    float _rcp_4 = approx_rcp(value_2 + _shfl_xor_16 + _shfl_xor_17 + sinkhorn_eps);
                    value_2 *= _rcp_4;
                }
                if (lane_in_warp < 24) {
                    if (lane_in_warp < 4) {
                        new_prev_mix[(unsigned long long)token * 4 + (unsigned long long)lane_in_warp] = mix_value;
                    } else if (lane_in_warp < 8) {
                        new_post_mix[(unsigned long long)token * 4 + (unsigned long long)lane_in_warp - 4] = mix_value;
                    } else {
                        new_comb_res_mix[(unsigned long long)token * 16 + (unsigned long long)lane_in_warp - 8] = value_2;
                    }
                }
                __syncwarp();
            }
        }
    }
    // ---- Role: norm ----
    if (warp >= 20 && warp <= 23) {
        // idle — no tasks assigned
    }

    // Kernel teardown ops
    unsigned int shifted_wg = warp / 4;
    if (shifted_wg == 1 || shifted_wg == 2 || shifted_wg == 5) {
        unsigned int pairs = (num_tokens + 1) / 2;
        unsigned int partitions = 2;
        unsigned int task_6 = bid * 4;
        if (task_6 < pairs * partitions) {
            asm volatile("setmaxnreg.inc.sync.aligned.u32 96;");
            while (1) {
                unsigned int ticket = 0;
                bool _elect_sync_0 = elect_sync();
                if (_elect_sync_0) {
                    unsigned int _atomic_old_1 = atomicAdd(queue, 1);
                    ticket = _atomic_old_1;
                }
                unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, ticket, 0);
                ticket = _shfl_0;
                task_6 = (unsigned int)(bid * 4) + ticket % 4 + ticket / 4 * 152 * 4;
                if (task_6 < pairs * partitions) {
                    unsigned int part = task_6 % partitions;
                    unsigned int begin = part * 20 / partitions;
                    unsigned int end = (part + 1) * 20 / partitions;
                    unsigned int token_1 = task_6 / partitions * 2;
                    unsigned long long target_1 = (launch_epoch[0] + 1) * 128 + 16;
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(split_barriers + (token_1 / 64 * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(target_1)) >= static_cast<unsigned long long>(1));
                    }
                    unsigned int current_pack[12];
                    unsigned int next_pack[12];
                    unsigned int first_hidden_idx = begin * 256 + lane_in_warp * 8;
                    unsigned long long off = (unsigned long long)token_1 * 5120 + (unsigned long long)first_hidden_idx;
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(y_bf16 + off + 0);
                        uint4* _vdst_0 = reinterpret_cast<uint4*>(&current_pack[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(y_bf16 + (off + (unsigned long long)(((token_1 + 1 < num_tokens) ? 5120 : 0))) + 0);
                        uint4* _vdst_1 = reinterpret_cast<uint4*>(&current_pack[4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rmsnorm_weight + first_hidden_idx + 0);
                        uint4* _vdst_2 = reinterpret_cast<uint4*>(&current_pack[8]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vdst_2[_blk] = _vptr_2[_blk];
                        }
                    }
                    if (end > begin + 1) {
                        unsigned long long off_0 = (unsigned long long)token_1 * 5120 + (unsigned long long)(first_hidden_idx + 256);
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(y_bf16 + off_0 + 0);
                            uint4* _vdst_3 = reinterpret_cast<uint4*>(&next_pack[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_3[_blk].x), "=r"(_vdst_3[_blk].y), "=r"(_vdst_3[_blk].z), "=r"(_vdst_3[_blk].w) : "l"((const void*)(_vptr_3 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        {
                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(y_bf16 + (off_0 + (unsigned long long)(((token_1 + 1 < num_tokens) ? 5120 : 0))) + 0);
                            uint4* _vdst_4 = reinterpret_cast<uint4*>(&next_pack[4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                    : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        {
                            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(rmsnorm_weight + (first_hidden_idx + 256) + 0);
                            uint4* _vdst_5 = reinterpret_cast<uint4*>(&next_pack[8]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_5[_blk] = _vptr_5[_blk];
                            }
                        }
                    }
                    float row_sum = 0.0f;
                    if (lane_in_warp < 2 && token_1 + lane_in_warp < num_tokens) {
                        #pragma unroll
                        for (int split_2 = 0; split_2 < 16; split_2++) {
                            row_sum += scratch[(unsigned long long)((num_tokens + 63) / 64 * 16) * 1536 + ((1) ? (unsigned long long)((num_tokens + 63) / 64 * 16) * 64 : (unsigned long long)0) + (unsigned long long)(token_1 / 64 * 16) * 64 + (unsigned long long)(split_2 * 64) + (unsigned long long)(token_1 % 64) + (unsigned long long)lane_in_warp];
                        }
                    }
                    float _rsqrt_1 = rsqrtf(row_sum * 0.0001953125f + rmsnorm_eps);
                    float scale = _rsqrt_1 * rmsnorm_scale;
                    float _shfl_1;
                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(scale), "r"(0));
                    float _shfl_2;
                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(scale), "r"(1));
                    unsigned int first_sf_idx = token_1 % 224;
                    unsigned int shared_row0 = token_1 / 224 * 256 + (first_sf_idx & 4294967168) + (first_sf_idx & 31) * 4 + (first_sf_idx >> 5 & 3);
                    unsigned int second_sf_idx = (token_1 + 1) % 224;
                    unsigned int shared_row1 = (token_1 + 1) / 224 * 256 + (second_sf_idx & 4294967168) + (second_sf_idx & 31) * 4 + (second_sf_idx >> 5 & 3);
                    #pragma unroll 1
                    for (int pack = begin; pack < end; pack++) {
                        unsigned int future_pack[12];
                        if (end > (unsigned int)(pack + 2)) {
                            unsigned int hidden_idx = (unsigned int)((pack + 2) * 256) + lane_in_warp * 8;
                            unsigned long long off_0_1 = (unsigned long long)token_1 * 5120 + (unsigned long long)hidden_idx;
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(y_bf16 + off_0_1 + 0);
                                uint4* _vdst_6 = reinterpret_cast<uint4*>(&future_pack[0]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(y_bf16 + (off_0_1 + (unsigned long long)(((token_1 + 1 < num_tokens) ? 5120 : 0))) + 0);
                                uint4* _vdst_7 = reinterpret_cast<uint4*>(&future_pack[4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                                        : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(rmsnorm_weight + hidden_idx + 0);
                                uint4* _vdst_8 = reinterpret_cast<uint4*>(&future_pack[8]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_8[_blk] = _vptr_8[_blk];
                                }
                            }
                        }
                        float values_1[16];
                        unsigned int normalized_words[8];
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 4; j_3++) {
                            #pragma unroll
                            for (int token_offset = 0; token_offset < 2; token_offset++) {
                                unsigned int word_4 = current_pack[token_offset * 4 + j_3];
                                unsigned int weight_word = current_pack[8 + j_3];
                                float lo_2 = __uint_as_float(word_4 << 16);
                                float hi_2 = __uint_as_float(word_4 & 4294901760);
                                float weight_lo = __uint_as_float(weight_word << 16);
                                float weight_hi = __uint_as_float(weight_word & 4294901760);
                                float rs = ((token_offset == 0) ? _shfl_1 : _shfl_2);
                                float2 _f2_16 = make_float2(lo_2, hi_2);
                                float2 _f2_17 = make_float2(rs, rs);
                                float2 _mul_f32x2_2;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_17));
                                float2 scaled = _mul_f32x2_2;
                                float2 _f2_18 = make_float2(weight_lo, weight_hi);
                                float2 _mul_f32x2_3;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f2_18));
                                float2 weighted = _mul_f32x2_3;
                                __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(weighted.x, weighted.y));
                                unsigned int rounded_bits = __as_u32(_bf16x2_4);
                                normalized_words[token_offset * 4 + j_3] = rounded_bits;
                                values_1[token_offset * 8 + 2 * j_3] = __uint_as_float(rounded_bits << 16);
                                values_1[token_offset * 8 + 2 * j_3 + 1] = __uint_as_float(rounded_bits & 4294901760);
                            }
                        }
                        unsigned long long off_0_2 = (unsigned long long)token_1 * 5120 + (unsigned long long)(pack * 256) + (unsigned long long)(lane_in_warp * 8);
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(values_1[0 + 0], values_1[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(values_1[0 + 2], values_1[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(values_1[0 + 4], values_1[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(values_1[0 + 6], values_1[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y_bf16 + off_0_2))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                        if (token_1 + 1 < num_tokens) {
                            unsigned long long off_1 = (unsigned long long)(token_1 + 1) * 5120 + (unsigned long long)(pack * 256) + (unsigned long long)(lane_in_warp * 8);
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(values_1[8 + 0], values_1[8 + 1]);
                                _pk[1] = __floats2bfloat162_rn(values_1[8 + 2], values_1[8 + 3]);
                                _pk[2] = __floats2bfloat162_rn(values_1[8 + 4], values_1[8 + 5]);
                                _pk[3] = __floats2bfloat162_rn(values_1[8 + 6], values_1[8 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y_bf16 + off_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                        uint32_t _bf16x2_abs_0;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(normalized_words[0]));
                        uint32_t _bf16x2_abs_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(normalized_words[2]));
                        uint32_t _bf16x2_max_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_1));
                        uint32_t _bf16x2_abs_2;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(normalized_words[1]));
                        uint32_t _bf16x2_abs_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(normalized_words[3]));
                        uint32_t _bf16x2_max_1;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_abs_2), "r"(_bf16x2_abs_3));
                        uint32_t _bf16x2_max_2;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_max_0), "r"(_bf16x2_max_1));
                        uint16_t _bf16_max_0;
                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(_bf16x2_max_2 & 65535)), "h"((uint16_t)(_bf16x2_max_2 >> 16)));
                        uint32_t _bf16x2_abs_4;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(normalized_words[4]));
                        uint32_t _bf16x2_abs_5;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(normalized_words[6]));
                        uint32_t _bf16x2_max_3;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(_bf16x2_abs_4), "r"(_bf16x2_abs_5));
                        uint32_t _bf16x2_abs_6;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(normalized_words[5]));
                        uint32_t _bf16x2_abs_7;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(normalized_words[7]));
                        uint32_t _bf16x2_max_4;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(_bf16x2_abs_6), "r"(_bf16x2_abs_7));
                        uint32_t _bf16x2_max_5;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(_bf16x2_max_3), "r"(_bf16x2_max_4));
                        uint16_t _bf16_max_1;
                        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_1) : "h"((uint16_t)(_bf16x2_max_5 & 65535)), "h"((uint16_t)(_bf16x2_max_5 >> 16)));
                        unsigned int second_amax = ((token_1 + 1 < num_tokens) ? (unsigned int)_bf16_max_1 : (unsigned int)0);
                        unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, (unsigned int)_bf16_max_0 | second_amax << 16, 2);
                        uint32_t _bf16x2_max_6;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"((unsigned int)_bf16_max_0 | second_amax << 16), "r"(_shfl_xor_18));
                        unsigned int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, _bf16x2_max_6, 1);
                        uint32_t _bf16x2_max_7;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_7) : "r"(_bf16x2_max_6), "r"(_shfl_xor_19));
                        unsigned int rounded = (_bf16x2_max_7 & 65535) + 31 >> 7;
                        unsigned int rounded_1 = (_bf16x2_max_7 >> 16) + 31 >> 7;
                        unsigned long long off_2 = (unsigned long long)token_1 * 5120 + (unsigned long long)(pack * 256) + (unsigned long long)(lane_in_warp * 8);
                        unsigned int word_5 = 0;
                        unsigned int inv_bits = 254 - (((rounded > 113) ? rounded : (unsigned int)113) - 8) << 23;
                        float inv = __uint_as_float(inv_bits);
                        float quantized[8];
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            quantized[j_4] = (float)(__nv_bfloat16)(values_1[j_4] * inv);
                        }
                        {
                            unsigned int _fp8_pk[2];
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[0]) : "f"(quantized[0 + 0]), "f"(quantized[0 + 1]), "f"(quantized[0 + 2]), "f"(quantized[0 + 3]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[1]) : "f"(quantized[0 + 4]), "f"(quantized[0 + 5]), "f"(quantized[0 + 6]), "f"(quantized[0 + 7]));
                            *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(y_fp8 + off_2) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                        }
                        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, ((rounded > 113) ? rounded : (unsigned int)113) - 8, lane_in_warp & 16);
                        unsigned int sf0 = _shfl_3;
                        unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, ((rounded > 113) ? rounded : (unsigned int)113) - 8, (lane_in_warp & 16) + 4);
                        unsigned int sf1 = _shfl_4;
                        unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, ((rounded > 113) ? rounded : (unsigned int)113) - 8, (lane_in_warp & 16) + 8);
                        unsigned int sf2 = _shfl_5;
                        unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, ((rounded > 113) ? rounded : (unsigned int)113) - 8, (lane_in_warp & 16) + 12);
                        unsigned int sf3 = _shfl_6;
                        word_5 = sf0 | sf1 << 8 | sf2 << 16 | sf3 << 24;
                        {
                            if (lane_in_warp % 16 == 0) {
                                unsigned int word_idx = (unsigned int)(pack * 2) + lane_in_warp / 16;
                                y_primary_sf[(unsigned long long)token_1 * primary_sf_stride_token + (unsigned long long)word_idx * primary_sf_stride_word] = word_5;
                                unsigned int shared_row = shared_row0;
                                y_shared_sf[(unsigned long long)word_idx * shared_sf_stride_word + (unsigned long long)shared_row] = word_5;
                            }
                        }
                        if (token_1 + 1 < num_tokens) {
                            unsigned long long off_1_1 = (unsigned long long)(token_1 + 1) * 5120 + (unsigned long long)(pack * 256) + (unsigned long long)(lane_in_warp * 8);
                            unsigned int word_2_1 = 0;
                            unsigned int inv_bits_3 = 254 - (((rounded_1 > 113) ? rounded_1 : (unsigned int)113) - 8) << 23;
                            float inv_4 = __uint_as_float(inv_bits_3);
                            float quantized_5[8];
                            #pragma unroll
                            for (int j_5 = 0; j_5 < 8; j_5++) {
                                quantized_5[j_5] = (float)(__nv_bfloat16)(values_1[8 + j_5] * inv_4);
                            }
                            {
                                unsigned int _fp8_pk[2];
                                asm("{\n\t"
                                    ".reg .b16 _lo, _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}\n"
                                    : "=r"(_fp8_pk[0]) : "f"(quantized_5[0 + 0]), "f"(quantized_5[0 + 1]), "f"(quantized_5[0 + 2]), "f"(quantized_5[0 + 3]));
                                asm("{\n\t"
                                    ".reg .b16 _lo, _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}\n"
                                    : "=r"(_fp8_pk[1]) : "f"(quantized_5[0 + 4]), "f"(quantized_5[0 + 5]), "f"(quantized_5[0 + 6]), "f"(quantized_5[0 + 7]));
                                *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(y_fp8 + off_1_1) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                            }
                            unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, ((rounded_1 > 113) ? rounded_1 : (unsigned int)113) - 8, lane_in_warp & 16);
                            unsigned int sf0_6 = _shfl_7;
                            unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, ((rounded_1 > 113) ? rounded_1 : (unsigned int)113) - 8, (lane_in_warp & 16) + 4);
                            unsigned int sf1_7 = _shfl_8;
                            unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, ((rounded_1 > 113) ? rounded_1 : (unsigned int)113) - 8, (lane_in_warp & 16) + 8);
                            unsigned int sf2_8 = _shfl_9;
                            unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, ((rounded_1 > 113) ? rounded_1 : (unsigned int)113) - 8, (lane_in_warp & 16) + 12);
                            unsigned int sf3_9 = _shfl_10;
                            word_2_1 = sf0_6 | sf1_7 << 8 | sf2_8 << 16 | sf3_9 << 24;
                            {
                                if (lane_in_warp % 16 == 0) {
                                    unsigned int word_idx_1 = (unsigned int)(pack * 2) + lane_in_warp / 16;
                                    y_primary_sf[(unsigned long long)(token_1 + 1) * primary_sf_stride_token + (unsigned long long)word_idx_1 * primary_sf_stride_word] = word_2_1;
                                    unsigned int shared_row_1 = shared_row1;
                                    y_shared_sf[(unsigned long long)word_idx_1 * shared_sf_stride_word + (unsigned long long)shared_row_1] = word_2_1;
                                }
                            }
                        }
                        if (end > (unsigned int)(pack + 1)) {
                            #pragma unroll
                            for (int j_6 = 0; j_6 < 12; j_6++) {
                                current_pack[j_6] = next_pack[j_6];
                            }
                        }
                        if (end > (unsigned int)(pack + 2)) {
                            #pragma unroll
                            for (int j_7 = 0; j_7 < 12; j_7++) {
                                next_pack[j_7] = future_pack[j_7];
                            }
                        }
                    }
                }
                if (task_6 >= pairs * partitions) {
                    break;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 3) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
