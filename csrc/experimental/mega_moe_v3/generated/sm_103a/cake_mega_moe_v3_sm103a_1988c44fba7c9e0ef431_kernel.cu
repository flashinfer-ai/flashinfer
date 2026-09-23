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
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 272
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAIN_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 51200
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 51200
#define SMEM_SMEM_SFA_OFF 50176
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 51200
#define SMEM_SMEM_SFB_OFF 51200
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 51200
#define SMEM_SMEM_HIST_OFF 205824
#define SMEM_SMEM_HIST_STAGE_BYTES 1536
#define SMEM_SMEM_HIST_STRIDE 1536
#define SMEM_SMEM_DST_ROW_OFF 207360
#define SMEM_SMEM_DST_ROW_STAGE_BYTES 8
#define SMEM_SMEM_DST_ROW_STRIDE 8
#define SMEM_TOTAL 207488
#define NUM_CTAS 64

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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo256(int lo) {
    const int SBO = 256;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
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

__global__ __launch_bounds__(256) __cluster_dims__(2,1,1) void
kernel_cake_mega_moe_v3_sm103a_1988c44fba7c9e0ef431(int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8, unsigned int* __restrict__ x_sf, CakeTensorMap const* A, int* __restrict__ pool_fp8, unsigned int* __restrict__ pool_sf, float* __restrict__ routing_weight_pool, int* __restrict__ token_to_permuted, int* __restrict__ meta_token, int* __restrict__ meta_slot, int* __restrict__ expert_counts, int* __restrict__ expert_row_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ tile_expert, int* __restrict__ tile_m_local, int* __restrict__ total_m_tiles_out, CakeTensorMap const* B1, unsigned int* __restrict__ SFB1, uint8_t* __restrict__ I_fp8_w, uint8_t* __restrict__ SF_I_w, __nv_bfloat16* __restrict__ l1_bf16_capture, CakeTensorMap const* A2, unsigned int* __restrict__ SFA2, CakeTensorMap const* B2, unsigned int* __restrict__ SFB2, __nv_bfloat16* __restrict__ expert_output, __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ histogram_done, unsigned int* __restrict__ prefix_done, unsigned int* __restrict__ dispatch_done, unsigned int* __restrict__ l1_arrival, unsigned int* __restrict__ l2_done, int num_tokens, int top_k, int num_experts, int N1, int K1, int grid_n1, int K1_tiles, int N2, int K2, int grid_n2, int K2_tiles, int total_m_tiles, int M_total, float activation_clamp)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 32)
    #define main_done_addr (mbar_base + 64)
    #define epi_done_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B2)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_sfa_addr = smem + 50176;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfb_addr = smem + 51200;
    int* smem_hist = reinterpret_cast<int*>(smem_raw + 205824);
    const int smem_hist_addr = smem + 205824;
    int* smem_dst_row = reinterpret_cast<int*>(smem_raw + 207360);
    const int smem_dst_row_addr = smem + 207360;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'main_pipe' ---
            // main_done: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // epi_done: 2 barriers, init_count=8
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 272 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 264;

    // ---- Role: dispatch ----
    if (warp >= 6 && warp <= 7) {
        { // dispatch_main
            int disp_warp = warp - 6;
            int disp_tid = (unsigned int)(disp_warp * 32) + lane;
            int total_pairs = num_tokens * top_k;
            int hidden_u32 = K1 / 4;
            int sf_words_x = K1 / 128;
            #pragma unroll 1
            for (int i = disp_tid; i < num_experts; i += 64) {
                smem_hist[i] = 0;
            }
            asm volatile("barrier.sync 10, 64;" ::: "memory");
            int grid_stride = num_bids * 64;
            int cta_base = bid * 64;
            #pragma unroll 1
            for (int idx = cta_base + disp_tid; idx < total_pairs; idx += grid_stride) {
                int e_h = topk_idx_i32[idx * 2];
                atomicAdd(&smem_hist[e_h], 1);
            }
            asm volatile("barrier.sync 10, 64;" ::: "memory");
            #pragma unroll 1
            for (int i_1 = disp_tid; i_1 < num_experts; i_1 += 64) {
                int c_h = smem_hist[i_1];
                if (c_h > 0) {
                    atomicAdd(&expert_counts[i_1], c_h);
                }
            }
            asm volatile("barrier.sync 10, 64;" ::: "memory");
            __threadfence_system();
            if (warp == 6) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(histogram_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.sys.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(histogram_done) + (0);
                unsigned int _gc_v;
                do {
                    _gc_v = atomicAdd_system(_gc_p, 0u);
                } while (_gc_v < (unsigned int)(NUM_CTAS));
                __threadfence_system();
            }
            if (bid == 0) {
                if (warp == 6) {
                    if (elect_sync()) {
                        int running = 0;
                        int tile_idx = 0;
                        #pragma unroll 1
                        for (int e = 0; e < num_experts; e++) {
                            int cnt_e = expert_counts[e];
                            int pad_e = (cnt_e + 256 - 1) / 256 * 256;
                            expert_row_offsets[e] = running;
                            int ntile_e = pad_e / 128;
                            #pragma unroll 1
                            for (int tt = 0; tt < ntile_e; tt++) {
                                tile_expert[tile_idx] = e;
                                tile_m_local[tile_idx] = tt * 128;
                                tile_idx += 1;
                            }
                            running += pad_e;
                        }
                        total_m_tiles_out[0] = tile_idx;
                    }
                }
                __threadfence_system();
                if (warp == 6) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gc_p = reinterpret_cast<unsigned int*>(prefix_done) + (0);
                            unsigned int _gc_old;
                            asm volatile("atom.release.sys.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                        }
                    }
                }
            }
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(prefix_done) + (0);
                unsigned int _gc_v;
                do {
                    _gc_v = atomicAdd_system(_gc_p, 0u);
                } while (_gc_v < (unsigned int)(1));
                __threadfence_system();
            }
            int global_warp_idx = bid * 2 + disp_warp;
            int warps_per_grid = NUM_CTAS * 2;
            #pragma unroll 1
            for (int pair_idx = global_warp_idx; pair_idx < total_pairs; pair_idx += warps_per_grid) {
                int e_s = topk_idx_i32[pair_idx * 2];
                int t_s = pair_idx / top_k;
                int k_s = pair_idx - t_s * top_k;
                if (lane == 0) {
                    float cached_rw = topk_weights[pair_idx];
                    int _atomic_old_0 = atomicAdd(&expert_scatter_offsets[e_s], 1);
                    int claim = _atomic_old_0;
                    int dst_row_e = expert_row_offsets[e_s] + claim;
                    smem_dst_row[disp_warp] = dst_row_e;
                    token_to_permuted[pair_idx] = dst_row_e;
                    meta_token[dst_row_e] = t_s;
                    meta_slot[dst_row_e] = k_s;
                    routing_weight_pool[dst_row_e] = cached_rw;
                }
                __syncwarp();
                int dst_row = smem_dst_row[disp_warp];
                unsigned long long src_u32_off = (unsigned long long)t_s * (unsigned long long)hidden_u32;
                unsigned long long dst_u32_off = (unsigned long long)dst_row * (unsigned long long)hidden_u32;
                #pragma unroll 1
                for (int base = lane * 4; base < hidden_u32; base += 128) {
                    int _vec_load_0[4];
                    {
                        const int4* _ivptr_0 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)base) + 0);
                        int4 _ivld_0;
                        _ivld_0 = *_ivptr_0;
                        _vec_load_0[0 + 0] = _ivld_0.x;
                        _vec_load_0[0 + 1] = _ivld_0.y;
                        _vec_load_0[0 + 2] = _ivld_0.z;
                        _vec_load_0[0 + 3] = _ivld_0.w;
                    }
                    reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)base))[0] = reinterpret_cast<int4*>(_vec_load_0)[0];
                }
                unsigned long long src_sf_off = (unsigned long long)t_s * (unsigned long long)sf_words_x;
                unsigned long long dst_sf_off = (unsigned long long)dst_row * (unsigned long long)sf_words_x;
                #pragma unroll 1
                for (int w = lane; w < sf_words_x; w += 32) {
                    pool_sf[dst_sf_off + (unsigned long long)w] = x_sf[src_sf_off + (unsigned long long)w];
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 10, 64;" ::: "memory");
            __threadfence_system();
            if (warp == 6) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.sys.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_tid = (unsigned int)(epi_warp * 32) + lane;
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                unsigned int _gc_v;
                do {
                    _gc_v = atomicAdd_system(_gc_p, 0u);
                } while (_gc_v < (unsigned int)(NUM_CTAS));
                __threadfence_system();
            }
            unsigned int e_stage = 0;
            int num_tiles1e = total_m_tiles * grid_n1;
            int n_out1 = N1 / 2;
            int sf_cols_out1 = n_out1 / 32;
            unsigned int _phase_main_done = 0;
            #pragma unroll 1
            for (unsigned int this_bid = bid; this_bid < num_tiles1e; this_bid += num_bids) {
                int m_tile = this_bid / (unsigned int)(grid_n1 * 2) * 2 + this_bid % 2;
                int n_tile = this_bid / 2 % (unsigned int)grid_n1;
                int exp_id = tile_expert[m_tile];
                int local_m = tile_m_local[m_tile];
                int off_m = expert_row_offsets[exp_id] + local_m;
                int out_n_base = n_tile * 64;
                int token = off_m + epi_tid;
                float rw = routing_weight_pool[token];
                mbarrier_wait(main_done_addr + (e_stage) * 8, _phase_main_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int tmem_row = cta_rank * 128 + epi_warp * 32;
                float amax = 0.0001f;
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    int pb = e_stage * 128 + (unsigned int)(c * 16);
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(_tmem_load_0[j * 2]);
                        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_4);
                        float _min_0 = fminf(_cvt_f32_0, activation_clamp);
                        float g0 = _min_0;
                        __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(_tmem_load_0[j * 2 + 1]);
                        float _cvt_f32_1 = __bfloat162float(_cvt_bf16_5);
                        float _min_1 = fminf(_cvt_f32_1, activation_clamp);
                        float g1 = _min_1;
                        __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(_tmem_load_1[j * 2]);
                        float _cvt_f32_2 = __bfloat162float(_cvt_bf16_6);
                        float _max_0 = max_noftz(_cvt_f32_2, -activation_clamp);
                        float _min_2 = fminf(_max_0, activation_clamp);
                        float u0 = _min_2;
                        __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_tmem_load_1[j * 2 + 1]);
                        float _cvt_f32_3 = __bfloat162float(_cvt_bf16_7);
                        float _max_1 = max_noftz(_cvt_f32_3, -activation_clamp);
                        float _min_3 = fminf(_max_1, activation_clamp);
                        float u1 = _min_3;
                        float _exp2_0 = approx_exp2((-g0) * 1.4426950408889634f);
                        float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                        float s0 = _rcp_0;
                        float _exp2_1 = approx_exp2((-g1) * 1.4426950408889634f);
                        float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                        float s1 = _rcp_1;
                        float r0 = g0 * s0 * u0 * rw;
                        float r1 = g1 * s1 * u1 * rw;
                        float _max_2 = max_noftz(r0, -r0);
                        float _max_3 = max_noftz(r1, -r1);
                        float _max_4 = max_noftz(_max_2, _max_3);
                        float _max_5 = max_noftz(amax, _max_4);
                        amax = _max_5;
                    }
                }
                unsigned int exp_e = 0;
                float sf_inv = 0.0f;
                {
                    unsigned int amax_bits = __as_u32(amax);
                    unsigned int _max_6 = ((amax_bits + 2097151 >> 23) > (113) ? (amax_bits + 2097151 >> 23) : (113));
                    exp_e = _max_6 - 8;
                    unsigned int inv_bits = 254 - exp_e << 23;
                    sf_inv = __uint_as_float(inv_bits);
                }
                uint8_t sf_byte = exp_e;
                SF_I_w[token * sf_cols_out1 + n_tile * 2] = sf_byte;
                #pragma unroll
                for (int c_1 = 0; c_1 < 4; c_1++) {
                    int pb2 = e_stage * 128 + (unsigned int)(c_1 * 16);
                    float _tmem_load_2[8];
                    tmem_ld_x8(&_tmem_load_2[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb2);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_3[8];
                    tmem_ld_x8(&_tmem_load_3[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb2 + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float result[8];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 4; j_1++) {
                        __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(_tmem_load_2[j_1 * 2]);
                        float _cvt_f32_4 = __bfloat162float(_cvt_bf16_8);
                        float _min_5 = fminf(_cvt_f32_4, activation_clamp);
                        float gg0 = _min_5;
                        __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(_tmem_load_2[j_1 * 2 + 1]);
                        float _cvt_f32_5 = __bfloat162float(_cvt_bf16_9);
                        float _min_6 = fminf(_cvt_f32_5, activation_clamp);
                        float gg1 = _min_6;
                        __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(_tmem_load_3[j_1 * 2]);
                        float _cvt_f32_6 = __bfloat162float(_cvt_bf16_10);
                        float _max_8 = max_noftz(_cvt_f32_6, -activation_clamp);
                        float _min_7 = fminf(_max_8, activation_clamp);
                        float uu0 = _min_7;
                        __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(_tmem_load_3[j_1 * 2 + 1]);
                        float _cvt_f32_7 = __bfloat162float(_cvt_bf16_11);
                        float _max_9 = max_noftz(_cvt_f32_7, -activation_clamp);
                        float _min_8 = fminf(_max_9, activation_clamp);
                        float uu1 = _min_8;
                        float _exp2_3 = approx_exp2((-gg0) * 1.4426950408889634f);
                        float _rcp_2 = approx_rcp(1.0f + _exp2_3);
                        float ss0 = _rcp_2;
                        float _exp2_4 = approx_exp2((-gg1) * 1.4426950408889634f);
                        float _rcp_3 = approx_rcp(1.0f + _exp2_4);
                        float ss1 = _rcp_3;
                        result[j_1 * 2] = gg0 * ss0 * uu0 * rw;
                        result[j_1 * 2 + 1] = gg1 * ss1 * uu1 * rw;
                    }
                    {
                        const float2 _prescale2_0 = {sf_inv, sf_inv};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&result[0])[_ps], _prescale2_0);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            result[0 + _ps] *= sf_inv;
                        #endif
                        unsigned int _fp8_pk[2];
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[0]) : "f"(result[0 + 0]), "f"(result[0 + 1]), "f"(result[0 + 2]), "f"(result[0 + 3]));
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[1]) : "f"(result[0 + 4]), "f"(result[0 + 5]), "f"(result[0 + 6]), "f"(result[0 + 7]));
                        *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(I_fp8_w + (token * n_out1 + out_n_base + c_1 * 8)) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                    }
                }
                float amax_0 = 0.0001f;
                #pragma unroll
                for (int c_2 = 0; c_2 < 4; c_2++) {
                    int pb_1 = e_stage * 128 + (unsigned int)((4 + c_2) * 16);
                    float _tmem_load_4[8];
                    tmem_ld_x8(&_tmem_load_4[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_5[8];
                    tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb_1 + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 4; j_2++) {
                        __nv_bfloat16 _cvt_bf16_16 = __float2bfloat16(_tmem_load_4[j_2 * 2]);
                        float _cvt_f32_8 = __bfloat162float(_cvt_bf16_16);
                        float _min_9 = fminf(_cvt_f32_8, activation_clamp);
                        float g0_1 = _min_9;
                        __nv_bfloat16 _cvt_bf16_17 = __float2bfloat16(_tmem_load_4[j_2 * 2 + 1]);
                        float _cvt_f32_9 = __bfloat162float(_cvt_bf16_17);
                        float _min_10 = fminf(_cvt_f32_9, activation_clamp);
                        float g1_1 = _min_10;
                        __nv_bfloat16 _cvt_bf16_18 = __float2bfloat16(_tmem_load_5[j_2 * 2]);
                        float _cvt_f32_10 = __bfloat162float(_cvt_bf16_18);
                        float _max_10 = max_noftz(_cvt_f32_10, -activation_clamp);
                        float _min_11 = fminf(_max_10, activation_clamp);
                        float u0_1 = _min_11;
                        __nv_bfloat16 _cvt_bf16_19 = __float2bfloat16(_tmem_load_5[j_2 * 2 + 1]);
                        float _cvt_f32_11 = __bfloat162float(_cvt_bf16_19);
                        float _max_11 = max_noftz(_cvt_f32_11, -activation_clamp);
                        float _min_12 = fminf(_max_11, activation_clamp);
                        float u1_1 = _min_12;
                        float _exp2_5 = approx_exp2((-g0_1) * 1.4426950408889634f);
                        float _rcp_4 = approx_rcp(1.0f + _exp2_5);
                        float s0_1 = _rcp_4;
                        float _exp2_6 = approx_exp2((-g1_1) * 1.4426950408889634f);
                        float _rcp_5 = approx_rcp(1.0f + _exp2_6);
                        float s1_1 = _rcp_5;
                        float r0_1 = g0_1 * s0_1 * u0_1 * rw;
                        float r1_1 = g1_1 * s1_1 * u1_1 * rw;
                        float _max_12 = max_noftz(r0_1, -r0_1);
                        float _max_13 = max_noftz(r1_1, -r1_1);
                        float _max_14 = max_noftz(_max_12, _max_13);
                        float _max_15 = max_noftz(amax_0, _max_14);
                        amax_0 = _max_15;
                    }
                }
                unsigned int exp_e_1 = 0;
                float sf_inv_2 = 0.0f;
                {
                    unsigned int amax_bits_1 = __as_u32(amax_0);
                    unsigned int _max_16 = ((amax_bits_1 + 2097151 >> 23) > (113) ? (amax_bits_1 + 2097151 >> 23) : (113));
                    exp_e_1 = _max_16 - 8;
                    unsigned int inv_bits_1 = 254 - exp_e_1 << 23;
                    sf_inv_2 = __uint_as_float(inv_bits_1);
                }
                uint8_t sf_byte_3 = exp_e_1;
                SF_I_w[token * sf_cols_out1 + n_tile * 2 + 1] = sf_byte_3;
                #pragma unroll
                for (int c_3 = 0; c_3 < 4; c_3++) {
                    int pb2_1 = e_stage * 128 + (unsigned int)((4 + c_3) * 16);
                    float _tmem_load_6[8];
                    tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb2_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_7[8];
                    tmem_ld_x8(&_tmem_load_7[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pb2_1 + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float result_1[8];
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 4; j_3++) {
                        __nv_bfloat16 _cvt_bf16_20 = __float2bfloat16(_tmem_load_6[j_3 * 2]);
                        float _cvt_f32_12 = __bfloat162float(_cvt_bf16_20);
                        float _min_14 = fminf(_cvt_f32_12, activation_clamp);
                        float gg0_1 = _min_14;
                        __nv_bfloat16 _cvt_bf16_21 = __float2bfloat16(_tmem_load_6[j_3 * 2 + 1]);
                        float _cvt_f32_13 = __bfloat162float(_cvt_bf16_21);
                        float _min_15 = fminf(_cvt_f32_13, activation_clamp);
                        float gg1_1 = _min_15;
                        __nv_bfloat16 _cvt_bf16_22 = __float2bfloat16(_tmem_load_7[j_3 * 2]);
                        float _cvt_f32_14 = __bfloat162float(_cvt_bf16_22);
                        float _max_18 = max_noftz(_cvt_f32_14, -activation_clamp);
                        float _min_16 = fminf(_max_18, activation_clamp);
                        float uu0_1 = _min_16;
                        __nv_bfloat16 _cvt_bf16_23 = __float2bfloat16(_tmem_load_7[j_3 * 2 + 1]);
                        float _cvt_f32_15 = __bfloat162float(_cvt_bf16_23);
                        float _max_19 = max_noftz(_cvt_f32_15, -activation_clamp);
                        float _min_17 = fminf(_max_19, activation_clamp);
                        float uu1_1 = _min_17;
                        float _exp2_8 = approx_exp2((-gg0_1) * 1.4426950408889634f);
                        float _rcp_6 = approx_rcp(1.0f + _exp2_8);
                        float ss0_1 = _rcp_6;
                        float _exp2_9 = approx_exp2((-gg1_1) * 1.4426950408889634f);
                        float _rcp_7 = approx_rcp(1.0f + _exp2_9);
                        float ss1_1 = _rcp_7;
                        result_1[j_3 * 2] = gg0_1 * ss0_1 * uu0_1 * rw;
                        result_1[j_3 * 2 + 1] = gg1_1 * ss1_1 * uu1_1 * rw;
                    }
                    {
                        const float2 _prescale2_1 = {sf_inv_2, sf_inv_2};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&result_1[0])[_ps], _prescale2_1);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            result_1[0 + _ps] *= sf_inv_2;
                        #endif
                        unsigned int _fp8_pk[2];
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[0]) : "f"(result_1[0 + 0]), "f"(result_1[0 + 1]), "f"(result_1[0 + 2]), "f"(result_1[0 + 3]));
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_fp8_pk[1]) : "f"(result_1[0 + 4]), "f"(result_1[0 + 5]), "f"(result_1[0 + 6]), "f"(result_1[0 + 7]));
                        *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(I_fp8_w + (token * n_out1 + out_n_base + 32 + c_3 * 8)) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                __threadfence();
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gc_p = reinterpret_cast<unsigned int*>(l1_arrival) + (m_tile * K2_tiles + n_tile / 4);
                            unsigned int _gc_old;
                            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epi_done_addr + (e_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                e_stage += 1;
                if (e_stage == 2) { e_stage = 0; _phase_main_done ^= 1; }
            }
            int num_tiles2e = total_m_tiles * grid_n2;
            #pragma unroll 1
            for (unsigned int this_bid_1 = bid; this_bid_1 < num_tiles2e; this_bid_1 += num_bids) {
                int m_tile2 = this_bid_1 / (unsigned int)(grid_n2 * 2) * 2 + this_bid_1 % 2;
                int n_tile2 = this_bid_1 / 2 % (unsigned int)grid_n2;
                int local_m2 = tile_m_local[m_tile2];
                int off_m2 = expert_row_offsets[tile_expert[m_tile2]] + local_m2;
                int out_n_base2 = n_tile2 * 128;
                mbarrier_wait(main_done_addr + (e_stage) * 8, _phase_main_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int token2 = off_m2 + epi_tid;
                int tmem_row2 = cta_rank * 128 + epi_warp * 32;
                #pragma unroll
                for (int n_chunk = 0; n_chunk < 16; n_chunk++) {
                    int col = e_stage * 128 + (unsigned int)(n_chunk * 8);
                    float _tmem_load_8[8];
                    tmem_ld_x8(&_tmem_load_8[0], taddr + (unsigned int)(tmem_row2 << 16) + (unsigned int)col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_8[0 + 0], _tmem_load_8[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_8[0 + 2], _tmem_load_8[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_8[0 + 4], _tmem_load_8[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_8[0 + 6], _tmem_load_8[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(expert_output + (token2 * N2 + out_n_base2 + n_chunk * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epi_done_addr + (e_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                e_stage += 1;
                if (e_stage == 2) { e_stage = 0; _phase_main_done ^= 1; }
            }
            __threadfence();
            asm volatile("barrier.sync 15, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(l2_done) + (0);
                        unsigned int _gc_old;
                        asm volatile("atom.release.sys.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                    }
                }
            }
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(l2_done) + (0);
                unsigned int _gc_v;
                do {
                    _gc_v = atomicAdd_system(_gc_p, 0u);
                } while (_gc_v < (unsigned int)(NUM_CTAS));
                __threadfence_system();
            }
            int ep_warp_g = epi_warp * NUM_CTAS + bid;
            int combine_warps = NUM_CTAS * 4;
            #pragma unroll 1
            for (int token_chunk = ep_warp_g; token_chunk < num_tokens; token_chunk += combine_warps) {
                int tok = token_chunk;
                int chunk = 0;
                int chunk_elems = N2;
                unsigned long long out_base = (unsigned long long)tok * (unsigned long long)N2;
                #pragma unroll 1
                for (int local_base = lane * 8; local_base < chunk_elems; local_base += 256) {
                    int base_1 = chunk * chunk_elems + local_base;
                    float acc[8];
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    #pragma unroll 1
                    for (int kk = 0; kk < top_k; kk++) {
                        int row_c = token_to_permuted[tok * top_k + kk];
                        unsigned long long row_off_c = (unsigned long long)row_c * (unsigned long long)N2;
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(expert_output + (row_off_c + (unsigned long long)base_1) + 0);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            acc[j_4] = acc[j_4] + _vec_load_1[j_4];
                        }
                    }
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(acc[0 + 4], acc[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(acc[0 + 6], acc[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y + (out_base + (unsigned long long)base_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int m_tma = 0;
            unsigned int m_epi = 0;
            int num_tiles1m = total_m_tiles * grid_n1;
            int num_tiles2m = total_m_tiles * grid_n2;
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                unsigned int _gc_v;
                do {
                    _gc_v = atomicAdd_system(_gc_p, 0u);
                } while (_gc_v < (unsigned int)(NUM_CTAS));
                __threadfence_system();
            }
            unsigned int _phase_epi_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int this_bid_2 = bid; this_bid_2 < num_tiles1m; this_bid_2 += num_bids) {
                    mbarrier_wait(epi_done_addr + (m_epi) * 8, _phase_epi_done);
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < K1_tiles; iter_k++) {
                        mbarrier_wait(tma_full_addr + (m_tma) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int init_flag = ((iter_k == 0) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (m_tma) * 3200)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (m_tma) * 3200 + 8)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (m_tma) * 3200)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (m_tma) * 3200 + 8)));
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            int _mma_b_lo_1 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (m_tma) * 8, (uint16_t)(3));
                        m_tma += 1;
                        if (m_tma == 4) { m_tma = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(main_done_addr + (m_epi) * 8, (uint16_t)(3));
                    m_epi += 1;
                    if (m_epi == 2) { m_epi = 0; _phase_epi_done ^= 1; }
                }
                #pragma unroll 1
                for (unsigned int this_bid_3 = bid; this_bid_3 < num_tiles2m; this_bid_3 += num_bids) {
                    mbarrier_wait(epi_done_addr + (m_epi) * 8, _phase_epi_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < K2_tiles; iter_k_1++) {
                        mbarrier_wait(tma_full_addr + (m_tma) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int init_flag2 = ((iter_k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (m_tma) * 3200)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (m_tma) * 3200 + 8)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (m_tma) * 3200)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (m_tma) * 3200 + 8)));
                            int _mma_a_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            int _mma_b_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag2) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _mma_a_lo_3 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            int _mma_b_lo_3 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (m_tma) * 3200;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 0, b_desc + 0,
                                    0x10a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 2, b_desc + 2,
                                    0x30a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 4, b_desc + 4,
                                    0x50a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (m_epi * 128)), a_desc + 6, b_desc + 6,
                                    0x70a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (m_tma) * 8, (uint16_t)(3));
                        m_tma += 1;
                        if (m_tma == 4) { m_tma = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(main_done_addr + (m_epi) * 8, (uint16_t)(3));
                    m_epi += 1;
                    if (m_epi == 2) { m_epi = 0; _phase_epi_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            {
                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(dispatch_done) + (0);
                unsigned int _gc_v;
                do {
                    _gc_v = atomicAdd_system(_gc_p, 0u);
                } while (_gc_v < (unsigned int)(NUM_CTAS));
                __threadfence_system();
            }
            unsigned int load_stage = 0;
            int num_tiles1 = total_m_tiles * grid_n1;
            int sf_words1 = K1 / 128;
            int sfb1_exp_stride = N1 * sf_words1;
            unsigned int _phase_mma_done = 1;
            #pragma unroll 1
            for (unsigned int this_bid_4 = bid; this_bid_4 < num_tiles1; this_bid_4 += num_bids) {
                int m_tile_1 = this_bid_4 / (unsigned int)(grid_n1 * 2) * 2 + this_bid_4 % 2;
                int n_tile_1 = this_bid_4 / 2 % (unsigned int)grid_n1;
                int exp_id_1 = tile_expert[m_tile_1];
                int local_m_1 = tile_m_local[m_tile_1];
                int off_m_1 = expert_row_offsets[exp_id_1] + local_m_1;
                int off_n = n_tile_1 * 128;
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < K1_tiles; iter_k_2++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int sfa_base_smem = smem_sfa_addr + load_stage * 51200;
                    int sfb_base_smem = smem_sfb_addr + load_stage * 51200;
                    int sf_row = lane;
                    int sf_c = lane / 8;
                    int sf_d = lane % 8;
                    int sf_dst0 = (sf_c * 2 * 8 + sf_d) * 16;
                    int sf_dst1 = ((sf_c * 2 + 1) * 8 + sf_d) * 16;
                    int sfa_word_off = (off_m_1 + sf_row) * sf_words1 + iter_k_2 * 2;
                    int sfb_word_off = exp_id_1 * sfb1_exp_stride + (off_n + sf_row) * sf_words1 + iter_k_2 * 2;
                    unsigned int a0 = pool_sf[sfa_word_off];
                    unsigned int a1 = pool_sf[sfa_word_off + 1];
                    unsigned int b0 = SFB1[sfb_word_off];
                    unsigned int b1 = SFB1[sfb_word_off + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0), "r"(a0));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1), "r"(a1));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0), "r"(b0));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1), "r"(b1));
                    int sf_row_0 = 32 + lane;
                    int sf_c_1 = lane / 8;
                    int sf_d_2 = lane % 8;
                    int sf_dst0_3 = (sf_c_1 * 2 * 8 + sf_d_2) * 16 + 4;
                    int sf_dst1_4 = ((sf_c_1 * 2 + 1) * 8 + sf_d_2) * 16 + 4;
                    int sfa_word_off_5 = (off_m_1 + sf_row_0) * sf_words1 + iter_k_2 * 2;
                    int sfb_word_off_6 = exp_id_1 * sfb1_exp_stride + (off_n + sf_row_0) * sf_words1 + iter_k_2 * 2;
                    unsigned int a0_7 = pool_sf[sfa_word_off_5];
                    unsigned int a1_8 = pool_sf[sfa_word_off_5 + 1];
                    unsigned int b0_9 = SFB1[sfb_word_off_6];
                    unsigned int b1_10 = SFB1[sfb_word_off_6 + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_3), "r"(a0_7));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_4), "r"(a1_8));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_3), "r"(b0_9));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_4), "r"(b1_10));
                    int sf_row_11 = 64 + lane;
                    int sf_c_12 = lane / 8;
                    int sf_d_13 = lane % 8;
                    int sf_dst0_14 = (sf_c_12 * 2 * 8 + sf_d_13) * 16 + 8;
                    int sf_dst1_15 = ((sf_c_12 * 2 + 1) * 8 + sf_d_13) * 16 + 8;
                    int sfa_word_off_16 = (off_m_1 + sf_row_11) * sf_words1 + iter_k_2 * 2;
                    int sfb_word_off_17 = exp_id_1 * sfb1_exp_stride + (off_n + sf_row_11) * sf_words1 + iter_k_2 * 2;
                    unsigned int a0_18 = pool_sf[sfa_word_off_16];
                    unsigned int a1_19 = pool_sf[sfa_word_off_16 + 1];
                    unsigned int b0_20 = SFB1[sfb_word_off_17];
                    unsigned int b1_21 = SFB1[sfb_word_off_17 + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_14), "r"(a0_18));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_15), "r"(a1_19));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_14), "r"(b0_20));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_15), "r"(b1_21));
                    int sf_row_22 = 96 + lane;
                    int sf_c_23 = lane / 8;
                    int sf_d_24 = lane % 8;
                    int sf_dst0_25 = (sf_c_23 * 2 * 8 + sf_d_24) * 16 + 12;
                    int sf_dst1_26 = ((sf_c_23 * 2 + 1) * 8 + sf_d_24) * 16 + 12;
                    int sfa_word_off_27 = (off_m_1 + sf_row_22) * sf_words1 + iter_k_2 * 2;
                    int sfb_word_off_28 = exp_id_1 * sfb1_exp_stride + (off_n + sf_row_22) * sf_words1 + iter_k_2 * 2;
                    unsigned int a0_29 = pool_sf[sfa_word_off_27];
                    unsigned int a1_30 = pool_sf[sfa_word_off_27 + 1];
                    unsigned int b0_31 = SFB1[sfb_word_off_28];
                    unsigned int b1_32 = SFB1[sfb_word_off_28 + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_25), "r"(a0_29));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_26), "r"(a1_30));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_25), "r"(b0_31));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_26), "r"(b1_32));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 51200, A, 0, off_m_1, iter_k_2 * 2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 51200, B1, 0, off_n + cta_rank * 64, iter_k_2 * 2, exp_id_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
            int num_tiles2 = total_m_tiles * grid_n2;
            int sf_words2 = K2 / 128;
            int sfb2_exp_stride = N2 * sf_words2;
            #pragma unroll 1
            for (unsigned int this_bid_5 = bid; this_bid_5 < num_tiles2; this_bid_5 += num_bids) {
                int m_tile2_1 = this_bid_5 / (unsigned int)(grid_n2 * 2) * 2 + this_bid_5 % 2;
                int n_tile2_1 = this_bid_5 / 2 % (unsigned int)grid_n2;
                int exp_id2 = tile_expert[m_tile2_1];
                int local_m2_1 = tile_m_local[m_tile2_1];
                int off_m2_1 = expert_row_offsets[exp_id2] + local_m2_1;
                int off_n2 = n_tile2_1 * 128;
                #pragma unroll 1
                for (int iter_k_3 = 0; iter_k_3 < K2_tiles; iter_k_3++) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(l1_arrival) + (m_tile2_1 * K2_tiles + iter_k_3);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(4)) break;
                        }
                    }
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int sfa2_base_smem = smem_sfa_addr + load_stage * 51200;
                    int sfb2_base_smem = smem_sfb_addr + load_stage * 51200;
                    int sf_row2 = lane;
                    int sf_c2 = lane / 8;
                    int sf_d2 = lane % 8;
                    int sf2_dst0 = (sf_c2 * 2 * 8 + sf_d2) * 16;
                    int sf2_dst1 = ((sf_c2 * 2 + 1) * 8 + sf_d2) * 16;
                    int sfa2_word_off = (off_m2_1 + sf_row2) * sf_words2 + iter_k_3 * 2;
                    int sfb2_word_off = exp_id2 * sfb2_exp_stride + (off_n2 + sf_row2) * sf_words2 + iter_k_3 * 2;
                    unsigned int a2w0 = SFA2[sfa2_word_off];
                    unsigned int a2w1 = SFA2[sfa2_word_off + 1];
                    unsigned int b2w0 = SFB2[sfb2_word_off];
                    unsigned int b2w1 = SFB2[sfb2_word_off + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst0), "r"(a2w0));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst1), "r"(a2w1));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst0), "r"(b2w0));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst1), "r"(b2w1));
                    int sf_row2_0 = 32 + lane;
                    int sf_c2_1 = lane / 8;
                    int sf_d2_2 = lane % 8;
                    int sf2_dst0_3 = (sf_c2_1 * 2 * 8 + sf_d2_2) * 16 + 4;
                    int sf2_dst1_4 = ((sf_c2_1 * 2 + 1) * 8 + sf_d2_2) * 16 + 4;
                    int sfa2_word_off_5 = (off_m2_1 + sf_row2_0) * sf_words2 + iter_k_3 * 2;
                    int sfb2_word_off_6 = exp_id2 * sfb2_exp_stride + (off_n2 + sf_row2_0) * sf_words2 + iter_k_3 * 2;
                    unsigned int a2w0_7 = SFA2[sfa2_word_off_5];
                    unsigned int a2w1_8 = SFA2[sfa2_word_off_5 + 1];
                    unsigned int b2w0_9 = SFB2[sfb2_word_off_6];
                    unsigned int b2w1_10 = SFB2[sfb2_word_off_6 + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst0_3), "r"(a2w0_7));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst1_4), "r"(a2w1_8));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst0_3), "r"(b2w0_9));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst1_4), "r"(b2w1_10));
                    int sf_row2_11 = 64 + lane;
                    int sf_c2_12 = lane / 8;
                    int sf_d2_13 = lane % 8;
                    int sf2_dst0_14 = (sf_c2_12 * 2 * 8 + sf_d2_13) * 16 + 8;
                    int sf2_dst1_15 = ((sf_c2_12 * 2 + 1) * 8 + sf_d2_13) * 16 + 8;
                    int sfa2_word_off_16 = (off_m2_1 + sf_row2_11) * sf_words2 + iter_k_3 * 2;
                    int sfb2_word_off_17 = exp_id2 * sfb2_exp_stride + (off_n2 + sf_row2_11) * sf_words2 + iter_k_3 * 2;
                    unsigned int a2w0_18 = SFA2[sfa2_word_off_16];
                    unsigned int a2w1_19 = SFA2[sfa2_word_off_16 + 1];
                    unsigned int b2w0_20 = SFB2[sfb2_word_off_17];
                    unsigned int b2w1_21 = SFB2[sfb2_word_off_17 + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst0_14), "r"(a2w0_18));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst1_15), "r"(a2w1_19));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst0_14), "r"(b2w0_20));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst1_15), "r"(b2w1_21));
                    int sf_row2_22 = 96 + lane;
                    int sf_c2_23 = lane / 8;
                    int sf_d2_24 = lane % 8;
                    int sf2_dst0_25 = (sf_c2_23 * 2 * 8 + sf_d2_24) * 16 + 12;
                    int sf2_dst1_26 = ((sf_c2_23 * 2 + 1) * 8 + sf_d2_24) * 16 + 12;
                    int sfa2_word_off_27 = (off_m2_1 + sf_row2_22) * sf_words2 + iter_k_3 * 2;
                    int sfb2_word_off_28 = exp_id2 * sfb2_exp_stride + (off_n2 + sf_row2_22) * sf_words2 + iter_k_3 * 2;
                    unsigned int a2w0_29 = SFA2[sfa2_word_off_27];
                    unsigned int a2w1_30 = SFA2[sfa2_word_off_27 + 1];
                    unsigned int b2w0_31 = SFB2[sfb2_word_off_28];
                    unsigned int b2w1_32 = SFB2[sfb2_word_off_28 + 1];
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst0_25), "r"(a2w0_29));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa2_base_smem + sf2_dst1_26), "r"(a2w1_30));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst0_25), "r"(b2w0_31));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb2_base_smem + sf2_dst1_26), "r"(b2w1_32));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 51200, A2, 0, off_m2_1, iter_k_3 * 2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 51200, B2, 0, off_n2 + cta_rank * 64, iter_k_3 * 2, exp_id2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
