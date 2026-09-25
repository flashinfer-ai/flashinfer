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
template <int N>
struct __align__(128) DeepgemmTensorMapPack { DeepgemmTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(DeepgemmTensorMap) >= alignof(CUtensorMap), "DeepgemmTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define DEEPGEMM_INF CUDART_INF_F
#define TMEM_NCOLS 272
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 2
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
#define SMEM_TOTAL 205824

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

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_deepgemm_mega_moe_v3_sm100a_fd3026b4e883d2df2003(DeepgemmTensorMap const* A, DeepgemmTensorMap const* B, uint8_t* __restrict__ SFA, uint8_t* __restrict__ SFB, uint8_t* __restrict__ C_fp8, uint8_t* __restrict__ SF_out, float* __restrict__ routing_weights, int* __restrict__ tile_expert, int* __restrict__ tile_m_local, int* __restrict__ expert_row_offsets, int M_total, int N, int K, int grid_n, int K_tiles, int total_m_tiles)
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
    #define mainloop_done_addr (mbar_base + 64)
    #define epilogue_done_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
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
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

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

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = (unsigned int)(epi_warp * 32) + lane;
            int num_tiles = total_m_tiles * grid_n;
            int n_out = N / 2;
            int sf_cols_out = n_out / 32;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
                int m_tile = this_bid / (unsigned int)(grid_n * 2) * 2 + this_bid % 2;
                int n_tile = this_bid / 2 % (unsigned int)grid_n;
                int exp_id = tile_expert[m_tile];
                int local_m = tile_m_local[m_tile];
                int row_off = expert_row_offsets[exp_id];
                int off_m = row_off + local_m;
                int out_n_base = n_tile * 64;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int token = off_m + epi_tid;
                float rw = routing_weights[token];
                int tmem_row = cta_rank * 128 + epi_warp * 32;
                float amax = 1e-07f;
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    int pair_base = epi_stage * 128 + (unsigned int)(c * 16);
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float g0 = _tmem_load_0[j * 2];
                        float g1 = _tmem_load_0[j * 2 + 1];
                        float u0 = _tmem_load_1[j * 2];
                        float u1 = _tmem_load_1[j * 2 + 1];
                        float _exp2_0 = approx_exp2((-g0) * 1.4426950408889634f);
                        float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                        float s0 = _rcp_0;
                        float _exp2_1 = approx_exp2((-g1) * 1.4426950408889634f);
                        float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                        float s1 = _rcp_1;
                        float r0 = g0 * s0 * u0 * rw;
                        float r1 = g1 * s1 * u1 * rw;
                        float _max_0 = max_noftz(r0, -r0);
                        float _max_1 = max_noftz(r1, -r1);
                        float _max_2 = max_noftz(_max_0, _max_1);
                        float _max_3 = max_noftz(amax, _max_2);
                        amax = _max_3;
                    }
                }
                float xsf = amax * 0.002232142857142857f;
                unsigned int xbits = __as_u32(xsf);
                unsigned int exp_e = (xbits >> 23 & 255) + ((xbits & 8388607) + 8388607 >> 23);
                unsigned int _max_4 = ((exp_e) > (1) ? (exp_e) : (1));
                exp_e = _max_4;
                unsigned int _min_0 = ((exp_e) < (254) ? (exp_e) : (254));
                exp_e = _min_0;
                int exp_i = exp_e;
                float neg_exp_f = 127 - exp_i;
                float _exp2_2 = approx_exp2(neg_exp_f);
                float sf_inv = _exp2_2;
                uint8_t sf_byte = exp_e;
                SF_out[token * sf_cols_out + n_tile * 2] = sf_byte;
                #pragma unroll
                for (int c_1 = 0; c_1 < 4; c_1++) {
                    int pair_base2 = epi_stage * 128 + (unsigned int)(c_1 * 16);
                    float _tmem_load_2[8];
                    tmem_ld_x8(&_tmem_load_2[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base2);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_3[8];
                    tmem_ld_x8(&_tmem_load_3[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base2 + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float result[8];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 4; j_1++) {
                        float gg0 = _tmem_load_2[j_1 * 2];
                        float gg1 = _tmem_load_2[j_1 * 2 + 1];
                        float uu0 = _tmem_load_3[j_1 * 2];
                        float uu1 = _tmem_load_3[j_1 * 2 + 1];
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
                        *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(C_fp8 + (token * n_out + out_n_base + c_1 * 8)) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                    }
                }
                float amax_0 = 1e-07f;
                #pragma unroll
                for (int c_2 = 0; c_2 < 4; c_2++) {
                    int pair_base_1 = epi_stage * 128 + (unsigned int)((4 + c_2) * 16);
                    float _tmem_load_4[8];
                    tmem_ld_x8(&_tmem_load_4[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_5[8];
                    tmem_ld_x8(&_tmem_load_5[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base_1 + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 4; j_2++) {
                        float g0_1 = _tmem_load_4[j_2 * 2];
                        float g1_1 = _tmem_load_4[j_2 * 2 + 1];
                        float u0_1 = _tmem_load_5[j_2 * 2];
                        float u1_1 = _tmem_load_5[j_2 * 2 + 1];
                        float _exp2_5 = approx_exp2((-g0_1) * 1.4426950408889634f);
                        float _rcp_4 = approx_rcp(1.0f + _exp2_5);
                        float s0_1 = _rcp_4;
                        float _exp2_6 = approx_exp2((-g1_1) * 1.4426950408889634f);
                        float _rcp_5 = approx_rcp(1.0f + _exp2_6);
                        float s1_1 = _rcp_5;
                        float r0_1 = g0_1 * s0_1 * u0_1 * rw;
                        float r1_1 = g1_1 * s1_1 * u1_1 * rw;
                        float _max_5 = max_noftz(r0_1, -r0_1);
                        float _max_6 = max_noftz(r1_1, -r1_1);
                        float _max_7 = max_noftz(_max_5, _max_6);
                        float _max_8 = max_noftz(amax_0, _max_7);
                        amax_0 = _max_8;
                    }
                }
                float xsf_1 = amax_0 * 0.002232142857142857f;
                unsigned int xbits_2 = __as_u32(xsf_1);
                unsigned int exp_e_3 = (xbits_2 >> 23 & 255) + ((xbits_2 & 8388607) + 8388607 >> 23);
                unsigned int _max_9 = ((exp_e_3) > (1) ? (exp_e_3) : (1));
                exp_e_3 = _max_9;
                unsigned int _min_1 = ((exp_e_3) < (254) ? (exp_e_3) : (254));
                exp_e_3 = _min_1;
                int exp_i_4 = exp_e_3;
                float neg_exp_f_5 = 127 - exp_i_4;
                float _exp2_7 = approx_exp2(neg_exp_f_5);
                float sf_inv_6 = _exp2_7;
                uint8_t sf_byte_7 = exp_e_3;
                SF_out[token * sf_cols_out + n_tile * 2 + 1] = sf_byte_7;
                #pragma unroll
                for (int c_3 = 0; c_3 < 4; c_3++) {
                    int pair_base2_1 = epi_stage * 128 + (unsigned int)((4 + c_3) * 16);
                    float _tmem_load_6[8];
                    tmem_ld_x8(&_tmem_load_6[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base2_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_7[8];
                    tmem_ld_x8(&_tmem_load_7[0], taddr + (unsigned int)(tmem_row << 16) + (unsigned int)pair_base2_1 + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float result_1[8];
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 4; j_3++) {
                        float gg0_1 = _tmem_load_6[j_3 * 2];
                        float gg1_1 = _tmem_load_6[j_3 * 2 + 1];
                        float uu0_1 = _tmem_load_7[j_3 * 2];
                        float uu1_1 = _tmem_load_7[j_3 * 2 + 1];
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
                        const float2 _prescale2_1 = {sf_inv_6, sf_inv_6};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&result_1[0])[_ps], _prescale2_1);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            result_1[0 + _ps] *= sf_inv_6;
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
                        *reinterpret_cast<uint2*>(reinterpret_cast<unsigned char*>(C_fp8 + (token * n_out + out_n_base + 32 + c_3 * 8)) + (0)) = *reinterpret_cast<uint2*>(_fp8_pk);
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            int num_tiles_1 = total_m_tiles * grid_n;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int this_bid_1 = bid; this_bid_1 < num_tiles_1; this_bid_1 += num_bids) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int tmem_buf = taddr + mma_epi_stage * 128;
                        int init_flag = ((iter_k == 0) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3200)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (mma_tma_stage) * 3200 + 8)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 3200)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (mma_tma_stage) * 3200 + 8)));
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x10a01400U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x30a01410U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 4, b_desc + 4,
                                    0x50a01420U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 6, b_desc + 6,
                                    0x70a01430U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            }
                            int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                            int _mma_b_lo_1 = (((smem_b_addr + 8192) >> 4) & 0x3FFF) + (mma_tma_stage) * 3200;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x10a01400U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x30a01410U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 4, b_desc + 4,
                                    0x50a01420U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_tmem_acc + (mma_epi_stage * 128)), a_desc + 6, b_desc + 6,
                                    0x70a01430U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            unsigned int load_stage = 0;
            int num_tiles_2 = total_m_tiles * grid_n;
            int sf_cols = K / 128;
            int sfb_exp_stride = N * sf_cols;
            unsigned int _phase_mma_done = 1;
            #pragma unroll 1
            for (unsigned int this_bid_2 = bid; this_bid_2 < num_tiles_2; this_bid_2 += num_bids) {
                int m_tile_1 = this_bid_2 / (unsigned int)(grid_n * 2) * 2 + this_bid_2 % 2;
                int n_tile_1 = this_bid_2 / 2 % (unsigned int)grid_n;
                int exp_id_1 = tile_expert[m_tile_1];
                int local_m_1 = tile_m_local[m_tile_1];
                int row_off_1 = expert_row_offsets[exp_id_1];
                int off_m_1 = row_off_1 + local_m_1;
                int off_n = n_tile_1 * 128;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    int sfa_base_smem = smem_sfa_addr + load_stage * 51200;
                    int sfb_base_smem = smem_sfb_addr + load_stage * 51200;
                    int sf_row = lane;
                    int sf_c = lane / 8;
                    int sf_d = lane % 8;
                    int sf_dst0 = (sf_c * 2 * 8 + sf_d) * 16;
                    int sf_dst1 = ((sf_c * 2 + 1) * 8 + sf_d) * 16;
                    int sfa_idx0 = (off_m_1 + sf_row) * sf_cols + iter_k_1 * 2;
                    int sfa_idx1 = sfa_idx0 + 1;
                    int sfb_idx0 = exp_id_1 * sfb_exp_stride + (off_n + sf_row) * sf_cols + iter_k_1 * 2;
                    int sfb_idx1 = sfb_idx0 + 1;
                    unsigned int sfa0 = SFA[sfa_idx0];
                    unsigned int sfa1 = SFA[sfa_idx1];
                    unsigned int sfb0 = SFB[sfb_idx0];
                    unsigned int sfb1 = SFB[sfb_idx1];
                    unsigned int sfa0_word = sfa0 | sfa0 << 8 | sfa0 << 16 | sfa0 << 24;
                    unsigned int sfa1_word = sfa1 | sfa1 << 8 | sfa1 << 16 | sfa1 << 24;
                    unsigned int sfb0_word = sfb0 | sfb0 << 8 | sfb0 << 16 | sfb0 << 24;
                    unsigned int sfb1_word = sfb1 | sfb1 << 8 | sfb1 << 16 | sfb1 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0), "r"(sfa0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1), "r"(sfa1_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1), "r"(sfb1_word));
                    int sf_row_0 = 32 + lane;
                    int sf_c_1 = lane / 8;
                    int sf_d_2 = lane % 8;
                    int sf_dst0_3 = (sf_c_1 * 2 * 8 + sf_d_2) * 16 + 4;
                    int sf_dst1_4 = ((sf_c_1 * 2 + 1) * 8 + sf_d_2) * 16 + 4;
                    int sfa_idx0_5 = (off_m_1 + sf_row_0) * sf_cols + iter_k_1 * 2;
                    int sfa_idx1_6 = sfa_idx0_5 + 1;
                    int sfb_idx0_7 = exp_id_1 * sfb_exp_stride + (off_n + sf_row_0) * sf_cols + iter_k_1 * 2;
                    int sfb_idx1_8 = sfb_idx0_7 + 1;
                    unsigned int sfa0_9 = SFA[sfa_idx0_5];
                    unsigned int sfa1_10 = SFA[sfa_idx1_6];
                    unsigned int sfb0_11 = SFB[sfb_idx0_7];
                    unsigned int sfb1_12 = SFB[sfb_idx1_8];
                    unsigned int sfa0_word_13 = sfa0_9 | sfa0_9 << 8 | sfa0_9 << 16 | sfa0_9 << 24;
                    unsigned int sfa1_word_14 = sfa1_10 | sfa1_10 << 8 | sfa1_10 << 16 | sfa1_10 << 24;
                    unsigned int sfb0_word_15 = sfb0_11 | sfb0_11 << 8 | sfb0_11 << 16 | sfb0_11 << 24;
                    unsigned int sfb1_word_16 = sfb1_12 | sfb1_12 << 8 | sfb1_12 << 16 | sfb1_12 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_3), "r"(sfa0_word_13));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_4), "r"(sfa1_word_14));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_3), "r"(sfb0_word_15));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_4), "r"(sfb1_word_16));
                    int sf_row_17 = 64 + lane;
                    int sf_c_18 = lane / 8;
                    int sf_d_19 = lane % 8;
                    int sf_dst0_20 = (sf_c_18 * 2 * 8 + sf_d_19) * 16 + 8;
                    int sf_dst1_21 = ((sf_c_18 * 2 + 1) * 8 + sf_d_19) * 16 + 8;
                    int sfa_idx0_22 = (off_m_1 + sf_row_17) * sf_cols + iter_k_1 * 2;
                    int sfa_idx1_23 = sfa_idx0_22 + 1;
                    int sfb_idx0_24 = exp_id_1 * sfb_exp_stride + (off_n + sf_row_17) * sf_cols + iter_k_1 * 2;
                    int sfb_idx1_25 = sfb_idx0_24 + 1;
                    unsigned int sfa0_26 = SFA[sfa_idx0_22];
                    unsigned int sfa1_27 = SFA[sfa_idx1_23];
                    unsigned int sfb0_28 = SFB[sfb_idx0_24];
                    unsigned int sfb1_29 = SFB[sfb_idx1_25];
                    unsigned int sfa0_word_30 = sfa0_26 | sfa0_26 << 8 | sfa0_26 << 16 | sfa0_26 << 24;
                    unsigned int sfa1_word_31 = sfa1_27 | sfa1_27 << 8 | sfa1_27 << 16 | sfa1_27 << 24;
                    unsigned int sfb0_word_32 = sfb0_28 | sfb0_28 << 8 | sfb0_28 << 16 | sfb0_28 << 24;
                    unsigned int sfb1_word_33 = sfb1_29 | sfb1_29 << 8 | sfb1_29 << 16 | sfb1_29 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_20), "r"(sfa0_word_30));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_21), "r"(sfa1_word_31));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_20), "r"(sfb0_word_32));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_21), "r"(sfb1_word_33));
                    int sf_row_34 = 96 + lane;
                    int sf_c_35 = lane / 8;
                    int sf_d_36 = lane % 8;
                    int sf_dst0_37 = (sf_c_35 * 2 * 8 + sf_d_36) * 16 + 12;
                    int sf_dst1_38 = ((sf_c_35 * 2 + 1) * 8 + sf_d_36) * 16 + 12;
                    int sfa_idx0_39 = (off_m_1 + sf_row_34) * sf_cols + iter_k_1 * 2;
                    int sfa_idx1_40 = sfa_idx0_39 + 1;
                    int sfb_idx0_41 = exp_id_1 * sfb_exp_stride + (off_n + sf_row_34) * sf_cols + iter_k_1 * 2;
                    int sfb_idx1_42 = sfb_idx0_41 + 1;
                    unsigned int sfa0_43 = SFA[sfa_idx0_39];
                    unsigned int sfa1_44 = SFA[sfa_idx1_40];
                    unsigned int sfb0_45 = SFB[sfb_idx0_41];
                    unsigned int sfb1_46 = SFB[sfb_idx1_42];
                    unsigned int sfa0_word_47 = sfa0_43 | sfa0_43 << 8 | sfa0_43 << 16 | sfa0_43 << 24;
                    unsigned int sfa1_word_48 = sfa1_44 | sfa1_44 << 8 | sfa1_44 << 16 | sfa1_44 << 24;
                    unsigned int sfb0_word_49 = sfb0_45 | sfb0_45 << 8 | sfb0_45 << 16 | sfb0_45 << 24;
                    unsigned int sfb1_word_50 = sfb1_46 | sfb1_46 << 8 | sfb1_46 << 16 | sfb1_46 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst0_37), "r"(sfa0_word_47));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_base_smem + sf_dst1_38), "r"(sfa1_word_48));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst0_37), "r"(sfb0_word_49));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_base_smem + sf_dst1_38), "r"(sfb1_word_50));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 51200, A, 0, off_m_1, iter_k_1 * 2, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 51200, B, 0, off_n + cta_rank * 64, iter_k_1 * 2, exp_id_1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(40960)) : "memory");
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
