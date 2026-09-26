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
#define TMEM_NCOLS 40
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SCALE_A_OFFSET 32
#define TMEM_SCALE_B_OFFSET 36
#define NUM_PIPE_STAGES 11
#define SMEM_HISTOGRAM_OFF 0
#define SMEM_HISTOGRAM_STAGE_BYTES 1536
#define SMEM_HISTOGRAM_STRIDE 1536
#define SMEM_SEND_BUFFER_OFF 2048
#define SMEM_SEND_BUFFER_STAGE_BYTES 20480
#define SMEM_SEND_BUFFER_STRIDE 20480
#define SMEM_SMEM_OUTPUT_OFF 22528
#define SMEM_SMEM_OUTPUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUTPUT_STRIDE 4096
#define SMEM_REUSABLE_OFF 0
#define SMEM_REUSABLE_STAGE_BYTES 229696
#define SMEM_REUSABLE_STRIDE 229696
#define SMEM_SMEM_A_OFF 26624
#define SMEM_SMEM_A_STAGE_BYTES 1024
#define SMEM_SMEM_A_STRIDE 1024
#define SMEM_SMEM_B_OFF 37888
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_SHARED_B_OFF 37888
#define SMEM_SMEM_SHARED_B_STAGE_BYTES 16384
#define SMEM_SMEM_SHARED_B_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 218112
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 512
#define SMEM_SMEM_SFB_OFF 223744
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 512
#define SMEM_AMAX_REDUCTION_OFF 229376
#define SMEM_AMAX_REDUCTION_STAGE_BYTES 256
#define SMEM_AMAX_REDUCTION_STRIDE 256
#define SMEM_TASK_INFOS_OFF 229632
#define SMEM_TASK_INFOS_STAGE_BYTES 64
#define SMEM_TASK_INFOS_STRIDE 64
#define SMEM_TOTAL 230528
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
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


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
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


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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


__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
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

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_deepgemm_source_mega_moe_sm103a_0d48de6e087e9d55de3d(DeepgemmTensorMap const* A1, DeepgemmTensorMap const* A2, DeepgemmTensorMap const* SA1, DeepgemmTensorMap const* SA2, DeepgemmTensorMap const* B1, DeepgemmTensorMap const* B2, DeepgemmTensorMap const* SB1, DeepgemmTensorMap const* SB2, DeepgemmTensorMap const* SFA1, DeepgemmTensorMap const* SFA2, DeepgemmTensorMap const* SSFA1, DeepgemmTensorMap const* SSFA2, DeepgemmTensorMap const* SFB1, DeepgemmTensorMap const* SFB2, DeepgemmTensorMap const* SSFB1, DeepgemmTensorMap const* SSFB2, DeepgemmTensorMap const* L1Output, DeepgemmTensorMap const* SharedL1Output, uint8_t* __restrict__ X, unsigned int* __restrict__ XSF, long long* __restrict__ TopK, float* __restrict__ Weights, uint8_t* __restrict__ L1Acts, unsigned int* __restrict__ L1SF, float* __restrict__ L1Weights, uint8_t* __restrict__ L2SF, uint8_t* __restrict__ SharedL2SF, unsigned int* __restrict__ SourceIndices, unsigned int* __restrict__ TokenMetadata, unsigned int* __restrict__ GridCounters, unsigned int* __restrict__ NvlCounter, unsigned int* __restrict__ NvlSignals, unsigned long long* __restrict__ PeerGrid, unsigned long long* __restrict__ ReadyGrid, unsigned long long* __restrict__ SendCounts, unsigned long long* __restrict__ RecvCounts, unsigned long long* __restrict__ RecvSum, unsigned int* __restrict__ L1Full, unsigned int* __restrict__ L1Empty, unsigned long long* __restrict__ L2Mask, unsigned int* __restrict__ L2Empty, unsigned int* __restrict__ SharedFull, unsigned int* __restrict__ L1Counter, unsigned int* __restrict__ L2Counter, unsigned int* __restrict__ SharedL1Counter, unsigned int* __restrict__ SharedL2Counter, unsigned int* __restrict__ Combine, uint8_t* __restrict__ CombineBytes, uint8_t* __restrict__ Y, unsigned int num_tokens)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 229696;
    #define pull_barriers_addr (mbar_base + 0)
    #define full_addr (mbar_base + 32)
    #define empty_addr (mbar_base + 120)
    #define tmem_full_addr (mbar_base + 208)
    #define tmem_empty_addr (mbar_base + 224)
    #define combine_barriers_addr (mbar_base + 240)
    #define task_full_addr (mbar_base + 688)
    #define task_empty_addr (mbar_base + 704)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SA1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SA2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SB1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SB2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFA1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFA2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SSFA1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SSFA2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFB1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFB2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SSFB1)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SSFB2)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(L1Output)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SharedL1Output)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    unsigned int* histogram = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int histogram_addr = smem + 0;
    uint8_t* send_buffer = reinterpret_cast<uint8_t*>(smem_raw + 2048);
    const int send_buffer_addr = smem + 2048;
    uint8_t* smem_output = reinterpret_cast<uint8_t*>(smem_raw + 22528);
    const int smem_output_addr = smem + 22528;
    uint8_t* reusable = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int reusable_addr = smem + 0;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 26624);
    const int smem_a_addr = smem + 26624;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_b_addr = smem + 37888;
    uint8_t* smem_shared_b = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_shared_b_addr = smem + 37888;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 218112);
    const int smem_sfa_addr = smem + 218112;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 223744);
    const int smem_sfb_addr = smem + 223744;
    float* amax_reduction = reinterpret_cast<float*>(smem_raw + 229376);
    const int amax_reduction_addr = smem + 229376;
    unsigned int* task_infos = reinterpret_cast<unsigned int*>(smem_raw + 229632);
    const int task_infos_addr = smem + 229632;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
            uint32_t _smem_bulk_zero_addr_0 = static_cast<uint32_t>(histogram_addr);
            asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;"
                :: "r"(_smem_bulk_zero_addr_0), "l"(static_cast<uint64_t>(2048)) : "memory");
            #elif defined(__CUDA_ARCH__)
            #error "SmemBulkZero requires SM100 or newer"
            #endif
        }
    }

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 90 barriers)
    // Mbarriers at smem_raw[229696..230416)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // pull_barriers: 4 barriers, init_count=1
            mbarrier_init(smem + 229696, 1);
            mbarrier_init(smem + 229704, 1);
            mbarrier_init(smem + 229712, 1);
            mbarrier_init(smem + 229720, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 11 barriers, init_count=4
            mbarrier_init(smem + 229728, 4);
            mbarrier_init(smem + 229736, 4);
            mbarrier_init(smem + 229744, 4);
            mbarrier_init(smem + 229752, 4);
            mbarrier_init(smem + 229760, 4);
            mbarrier_init(smem + 229768, 4);
            mbarrier_init(smem + 229776, 4);
            mbarrier_init(smem + 229784, 4);
            mbarrier_init(smem + 229792, 4);
            mbarrier_init(smem + 229800, 4);
            mbarrier_init(smem + 229808, 4);
            // empty: 11 barriers, init_count=1
            mbarrier_init(smem + 229816, 1);
            mbarrier_init(smem + 229824, 1);
            mbarrier_init(smem + 229832, 1);
            mbarrier_init(smem + 229840, 1);
            mbarrier_init(smem + 229848, 1);
            mbarrier_init(smem + 229856, 1);
            mbarrier_init(smem + 229864, 1);
            mbarrier_init(smem + 229872, 1);
            mbarrier_init(smem + 229880, 1);
            mbarrier_init(smem + 229888, 1);
            mbarrier_init(smem + 229896, 1);
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 229904, 1);
            mbarrier_init(smem + 229912, 1);
            // tmem_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 229920, 512);
            mbarrier_init(smem + 229928, 512);
            // combine_barriers: 56 barriers, init_count=1
            mbarrier_init(smem + 229936, 1);
            mbarrier_init(smem + 229944, 1);
            mbarrier_init(smem + 229952, 1);
            mbarrier_init(smem + 229960, 1);
            mbarrier_init(smem + 229968, 1);
            mbarrier_init(smem + 229976, 1);
            mbarrier_init(smem + 229984, 1);
            mbarrier_init(smem + 229992, 1);
            mbarrier_init(smem + 230000, 1);
            mbarrier_init(smem + 230008, 1);
            mbarrier_init(smem + 230016, 1);
            mbarrier_init(smem + 230024, 1);
            mbarrier_init(smem + 230032, 1);
            mbarrier_init(smem + 230040, 1);
            mbarrier_init(smem + 230048, 1);
            mbarrier_init(smem + 230056, 1);
            mbarrier_init(smem + 230064, 1);
            mbarrier_init(smem + 230072, 1);
            mbarrier_init(smem + 230080, 1);
            mbarrier_init(smem + 230088, 1);
            mbarrier_init(smem + 230096, 1);
            mbarrier_init(smem + 230104, 1);
            mbarrier_init(smem + 230112, 1);
            mbarrier_init(smem + 230120, 1);
            mbarrier_init(smem + 230128, 1);
            mbarrier_init(smem + 230136, 1);
            mbarrier_init(smem + 230144, 1);
            mbarrier_init(smem + 230152, 1);
            mbarrier_init(smem + 230160, 1);
            mbarrier_init(smem + 230168, 1);
            mbarrier_init(smem + 230176, 1);
            mbarrier_init(smem + 230184, 1);
            mbarrier_init(smem + 230192, 1);
            mbarrier_init(smem + 230200, 1);
            mbarrier_init(smem + 230208, 1);
            mbarrier_init(smem + 230216, 1);
            mbarrier_init(smem + 230224, 1);
            mbarrier_init(smem + 230232, 1);
            mbarrier_init(smem + 230240, 1);
            mbarrier_init(smem + 230248, 1);
            mbarrier_init(smem + 230256, 1);
            mbarrier_init(smem + 230264, 1);
            mbarrier_init(smem + 230272, 1);
            mbarrier_init(smem + 230280, 1);
            mbarrier_init(smem + 230288, 1);
            mbarrier_init(smem + 230296, 1);
            mbarrier_init(smem + 230304, 1);
            mbarrier_init(smem + 230312, 1);
            mbarrier_init(smem + 230320, 1);
            mbarrier_init(smem + 230328, 1);
            mbarrier_init(smem + 230336, 1);
            mbarrier_init(smem + 230344, 1);
            mbarrier_init(smem + 230352, 1);
            mbarrier_init(smem + 230360, 1);
            mbarrier_init(smem + 230368, 1);
            mbarrier_init(smem + 230376, 1);
            // task_full: 2 barriers, init_count=1
            mbarrier_init(smem + 230384, 1);
            mbarrier_init(smem + 230392, 1);
            // task_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 230400, 512);
            mbarrier_init(smem + 230408, 512);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 40 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 230416);
    if (warp == 3) {
        int _tmem_hold = smem + 230416;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_scale_a = taddr + 32;
    const int tmem_scale_b = taddr + 36;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: dispatch ----
    if (warp <= 3) {
        { // dispatch_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            unsigned int counts[12];
            unsigned int state[2];
            unsigned int first_token = ((unsigned int)(bid * 4) + warp) * 5;
            #pragma unroll
            for (int token_base = first_token; token_base < num_tokens; token_base += 3040) {
                if ((unsigned int)token_base + lane / 6 < num_tokens && lane < 30) {
                    int expert = (int)TopK[(unsigned int)(token_base * 6) + lane];
                    if (expert >= 0) {
                        uint32_t _shared_atomic_old_0;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((histogram_addr + 4 * (expert)))), "r"(static_cast<uint32_t>(1)) : "memory");
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            #pragma unroll
            for (int expert_1 = tid; expert_1 < 384; expert_1 += 128) {
                unsigned int local_count = histogram[expert_1];
                unsigned long long send_value = (unsigned long long)1 << 32 | (unsigned long long)local_count;
                unsigned long long _atomic_old_0 = atomicAdd(&SendCounts[expert_1], send_value);
                unsigned long long old = _atomic_old_0;
                histogram[expert_1] = (unsigned int)old;
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            #pragma unroll
            for (int token_base_1 = first_token; token_base_1 < num_tokens; token_base_1 += 3040) {
                if ((unsigned int)token_base_1 + lane / 6 < num_tokens && lane < 30) {
                    int expert_2 = (int)TopK[(unsigned int)(token_base_1 * 6) + lane];
                    if (expert_2 >= 0) {
                        uint32_t _shared_atomic_old_1;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_1) : "r"(static_cast<uint32_t>((histogram_addr + 4 * (expert_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        unsigned int slot = _shared_atomic_old_1;
                        SourceIndices[(unsigned int)(expert_2 * 1920) + slot] = (unsigned int)(token_base_1 * 6) + lane;
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (tid == 0) {
                unsigned int increment = 1;
                if (bid == 0) {
                    increment = 2147483497;
                }
                unsigned int _atomic_old_1;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_1) : "l"(&GridCounters[0]), "r"(static_cast<uint32_t>(increment)) : "memory");
                unsigned int old_1 = _atomic_old_1;
                unsigned int _wait_acquire_mask_0;
                do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_0) : "l"((reinterpret_cast<unsigned int*>(GridCounters) + (0))) : "memory");
                } while (((_wait_acquire_mask_0 ^ static_cast<unsigned int>((old_1 ^ 2147483648) & 2147483648)) & static_cast<unsigned int>(2147483648)) != 0);
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (bid == 0) {
                if (tid == 0) {
                    uint64_t _grid_id_0;
                    asm volatile("mov.u64 %0, %%gridid;" : "=l"(_grid_id_0));
                    PeerGrid[0] = _grid_id_0 + 1;
                }
                __syncwarp();
                #pragma unroll
                for (int expert_3 = tid; expert_3 < 384; expert_3 += 128) {
                    unsigned long long status = SendCounts[expert_3];
                    RecvCounts[expert_3] = status & 4294967295;
                    asm volatile("red.sys.global.add.u64 [%0], %1;" :: "l"(&RecvSum[expert_3]), "l"(static_cast<uint64_t>(status)) : "memory");
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (bid == 0) {
                unsigned int status_1 = NvlCounter[0] & 3;
                unsigned int phase = status_1 & 1;
                unsigned int sign = status_1 >> 1;
                if (tid == 0) {
                    unsigned int delta = ((sign == 0) ? 1 : 4294967295);
                    asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(NvlSignals) + (phase))), "r"(static_cast<unsigned int>(delta)) : "memory");
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (tid == 0) {
                    atomicAdd(&NvlCounter[0], 1);
                    unsigned int target = ((sign == 0) ? 1 : 0);
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(NvlSignals) + (phase))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(target)) >= static_cast<unsigned int>(1));
                    }
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (tid == 0) {
                unsigned int increment_1 = 1;
                if (bid == 0) {
                    increment_1 = 2147483497;
                }
                unsigned int _atomic_old_2;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_2) : "l"(&GridCounters[0]), "r"(static_cast<uint32_t>(increment_1)) : "memory");
                unsigned int old_2 = _atomic_old_2;
                unsigned int _wait_acquire_mask_1;
                do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_1) : "l"((reinterpret_cast<unsigned int*>(GridCounters) + (0))) : "memory");
                } while (((_wait_acquire_mask_1 ^ static_cast<unsigned int>((old_2 ^ 2147483648) & 2147483648)) & static_cast<unsigned int>(2147483648)) != 0);
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int pull_phase = 0;
            int current_expert = -1;
            unsigned int expert_start = 0;
            unsigned int expert_end = 0;
            unsigned int pool_offset = 0;
            uint32_t lane_read_0;
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_read_0));
            unsigned int expert_4 = lane_read_0;
            unsigned long long received = 0;
            if (expert_4 < 384) {
                while (1) {
                    received = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_4];
                    if ((unsigned int)(received >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[0] = (unsigned int)received;
            unsigned int expert_1_1 = 32 + lane_read_0;
            unsigned long long received_2 = 0;
            if (expert_1_1 < 384) {
                while (1) {
                    received_2 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_1_1];
                    if ((unsigned int)(received_2 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[1] = (unsigned int)received_2;
            unsigned int expert_3_1 = 64 + lane_read_0;
            unsigned long long received_4 = 0;
            if (expert_3_1 < 384) {
                while (1) {
                    received_4 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_3_1];
                    if ((unsigned int)(received_4 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[2] = (unsigned int)received_4;
            unsigned int expert_5 = 96 + lane_read_0;
            unsigned long long received_6 = 0;
            if (expert_5 < 384) {
                while (1) {
                    received_6 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_5];
                    if ((unsigned int)(received_6 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[3] = (unsigned int)received_6;
            unsigned int expert_7 = 128 + lane_read_0;
            unsigned long long received_8 = 0;
            if (expert_7 < 384) {
                while (1) {
                    received_8 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_7];
                    if ((unsigned int)(received_8 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[4] = (unsigned int)received_8;
            unsigned int expert_9 = 160 + lane_read_0;
            unsigned long long received_10 = 0;
            if (expert_9 < 384) {
                while (1) {
                    received_10 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_9];
                    if ((unsigned int)(received_10 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[5] = (unsigned int)received_10;
            unsigned int expert_11 = 192 + lane_read_0;
            unsigned long long received_12 = 0;
            if (expert_11 < 384) {
                while (1) {
                    received_12 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_11];
                    if ((unsigned int)(received_12 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[6] = (unsigned int)received_12;
            unsigned int expert_13 = 224 + lane_read_0;
            unsigned long long received_14 = 0;
            if (expert_13 < 384) {
                while (1) {
                    received_14 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_13];
                    if ((unsigned int)(received_14 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[7] = (unsigned int)received_14;
            unsigned int expert_15 = 256 + lane_read_0;
            unsigned long long received_16 = 0;
            if (expert_15 < 384) {
                while (1) {
                    received_16 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_15];
                    if ((unsigned int)(received_16 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[8] = (unsigned int)received_16;
            unsigned int expert_17 = 288 + lane_read_0;
            unsigned long long received_18 = 0;
            if (expert_17 < 384) {
                while (1) {
                    received_18 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_17];
                    if ((unsigned int)(received_18 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[9] = (unsigned int)received_18;
            unsigned int expert_19 = 320 + lane_read_0;
            unsigned long long received_20 = 0;
            if (expert_19 < 384) {
                while (1) {
                    received_20 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_19];
                    if ((unsigned int)(received_20 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[10] = (unsigned int)received_20;
            unsigned int expert_21 = 352 + lane_read_0;
            unsigned long long received_22 = 0;
            if (expert_21 < 384) {
                while (1) {
                    received_22 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_21];
                    if ((unsigned int)(received_22 >> 32) == 152) {
                        break;
                    }
                }
            }
            counts[11] = (unsigned int)received_22;
            __syncwarp();
            unsigned int num_blocks = 0;
            if (lane < 384) {
                num_blocks += (counts[0] + 16 - 1) / 16;
            }
            if (32 + lane < 384) {
                num_blocks += (counts[1] + 16 - 1) / 16;
            }
            if (64 + lane < 384) {
                num_blocks += (counts[2] + 16 - 1) / 16;
            }
            if (96 + lane < 384) {
                num_blocks += (counts[3] + 16 - 1) / 16;
            }
            if (128 + lane < 384) {
                num_blocks += (counts[4] + 16 - 1) / 16;
            }
            if (160 + lane < 384) {
                num_blocks += (counts[5] + 16 - 1) / 16;
            }
            if (192 + lane < 384) {
                num_blocks += (counts[6] + 16 - 1) / 16;
            }
            if (224 + lane < 384) {
                num_blocks += (counts[7] + 16 - 1) / 16;
            }
            if (256 + lane < 384) {
                num_blocks += (counts[8] + 16 - 1) / 16;
            }
            if (288 + lane < 384) {
                num_blocks += (counts[9] + 16 - 1) / 16;
            }
            if (320 + lane < 384) {
                num_blocks += (counts[10] + 16 - 1) / 16;
            }
            if (352 + lane < 384) {
                num_blocks += (counts[11] + 16 - 1) / 16;
            }
            unsigned int _warp_redux_u32_0;
            asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(num_blocks));
            unsigned int total = _warp_redux_u32_0;
            unsigned int waves = (total * 18 + 76 - 1) / 76;
            unsigned int interleave_waves = 2;
            state[0] = total;
            int _max_0 = ((1) > (interleave_waves) ? (1) : (interleave_waves));
            int _min_0 = ((_max_0) < (waves) ? (_max_0) : (waves));
            state[1] = _min_0;
            #pragma unroll 1
            for (int token = (unsigned int)(bid * 4) + warp; token < num_tokens * 6; token += 608) {
                #pragma unroll 1
                for (int advance_expert = 0; advance_expert < 385; advance_expert++) {
                    if (expert_end > (unsigned int)token) {
                        break;
                    }
                    current_expert += 1;
                    if (current_expert >= 384) {
                        break;
                    }
                    pool_offset += (expert_end - expert_start + 16 - 1) / 16;
                    expert_start = expert_end;
                    unsigned int selected = 0;
                    if ((unsigned int)current_expert == lane) {
                        selected = counts[0];
                    }
                    if ((unsigned int)current_expert == 32 + lane) {
                        selected = counts[1];
                    }
                    if ((unsigned int)current_expert == 64 + lane) {
                        selected = counts[2];
                    }
                    if ((unsigned int)current_expert == 96 + lane) {
                        selected = counts[3];
                    }
                    if ((unsigned int)current_expert == 128 + lane) {
                        selected = counts[4];
                    }
                    if ((unsigned int)current_expert == 160 + lane) {
                        selected = counts[5];
                    }
                    if ((unsigned int)current_expert == 192 + lane) {
                        selected = counts[6];
                    }
                    if ((unsigned int)current_expert == 224 + lane) {
                        selected = counts[7];
                    }
                    if ((unsigned int)current_expert == 256 + lane) {
                        selected = counts[8];
                    }
                    if ((unsigned int)current_expert == 288 + lane) {
                        selected = counts[9];
                    }
                    if ((unsigned int)current_expert == 320 + lane) {
                        selected = counts[10];
                    }
                    if ((unsigned int)current_expert == 352 + lane) {
                        selected = counts[11];
                    }
                    unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, selected, (unsigned int)current_expert % 32);
                    expert_end += _shfl_0;
                }
                if (current_expert >= 384) {
                    break;
                }
                unsigned int in_expert = (unsigned int)token - expert_start;
                unsigned int source_topk = SourceIndices[(unsigned int)(current_expert * 1920) + in_expert];
                unsigned int source_token = source_topk / 6;
                unsigned int pool_token = pool_offset * 16 + in_expert;
                unsigned int pool_block = pool_token / 16;
                unsigned int ring_block = pool_block % 960;
                unsigned int ring_token = pool_token % 15360;
                unsigned int target_1 = pool_block / 960 * 36;
                if (target_1 > 0) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(L1Empty) + (ring_block))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(target_1)) >= static_cast<unsigned int>((unsigned int)0 - target_1));
                    }
                }
                unsigned long long source_byte = (unsigned long long)source_token * 5120;
                unsigned long long destination_byte = (unsigned long long)ring_token * 5120;
                if (elect_sync()) {
                    #pragma unroll
                    for (int chunk = 0; chunk < 1; chunk++) {
                        cp_async_bulk_gmem2smem(send_buffer_addr + warp * 5120, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(X) + ((unsigned long long)(source_byte + (unsigned long long)(chunk * 5120)) * (unsigned long long)1)), 5120, pull_barriers_addr + (warp) * 8);
                        mbarrier_arrive_expect_tx(pull_barriers_addr + (warp) * 8, 5120);
                        if (chunk != 0) {
                            mbarrier_wait(pull_barriers_addr + (warp) * 8, pull_phase);
                            pull_phase ^= 1;
                            {
                                void* _cpbulk_dst_0 = reinterpret_cast<void*>(L1Acts + (destination_byte + (unsigned long long)(chunk * 5120)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_0), "r"(send_buffer_addr + warp * 5120), "r"((uint32_t)(5120))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group 0;");
                        }
                    }
                }
                __syncwarp();
                float weight = Weights[source_topk];
                unsigned int in_block = in_expert % 16;
                unsigned int sf_token = ring_block * 128 + (in_block & 4294967168) + (in_block & 31) * 4 + (in_block >> 5 & 3);
                #pragma unroll
                for (int sf_group = 0; sf_group < 2; sf_group++) {
                    unsigned int sf_word = (unsigned int)(sf_group * 32) + lane;
                    if (sf_word < 40) {
                        L1SF[sf_word * 245760 + sf_token] = XSF[source_token * 40 + sf_word];
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    L1Weights[ring_token] = weight;
                    TokenMetadata[pool_token * 3] = 0;
                    TokenMetadata[pool_token * 3 + 1] = source_token;
                    TokenMetadata[pool_token * 3 + 2] = source_topk % 6;
                    mbarrier_wait(pull_barriers_addr + (warp) * 8, pull_phase);
                    pull_phase ^= 1;
                    {
                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(L1Acts + destination_byte);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_1), "r"(send_buffer_addr + warp * 5120), "r"((uint32_t)(5120))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    unsigned int arrivals = (((unsigned int)token == expert_end - 1) ? 16 - in_block : (unsigned int)1);
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(L1Full) + (ring_block))), "r"(static_cast<unsigned int>(arrivals)) : "memory");
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (bid == 0) {
                #pragma unroll
                for (int expert_6 = tid; expert_6 < 384; expert_6 += 128) {
                    SendCounts[expert_6] = 0;
                }
                if (tid == 0) {
                    L1Counter[0] = 0;
                    L2Counter[0] = 0;
                    SharedL1Counter[0] = 0;
                    SharedL2Counter[0] = 0;
                }
                __syncwarp();
                #pragma unroll
                for (int block = tid; block < 240; block += 128) {
                    SharedFull[block] = 0;
                }
                __syncwarp();
            } else {
                #pragma unroll 1
                for (int expert_8 = bid - 1; expert_8 < 384; expert_8 += 151) {
                    unsigned int num_received = (unsigned int)RecvSum[expert_8];
                    unsigned int blocks = (num_received + 16 - 1) / 16;
                    unsigned int num_blocks_0 = 0;
                    if (lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[0] + 16 - 1) / 16;
                    }
                    if (32 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[1] + 16 - 1) / 16;
                    }
                    if (64 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[2] + 16 - 1) / 16;
                    }
                    if (96 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[3] + 16 - 1) / 16;
                    }
                    if (128 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[4] + 16 - 1) / 16;
                    }
                    if (160 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[5] + 16 - 1) / 16;
                    }
                    if (192 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[6] + 16 - 1) / 16;
                    }
                    if (224 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[7] + 16 - 1) / 16;
                    }
                    if (256 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[8] + 16 - 1) / 16;
                    }
                    if (288 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[9] + 16 - 1) / 16;
                    }
                    if (320 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[10] + 16 - 1) / 16;
                    }
                    if (352 + lane < (unsigned int)expert_8) {
                        num_blocks_0 += (counts[11] + 16 - 1) / 16;
                    }
                    unsigned int _warp_redux_u32_1;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(num_blocks_0));
                    unsigned int offset = _warp_redux_u32_1;
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    if (warp == 0) {
                        RecvSum[expert_8] = 0;
                    }
                    if (tid == 0) {
                        RecvCounts[expert_8] = 0;
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (int block_1 = tid; block_1 < blocks; block_1 += 128) {
                        unsigned int ring = (offset + (unsigned int)block_1) % 960;
                        L1Full[ring] = 0;
                        L1Empty[ring] = 0;
                        L2Mask[ring] = 0;
                        L2Empty[ring] = 0;
                    }
                    __syncwarp();
                }
            }
        }
    // ---- Role: load_a ----
    } else if (warp == 4) {
        { // load_a_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int task[8];
            unsigned int a_stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (int task_iteration = 0; task_iteration < 250801; task_iteration++) {
                mbarrier_wait(task_full_addr + (task_iteration % 2) * 8, task_iteration / 2 & 1);
                task[0] = task_infos[task_iteration % 2 * 8];
                task[1] = task_infos[task_iteration % 2 * 8 + 1];
                task[2] = task_infos[task_iteration % 2 * 8 + 2];
                task[3] = task_infos[task_iteration % 2 * 8 + 3];
                task[4] = task_infos[task_iteration % 2 * 8 + 4];
                task[5] = task_infos[task_iteration % 2 * 8 + 5];
                task[6] = task_infos[task_iteration % 2 * 8 + 6];
                task[7] = task_infos[task_iteration % 2 * 8 + 7];
                if (task[0] == 0) {
                    break;
                }
                const void* selected_activation_map = ((task[0] == 1) ? A1 : ((task[0] == 2) ? A2 : ((task[0] == 3) ? SA1 : SA2)));
                const void* selected_activation_sf_map = ((task[0] == 1) ? SFA1 : ((task[0] == 2) ? SFA2 : ((task[0] == 3) ? SSFA1 : SSFA2)));
                unsigned int pool_block_1 = task[4];
                unsigned int ring_block_1 = pool_block_1 % 960;
                unsigned int block_2 = ring_block_1;
                if (task[0] > 2) {
                    block_2 = pool_block_1;
                }
                if (task[0] == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(L1Full) + (ring_block_1))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(16 * (pool_block_1 / 960 + 1))) >= static_cast<unsigned int>(1));
                    }
                }
                if (task[0] == 4) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(SharedFull) + (block_2))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(36)) >= static_cast<unsigned int>(1));
                    }
                }
                unsigned long long expected = 68719476735;
                if ((pool_block_1 / 960 & 1) != 0) {
                    expected = 0;
                }
                unsigned long long pending = 68719476735;
                unsigned int token_offset = block_2 * 16 + (unsigned int)cta_rank * ((task[5] + 15) / 16 * 8);
                unsigned int sf_offset = block_2 * 128;
                for (int k = 0; k < task[7] / 128; k++) {
                    if (task[0] == 2) {
                        unsigned long long k_mask = (unsigned long long)3 << (unsigned long long)(k * 2);
                        if ((pending & k_mask) != 0) {
                            unsigned long long _wait_acquire_mask_2;
                            do {
                            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_wait_acquire_mask_2) : "l"((reinterpret_cast<unsigned long long*>(L2Mask) + (ring_block_1))) : "memory");
                            } while (((_wait_acquire_mask_2 ^ static_cast<unsigned long long>(expected)) & static_cast<unsigned long long>(k_mask)) != 0);
                            unsigned long long observed = _wait_acquire_mask_2;
                            pending = observed ^ expected;
                        }
                    }
                    mbarrier_wait(empty_addr + (a_stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + a_stage * 1024, selected_activation_map, 0, token_offset, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 512, selected_activation_sf_map, sf_offset, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (a_stage) * 8, 3072);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (a_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    __syncwarp();
                    a_stage += 1;
                    if (a_stage == 11) { a_stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    // ---- Role: load_b ----
    } else if (warp == 5) {
        { // load_b_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int task_1[8];
            unsigned int b_stage = 0;
            unsigned int _phase_empty_1 = 1;
            #pragma unroll 1
            for (int task_iteration_1 = 0; task_iteration_1 < 250801; task_iteration_1++) {
                mbarrier_wait(task_full_addr + (task_iteration_1 % 2) * 8, task_iteration_1 / 2 & 1);
                task_1[0] = task_infos[task_iteration_1 % 2 * 8];
                task_1[1] = task_infos[task_iteration_1 % 2 * 8 + 1];
                task_1[2] = task_infos[task_iteration_1 % 2 * 8 + 2];
                task_1[3] = task_infos[task_iteration_1 % 2 * 8 + 3];
                task_1[4] = task_infos[task_iteration_1 % 2 * 8 + 4];
                task_1[5] = task_infos[task_iteration_1 % 2 * 8 + 5];
                task_1[6] = task_infos[task_iteration_1 % 2 * 8 + 6];
                task_1[7] = task_infos[task_iteration_1 % 2 * 8 + 7];
                if (task_1[0] == 0) {
                    break;
                }
                const void* selected_weight_map = ((task_1[0] == 1) ? B1 : ((task_1[0] == 2) ? B2 : ((task_1[0] == 3) ? SB1 : SB2)));
                const void* selected_weight_sf_map = ((task_1[0] == 1) ? SFB1 : ((task_1[0] == 2) ? SFB2 : ((task_1[0] == 3) ? SSFB1 : SSFB2)));
                unsigned int n_offset = (task_1[3] * 2 + (unsigned int)cta_rank) * 128;
                unsigned int row_offset = n_offset;
                unsigned int sf_k_offset = 0;
                if (task_1[0] <= 2) {
                    row_offset += task_1[1] * task_1[6];
                    sf_k_offset = task_1[1] * (task_1[7] / 128);
                }
                #pragma unroll 2
                for (int k_1 = 0; k_1 < task_1[7] / 128; k_1++) {
                    mbarrier_wait(empty_addr + (b_stage) * 8, _phase_empty_1);
                    if (elect_sync()) {
                        if (task_1[0] > 2) {
                            tma_3d_gmem2smem_cta2(smem_b_addr + b_stage * 16384, selected_weight_map, 0, row_offset, k_1, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                            tma_2d_gmem2smem_cta2(smem_sfb_addr + b_stage * 512, selected_weight_sf_map, n_offset, k_1, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                            if (cta_rank == 0) {
                                mbarrier_arrive_expect_tx(full_addr + (b_stage) * 8, 33792);
                            } else {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((full_addr + (b_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_b_addr + b_stage * 16384), "l"(selected_weight_map), "r"(0), "r"(row_offset), "r"(k_1),
                                   "r"(((full_addr + (b_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_sfb_addr + b_stage * 512), "l"(selected_weight_sf_map), "r"(n_offset), "r"(sf_k_offset + (unsigned int)k_1),
                                   "r"(((full_addr + (b_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            if (cta_rank == 0) {
                                mbarrier_arrive_expect_tx(full_addr + (b_stage) * 8, 17408);
                            } else {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((full_addr + (b_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                        }
                    }
                    __syncwarp();
                    b_stage += 1;
                    if (b_stage == 11) { b_stage = 0; _phase_empty_1 ^= 1; }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 6) {
        { // mma_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int task_2[8];
            unsigned int mma_stage = 0;
            unsigned int completed_tasks = 0;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                unsigned int persistent_routed_mma_idesc = 277086848;
                unsigned int persistent_shared_mma_idesc = 277086208;
                #pragma unroll 1
                for (int task_iteration_2 = 0; task_iteration_2 < 250801; task_iteration_2++) {
                    mbarrier_wait(task_full_addr + (task_iteration_2 % 2) * 8, task_iteration_2 / 2 & 1);
                    task_2[0] = task_infos[task_iteration_2 % 2 * 8];
                    task_2[1] = task_infos[task_iteration_2 % 2 * 8 + 1];
                    task_2[2] = task_infos[task_iteration_2 % 2 * 8 + 2];
                    task_2[3] = task_infos[task_iteration_2 % 2 * 8 + 3];
                    task_2[4] = task_infos[task_iteration_2 % 2 * 8 + 4];
                    task_2[5] = task_infos[task_iteration_2 % 2 * 8 + 5];
                    task_2[6] = task_infos[task_iteration_2 % 2 * 8 + 6];
                    task_2[7] = task_infos[task_iteration_2 % 2 * 8 + 7];
                    if (task_2[0] == 0) {
                        break;
                    }
                    unsigned int accum_stage = task_iteration_2 % 2;
                    unsigned int aligned_m = 16;
                    if (task_2[0] > 2) {
                        persistent_shared_mma_idesc = persistent_shared_mma_idesc & (unsigned int)4286709759 | (aligned_m >> 3 & 63) << 17;
                    } else {
                        persistent_routed_mma_idesc = persistent_routed_mma_idesc & (unsigned int)4286709759 | (aligned_m >> 3 & 63) << 17;
                    }
                    unsigned int selected_mma_idesc = ((task_2[0] > 2) ? persistent_shared_mma_idesc : persistent_routed_mma_idesc);
                    mbarrier_wait_hint(tmem_empty_addr + (accum_stage) * 8, task_iteration_2 / 2 & 1 ^ 1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 2
                    for (int k_2 = 0; k_2 < task_2[7] / 128; k_2++) {
                        mbarrier_wait(full_addr + (mma_stage) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((k_2 == 0) ? 1 : 0);
                        { // Pre-election register operand materialization
                            uint32_t _election_operand_0_0 = (tmem_scale_a);
                            uint64_t _election_operand_0_1 = (make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 32)));
                            asm volatile("" :: "r"(_election_operand_0_0), "l"(_election_operand_0_1));
                            uint32_t _election_operand_1_0 = (tmem_scale_b);
                            uint64_t _election_operand_1_1 = (make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_stage) * 32)));
                            asm volatile("" :: "r"(_election_operand_1_0), "l"(_election_operand_1_1));
                            int _mma_a_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024;
                            int _mma_b_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 64;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            uint32_t _election_operand_2_0 = ((tmem_accum + (accum_stage * 16)));
                            uint64_t _election_operand_2_1 = (a_desc + 0);
                            uint64_t _election_operand_2_2 = (b_desc + 0);
                            uint32_t _election_operand_2_3 = ((selected_mma_idesc | ((0) << 29) | ((0) << 4)));
                            uint32_t _election_operand_2_4 = (tmem_scale_b);
                            uint32_t _election_operand_2_5 = (tmem_scale_a);
                            uint32_t _election_operand_2_6 = (((init_flag) ? 0 : 1));
                            asm volatile("" :: "r"(_election_operand_2_0), "l"(_election_operand_2_1), "l"(_election_operand_2_2), "r"(_election_operand_2_3), "r"(_election_operand_2_4), "r"(_election_operand_2_5), "r"(_election_operand_2_6));
                            uint32_t _election_operand_3_0 = ((tmem_accum + (accum_stage * 16)));
                            uint64_t _election_operand_3_1 = (a_desc + 2);
                            uint64_t _election_operand_3_2 = (b_desc + 2);
                            uint32_t _election_operand_3_3 = ((selected_mma_idesc | ((1) << 29) | ((1) << 4)));
                            uint32_t _election_operand_3_4 = (tmem_scale_b);
                            uint32_t _election_operand_3_5 = (tmem_scale_a);
                            uint32_t _election_operand_3_6 = (1);
                            asm volatile("" :: "r"(_election_operand_3_0), "l"(_election_operand_3_1), "l"(_election_operand_3_2), "r"(_election_operand_3_3), "r"(_election_operand_3_4), "r"(_election_operand_3_5), "r"(_election_operand_3_6));
                            uint32_t _election_operand_4_0 = ((tmem_accum + (accum_stage * 16)));
                            uint64_t _election_operand_4_1 = (a_desc + 4);
                            uint64_t _election_operand_4_2 = (b_desc + 4);
                            uint32_t _election_operand_4_3 = ((selected_mma_idesc | ((2) << 29) | ((2) << 4)));
                            uint32_t _election_operand_4_4 = (tmem_scale_b);
                            uint32_t _election_operand_4_5 = (tmem_scale_a);
                            uint32_t _election_operand_4_6 = (1);
                            asm volatile("" :: "r"(_election_operand_4_0), "l"(_election_operand_4_1), "l"(_election_operand_4_2), "r"(_election_operand_4_3), "r"(_election_operand_4_4), "r"(_election_operand_4_5), "r"(_election_operand_4_6));
                            uint32_t _election_operand_5_0 = ((tmem_accum + (accum_stage * 16)));
                            uint64_t _election_operand_5_1 = (a_desc + 6);
                            uint64_t _election_operand_5_2 = (b_desc + 6);
                            uint32_t _election_operand_5_3 = ((selected_mma_idesc | ((3) << 29) | ((3) << 4)));
                            uint32_t _election_operand_5_4 = (tmem_scale_b);
                            uint32_t _election_operand_5_5 = (tmem_scale_a);
                            uint32_t _election_operand_5_6 = (1);
                            asm volatile("" :: "r"(_election_operand_5_0), "l"(_election_operand_5_1), "l"(_election_operand_5_2), "r"(_election_operand_5_3), "r"(_election_operand_5_4), "r"(_election_operand_5_5), "r"(_election_operand_5_6));
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(_election_operand_0_0, _election_operand_0_1);
                                tcgen05_cp_32x128b_warpx4_cta2(_election_operand_1_0, _election_operand_1_1);
                                tcgen05_mma_mxf8_bs_cta2(_election_operand_2_0, _election_operand_2_1, _election_operand_2_2,
                                    _election_operand_2_3, _election_operand_2_4, _election_operand_2_5, _election_operand_2_6);
                                tcgen05_mma_mxf8_bs_cta2(_election_operand_3_0, _election_operand_3_1, _election_operand_3_2,
                                    _election_operand_3_3, _election_operand_3_4, _election_operand_3_5, _election_operand_3_6);
                                tcgen05_mma_mxf8_bs_cta2(_election_operand_4_0, _election_operand_4_1, _election_operand_4_2,
                                    _election_operand_4_3, _election_operand_4_4, _election_operand_4_5, _election_operand_4_6);
                                tcgen05_mma_mxf8_bs_cta2(_election_operand_5_0, _election_operand_5_1, _election_operand_5_2,
                                    _election_operand_5_3, _election_operand_5_4, _election_operand_5_5, _election_operand_5_6);
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        if ((unsigned int)k_2 == task_2[7] / 128 - 1) {
                            elect_commit_cg2_multicast(tmem_full_addr + (accum_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        mma_stage += 1;
                        if (mma_stage == 11) { mma_stage = 0; _phase_full ^= 1; }
                    }
                    completed_tasks += 1;
                }
                if (completed_tasks > 0) {
                    mbarrier_wait_hint(tmem_empty_addr + ((completed_tasks - 1) % 2) * 8, (completed_tasks - 1) / 2 & 1, 10000000);
                }
            }
        }
    // ---- Role: schedule ----
    } else if (warp == 7) {
        { // schedule_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int counts_1[12];
            unsigned int state_1[2];
            unsigned int task_3[8];
            unsigned int schedule_iteration[1];
            schedule_iteration[0] = 0;
            unsigned int early_shared_l2 = (unsigned int)((num_tokens + 16 - 1) / 16 * 20 <= 76);
            if (cta_rank == 0) {
                unsigned int total_1 = (num_tokens + 16 - 1) / 16 * 18;
                #pragma unroll 1
                for (int iteration = 0; iteration < 250801; iteration++) {
                    mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                    unsigned int claimed = 0;
                    if (lane == 0) {
                        unsigned int _atomic_old_3 = atomicAdd(SharedL1Counter, 1);
                        claimed = _atomic_old_3;
                    }
                    unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, claimed, 0);
                    unsigned int claimed_0 = _shfl_1;
                    if (claimed_0 >= total_1) {
                        break;
                    }
                    unsigned int block_3 = claimed_0 / 18;
                    task_3[0] = 3;
                    task_3[1] = 0;
                    task_3[2] = block_3;
                    task_3[3] = claimed_0 % 18;
                    task_3[4] = block_3;
                    unsigned int _min_1 = ((num_tokens - block_3 * 16) < (16) ? (num_tokens - block_3 * 16) : (16));
                    task_3[5] = _min_1;
                    task_3[6] = 4608;
                    task_3[7] = 5120;
                    if (lane < 2) {
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(task_infos_addr + schedule_iteration[0] % 2 * 32), "r"(lane));
                        unsigned int destination = _mapa_0;
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(task_full_addr + schedule_iteration[0] % 2 * 8), "r"(lane));
                        unsigned int barrier = _mapa_1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"(barrier), "r"((uint32_t)(32)) : "memory");
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(destination), "r"(task_3[0]), "r"(task_3[1]), "r"(task_3[2]), "r"(task_3[3]), "r"(barrier) : "memory");
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(destination + 16), "r"(task_3[4]), "r"(task_3[5]), "r"(task_3[6]), "r"(task_3[7]), "r"(barrier) : "memory");
                    }
                    __syncwarp();
                    schedule_iteration[0] = schedule_iteration[0] + 1;
                }
                if (early_shared_l2 != 0) {
                    unsigned int total_0 = (num_tokens + 16 - 1) / 16 * 20;
                    #pragma unroll 1
                    for (int iteration_1 = 0; iteration_1 < 250801; iteration_1++) {
                        mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                        unsigned int claimed_1 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_4 = atomicAdd(SharedL2Counter, 1);
                            claimed_1 = _atomic_old_4;
                        }
                        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, claimed_1, 0);
                        unsigned int claimed_0_1 = _shfl_2;
                        if (claimed_0_1 >= total_0) {
                            break;
                        }
                        unsigned int block_4 = claimed_0_1 / 20;
                        task_3[0] = 4;
                        task_3[1] = 0;
                        task_3[2] = block_4;
                        task_3[3] = claimed_0_1 % 20;
                        task_3[4] = block_4;
                        unsigned int _min_2 = ((num_tokens - block_4 * 16) < (16) ? (num_tokens - block_4 * 16) : (16));
                        task_3[5] = _min_2;
                        task_3[6] = 5120;
                        task_3[7] = 2304;
                        if (lane < 2) {
                            uint32_t _mapa_2;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_2) : "r"(task_infos_addr + schedule_iteration[0] % 2 * 32), "r"(lane));
                            unsigned int destination_1 = _mapa_2;
                            uint32_t _mapa_3;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_3) : "r"(task_full_addr + schedule_iteration[0] % 2 * 8), "r"(lane));
                            unsigned int barrier_1 = _mapa_3;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"(barrier_1), "r"((uint32_t)(32)) : "memory");
                            asm volatile(
                                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                :: "r"(destination_1), "r"(task_3[0]), "r"(task_3[1]), "r"(task_3[2]), "r"(task_3[3]), "r"(barrier_1) : "memory");
                            asm volatile(
                                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                :: "r"(destination_1 + 16), "r"(task_3[4]), "r"(task_3[5]), "r"(task_3[6]), "r"(task_3[7]), "r"(barrier_1) : "memory");
                        }
                        __syncwarp();
                        schedule_iteration[0] = schedule_iteration[0] + 1;
                    }
                }
                uint32_t lane_read_0_1;
                asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_read_0_1));
                unsigned int expert_10 = lane_read_0_1;
                unsigned long long received_1 = 0;
                if (expert_10 < 384) {
                    while (1) {
                        received_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_10];
                        if ((unsigned int)(received_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[0] = (unsigned int)received_1;
                unsigned int expert_1_2 = 32 + lane_read_0_1;
                unsigned long long received_2_1 = 0;
                if (expert_1_2 < 384) {
                    while (1) {
                        received_2_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_1_2];
                        if ((unsigned int)(received_2_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[1] = (unsigned int)received_2_1;
                unsigned int expert_3_2 = 64 + lane_read_0_1;
                unsigned long long received_4_1 = 0;
                if (expert_3_2 < 384) {
                    while (1) {
                        received_4_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_3_2];
                        if ((unsigned int)(received_4_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[2] = (unsigned int)received_4_1;
                unsigned int expert_5_1 = 96 + lane_read_0_1;
                unsigned long long received_6_1 = 0;
                if (expert_5_1 < 384) {
                    while (1) {
                        received_6_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_5_1];
                        if ((unsigned int)(received_6_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[3] = (unsigned int)received_6_1;
                unsigned int expert_7_1 = 128 + lane_read_0_1;
                unsigned long long received_8_1 = 0;
                if (expert_7_1 < 384) {
                    while (1) {
                        received_8_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_7_1];
                        if ((unsigned int)(received_8_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[4] = (unsigned int)received_8_1;
                unsigned int expert_9_1 = 160 + lane_read_0_1;
                unsigned long long received_10_1 = 0;
                if (expert_9_1 < 384) {
                    while (1) {
                        received_10_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_9_1];
                        if ((unsigned int)(received_10_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[5] = (unsigned int)received_10_1;
                unsigned int expert_11_1 = 192 + lane_read_0_1;
                unsigned long long received_12_1 = 0;
                if (expert_11_1 < 384) {
                    while (1) {
                        received_12_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_11_1];
                        if ((unsigned int)(received_12_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[6] = (unsigned int)received_12_1;
                unsigned int expert_13_1 = 224 + lane_read_0_1;
                unsigned long long received_14_1 = 0;
                if (expert_13_1 < 384) {
                    while (1) {
                        received_14_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_13_1];
                        if ((unsigned int)(received_14_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[7] = (unsigned int)received_14_1;
                unsigned int expert_15_1 = 256 + lane_read_0_1;
                unsigned long long received_16_1 = 0;
                if (expert_15_1 < 384) {
                    while (1) {
                        received_16_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_15_1];
                        if ((unsigned int)(received_16_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[8] = (unsigned int)received_16_1;
                unsigned int expert_17_1 = 288 + lane_read_0_1;
                unsigned long long received_18_1 = 0;
                if (expert_17_1 < 384) {
                    while (1) {
                        received_18_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_17_1];
                        if ((unsigned int)(received_18_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[9] = (unsigned int)received_18_1;
                unsigned int expert_19_1 = 320 + lane_read_0_1;
                unsigned long long received_20_1 = 0;
                if (expert_19_1 < 384) {
                    while (1) {
                        received_20_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_19_1];
                        if ((unsigned int)(received_20_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[10] = (unsigned int)received_20_1;
                unsigned int expert_21_1 = 352 + lane_read_0_1;
                unsigned long long received_22_1 = 0;
                if (expert_21_1 < 384) {
                    while (1) {
                        received_22_1 = reinterpret_cast<volatile unsigned long long*>(RecvSum)[(unsigned int)expert_21_1];
                        if ((unsigned int)(received_22_1 >> 32) == 152) {
                            break;
                        }
                    }
                }
                counts_1[11] = (unsigned int)received_22_1;
                __syncwarp();
                unsigned int num_blocks_1 = 0;
                if (lane < 384) {
                    num_blocks_1 += (counts_1[0] + 16 - 1) / 16;
                }
                if (32 + lane < 384) {
                    num_blocks_1 += (counts_1[1] + 16 - 1) / 16;
                }
                if (64 + lane < 384) {
                    num_blocks_1 += (counts_1[2] + 16 - 1) / 16;
                }
                if (96 + lane < 384) {
                    num_blocks_1 += (counts_1[3] + 16 - 1) / 16;
                }
                if (128 + lane < 384) {
                    num_blocks_1 += (counts_1[4] + 16 - 1) / 16;
                }
                if (160 + lane < 384) {
                    num_blocks_1 += (counts_1[5] + 16 - 1) / 16;
                }
                if (192 + lane < 384) {
                    num_blocks_1 += (counts_1[6] + 16 - 1) / 16;
                }
                if (224 + lane < 384) {
                    num_blocks_1 += (counts_1[7] + 16 - 1) / 16;
                }
                if (256 + lane < 384) {
                    num_blocks_1 += (counts_1[8] + 16 - 1) / 16;
                }
                if (288 + lane < 384) {
                    num_blocks_1 += (counts_1[9] + 16 - 1) / 16;
                }
                if (320 + lane < 384) {
                    num_blocks_1 += (counts_1[10] + 16 - 1) / 16;
                }
                if (352 + lane < 384) {
                    num_blocks_1 += (counts_1[11] + 16 - 1) / 16;
                }
                unsigned int _warp_redux_u32_2;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(num_blocks_1));
                unsigned int total_23 = _warp_redux_u32_2;
                unsigned int waves_1 = (total_23 * 18 + 76 - 1) / 76;
                unsigned int interleave_waves_1 = 2;
                state_1[0] = total_23;
                int _max_1 = ((1) > (interleave_waves_1) ? (1) : (interleave_waves_1));
                int _min_3 = ((_max_1) < (waves_1) ? (_max_1) : (waves_1));
                state_1[1] = _min_3;
                #pragma unroll 1
                for (int iteration_2 = 0; iteration_2 < 250801; iteration_2++) {
                    mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                    unsigned int phase_1 = 0;
                    unsigned int claimed_2 = 0;
                    unsigned int clusters = 18;
                    unsigned int shape_n = 4608;
                    unsigned int shape_k = 5120;
                    if (state_1[1] != 4294967295 && state_1[1] != 0) {
                        state_1[1] = state_1[1] - 1;
                        unsigned int claimed_0_2 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_5 = atomicAdd(L1Counter, 1);
                            claimed_0_2 = _atomic_old_5;
                        }
                        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, claimed_0_2, 0);
                        claimed_2 = _shfl_3;
                        if (claimed_2 >= state_1[0] * 18) {
                            state_1[1] = 4294967295;
                        } else {
                            phase_1 = 1;
                        }
                    }
                    if (phase_1 == 0) {
                        unsigned int claimed_0_3 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_6 = atomicAdd(L2Counter, 1);
                            claimed_0_3 = _atomic_old_6;
                        }
                        unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, claimed_0_3, 0);
                        claimed_2 = _shfl_4;
                        if (claimed_2 < state_1[0] * 20) {
                            if (state_1[1] != 4294967295) {
                                state_1[1] = 1;
                            }
                            phase_1 = 2;
                            clusters = 20;
                            shape_n = 5120;
                            shape_k = 2304;
                        }
                    }
                    task_3[0] = 0;
                    task_3[1] = 0;
                    task_3[2] = 0;
                    task_3[3] = 0;
                    task_3[4] = 0;
                    task_3[5] = 0;
                    task_3[6] = 0;
                    task_3[7] = 0;
                    if (phase_1 != 0) {
                        unsigned int pool_block_2 = claimed_2 / clusters;
                        task_3[0] = phase_1;
                        task_3[1] = 0;
                        task_3[2] = 0;
                        task_3[3] = claimed_2 % clusters;
                        task_3[4] = pool_block_2;
                        task_3[5] = 0;
                        task_3[6] = shape_n;
                        task_3[7] = shape_k;
                        unsigned int block_offset = 0;
                        unsigned int expert_0 = lane;
                        unsigned int tokens = counts_1[0];
                        unsigned int blocks_1 = (tokens + 16 - 1) / 16;
                        unsigned int inclusive = blocks_1;
                        unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
                        unsigned int previous = _shfl_up_0;
                        if (lane >= 1) {
                            inclusive += previous;
                        }
                        unsigned int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
                        unsigned int previous_1 = _shfl_up_1;
                        if (lane >= 2) {
                            inclusive += previous_1;
                        }
                        unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
                        unsigned int previous_2 = _shfl_up_2;
                        if (lane >= 4) {
                            inclusive += previous_2;
                        }
                        unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
                        unsigned int previous_3 = _shfl_up_3;
                        if (lane >= 8) {
                            inclusive += previous_3;
                        }
                        unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
                        unsigned int previous_4 = _shfl_up_4;
                        if (lane >= 16) {
                            inclusive += previous_4;
                        }
                        unsigned int lane_offset = block_offset + inclusive - blocks_1;
                        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, expert_0 < 384 && (pool_block_2 >= lane_offset && pool_block_2 < lane_offset + blocks_1));
                        unsigned int owner_mask = _vote_0;
                        if (owner_mask != 0) {
                            int _ffs_0 = __ffs(owner_mask);
                            unsigned int owner_lane = _ffs_0 - 1;
                            unsigned int local_block = pool_block_2 - lane_offset;
                            unsigned int _min_4 = ((tokens - local_block * 16) < (16) ? (tokens - local_block * 16) : (16));
                            unsigned int valid = _min_4;
                            unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, expert_0, owner_lane);
                            task_3[1] = _shfl_5;
                            unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, local_block, owner_lane);
                            task_3[2] = _shfl_6;
                            unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, valid, owner_lane);
                            task_3[5] = _shfl_7;
                        }
                        unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
                        block_offset += _shfl_8;
                        unsigned int expert_6_1 = 32 + lane;
                        unsigned int tokens_7 = counts_1[1];
                        unsigned int blocks_8 = (tokens_7 + 16 - 1) / 16;
                        unsigned int inclusive_9 = blocks_8;
                        unsigned int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, inclusive_9, 1, 32);
                        unsigned int previous_10 = _shfl_up_5;
                        if (lane >= 1) {
                            inclusive_9 += previous_10;
                        }
                        unsigned int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, inclusive_9, 2, 32);
                        unsigned int previous_11 = _shfl_up_6;
                        if (lane >= 2) {
                            inclusive_9 += previous_11;
                        }
                        unsigned int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, inclusive_9, 4, 32);
                        unsigned int previous_12 = _shfl_up_7;
                        if (lane >= 4) {
                            inclusive_9 += previous_12;
                        }
                        unsigned int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, inclusive_9, 8, 32);
                        unsigned int previous_13 = _shfl_up_8;
                        if (lane >= 8) {
                            inclusive_9 += previous_13;
                        }
                        unsigned int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, inclusive_9, 16, 32);
                        unsigned int previous_14 = _shfl_up_9;
                        if (lane >= 16) {
                            inclusive_9 += previous_14;
                        }
                        unsigned int lane_offset_15 = block_offset + inclusive_9 - blocks_8;
                        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, expert_6_1 < 384 && (pool_block_2 >= lane_offset_15 && pool_block_2 < lane_offset_15 + blocks_8));
                        unsigned int owner_mask_16 = _vote_1;
                        if (owner_mask_16 != 0) {
                            int _ffs_1 = __ffs(owner_mask_16);
                            unsigned int owner_lane_1 = _ffs_1 - 1;
                            unsigned int local_block_1 = pool_block_2 - lane_offset_15;
                            unsigned int _min_5 = ((tokens_7 - local_block_1 * 16) < (16) ? (tokens_7 - local_block_1 * 16) : (16));
                            unsigned int valid_1 = _min_5;
                            unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, expert_6_1, owner_lane_1);
                            task_3[1] = _shfl_9;
                            unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, local_block_1, owner_lane_1);
                            task_3[2] = _shfl_10;
                            unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, valid_1, owner_lane_1);
                            task_3[5] = _shfl_11;
                        }
                        unsigned int _shfl_12 = __shfl_sync(0xFFFFFFFF, inclusive_9, 31);
                        block_offset += _shfl_12;
                        unsigned int expert_18 = 64 + lane;
                        unsigned int tokens_19 = counts_1[2];
                        unsigned int blocks_20 = (tokens_19 + 16 - 1) / 16;
                        unsigned int inclusive_21 = blocks_20;
                        unsigned int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, inclusive_21, 1, 32);
                        unsigned int previous_22 = _shfl_up_10;
                        if (lane >= 1) {
                            inclusive_21 += previous_22;
                        }
                        unsigned int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, inclusive_21, 2, 32);
                        unsigned int previous_23 = _shfl_up_11;
                        if (lane >= 2) {
                            inclusive_21 += previous_23;
                        }
                        unsigned int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, inclusive_21, 4, 32);
                        unsigned int previous_24 = _shfl_up_12;
                        if (lane >= 4) {
                            inclusive_21 += previous_24;
                        }
                        unsigned int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, inclusive_21, 8, 32);
                        unsigned int previous_25 = _shfl_up_13;
                        if (lane >= 8) {
                            inclusive_21 += previous_25;
                        }
                        unsigned int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, inclusive_21, 16, 32);
                        unsigned int previous_26 = _shfl_up_14;
                        if (lane >= 16) {
                            inclusive_21 += previous_26;
                        }
                        unsigned int lane_offset_27 = block_offset + inclusive_21 - blocks_20;
                        unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, expert_18 < 384 && (pool_block_2 >= lane_offset_27 && pool_block_2 < lane_offset_27 + blocks_20));
                        unsigned int owner_mask_28 = _vote_2;
                        if (owner_mask_28 != 0) {
                            int _ffs_2 = __ffs(owner_mask_28);
                            unsigned int owner_lane_2 = _ffs_2 - 1;
                            unsigned int local_block_2 = pool_block_2 - lane_offset_27;
                            unsigned int _min_6 = ((tokens_19 - local_block_2 * 16) < (16) ? (tokens_19 - local_block_2 * 16) : (16));
                            unsigned int valid_2 = _min_6;
                            unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, expert_18, owner_lane_2);
                            task_3[1] = _shfl_13;
                            unsigned int _shfl_14 = __shfl_sync(0xFFFFFFFF, local_block_2, owner_lane_2);
                            task_3[2] = _shfl_14;
                            unsigned int _shfl_15 = __shfl_sync(0xFFFFFFFF, valid_2, owner_lane_2);
                            task_3[5] = _shfl_15;
                        }
                        unsigned int _shfl_16 = __shfl_sync(0xFFFFFFFF, inclusive_21, 31);
                        block_offset += _shfl_16;
                        unsigned int expert_29 = 96 + lane;
                        unsigned int tokens_30 = counts_1[3];
                        unsigned int blocks_31 = (tokens_30 + 16 - 1) / 16;
                        unsigned int inclusive_32 = blocks_31;
                        unsigned int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, inclusive_32, 1, 32);
                        unsigned int previous_33 = _shfl_up_15;
                        if (lane >= 1) {
                            inclusive_32 += previous_33;
                        }
                        unsigned int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, inclusive_32, 2, 32);
                        unsigned int previous_34 = _shfl_up_16;
                        if (lane >= 2) {
                            inclusive_32 += previous_34;
                        }
                        unsigned int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, inclusive_32, 4, 32);
                        unsigned int previous_35 = _shfl_up_17;
                        if (lane >= 4) {
                            inclusive_32 += previous_35;
                        }
                        unsigned int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, inclusive_32, 8, 32);
                        unsigned int previous_36 = _shfl_up_18;
                        if (lane >= 8) {
                            inclusive_32 += previous_36;
                        }
                        unsigned int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, inclusive_32, 16, 32);
                        unsigned int previous_37 = _shfl_up_19;
                        if (lane >= 16) {
                            inclusive_32 += previous_37;
                        }
                        unsigned int lane_offset_38 = block_offset + inclusive_32 - blocks_31;
                        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, expert_29 < 384 && (pool_block_2 >= lane_offset_38 && pool_block_2 < lane_offset_38 + blocks_31));
                        unsigned int owner_mask_39 = _vote_3;
                        if (owner_mask_39 != 0) {
                            int _ffs_3 = __ffs(owner_mask_39);
                            unsigned int owner_lane_3 = _ffs_3 - 1;
                            unsigned int local_block_3 = pool_block_2 - lane_offset_38;
                            unsigned int _min_7 = ((tokens_30 - local_block_3 * 16) < (16) ? (tokens_30 - local_block_3 * 16) : (16));
                            unsigned int valid_3 = _min_7;
                            unsigned int _shfl_17 = __shfl_sync(0xFFFFFFFF, expert_29, owner_lane_3);
                            task_3[1] = _shfl_17;
                            unsigned int _shfl_18 = __shfl_sync(0xFFFFFFFF, local_block_3, owner_lane_3);
                            task_3[2] = _shfl_18;
                            unsigned int _shfl_19 = __shfl_sync(0xFFFFFFFF, valid_3, owner_lane_3);
                            task_3[5] = _shfl_19;
                        }
                        unsigned int _shfl_20 = __shfl_sync(0xFFFFFFFF, inclusive_32, 31);
                        block_offset += _shfl_20;
                        unsigned int expert_40 = 128 + lane;
                        unsigned int tokens_41 = counts_1[4];
                        unsigned int blocks_42 = (tokens_41 + 16 - 1) / 16;
                        unsigned int inclusive_43 = blocks_42;
                        unsigned int _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, inclusive_43, 1, 32);
                        unsigned int previous_44 = _shfl_up_20;
                        if (lane >= 1) {
                            inclusive_43 += previous_44;
                        }
                        unsigned int _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, inclusive_43, 2, 32);
                        unsigned int previous_45 = _shfl_up_21;
                        if (lane >= 2) {
                            inclusive_43 += previous_45;
                        }
                        unsigned int _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, inclusive_43, 4, 32);
                        unsigned int previous_46 = _shfl_up_22;
                        if (lane >= 4) {
                            inclusive_43 += previous_46;
                        }
                        unsigned int _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, inclusive_43, 8, 32);
                        unsigned int previous_47 = _shfl_up_23;
                        if (lane >= 8) {
                            inclusive_43 += previous_47;
                        }
                        unsigned int _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, inclusive_43, 16, 32);
                        unsigned int previous_48 = _shfl_up_24;
                        if (lane >= 16) {
                            inclusive_43 += previous_48;
                        }
                        unsigned int lane_offset_49 = block_offset + inclusive_43 - blocks_42;
                        unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, expert_40 < 384 && (pool_block_2 >= lane_offset_49 && pool_block_2 < lane_offset_49 + blocks_42));
                        unsigned int owner_mask_50 = _vote_4;
                        if (owner_mask_50 != 0) {
                            int _ffs_4 = __ffs(owner_mask_50);
                            unsigned int owner_lane_4 = _ffs_4 - 1;
                            unsigned int local_block_4 = pool_block_2 - lane_offset_49;
                            unsigned int _min_8 = ((tokens_41 - local_block_4 * 16) < (16) ? (tokens_41 - local_block_4 * 16) : (16));
                            unsigned int valid_4 = _min_8;
                            unsigned int _shfl_21 = __shfl_sync(0xFFFFFFFF, expert_40, owner_lane_4);
                            task_3[1] = _shfl_21;
                            unsigned int _shfl_22 = __shfl_sync(0xFFFFFFFF, local_block_4, owner_lane_4);
                            task_3[2] = _shfl_22;
                            unsigned int _shfl_23 = __shfl_sync(0xFFFFFFFF, valid_4, owner_lane_4);
                            task_3[5] = _shfl_23;
                        }
                        unsigned int _shfl_24 = __shfl_sync(0xFFFFFFFF, inclusive_43, 31);
                        block_offset += _shfl_24;
                        unsigned int expert_51 = 160 + lane;
                        unsigned int tokens_52 = counts_1[5];
                        unsigned int blocks_53 = (tokens_52 + 16 - 1) / 16;
                        unsigned int inclusive_54 = blocks_53;
                        unsigned int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, inclusive_54, 1, 32);
                        unsigned int previous_55 = _shfl_up_25;
                        if (lane >= 1) {
                            inclusive_54 += previous_55;
                        }
                        unsigned int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, inclusive_54, 2, 32);
                        unsigned int previous_56 = _shfl_up_26;
                        if (lane >= 2) {
                            inclusive_54 += previous_56;
                        }
                        unsigned int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, inclusive_54, 4, 32);
                        unsigned int previous_57 = _shfl_up_27;
                        if (lane >= 4) {
                            inclusive_54 += previous_57;
                        }
                        unsigned int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, inclusive_54, 8, 32);
                        unsigned int previous_58 = _shfl_up_28;
                        if (lane >= 8) {
                            inclusive_54 += previous_58;
                        }
                        unsigned int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, inclusive_54, 16, 32);
                        unsigned int previous_59 = _shfl_up_29;
                        if (lane >= 16) {
                            inclusive_54 += previous_59;
                        }
                        unsigned int lane_offset_60 = block_offset + inclusive_54 - blocks_53;
                        unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, expert_51 < 384 && (pool_block_2 >= lane_offset_60 && pool_block_2 < lane_offset_60 + blocks_53));
                        unsigned int owner_mask_61 = _vote_5;
                        if (owner_mask_61 != 0) {
                            int _ffs_5 = __ffs(owner_mask_61);
                            unsigned int owner_lane_5 = _ffs_5 - 1;
                            unsigned int local_block_5 = pool_block_2 - lane_offset_60;
                            unsigned int _min_9 = ((tokens_52 - local_block_5 * 16) < (16) ? (tokens_52 - local_block_5 * 16) : (16));
                            unsigned int valid_5 = _min_9;
                            unsigned int _shfl_25 = __shfl_sync(0xFFFFFFFF, expert_51, owner_lane_5);
                            task_3[1] = _shfl_25;
                            unsigned int _shfl_26 = __shfl_sync(0xFFFFFFFF, local_block_5, owner_lane_5);
                            task_3[2] = _shfl_26;
                            unsigned int _shfl_27 = __shfl_sync(0xFFFFFFFF, valid_5, owner_lane_5);
                            task_3[5] = _shfl_27;
                        }
                        unsigned int _shfl_28 = __shfl_sync(0xFFFFFFFF, inclusive_54, 31);
                        block_offset += _shfl_28;
                        unsigned int expert_62 = 192 + lane;
                        unsigned int tokens_63 = counts_1[6];
                        unsigned int blocks_64 = (tokens_63 + 16 - 1) / 16;
                        unsigned int inclusive_65 = blocks_64;
                        unsigned int _shfl_up_30 = __shfl_up_sync(0xFFFFFFFF, inclusive_65, 1, 32);
                        unsigned int previous_66 = _shfl_up_30;
                        if (lane >= 1) {
                            inclusive_65 += previous_66;
                        }
                        unsigned int _shfl_up_31 = __shfl_up_sync(0xFFFFFFFF, inclusive_65, 2, 32);
                        unsigned int previous_67 = _shfl_up_31;
                        if (lane >= 2) {
                            inclusive_65 += previous_67;
                        }
                        unsigned int _shfl_up_32 = __shfl_up_sync(0xFFFFFFFF, inclusive_65, 4, 32);
                        unsigned int previous_68 = _shfl_up_32;
                        if (lane >= 4) {
                            inclusive_65 += previous_68;
                        }
                        unsigned int _shfl_up_33 = __shfl_up_sync(0xFFFFFFFF, inclusive_65, 8, 32);
                        unsigned int previous_69 = _shfl_up_33;
                        if (lane >= 8) {
                            inclusive_65 += previous_69;
                        }
                        unsigned int _shfl_up_34 = __shfl_up_sync(0xFFFFFFFF, inclusive_65, 16, 32);
                        unsigned int previous_70 = _shfl_up_34;
                        if (lane >= 16) {
                            inclusive_65 += previous_70;
                        }
                        unsigned int lane_offset_71 = block_offset + inclusive_65 - blocks_64;
                        unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, expert_62 < 384 && (pool_block_2 >= lane_offset_71 && pool_block_2 < lane_offset_71 + blocks_64));
                        unsigned int owner_mask_72 = _vote_6;
                        if (owner_mask_72 != 0) {
                            int _ffs_6 = __ffs(owner_mask_72);
                            unsigned int owner_lane_6 = _ffs_6 - 1;
                            unsigned int local_block_6 = pool_block_2 - lane_offset_71;
                            unsigned int _min_10 = ((tokens_63 - local_block_6 * 16) < (16) ? (tokens_63 - local_block_6 * 16) : (16));
                            unsigned int valid_6 = _min_10;
                            unsigned int _shfl_29 = __shfl_sync(0xFFFFFFFF, expert_62, owner_lane_6);
                            task_3[1] = _shfl_29;
                            unsigned int _shfl_30 = __shfl_sync(0xFFFFFFFF, local_block_6, owner_lane_6);
                            task_3[2] = _shfl_30;
                            unsigned int _shfl_31 = __shfl_sync(0xFFFFFFFF, valid_6, owner_lane_6);
                            task_3[5] = _shfl_31;
                        }
                        unsigned int _shfl_32 = __shfl_sync(0xFFFFFFFF, inclusive_65, 31);
                        block_offset += _shfl_32;
                        unsigned int expert_73 = 224 + lane;
                        unsigned int tokens_74 = counts_1[7];
                        unsigned int blocks_75 = (tokens_74 + 16 - 1) / 16;
                        unsigned int inclusive_76 = blocks_75;
                        unsigned int _shfl_up_35 = __shfl_up_sync(0xFFFFFFFF, inclusive_76, 1, 32);
                        unsigned int previous_77 = _shfl_up_35;
                        if (lane >= 1) {
                            inclusive_76 += previous_77;
                        }
                        unsigned int _shfl_up_36 = __shfl_up_sync(0xFFFFFFFF, inclusive_76, 2, 32);
                        unsigned int previous_78 = _shfl_up_36;
                        if (lane >= 2) {
                            inclusive_76 += previous_78;
                        }
                        unsigned int _shfl_up_37 = __shfl_up_sync(0xFFFFFFFF, inclusive_76, 4, 32);
                        unsigned int previous_79 = _shfl_up_37;
                        if (lane >= 4) {
                            inclusive_76 += previous_79;
                        }
                        unsigned int _shfl_up_38 = __shfl_up_sync(0xFFFFFFFF, inclusive_76, 8, 32);
                        unsigned int previous_80 = _shfl_up_38;
                        if (lane >= 8) {
                            inclusive_76 += previous_80;
                        }
                        unsigned int _shfl_up_39 = __shfl_up_sync(0xFFFFFFFF, inclusive_76, 16, 32);
                        unsigned int previous_81 = _shfl_up_39;
                        if (lane >= 16) {
                            inclusive_76 += previous_81;
                        }
                        unsigned int lane_offset_82 = block_offset + inclusive_76 - blocks_75;
                        unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, expert_73 < 384 && (pool_block_2 >= lane_offset_82 && pool_block_2 < lane_offset_82 + blocks_75));
                        unsigned int owner_mask_83 = _vote_7;
                        if (owner_mask_83 != 0) {
                            int _ffs_7 = __ffs(owner_mask_83);
                            unsigned int owner_lane_7 = _ffs_7 - 1;
                            unsigned int local_block_7 = pool_block_2 - lane_offset_82;
                            unsigned int _min_11 = ((tokens_74 - local_block_7 * 16) < (16) ? (tokens_74 - local_block_7 * 16) : (16));
                            unsigned int valid_7 = _min_11;
                            unsigned int _shfl_33 = __shfl_sync(0xFFFFFFFF, expert_73, owner_lane_7);
                            task_3[1] = _shfl_33;
                            unsigned int _shfl_34 = __shfl_sync(0xFFFFFFFF, local_block_7, owner_lane_7);
                            task_3[2] = _shfl_34;
                            unsigned int _shfl_35 = __shfl_sync(0xFFFFFFFF, valid_7, owner_lane_7);
                            task_3[5] = _shfl_35;
                        }
                        unsigned int _shfl_36 = __shfl_sync(0xFFFFFFFF, inclusive_76, 31);
                        block_offset += _shfl_36;
                        unsigned int expert_84 = 256 + lane;
                        unsigned int tokens_85 = counts_1[8];
                        unsigned int blocks_86 = (tokens_85 + 16 - 1) / 16;
                        unsigned int inclusive_87 = blocks_86;
                        unsigned int _shfl_up_40 = __shfl_up_sync(0xFFFFFFFF, inclusive_87, 1, 32);
                        unsigned int previous_88 = _shfl_up_40;
                        if (lane >= 1) {
                            inclusive_87 += previous_88;
                        }
                        unsigned int _shfl_up_41 = __shfl_up_sync(0xFFFFFFFF, inclusive_87, 2, 32);
                        unsigned int previous_89 = _shfl_up_41;
                        if (lane >= 2) {
                            inclusive_87 += previous_89;
                        }
                        unsigned int _shfl_up_42 = __shfl_up_sync(0xFFFFFFFF, inclusive_87, 4, 32);
                        unsigned int previous_90 = _shfl_up_42;
                        if (lane >= 4) {
                            inclusive_87 += previous_90;
                        }
                        unsigned int _shfl_up_43 = __shfl_up_sync(0xFFFFFFFF, inclusive_87, 8, 32);
                        unsigned int previous_91 = _shfl_up_43;
                        if (lane >= 8) {
                            inclusive_87 += previous_91;
                        }
                        unsigned int _shfl_up_44 = __shfl_up_sync(0xFFFFFFFF, inclusive_87, 16, 32);
                        unsigned int previous_92 = _shfl_up_44;
                        if (lane >= 16) {
                            inclusive_87 += previous_92;
                        }
                        unsigned int lane_offset_93 = block_offset + inclusive_87 - blocks_86;
                        unsigned int _vote_8 = __ballot_sync(0xFFFFFFFF, expert_84 < 384 && (pool_block_2 >= lane_offset_93 && pool_block_2 < lane_offset_93 + blocks_86));
                        unsigned int owner_mask_94 = _vote_8;
                        if (owner_mask_94 != 0) {
                            int _ffs_8 = __ffs(owner_mask_94);
                            unsigned int owner_lane_8 = _ffs_8 - 1;
                            unsigned int local_block_8 = pool_block_2 - lane_offset_93;
                            unsigned int _min_12 = ((tokens_85 - local_block_8 * 16) < (16) ? (tokens_85 - local_block_8 * 16) : (16));
                            unsigned int valid_8 = _min_12;
                            unsigned int _shfl_37 = __shfl_sync(0xFFFFFFFF, expert_84, owner_lane_8);
                            task_3[1] = _shfl_37;
                            unsigned int _shfl_38 = __shfl_sync(0xFFFFFFFF, local_block_8, owner_lane_8);
                            task_3[2] = _shfl_38;
                            unsigned int _shfl_39 = __shfl_sync(0xFFFFFFFF, valid_8, owner_lane_8);
                            task_3[5] = _shfl_39;
                        }
                        unsigned int _shfl_40 = __shfl_sync(0xFFFFFFFF, inclusive_87, 31);
                        block_offset += _shfl_40;
                        unsigned int expert_95 = 288 + lane;
                        unsigned int tokens_96 = counts_1[9];
                        unsigned int blocks_97 = (tokens_96 + 16 - 1) / 16;
                        unsigned int inclusive_98 = blocks_97;
                        unsigned int _shfl_up_45 = __shfl_up_sync(0xFFFFFFFF, inclusive_98, 1, 32);
                        unsigned int previous_99 = _shfl_up_45;
                        if (lane >= 1) {
                            inclusive_98 += previous_99;
                        }
                        unsigned int _shfl_up_46 = __shfl_up_sync(0xFFFFFFFF, inclusive_98, 2, 32);
                        unsigned int previous_100 = _shfl_up_46;
                        if (lane >= 2) {
                            inclusive_98 += previous_100;
                        }
                        unsigned int _shfl_up_47 = __shfl_up_sync(0xFFFFFFFF, inclusive_98, 4, 32);
                        unsigned int previous_101 = _shfl_up_47;
                        if (lane >= 4) {
                            inclusive_98 += previous_101;
                        }
                        unsigned int _shfl_up_48 = __shfl_up_sync(0xFFFFFFFF, inclusive_98, 8, 32);
                        unsigned int previous_102 = _shfl_up_48;
                        if (lane >= 8) {
                            inclusive_98 += previous_102;
                        }
                        unsigned int _shfl_up_49 = __shfl_up_sync(0xFFFFFFFF, inclusive_98, 16, 32);
                        unsigned int previous_103 = _shfl_up_49;
                        if (lane >= 16) {
                            inclusive_98 += previous_103;
                        }
                        unsigned int lane_offset_104 = block_offset + inclusive_98 - blocks_97;
                        unsigned int _vote_9 = __ballot_sync(0xFFFFFFFF, expert_95 < 384 && (pool_block_2 >= lane_offset_104 && pool_block_2 < lane_offset_104 + blocks_97));
                        unsigned int owner_mask_105 = _vote_9;
                        if (owner_mask_105 != 0) {
                            int _ffs_9 = __ffs(owner_mask_105);
                            unsigned int owner_lane_9 = _ffs_9 - 1;
                            unsigned int local_block_9 = pool_block_2 - lane_offset_104;
                            unsigned int _min_13 = ((tokens_96 - local_block_9 * 16) < (16) ? (tokens_96 - local_block_9 * 16) : (16));
                            unsigned int valid_9 = _min_13;
                            unsigned int _shfl_41 = __shfl_sync(0xFFFFFFFF, expert_95, owner_lane_9);
                            task_3[1] = _shfl_41;
                            unsigned int _shfl_42 = __shfl_sync(0xFFFFFFFF, local_block_9, owner_lane_9);
                            task_3[2] = _shfl_42;
                            unsigned int _shfl_43 = __shfl_sync(0xFFFFFFFF, valid_9, owner_lane_9);
                            task_3[5] = _shfl_43;
                        }
                        unsigned int _shfl_44 = __shfl_sync(0xFFFFFFFF, inclusive_98, 31);
                        block_offset += _shfl_44;
                        unsigned int expert_106 = 320 + lane;
                        unsigned int tokens_107 = counts_1[10];
                        unsigned int blocks_108 = (tokens_107 + 16 - 1) / 16;
                        unsigned int inclusive_109 = blocks_108;
                        unsigned int _shfl_up_50 = __shfl_up_sync(0xFFFFFFFF, inclusive_109, 1, 32);
                        unsigned int previous_110 = _shfl_up_50;
                        if (lane >= 1) {
                            inclusive_109 += previous_110;
                        }
                        unsigned int _shfl_up_51 = __shfl_up_sync(0xFFFFFFFF, inclusive_109, 2, 32);
                        unsigned int previous_111 = _shfl_up_51;
                        if (lane >= 2) {
                            inclusive_109 += previous_111;
                        }
                        unsigned int _shfl_up_52 = __shfl_up_sync(0xFFFFFFFF, inclusive_109, 4, 32);
                        unsigned int previous_112 = _shfl_up_52;
                        if (lane >= 4) {
                            inclusive_109 += previous_112;
                        }
                        unsigned int _shfl_up_53 = __shfl_up_sync(0xFFFFFFFF, inclusive_109, 8, 32);
                        unsigned int previous_113 = _shfl_up_53;
                        if (lane >= 8) {
                            inclusive_109 += previous_113;
                        }
                        unsigned int _shfl_up_54 = __shfl_up_sync(0xFFFFFFFF, inclusive_109, 16, 32);
                        unsigned int previous_114 = _shfl_up_54;
                        if (lane >= 16) {
                            inclusive_109 += previous_114;
                        }
                        unsigned int lane_offset_115 = block_offset + inclusive_109 - blocks_108;
                        unsigned int _vote_10 = __ballot_sync(0xFFFFFFFF, expert_106 < 384 && (pool_block_2 >= lane_offset_115 && pool_block_2 < lane_offset_115 + blocks_108));
                        unsigned int owner_mask_116 = _vote_10;
                        if (owner_mask_116 != 0) {
                            int _ffs_10 = __ffs(owner_mask_116);
                            unsigned int owner_lane_10 = _ffs_10 - 1;
                            unsigned int local_block_10 = pool_block_2 - lane_offset_115;
                            unsigned int _min_14 = ((tokens_107 - local_block_10 * 16) < (16) ? (tokens_107 - local_block_10 * 16) : (16));
                            unsigned int valid_10 = _min_14;
                            unsigned int _shfl_45 = __shfl_sync(0xFFFFFFFF, expert_106, owner_lane_10);
                            task_3[1] = _shfl_45;
                            unsigned int _shfl_46 = __shfl_sync(0xFFFFFFFF, local_block_10, owner_lane_10);
                            task_3[2] = _shfl_46;
                            unsigned int _shfl_47 = __shfl_sync(0xFFFFFFFF, valid_10, owner_lane_10);
                            task_3[5] = _shfl_47;
                        }
                        unsigned int _shfl_48 = __shfl_sync(0xFFFFFFFF, inclusive_109, 31);
                        block_offset += _shfl_48;
                        unsigned int expert_117 = 352 + lane;
                        unsigned int tokens_118 = counts_1[11];
                        unsigned int blocks_119 = (tokens_118 + 16 - 1) / 16;
                        unsigned int inclusive_120 = blocks_119;
                        unsigned int _shfl_up_55 = __shfl_up_sync(0xFFFFFFFF, inclusive_120, 1, 32);
                        unsigned int previous_121 = _shfl_up_55;
                        if (lane >= 1) {
                            inclusive_120 += previous_121;
                        }
                        unsigned int _shfl_up_56 = __shfl_up_sync(0xFFFFFFFF, inclusive_120, 2, 32);
                        unsigned int previous_122 = _shfl_up_56;
                        if (lane >= 2) {
                            inclusive_120 += previous_122;
                        }
                        unsigned int _shfl_up_57 = __shfl_up_sync(0xFFFFFFFF, inclusive_120, 4, 32);
                        unsigned int previous_123 = _shfl_up_57;
                        if (lane >= 4) {
                            inclusive_120 += previous_123;
                        }
                        unsigned int _shfl_up_58 = __shfl_up_sync(0xFFFFFFFF, inclusive_120, 8, 32);
                        unsigned int previous_124 = _shfl_up_58;
                        if (lane >= 8) {
                            inclusive_120 += previous_124;
                        }
                        unsigned int _shfl_up_59 = __shfl_up_sync(0xFFFFFFFF, inclusive_120, 16, 32);
                        unsigned int previous_125 = _shfl_up_59;
                        if (lane >= 16) {
                            inclusive_120 += previous_125;
                        }
                        unsigned int lane_offset_126 = block_offset + inclusive_120 - blocks_119;
                        unsigned int _vote_11 = __ballot_sync(0xFFFFFFFF, expert_117 < 384 && (pool_block_2 >= lane_offset_126 && pool_block_2 < lane_offset_126 + blocks_119));
                        unsigned int owner_mask_127 = _vote_11;
                        if (owner_mask_127 != 0) {
                            int _ffs_11 = __ffs(owner_mask_127);
                            unsigned int owner_lane_11 = _ffs_11 - 1;
                            unsigned int local_block_11 = pool_block_2 - lane_offset_126;
                            unsigned int _min_15 = ((tokens_118 - local_block_11 * 16) < (16) ? (tokens_118 - local_block_11 * 16) : (16));
                            unsigned int valid_11 = _min_15;
                            unsigned int _shfl_49 = __shfl_sync(0xFFFFFFFF, expert_117, owner_lane_11);
                            task_3[1] = _shfl_49;
                            unsigned int _shfl_50 = __shfl_sync(0xFFFFFFFF, local_block_11, owner_lane_11);
                            task_3[2] = _shfl_50;
                            unsigned int _shfl_51 = __shfl_sync(0xFFFFFFFF, valid_11, owner_lane_11);
                            task_3[5] = _shfl_51;
                        }
                        unsigned int _shfl_52 = __shfl_sync(0xFFFFFFFF, inclusive_120, 31);
                        block_offset += _shfl_52;
                        if (phase_1 == 2) {
                            unsigned int required = (task_3[4] + 1) * 18;
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"(reinterpret_cast<unsigned int*>(L1Counter)) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(required)) >= static_cast<unsigned int>((unsigned int)0 - required));
                            }
                        }
                    }
                    if (task_3[0] == 0) {
                        break;
                    }
                    if (lane < 2) {
                        uint32_t _mapa_4;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_4) : "r"(task_infos_addr + schedule_iteration[0] % 2 * 32), "r"(lane));
                        unsigned int destination_2 = _mapa_4;
                        uint32_t _mapa_5;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_5) : "r"(task_full_addr + schedule_iteration[0] % 2 * 8), "r"(lane));
                        unsigned int barrier_2 = _mapa_5;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"(barrier_2), "r"((uint32_t)(32)) : "memory");
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(destination_2), "r"(task_3[0]), "r"(task_3[1]), "r"(task_3[2]), "r"(task_3[3]), "r"(barrier_2) : "memory");
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(destination_2 + 16), "r"(task_3[4]), "r"(task_3[5]), "r"(task_3[6]), "r"(task_3[7]), "r"(barrier_2) : "memory");
                    }
                    __syncwarp();
                    schedule_iteration[0] = schedule_iteration[0] + 1;
                }
                if (early_shared_l2 == 0) {
                    unsigned int total_0_1 = (num_tokens + 16 - 1) / 16 * 20;
                    #pragma unroll 1
                    for (int iteration_3 = 0; iteration_3 < 250801; iteration_3++) {
                        mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                        unsigned int claimed_3 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_7 = atomicAdd(SharedL2Counter, 1);
                            claimed_3 = _atomic_old_7;
                        }
                        unsigned int _shfl_53 = __shfl_sync(0xFFFFFFFF, claimed_3, 0);
                        unsigned int claimed_0_4 = _shfl_53;
                        if (claimed_0_4 >= total_0_1) {
                            break;
                        }
                        unsigned int block_5 = claimed_0_4 / 20;
                        task_3[0] = 4;
                        task_3[1] = 0;
                        task_3[2] = block_5;
                        task_3[3] = claimed_0_4 % 20;
                        task_3[4] = block_5;
                        unsigned int _min_16 = ((num_tokens - block_5 * 16) < (16) ? (num_tokens - block_5 * 16) : (16));
                        task_3[5] = _min_16;
                        task_3[6] = 5120;
                        task_3[7] = 2304;
                        if (lane < 2) {
                            uint32_t _mapa_6;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_6) : "r"(task_infos_addr + schedule_iteration[0] % 2 * 32), "r"(lane));
                            unsigned int destination_3 = _mapa_6;
                            uint32_t _mapa_7;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_7) : "r"(task_full_addr + schedule_iteration[0] % 2 * 8), "r"(lane));
                            unsigned int barrier_3 = _mapa_7;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"(barrier_3), "r"((uint32_t)(32)) : "memory");
                            asm volatile(
                                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                :: "r"(destination_3), "r"(task_3[0]), "r"(task_3[1]), "r"(task_3[2]), "r"(task_3[3]), "r"(barrier_3) : "memory");
                            asm volatile(
                                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                :: "r"(destination_3 + 16), "r"(task_3[4]), "r"(task_3[5]), "r"(task_3[6]), "r"(task_3[7]), "r"(barrier_3) : "memory");
                        }
                        __syncwarp();
                        schedule_iteration[0] = schedule_iteration[0] + 1;
                    }
                }
                mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                task_3[0] = 0;
                task_3[1] = 0;
                task_3[2] = 0;
                task_3[3] = 0;
                task_3[4] = 0;
                task_3[5] = 0;
                task_3[6] = 0;
                task_3[7] = 0;
                if (lane < 2) {
                    uint32_t _mapa_8;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_8) : "r"(task_infos_addr + schedule_iteration[0] % 2 * 32), "r"(lane));
                    unsigned int destination_4 = _mapa_8;
                    uint32_t _mapa_9;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_9) : "r"(task_full_addr + schedule_iteration[0] % 2 * 8), "r"(lane));
                    unsigned int barrier_4 = _mapa_9;
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"(barrier_4), "r"((uint32_t)(32)) : "memory");
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(destination_4), "r"(task_3[0]), "r"(task_3[1]), "r"(task_3[2]), "r"(task_3[3]), "r"(barrier_4) : "memory");
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(destination_4 + 16), "r"(task_3[4]), "r"(task_3[5]), "r"(task_3[6]), "r"(task_3[7]), "r"(barrier_4) : "memory");
                }
                __syncwarp();
            }
        }
    // ---- Role: epilogue ----
    } else if (warp >= 8 && warp <= 15) {
        { // epilogue_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
            uint32_t lane_read_0_2;
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_read_0_2));
            unsigned int task_4[8];
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            #pragma unroll 1
            for (int task_iteration_3 = 0; task_iteration_3 < 250801; task_iteration_3++) {
                mbarrier_wait(task_full_addr + (task_iteration_3 % 2) * 8, task_iteration_3 / 2 & 1);
                task_4[0] = task_infos[task_iteration_3 % 2 * 8];
                task_4[1] = task_infos[task_iteration_3 % 2 * 8 + 1];
                task_4[2] = task_infos[task_iteration_3 % 2 * 8 + 2];
                task_4[3] = task_infos[task_iteration_3 % 2 * 8 + 3];
                task_4[4] = task_infos[task_iteration_3 % 2 * 8 + 4];
                task_4[5] = task_infos[task_iteration_3 % 2 * 8 + 5];
                task_4[6] = task_infos[task_iteration_3 % 2 * 8 + 6];
                task_4[7] = task_infos[task_iteration_3 % 2 * 8 + 7];
                if (task_4[0] == 0) {
                    break;
                }
                unsigned int accum_stage_1 = task_iteration_3 % 2;
                mbarrier_wait(tmem_full_addr + (accum_stage_1) * 8, task_iteration_3 / 2 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((task_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                if (task_4[0] == 1 || task_4[0] == 3) {
                    unsigned int epi_warp = warp - 8;
                    unsigned int epi_wg = epi_warp / 4;
                    unsigned int warp_in_wg = epi_warp % 4;
                    unsigned int _shfl_54 = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int valid_m = _shfl_54;
                    unsigned int pool_block_3 = task_4[4];
                    unsigned int ring_block_2 = pool_block_3 % 960;
                    unsigned int block_6 = ring_block_2;
                    unsigned int sf_stride = 983040;
                    if (task_4[0] == 3) {
                        block_6 = pool_block_3;
                        sf_stride = 122880;
                    }
                    unsigned int n_block = task_4[3] * 2 + (unsigned int)cta_rank;
                    float cached_weight = 1.0f;
                    float activation[4];
                    float amax[2];
                    float quantized[4];
                    if (task_4[0] == 1) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(L2Empty) + (ring_block_2))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(40 * (pool_block_3 / 960))) >= static_cast<unsigned int>(1));
                        }
                    }
                    #pragma unroll
                    for (int s = 0; s < 1; s++) {
                        if (valid_m <= epi_wg * 8 + (unsigned int)(s * 8)) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            break;
                        }
                        #pragma unroll
                        for (int i = 0; i < 1; i++) {
                            unsigned int j = s + i;
                            if (task_4[0] == 1 && j * 8 % 32 == 0) {
                                if (j * 8 + lane_read_0_2 < 8) {
                                    cached_weight = L1Weights[ring_block_2 * 16 + epi_wg * 8 + j * 8 + lane_read_0_2];
                                }
                            }
                            float _shfl_55;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_55) : "f"(cached_weight), "r"(j * 8 % 32 + lane_read_0_2 % 4 * 2));
                            float first_weight = _shfl_55;
                            float _shfl_56;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_56) : "f"(cached_weight), "r"(j * 8 % 32 + lane_read_0_2 % 4 * 2 + 1));
                            float second_weight = _shfl_56;
                            unsigned int address = accum_stage_1 * 16 + epi_wg * 8 + j * 8;
                            float _tmem_load_0[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                                : "r"(address));
                            float _tmem_load_1[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                                : "r"(address | 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (j == 0) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                            __nv_bfloat162 _min_17 = __hmin2(_bf16x2_0, _bf16x2_2);
                            __nv_bfloat162 _max_2 = __hmax2(_bf16x2_1, _bf16x2_3);
                            __nv_bfloat162 _min_18 = __hmin2(_max_2, _bf16x2_2);
                            float2 _cvt_f32_0 = __bfloat1622float2(_min_17);
                            float2 _cvt_f32_1 = __bfloat1622float2(_min_18);
                            float denominator[2];
                            float2 activated_gate;
                            float _expf_0 = __expf(-_cvt_f32_0.x);
                            denominator[0] = _expf_0;
                            float _expf_1 = __expf(-_cvt_f32_0.y);
                            denominator[1] = _expf_1;
                            const float2 _add2_0 = {1.0f, 1.0f};
                            #pragma unroll
                            for (int _la = 0; _la < 1; _la++) {
                                reinterpret_cast<float2*>(denominator)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(denominator)[_la], _add2_0);
                            }
                            float _rcp_0 = approx_rcp(denominator[0]);
                            float _rcp_1 = approx_rcp(denominator[1]);
                            float2 _f2_0 = make_float2(_rcp_0, _rcp_1);
                            float2 _mul_f32x2_0;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_cvt_f32_0), "l"(*(const unsigned long long*)&_f2_0));
                            activated_gate = _mul_f32x2_0;
                            float2 _mul_f32x2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&activated_gate), "l"(*(const unsigned long long*)&_cvt_f32_1));
                            float2 _f2_1 = make_float2(first_weight, second_weight);
                            float2 _mul_f32x2_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_mul_f32x2_1), "l"(*(const unsigned long long*)&_f2_1));
                            activation[i * 4] = _mul_f32x2_2.x;
                            activation[i * 4 + 1] = _mul_f32x2_2.y;
                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                            __nv_bfloat162 _min_19 = __hmin2(_bf16x2_4, _bf16x2_6);
                            __nv_bfloat162 _max_3 = __hmax2(_bf16x2_5, _bf16x2_7);
                            __nv_bfloat162 _min_20 = __hmin2(_max_3, _bf16x2_6);
                            float2 _cvt_f32_2 = __bfloat1622float2(_min_19);
                            float2 _cvt_f32_3 = __bfloat1622float2(_min_20);
                            float denominator_0[2];
                            float2 activated_gate_1;
                            float _expf_2 = __expf(-_cvt_f32_2.x);
                            denominator_0[0] = _expf_2;
                            float _expf_3 = __expf(-_cvt_f32_2.y);
                            denominator_0[1] = _expf_3;
                            const float2 _add2_1 = {1.0f, 1.0f};
                            #pragma unroll
                            for (int _la = 0; _la < 1; _la++) {
                                reinterpret_cast<float2*>(denominator_0)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(denominator_0)[_la], _add2_1);
                            }
                            float _rcp_2 = approx_rcp(denominator_0[0]);
                            float _rcp_3 = approx_rcp(denominator_0[1]);
                            float2 _f2_2 = make_float2(_rcp_2, _rcp_3);
                            float2 _mul_f32x2_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_cvt_f32_2), "l"(*(const unsigned long long*)&_f2_2));
                            activated_gate_1 = _mul_f32x2_3;
                            float2 _mul_f32x2_4;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&activated_gate_1), "l"(*(const unsigned long long*)&_cvt_f32_3));
                            float2 _f2_3 = make_float2(first_weight, second_weight);
                            float2 _mul_f32x2_5;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_mul_f32x2_4), "l"(*(const unsigned long long*)&_f2_3));
                            activation[i * 4 + 2] = _mul_f32x2_5.x;
                            activation[i * 4 + 3] = _mul_f32x2_5.y;
                            float _fabs_0 = fabsf(activation[i * 4]);
                            float _fmax_0 = fmaxf(0.0f, _fabs_0);
                            float _fabs_1 = fabsf(activation[i * 4 + 2]);
                            float _fmax_1 = fmaxf(_fmax_0, _fabs_1);
                            float first_max = _fmax_1;
                            float _fabs_2 = fabsf(activation[i * 4 + 1]);
                            float _fmax_2 = fmaxf(0.0f, _fabs_2);
                            float _fabs_3 = fabsf(activation[i * 4 + 3]);
                            float _fmax_3 = fmaxf(_fmax_2, _fabs_3);
                            float second_max = _fmax_3;
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, first_max, 4);
                            float _max_4 = max_noftz(first_max, _shfl_xor_0);
                            first_max = _max_4;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, second_max, 4);
                            float _max_5 = max_noftz(second_max, _shfl_xor_1);
                            second_max = _max_5;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, first_max, 8);
                            float _max_6 = max_noftz(first_max, _shfl_xor_2);
                            first_max = _max_6;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, second_max, 8);
                            float _max_7 = max_noftz(second_max, _shfl_xor_3);
                            second_max = _max_7;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, first_max, 16);
                            float _max_8 = max_noftz(first_max, _shfl_xor_4);
                            first_max = _max_8;
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, second_max, 16);
                            float _max_9 = max_noftz(second_max, _shfl_xor_5);
                            second_max = _max_9;
                            amax[i * 2] = first_max;
                            amax[i * 2 + 1] = second_max;
                            if (lane_read_0_2 < 4) {
                                amax_reduction[epi_warp * 8 + (unsigned int)(i * 8) + lane_read_0_2 * 2] = first_max;
                                amax_reduction[epi_warp * 8 + (unsigned int)(i * 8) + lane_read_0_2 * 2 + 1] = second_max;
                            }
                            __syncwarp();
                        }
                        unsigned int tma_stage = s % 2;
                        unsigned int store_address = smem_output_addr + (epi_wg * 2 + tma_stage) * 8 * 64;
                        asm volatile("cp.async.bulk.wait_group 1;");
                        asm volatile("bar.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 1; i_1++) {
                            unsigned int paired_index = (epi_warp ^ 1) * 8 + (unsigned int)(i_1 * 8) + lane_read_0_2 % 4 * 2;
                            float _max_10 = max_noftz(amax[i_1 * 2], amax_reduction[paired_index]);
                            float first_max_1 = _max_10;
                            float _max_11 = max_noftz(amax[i_1 * 2 + 1], amax_reduction[paired_index + 1]);
                            float second_max_1 = _max_11;
                            unsigned int first_bits = __as_u32(first_max_1);
                            unsigned int second_bits = __as_u32(second_max_1);
                            unsigned int first_rounded_exp = first_bits + 2097151 >> 23;
                            unsigned int first_exp = ((first_rounded_exp < 113) ? (unsigned int)113 : first_rounded_exp) - 8;
                            unsigned int second_rounded_exp = second_bits + 2097151 >> 23;
                            unsigned int second_exp = ((second_rounded_exp < 113) ? (unsigned int)113 : second_rounded_exp) - 8;
                            unsigned int first_inverse_bits = 254 - first_exp << 23;
                            unsigned int second_inverse_bits = 254 - second_exp << 23;
                            float first_inverse = __uint_as_float(first_inverse_bits);
                            float second_inverse = __uint_as_float(second_inverse_bits);
                            float2 _f2_4 = make_float2(first_inverse, second_inverse);
                            float2 _f2_5 = make_float2(activation[i_1 * 4], activation[i_1 * 4 + 1]);
                            float2 _mul_f32x2_6;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_5), "l"(*(const unsigned long long*)&_f2_4));
                            float2 _f2_6 = make_float2(activation[i_1 * 4 + 2], activation[i_1 * 4 + 3]);
                            float2 _mul_f32x2_7;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_4));
                            quantized[0] = _mul_f32x2_6.x;
                            quantized[1] = _mul_f32x2_6.y;
                            quantized[2] = _mul_f32x2_7.x;
                            quantized[3] = _mul_f32x2_7.y;
                            uint32_t _fp8_0[1];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(quantized[0]), "f"(quantized[1]),
                                                       "f"(quantized[2]), "f"(quantized[3]));
                                _fp8_0[0] = _packed;
                            }
                            unsigned int stsm_address = store_address + (unsigned int)(i_1 * 8 * 64) + lane_read_0_2 * 64 + (warp_in_wg ^ lane_read_0_2 / 2) * 16;
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                            uint32_t _stmatrix_b8_x1_addr_2 = static_cast<uint32_t>(stsm_address);
                            asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                                :: "r"(_stmatrix_b8_x1_addr_2), "r"(_fp8_0[0]) : "memory");
                            #elif defined(__CUDA_ARCH__)
                            #error "StmatrixTransB8X1 requires SM100 or newer"
                            #endif
                            if (warp_in_wg % 2 == 0 && lane_read_0_2 < 4) {
                                unsigned int sf_k = n_block * 2 + warp_in_wg / 2;
                                unsigned int token_base_2 = epi_wg * 8 + (unsigned int)(s * 8) + (unsigned int)(i_1 * 8);
                                unsigned int permuted_base = (token_base_2 & 4294967168) + (token_base_2 & 31) * 4 + (token_base_2 >> 5 & 3);
                                unsigned int sf_token_1 = block_6 * 128 + permuted_base + lane_read_0_2 * 8;
                                unsigned int sf_address = sf_k / 4 * sf_stride + sf_token_1 * 4 + sf_k % 4;
                                if (task_4[0] == 3) {
                                    SharedL2SF[sf_address] = (uint8_t)first_exp;
                                    SharedL2SF[sf_address + 16] = (uint8_t)second_exp;
                                } else {
                                    L2SF[sf_address] = (uint8_t)first_exp;
                                    L2SF[sf_address + 16] = (uint8_t)second_exp;
                                }
                            }
                            __syncwarp();
                        }
                        asm volatile("bar.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                        if (warp_in_wg == 0) {
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                if (task_4[0] == 3) {
                                    tma_store_2d(SharedL1Output, n_block * 64, block_6 * 16 + epi_wg * 8 + (unsigned int)(s * 8), store_address);
                                } else {
                                    tma_store_2d(L1Output, n_block * 64, block_6 * 16 + epi_wg * 8 + (unsigned int)(s * 8), store_address);
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        __syncwarp();
                    }
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            if (task_4[0] == 3) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(SharedFull) + (pool_block_3))), "r"(static_cast<unsigned int>(1)) : "memory");
                            } else {
                                asm volatile("red.release.gpu.global.xor.b64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(L2Mask) + (ring_block_2))), "l"(static_cast<unsigned long long>((unsigned long long)1 << (unsigned long long)n_block)) : "memory");
                                atomicAdd(&L1Empty[ring_block_2], 1);
                            }
                        }
                    }
                    __syncwarp();
                } else {
                    unsigned int epi_warp_1 = warp - 8;
                    unsigned int epi_wg_1 = epi_warp_1 / 4;
                    unsigned int warp_in_wg_1 = epi_warp_1 % 4;
                    unsigned int _shfl_57 = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int valid_m_1 = _shfl_57;
                    unsigned int pool_m = task_4[4] * 16;
                    unsigned int n_offset_1 = (task_4[3] * 2 + (unsigned int)cta_rank) * 128;
                    unsigned int output_base = smem_output_addr + epi_wg_1 * 8 * 128 * 2;
                    unsigned int cached_token[1];
                    unsigned int cached_topk[1];
                    unsigned int packed[4];
                    if (task_4[0] == 2) {
                        if (epi_warp_1 == 0) {
                            if (elect_sync()) {
                                atomicAdd(&L2Empty[task_4[4] % 960], 1);
                            }
                        }
                        __syncwarp();
                    }
                    #pragma unroll
                    for (int s_1 = 0; s_1 < 1; s_1++) {
                        if (valid_m_1 <= epi_wg_1 * 8 + (unsigned int)(s_1 * 8)) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            break;
                        }
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 1; j_1++) {
                            unsigned int row = epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + (unsigned int)(j_1 * 8) + warp_in_wg_1 * 2 + lane_read_0_2 / 16;
                            if (row < valid_m_1) {
                                if (task_4[0] == 4) {
                                    cached_token[j_1] = pool_m + row;
                                    cached_topk[j_1] = 6;
                                } else {
                                    cached_token[j_1] = TokenMetadata[(pool_m + row) * 3 + 1];
                                    cached_topk[j_1] = TokenMetadata[(pool_m + row) * 3 + 2];
                                }
                            }
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 1; i_2++) {
                            unsigned int address_1 = accum_stage_1 * 16 + epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + (unsigned int)(i_2 * 8);
                            float _tmem_load_2[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3]))
                                : "r"(address_1));
                            float _tmem_load_3[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3]))
                                : "r"(address_1 | 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (i_2 == 0 && s_1 > 0) {
                                asm volatile("bar.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                            }
                            if (s_1 == 0 && i_2 == 0) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            uint32_t _tmem_load_2_bf16[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                                _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _tmem_load_3_bf16[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                                _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int row_1 = lane_read_0_2 % 8;
                            unsigned int col = epi_warp_1 % 2 * 4 + lane_read_0_2 / 8;
                            unsigned int write_address = output_base + warp_in_wg_1 / 2 * 8 * 128 + (unsigned int)(i_2 * 8 * 128) + row_1 * 128 + (col ^ row_1) * 16;
                            uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(write_address);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1]))
                                : "memory");
                        }
                        asm volatile("bar.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 1; j_2++) {
                            unsigned int row_in_store = (unsigned int)(j_2 * 8) + warp_in_wg_1 * 2 + lane_read_0_2 / 16;
                            unsigned int row_2 = epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + row_in_store;
                            if (row_2 >= valid_m_1) {
                                break;
                            }
                            unsigned int row_in_atom = (warp_in_wg_1 * 2 + lane_read_0_2 / 16) % 8;
                            unsigned int read_address = output_base + lane_read_0_2 % 16 / 8 * 8 * 128 + row_in_store * 128 + (lane_read_0_2 % 8 ^ row_in_atom) * 16;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                : "r"(read_address));
                            unsigned long long dst_index = ((unsigned long long)cached_topk[j_2] * 1920 + (unsigned long long)cached_token[j_2]) * 2560 + (unsigned long long)(n_offset_1 / 2) + (unsigned long long)(lane_read_0_2 % 16 * 4);
                            reinterpret_cast<int4*>(Combine + dst_index)[0] = reinterpret_cast<int4*>(packed)[0];
                        }
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                }
            }
            if (warp == 8) {
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(64));
            }
            unsigned int epi_warp_2 = warp - 8;
            unsigned int epi_thread = epi_warp_2 * 32 + lane;
            unsigned int phases = 0;
            unsigned long long peer = 0;
            if (bid == 0 && epi_thread == 0) {
                peer = PeerGrid[0];
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (epi_thread == 0) {
                unsigned int increment_2 = 1;
                if (bid == 0) {
                    increment_2 = 2147483497;
                }
                unsigned int _atomic_old_8;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_8) : "l"(&GridCounters[1]), "r"(static_cast<uint32_t>(increment_2)) : "memory");
                unsigned int old_3 = _atomic_old_8;
                unsigned int _wait_acquire_mask_3;
                do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_3) : "l"((reinterpret_cast<unsigned int*>(GridCounters) + (1))) : "memory");
                } while (((_wait_acquire_mask_3 ^ static_cast<unsigned int>((old_3 ^ 2147483648) & 2147483648)) & static_cast<unsigned int>(2147483648)) != 0);
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (bid == 0 && epi_thread == 0) {
                asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(reinterpret_cast<unsigned long long*>(ReadyGrid)), "l"(static_cast<unsigned long long>(peer)) : "memory");
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            uint64_t _grid_id_1;
            asm volatile("mov.u64 %0, %%gridid;" : "=l"(_grid_id_1));
            unsigned long long grid_index = _grid_id_1 + 1;
            unsigned int values[4];
            float reduced[40];
            unsigned int store_values[4];
            #pragma unroll 1
            for (unsigned int token_chunk = epi_warp_2 * 152 + (unsigned int)bid; token_chunk < num_tokens * 4; token_chunk += 1216) {
                unsigned int token_1 = token_chunk / 4;
                unsigned int chunk_1 = token_chunk % 4;
                int expert_12 = -1;
                if (lane < 6) {
                    expert_12 = (int)TopK[token_1 * 6 + lane];
                }
                if (lane == 6) {
                    expert_12 = 6;
                }
                unsigned int _vote_12 = __ballot_sync(0xFFFFFFFF, expert_12 >= 0);
                unsigned int total_mask = _vote_12;
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                {
                bool _warp_acquire_ready;
                do {
                _warp_acquire_ready = true;
                if (lane < 6 && expert_12 >= 0) {
                unsigned long long _warp_acquire_observed;
                asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(_warp_acquire_observed) : "l"(reinterpret_cast<unsigned long long*>(ReadyGrid)) : "memory");
                _warp_acquire_ready = _warp_acquire_observed == static_cast<unsigned long long>(grid_index);
                }
                } while (!__all_sync(static_cast<unsigned>(4294967295), _warp_acquire_ready));
                }
                #elif defined(__CUDA_ARCH__)
                #error "GlobalWarpWaitAcquire requires SM70 or newer"
                #endif
                unsigned int mask = total_mask;
                unsigned int issued = 0;
                #pragma unroll 1
                for (unsigned int i_3 = 0; i_3 < 7; i_3++) {
                    if (mask != 0) {
                        int _ffs_12 = __ffs(mask);
                        unsigned int slot_1 = (unsigned int)(_ffs_12 - 1);
                        mask ^= (unsigned int)1 << slot_1;
                        if (elect_sync()) {
                            cp_async_bulk_gmem2smem(reusable_addr + (epi_warp_2 * 7 + i_3) * 2560, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(CombineBytes) + ((unsigned long long)(((unsigned long long)slot_1 * 1920 + (unsigned long long)token_1) * 5120 * 2 + (unsigned long long)(chunk_1 * 2560)) * (unsigned long long)1)), 2560, combine_barriers_addr + (epi_warp_2 * 7 + i_3) * 8);
                            mbarrier_arrive_expect_tx(combine_barriers_addr + (epi_warp_2 * 7 + i_3) * 8, 2560);
                        }
                        __syncwarp();
                        issued += 1;
                    }
                }
                reduced[0] = 0.0f;
                reduced[1] = 0.0f;
                reduced[2] = 0.0f;
                reduced[3] = 0.0f;
                reduced[4] = 0.0f;
                reduced[5] = 0.0f;
                reduced[6] = 0.0f;
                reduced[7] = 0.0f;
                reduced[8] = 0.0f;
                reduced[9] = 0.0f;
                reduced[10] = 0.0f;
                reduced[11] = 0.0f;
                reduced[12] = 0.0f;
                reduced[13] = 0.0f;
                reduced[14] = 0.0f;
                reduced[15] = 0.0f;
                reduced[16] = 0.0f;
                reduced[17] = 0.0f;
                reduced[18] = 0.0f;
                reduced[19] = 0.0f;
                reduced[20] = 0.0f;
                reduced[21] = 0.0f;
                reduced[22] = 0.0f;
                reduced[23] = 0.0f;
                reduced[24] = 0.0f;
                reduced[25] = 0.0f;
                reduced[26] = 0.0f;
                reduced[27] = 0.0f;
                reduced[28] = 0.0f;
                reduced[29] = 0.0f;
                reduced[30] = 0.0f;
                reduced[31] = 0.0f;
                reduced[32] = 0.0f;
                reduced[33] = 0.0f;
                reduced[34] = 0.0f;
                reduced[35] = 0.0f;
                reduced[36] = 0.0f;
                reduced[37] = 0.0f;
                reduced[38] = 0.0f;
                reduced[39] = 0.0f;
                #pragma unroll 1
                for (unsigned int i_4 = 0; i_4 < 7; i_4++) {
                    if (issued > i_4) {
                        mbarrier_wait(combine_barriers_addr + (epi_warp_2 * 7 + i_4) * 8, phases >> i_4 & 1);
                        #pragma unroll
                        for (unsigned int j_3 = 0; j_3 < 5; j_3++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&values[0])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 3]))
                                : "r"(reusable_addr + (epi_warp_2 * 7 + i_4) * 2560 + (j_3 * 32 + lane) * 16));
                            float _bf16x2_add_f32_0[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(values[0]), "f"(reduced[j_3 * 8]), "f"(reduced[j_3 * 8 + 1]));
                            reduced[j_3 * 8] = _bf16x2_add_f32_0[0];
                            reduced[j_3 * 8 + 1] = _bf16x2_add_f32_0[1];
                            float _bf16x2_add_f32_1[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_1[0]), "=&f"(_bf16x2_add_f32_1[1]) : "r"(values[1]), "f"(reduced[j_3 * 8 + 2]), "f"(reduced[j_3 * 8 + 2 + 1]));
                            reduced[j_3 * 8 + 2] = _bf16x2_add_f32_1[0];
                            reduced[j_3 * 8 + 2 + 1] = _bf16x2_add_f32_1[1];
                            float _bf16x2_add_f32_2[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_2[0]), "=&f"(_bf16x2_add_f32_2[1]) : "r"(values[2]), "f"(reduced[j_3 * 8 + 4]), "f"(reduced[j_3 * 8 + 4 + 1]));
                            reduced[j_3 * 8 + 4] = _bf16x2_add_f32_2[0];
                            reduced[j_3 * 8 + 4 + 1] = _bf16x2_add_f32_2[1];
                            float _bf16x2_add_f32_3[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_3[0]), "=&f"(_bf16x2_add_f32_3[1]) : "r"(values[3]), "f"(reduced[j_3 * 8 + 6]), "f"(reduced[j_3 * 8 + 6 + 1]));
                            reduced[j_3 * 8 + 6] = _bf16x2_add_f32_3[0];
                            reduced[j_3 * 8 + 6 + 1] = _bf16x2_add_f32_3[1];
                        }
                        phases ^= (unsigned int)1 << i_4;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                #pragma unroll
                for (unsigned int j_4 = 0; j_4 < 5; j_4++) {
                    __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(reduced[j_4 * 8], reduced[j_4 * 8 + 1]));
                    store_values[0] = __as_u32(_bf16x2_8);
                    __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(reduced[j_4 * 8 + 2], reduced[j_4 * 8 + 2 + 1]));
                    store_values[1] = __as_u32(_bf16x2_9);
                    __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(reduced[j_4 * 8 + 4], reduced[j_4 * 8 + 4 + 1]));
                    store_values[2] = __as_u32(_bf16x2_10);
                    __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(reduced[j_4 * 8 + 6], reduced[j_4 * 8 + 6 + 1]));
                    store_values[3] = __as_u32(_bf16x2_11);
                    if (j_4 == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        __syncwarp();
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(reusable_addr + (56 + epi_warp_2) * 2560 + (j_4 * 32 + lane) * 16), "r"(*reinterpret_cast<uint32_t*>(&store_values[0])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 3])));
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        void* _cpbulk_dst_4 = reinterpret_cast<void*>(Y + ((unsigned long long)token_1 * 5120 * 2 + (unsigned long long)(chunk_1 * 2560)));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_4), "r"(reusable_addr + (56 + epi_warp_2) * 2560), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                __syncwarp();
            }
        }
    }

    // Cleanup
}

} // extern "C"
