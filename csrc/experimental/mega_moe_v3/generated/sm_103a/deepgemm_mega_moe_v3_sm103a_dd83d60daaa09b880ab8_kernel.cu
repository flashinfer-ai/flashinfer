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
#define SMEM_COMBINE_SMEM_OFF 0
#define SMEM_COMBINE_SMEM_STAGE_BYTES 143360
#define SMEM_COMBINE_SMEM_STRIDE 143360
#define SMEM_SMEM_HIST_OFF 0
#define SMEM_SMEM_HIST_STAGE_BYTES 1536
#define SMEM_SMEM_HIST_STRIDE 1536
#define SMEM_SMEM_DST_ROW_OFF 1536
#define SMEM_SMEM_DST_ROW_STAGE_BYTES 16
#define SMEM_SMEM_DST_ROW_STRIDE 16
#define SMEM_SEND_BUFFER_OFF 2048
#define SMEM_SEND_BUFFER_STAGE_BYTES 20480
#define SMEM_SEND_BUFFER_STRIDE 20480
#define SMEM_SMEM_OUTPUT_OFF 22528
#define SMEM_SMEM_OUTPUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUTPUT_STRIDE 4096
#define SMEM_SMEM_A_OFF 26624
#define SMEM_SMEM_A_STAGE_BYTES 1024
#define SMEM_SMEM_A_STRIDE 1024
#define SMEM_SMEM_B_OFF 37888
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
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
#define SMEM_TOTAL 230400
#define THREADS 512
#define NUM_CTAS 152

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
kernel_deepgemm_mega_moe_v3_sm103a_dd83d60daaa09b880ab8(int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8, unsigned int* __restrict__ x_sf, const __grid_constant__ CUtensorMap A, int* __restrict__ pool_fp8, unsigned int* __restrict__ pool_sf, float* __restrict__ routing_weight_pool, int* __restrict__ token_to_permuted, int* __restrict__ meta_token, int* __restrict__ meta_slot, int* __restrict__ expert_counts, int* __restrict__ expert_row_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ tile_expert, int* __restrict__ tile_m_local, int* __restrict__ total_m_tiles_out, const __grid_constant__ CUtensorMap B1, const __grid_constant__ CUtensorMap SFB1, const __grid_constant__ CUtensorMap I_fp8_w, uint8_t* __restrict__ SF_I_w, __nv_bfloat16* __restrict__ l1_bf16_capture, const __grid_constant__ CUtensorMap A2, unsigned int* __restrict__ SFA2, const __grid_constant__ CUtensorMap B2, const __grid_constant__ CUtensorMap SFB2, __nv_bfloat16* __restrict__ expert_output, __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ histogram_done, unsigned int* __restrict__ prefix_done, unsigned int* __restrict__ dispatch_done, unsigned int* __restrict__ l1_arrival, unsigned int* __restrict__ l2_done, int num_tokens, int top_k, int num_experts, int N1, int K1, int grid_n1, int K1_tiles, int N2, int K2, int grid_n2, int K2_tiles, int total_m_tiles, int M_total, float activation_clamp, unsigned int* __restrict__ PrivateCounters, unsigned long long* __restrict__ PrivateMasks, unsigned int* __restrict__ PublicExpertOutput, unsigned int* __restrict__ PrivateSFBlockOffsets, const __grid_constant__ CUtensorMap PrivateSF1, const __grid_constant__ CUtensorMap PrivateSF2, unsigned int* __restrict__ PrivateSF1Words, uint8_t* __restrict__ PrivateSF2Bytes)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 229696;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 88)
    #define tmem_full_addr (mbar_base + 176)
    #define tmem_empty_addr (mbar_base + 192)
    #define task_full_addr (mbar_base + 208)
    #define task_empty_addr (mbar_base + 224)
    #define combine_barriers_addr (mbar_base + 240)
    #define pull_barriers_addr (mbar_base + 624)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* combine_smem = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int combine_smem_addr = smem + 0;
    int* smem_hist = reinterpret_cast<int*>(smem_raw + 0);
    const int smem_hist_addr = smem + 0;
    int* smem_dst_row = reinterpret_cast<int*>(smem_raw + 1536);
    const int smem_dst_row_addr = smem + 1536;
    uint8_t* send_buffer = reinterpret_cast<uint8_t*>(smem_raw + 2048);
    const int send_buffer_addr = smem + 2048;
    uint8_t* smem_output = reinterpret_cast<uint8_t*>(smem_raw + 22528);
    const int smem_output_addr = smem + 22528;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 26624);
    const int smem_a_addr = smem + 26624;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_b_addr = smem + 37888;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 218112);
    const int smem_sfa_addr = smem + 218112;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 223744);
    const int smem_sfb_addr = smem + 223744;
    float* amax_reduction = reinterpret_cast<float*>(smem_raw + 229376);
    const int amax_reduction_addr = smem + 229376;
    unsigned int* task_infos = reinterpret_cast<unsigned int*>(smem_raw + 229632);
    const int task_infos_addr = smem + 229632;
    {
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B1))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB1))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&I_fp8_w))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A2))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B2))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB2))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&PrivateSF1))) : "memory"); }
        if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&PrivateSF2))) : "memory"); }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 82 barriers)
    // Mbarriers at smem_raw[229696..230352)

    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 11 barriers, init_count=4
            mbarrier_init(smem + 229696, 4);
            mbarrier_init(smem + 229704, 4);
            mbarrier_init(smem + 229712, 4);
            mbarrier_init(smem + 229720, 4);
            mbarrier_init(smem + 229728, 4);
            mbarrier_init(smem + 229736, 4);
            mbarrier_init(smem + 229744, 4);
            mbarrier_init(smem + 229752, 4);
            mbarrier_init(smem + 229760, 4);
            mbarrier_init(smem + 229768, 4);
            mbarrier_init(smem + 229776, 4);
            // empty: 11 barriers, init_count=1
            mbarrier_init(smem + 229784, 1);
            mbarrier_init(smem + 229792, 1);
            mbarrier_init(smem + 229800, 1);
            mbarrier_init(smem + 229808, 1);
            mbarrier_init(smem + 229816, 1);
            mbarrier_init(smem + 229824, 1);
            mbarrier_init(smem + 229832, 1);
            mbarrier_init(smem + 229840, 1);
            mbarrier_init(smem + 229848, 1);
            mbarrier_init(smem + 229856, 1);
            mbarrier_init(smem + 229864, 1);
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 229872, 1);
            mbarrier_init(smem + 229880, 1);
            // tmem_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 229888, 512);
            mbarrier_init(smem + 229896, 512);
            // task_full: 2 barriers, init_count=1
            mbarrier_init(smem + 229904, 1);
            mbarrier_init(smem + 229912, 1);
            // task_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 229920, 512);
            mbarrier_init(smem + 229928, 512);
            // combine_barriers: 48 barriers, init_count=1
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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // pull_barriers: 4 barriers, init_count=1
            mbarrier_init(smem + 230320, 1);
            mbarrier_init(smem + 230328, 1);
            mbarrier_init(smem + 230336, 1);
            mbarrier_init(smem + 230344, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 40 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 230352);
    if (warp == 3) {
        int _tmem_hold = smem + 230352;
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

    // ---- Role: dispatch ----
    if (warp <= 3) {
        { // dispatch_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
            int disp_warp = warp;
            int disp_tid = (unsigned int)(disp_warp * 32) + lane;
            int total_pairs = num_tokens * top_k;
            int hidden_u32 = K1 / 4;
            int sf_words_x = K1 / 128;
            unsigned int sync_tag = (unsigned int)1 << 31;
            unsigned int hist_addend = ((bid == 0) ? sync_tag - (unsigned int)(NUM_CTAS - 1) : (unsigned int)1);
            unsigned int prefix_participants = (unsigned int)NUM_CTAS;
            prefix_participants = prefix_participants + (unsigned int)NUM_CTAS / 2;
            unsigned int prefix_addend = ((bid == 0) ? sync_tag - (prefix_participants - 1) : (unsigned int)1);
            unsigned int done_participants = (unsigned int)NUM_CTAS;
            unsigned int done_addend = ((bid == 0) ? sync_tag - (done_participants - 1) : (unsigned int)1);
            #pragma unroll 1
            for (int i = disp_tid; i < num_experts; i += 128) {
                smem_hist[i] = 0;
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            int grid_stride = num_bids * 128;
            int cta_base = bid * 128;
            #pragma unroll 1
            for (int idx = cta_base + disp_tid; idx < total_pairs; idx += grid_stride) {
                int e_h = topk_idx_i32[idx * 2];
                atomicAdd(&smem_hist[e_h], 1);
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            #pragma unroll 1
            for (int i_1 = disp_tid; i_1 < num_experts; i_1 += 128) {
                int c_h = smem_hist[i_1];
                if (c_h > 0) {
                    atomicAdd(&expert_counts[i_1], c_h);
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            unsigned int hist_ticket = 0;
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_old_0;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_0) : "l"(histogram_done), "r"(static_cast<uint32_t>(hist_addend)) : "memory");
                    hist_ticket = _atomic_old_0;
                    unsigned int _wait_acquire_mask_0;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_0) : "l"(reinterpret_cast<unsigned int*>(histogram_done)) : "memory");
                    } while (((_wait_acquire_mask_0 ^ static_cast<unsigned int>(hist_ticket ^ sync_tag)) & static_cast<unsigned int>(sync_tag)) != 0);
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (bid == 0) {
                if (warp == 0) {
                    unsigned int public_prior = 0;
                    unsigned int source_prior = 0;
                    #pragma unroll 1
                    for (int stripe = 0; stripe < 12; stripe++) {
                        unsigned int expert = (unsigned int)(stripe * 32) + lane;
                        unsigned int prefix_count = 0;
                        if (expert < (unsigned int)num_experts) {
                            prefix_count = (unsigned int)expert_counts[expert];
                        }
                        unsigned int public_tiles = (prefix_count + 256 - 1) / 256 * 2;
                        unsigned int source_blocks = (prefix_count + 16 - 1) / 16;
                        unsigned int public_inclusive = public_tiles;
                        unsigned int source_inclusive = source_blocks;
                        unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 1, 32);
                        unsigned int public_previous = _shfl_up_0;
                        unsigned int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 1, 32);
                        unsigned int source_previous = _shfl_up_1;
                        if (lane >= 1) {
                            public_inclusive += public_previous;
                            source_inclusive += source_previous;
                        }
                        unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 2, 32);
                        unsigned int public_previous_0 = _shfl_up_2;
                        unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 2, 32);
                        unsigned int source_previous_1 = _shfl_up_3;
                        if (lane >= 2) {
                            public_inclusive += public_previous_0;
                            source_inclusive += source_previous_1;
                        }
                        unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 4, 32);
                        unsigned int public_previous_2 = _shfl_up_4;
                        unsigned int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 4, 32);
                        unsigned int source_previous_3 = _shfl_up_5;
                        if (lane >= 4) {
                            public_inclusive += public_previous_2;
                            source_inclusive += source_previous_3;
                        }
                        unsigned int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 8, 32);
                        unsigned int public_previous_4 = _shfl_up_6;
                        unsigned int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 8, 32);
                        unsigned int source_previous_5 = _shfl_up_7;
                        if (lane >= 8) {
                            public_inclusive += public_previous_4;
                            source_inclusive += source_previous_5;
                        }
                        unsigned int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 16, 32);
                        unsigned int public_previous_6 = _shfl_up_8;
                        unsigned int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 16, 32);
                        unsigned int source_previous_7 = _shfl_up_9;
                        if (lane >= 16) {
                            public_inclusive += public_previous_6;
                            source_inclusive += source_previous_7;
                        }
                        unsigned int public_offset = public_prior + public_inclusive - public_tiles;
                        unsigned int source_offset = source_prior + source_inclusive - source_blocks;
                        if (expert < (unsigned int)num_experts) {
                            expert_row_offsets[expert] = (int)(public_offset * 128);
                            PrivateSFBlockOffsets[expert] = source_offset;
                            #pragma unroll 1
                            for (int tt = 0; tt < public_tiles; tt++) {
                                tile_expert[public_offset + (unsigned int)tt] = (int)expert;
                                tile_m_local[public_offset + (unsigned int)tt] = (int)(tt * 128);
                            }
                        }
                        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, public_inclusive, 31);
                        public_prior += _shfl_0;
                        unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, source_inclusive, 31);
                        source_prior += _shfl_1;
                    }
                    if (lane == 0) {
                        total_m_tiles_out[0] = (int)public_prior;
                    }
                }
                if (warp == 0) {
                    __syncwarp();
                }
            }
            unsigned int prefix_ticket = 0;
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_old_1;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_1) : "l"(prefix_done), "r"(static_cast<uint32_t>(prefix_addend)) : "memory");
                    prefix_ticket = _atomic_old_1;
                    unsigned int _wait_acquire_mask_1;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_1) : "l"(reinterpret_cast<unsigned int*>(prefix_done)) : "memory");
                    } while (((_wait_acquire_mask_1 ^ static_cast<unsigned int>(prefix_ticket ^ sync_tag)) & static_cast<unsigned int>(sync_tag)) != 0);
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            unsigned int pull_phase = 0;
            int global_warp_idx = bid * 4 + disp_warp;
            int warps_per_grid = NUM_CTAS * 4;
            #pragma unroll 1
            for (int pair_idx = global_warp_idx; pair_idx < total_pairs; pair_idx += warps_per_grid) {
                int e_s = topk_idx_i32[pair_idx * 2];
                int t_s = pair_idx / top_k;
                int k_s = pair_idx - t_s * top_k;
                if (lane == 0) {
                    float cached_rw = topk_weights[pair_idx];
                    int _atomic_old_2 = atomicAdd(&expert_scatter_offsets[e_s], 1);
                    int claim = _atomic_old_2;
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
                if (elect_sync()) {
                    cp_async_bulk_gmem2smem(send_buffer_addr + (unsigned int)(disp_warp * 5120), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x_fp8) + ((unsigned long long)src_u32_off * (unsigned long long)4)), 5120, pull_barriers_addr + (disp_warp) * 8);
                    mbarrier_arrive_expect_tx(pull_barriers_addr + (disp_warp) * 8, 5120);
                }
                __syncwarp();
                unsigned long long src_sf_off = (unsigned long long)t_s * (unsigned long long)sf_words_x;
                unsigned long long dst_sf_off = (unsigned long long)dst_row * (unsigned long long)sf_words_x;
                unsigned int in_expert = (unsigned int)(dst_row - expert_row_offsets[e_s]);
                unsigned int sf_block = PrivateSFBlockOffsets[e_s] + in_expert / 16;
                unsigned int source_sf_token = sf_block * 128 + in_expert % 16 * 4;
                #pragma unroll
                for (int sf_chunk = 0; sf_chunk < 2; sf_chunk++) {
                    int w = (unsigned int)(sf_chunk * 32) + lane;
                    if (w < 40) {
                        unsigned int sf_word = x_sf[src_sf_off + (unsigned long long)w];
                        pool_sf[dst_sf_off + (unsigned long long)w] = sf_word;
                        PrivateSF1Words[(unsigned int)(w * 49920) + source_sf_token] = sf_word;
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_wait(pull_barriers_addr + (disp_warp) * 8, pull_phase);
                    pull_phase ^= 1;
                    {
                        void* _cpbulk_dst_0 = reinterpret_cast<void*>(pool_fp8 + dst_u32_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_0), "r"(send_buffer_addr + (unsigned int)(disp_warp * 5120)), "r"((uint32_t)(5120))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __threadfence();
                __syncwarp();
                if (elect_sync()) {
                    unsigned int arrivals = 1;
                    if (in_expert + 1 == (unsigned int)expert_counts[e_s]) {
                        arrivals = 16 - in_expert % 16;
                    }
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (782 + sf_block))), "r"(static_cast<unsigned int>(arrivals)) : "memory");
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            __threadfence_system();
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_old_3;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_3) : "l"(dispatch_done), "r"(static_cast<uint32_t>(done_addend)) : "memory");
                }
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (bid == 0) {
                #pragma unroll 1
                for (int i_2 = disp_tid; i_2 < num_experts; i_2 += 128) {
                    expert_counts[i_2] = 0;
                    expert_scatter_offsets[i_2] = 0;
                }
                if (disp_tid == 0) {
                    PrivateCounters[0] = (unsigned int)0;
                    PrivateCounters[1] = (unsigned int)0;
                }
            } else {
                #pragma unroll 1
                for (int block = (bid - 1) * 128 + disp_tid; block < 390; block += (NUM_CTAS - 1) * 128) {
                    PrivateMasks[block] = (unsigned long long)0;
                    PrivateCounters[782 + block] = (unsigned int)0;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 4) {
        { // load_a_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int task[8];
            unsigned int a_stage = 0;
            int task_iteration = 0;
            unsigned int _phase_empty = 1;
            while (1) {
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
                unsigned int public_m = (unsigned int)expert_row_offsets[task[1]] + task[2] * 16;
                unsigned int aligned_m = (task[5] + 15) / 16 * 16;
                unsigned int token_offset = public_m + (unsigned int)cta_rank * (aligned_m / 2);
                unsigned long long pending = 68719476735;
                if (task[0] == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (782 + task[4]))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(16)) >= static_cast<unsigned int>(1));
                    }
                }
                #pragma unroll 2
                for (int k = 0; k < task[7] / 128; k++) {
                    if (task[0] == 2) {
                        unsigned long long k_mask = (unsigned long long)3 << (unsigned long long)(k * 2);
                        if ((pending & k_mask) != 0) {
                            unsigned long long _wait_acquire_mask_3;
                            do {
                            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_wait_acquire_mask_3) : "l"((reinterpret_cast<unsigned long long*>(PrivateMasks) + (task[4]))) : "memory");
                            } while (((_wait_acquire_mask_3 ^ static_cast<unsigned long long>((unsigned long long)68719476735)) & static_cast<unsigned long long>(k_mask)) != 0);
                            unsigned long long observed = _wait_acquire_mask_3;
                            pending = observed ^ 68719476735;
                        }
                    }
                    mbarrier_wait(empty_addr + (a_stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        if (task[0] == 1) {
                            tma_2d_gmem2smem_cta2(smem_a_addr + a_stage * 1024, (&A), k * 128, token_offset, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 512, (&PrivateSF1), task[4] * 128, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        } else {
                            tma_2d_gmem2smem_cta2(smem_a_addr + a_stage * 1024, (&A2), k * 128, token_offset, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 512, (&PrivateSF2), task[4] * 128, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        }
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
                task_iteration += 1;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 5) {
        { // load_b_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int task_1[8];
            unsigned int b_stage = 0;
            int task_iteration_1 = 0;
            unsigned int _phase_empty_1 = 1;
            while (1) {
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
                const void* selected_weight_map = ((task_1[0] == 1) ? ((&B1)) : ((&B2)));
                const void* selected_weight_sf_map = ((task_1[0] == 1) ? ((&SFB1)) : ((&SFB2)));
                unsigned int n_offset = (task_1[3] * 2 + (unsigned int)cta_rank) * 128;
                #pragma unroll 2
                for (int k_1 = 0; k_1 < task_1[7] / 128; k_1++) {
                    mbarrier_wait(empty_addr + (b_stage) * 8, _phase_empty_1);
                    unsigned int sf_k_offset = task_1[1] * (task_1[7] / 128) + (unsigned int)k_1;
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_b_addr + b_stage * 16384, selected_weight_map, k_1 * 128, task_1[1] * task_1[6] + n_offset, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfb_addr + b_stage * 512, selected_weight_sf_map, n_offset, sf_k_offset, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (b_stage) * 8, 17408);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (b_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    __syncwarp();
                    b_stage += 1;
                    if (b_stage == 11) { b_stage = 0; _phase_empty_1 ^= 1; }
                }
                task_iteration_1 += 1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        { // mma_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int task_2[8];
            unsigned int mma_stage = 0;
            unsigned int completed_tasks = 0;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                int task_iteration_2 = 0;
                while (1) {
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
                    unsigned int aligned_m_1 = (task_2[5] + 15) / 16 * 16;
                    mbarrier_wait(tmem_empty_addr + (accum_stage) * 8, task_iteration_2 / 2 & 1 ^ 1);
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
                            uint32_t _election_operand_2_3 = ((((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((0) << 29) | ((0) << 4)));
                            uint32_t _election_operand_2_4 = (tmem_scale_b);
                            uint32_t _election_operand_2_5 = (tmem_scale_a);
                            uint32_t _election_operand_2_6 = (((init_flag) ? 0 : 1));
                            asm volatile("" :: "r"(_election_operand_2_0), "l"(_election_operand_2_1), "l"(_election_operand_2_2), "r"(_election_operand_2_3), "r"(_election_operand_2_4), "r"(_election_operand_2_5), "r"(_election_operand_2_6));
                            uint32_t _election_operand_3_0 = ((tmem_accum + (accum_stage * 16)));
                            uint64_t _election_operand_3_1 = (a_desc + 2);
                            uint64_t _election_operand_3_2 = (b_desc + 2);
                            uint32_t _election_operand_3_3 = ((((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((1) << 29) | ((1) << 4)));
                            uint32_t _election_operand_3_4 = (tmem_scale_b);
                            uint32_t _election_operand_3_5 = (tmem_scale_a);
                            uint32_t _election_operand_3_6 = (1);
                            asm volatile("" :: "r"(_election_operand_3_0), "l"(_election_operand_3_1), "l"(_election_operand_3_2), "r"(_election_operand_3_3), "r"(_election_operand_3_4), "r"(_election_operand_3_5), "r"(_election_operand_3_6));
                            uint32_t _election_operand_4_0 = ((tmem_accum + (accum_stage * 16)));
                            uint64_t _election_operand_4_1 = (a_desc + 4);
                            uint64_t _election_operand_4_2 = (b_desc + 4);
                            uint32_t _election_operand_4_3 = ((((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((2) << 29) | ((2) << 4)));
                            uint32_t _election_operand_4_4 = (tmem_scale_b);
                            uint32_t _election_operand_4_5 = (tmem_scale_a);
                            uint32_t _election_operand_4_6 = (1);
                            asm volatile("" :: "r"(_election_operand_4_0), "l"(_election_operand_4_1), "l"(_election_operand_4_2), "r"(_election_operand_4_3), "r"(_election_operand_4_4), "r"(_election_operand_4_5), "r"(_election_operand_4_6));
                            uint32_t _election_operand_5_0 = ((tmem_accum + (accum_stage * 16)));
                            uint64_t _election_operand_5_1 = (a_desc + 6);
                            uint64_t _election_operand_5_2 = (b_desc + 6);
                            uint32_t _election_operand_5_3 = ((((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((3) << 29) | ((3) << 4)));
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
                    task_iteration_2 += 1;
                }
                if (completed_tasks > 0) {
                    mbarrier_wait(tmem_empty_addr + ((completed_tasks - 1) % 2) * 8, (completed_tasks - 1) / 2 & 1);
                }
            }
        }
    }
    // ---- Role: schedule ----
    if (warp == 7) {
        { // schedule_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
            unsigned int counts[12];
            unsigned int state[2];
            unsigned int task_3[8];
            unsigned int schedule_iteration = 0;
            unsigned int sched_tag = (unsigned int)1 << 31;
            unsigned int sched_ticket = 0;
            if (cta_rank == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_old_4;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_4) : "l"(prefix_done), "r"(static_cast<uint32_t>(1)) : "memory");
                    sched_ticket = _atomic_old_4;
                    unsigned int _wait_acquire_mask_2;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_2) : "l"(reinterpret_cast<unsigned int*>(prefix_done)) : "memory");
                    } while (((_wait_acquire_mask_2 ^ static_cast<unsigned int>(sched_ticket ^ sched_tag)) & static_cast<unsigned int>(sched_tag)) != 0);
                }
                __syncwarp();
                unsigned int expert_1 = lane;
                counts[0] = 0;
                if (expert_1 < 384) {
                    counts[0] = (unsigned int)expert_counts[expert_1];
                }
                unsigned int expert_0 = 32 + lane;
                counts[1] = 0;
                if (expert_0 < 384) {
                    counts[1] = (unsigned int)expert_counts[expert_0];
                }
                unsigned int expert_1_1 = 64 + lane;
                counts[2] = 0;
                if (expert_1_1 < 384) {
                    counts[2] = (unsigned int)expert_counts[expert_1_1];
                }
                unsigned int expert_2 = 96 + lane;
                counts[3] = 0;
                if (expert_2 < 384) {
                    counts[3] = (unsigned int)expert_counts[expert_2];
                }
                unsigned int expert_3 = 128 + lane;
                counts[4] = 0;
                if (expert_3 < 384) {
                    counts[4] = (unsigned int)expert_counts[expert_3];
                }
                unsigned int expert_4 = 160 + lane;
                counts[5] = 0;
                if (expert_4 < 384) {
                    counts[5] = (unsigned int)expert_counts[expert_4];
                }
                unsigned int expert_5 = 192 + lane;
                counts[6] = 0;
                if (expert_5 < 384) {
                    counts[6] = (unsigned int)expert_counts[expert_5];
                }
                unsigned int expert_6 = 224 + lane;
                counts[7] = 0;
                if (expert_6 < 384) {
                    counts[7] = (unsigned int)expert_counts[expert_6];
                }
                unsigned int expert_7 = 256 + lane;
                counts[8] = 0;
                if (expert_7 < 384) {
                    counts[8] = (unsigned int)expert_counts[expert_7];
                }
                unsigned int expert_8 = 288 + lane;
                counts[9] = 0;
                if (expert_8 < 384) {
                    counts[9] = (unsigned int)expert_counts[expert_8];
                }
                unsigned int expert_9 = 320 + lane;
                counts[10] = 0;
                if (expert_9 < 384) {
                    counts[10] = (unsigned int)expert_counts[expert_9];
                }
                unsigned int expert_10 = 352 + lane;
                counts[11] = 0;
                if (expert_10 < 384) {
                    counts[11] = (unsigned int)expert_counts[expert_10];
                }
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
                for (int iteration = 0; iteration < 14821; iteration++) {
                    mbarrier_wait(task_empty_addr + (schedule_iteration % 2) * 8, schedule_iteration / 2 & 1 ^ 1);
                    unsigned int phase = 0;
                    unsigned int claimed = 0;
                    unsigned int clusters = 18;
                    unsigned int shape_n = 4608;
                    unsigned int shape_k = 5120;
                    if (state[1] != 4294967295 && state[1] != 0) {
                        state[1] = state[1] - 1;
                        unsigned int claimed_0 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_5 = atomicAdd(PrivateCounters, 1);
                            claimed_0 = _atomic_old_5;
                        }
                        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, claimed_0, 0);
                        claimed = _shfl_2;
                        if (claimed >= state[0] * 18) {
                            state[1] = 4294967295;
                        } else {
                            phase = 1;
                        }
                    }
                    if (phase == 0) {
                        unsigned int claimed_0_1 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_6 = atomicAdd(PrivateCounters + 1, 1);
                            claimed_0_1 = _atomic_old_6;
                        }
                        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, claimed_0_1, 0);
                        claimed = _shfl_3;
                        if (claimed < state[0] * 20) {
                            if (state[1] != 4294967295) {
                                state[1] = 1;
                            }
                            phase = 2;
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
                    if (phase != 0) {
                        unsigned int pool_block = ((phase == 1) ? claimed / 18 : claimed / 20);
                        task_3[0] = phase;
                        task_3[1] = 0;
                        task_3[2] = 0;
                        task_3[3] = ((phase == 1) ? claimed % 18 : claimed % 20);
                        task_3[4] = pool_block;
                        task_3[5] = 0;
                        task_3[6] = shape_n;
                        task_3[7] = shape_k;
                        unsigned int block_offset = 0;
                        unsigned int expert_11 = lane;
                        unsigned int tokens = counts[0];
                        unsigned int blocks = (tokens + 16 - 1) / 16;
                        unsigned int inclusive = blocks;
                        unsigned int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
                        unsigned int previous = _shfl_up_10;
                        if (lane >= 1) {
                            inclusive += previous;
                        }
                        unsigned int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
                        unsigned int previous_12 = _shfl_up_11;
                        if (lane >= 2) {
                            inclusive += previous_12;
                        }
                        unsigned int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
                        unsigned int previous_13 = _shfl_up_12;
                        if (lane >= 4) {
                            inclusive += previous_13;
                        }
                        unsigned int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
                        unsigned int previous_14 = _shfl_up_13;
                        if (lane >= 8) {
                            inclusive += previous_14;
                        }
                        unsigned int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
                        unsigned int previous_15 = _shfl_up_14;
                        if (lane >= 16) {
                            inclusive += previous_15;
                        }
                        unsigned int lane_offset = block_offset + inclusive - blocks;
                        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, expert_11 < 384 && (pool_block >= lane_offset && pool_block < lane_offset + blocks));
                        unsigned int owner_mask = _vote_0;
                        if (owner_mask != 0) {
                            int _ffs_0 = __ffs(owner_mask);
                            unsigned int owner_lane = _ffs_0 - 1;
                            unsigned int local_block = pool_block - lane_offset;
                            unsigned int _min_1 = ((tokens - local_block * 16) < (16) ? (tokens - local_block * 16) : (16));
                            unsigned int valid = _min_1;
                            unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, expert_11, owner_lane);
                            task_3[1] = _shfl_4;
                            unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, local_block, owner_lane);
                            task_3[2] = _shfl_5;
                            unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, valid, owner_lane);
                            task_3[5] = _shfl_6;
                        }
                        unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
                        block_offset += _shfl_7;
                        unsigned int expert_16 = 32 + lane;
                        unsigned int tokens_17 = counts[1];
                        unsigned int blocks_18 = (tokens_17 + 16 - 1) / 16;
                        unsigned int inclusive_19 = blocks_18;
                        unsigned int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, inclusive_19, 1, 32);
                        unsigned int previous_20 = _shfl_up_15;
                        if (lane >= 1) {
                            inclusive_19 += previous_20;
                        }
                        unsigned int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, inclusive_19, 2, 32);
                        unsigned int previous_21 = _shfl_up_16;
                        if (lane >= 2) {
                            inclusive_19 += previous_21;
                        }
                        unsigned int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, inclusive_19, 4, 32);
                        unsigned int previous_22 = _shfl_up_17;
                        if (lane >= 4) {
                            inclusive_19 += previous_22;
                        }
                        unsigned int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, inclusive_19, 8, 32);
                        unsigned int previous_23 = _shfl_up_18;
                        if (lane >= 8) {
                            inclusive_19 += previous_23;
                        }
                        unsigned int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, inclusive_19, 16, 32);
                        unsigned int previous_24 = _shfl_up_19;
                        if (lane >= 16) {
                            inclusive_19 += previous_24;
                        }
                        unsigned int lane_offset_25 = block_offset + inclusive_19 - blocks_18;
                        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, expert_16 < 384 && (pool_block >= lane_offset_25 && pool_block < lane_offset_25 + blocks_18));
                        unsigned int owner_mask_26 = _vote_1;
                        if (owner_mask_26 != 0) {
                            int _ffs_1 = __ffs(owner_mask_26);
                            unsigned int owner_lane_1 = _ffs_1 - 1;
                            unsigned int local_block_1 = pool_block - lane_offset_25;
                            unsigned int _min_2 = ((tokens_17 - local_block_1 * 16) < (16) ? (tokens_17 - local_block_1 * 16) : (16));
                            unsigned int valid_1 = _min_2;
                            unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, expert_16, owner_lane_1);
                            task_3[1] = _shfl_8;
                            unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, local_block_1, owner_lane_1);
                            task_3[2] = _shfl_9;
                            unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, valid_1, owner_lane_1);
                            task_3[5] = _shfl_10;
                        }
                        unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, inclusive_19, 31);
                        block_offset += _shfl_11;
                        unsigned int expert_27 = 64 + lane;
                        unsigned int tokens_28 = counts[2];
                        unsigned int blocks_29 = (tokens_28 + 16 - 1) / 16;
                        unsigned int inclusive_30 = blocks_29;
                        unsigned int _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, inclusive_30, 1, 32);
                        unsigned int previous_31 = _shfl_up_20;
                        if (lane >= 1) {
                            inclusive_30 += previous_31;
                        }
                        unsigned int _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, inclusive_30, 2, 32);
                        unsigned int previous_32 = _shfl_up_21;
                        if (lane >= 2) {
                            inclusive_30 += previous_32;
                        }
                        unsigned int _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, inclusive_30, 4, 32);
                        unsigned int previous_33 = _shfl_up_22;
                        if (lane >= 4) {
                            inclusive_30 += previous_33;
                        }
                        unsigned int _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, inclusive_30, 8, 32);
                        unsigned int previous_34 = _shfl_up_23;
                        if (lane >= 8) {
                            inclusive_30 += previous_34;
                        }
                        unsigned int _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, inclusive_30, 16, 32);
                        unsigned int previous_35 = _shfl_up_24;
                        if (lane >= 16) {
                            inclusive_30 += previous_35;
                        }
                        unsigned int lane_offset_36 = block_offset + inclusive_30 - blocks_29;
                        unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, expert_27 < 384 && (pool_block >= lane_offset_36 && pool_block < lane_offset_36 + blocks_29));
                        unsigned int owner_mask_37 = _vote_2;
                        if (owner_mask_37 != 0) {
                            int _ffs_2 = __ffs(owner_mask_37);
                            unsigned int owner_lane_2 = _ffs_2 - 1;
                            unsigned int local_block_2 = pool_block - lane_offset_36;
                            unsigned int _min_3 = ((tokens_28 - local_block_2 * 16) < (16) ? (tokens_28 - local_block_2 * 16) : (16));
                            unsigned int valid_2 = _min_3;
                            unsigned int _shfl_12 = __shfl_sync(0xFFFFFFFF, expert_27, owner_lane_2);
                            task_3[1] = _shfl_12;
                            unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, local_block_2, owner_lane_2);
                            task_3[2] = _shfl_13;
                            unsigned int _shfl_14 = __shfl_sync(0xFFFFFFFF, valid_2, owner_lane_2);
                            task_3[5] = _shfl_14;
                        }
                        unsigned int _shfl_15 = __shfl_sync(0xFFFFFFFF, inclusive_30, 31);
                        block_offset += _shfl_15;
                        unsigned int expert_38 = 96 + lane;
                        unsigned int tokens_39 = counts[3];
                        unsigned int blocks_40 = (tokens_39 + 16 - 1) / 16;
                        unsigned int inclusive_41 = blocks_40;
                        unsigned int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, inclusive_41, 1, 32);
                        unsigned int previous_42 = _shfl_up_25;
                        if (lane >= 1) {
                            inclusive_41 += previous_42;
                        }
                        unsigned int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, inclusive_41, 2, 32);
                        unsigned int previous_43 = _shfl_up_26;
                        if (lane >= 2) {
                            inclusive_41 += previous_43;
                        }
                        unsigned int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, inclusive_41, 4, 32);
                        unsigned int previous_44 = _shfl_up_27;
                        if (lane >= 4) {
                            inclusive_41 += previous_44;
                        }
                        unsigned int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, inclusive_41, 8, 32);
                        unsigned int previous_45 = _shfl_up_28;
                        if (lane >= 8) {
                            inclusive_41 += previous_45;
                        }
                        unsigned int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, inclusive_41, 16, 32);
                        unsigned int previous_46 = _shfl_up_29;
                        if (lane >= 16) {
                            inclusive_41 += previous_46;
                        }
                        unsigned int lane_offset_47 = block_offset + inclusive_41 - blocks_40;
                        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, expert_38 < 384 && (pool_block >= lane_offset_47 && pool_block < lane_offset_47 + blocks_40));
                        unsigned int owner_mask_48 = _vote_3;
                        if (owner_mask_48 != 0) {
                            int _ffs_3 = __ffs(owner_mask_48);
                            unsigned int owner_lane_3 = _ffs_3 - 1;
                            unsigned int local_block_3 = pool_block - lane_offset_47;
                            unsigned int _min_4 = ((tokens_39 - local_block_3 * 16) < (16) ? (tokens_39 - local_block_3 * 16) : (16));
                            unsigned int valid_3 = _min_4;
                            unsigned int _shfl_16 = __shfl_sync(0xFFFFFFFF, expert_38, owner_lane_3);
                            task_3[1] = _shfl_16;
                            unsigned int _shfl_17 = __shfl_sync(0xFFFFFFFF, local_block_3, owner_lane_3);
                            task_3[2] = _shfl_17;
                            unsigned int _shfl_18 = __shfl_sync(0xFFFFFFFF, valid_3, owner_lane_3);
                            task_3[5] = _shfl_18;
                        }
                        unsigned int _shfl_19 = __shfl_sync(0xFFFFFFFF, inclusive_41, 31);
                        block_offset += _shfl_19;
                        unsigned int expert_49 = 128 + lane;
                        unsigned int tokens_50 = counts[4];
                        unsigned int blocks_51 = (tokens_50 + 16 - 1) / 16;
                        unsigned int inclusive_52 = blocks_51;
                        unsigned int _shfl_up_30 = __shfl_up_sync(0xFFFFFFFF, inclusive_52, 1, 32);
                        unsigned int previous_53 = _shfl_up_30;
                        if (lane >= 1) {
                            inclusive_52 += previous_53;
                        }
                        unsigned int _shfl_up_31 = __shfl_up_sync(0xFFFFFFFF, inclusive_52, 2, 32);
                        unsigned int previous_54 = _shfl_up_31;
                        if (lane >= 2) {
                            inclusive_52 += previous_54;
                        }
                        unsigned int _shfl_up_32 = __shfl_up_sync(0xFFFFFFFF, inclusive_52, 4, 32);
                        unsigned int previous_55 = _shfl_up_32;
                        if (lane >= 4) {
                            inclusive_52 += previous_55;
                        }
                        unsigned int _shfl_up_33 = __shfl_up_sync(0xFFFFFFFF, inclusive_52, 8, 32);
                        unsigned int previous_56 = _shfl_up_33;
                        if (lane >= 8) {
                            inclusive_52 += previous_56;
                        }
                        unsigned int _shfl_up_34 = __shfl_up_sync(0xFFFFFFFF, inclusive_52, 16, 32);
                        unsigned int previous_57 = _shfl_up_34;
                        if (lane >= 16) {
                            inclusive_52 += previous_57;
                        }
                        unsigned int lane_offset_58 = block_offset + inclusive_52 - blocks_51;
                        unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, expert_49 < 384 && (pool_block >= lane_offset_58 && pool_block < lane_offset_58 + blocks_51));
                        unsigned int owner_mask_59 = _vote_4;
                        if (owner_mask_59 != 0) {
                            int _ffs_4 = __ffs(owner_mask_59);
                            unsigned int owner_lane_4 = _ffs_4 - 1;
                            unsigned int local_block_4 = pool_block - lane_offset_58;
                            unsigned int _min_5 = ((tokens_50 - local_block_4 * 16) < (16) ? (tokens_50 - local_block_4 * 16) : (16));
                            unsigned int valid_4 = _min_5;
                            unsigned int _shfl_20 = __shfl_sync(0xFFFFFFFF, expert_49, owner_lane_4);
                            task_3[1] = _shfl_20;
                            unsigned int _shfl_21 = __shfl_sync(0xFFFFFFFF, local_block_4, owner_lane_4);
                            task_3[2] = _shfl_21;
                            unsigned int _shfl_22 = __shfl_sync(0xFFFFFFFF, valid_4, owner_lane_4);
                            task_3[5] = _shfl_22;
                        }
                        unsigned int _shfl_23 = __shfl_sync(0xFFFFFFFF, inclusive_52, 31);
                        block_offset += _shfl_23;
                        unsigned int expert_60 = 160 + lane;
                        unsigned int tokens_61 = counts[5];
                        unsigned int blocks_62 = (tokens_61 + 16 - 1) / 16;
                        unsigned int inclusive_63 = blocks_62;
                        unsigned int _shfl_up_35 = __shfl_up_sync(0xFFFFFFFF, inclusive_63, 1, 32);
                        unsigned int previous_64 = _shfl_up_35;
                        if (lane >= 1) {
                            inclusive_63 += previous_64;
                        }
                        unsigned int _shfl_up_36 = __shfl_up_sync(0xFFFFFFFF, inclusive_63, 2, 32);
                        unsigned int previous_65 = _shfl_up_36;
                        if (lane >= 2) {
                            inclusive_63 += previous_65;
                        }
                        unsigned int _shfl_up_37 = __shfl_up_sync(0xFFFFFFFF, inclusive_63, 4, 32);
                        unsigned int previous_66 = _shfl_up_37;
                        if (lane >= 4) {
                            inclusive_63 += previous_66;
                        }
                        unsigned int _shfl_up_38 = __shfl_up_sync(0xFFFFFFFF, inclusive_63, 8, 32);
                        unsigned int previous_67 = _shfl_up_38;
                        if (lane >= 8) {
                            inclusive_63 += previous_67;
                        }
                        unsigned int _shfl_up_39 = __shfl_up_sync(0xFFFFFFFF, inclusive_63, 16, 32);
                        unsigned int previous_68 = _shfl_up_39;
                        if (lane >= 16) {
                            inclusive_63 += previous_68;
                        }
                        unsigned int lane_offset_69 = block_offset + inclusive_63 - blocks_62;
                        unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, expert_60 < 384 && (pool_block >= lane_offset_69 && pool_block < lane_offset_69 + blocks_62));
                        unsigned int owner_mask_70 = _vote_5;
                        if (owner_mask_70 != 0) {
                            int _ffs_5 = __ffs(owner_mask_70);
                            unsigned int owner_lane_5 = _ffs_5 - 1;
                            unsigned int local_block_5 = pool_block - lane_offset_69;
                            unsigned int _min_6 = ((tokens_61 - local_block_5 * 16) < (16) ? (tokens_61 - local_block_5 * 16) : (16));
                            unsigned int valid_5 = _min_6;
                            unsigned int _shfl_24 = __shfl_sync(0xFFFFFFFF, expert_60, owner_lane_5);
                            task_3[1] = _shfl_24;
                            unsigned int _shfl_25 = __shfl_sync(0xFFFFFFFF, local_block_5, owner_lane_5);
                            task_3[2] = _shfl_25;
                            unsigned int _shfl_26 = __shfl_sync(0xFFFFFFFF, valid_5, owner_lane_5);
                            task_3[5] = _shfl_26;
                        }
                        unsigned int _shfl_27 = __shfl_sync(0xFFFFFFFF, inclusive_63, 31);
                        block_offset += _shfl_27;
                        unsigned int expert_71 = 192 + lane;
                        unsigned int tokens_72 = counts[6];
                        unsigned int blocks_73 = (tokens_72 + 16 - 1) / 16;
                        unsigned int inclusive_74 = blocks_73;
                        unsigned int _shfl_up_40 = __shfl_up_sync(0xFFFFFFFF, inclusive_74, 1, 32);
                        unsigned int previous_75 = _shfl_up_40;
                        if (lane >= 1) {
                            inclusive_74 += previous_75;
                        }
                        unsigned int _shfl_up_41 = __shfl_up_sync(0xFFFFFFFF, inclusive_74, 2, 32);
                        unsigned int previous_76 = _shfl_up_41;
                        if (lane >= 2) {
                            inclusive_74 += previous_76;
                        }
                        unsigned int _shfl_up_42 = __shfl_up_sync(0xFFFFFFFF, inclusive_74, 4, 32);
                        unsigned int previous_77 = _shfl_up_42;
                        if (lane >= 4) {
                            inclusive_74 += previous_77;
                        }
                        unsigned int _shfl_up_43 = __shfl_up_sync(0xFFFFFFFF, inclusive_74, 8, 32);
                        unsigned int previous_78 = _shfl_up_43;
                        if (lane >= 8) {
                            inclusive_74 += previous_78;
                        }
                        unsigned int _shfl_up_44 = __shfl_up_sync(0xFFFFFFFF, inclusive_74, 16, 32);
                        unsigned int previous_79 = _shfl_up_44;
                        if (lane >= 16) {
                            inclusive_74 += previous_79;
                        }
                        unsigned int lane_offset_80 = block_offset + inclusive_74 - blocks_73;
                        unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, expert_71 < 384 && (pool_block >= lane_offset_80 && pool_block < lane_offset_80 + blocks_73));
                        unsigned int owner_mask_81 = _vote_6;
                        if (owner_mask_81 != 0) {
                            int _ffs_6 = __ffs(owner_mask_81);
                            unsigned int owner_lane_6 = _ffs_6 - 1;
                            unsigned int local_block_6 = pool_block - lane_offset_80;
                            unsigned int _min_7 = ((tokens_72 - local_block_6 * 16) < (16) ? (tokens_72 - local_block_6 * 16) : (16));
                            unsigned int valid_6 = _min_7;
                            unsigned int _shfl_28 = __shfl_sync(0xFFFFFFFF, expert_71, owner_lane_6);
                            task_3[1] = _shfl_28;
                            unsigned int _shfl_29 = __shfl_sync(0xFFFFFFFF, local_block_6, owner_lane_6);
                            task_3[2] = _shfl_29;
                            unsigned int _shfl_30 = __shfl_sync(0xFFFFFFFF, valid_6, owner_lane_6);
                            task_3[5] = _shfl_30;
                        }
                        unsigned int _shfl_31 = __shfl_sync(0xFFFFFFFF, inclusive_74, 31);
                        block_offset += _shfl_31;
                        unsigned int expert_82 = 224 + lane;
                        unsigned int tokens_83 = counts[7];
                        unsigned int blocks_84 = (tokens_83 + 16 - 1) / 16;
                        unsigned int inclusive_85 = blocks_84;
                        unsigned int _shfl_up_45 = __shfl_up_sync(0xFFFFFFFF, inclusive_85, 1, 32);
                        unsigned int previous_86 = _shfl_up_45;
                        if (lane >= 1) {
                            inclusive_85 += previous_86;
                        }
                        unsigned int _shfl_up_46 = __shfl_up_sync(0xFFFFFFFF, inclusive_85, 2, 32);
                        unsigned int previous_87 = _shfl_up_46;
                        if (lane >= 2) {
                            inclusive_85 += previous_87;
                        }
                        unsigned int _shfl_up_47 = __shfl_up_sync(0xFFFFFFFF, inclusive_85, 4, 32);
                        unsigned int previous_88 = _shfl_up_47;
                        if (lane >= 4) {
                            inclusive_85 += previous_88;
                        }
                        unsigned int _shfl_up_48 = __shfl_up_sync(0xFFFFFFFF, inclusive_85, 8, 32);
                        unsigned int previous_89 = _shfl_up_48;
                        if (lane >= 8) {
                            inclusive_85 += previous_89;
                        }
                        unsigned int _shfl_up_49 = __shfl_up_sync(0xFFFFFFFF, inclusive_85, 16, 32);
                        unsigned int previous_90 = _shfl_up_49;
                        if (lane >= 16) {
                            inclusive_85 += previous_90;
                        }
                        unsigned int lane_offset_91 = block_offset + inclusive_85 - blocks_84;
                        unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, expert_82 < 384 && (pool_block >= lane_offset_91 && pool_block < lane_offset_91 + blocks_84));
                        unsigned int owner_mask_92 = _vote_7;
                        if (owner_mask_92 != 0) {
                            int _ffs_7 = __ffs(owner_mask_92);
                            unsigned int owner_lane_7 = _ffs_7 - 1;
                            unsigned int local_block_7 = pool_block - lane_offset_91;
                            unsigned int _min_8 = ((tokens_83 - local_block_7 * 16) < (16) ? (tokens_83 - local_block_7 * 16) : (16));
                            unsigned int valid_7 = _min_8;
                            unsigned int _shfl_32 = __shfl_sync(0xFFFFFFFF, expert_82, owner_lane_7);
                            task_3[1] = _shfl_32;
                            unsigned int _shfl_33 = __shfl_sync(0xFFFFFFFF, local_block_7, owner_lane_7);
                            task_3[2] = _shfl_33;
                            unsigned int _shfl_34 = __shfl_sync(0xFFFFFFFF, valid_7, owner_lane_7);
                            task_3[5] = _shfl_34;
                        }
                        unsigned int _shfl_35 = __shfl_sync(0xFFFFFFFF, inclusive_85, 31);
                        block_offset += _shfl_35;
                        unsigned int expert_93 = 256 + lane;
                        unsigned int tokens_94 = counts[8];
                        unsigned int blocks_95 = (tokens_94 + 16 - 1) / 16;
                        unsigned int inclusive_96 = blocks_95;
                        unsigned int _shfl_up_50 = __shfl_up_sync(0xFFFFFFFF, inclusive_96, 1, 32);
                        unsigned int previous_97 = _shfl_up_50;
                        if (lane >= 1) {
                            inclusive_96 += previous_97;
                        }
                        unsigned int _shfl_up_51 = __shfl_up_sync(0xFFFFFFFF, inclusive_96, 2, 32);
                        unsigned int previous_98 = _shfl_up_51;
                        if (lane >= 2) {
                            inclusive_96 += previous_98;
                        }
                        unsigned int _shfl_up_52 = __shfl_up_sync(0xFFFFFFFF, inclusive_96, 4, 32);
                        unsigned int previous_99 = _shfl_up_52;
                        if (lane >= 4) {
                            inclusive_96 += previous_99;
                        }
                        unsigned int _shfl_up_53 = __shfl_up_sync(0xFFFFFFFF, inclusive_96, 8, 32);
                        unsigned int previous_100 = _shfl_up_53;
                        if (lane >= 8) {
                            inclusive_96 += previous_100;
                        }
                        unsigned int _shfl_up_54 = __shfl_up_sync(0xFFFFFFFF, inclusive_96, 16, 32);
                        unsigned int previous_101 = _shfl_up_54;
                        if (lane >= 16) {
                            inclusive_96 += previous_101;
                        }
                        unsigned int lane_offset_102 = block_offset + inclusive_96 - blocks_95;
                        unsigned int _vote_8 = __ballot_sync(0xFFFFFFFF, expert_93 < 384 && (pool_block >= lane_offset_102 && pool_block < lane_offset_102 + blocks_95));
                        unsigned int owner_mask_103 = _vote_8;
                        if (owner_mask_103 != 0) {
                            int _ffs_8 = __ffs(owner_mask_103);
                            unsigned int owner_lane_8 = _ffs_8 - 1;
                            unsigned int local_block_8 = pool_block - lane_offset_102;
                            unsigned int _min_9 = ((tokens_94 - local_block_8 * 16) < (16) ? (tokens_94 - local_block_8 * 16) : (16));
                            unsigned int valid_8 = _min_9;
                            unsigned int _shfl_36 = __shfl_sync(0xFFFFFFFF, expert_93, owner_lane_8);
                            task_3[1] = _shfl_36;
                            unsigned int _shfl_37 = __shfl_sync(0xFFFFFFFF, local_block_8, owner_lane_8);
                            task_3[2] = _shfl_37;
                            unsigned int _shfl_38 = __shfl_sync(0xFFFFFFFF, valid_8, owner_lane_8);
                            task_3[5] = _shfl_38;
                        }
                        unsigned int _shfl_39 = __shfl_sync(0xFFFFFFFF, inclusive_96, 31);
                        block_offset += _shfl_39;
                        unsigned int expert_104 = 288 + lane;
                        unsigned int tokens_105 = counts[9];
                        unsigned int blocks_106 = (tokens_105 + 16 - 1) / 16;
                        unsigned int inclusive_107 = blocks_106;
                        unsigned int _shfl_up_55 = __shfl_up_sync(0xFFFFFFFF, inclusive_107, 1, 32);
                        unsigned int previous_108 = _shfl_up_55;
                        if (lane >= 1) {
                            inclusive_107 += previous_108;
                        }
                        unsigned int _shfl_up_56 = __shfl_up_sync(0xFFFFFFFF, inclusive_107, 2, 32);
                        unsigned int previous_109 = _shfl_up_56;
                        if (lane >= 2) {
                            inclusive_107 += previous_109;
                        }
                        unsigned int _shfl_up_57 = __shfl_up_sync(0xFFFFFFFF, inclusive_107, 4, 32);
                        unsigned int previous_110 = _shfl_up_57;
                        if (lane >= 4) {
                            inclusive_107 += previous_110;
                        }
                        unsigned int _shfl_up_58 = __shfl_up_sync(0xFFFFFFFF, inclusive_107, 8, 32);
                        unsigned int previous_111 = _shfl_up_58;
                        if (lane >= 8) {
                            inclusive_107 += previous_111;
                        }
                        unsigned int _shfl_up_59 = __shfl_up_sync(0xFFFFFFFF, inclusive_107, 16, 32);
                        unsigned int previous_112 = _shfl_up_59;
                        if (lane >= 16) {
                            inclusive_107 += previous_112;
                        }
                        unsigned int lane_offset_113 = block_offset + inclusive_107 - blocks_106;
                        unsigned int _vote_9 = __ballot_sync(0xFFFFFFFF, expert_104 < 384 && (pool_block >= lane_offset_113 && pool_block < lane_offset_113 + blocks_106));
                        unsigned int owner_mask_114 = _vote_9;
                        if (owner_mask_114 != 0) {
                            int _ffs_9 = __ffs(owner_mask_114);
                            unsigned int owner_lane_9 = _ffs_9 - 1;
                            unsigned int local_block_9 = pool_block - lane_offset_113;
                            unsigned int _min_10 = ((tokens_105 - local_block_9 * 16) < (16) ? (tokens_105 - local_block_9 * 16) : (16));
                            unsigned int valid_9 = _min_10;
                            unsigned int _shfl_40 = __shfl_sync(0xFFFFFFFF, expert_104, owner_lane_9);
                            task_3[1] = _shfl_40;
                            unsigned int _shfl_41 = __shfl_sync(0xFFFFFFFF, local_block_9, owner_lane_9);
                            task_3[2] = _shfl_41;
                            unsigned int _shfl_42 = __shfl_sync(0xFFFFFFFF, valid_9, owner_lane_9);
                            task_3[5] = _shfl_42;
                        }
                        unsigned int _shfl_43 = __shfl_sync(0xFFFFFFFF, inclusive_107, 31);
                        block_offset += _shfl_43;
                        unsigned int expert_115 = 320 + lane;
                        unsigned int tokens_116 = counts[10];
                        unsigned int blocks_117 = (tokens_116 + 16 - 1) / 16;
                        unsigned int inclusive_118 = blocks_117;
                        unsigned int _shfl_up_60 = __shfl_up_sync(0xFFFFFFFF, inclusive_118, 1, 32);
                        unsigned int previous_119 = _shfl_up_60;
                        if (lane >= 1) {
                            inclusive_118 += previous_119;
                        }
                        unsigned int _shfl_up_61 = __shfl_up_sync(0xFFFFFFFF, inclusive_118, 2, 32);
                        unsigned int previous_120 = _shfl_up_61;
                        if (lane >= 2) {
                            inclusive_118 += previous_120;
                        }
                        unsigned int _shfl_up_62 = __shfl_up_sync(0xFFFFFFFF, inclusive_118, 4, 32);
                        unsigned int previous_121 = _shfl_up_62;
                        if (lane >= 4) {
                            inclusive_118 += previous_121;
                        }
                        unsigned int _shfl_up_63 = __shfl_up_sync(0xFFFFFFFF, inclusive_118, 8, 32);
                        unsigned int previous_122 = _shfl_up_63;
                        if (lane >= 8) {
                            inclusive_118 += previous_122;
                        }
                        unsigned int _shfl_up_64 = __shfl_up_sync(0xFFFFFFFF, inclusive_118, 16, 32);
                        unsigned int previous_123 = _shfl_up_64;
                        if (lane >= 16) {
                            inclusive_118 += previous_123;
                        }
                        unsigned int lane_offset_124 = block_offset + inclusive_118 - blocks_117;
                        unsigned int _vote_10 = __ballot_sync(0xFFFFFFFF, expert_115 < 384 && (pool_block >= lane_offset_124 && pool_block < lane_offset_124 + blocks_117));
                        unsigned int owner_mask_125 = _vote_10;
                        if (owner_mask_125 != 0) {
                            int _ffs_10 = __ffs(owner_mask_125);
                            unsigned int owner_lane_10 = _ffs_10 - 1;
                            unsigned int local_block_10 = pool_block - lane_offset_124;
                            unsigned int _min_11 = ((tokens_116 - local_block_10 * 16) < (16) ? (tokens_116 - local_block_10 * 16) : (16));
                            unsigned int valid_10 = _min_11;
                            unsigned int _shfl_44 = __shfl_sync(0xFFFFFFFF, expert_115, owner_lane_10);
                            task_3[1] = _shfl_44;
                            unsigned int _shfl_45 = __shfl_sync(0xFFFFFFFF, local_block_10, owner_lane_10);
                            task_3[2] = _shfl_45;
                            unsigned int _shfl_46 = __shfl_sync(0xFFFFFFFF, valid_10, owner_lane_10);
                            task_3[5] = _shfl_46;
                        }
                        unsigned int _shfl_47 = __shfl_sync(0xFFFFFFFF, inclusive_118, 31);
                        block_offset += _shfl_47;
                        unsigned int expert_126 = 352 + lane;
                        unsigned int tokens_127 = counts[11];
                        unsigned int blocks_128 = (tokens_127 + 16 - 1) / 16;
                        unsigned int inclusive_129 = blocks_128;
                        unsigned int _shfl_up_65 = __shfl_up_sync(0xFFFFFFFF, inclusive_129, 1, 32);
                        unsigned int previous_130 = _shfl_up_65;
                        if (lane >= 1) {
                            inclusive_129 += previous_130;
                        }
                        unsigned int _shfl_up_66 = __shfl_up_sync(0xFFFFFFFF, inclusive_129, 2, 32);
                        unsigned int previous_131 = _shfl_up_66;
                        if (lane >= 2) {
                            inclusive_129 += previous_131;
                        }
                        unsigned int _shfl_up_67 = __shfl_up_sync(0xFFFFFFFF, inclusive_129, 4, 32);
                        unsigned int previous_132 = _shfl_up_67;
                        if (lane >= 4) {
                            inclusive_129 += previous_132;
                        }
                        unsigned int _shfl_up_68 = __shfl_up_sync(0xFFFFFFFF, inclusive_129, 8, 32);
                        unsigned int previous_133 = _shfl_up_68;
                        if (lane >= 8) {
                            inclusive_129 += previous_133;
                        }
                        unsigned int _shfl_up_69 = __shfl_up_sync(0xFFFFFFFF, inclusive_129, 16, 32);
                        unsigned int previous_134 = _shfl_up_69;
                        if (lane >= 16) {
                            inclusive_129 += previous_134;
                        }
                        unsigned int lane_offset_135 = block_offset + inclusive_129 - blocks_128;
                        unsigned int _vote_11 = __ballot_sync(0xFFFFFFFF, expert_126 < 384 && (pool_block >= lane_offset_135 && pool_block < lane_offset_135 + blocks_128));
                        unsigned int owner_mask_136 = _vote_11;
                        if (owner_mask_136 != 0) {
                            int _ffs_11 = __ffs(owner_mask_136);
                            unsigned int owner_lane_11 = _ffs_11 - 1;
                            unsigned int local_block_11 = pool_block - lane_offset_135;
                            unsigned int _min_12 = ((tokens_127 - local_block_11 * 16) < (16) ? (tokens_127 - local_block_11 * 16) : (16));
                            unsigned int valid_11 = _min_12;
                            unsigned int _shfl_48 = __shfl_sync(0xFFFFFFFF, expert_126, owner_lane_11);
                            task_3[1] = _shfl_48;
                            unsigned int _shfl_49 = __shfl_sync(0xFFFFFFFF, local_block_11, owner_lane_11);
                            task_3[2] = _shfl_49;
                            unsigned int _shfl_50 = __shfl_sync(0xFFFFFFFF, valid_11, owner_lane_11);
                            task_3[5] = _shfl_50;
                        }
                        unsigned int _shfl_51 = __shfl_sync(0xFFFFFFFF, inclusive_129, 31);
                        block_offset += _shfl_51;
                        if (phase == 2) {
                            unsigned int required = (task_3[4] + 1) * 18;
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"(reinterpret_cast<unsigned int*>(PrivateCounters)) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(required)) >= static_cast<unsigned int>((unsigned int)0 - required));
                            }
                        }
                    }
                    if (task_3[0] == 0) {
                        break;
                    }
                    if (lane < 2) {
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(task_infos_addr + schedule_iteration % 2 * 32), "r"(lane));
                        unsigned int destination = _mapa_0;
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(task_full_addr + schedule_iteration % 2 * 8), "r"(lane));
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
                    schedule_iteration += 1;
                }
                mbarrier_wait(task_empty_addr + (schedule_iteration % 2) * 8, schedule_iteration / 2 & 1 ^ 1);
                task_3[0] = 0;
                task_3[1] = 0;
                task_3[2] = 0;
                task_3[3] = 0;
                task_3[4] = 0;
                task_3[5] = 0;
                task_3[6] = 0;
                task_3[7] = 0;
                if (lane < 2) {
                    uint32_t _mapa_2;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_2) : "r"(task_infos_addr + schedule_iteration % 2 * 32), "r"(lane));
                    unsigned int destination_1 = _mapa_2;
                    uint32_t _mapa_3;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_3) : "r"(task_full_addr + schedule_iteration % 2 * 8), "r"(lane));
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
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 8 && warp <= 15) {
        { // epilogue_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
            unsigned int task_4[8];
            int task_iteration_3 = 0;
            while (1) {
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
                if (task_4[0] == 1) {
                    unsigned int epi_warp = warp - 8;
                    unsigned int epi_wg = epi_warp / 4;
                    unsigned int warp_in_wg = epi_warp % 4;
                    unsigned int _shfl_52 = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int valid_m = _shfl_52;
                    unsigned int pool_block_1 = task_4[4];
                    unsigned int ring_block = pool_block_1 % 390;
                    unsigned int block_1 = ((unsigned int)expert_row_offsets[task_4[1]] + task_4[2] * 16) / 16;
                    unsigned int sf_stride = 199680;
                    if (task_4[0] == 3) {
                        block_1 = pool_block_1;
                        sf_stride = 0;
                    }
                    unsigned int n_block = task_4[3] * 2 + (unsigned int)cta_rank;
                    float cached_weight = 1.0f;
                    float activation[4];
                    float amax[2];
                    float quantized[4];
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
                        for (int i_3 = 0; i_3 < 1; i_3++) {
                            unsigned int j = s + i_3;
                            if (task_4[0] == 1 && j * 8 % 32 == 0) {
                                if (j * 8 + lane < 8) {
                                    cached_weight = routing_weight_pool[block_1 * 16 + epi_wg * 8 + j * 8 + lane];
                                }
                            }
                            float _shfl_53;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_53) : "f"(cached_weight), "r"(j * 8 % 32 + lane % 4 * 2));
                            float first_weight = _shfl_53;
                            float _shfl_54;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_54) : "f"(cached_weight), "r"(j * 8 % 32 + lane % 4 * 2 + 1));
                            float second_weight = _shfl_54;
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
                            __nv_bfloat162 _min_13 = __hmin2(_bf16x2_0, _bf16x2_2);
                            __nv_bfloat162 _max_1 = __hmax2(_bf16x2_1, _bf16x2_3);
                            __nv_bfloat162 _min_14 = __hmin2(_max_1, _bf16x2_2);
                            float2 _cvt_f32_0 = __bfloat1622float2(_min_13);
                            float2 _cvt_f32_1 = __bfloat1622float2(_min_14);
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
                            activation[i_3 * 4] = _mul_f32x2_2.x;
                            activation[i_3 * 4 + 1] = _mul_f32x2_2.y;
                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                            __nv_bfloat162 _min_15 = __hmin2(_bf16x2_4, _bf16x2_6);
                            __nv_bfloat162 _max_2 = __hmax2(_bf16x2_5, _bf16x2_7);
                            __nv_bfloat162 _min_16 = __hmin2(_max_2, _bf16x2_6);
                            float2 _cvt_f32_2 = __bfloat1622float2(_min_15);
                            float2 _cvt_f32_3 = __bfloat1622float2(_min_16);
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
                            activation[i_3 * 4 + 2] = _mul_f32x2_5.x;
                            activation[i_3 * 4 + 3] = _mul_f32x2_5.y;
                            float _fabs_0 = fabsf(activation[i_3 * 4]);
                            float _fabs_1 = fabsf(activation[i_3 * 4 + 2]);
                            float _max_3 = max_noftz(_fabs_0, _fabs_1);
                            float first_max = _max_3;
                            float _fabs_2 = fabsf(activation[i_3 * 4 + 1]);
                            float _fabs_3 = fabsf(activation[i_3 * 4 + 3]);
                            float _max_4 = max_noftz(_fabs_2, _fabs_3);
                            float second_max = _max_4;
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, first_max, 4);
                            float _max_5 = max_noftz(first_max, _shfl_xor_0);
                            first_max = _max_5;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, second_max, 4);
                            float _max_6 = max_noftz(second_max, _shfl_xor_1);
                            second_max = _max_6;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, first_max, 8);
                            float _max_7 = max_noftz(first_max, _shfl_xor_2);
                            first_max = _max_7;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, second_max, 8);
                            float _max_8 = max_noftz(second_max, _shfl_xor_3);
                            second_max = _max_8;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, first_max, 16);
                            float _max_9 = max_noftz(first_max, _shfl_xor_4);
                            first_max = _max_9;
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, second_max, 16);
                            float _max_10 = max_noftz(second_max, _shfl_xor_5);
                            second_max = _max_10;
                            amax[i_3 * 2] = first_max;
                            amax[i_3 * 2 + 1] = second_max;
                            if (lane < 4) {
                                amax_reduction[epi_warp * 8 + (unsigned int)(i_3 * 8) + lane * 2] = first_max;
                                amax_reduction[epi_warp * 8 + (unsigned int)(i_3 * 8) + lane * 2 + 1] = second_max;
                            }
                            __syncwarp();
                        }
                        unsigned int tma_stage = s % 2;
                        unsigned int store_address = smem_output_addr + (epi_wg * 2 + tma_stage) * 8 * 64;
                        asm volatile("cp.async.bulk.wait_group 1;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                        #pragma unroll
                        for (int i_4 = 0; i_4 < 1; i_4++) {
                            unsigned int paired_index = (epi_warp ^ 1) * 8 + (unsigned int)(i_4 * 8) + lane % 4 * 2;
                            float _max_11 = max_noftz(amax[i_4 * 2], amax_reduction[paired_index]);
                            float first_max_1 = _max_11;
                            float _max_12 = max_noftz(amax[i_4 * 2 + 1], amax_reduction[paired_index + 1]);
                            float second_max_1 = _max_12;
                            unsigned int first_bits = __as_u32(first_max_1);
                            unsigned int second_bits = __as_u32(second_max_1);
                            unsigned int _max_13 = ((first_bits + 2097151 >> 23) > (113) ? (first_bits + 2097151 >> 23) : (113));
                            unsigned int first_exp = _max_13 - 8;
                            unsigned int _max_14 = ((second_bits + 2097151 >> 23) > (113) ? (second_bits + 2097151 >> 23) : (113));
                            unsigned int second_exp = _max_14 - 8;
                            unsigned int first_inverse_bits = 254 - first_exp << 23;
                            unsigned int second_inverse_bits = 254 - second_exp << 23;
                            float first_inverse = __uint_as_float(first_inverse_bits);
                            float second_inverse = __uint_as_float(second_inverse_bits);
                            quantized[0] = activation[i_4 * 4] * first_inverse;
                            quantized[1] = activation[i_4 * 4 + 1] * second_inverse;
                            quantized[2] = activation[i_4 * 4 + 2] * first_inverse;
                            quantized[3] = activation[i_4 * 4 + 3] * second_inverse;
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
                            unsigned int stsm_address = store_address + (unsigned int)(i_4 * 8 * 64) + lane * 64 + (warp_in_wg ^ lane / 2) * 16;
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                            uint32_t _stmatrix_b8_x1_addr_2 = static_cast<uint32_t>(stsm_address);
                            asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                                :: "r"(_stmatrix_b8_x1_addr_2), "r"(_fp8_0[0]) : "memory");
                            #elif defined(__CUDA_ARCH__)
                            #error "StmatrixTransB8X1 requires SM100 or newer"
                            #endif
                            if (warp_in_wg % 2 == 0 && lane < 4) {
                                unsigned int sf_k = n_block * 2 + warp_in_wg / 2;
                                unsigned int token_base = epi_wg * 8 + (unsigned int)(s * 8) + (unsigned int)(i_4 * 8);
                                unsigned int sf_address = (block_1 * 16 + token_base + lane * 2) * 72 + sf_k;
                                SF_I_w[sf_address] = (uint8_t)first_exp;
                                SF_I_w[sf_address + 72] = (uint8_t)second_exp;
                                unsigned int source_token = pool_block_1 * 128 + token_base * 4 + lane * 8;
                                unsigned int source_address = sf_k / 4 * 199680 + source_token * 4 + sf_k % 4;
                                PrivateSF2Bytes[source_address] = (uint8_t)first_exp;
                                PrivateSF2Bytes[source_address + 16] = (uint8_t)second_exp;
                            }
                            __syncwarp();
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                        if (warp_in_wg == 0) {
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                if (task_4[0] == 3) {
                                    tma_store_2d((&I_fp8_w), n_block * 64, block_1 * 16 + epi_wg * 8 + (unsigned int)(s * 8), store_address);
                                } else {
                                    tma_store_2d((&I_fp8_w), n_block * 64, block_1 * 16 + epi_wg * 8 + (unsigned int)(s * 8), store_address);
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        __syncwarp();
                    }
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            if (task_4[0] == 3) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (pool_block_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                            } else {
                                asm volatile("red.release.gpu.global.xor.b64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(PrivateMasks) + (ring_block))), "l"(static_cast<unsigned long long>((unsigned long long)1 << (unsigned long long)n_block)) : "memory");
                            }
                        }
                    }
                    __syncwarp();
                } else {
                    unsigned int epi_warp_1 = warp - 8;
                    unsigned int epi_wg_1 = epi_warp_1 / 4;
                    unsigned int warp_in_wg_1 = epi_warp_1 % 4;
                    unsigned int _shfl_55 = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int valid_m_1 = _shfl_55;
                    unsigned int pool_m = (unsigned int)expert_row_offsets[task_4[1]] + task_4[2] * 16;
                    unsigned int n_offset_1 = (task_4[3] * 2 + (unsigned int)cta_rank) * 128;
                    unsigned int output_base = smem_output_addr + epi_wg_1 * 8 * 128 * 2;
                    unsigned int cached_token[1];
                    unsigned int cached_topk[1];
                    unsigned int packed[4];
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
                            unsigned int row = epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + (unsigned int)(j_1 * 8) + warp_in_wg_1 * 2 + lane / 16;
                            if (row < valid_m_1) {
                                if (task_4[0] == 4) {
                                    cached_token[j_1] = pool_m + row;
                                    cached_topk[j_1] = 6;
                                } else {
                                    cached_token[j_1] = pool_m + row;
                                    cached_topk[j_1] = 0;
                                }
                            }
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 1; i_5++) {
                            unsigned int address_1 = accum_stage_1 * 16 + epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + (unsigned int)(i_5 * 8);
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
                            if (i_5 == 0 && s_1 > 0) {
                                asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                            }
                            if (s_1 == 0 && i_5 == 0) {
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
                            unsigned int row_1 = lane % 8;
                            unsigned int col = epi_warp_1 % 2 * 4 + lane / 8;
                            unsigned int write_address = output_base + warp_in_wg_1 / 2 * 8 * 128 + (unsigned int)(i_5 * 8 * 128) + row_1 * 128 + (col ^ row_1) * 16;
                            uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(write_address);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1]))
                                : "memory");
                        }
                        asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 1; j_2++) {
                            unsigned int row_in_store = (unsigned int)(j_2 * 8) + warp_in_wg_1 * 2 + lane / 16;
                            unsigned int row_2 = epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + row_in_store;
                            if (row_2 >= valid_m_1) {
                                break;
                            }
                            unsigned int row_in_atom = (warp_in_wg_1 * 2 + lane / 16) % 8;
                            unsigned int read_address = output_base + lane % 16 / 8 * 8 * 128 + row_in_store * 128 + (lane % 8 ^ row_in_atom) * 16;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                : "r"(read_address));
                            unsigned long long dst_index = (unsigned long long)(pool_m + row_2) * 2560 + (unsigned long long)(n_offset_1 / 2) + (unsigned long long)(lane % 16 * 4);
                            reinterpret_cast<int4*>(PublicExpertOutput + dst_index)[0] = reinterpret_cast<int4*>(packed)[0];
                        }
                    }
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                }
                task_iteration_3 += 1;
            }
            if (warp == 8) {
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(64));
            }
            __threadfence();
            asm volatile("barrier.sync 15, 256;" ::: "memory");
            unsigned int l2_tag = (unsigned int)1 << 31;
            unsigned int l2_addend = 1;
            if (bid == 0) {
                l2_addend = l2_tag - (unsigned int)(NUM_CTAS - 1);
            }
            unsigned int l2_ticket = 0;
            if (warp == 8) {
                if (lane == 0) {
                    unsigned int _atomic_old_7;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_7) : "l"(l2_done), "r"(static_cast<uint32_t>(l2_addend)) : "memory");
                    l2_ticket = _atomic_old_7;
                }
            }
            if (warp == 8) {
                if (lane == 0) {
                    unsigned int _wait_acquire_mask_4;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_4) : "l"(reinterpret_cast<unsigned int*>(l2_done)) : "memory");
                    } while (((_wait_acquire_mask_4 ^ static_cast<unsigned int>(l2_ticket ^ l2_tag)) & static_cast<unsigned int>(l2_tag)) != 0);
                }
            }
            asm volatile("barrier.sync 15, 256;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int epi_warp_2 = warp - 8;
            unsigned int phase_1 = 0;
            unsigned int values[4];
            float reduced[40];
            unsigned int store_values[4];
            #pragma unroll 1
            for (unsigned int token_chunk = epi_warp_2 * 152 + (unsigned int)bid; token_chunk < num_tokens * 4; token_chunk += 1216) {
                unsigned int token = token_chunk / 4;
                unsigned int chunk = token_chunk % 4;
                unsigned int stored_row = 0;
                if (lane < 6) {
                    stored_row = (unsigned int)token_to_permuted[token * 6 + lane];
                }
                #pragma unroll 1
                for (unsigned int i_6 = 0; i_6 < 6; i_6++) {
                    unsigned int _shfl_56 = __shfl_sync(0xFFFFFFFF, stored_row, i_6);
                    unsigned int row_3 = _shfl_56;
                    if (elect_sync()) {
                        cp_async_bulk_gmem2smem(combine_smem_addr + (epi_warp_2 * 6 + i_6) * 2560, expert_output + ((unsigned long long)row_3 * 5120 + (unsigned long long)(chunk * 1280)), 2560, combine_barriers_addr + (epi_warp_2 * 6 + i_6) * 8);
                        mbarrier_arrive_expect_tx(combine_barriers_addr + (epi_warp_2 * 6 + i_6) * 8, 2560);
                    }
                    __syncwarp();
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
                for (unsigned int i_7 = 0; i_7 < 6; i_7++) {
                    mbarrier_wait(combine_barriers_addr + (epi_warp_2 * 6 + i_7) * 8, phase_1);
                    #pragma unroll
                    for (unsigned int j_3 = 0; j_3 < 5; j_3++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&values[0])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 3]))
                            : "r"(combine_smem_addr + (epi_warp_2 * 6 + i_7) * 2560 + (j_3 * 32 + lane) * 16));
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
                }
                phase_1 ^= 1;
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
                        "r"(combine_smem_addr + (48 + epi_warp_2) * 2560 + (j_4 * 32 + lane) * 16), "r"(*reinterpret_cast<uint32_t*>(&store_values[0])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 3])));
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        void* _cpbulk_dst_4 = reinterpret_cast<void*>(y + ((unsigned long long)token * 5120 + (unsigned long long)(chunk * 1280)));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_4), "r"(combine_smem_addr + (48 + epi_warp_2) * 2560), "r"((uint32_t)(2560))
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
