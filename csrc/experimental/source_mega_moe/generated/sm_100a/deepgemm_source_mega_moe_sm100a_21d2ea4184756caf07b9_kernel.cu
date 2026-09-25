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
#define TMEM_NCOLS 396
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SCALE_A_OFFSET 384
#define TMEM_SCALE_B_OFFSET 392
#define NUM_PIPE_STAGES 7
#define SMEM_HISTOGRAM_OFF 0
#define SMEM_HISTOGRAM_STAGE_BYTES 32
#define SMEM_HISTOGRAM_STRIDE 32
#define SMEM_SEND_BUFFER_OFF 1024
#define SMEM_SEND_BUFFER_STAGE_BYTES 2048
#define SMEM_SEND_BUFFER_STRIDE 2048
#define SMEM_SMEM_OUTPUT_OFF 3072
#define SMEM_SMEM_OUTPUT_STAGE_BYTES 16384
#define SMEM_SMEM_OUTPUT_STRIDE 16384
#define SMEM_REUSABLE_OFF 0
#define SMEM_REUSABLE_STAGE_BYTES 232000
#define SMEM_REUSABLE_STRIDE 232000
#define SMEM_SMEM_A_OFF 19456
#define SMEM_SMEM_A_STAGE_BYTES 12288
#define SMEM_SMEM_A_STRIDE 12288
#define SMEM_SMEM_B_OFF 105472
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_SHARED_B_OFF 105472
#define SMEM_SMEM_SHARED_B_STAGE_BYTES 16384
#define SMEM_SMEM_SHARED_B_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 220160
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 1024
#define SMEM_SMEM_SFB_OFF 227328
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 512
#define SMEM_AMAX_REDUCTION_OFF 230912
#define SMEM_AMAX_REDUCTION_STAGE_BYTES 1024
#define SMEM_AMAX_REDUCTION_STRIDE 1024
#define SMEM_TASK_INFOS_OFF 231936
#define SMEM_TASK_INFOS_STAGE_BYTES 64
#define SMEM_TASK_INFOS_STRIDE 64
#define SMEM_TOTAL 232448
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
kernel_deepgemm_source_mega_moe_sm100a_21d2ea4184756caf07b9(DeepgemmTensorMap const* A1, DeepgemmTensorMap const* A2, DeepgemmTensorMap const* SA1, DeepgemmTensorMap const* SA2, DeepgemmTensorMap const* B1, DeepgemmTensorMap const* B2, DeepgemmTensorMap const* SB1, DeepgemmTensorMap const* SB2, DeepgemmTensorMap const* SFA1, DeepgemmTensorMap const* SFA2, DeepgemmTensorMap const* SSFA1, DeepgemmTensorMap const* SSFA2, DeepgemmTensorMap const* SFB1, DeepgemmTensorMap const* SFB2, DeepgemmTensorMap const* SSFB1, DeepgemmTensorMap const* SSFB2, DeepgemmTensorMap const* L1Output, DeepgemmTensorMap const* SharedL1Output, uint8_t* __restrict__ X, unsigned int* __restrict__ XSF, long long* __restrict__ TopK, float* __restrict__ Weights, uint8_t* __restrict__ L1Acts, unsigned int* __restrict__ L1SF, float* __restrict__ L1Weights, uint8_t* __restrict__ L2SF, uint8_t* __restrict__ SharedL2SF, unsigned int* __restrict__ SourceIndices, unsigned int* __restrict__ TokenMetadata, unsigned int* __restrict__ GridCounters, unsigned int* __restrict__ NvlCounter, unsigned int* __restrict__ NvlSignals, unsigned long long* __restrict__ PeerGrid, unsigned long long* __restrict__ ReadyGrid, unsigned long long* __restrict__ SendCounts, unsigned long long* __restrict__ RecvCounts, unsigned long long* __restrict__ RecvSum, unsigned int* __restrict__ L1Full, unsigned int* __restrict__ L1Empty, unsigned long long* __restrict__ L2Mask, unsigned int* __restrict__ L2Empty, unsigned int* __restrict__ SharedFull, unsigned int* __restrict__ L1Counter, unsigned int* __restrict__ L2Counter, unsigned int* __restrict__ SharedL1Counter, unsigned int* __restrict__ SharedL2Counter, unsigned int* __restrict__ Combine, uint8_t* __restrict__ CombineBytes, uint8_t* __restrict__ Y, unsigned int num_tokens)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 232000;
    #define pull_barriers_addr (mbar_base + 0)
    #define full_addr (mbar_base + 32)
    #define empty_addr (mbar_base + 88)
    #define tmem_full_addr (mbar_base + 144)
    #define tmem_empty_addr (mbar_base + 160)
    #define combine_barriers_addr (mbar_base + 176)
    #define task_full_addr (mbar_base + 304)
    #define task_empty_addr (mbar_base + 320)

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
    uint8_t* send_buffer = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int send_buffer_addr = smem + 1024;
    uint8_t* smem_output = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_output_addr = smem + 3072;
    uint8_t* reusable = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int reusable_addr = smem + 0;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_a_addr = smem + 19456;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 105472);
    const int smem_b_addr = smem + 105472;
    uint8_t* smem_shared_b = reinterpret_cast<uint8_t*>(smem_raw + 105472);
    const int smem_shared_b_addr = smem + 105472;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 220160);
    const int smem_sfa_addr = smem + 220160;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 227328);
    const int smem_sfb_addr = smem + 227328;
    float* amax_reduction = reinterpret_cast<float*>(smem_raw + 230912);
    const int amax_reduction_addr = smem + 230912;
    unsigned int* task_infos = reinterpret_cast<unsigned int*>(smem_raw + 231936);
    const int task_infos_addr = smem + 231936;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
            uint32_t _smem_bulk_zero_addr_0 = static_cast<uint32_t>(histogram_addr);
            asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;"
                :: "r"(_smem_bulk_zero_addr_0), "l"(static_cast<uint64_t>(1024)) : "memory");
            #elif defined(__CUDA_ARCH__)
            #error "SmemBulkZero requires SM100 or newer"
            #endif
        }
    }

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 42 barriers)
    // Mbarriers at smem_raw[232000..232336)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // pull_barriers: 4 barriers, init_count=1
            mbarrier_init(smem + 232000, 1);
            mbarrier_init(smem + 232008, 1);
            mbarrier_init(smem + 232016, 1);
            mbarrier_init(smem + 232024, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 7 barriers, init_count=4
            mbarrier_init(smem + 232032, 4);
            mbarrier_init(smem + 232040, 4);
            mbarrier_init(smem + 232048, 4);
            mbarrier_init(smem + 232056, 4);
            mbarrier_init(smem + 232064, 4);
            mbarrier_init(smem + 232072, 4);
            mbarrier_init(smem + 232080, 4);
            // empty: 7 barriers, init_count=1
            mbarrier_init(smem + 232088, 1);
            mbarrier_init(smem + 232096, 1);
            mbarrier_init(smem + 232104, 1);
            mbarrier_init(smem + 232112, 1);
            mbarrier_init(smem + 232120, 1);
            mbarrier_init(smem + 232128, 1);
            mbarrier_init(smem + 232136, 1);
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232144, 1);
            mbarrier_init(smem + 232152, 1);
            // tmem_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 232160, 512);
            mbarrier_init(smem + 232168, 512);
            // combine_barriers: 16 barriers, init_count=1
            mbarrier_init(smem + 232176, 1);
            mbarrier_init(smem + 232184, 1);
            mbarrier_init(smem + 232192, 1);
            mbarrier_init(smem + 232200, 1);
            mbarrier_init(smem + 232208, 1);
            mbarrier_init(smem + 232216, 1);
            mbarrier_init(smem + 232224, 1);
            mbarrier_init(smem + 232232, 1);
            mbarrier_init(smem + 232240, 1);
            mbarrier_init(smem + 232248, 1);
            mbarrier_init(smem + 232256, 1);
            mbarrier_init(smem + 232264, 1);
            mbarrier_init(smem + 232272, 1);
            mbarrier_init(smem + 232280, 1);
            mbarrier_init(smem + 232288, 1);
            mbarrier_init(smem + 232296, 1);
            // task_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232304, 1);
            mbarrier_init(smem + 232312, 1);
            // task_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 232320, 512);
            mbarrier_init(smem + 232328, 512);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 396 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232336);
    if (warp == 3) {
        int _tmem_hold = smem + 232336;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_scale_a = taddr + 384;
    const int tmem_scale_b = taddr + 392;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: dispatch ----
    if (warp <= 3) {
        { // dispatch_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
            unsigned int counts[1];
            unsigned int state[2];
            unsigned int first_token = ((unsigned int)(bid * 4) + warp) * 4;
            #pragma unroll
            for (int token_base = first_token; token_base < num_tokens; token_base += 32) {
                if ((unsigned int)token_base + lane / 8 < num_tokens && lane < 32) {
                    int expert = (int)TopK[(unsigned int)(token_base * 8) + lane];
                    if (expert >= 0) {
                        uint32_t _shared_atomic_old_0;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((histogram_addr + 4 * (expert)))), "r"(static_cast<uint32_t>(1)) : "memory");
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            #pragma unroll
            for (int expert_1 = tid; expert_1 < 8; expert_1 += 128) {
                unsigned int local_count = histogram[expert_1];
                unsigned long long send_value = (unsigned long long)1 << 32 | (unsigned long long)local_count;
                unsigned long long _atomic_old_0 = atomicAdd(&SendCounts[expert_1], send_value);
                unsigned long long old = _atomic_old_0;
                histogram[expert_1] = (unsigned int)old;
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            #pragma unroll
            for (int token_base_1 = first_token; token_base_1 < num_tokens; token_base_1 += 32) {
                if ((unsigned int)token_base_1 + lane / 8 < num_tokens && lane < 32) {
                    int expert_2 = (int)TopK[(unsigned int)(token_base_1 * 8) + lane];
                    if (expert_2 >= 0) {
                        uint32_t _shared_atomic_old_1;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_1) : "r"(static_cast<uint32_t>((histogram_addr + 4 * (expert_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        unsigned int slot = _shared_atomic_old_1;
                        SourceIndices[(unsigned int)(expert_2 * 1920) + slot] = (unsigned int)(token_base_1 * 8) + lane;
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (tid == 0) {
                unsigned int increment = 1;
                if (bid == 0) {
                    increment = 2147483647;
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
                for (int expert_3 = tid; expert_3 < 8; expert_3 += 128) {
                    unsigned long long status = SendCounts[expert_3];
                    RecvCounts[expert_3] = status & 4294967295;
                    unsigned long long _atomic_old_2;
                    asm volatile("atom.sys.add.u64 %0, [%1], %2;"
                        : "=l"(_atomic_old_2) : "l"(&RecvSum[expert_3]), "l"(static_cast<uint64_t>(status)) : "memory");
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
                    increment_1 = 2147483647;
                }
                unsigned int _atomic_old_3;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_3) : "l"(&GridCounters[0]), "r"(static_cast<uint32_t>(increment_1)) : "memory");
                unsigned int old_2 = _atomic_old_3;
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
            unsigned int expert_4 = lane;
            unsigned long long received = 0;
            if (expert_4 < 8) {
                unsigned long long _wait_acquire_mask_2;
                do {
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_wait_acquire_mask_2) : "l"((reinterpret_cast<unsigned long long*>(RecvSum) + (expert_4))) : "memory");
                } while (((_wait_acquire_mask_2 ^ static_cast<unsigned long long>((unsigned long long)2 << 32)) & static_cast<unsigned long long>((unsigned long long)4294967295 << 32)) != 0);
                received = _wait_acquire_mask_2;
            }
            counts[0] = (unsigned int)received;
            __syncwarp();
            unsigned int num_blocks = 0;
            if (lane < 8) {
                num_blocks += (counts[0] + 192 - 1) / 192;
            }
            unsigned int _warp_redux_u32_0;
            asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(num_blocks));
            unsigned int total = _warp_redux_u32_0;
            unsigned int waves = total * 2 + 1 - 1;
            unsigned int interleave_waves = 3;
            state[0] = total;
            int _max_0 = ((2) > (interleave_waves) ? (2) : (interleave_waves));
            int _min_0 = ((_max_0) < (waves) ? (_max_0) : (waves));
            state[1] = _min_0;
            #pragma unroll 1
            for (int token = (unsigned int)(bid * 4) + warp; token < num_tokens * 8; token += 8) {
                #pragma unroll 1
                for (int advance_expert = 0; advance_expert < 9; advance_expert++) {
                    if (expert_end > (unsigned int)token) {
                        break;
                    }
                    current_expert += 1;
                    if (current_expert >= 8) {
                        break;
                    }
                    pool_offset += (expert_end - expert_start + 192 - 1) / 192;
                    expert_start = expert_end;
                    unsigned int selected = 0;
                    if ((unsigned int)current_expert == lane) {
                        selected = counts[0];
                    }
                    unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, selected, current_expert % 32);
                    expert_end += _shfl_0;
                }
                if (current_expert >= 8) {
                    break;
                }
                unsigned int in_expert = (unsigned int)token - expert_start;
                unsigned int source_topk = SourceIndices[(unsigned int)(current_expert * 1920) + in_expert];
                unsigned int source_token = source_topk / 8;
                unsigned int pool_token = pool_offset * 192 + in_expert;
                unsigned int pool_block = pool_token / 192;
                unsigned int ring_block = pool_block % 10;
                unsigned int ring_token = pool_token % 1920;
                unsigned int target_1 = pool_block / 10 * 4;
                if (target_1 > 0) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(L1Empty) + (ring_block))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(target_1)) >= static_cast<unsigned int>((unsigned int)0 - target_1));
                    }
                }
                unsigned long long source_byte = (unsigned long long)source_token * 512;
                unsigned long long destination_byte = (unsigned long long)ring_token * 512;
                if (elect_sync()) {
                    #pragma unroll
                    for (int chunk = 0; chunk < 1; chunk++) {
                        cp_async_bulk_gmem2smem(send_buffer_addr + warp * 512, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(X) + ((unsigned long long)(source_byte + (unsigned long long)(chunk * 512)) * (unsigned long long)1)), 512, pull_barriers_addr + (warp) * 8);
                        mbarrier_arrive_expect_tx(pull_barriers_addr + (warp) * 8, 512);
                        if (chunk != 0) {
                            mbarrier_wait(pull_barriers_addr + (warp) * 8, pull_phase);
                            pull_phase ^= 1;
                            {
                                void* _cpbulk_dst_0 = reinterpret_cast<void*>(L1Acts + (destination_byte + (unsigned long long)(chunk * 512)));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_0), "r"(send_buffer_addr + warp * 512), "r"((uint32_t)(512))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group 0;");
                        }
                    }
                }
                __syncwarp();
                float weight = Weights[source_topk];
                unsigned int in_block = in_expert % 192;
                unsigned int sf_token = ring_block * 256 + (in_block & 4294967168) + (in_block & 31) * 4 + (in_block >> 5 & 3);
                #pragma unroll
                for (int sf_group = 0; sf_group < 1; sf_group++) {
                    unsigned int sf_word = (unsigned int)(sf_group * 32) + lane;
                    if (sf_word < 4) {
                        L1SF[sf_word * 30720 + sf_token] = XSF[source_token * 4 + sf_word];
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    L1Weights[ring_token] = weight;
                    TokenMetadata[pool_token * 3] = 0;
                    TokenMetadata[pool_token * 3 + 1] = source_token;
                    TokenMetadata[pool_token * 3 + 2] = source_topk % 8;
                    mbarrier_wait(pull_barriers_addr + (warp) * 8, pull_phase);
                    pull_phase ^= 1;
                    {
                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(L1Acts + destination_byte);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_1), "r"(send_buffer_addr + warp * 512), "r"((uint32_t)(512))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    unsigned int arrivals = (((unsigned int)token == expert_end - 1) ? 192 - in_block : (unsigned int)1);
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(L1Full) + (ring_block))), "r"(static_cast<unsigned int>(arrivals)) : "memory");
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (bid == 0) {
                #pragma unroll
                for (int expert_5 = tid; expert_5 < 8; expert_5 += 128) {
                    SendCounts[expert_5] = 0;
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
                for (int expert_6 = bid - 1; expert_6 < 8; expert_6++) {
                    unsigned int num_received = (unsigned int)RecvSum[expert_6];
                    unsigned int blocks = (num_received + 192 - 1) / 192;
                    unsigned int num_blocks_0 = 0;
                    if (lane < (unsigned int)expert_6) {
                        num_blocks_0 += (counts[0] + 192 - 1) / 192;
                    }
                    unsigned int _warp_redux_u32_1;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(num_blocks_0));
                    unsigned int offset = _warp_redux_u32_1;
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    if (warp == 0) {
                        RecvSum[expert_6] = 0;
                    }
                    if (tid == 0) {
                        RecvCounts[expert_6] = 0;
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (int block_1 = tid; block_1 < blocks; block_1 += 128) {
                        unsigned int ring = (offset + (unsigned int)block_1) % 10;
                        L1Full[ring] = 0;
                        L1Empty[ring] = 0;
                        L2Mask[ring] = 0;
                        L2Empty[ring] = 0;
                    }
                    __syncwarp();
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (tid == 0) {
                unsigned int increment_2 = 1;
                if (bid == 0) {
                    increment_2 = 2147483647;
                }
                unsigned int _atomic_old_4;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_4) : "l"(&GridCounters[0]), "r"(static_cast<uint32_t>(increment_2)) : "memory");
                unsigned int old_3 = _atomic_old_4;
                unsigned int _wait_acquire_mask_3;
                do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_3) : "l"((reinterpret_cast<unsigned int*>(GridCounters) + (0))) : "memory");
                } while (((_wait_acquire_mask_3 ^ static_cast<unsigned int>((old_3 ^ 2147483648) & 2147483648)) & static_cast<unsigned int>(2147483648)) != 0);
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (bid == 0) {
                unsigned int status_2 = NvlCounter[0] & 3;
                unsigned int phase_1 = status_2 & 1;
                unsigned int sign_1 = status_2 >> 1;
                if (tid == 0) {
                    unsigned int delta_1 = ((sign_1 == 0) ? 1 : 4294967295);
                    asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(NvlSignals) + (phase_1))), "r"(static_cast<unsigned int>(delta_1)) : "memory");
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (tid == 0) {
                    atomicAdd(&NvlCounter[0], 1);
                    unsigned int target_2 = ((sign_1 == 0) ? 1 : 0);
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(NvlSignals) + (phase_1))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(target_2)) >= static_cast<unsigned int>(1));
                    }
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 4) {
        { // load_a_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
            unsigned int task[8];
            unsigned int a_stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (int task_iteration = 0; task_iteration < 401; task_iteration++) {
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
                unsigned int ring_block_1 = pool_block_1 % 10;
                unsigned int block_2 = ring_block_1;
                if (task[0] > 2) {
                    block_2 = pool_block_1;
                }
                if (task[0] == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(L1Full) + (ring_block_1))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(192 * (pool_block_1 / 10 + 1))) >= static_cast<unsigned int>(1));
                    }
                }
                if (task[0] == 4) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(SharedFull) + (block_2))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(4)) >= static_cast<unsigned int>(1));
                    }
                }
                unsigned long long expected = 15;
                if ((pool_block_1 / 10 & 1) != 0) {
                    expected = 0;
                }
                unsigned long long pending = 15;
                unsigned int token_offset = block_2 * 192 + (unsigned int)cta_rank * ((task[5] + 15) / 16 * 8);
                unsigned int sf_offset = block_2 * 256;
                for (int k = 0; k < task[7] / 128; k++) {
                    if (task[0] == 2) {
                        unsigned long long k_mask = (unsigned long long)3 << (unsigned long long)(k * 2);
                        if ((pending & k_mask) != 0) {
                            unsigned long long _wait_acquire_mask_5;
                            do {
                            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_wait_acquire_mask_5) : "l"((reinterpret_cast<unsigned long long*>(L2Mask) + (ring_block_1))) : "memory");
                            } while (((_wait_acquire_mask_5 ^ static_cast<unsigned long long>(expected)) & static_cast<unsigned long long>(k_mask)) != 0);
                            unsigned long long observed = _wait_acquire_mask_5;
                            pending = observed ^ expected;
                        }
                    }
                    mbarrier_wait(empty_addr + (a_stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + a_stage * 12288, selected_activation_map, 0, token_offset, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 1024, selected_activation_sf_map, sf_offset, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (a_stage) * 8, 26624);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (a_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    __syncwarp();
                    a_stage += 1;
                    if (a_stage == 7) { a_stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 5) {
        { // load_b_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
            unsigned int task_1[8];
            unsigned int b_stage = 0;
            unsigned int _phase_empty_1 = 1;
            #pragma unroll 1
            for (int task_iteration_1 = 0; task_iteration_1 < 401; task_iteration_1++) {
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
                        } else {
                            tma_3d_gmem2smem_cta2(smem_b_addr + b_stage * 16384, selected_weight_map, 0, row_offset, k_1, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                            tma_2d_gmem2smem_cta2(smem_sfb_addr + b_stage * 512, selected_weight_sf_map, n_offset, sf_k_offset + (unsigned int)k_1, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                        }
                        if (cta_rank == 0) {
                            unsigned int transfer_bytes = 17408;
                            if (task_1[0] > 2) {
                                transfer_bytes = 33792;
                            }
                            mbarrier_arrive_expect_tx(full_addr + (b_stage) * 8, transfer_bytes);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (b_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    __syncwarp();
                    b_stage += 1;
                    if (b_stage == 7) { b_stage = 0; _phase_empty_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        { // mma_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
            unsigned int task_2[8];
            unsigned int mma_stage = 0;
            unsigned int completed_tasks = 0;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int task_iteration_2 = 0; task_iteration_2 < 401; task_iteration_2++) {
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
                    unsigned int aligned_m = (task_2[5] + 15) / 16 * 16;
                    unsigned int selected_mma_idesc = (unsigned int)((unsigned int)(((task_2[0] > 2) ? 279969792 : 279970432) & -8257537) | aligned_m >> 3 << 17);
                    mbarrier_wait(tmem_empty_addr + (accum_stage) * 8, task_iteration_2 / 2 & 1 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 2
                    for (int k_2 = 0; k_2 < task_2[7] / 128; k_2++) {
                        mbarrier_wait(full_addr + (mma_stage) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((k_2 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_a, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_a + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 64 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_b, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_stage) * 32)));
                            int _mma_a_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024;
                            int _mma_b_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 768;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 192)), a_desc + 0, b_desc + 0,
                                    (selected_mma_idesc | ((0) << 29) | ((0) << 4)), tmem_scale_b, tmem_scale_a, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 192)), a_desc + 2, b_desc + 2,
                                    (selected_mma_idesc | ((1) << 29) | ((1) << 4)), tmem_scale_b, tmem_scale_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 192)), a_desc + 4, b_desc + 4,
                                    (selected_mma_idesc | ((2) << 29) | ((2) << 4)), tmem_scale_b, tmem_scale_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 192)), a_desc + 6, b_desc + 6,
                                    (selected_mma_idesc | ((3) << 29) | ((3) << 4)), tmem_scale_b, tmem_scale_a, 1);
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        if ((unsigned int)k_2 == task_2[7] / 128 - 1) {
                            elect_commit_cg2_multicast(tmem_full_addr + (accum_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        mma_stage += 1;
                        if (mma_stage == 7) { mma_stage = 0; _phase_full ^= 1; }
                    }
                    completed_tasks += 1;
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
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
            unsigned int counts_1[1];
            unsigned int state_1[2];
            unsigned int task_3[8];
            unsigned int schedule_iteration[1];
            schedule_iteration[0] = 0;
            unsigned int early_shared_l2 = (unsigned int)((num_tokens + 192 - 1) / 192 * 2 <= 1);
            if (cta_rank == 0) {
                unsigned int total_1 = (num_tokens + 192 - 1) / 192 * 2;
                #pragma unroll 1
                for (int iteration = 0; iteration < 401; iteration++) {
                    mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                    unsigned int claimed = 0;
                    if (lane == 0) {
                        unsigned int _atomic_old_5 = atomicAdd(SharedL1Counter, 1);
                        claimed = _atomic_old_5;
                    }
                    unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, claimed, 0);
                    unsigned int claimed_0 = _shfl_1;
                    if (claimed_0 >= total_1) {
                        break;
                    }
                    unsigned int block_3 = claimed_0 / 2;
                    task_3[0] = 3;
                    task_3[1] = 0;
                    task_3[2] = block_3;
                    task_3[3] = claimed_0 % 2;
                    task_3[4] = block_3;
                    unsigned int _min_1 = ((num_tokens - block_3 * 192) < (192) ? (num_tokens - block_3 * 192) : (192));
                    task_3[5] = _min_1;
                    task_3[6] = 512;
                    task_3[7] = 512;
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
                    unsigned int total_0 = (num_tokens + 192 - 1) / 192 * 2;
                    #pragma unroll 1
                    for (int iteration_1 = 0; iteration_1 < 401; iteration_1++) {
                        mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                        unsigned int claimed_1 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_6 = atomicAdd(SharedL2Counter, 1);
                            claimed_1 = _atomic_old_6;
                        }
                        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, claimed_1, 0);
                        unsigned int claimed_0_1 = _shfl_2;
                        if (claimed_0_1 >= total_0) {
                            break;
                        }
                        unsigned int block_4 = claimed_0_1 / 2;
                        task_3[0] = 4;
                        task_3[1] = 0;
                        task_3[2] = block_4;
                        task_3[3] = claimed_0_1 % 2;
                        task_3[4] = block_4;
                        unsigned int _min_2 = ((num_tokens - block_4 * 192) < (192) ? (num_tokens - block_4 * 192) : (192));
                        task_3[5] = _min_2;
                        task_3[6] = 512;
                        task_3[7] = 256;
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
                unsigned int expert_7 = lane;
                unsigned long long received_1 = 0;
                if (expert_7 < 8) {
                    unsigned long long _wait_acquire_mask_4;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_wait_acquire_mask_4) : "l"((reinterpret_cast<unsigned long long*>(RecvSum) + (expert_7))) : "memory");
                    } while (((_wait_acquire_mask_4 ^ static_cast<unsigned long long>((unsigned long long)2 << 32)) & static_cast<unsigned long long>((unsigned long long)4294967295 << 32)) != 0);
                    received_1 = _wait_acquire_mask_4;
                }
                counts_1[0] = (unsigned int)received_1;
                __syncwarp();
                unsigned int num_blocks_1 = 0;
                if (lane < 8) {
                    num_blocks_1 += (counts_1[0] + 192 - 1) / 192;
                }
                unsigned int _warp_redux_u32_2;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(num_blocks_1));
                unsigned int total_0_1 = _warp_redux_u32_2;
                unsigned int waves_1 = total_0_1 * 2 + 1 - 1;
                unsigned int interleave_waves_1 = 3;
                state_1[0] = total_0_1;
                int _max_1 = ((2) > (interleave_waves_1) ? (2) : (interleave_waves_1));
                int _min_3 = ((_max_1) < (waves_1) ? (_max_1) : (waves_1));
                state_1[1] = _min_3;
                #pragma unroll 1
                for (int iteration_2 = 0; iteration_2 < 401; iteration_2++) {
                    mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                    unsigned int phase_2 = 0;
                    unsigned int claimed_2 = 0;
                    unsigned int clusters = 2;
                    unsigned int shape_n = 512;
                    unsigned int shape_k = 512;
                    if (state_1[1] != 4294967295 && state_1[1] != 0) {
                        state_1[1] = state_1[1] - 1;
                        unsigned int claimed_0_2 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_7 = atomicAdd(L1Counter, 1);
                            claimed_0_2 = _atomic_old_7;
                        }
                        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, claimed_0_2, 0);
                        claimed_2 = _shfl_3;
                        if (claimed_2 >= state_1[0] * 2) {
                            state_1[1] = 4294967295;
                        } else {
                            phase_2 = 1;
                        }
                    }
                    if (phase_2 == 0) {
                        unsigned int claimed_0_3 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_8 = atomicAdd(L2Counter, 1);
                            claimed_0_3 = _atomic_old_8;
                        }
                        unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, claimed_0_3, 0);
                        claimed_2 = _shfl_4;
                        if (claimed_2 < state_1[0] * 2) {
                            if (state_1[1] != 4294967295) {
                                state_1[1] = 1;
                            }
                            phase_2 = 2;
                            clusters = 2;
                            shape_n = 512;
                            shape_k = 256;
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
                    if (phase_2 != 0) {
                        unsigned int pool_block_2 = claimed_2 / clusters;
                        task_3[0] = phase_2;
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
                        unsigned int blocks_1 = (tokens + 192 - 1) / 192;
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
                        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, expert_0 < 8 && (pool_block_2 >= lane_offset && pool_block_2 < lane_offset + blocks_1));
                        unsigned int owner_mask = _vote_0;
                        if (owner_mask != 0) {
                            int _ffs_0 = __ffs(owner_mask);
                            unsigned int owner_lane = _ffs_0 - 1;
                            unsigned int local_block = pool_block_2 - lane_offset;
                            unsigned int _min_4 = ((tokens - local_block * 192) < (192) ? (tokens - local_block * 192) : (192));
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
                        if (phase_2 == 2) {
                            unsigned int required = (task_3[4] + 1) * 2;
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
                    unsigned int total_1_1 = (num_tokens + 192 - 1) / 192 * 2;
                    #pragma unroll 1
                    for (int iteration_3 = 0; iteration_3 < 401; iteration_3++) {
                        mbarrier_wait(task_empty_addr + (schedule_iteration[0] % 2) * 8, schedule_iteration[0] / 2 & 1 ^ 1);
                        unsigned int claimed_3 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_9 = atomicAdd(SharedL2Counter, 1);
                            claimed_3 = _atomic_old_9;
                        }
                        unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, claimed_3, 0);
                        unsigned int claimed_0_4 = _shfl_9;
                        if (claimed_0_4 >= total_1_1) {
                            break;
                        }
                        unsigned int block_5 = claimed_0_4 / 2;
                        task_3[0] = 4;
                        task_3[1] = 0;
                        task_3[2] = block_5;
                        task_3[3] = claimed_0_4 % 2;
                        task_3[4] = block_5;
                        unsigned int _min_5 = ((num_tokens - block_5 * 192) < (192) ? (num_tokens - block_5 * 192) : (192));
                        task_3[5] = _min_5;
                        task_3[6] = 512;
                        task_3[7] = 256;
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
    }
    // ---- Role: epilogue ----
    if (warp >= 8 && warp <= 15) {
        { // epilogue_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
            unsigned int task_4[8];
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            #pragma unroll 1
            for (int task_iteration_3 = 0; task_iteration_3 < 401; task_iteration_3++) {
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
                    unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int valid_m = _shfl_10;
                    unsigned int pool_block_3 = task_4[4];
                    unsigned int ring_block_2 = pool_block_3 % 10;
                    unsigned int block_6 = ring_block_2;
                    unsigned int sf_stride = 122880;
                    if (task_4[0] == 3) {
                        block_6 = pool_block_3;
                        sf_stride = 122880;
                    }
                    unsigned int n_block = task_4[3] * 2 + (unsigned int)cta_rank;
                    float cached_weight = 1.0f;
                    float activation[16];
                    float amax[8];
                    float quantized[4];
                    if (task_4[0] == 1) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(L2Empty) + (ring_block_2))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(4 * (pool_block_3 / 10))) >= static_cast<unsigned int>(1));
                        }
                    }
                    #pragma unroll
                    for (int s = 0; s < 3; s++) {
                        if (valid_m <= epi_wg * 96 + (unsigned int)(s * 32)) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            break;
                        }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            unsigned int j = s * 4 + i;
                            if (task_4[0] == 1 && j * 8 % 32 == 0) {
                                {
                                    cached_weight = L1Weights[ring_block_2 * 192 + epi_wg * 96 + j * 8 + lane];
                                }
                            }
                            float _shfl_11;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_11) : "f"(cached_weight), "r"(j * 8 % 32 + lane % 4 * 2));
                            float first_weight = _shfl_11;
                            float _shfl_12;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_12) : "f"(cached_weight), "r"(j * 8 % 32 + lane % 4 * 2 + 1));
                            float second_weight = _shfl_12;
                            unsigned int address = accum_stage_1 * 192 + epi_wg * 96 + j * 8;
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
                            if (j == 11) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                            __nv_bfloat162 _min_6 = __hmin2(_bf16x2_0, _bf16x2_2);
                            __nv_bfloat162 _max_2 = __hmax2(_bf16x2_1, _bf16x2_3);
                            __nv_bfloat162 _min_7 = __hmin2(_max_2, _bf16x2_2);
                            float2 _cvt_f32_0 = __bfloat1622float2(_min_6);
                            float2 _cvt_f32_1 = __bfloat1622float2(_min_7);
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
                            __nv_bfloat162 _min_8 = __hmin2(_bf16x2_4, _bf16x2_6);
                            __nv_bfloat162 _max_3 = __hmax2(_bf16x2_5, _bf16x2_7);
                            __nv_bfloat162 _min_9 = __hmin2(_max_3, _bf16x2_6);
                            float2 _cvt_f32_2 = __bfloat1622float2(_min_8);
                            float2 _cvt_f32_3 = __bfloat1622float2(_min_9);
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
                            float _fabs_1 = fabsf(activation[i * 4 + 2]);
                            float _max_4 = max_noftz(_fabs_0, _fabs_1);
                            float first_max = _max_4;
                            float _fabs_2 = fabsf(activation[i * 4 + 1]);
                            float _fabs_3 = fabsf(activation[i * 4 + 3]);
                            float _max_5 = max_noftz(_fabs_2, _fabs_3);
                            float second_max = _max_5;
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, first_max, 4);
                            float _max_6 = max_noftz(first_max, _shfl_xor_0);
                            first_max = _max_6;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, second_max, 4);
                            float _max_7 = max_noftz(second_max, _shfl_xor_1);
                            second_max = _max_7;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, first_max, 8);
                            float _max_8 = max_noftz(first_max, _shfl_xor_2);
                            first_max = _max_8;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, second_max, 8);
                            float _max_9 = max_noftz(second_max, _shfl_xor_3);
                            second_max = _max_9;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, first_max, 16);
                            float _max_10 = max_noftz(first_max, _shfl_xor_4);
                            first_max = _max_10;
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, second_max, 16);
                            float _max_11 = max_noftz(second_max, _shfl_xor_5);
                            second_max = _max_11;
                            amax[i * 2] = first_max;
                            amax[i * 2 + 1] = second_max;
                            if (lane < 4) {
                                amax_reduction[epi_warp * 32 + (unsigned int)(i * 8) + lane * 2] = first_max;
                                amax_reduction[epi_warp * 32 + (unsigned int)(i * 8) + lane * 2 + 1] = second_max;
                            }
                            __syncwarp();
                        }
                        unsigned int tma_stage = s % 2;
                        unsigned int store_address = smem_output_addr + (epi_wg * 2 + tma_stage) * 32 * 64;
                        asm volatile("cp.async.bulk.wait_group 1;");
                        asm volatile("bar.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            unsigned int paired_index = (epi_warp ^ 1) * 32 + (unsigned int)(i_1 * 8) + lane % 4 * 2;
                            float _max_12 = max_noftz(amax[i_1 * 2], amax_reduction[paired_index]);
                            float first_max_1 = _max_12;
                            float _max_13 = max_noftz(amax[i_1 * 2 + 1], amax_reduction[paired_index + 1]);
                            float second_max_1 = _max_13;
                            unsigned int first_bits = __as_u32(first_max_1);
                            unsigned int second_bits = __as_u32(second_max_1);
                            unsigned int _max_14 = ((first_bits + 2097151 >> 23) > (113) ? (first_bits + 2097151 >> 23) : (113));
                            unsigned int first_exp = _max_14 - 8;
                            unsigned int _max_15 = ((second_bits + 2097151 >> 23) > (113) ? (second_bits + 2097151 >> 23) : (113));
                            unsigned int second_exp = _max_15 - 8;
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
                            unsigned int stsm_address = store_address + (unsigned int)(i_1 * 8 * 64) + lane * 64 + (warp_in_wg ^ lane / 2) * 16;
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                            uint32_t _stmatrix_b8_x1_addr_2 = static_cast<uint32_t>(stsm_address);
                            asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                                :: "r"(_stmatrix_b8_x1_addr_2), "r"(_fp8_0[0]) : "memory");
                            #elif defined(__CUDA_ARCH__)
                            #error "StmatrixTransB8X1 requires SM100 or newer"
                            #endif
                            if (warp_in_wg % 2 == 0 && lane < 4) {
                                unsigned int sf_k = n_block * 2 + warp_in_wg / 2;
                                unsigned int token_base_2 = epi_wg * 96 + (unsigned int)(s * 32) + (unsigned int)(i_1 * 8);
                                unsigned int permuted_base = (token_base_2 & 4294967168) + (token_base_2 & 31) * 4 + (token_base_2 >> 5 & 3);
                                unsigned int sf_token_1 = block_6 * 256 + permuted_base + lane * 8;
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
                                    tma_store_2d(SharedL1Output, n_block * 64, block_6 * 192 + epi_wg * 96 + (unsigned int)(s * 32), store_address);
                                } else {
                                    tma_store_2d(L1Output, n_block * 64, block_6 * 192 + epi_wg * 96 + (unsigned int)(s * 32), store_address);
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
                    unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int valid_m_1 = _shfl_13;
                    unsigned int pool_m = task_4[4] * 192;
                    unsigned int n_offset_1 = (task_4[3] * 2 + (unsigned int)cta_rank) * 128;
                    unsigned int output_base = smem_output_addr + epi_wg_1 * 32 * 128 * 2;
                    unsigned int cached_token[4];
                    unsigned int cached_topk[4];
                    unsigned int packed[4];
                    if (task_4[0] == 2) {
                        if (epi_warp_1 == 0) {
                            if (elect_sync()) {
                                atomicAdd(&L2Empty[task_4[4] % 10], 1);
                            }
                        }
                        __syncwarp();
                    }
                    #pragma unroll
                    for (int s_1 = 0; s_1 < 3; s_1++) {
                        if (valid_m_1 <= epi_wg_1 * 96 + (unsigned int)(s_1 * 32)) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            break;
                        }
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            unsigned int row = epi_wg_1 * 96 + (unsigned int)(s_1 * 32) + (unsigned int)(j_1 * 8) + warp_in_wg_1 * 2 + lane / 16;
                            if (row < valid_m_1) {
                                if (task_4[0] == 4) {
                                    cached_token[j_1] = pool_m + row;
                                    cached_topk[j_1] = 8;
                                } else {
                                    cached_token[j_1] = TokenMetadata[(pool_m + row) * 3 + 1];
                                    cached_topk[j_1] = TokenMetadata[(pool_m + row) * 3 + 2];
                                }
                            }
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 4; i_2++) {
                            unsigned int address_1 = accum_stage_1 * 192 + epi_wg_1 * 96 + (unsigned int)(s_1 * 32) + (unsigned int)(i_2 * 8);
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
                            if (s_1 == 2 && i_2 == 3) {
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
                            unsigned int write_address = output_base + warp_in_wg_1 / 2 * 32 * 128 + (unsigned int)(i_2 * 8 * 128) + row_1 * 128 + (col ^ row_1) * 16;
                            uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(write_address);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1]))
                                : "memory");
                        }
                        asm volatile("bar.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 4; j_2++) {
                            unsigned int row_in_store = (unsigned int)(j_2 * 8) + warp_in_wg_1 * 2 + lane / 16;
                            unsigned int row_2 = epi_wg_1 * 96 + (unsigned int)(s_1 * 32) + row_in_store;
                            if (row_2 >= valid_m_1) {
                                break;
                            }
                            unsigned int row_in_atom = (warp_in_wg_1 * 2 + lane / 16) % 8;
                            unsigned int read_address = output_base + lane % 16 / 8 * 32 * 128 + row_in_store * 128 + (lane % 8 ^ row_in_atom) * 16;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                : "r"(read_address));
                            unsigned long long dst_index = ((unsigned long long)cached_topk[j_2] * 1920 + (unsigned long long)cached_token[j_2]) * 256 + (unsigned long long)(n_offset_1 / 2) + (unsigned long long)(lane % 16 * 4);
                            reinterpret_cast<int4*>(Combine + dst_index)[0] = reinterpret_cast<int4*>(packed)[0];
                        }
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                }
            }
            if (warp == 8) {
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
            unsigned int epi_warp_2 = warp - 8;
            unsigned int epi_thread = epi_warp_2 * 32 + lane;
            unsigned int phase_3 = 0;
            unsigned int stage = 0;
            unsigned long long peer = 0;
            if (bid == 0 && epi_thread == 0) {
                peer = PeerGrid[0];
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (epi_thread == 0) {
                unsigned int increment_3 = 1;
                if (bid == 0) {
                    increment_3 = 2147483647;
                }
                unsigned int _atomic_old_10;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_10) : "l"(&GridCounters[1]), "r"(static_cast<uint32_t>(increment_3)) : "memory");
                unsigned int old_4 = _atomic_old_10;
                unsigned int _wait_acquire_mask_6;
                do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_6) : "l"((reinterpret_cast<unsigned int*>(GridCounters) + (1))) : "memory");
                } while (((_wait_acquire_mask_6 ^ static_cast<unsigned int>((old_4 ^ 2147483648) & 2147483648)) & static_cast<unsigned int>(2147483648)) != 0);
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
            float reduced[16];
            unsigned int store_values[4];
            #pragma unroll 1
            for (unsigned int token_chunk = epi_warp_2 * 2 + (unsigned int)bid; token_chunk < num_tokens; token_chunk += 16) {
                unsigned int token_1 = token_chunk;
                unsigned int chunk_1 = 0;
                int expert_8 = -1;
                if (lane < 8) {
                    expert_8 = (int)TopK[token_1 * 8 + lane];
                }
                if (lane == 8) {
                    expert_8 = 8;
                }
                unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, expert_8 >= 0);
                unsigned int total_mask = _vote_1;
                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                {
                bool _warp_acquire_ready;
                do {
                _warp_acquire_ready = true;
                if (lane < 8 && expert_8 >= 0) {
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
                if (mask != 0) {
                    int _ffs_1 = __ffs(mask);
                    unsigned int slot_1 = (unsigned int)(_ffs_1 - 1);
                    mask ^= (unsigned int)1 << slot_1;
                    if (elect_sync()) {
                        cp_async_bulk_gmem2smem(reusable_addr + (epi_warp_2 + stage * 8) * 1024, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(CombineBytes) + ((unsigned long long)(((unsigned long long)slot_1 * 1920 + (unsigned long long)token_1) * 512 * 2 + (unsigned long long)(chunk_1 * 1024)) * (unsigned long long)1)), 1024, combine_barriers_addr + (epi_warp_2 * 2 + stage) * 8);
                        mbarrier_arrive_expect_tx(combine_barriers_addr + (epi_warp_2 * 2 + stage) * 8, 1024);
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
                if (total_mask != 0) {
                    #pragma unroll 1
                    for (unsigned int selection = 0; selection < 9; selection++) {
                        unsigned int has_next = (unsigned int)(mask != 0);
                        if (has_next != 0) {
                            int _ffs_2 = __ffs(mask);
                            unsigned int slot_2 = (unsigned int)(_ffs_2 - 1);
                            mask ^= (unsigned int)1 << slot_2;
                            if (elect_sync()) {
                                cp_async_bulk_gmem2smem(reusable_addr + (epi_warp_2 + (stage ^ 1) * 8) * 1024, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(CombineBytes) + ((unsigned long long)(((unsigned long long)slot_2 * 1920 + (unsigned long long)token_1) * 512 * 2 + (unsigned long long)(chunk_1 * 1024)) * (unsigned long long)1)), 1024, combine_barriers_addr + (epi_warp_2 * 2 + (stage ^ 1)) * 8);
                                mbarrier_arrive_expect_tx(combine_barriers_addr + (epi_warp_2 * 2 + (stage ^ 1)) * 8, 1024);
                            }
                            __syncwarp();
                        }
                        mbarrier_wait(combine_barriers_addr + (epi_warp_2 * 2 + stage) * 8, phase_3);
                        #pragma unroll
                        for (unsigned int j_3 = 0; j_3 < 2; j_3++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&values[0])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 3]))
                                : "r"(reusable_addr + (epi_warp_2 + stage * 8) * 1024 + (j_3 * 32 + lane) * 16));
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
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        phase_3 ^= stage;
                        stage ^= 1;
                        if (has_next == 0) {
                            break;
                        }
                    }
                }
                #pragma unroll
                for (unsigned int j_4 = 0; j_4 < 2; j_4++) {
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
                        "r"(reusable_addr + (epi_warp_2 + 16) * 1024 + (j_4 * 32 + lane) * 16), "r"(*reinterpret_cast<uint32_t*>(&store_values[0])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&store_values[(0) + 3])));
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        void* _cpbulk_dst_4 = reinterpret_cast<void*>(Y + ((unsigned long long)token_1 * 512 * 2 + (unsigned long long)(chunk_1 * 1024)));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_4), "r"(reusable_addr + (epi_warp_2 + 16) * 1024), "r"((uint32_t)(1024))
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
