typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "A 64-bit CUDA host ABI is required");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Dsv4TensorMap { uint64_t opaque[16]; };
struct __align__(64) Dsv4TensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Dsv4TensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Dsv4TensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(Dsv4TensorMap) >= alignof(CUtensorMap), "Dsv4TensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define DSV4_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 16
#define TMEM_SFB_OFFSET 48
#define NUM_WORK_PIPE_STAGES 1
#define NUM_RELAY_PIPE_STAGES 1
#define NUM_PRIVATE_PIPE_STAGES 1
#define NUM_K_PIPE_STAGES 3
#define NUM_SFA_PIPE_STAGES 2
#define NUM_SFB_PIPE_STAGES 2
#define NUM_TMEM_SFB_PIPE_STAGES 1
#define NUM_MMA_PIPE_STAGES 2
#define NUM_TMEM_SFA_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 50176
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_EPI_STAGING_OFF 56320
#define SMEM_EPI_STAGING_STAGE_BYTES 128
#define SMEM_EPI_STAGING_STRIDE 128
#define SMEM_SMEM_SFA_OFF 57344
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 61440
#define SMEM_SMEM_SFB_STAGE_BYTES 1280
#define SMEM_SMEM_SFB_STRIDE 1280
#define SMEM_WORK_RESPONSE_OFF 64000
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_PRIVATE_RESPONSE_OFF 64016
#define SMEM_PRIVATE_RESPONSE_STAGE_BYTES 16
#define SMEM_PRIVATE_RESPONSE_STRIDE 16
#define SMEM_RELAY_INBOX_OFF 64032
#define SMEM_RELAY_INBOX_STAGE_BYTES 16
#define SMEM_RELAY_INBOX_STRIDE 16
#define SMEM_TOTAL 64128
#define THREADS 384
#define BLOCK_M 64
#define BLOCK_N 8
#define BLOCK_K 512
#define WEIGHTS_SHUFFLED 1

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


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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

extern "C" {

__global__ __launch_bounds__(384, 3) __cluster_dims__(2,1,1) void
kernel_dsv4_flash_moe_5184_fc1_weight_pdl_overlap_sm100(const __grid_constant__ CUtensorMap A, uint8_t* __restrict__ B, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ SFB, const __grid_constant__ CUtensorMap C, uint8_t* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int* __restrict__ work_counter, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles, uint8_t* __restrict__ SFA_raw, uint8_t* __restrict__ C_raw)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define work_full_addr (mbar_base + 0)
    #define work_empty_addr (mbar_base + 8)
    #define relay_full_addr (mbar_base + 16)
    #define relay_empty_addr (mbar_base + 24)
    #define private_full_addr (mbar_base + 32)
    #define a_full_addr (mbar_base + 40)
    #define b_full_addr (mbar_base + 64)
    #define sfa_full_addr (mbar_base + 88)
    #define sfb_full_addr (mbar_base + 104)
    #define sfa_free_addr (mbar_base + 120)
    #define sfb_free_addr (mbar_base + 136)
    #define tmem_sfa_full_addr (mbar_base + 152)
    #define tmem_sfb_full_addr (mbar_base + 168)
    #define tmem_sfb_free_addr (mbar_base + 176)
    #define k_done_addr (mbar_base + 184)
    #define tmem_sfa_free_addr (mbar_base + 208)
    #define mma_full_addr (mbar_base + 224)
    #define mma_free_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_b_addr = smem + 50176;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 56320);
    const int epi_staging_addr = smem + 56320;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 57344);
    const int smem_sfa_addr = smem + 57344;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 61440);
    const int smem_sfb_addr = smem + 61440;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 64000);
    const int work_response_addr = smem + 64000;
    unsigned int* private_response = reinterpret_cast<unsigned int*>(smem_raw + 64016);
    const int private_response_addr = smem + 64016;
    unsigned int* relay_inbox = reinterpret_cast<unsigned int*>(smem_raw + 64032);
    const int relay_inbox_addr = smem + 64032;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 4) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'work_pipe' ---
            // work_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // work_empty: 1 barriers, init_count=(cta_rank == 0 ? 384 : 256)
            mbarrier_init(smem + 8, (cta_rank == 0 ? 384 : 256));
            // --- pipeline 'relay_pipe' ---
            // relay_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // relay_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'private_pipe' ---
            // private_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // --- pipeline 'k_pipe' ---
            // a_full: 3 barriers, init_count=2
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // b_full: 3 barriers, init_count=2
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            // --- pipeline 'sfa_pipe' ---
            // sfa_full: 2 barriers, init_count=4
            mbarrier_init(smem + 88, 4);
            mbarrier_init(smem + 96, 4);
            // --- pipeline 'sfb_pipe' ---
            // sfb_full: 2 barriers, init_count=2
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            // --- pipeline 'sfa_pipe' ---
            // sfa_free: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // --- pipeline 'sfb_pipe' ---
            // sfb_free: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // --- pipeline 'tmem_sfa_pipe' ---
            // tmem_sfa_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // --- pipeline 'tmem_sfb_pipe' ---
            // tmem_sfb_full: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // tmem_sfb_free: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // --- pipeline 'k_pipe' ---
            // k_done: 3 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // --- pipeline 'tmem_sfa_pipe' ---
            // tmem_sfa_free: 2 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // mma_free: 2 barriers, init_count=4
            mbarrier_init(smem + 240, 4);
            mbarrier_init(smem + 248, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 10) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 16;
    const int tmem_sfb = taddr + 48;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epilogue ----
    if (warp <= 1) {
        { // epilogue_main
            __threadfence();
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            unsigned int acc_stage = 0;
            float quant_pair[8] = {0};
            unsigned int cluster_work = blockIdx.y * (64 / 2) + blockIdx.x / 2;
            unsigned int work_stage = 0;
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _work_iter = 0; _work_iter < 64 / 2 * grid_n + 1; _work_iter++) {
                if (cluster_work < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                    unsigned int m_tile = cluster_work % (unsigned int)(64 / 2) * 2 + (unsigned int)cta_rank;
                    unsigned int n_tile = cluster_work / (unsigned int)(64 / 2);
                    int expert = tile_expert[n_tile];
                    int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * (unsigned int)BLOCK_N;
                    float sc = scale_c[expert];
                    float sg = scale_gate[expert];
                    float cl = clamp_limit[expert];
                    float al = act_alpha[expert];
                    float be = act_beta[expert];
                    float neg_cl = -cl;
                    float beta_sg = be * sg;
                    float alpha_sg = al * sg;
                    mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)(cta_rank * 64 + warp_0 * 32 << 16) + acc_stage * 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 7, 64;" ::: "memory");
                    for (int token_pair = 0; token_pair < 4; token_pair++) {
                        if (valid_rows > token_pair * 2) {
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _tmem_load_0[token_pair * 2], 8);
                            float partner_even = _shfl_xor_0;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _tmem_load_0[token_pair * 2 + 1], 8);
                            float partner_odd = _shfl_xor_1;
                            float up = (((lane_1 & 8) == 0) ? _tmem_load_0[token_pair * 2] : partner_odd);
                            float gate_raw = (((lane_1 & 8) == 0) ? partner_even : _tmem_load_0[token_pair * 2 + 1]);
                            int token = token_pair * 2 + (lane_1 >> 3 & 1);
                            float _max_0 = max_noftz(up, neg_cl);
                            float _min_0 = fminf(_max_0, cl);
                            float lin = _min_0;
                            float _min_1 = fminf(gate_raw, cl);
                            float gate = _min_1;
                            float _expf_0 = __expf(-(alpha_sg * gate));
                            float value = (lin * sc * sg + beta_sg) * gate / (1.0f + _expf_0);
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, value, 16);
                            float other = _shfl_xor_2;
                            float _fabs_0 = fabsf(value);
                            float _fabs_1 = fabsf(other);
                            float _max_1 = max_noftz(_fabs_0, _fabs_1);
                            float block_max = _max_1;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                            float _max_2 = max_noftz(block_max, _shfl_xor_3);
                            block_max = _max_2;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                            float _max_3 = max_noftz(block_max, _shfl_xor_4);
                            block_max = _max_3;
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                            float _max_4 = max_noftz(block_max, _shfl_xor_5);
                            block_max = _max_4;
                            float _fp8_rt_0;
                            uint16_t _e4m3x2_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_0) : "f"(0.0f), "f"(block_max * 0.16666666666666666f));
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _fp8_h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_0));
                            float scale = _fp8_rt_0;
                            float inv_scale = 0.0f;
                            if (scale != 0.0f) {
                                inv_scale = 1.0f / scale;
                            }
                            if (lane_1 < 16) {
                                quant_pair[0] = value * inv_scale;
                                quant_pair[1] = other * inv_scale;
                                uint32_t _slice_lo_mask_0;
                                {
                                    int _lim_1 = 2;
                                    if (_lim_1 <= 0) { _slice_lo_mask_0 = 0u; }
                                    else if (_lim_1 >= 8) { _slice_lo_mask_0 = ((1u << 8) - 1u); }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_1));
                                    }
                                }
                                uint32_t _slice_hi_mask_0;
                                {
                                    int _lim_2 = 8;
                                    if (_lim_2 <= 0) { _slice_hi_mask_0 = 0u; }
                                    else if (_lim_2 >= 8) { _slice_hi_mask_0 = ((1u << 8) - 1u); }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_hi_mask_0) : "r"(_lim_2));
                                    }
                                }
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 0))) quant_pair[0] = 0.0f;
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 1))) quant_pair[1] = 0.0f;
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 2))) quant_pair[2] = 0.0f;
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 3))) quant_pair[3] = 0.0f;
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 4))) quant_pair[4] = 0.0f;
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 5))) quant_pair[5] = 0.0f;
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 6))) quant_pair[6] = 0.0f;
                                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 7))) quant_pair[7] = 0.0f;
                                uint32_t _fp4_0[1];
                                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                                if (token < valid_rows) {
                                    int output_byte = (n_tile * 8 + (unsigned int)token) * (unsigned int)(M_out / 2) + m_tile * 16 + (unsigned int)(warp_0 * 8) + (unsigned int)(lane_1 & 7);
                                    C_raw[output_byte] = _fp4_0[0];
                                }
                            }
                            if ((lane_1 & 23) == 0) {
                                if (token < valid_rows) {
                                    int sf_feature = m_tile * 2 + (unsigned int)warp_0;
                                    int sf_base = n_tile * (unsigned int)(M_out / 64) * 32 + (unsigned int)(sf_feature / 4 * 32) + (unsigned int)(token * 4) + (unsigned int)(sf_feature % 4);
                                    {
                                        unsigned short _fp8_pair;
                                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale));
                                        *(reinterpret_cast<unsigned char*>(SFC + sf_base) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 7, 64;" ::: "memory");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((mma_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                unsigned int valid = work_response[0];
                unsigned int next_linear = work_response[1];
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                _phase_work_full ^= 1;
                if (valid == 0) {
                    break;
                }
                cluster_work = next_linear;
            }
        }
    }
    // ---- Role: copy_sfb ----
    if (warp == 2) {
        { // copy_sfb_main
            unsigned int stage = 0;
            unsigned int sfb_stage = 0;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_tmem_sfb_free = 1;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int cluster_work_1 = blockIdx.y * (64 / 2) + blockIdx.x / 2;
                unsigned int work_stage_1 = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_1 = 0; _work_iter_1 < 64 / 2 * grid_n + 1; _work_iter_1++) {
                    if (cluster_work_1 < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            mbarrier_wait(sfb_full_addr + (sfb_stage) * 8, _phase_sfb_full);
                            mbarrier_wait(tmem_sfb_free_addr + (stage) * 8, _phase_tmem_sfb_free);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 1280)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + stage * 16)), "l"(_tcgen05_cp_desc_0)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 1280 + 256)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 16 + 4))), "l"(_tcgen05_cp_desc_1)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 1280 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 16 + 8))), "l"(_tcgen05_cp_desc_2)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 1280 + 768)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 16 + 12))), "l"(_tcgen05_cp_desc_3)
                                        : "memory");
                                }
                                tcgen05_commit_cg2_multicast(tmem_sfb_full_addr + (stage) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(sfb_free_addr + (sfb_stage) * 8, (uint16_t)(3));
                            }
                            _phase_tmem_sfb_free ^= 1;
                            sfb_stage += 1;
                            if (sfb_stage == 2) { sfb_stage = 0; _phase_sfb_full ^= 1; }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    unsigned int valid_1 = work_response[0];
                    unsigned int next_linear_1 = work_response[1];
                    mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                    _phase_work_full_1 ^= 1;
                    if (valid_1 == 0) {
                        break;
                    }
                    cluster_work_1 = next_linear_1;
                }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 3) {
        { // copy_sfa_main
            unsigned int stage_1 = 0;
            unsigned int sfa_stage = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_tmem_sfa_free = 1;
            unsigned int _phase_work_full_2 = 0;
            if (cta_rank == 0) {
                unsigned int cluster_work_2 = blockIdx.y * (64 / 2) + blockIdx.x / 2;
                unsigned int work_stage_2 = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_2 = 0; _work_iter_2 < 64 / 2 * grid_n + 1; _work_iter_2++) {
                    if (cluster_work_2 < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                            mbarrier_wait(sfa_full_addr + (sfa_stage) * 8, _phase_sfa_full);
                            mbarrier_wait(tmem_sfa_free_addr + (stage_1) * 8, _phase_tmem_sfa_free);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_1 * 16)), "l"(_tcgen05_cp_desc_0)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 16 + 4))), "l"(_tcgen05_cp_desc_1)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 16 + 8))), "l"(_tcgen05_cp_desc_2)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 16 + 12))), "l"(_tcgen05_cp_desc_3)
                                        : "memory");
                                }
                                tcgen05_commit_cg2_multicast(tmem_sfa_full_addr + (stage_1) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(sfa_free_addr + (sfa_stage) * 8, (uint16_t)(3));
                            }
                            stage_1 += 1;
                            if (stage_1 == 2) { stage_1 = 0; _phase_tmem_sfa_free ^= 1; }
                            sfa_stage += 1;
                            if (sfa_stage == 2) { sfa_stage = 0; _phase_sfa_full ^= 1; }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                    unsigned int valid_2 = work_response[0];
                    unsigned int next_linear_2 = work_response[1];
                    mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                    _phase_work_full_2 ^= 1;
                    if (valid_2 == 0) {
                        break;
                    }
                    cluster_work_2 = next_linear_2;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int stage_2 = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int sfb_tmem_stage = 0;
            unsigned int sfa_tmem_stage = 0;
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_tmem_sfb_full = 0;
            unsigned int _phase_work_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int cluster_work_3 = blockIdx.y * (64 / 2) + blockIdx.x / 2;
                unsigned int work_stage_3 = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_3 = 0; _work_iter_3 < 64 / 2 * grid_n + 1; _work_iter_3++) {
                    if (cluster_work_3 < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                        mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            mbarrier_wait(a_full_addr + (stage_2) * 8, _phase_a_full);
                            mbarrier_wait(b_full_addr + (stage_2) * 8, _phase_b_full);
                            mbarrier_wait(tmem_sfa_full_addr + (sfa_tmem_stage) * 8, _phase_tmem_sfa_full);
                            mbarrier_wait(tmem_sfb_full_addr + (sfb_tmem_stage) * 8, _phase_tmem_sfb_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + sfa_tmem_stage * 16 + 0, (unsigned int)tmem_sfb + sfb_tmem_stage * 16 + 0, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + (sfa_tmem_stage * 16 + 2) + 0, (unsigned int)tmem_sfb + (sfb_tmem_stage * 16 + 2) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + (sfa_tmem_stage * 16 + 4) + 0, (unsigned int)tmem_sfb + (sfb_tmem_stage * 16 + 4) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + (sfa_tmem_stage * 16 + 6) + 0, (unsigned int)tmem_sfb + (sfb_tmem_stage * 16 + 4 + 2) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_4 = (((smem_a_addr + 8192) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_4 = (((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + (sfa_tmem_stage * 16 + 8) + 0, (unsigned int)tmem_sfb + (sfb_tmem_stage * 16 + 8) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_5 = (((smem_a_addr + 8224) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_5 = (((smem_b_addr + 1056) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + (sfa_tmem_stage * 16 + 10) + 0, (unsigned int)tmem_sfb + (sfb_tmem_stage * 16 + 8 + 2) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_6 = (((smem_a_addr + 8256) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_6 = (((smem_b_addr + 1088) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + (sfa_tmem_stage * 16 + 12) + 0, (unsigned int)tmem_sfb + (sfb_tmem_stage * 16 + 12) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_7 = (((smem_a_addr + 8288) >> 4) & 0x3FFF) + (stage_2) * 1024;
                                int _mma_b_lo_7 = (((smem_b_addr + 1120) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                        0x8040480U, (unsigned int)tmem_sfa + (sfa_tmem_stage * 16 + 14) + 0, (unsigned int)tmem_sfb + (sfb_tmem_stage * 16 + 12 + 2) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                tcgen05_commit_cg2_multicast(tmem_sfb_free_addr + (sfb_tmem_stage) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(k_done_addr + (stage_2) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(tmem_sfa_free_addr + (sfa_tmem_stage) * 8, (uint16_t)(3));
                                if (iter_k_2 + 1 == K_tiles) {
                                    tcgen05_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                                }
                            }
                            stage_2 += 1;
                            if (stage_2 == 3) { stage_2 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                            _phase_tmem_sfb_full ^= 1;
                            sfa_tmem_stage += 1;
                            if (sfa_tmem_stage == 2) { sfa_tmem_stage = 0; _phase_tmem_sfa_full ^= 1; }
                        }
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                    unsigned int valid_3 = work_response[0];
                    unsigned int next_linear_3 = work_response[1];
                    mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                    _phase_work_full_3 ^= 1;
                    if (valid_3 == 0) {
                        break;
                    }
                    cluster_work_3 = next_linear_3;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 6 && warp <= 7) {
        { // load_b_main
            unsigned int stage_3 = 0;
            unsigned int issue_phase = 1;
            int local_thread = (warp - 6) * 32 + lane;
            int row_stride_bytes = K / 2;
            unsigned int cluster_work_4 = blockIdx.y * (64 / 2) + blockIdx.x / 2;
            unsigned int work_stage_4 = 0;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_4 = 0; _work_iter_4 < 64 / 2 * grid_n + 1; _work_iter_4++) {
                if (cluster_work_4 < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                    unsigned int n_tile_1 = cluster_work_4 / (unsigned int)(64 / 2);
                    int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * (unsigned int)BLOCK_N;
                    unsigned int publish_stage = stage_3;
                    if (K_tiles >= 5) {
                        mbarrier_wait(k_done_addr + (stage_3) * 8, issue_phase);
                        int dst_base = smem_b_addr + stage_3 * 2048;
                        int elt_offset = local_thread * 32;
                        int row = elt_offset / 256;
                        int col = elt_offset % 256;
                        int routed = 0;
                        routed = route_map[n_tile_1 * 8 + (unsigned int)row];
                        int src_base = routed * row_stride_bytes + col / 2;
                        int dst_chunk = elt_offset / 2 ^ row % 8 * 16;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, %0, 0;\n\t"
                            "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                            "}"
                            :: "r"((row < valid_rows_1) ? 1 : 0), "r"(dst_base + dst_chunk), "l"(B + src_base));
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p;\n\t"
                            "setp.ne.b32 p, %0, 0;\n\t"
                            "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                            "}"
                            :: "r"((row < valid_rows_1) ? 1 : 0), "r"(dst_base + 1024 + dst_chunk), "l"(B + (src_base + 128)));
                        asm volatile("cp.async.commit_group;");
                        stage_3 += 1;
                        if (stage_3 == 3) { stage_3 = 0; issue_phase ^= 1; }
                        #pragma unroll 1
                        for (int iter_k_3 = 1; iter_k_3 < K_tiles; iter_k_3++) {
                            mbarrier_wait(k_done_addr + (stage_3) * 8, issue_phase);
                            int dst_base_0 = smem_b_addr + stage_3 * 2048;
                            int elt_offset_1 = local_thread * 32;
                            int row_2 = elt_offset_1 / 256;
                            int col_3 = elt_offset_1 % 256;
                            int routed_4 = 0;
                            routed_4 = route_map[n_tile_1 * 8 + (unsigned int)row_2];
                            int src_base_5 = routed_4 * row_stride_bytes + iter_k_3 * 256 + col_3 / 2;
                            int dst_chunk_6 = elt_offset_1 / 2 ^ row_2 % 8 * 16;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.ne.b32 p, %0, 0;\n\t"
                                "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                                "}"
                                :: "r"((row_2 < valid_rows_1) ? 1 : 0), "r"(dst_base_0 + dst_chunk_6), "l"(B + src_base_5));
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.ne.b32 p, %0, 0;\n\t"
                                "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                                "}"
                                :: "r"((row_2 < valid_rows_1) ? 1 : 0), "r"(dst_base_0 + 1024 + dst_chunk_6), "l"(B + (src_base_5 + 128)));
                            asm volatile("cp.async.commit_group;");
                            stage_3 += 1;
                            if (stage_3 == 3) { stage_3 = 0; issue_phase ^= 1; }
                            asm volatile("cp.async.wait_group 1;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 1, 64;" ::: "memory");
                            if (warp == 6) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((b_full_addr + (publish_stage) * 8) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            publish_stage += 1;
                            if (publish_stage == 3) { publish_stage = 0; }
                        }
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 64;" ::: "memory");
                        if (warp == 6) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((b_full_addr + (publish_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        publish_stage += 1;
                        if (publish_stage == 3) { publish_stage = 0; }
                    } else {
                        #pragma unroll 1
                        for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                            mbarrier_wait(k_done_addr + (stage_3) * 8, issue_phase);
                            int dst_base_1 = smem_b_addr + stage_3 * 2048;
                            int elt_offset_2 = local_thread * 32;
                            int row_1 = elt_offset_2 / 256;
                            int col_1 = elt_offset_2 % 256;
                            int routed_1 = 0;
                            routed_1 = route_map[n_tile_1 * 8 + (unsigned int)row_1];
                            int src_base_1 = routed_1 * row_stride_bytes + iter_k_4 * 256 + col_1 / 2;
                            int dst_chunk_1 = elt_offset_2 / 2 ^ row_1 % 8 * 16;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.ne.b32 p, %0, 0;\n\t"
                                "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                                "}"
                                :: "r"((row_1 < valid_rows_1) ? 1 : 0), "r"(dst_base_1 + dst_chunk_1), "l"(B + src_base_1));
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.ne.b32 p, %0, 0;\n\t"
                                "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                                "}"
                                :: "r"((row_1 < valid_rows_1) ? 1 : 0), "r"(dst_base_1 + 1024 + dst_chunk_1), "l"(B + (src_base_1 + 128)));
                            asm volatile("cp.async.commit_group;");
                            stage_3 += 1;
                            if (stage_3 == 3) { stage_3 = 0; issue_phase ^= 1; }
                            asm volatile("cp.async.wait_group 0;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 1, 64;" ::: "memory");
                            if (warp == 6) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((b_full_addr + (publish_stage) * 8) & 0xFEFFFFFF) : "memory");
                                }
                            }
                            publish_stage += 1;
                            if (publish_stage == 3) { publish_stage = 0; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                unsigned int valid_4 = work_response[0];
                unsigned int next_linear_4 = work_response[1];
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                _phase_work_full_4 ^= 1;
                if (valid_4 == 0) {
                    break;
                }
                cluster_work_4 = next_linear_4;
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 8) {
        { // load_sfb_main
            unsigned int stage_4 = 0;
            unsigned int publish_stage_1 = 0;
            unsigned int zero[1];
            zero[0] = 0;
            #pragma unroll 1
            for (int i = 0; i < 20; i++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + (unsigned int)((i * 32 + lane) * 4)), "r"((zero[0])));
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            unsigned int cluster_work_5 = blockIdx.y * (64 / 2) + blockIdx.x / 2;
            unsigned int work_stage_5 = 0;
            unsigned int _phase_sfb_free = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_5 = 0; _work_iter_5 < 64 / 2 * grid_n + 1; _work_iter_5++) {
                if (cluster_work_5 < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                    unsigned int n_tile_2 = cluster_work_5 / (unsigned int)(64 / 2);
                    int valid_rows_2 = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * (unsigned int)BLOCK_N;
                    #pragma unroll 1
                    for (int iter_k_5 = 0; iter_k_5 < K_tiles; iter_k_5++) {
                        mbarrier_wait(sfb_free_addr + (stage_4) * 8, _phase_sfb_free);
                        int q = lane % 8;
                        int row_3 = lane / 8;
                        int routed_2 = route_map[n_tile_2 * 8 + (unsigned int)(row_3 % 8)];
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_sfb_addr + stage_4 * 1280 + (unsigned int)(q / 2 * 256) + (unsigned int)(row_3 * 16) + (unsigned int)(q % 2 * 8)), "l"(SFB + (routed_2 * (K / 16) + iter_k_5 * 32 + q * 4)), "r"((valid_rows_2 > row_3 % 8) ? 4 : 0));
                        int q_0 = lane % 8;
                        int row_1_1 = lane / 8 + 4;
                        int routed_2_1 = route_map[n_tile_2 * 8 + (unsigned int)(row_1_1 % 8)];
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_sfb_addr + stage_4 * 1280 + (unsigned int)(q_0 / 2 * 256) + (unsigned int)(row_1_1 * 16) + (unsigned int)(q_0 % 2 * 8)), "l"(SFB + (routed_2_1 * (K / 16) + iter_k_5 * 32 + q_0 * 4)), "r"((valid_rows_2 > row_1_1 % 8) ? 4 : 0));
                        asm volatile("cp.async.commit_group;");
                        stage_4 += 1;
                        if (stage_4 == 2) { stage_4 = 0; _phase_sfb_free ^= 1; }
                        if (iter_k_5 > 0) {
                            asm volatile("cp.async.wait_group 1;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((sfb_full_addr + (publish_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            publish_stage_1 += 1;
                            if (publish_stage_1 == 2) { publish_stage_1 = 0; }
                        }
                    }
                    if (K_tiles > 0) {
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((sfb_full_addr + (publish_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        publish_stage_1 += 1;
                        if (publish_stage_1 == 2) { publish_stage_1 = 0; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                unsigned int valid_5 = work_response[0];
                unsigned int next_linear_5 = work_response[1];
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                _phase_work_full_5 ^= 1;
                if (valid_5 == 0) {
                    break;
                }
                cluster_work_5 = next_linear_5;
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 9) {
        { // load_a_main
            unsigned int stage_5 = 0;
            unsigned int cluster_work_6 = blockIdx.y * (64 / 2) + blockIdx.x / 2;
            unsigned int work_stage_6 = 0;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_6 = 0; _work_iter_6 < 64 / 2 * grid_n + 1; _work_iter_6++) {
                if (cluster_work_6 < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                    unsigned int m_tile_1 = cluster_work_6 % (unsigned int)(64 / 2) * 2 + (unsigned int)cta_rank;
                    unsigned int n_tile_3 = cluster_work_6 / (unsigned int)(64 / 2);
                    int expert_1 = tile_expert[n_tile_3];
                    #pragma unroll 1
                    for (int iter_k_6 = 0; iter_k_6 < K_tiles; iter_k_6++) {
                        mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done);
                        if (elect_sync()) {
                            tma_4d_gmem2smem_cta2(smem_a_addr + stage_5 * 16384, (&A), 0, m_tile_1 * 64, iter_k_6 * 2, expert_1, ((a_full_addr + (stage_5) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((a_full_addr + (stage_5) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        }
                        stage_5 += 1;
                        if (stage_5 == 3) { stage_5 = 0; _phase_k_done ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                unsigned int valid_6 = work_response[0];
                unsigned int next_linear_6 = work_response[1];
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                _phase_work_full_6 ^= 1;
                if (valid_6 == 0) {
                    break;
                }
                cluster_work_6 = next_linear_6;
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp >= 10 && warp <= 11) {
        { // load_sfa_main
            int sfa_warp = warp - 10;
            unsigned int stage_6 = 0;
            unsigned int publish_stage_2 = 0;
            unsigned int pending = 0;
            unsigned int zero_sfa[1];
            zero_sfa[0] = 0;
            #pragma unroll 1
            for (int i_1 = 0; i_1 < 16; i_1++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + (unsigned int)(i_1 / 8 * 2048) + (unsigned int)(sfa_warp * 1024) + (unsigned int)(i_1 % 8 * 128) + (unsigned int)(lane * 4)), "r"((zero_sfa[0])));
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            unsigned int cluster_work_7 = blockIdx.y * (64 / 2) + blockIdx.x / 2;
            unsigned int work_stage_7 = 0;
            unsigned int _phase_sfa_free = 1;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_7 = 0; _work_iter_7 < 64 / 2 * grid_n + 1; _work_iter_7++) {
                if (cluster_work_7 < (unsigned int)(64 / 2 * num_non_exiting_ctas[0])) {
                    unsigned int m_tile_2 = cluster_work_7 % (unsigned int)(64 / 2) * 2 + (unsigned int)cta_rank;
                    unsigned int n_tile_4 = cluster_work_7 / (unsigned int)(64 / 2);
                    int expert_2 = tile_expert[n_tile_4];
                    int sf_parent = (unsigned int)(expert_2 * (M_out / 64)) + m_tile_2 / 2;
                    #pragma unroll 1
                    for (int iter_k_7 = 0; iter_k_7 < K_tiles; iter_k_7++) {
                        mbarrier_wait(sfa_free_addr + (stage_6) * 8, _phase_sfa_free);
                        int q_1 = sfa_warp * 4;
                        int sf_source = (unsigned int)((sf_parent * (K / 64) + iter_k_7 * 8 + q_1) * 512 + lane * 16) + m_tile_2 % 2 * 8;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + stage_6 * 2048 + (unsigned int)(q_1 / 2 * 512) + (unsigned int)(lane * 16) + (unsigned int)(q_1 % 2 * 8)), "l"(SFA_raw + sf_source));
                        int q_0_1 = sfa_warp * 4 + 1;
                        int sf_source_1 = (unsigned int)((sf_parent * (K / 64) + iter_k_7 * 8 + q_0_1) * 512 + lane * 16) + m_tile_2 % 2 * 8;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + stage_6 * 2048 + (unsigned int)(q_0_1 / 2 * 512) + (unsigned int)(lane * 16) + (unsigned int)(q_0_1 % 2 * 8)), "l"(SFA_raw + sf_source_1));
                        int q_2 = sfa_warp * 4 + 2;
                        int sf_source_3 = (unsigned int)((sf_parent * (K / 64) + iter_k_7 * 8 + q_2) * 512 + lane * 16) + m_tile_2 % 2 * 8;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + stage_6 * 2048 + (unsigned int)(q_2 / 2 * 512) + (unsigned int)(lane * 16) + (unsigned int)(q_2 % 2 * 8)), "l"(SFA_raw + sf_source_3));
                        int q_4 = sfa_warp * 4 + 3;
                        int sf_source_5 = (unsigned int)((sf_parent * (K / 64) + iter_k_7 * 8 + q_4) * 512 + lane * 16) + m_tile_2 % 2 * 8;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + stage_6 * 2048 + (unsigned int)(q_4 / 2 * 512) + (unsigned int)(lane * 16) + (unsigned int)(q_4 % 2 * 8)), "l"(SFA_raw + sf_source_5));
                        asm volatile("cp.async.commit_group;");
                        stage_6 += 1;
                        if (stage_6 == 2) { stage_6 = 0; _phase_sfa_free ^= 1; }
                        if (pending != 0) {
                            asm volatile("cp.async.wait_group 1;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((sfa_full_addr + (publish_stage_2) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            publish_stage_2 += 1;
                            if (publish_stage_2 == 2) { publish_stage_2 = 0; }
                        } else {
                            pending = 1;
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
                unsigned int valid_7 = work_response[0];
                unsigned int next_linear_7 = work_response[1];
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                _phase_work_full_7 ^= 1;
                if (valid_7 == 0) {
                    break;
                }
                cluster_work_7 = next_linear_7;
            }
            if (pending != 0) {
                asm volatile("cp.async.wait_group 0;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((sfa_full_addr + (publish_stage_2) * 8) & 0xFEFFFFFF) : "memory");
                }
                publish_stage_2 += 1;
                if (publish_stage_2 == 2) { publish_stage_2 = 0; }
                pending = 0;
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 5) {
        { // work_id_main
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_private_full = 0;
            unsigned int _phase_relay_empty = 1;
            unsigned int _phase_work_full_8 = 0;
            if (cta_rank == 0) {
                unsigned int work_stage_8 = 0;
                unsigned int private_stage = 0;
                unsigned int relay_stage = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_8 = 0; _work_iter_8 < 64 / 2 * grid_n + 1; _work_iter_8++) {
                    mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(private_full_addr + (private_stage) * 8, 16);
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.b128"
                                " [%0], [%1];"
                            :: "r"(private_response_addr + private_stage * 16 + 0 * 16), "r"(private_full_addr + private_stage * 8)
                            : "memory");
                    }
                    mbarrier_wait(private_full_addr + (private_stage) * 8, _phase_private_full);
                    uint32_t _clc_valid_0 = 0;
                    uint32_t _clc_ctaid_x_0;
                    uint32_t _clc_ctaid_y_0;
                    uint32_t _clc_ctaid_z_0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                        : "r"(private_response_addr + private_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    _phase_private_full ^= 1;
                    unsigned int publish = 1;
                    unsigned int next_linear_8 = 0;
                    if (_clc_valid_0 != 0) {
                        if (_clc_ctaid_y_0 >= (unsigned int)num_non_exiting_ctas[0]) {
                            publish = 0;
                        } else {
                            next_linear_8 = _clc_ctaid_y_0 * (unsigned int)(64 / 2) + _clc_ctaid_x_0 / 2;
                        }
                    }
                    if (publish != 0) {
                        mbarrier_wait_cluster_hint(relay_empty_addr + (relay_stage) * 8, _phase_relay_empty, 10000000);
                        if (elect_sync()) {
                            work_response[0] = _clc_valid_0;
                            work_response[1] = next_linear_8;
                            work_response[2] = 0;
                            work_response[3] = 0;
                            mbarrier_arrive(work_full_addr + (work_stage_8) * 8);
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(relay_full_addr + relay_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                            uint32_t _mapa_0;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_0) : "r"(relay_inbox_addr), "r"(1));
                            uint32_t _mapa_1;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_1) : "r"(relay_full_addr + relay_stage * 8), "r"(1));
                            asm volatile(
                                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                :: "r"(_mapa_0), "r"(_clc_valid_0), "r"(next_linear_8), "r"(0), "r"(0), "r"(_mapa_1) : "memory");
                        }
                        _phase_relay_empty ^= 1;
                        mbarrier_wait(work_full_addr + (work_stage_8) * 8, _phase_work_full_8);
                        unsigned int valid_8 = work_response[0];
                        unsigned int next_linear_0 = work_response[1];
                        mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                        _phase_work_empty ^= 1;
                        _phase_work_full_8 ^= 1;
                        if (valid_8 == 0) {
                            break;
                        }
                    }
                }
                mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                _phase_work_empty ^= 1;
                _phase_work_full_8 ^= 1;
                mbarrier_wait_cluster_hint(relay_empty_addr + (relay_stage) * 8, _phase_relay_empty, 10000000);
                _phase_relay_empty ^= 1;
            }
            unsigned int _phase_relay_full = 0;
            unsigned int _phase_work_empty_1 = 1;
            if (cta_rank == 1) {
                if (elect_sync()) {
                    unsigned int local_stage = 0;
                    unsigned int inbox_stage = 0;
                    #pragma unroll 1
                    for (unsigned int _relay_iter = 0; _relay_iter < 64 / 2 * grid_n + 1; _relay_iter++) {
                        mbarrier_wait_cluster_hint(relay_full_addr + (inbox_stage) * 8, _phase_relay_full, 10000000);
                        mbarrier_wait(work_empty_addr + (local_stage) * 8, _phase_work_empty_1);
                        unsigned int relay_valid = relay_inbox[0];
                        unsigned int relay_linear = relay_inbox[1];
                        work_response[0] = relay_valid;
                        work_response[1] = relay_linear;
                        work_response[2] = 0;
                        work_response[3] = 0;
                        mbarrier_arrive(work_full_addr + (local_stage) * 8);
                        _phase_work_empty ^= 1;
                        _phase_work_full_8 ^= 1;
                        _phase_work_empty_1 ^= 1;
                        if (relay_valid == 0) {
                            mbarrier_wait(work_empty_addr + (local_stage) * 8, _phase_work_empty_1);
                            _phase_work_empty ^= 1;
                            _phase_work_full_8 ^= 1;
                            _phase_work_empty_1 ^= 1;
                        }
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(relay_empty_addr + inbox_stage * 8), "r"(0) : "memory");
                        _phase_relay_empty ^= 1;
                        _phase_relay_full ^= 1;
                        if (relay_valid == 0) {
                            break;
                        }
                    }
                }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(64));
    }

    // Cleanup
}

} // extern "C"

