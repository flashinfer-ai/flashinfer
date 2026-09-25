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
#define TMEM_NCOLS 160
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 32
#define TMEM_SFB_OFFSET 96
#define NUM_WORK_PIPE_STAGES 1
#define NUM_PRIVATE_PIPE_STAGES 1
#define NUM_K_PIPE_STAGES 2
#define NUM_SFA_PIPE_STAGES 2
#define NUM_SFB_PIPE_STAGES 1
#define NUM_MMA_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 66560
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_SMEM_A_MMA0_OFF 1024
#define SMEM_SMEM_A_MMA0_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA0_STRIDE 32768
#define SMEM_SMEM_A_MMA1_OFF 1056
#define SMEM_SMEM_A_MMA1_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA1_STRIDE 32768
#define SMEM_SMEM_A_MMA2_OFF 1088
#define SMEM_SMEM_A_MMA2_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA2_STRIDE 32768
#define SMEM_SMEM_A_MMA3_OFF 1120
#define SMEM_SMEM_A_MMA3_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA3_STRIDE 32768
#define SMEM_SMEM_A_MMA4_OFF 17408
#define SMEM_SMEM_A_MMA4_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA4_STRIDE 32768
#define SMEM_SMEM_A_MMA5_OFF 17440
#define SMEM_SMEM_A_MMA5_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA5_STRIDE 32768
#define SMEM_SMEM_A_MMA6_OFF 17472
#define SMEM_SMEM_A_MMA6_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA6_STRIDE 32768
#define SMEM_SMEM_A_MMA7_OFF 17504
#define SMEM_SMEM_A_MMA7_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA7_STRIDE 32768
#define SMEM_SMEM_B_MMA0_OFF 66560
#define SMEM_SMEM_B_MMA0_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA0_STRIDE 2048
#define SMEM_SMEM_B_MMA1_OFF 66592
#define SMEM_SMEM_B_MMA1_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA1_STRIDE 2048
#define SMEM_SMEM_B_MMA2_OFF 66624
#define SMEM_SMEM_B_MMA2_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA2_STRIDE 2048
#define SMEM_SMEM_B_MMA3_OFF 66656
#define SMEM_SMEM_B_MMA3_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA3_STRIDE 2048
#define SMEM_SMEM_B_MMA4_OFF 67584
#define SMEM_SMEM_B_MMA4_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA4_STRIDE 2048
#define SMEM_SMEM_B_MMA5_OFF 67616
#define SMEM_SMEM_B_MMA5_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA5_STRIDE 2048
#define SMEM_SMEM_B_MMA6_OFF 67648
#define SMEM_SMEM_B_MMA6_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA6_STRIDE 2048
#define SMEM_SMEM_B_MMA7_OFF 67680
#define SMEM_SMEM_B_MMA7_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA7_STRIDE 2048
#define SMEM_EPI_STAGING_OFF 70656
#define SMEM_EPI_STAGING_STAGE_BYTES 2048
#define SMEM_EPI_STAGING_STRIDE 2048
#define SMEM_EPI_STAGING_U64_OFF 70656
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 2048
#define SMEM_EPI_STAGING_U64_STRIDE 2048
#define SMEM_SMEM_SFA_OFF 72704
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 80896
#define SMEM_SMEM_SFB_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 84992
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_PRIVATE_RESPONSE_OFF 85008
#define SMEM_PRIVATE_RESPONSE_STAGE_BYTES 16
#define SMEM_PRIVATE_RESPONSE_STRIDE 16
#define SMEM_TOTAL 85120
#define THREADS 416
#define BLOCK_M 128
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


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
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

extern "C" {

__global__ __launch_bounds__(416, 2) __cluster_dims__(2,1,1) void
kernel_dsv4_flash_moe_5184_fc2_weight_pdl_overlap_sm100(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ SFB, const __grid_constant__ CUtensorMap C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ scale_c, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int M, int K, int grid_m, int grid_n, int K_tiles, int* __restrict__ total_tiles)
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
    #define private_full_addr (mbar_base + 16)
    #define a_full_addr (mbar_base + 24)
    #define b_full_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 56)
    #define sfb_full_addr (mbar_base + 72)
    #define sfa_free_addr (mbar_base + 80)
    #define sfb_free_addr (mbar_base + 96)
    #define tmem_sfa_full_addr (mbar_base + 104)
    #define tmem_sfb_full_addr (mbar_base + 120)
    #define k_done_addr (mbar_base + 136)
    #define mma_full_addr (mbar_base + 152)
    #define mma_free_addr (mbar_base + 168)

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
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_addr = smem + 66560;
    uint8_t* smem_a_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma0_addr = smem + 1024;
    uint8_t* smem_a_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_a_mma1_addr = smem + 1056;
    uint8_t* smem_a_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_a_mma2_addr = smem + 1088;
    uint8_t* smem_a_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_a_mma3_addr = smem + 1120;
    uint8_t* smem_a_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_a_mma4_addr = smem + 17408;
    uint8_t* smem_a_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_a_mma5_addr = smem + 17440;
    uint8_t* smem_a_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_a_mma6_addr = smem + 17472;
    uint8_t* smem_a_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_a_mma7_addr = smem + 17504;
    uint8_t* smem_b_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma0_addr = smem + 66560;
    uint8_t* smem_b_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 66592);
    const int smem_b_mma1_addr = smem + 66592;
    uint8_t* smem_b_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 66624);
    const int smem_b_mma2_addr = smem + 66624;
    uint8_t* smem_b_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 66656);
    const int smem_b_mma3_addr = smem + 66656;
    uint8_t* smem_b_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_b_mma4_addr = smem + 67584;
    uint8_t* smem_b_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 67616);
    const int smem_b_mma5_addr = smem + 67616;
    uint8_t* smem_b_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 67648);
    const int smem_b_mma6_addr = smem + 67648;
    uint8_t* smem_b_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 67680);
    const int smem_b_mma7_addr = smem + 67680;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 70656);
    const int epi_staging_addr = smem + 70656;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 70656);
    const int epi_staging_u64_addr = smem + 70656;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 72704);
    const int smem_sfa_addr = smem + 72704;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 80896);
    const int smem_sfb_addr = smem + 80896;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 84992);
    const int work_response_addr = smem + 84992;
    unsigned int* private_response = reinterpret_cast<unsigned int*>(smem_raw + 85008);
    const int private_response_addr = smem + 85008;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 23 barriers)
    // Mbarriers at smem_raw[0..184)

    if (warp == 6) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'work_pipe' ---
            // work_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // work_empty: 1 barriers, init_count=640
            mbarrier_init(smem + 8, 640);
            // --- pipeline 'private_pipe' ---
            // private_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'k_pipe' ---
            // a_full: 2 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // b_full: 2 barriers, init_count=2
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // --- pipeline 'sfa_pipe' ---
            // sfa_full: 2 barriers, init_count=2
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            // --- pipeline 'sfb_pipe' ---
            // sfb_full: 1 barriers, init_count=2
            mbarrier_init(smem + 72, 2);
            // --- pipeline 'sfa_pipe' ---
            // sfa_free: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'sfb_pipe' ---
            // sfb_free: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // --- pipeline 'k_pipe' ---
            // tmem_sfa_full: 2 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // tmem_sfb_full: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // k_done: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // mma_free: 2 barriers, init_count=8
            mbarrier_init(smem + 168, 8);
            mbarrier_init(smem + 176, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 160 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184);
    if (warp == 12) {
        int _tmem_hold = smem + 184;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
    const int tmem_sfa = taddr + 32;
    const int tmem_sfb = taddr + 96;
    if (warp < 11) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            const int epi_warp = warp;
            int row = epi_warp * 32;
            int wide_feature = row + lane / 4 * 4;
            int wide_token = lane % 4 * 2;
            float wide_values[4];
            unsigned int wide_packed[2];
            unsigned long long wide_word = 0;
            unsigned int acc_stage = 0;
            unsigned int cluster_work = blockIdx.y * (32 / 2) + blockIdx.x / 2;
            unsigned int work_stage = 0;
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _work_iter = 0; _work_iter < 32 / 2 * grid_n + 1; _work_iter++) {
                if (cluster_work < (unsigned int)(32 / 2 * total_tiles[0])) {
                    unsigned int m_tile = cluster_work % (unsigned int)(32 / 2) * 2 + (unsigned int)cta_rank;
                    unsigned int n_tile = cluster_work / (unsigned int)(32 / 2);
                    int expert = tile_expert[n_tile];
                    float output_scale = scale_c[expert];
                    int mn_limit = tile_mn_limit[n_tile];
                    mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                        : "r"(taddr + (unsigned int)(cta_rank * 128 << 16) + acc_stage * 16));
                    float _tmem_load_1[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                        : "r"(taddr + (unsigned int)(cta_rank * 128 << 16) + 1048576 + acc_stage * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                    int token = wide_token;
                    wide_values[0] = _tmem_load_0[0] * output_scale;
                    wide_values[1] = _tmem_load_0[2] * output_scale;
                    wide_values[2] = _tmem_load_1[0] * output_scale;
                    wide_values[3] = _tmem_load_1[2] * output_scale;
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                        wide_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                    if (wide_feature < 64) {
                        epi_staging_u64[(token * 64 + wide_feature) / 4] = wide_word;
                    } else {
                        epi_staging_u64[(512 + token * 64 + wide_feature - 64) / 4] = wide_word;
                    }
                    int token_0 = wide_token + 1;
                    wide_values[0] = _tmem_load_0[1] * output_scale;
                    wide_values[1] = _tmem_load_0[3] * output_scale;
                    wide_values[2] = _tmem_load_1[1] * output_scale;
                    wide_values[3] = _tmem_load_1[3] * output_scale;
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                        wide_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                    if (wide_feature < 64) {
                        epi_staging_u64[(token_0 * 64 + wide_feature) / 4] = wide_word;
                    } else {
                        epi_staging_u64[(512 + token_0 * 64 + wide_feature - 64) / 4] = wide_word;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            int padding_rows = (8 - mn_limit % 8) % 8;
                            tma_store_4d((&C_tma), m_tile * 128, padding_rows, 1073741824, n_tile * 8 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                            tma_store_4d((&C_tma), m_tile * 128 + 64, padding_rows, 1073741824, n_tile * 8 - (unsigned int)padding_rows + 1073741824, epi_staging_addr + 1024);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
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
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                _phase_work_full ^= 1;
                if (valid == 0) {
                    break;
                }
                cluster_work = next_linear;
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: copy_sfb ----
    if (warp == 4) {
        { // copy_sfb_main
            unsigned int stage = 0;
            unsigned int sfb_stage = 0;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int cluster_work_1 = blockIdx.y * (32 / 2) + blockIdx.x / 2;
                unsigned int work_stage_1 = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_1 = 0; _work_iter_1 < 32 / 2 * grid_n + 1; _work_iter_1++) {
                    if (cluster_work_1 < (unsigned int)(32 / 2 * total_tiles[0])) {
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            mbarrier_wait(sfb_full_addr + (sfb_stage) * 8, _phase_sfb_full);
                            mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + stage * 32)), "l"(_tcgen05_cp_desc_0)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfb_addr + sfb_stage * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                                        : "memory");
                                }
                                tcgen05_commit_cg2_multicast(tmem_sfb_full_addr + (stage) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(sfb_free_addr + (sfb_stage) * 8, (uint16_t)(3));
                            }
                            stage += 1;
                            if (stage == 2) { stage = 0; _phase_k_done ^= 1; }
                            _phase_sfb_full ^= 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    unsigned int valid_1 = work_response[0];
                    unsigned int next_linear_1 = work_response[1];
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
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
    if (warp == 5) {
        { // copy_sfa_main
            unsigned int stage_1 = 0;
            unsigned int sfa_stage = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            if (cta_rank == 0) {
                unsigned int cluster_work_2 = blockIdx.y * (32 / 2) + blockIdx.x / 2;
                unsigned int work_stage_2 = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_2 = 0; _work_iter_2 < 32 / 2 * grid_n + 1; _work_iter_2++) {
                    if (cluster_work_2 < (unsigned int)(32 / 2 * total_tiles[0])) {
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                            mbarrier_wait(sfa_full_addr + (sfa_stage) * 8, _phase_sfa_full);
                            mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_1 * 32)), "l"(_tcgen05_cp_desc_0)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfa_addr + sfa_stage * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_1 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                                        : "memory");
                                }
                                tcgen05_commit_cg2_multicast(tmem_sfa_full_addr + (stage_1) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(sfa_free_addr + (sfa_stage) * 8, (uint16_t)(3));
                            }
                            stage_1 += 1;
                            if (stage_1 == 2) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                            sfa_stage += 1;
                            if (sfa_stage == 2) { sfa_stage = 0; _phase_sfa_full ^= 1; }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                    unsigned int valid_2 = work_response[0];
                    unsigned int next_linear_2 = work_response[1];
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
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
    if (warp == 6) {
        { // mma_main
            unsigned int stage_2 = 0;
            unsigned int readiness_phase = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_work_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int cluster_work_3 = blockIdx.y * (32 / 2) + blockIdx.x / 2;
                unsigned int work_stage_3 = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_3 = 0; _work_iter_3 < 32 / 2 * grid_n + 1; _work_iter_3++) {
                    if (cluster_work_3 < (unsigned int)(32 / 2 * total_tiles[0])) {
                        mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            uint32_t _mbar_token_0 = mbarrier_try_wait(a_full_addr + (stage_2) * 8, readiness_phase);
                            unsigned int a_ready = _mbar_token_0;
                            uint32_t _mbar_token_1 = mbarrier_try_wait(b_full_addr + (stage_2) * 8, readiness_phase);
                            unsigned int b_ready = _mbar_token_1;
                            uint32_t _mbar_token_2 = mbarrier_try_wait(tmem_sfa_full_addr + (stage_2) * 8, readiness_phase);
                            unsigned int sfa_ready = _mbar_token_2;
                            uint32_t _mbar_token_3 = mbarrier_try_wait(tmem_sfb_full_addr + (stage_2) * 8, readiness_phase);
                            unsigned int sfb_ready = _mbar_token_3;
                            mbarrier_wait_token(a_full_addr + (stage_2) * 8, readiness_phase, a_ready);
                            mbarrier_wait_token(b_full_addr + (stage_2) * 8, readiness_phase, b_ready);
                            mbarrier_wait_token(tmem_sfa_full_addr + (stage_2) * 8, readiness_phase, sfa_ready);
                            mbarrier_wait_token(tmem_sfb_full_addr + (stage_2) * 8, readiness_phase, sfb_ready);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                int _mma_a_lo_0 = (((smem_a_mma0_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_0 = (((smem_b_mma0_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + stage_2 * 32 + 0, (unsigned int)tmem_sfb + stage_2 * 32 + 0, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_1 = (((smem_a_mma1_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_1 = (((smem_b_mma1_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + (stage_2 * 32 + 4) + 0, (unsigned int)tmem_sfb + (stage_2 * 32 + 4) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_2 = (((smem_a_mma2_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_2 = (((smem_b_mma2_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + (stage_2 * 32 + 8) + 0, (unsigned int)tmem_sfb + (stage_2 * 32 + 8) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_3 = (((smem_a_mma3_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_3 = (((smem_b_mma3_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + (stage_2 * 32 + 12) + 0, (unsigned int)tmem_sfb + (stage_2 * 32 + 12) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_4 = (((smem_a_mma4_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_4 = (((smem_b_mma4_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + (stage_2 * 32 + 16) + 0, (unsigned int)tmem_sfb + (stage_2 * 32 + 16) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_5 = (((smem_a_mma5_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_5 = (((smem_b_mma5_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + (stage_2 * 32 + 20) + 0, (unsigned int)tmem_sfb + (stage_2 * 32 + 20) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_6 = (((smem_a_mma6_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_6 = (((smem_b_mma6_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + (stage_2 * 32 + 24) + 0, (unsigned int)tmem_sfb + (stage_2 * 32 + 24) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                int _mma_a_lo_7 = (((smem_a_mma7_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                                int _mma_b_lo_7 = (((smem_b_mma7_addr) >> 4) & 0x3FFF) + (stage_2) * 128;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 16)), a_desc + 0, b_desc + 0,
                                        0x10040480U, (unsigned int)tmem_sfa + (stage_2 * 32 + 28) + 0, (unsigned int)tmem_sfb + (stage_2 * 32 + 28) + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                                tcgen05_commit_cg2_multicast(k_done_addr + (stage_2) * 8, (uint16_t)(3));
                                if (iter_k_2 + 1 == K_tiles) {
                                    tcgen05_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                                }
                            }
                            stage_2 += 1;
                            if (stage_2 == 2) { stage_2 = 0; readiness_phase ^= 1; }
                        }
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                    unsigned int valid_3 = work_response[0];
                    unsigned int next_linear_3 = work_response[1];
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
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
    if (warp == 8) {
        { // load_b_main
            unsigned int stage_3 = 0;
            unsigned int cluster_work_4 = blockIdx.y * (32 / 2) + blockIdx.x / 2;
            unsigned int work_stage_4 = 0;
            unsigned int _phase_k_done_2 = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_4 = 0; _work_iter_4 < 32 / 2 * grid_n + 1; _work_iter_4++) {
                if (cluster_work_4 < (unsigned int)(32 / 2 * total_tiles[0])) {
                    unsigned int n_tile_1 = cluster_work_4 / (unsigned int)(32 / 2);
                    #pragma unroll 1
                    for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                        mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
                        if (elect_sync()) {
                            tma_3d_gmem2smem_cta2(smem_b_addr + stage_3 * 2048, (&B), iter_k_3 * 512, 0, n_tile_1, ((b_full_addr + (stage_3) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + stage_3 * 2048 + 1024, (&B), iter_k_3 * 512 + 256, 0, n_tile_1, ((b_full_addr + (stage_3) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage_3) * 8) & 0xFEFFFFFF), "r"((uint32_t)(2048)) : "memory");
                        }
                        stage_3 += 1;
                        if (stage_3 == 2) { stage_3 = 0; _phase_k_done_2 ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                unsigned int valid_4 = work_response[0];
                unsigned int next_linear_4 = work_response[1];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_4 * 8), "r"(0) : "memory");
                _phase_work_full_4 ^= 1;
                if (valid_4 == 0) {
                    break;
                }
                cluster_work_4 = next_linear_4;
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 10) {
        { // load_sfb_main
            unsigned int stage_4 = 0;
            unsigned int zero[1];
            zero[0] = 0;
            #pragma unroll 1
            for (int i = 0; i < 32; i++) {
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + (unsigned int)((i * 32 + lane) * 4)), "r"((zero[0])));
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            unsigned int cluster_work_5 = blockIdx.y * (32 / 2) + blockIdx.x / 2;
            unsigned int work_stage_5 = 0;
            unsigned int _phase_sfb_free = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_5 = 0; _work_iter_5 < 32 / 2 * grid_n + 1; _work_iter_5++) {
                if (cluster_work_5 < (unsigned int)(32 / 2 * total_tiles[0])) {
                    unsigned int n_tile_2 = cluster_work_5 / (unsigned int)(32 / 2);
                    int valid_rows = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * (unsigned int)BLOCK_N;
                    #pragma unroll 1
                    for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                        mbarrier_wait(sfb_free_addr + (stage_4) * 8, _phase_sfb_free);
                        int q = lane % 8;
                        int row_1 = lane / 8;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_sfb_addr + stage_4 * 4096 + (unsigned int)(q * 512) + (unsigned int)(row_1 * 16)), "l"(SFB + ((n_tile_2 * (unsigned int)(K / 64) + (unsigned int)(iter_k_4 * 8) + (unsigned int)q) * 32 + (unsigned int)(row_1 % 8 * 4))), "r"((valid_rows > row_1 % 8) ? 4 : 0));
                        int q_0 = lane % 8;
                        int row_1_1 = lane / 8 + 4;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_sfb_addr + stage_4 * 4096 + (unsigned int)(q_0 * 512) + (unsigned int)(row_1_1 * 16)), "l"(SFB + ((n_tile_2 * (unsigned int)(K / 64) + (unsigned int)(iter_k_4 * 8) + (unsigned int)q_0) * 32 + (unsigned int)(row_1_1 % 8 * 4))), "r"((valid_rows > row_1_1 % 8) ? 4 : 0));
                        int q_2 = lane % 8;
                        int row_3 = lane / 8 + 8;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_sfb_addr + stage_4 * 4096 + (unsigned int)(q_2 * 512) + (unsigned int)(row_3 * 16)), "l"(SFB + ((n_tile_2 * (unsigned int)(K / 64) + (unsigned int)(iter_k_4 * 8) + (unsigned int)q_2) * 32 + (unsigned int)(row_3 % 8 * 4))), "r"((valid_rows > row_3 % 8) ? 4 : 0));
                        int q_4 = lane % 8;
                        int row_5 = lane / 8 + 12;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_sfb_addr + stage_4 * 4096 + (unsigned int)(q_4 * 512) + (unsigned int)(row_5 * 16)), "l"(SFB + ((n_tile_2 * (unsigned int)(K / 64) + (unsigned int)(iter_k_4 * 8) + (unsigned int)q_4) * 32 + (unsigned int)(row_5 % 8 * 4))), "r"((valid_rows > row_5 % 8) ? 4 : 0));
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((sfb_full_addr + (stage_4) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        _phase_sfb_free ^= 1;
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                unsigned int valid_5 = work_response[0];
                unsigned int next_linear_5 = work_response[1];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_5 * 8), "r"(0) : "memory");
                _phase_work_full_5 ^= 1;
                if (valid_5 == 0) {
                    break;
                }
                cluster_work_5 = next_linear_5;
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 11) {
        { // load_a_main
            unsigned int stage_5 = 0;
            unsigned int cluster_work_6 = blockIdx.y * (32 / 2) + blockIdx.x / 2;
            unsigned int work_stage_6 = 0;
            unsigned int _phase_k_done_3 = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_6 = 0; _work_iter_6 < 32 / 2 * grid_n + 1; _work_iter_6++) {
                if (cluster_work_6 < (unsigned int)(32 / 2 * total_tiles[0])) {
                    unsigned int m_tile_1 = cluster_work_6 % (unsigned int)(32 / 2) * 2 + (unsigned int)cta_rank;
                    unsigned int n_tile_3 = cluster_work_6 / (unsigned int)(32 / 2);
                    int expert_1 = tile_expert[n_tile_3];
                    #pragma unroll 1
                    for (int iter_k_5 = 0; iter_k_5 < K_tiles; iter_k_5++) {
                        mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
                        if (elect_sync()) {
                            tma_3d_gmem2smem_cta2(smem_a_addr + stage_5 * 32768, (&A), iter_k_5 * 512, m_tile_1 * 128, expert_1, ((a_full_addr + (stage_5) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_a_addr + stage_5 * 32768 + 16384, (&A), iter_k_5 * 512 + 256, m_tile_1 * 128, expert_1, ((a_full_addr + (stage_5) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((a_full_addr + (stage_5) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        }
                        stage_5 += 1;
                        if (stage_5 == 2) { stage_5 = 0; _phase_k_done_3 ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                unsigned int valid_6 = work_response[0];
                unsigned int next_linear_6 = work_response[1];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_6 * 8), "r"(0) : "memory");
                _phase_work_full_6 ^= 1;
                if (valid_6 == 0) {
                    break;
                }
                cluster_work_6 = next_linear_6;
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 12) {
        { // load_sfa_main
            unsigned int stage_6 = 0;
            unsigned int cluster_work_7 = blockIdx.y * (32 / 2) + blockIdx.x / 2;
            unsigned int work_stage_7 = 0;
            unsigned int _phase_sfa_free = 1;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter_7 = 0; _work_iter_7 < 32 / 2 * grid_n + 1; _work_iter_7++) {
                if (cluster_work_7 < (unsigned int)(32 / 2 * total_tiles[0])) {
                    unsigned int m_tile_2 = cluster_work_7 % (unsigned int)(32 / 2) * 2 + (unsigned int)cta_rank;
                    unsigned int n_tile_4 = cluster_work_7 / (unsigned int)(32 / 2);
                    int expert_2 = tile_expert[n_tile_4];
                    #pragma unroll 1
                    for (int iter_k_6 = 0; iter_k_6 < K_tiles; iter_k_6++) {
                        mbarrier_wait(sfa_free_addr + (stage_6) * 8, _phase_sfa_free);
                        if (elect_sync()) {
                            tma_4d_gmem2smem_cta2(smem_sfa_addr + stage_6 * 4096, (&SFA), 0, 0, iter_k_6 * 8, (unsigned int)(expert_2 * 32) + m_tile_2, ((sfa_full_addr + (stage_6) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((sfa_full_addr + (stage_6) * 8) & 0xFEFFFFFF), "r"((uint32_t)(4096)) : "memory");
                        }
                        stage_6 += 1;
                        if (stage_6 == 2) { stage_6 = 0; _phase_sfa_free ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
                unsigned int valid_7 = work_response[0];
                unsigned int next_linear_7 = work_response[1];
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_7 * 8), "r"(0) : "memory");
                _phase_work_full_7 ^= 1;
                if (valid_7 == 0) {
                    break;
                }
                cluster_work_7 = next_linear_7;
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 7) {
        { // work_id_main
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_private_full = 0;
            unsigned int _phase_work_full_8 = 0;
            if (cta_rank == 0) {
                unsigned int work_stage_8 = 0;
                unsigned int private_stage = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_8 = 0; _work_iter_8 < 32 / 2 * grid_n + 1; _work_iter_8++) {
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
                        if (_clc_ctaid_y_0 >= (unsigned int)total_tiles[0]) {
                            publish = 0;
                        } else {
                            next_linear_8 = _clc_ctaid_y_0 * (unsigned int)(32 / 2) + _clc_ctaid_x_0 / 2;
                        }
                    }
                    if (publish != 0) {
                        if (lane < 2) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + work_stage_8 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                            uint32_t _mapa_0;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_0) : "r"(work_response_addr), "r"(lane));
                            uint32_t _mapa_1;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_1) : "r"(work_full_addr + work_stage_8 * 8), "r"(lane));
                            asm volatile(
                                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                :: "r"(_mapa_0), "r"(_clc_valid_0), "r"(next_linear_8), "r"(0), "r"(0), "r"(_mapa_1) : "memory");
                        }
                        mbarrier_wait(work_full_addr + (work_stage_8) * 8, _phase_work_full_8);
                        unsigned int valid_8 = work_response[0];
                        unsigned int next_linear_0 = work_response[1];
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + work_stage_8 * 8), "r"(0) : "memory");
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
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
    }

    // Cleanup
}

} // extern "C"

