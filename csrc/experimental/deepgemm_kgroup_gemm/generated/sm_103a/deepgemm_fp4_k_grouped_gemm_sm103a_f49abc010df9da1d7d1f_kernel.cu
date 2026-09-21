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
#define TMEM_NCOLS 504
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SCALE_A_OFFSET 480
#define TMEM_SCALE_B_OFFSET 496
#define NUM_PIPE_STAGES 6
#define NUM_EPI_PIPE_STAGES 2
#define SMEM_EPI_OFF 0
#define SMEM_EPI_STAGE_BYTES 8192
#define SMEM_EPI_STRIDE 8192
#define SMEM_SMEM_A_OFF 16384
#define SMEM_SMEM_A_STAGE_BYTES 15360
#define SMEM_SMEM_A_STRIDE 15360
#define SMEM_SMEM_B_OFF 108544
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 206848
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 219136
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 1024
#define SMEM_TOTAL 225536
#define THREADS 256

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


__device__ __forceinline__ void tcgen05_mma_mxf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4.block_scale"
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_deepgemm_fp4_k_grouped_gemm_sm103a_f49abc010df9da1d7d1f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, int* __restrict__ grouped_layout, int M, int N, int K, int grid_m, int grid_n, int num_groups)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 225280;
    #define full_addr (mbar_base + 0)
    #define sf_full_addr (mbar_base + 48)
    #define empty_addr (mbar_base + 96)
    #define tmem_full_addr (mbar_base + 144)
    #define tmem_empty_addr (mbar_base + 160)
    #define tmem_overlap_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* epi = reinterpret_cast<float*>(smem_raw + 0);
    const int epi_addr = smem + 0;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 16384);
    const int smem_a_addr = smem + 16384;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 108544);
    const int smem_b_addr = smem + 108544;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 206848);
    const int smem_sfa_addr = smem + 206848;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 219136);
    const int smem_sfb_addr = smem + 219136;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[225280..225472)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // Stage-major initialization across barrier declarations.
            mbarrier_init(smem + 225280, 130);
            mbarrier_init(smem + 225328, 1);
            mbarrier_init(smem + 225376, 1);
            mbarrier_init(smem + 225424, 1);
            mbarrier_init(smem + 225440, 256);
            mbarrier_init(smem + 225456, 256);
            mbarrier_init(smem + 225288, 130);
            mbarrier_init(smem + 225336, 1);
            mbarrier_init(smem + 225384, 1);
            mbarrier_init(smem + 225432, 1);
            mbarrier_init(smem + 225448, 256);
            mbarrier_init(smem + 225464, 256);
            mbarrier_init(smem + 225296, 130);
            mbarrier_init(smem + 225344, 1);
            mbarrier_init(smem + 225392, 1);
            mbarrier_init(smem + 225304, 130);
            mbarrier_init(smem + 225352, 1);
            mbarrier_init(smem + 225400, 1);
            mbarrier_init(smem + 225312, 130);
            mbarrier_init(smem + 225360, 1);
            mbarrier_init(smem + 225408, 1);
            mbarrier_init(smem + 225320, 130);
            mbarrier_init(smem + 225368, 1);
            mbarrier_init(smem + 225416, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 504 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 225472);
    if (warp == 2) {
        int _tmem_hold = smem + 225472;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_scale_a = taddr + 480;
    const int tmem_scale_b = taddr + 496;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int stage = 0;
            int current_group = 0;
            int current_k_start = 0;
            int current_sf_start = 0;
            int current_shape_k = grouped_layout[0];
            int current_k_end = current_shape_k;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (unsigned int bid_1 = bid; bid_1 < 1188; bid_1 += 152) {
                while (bid_1 >= (unsigned int)((current_group + 1) * 396)) {
                    int aligned_k = (current_shape_k + 256 - 1) / 256 * 256;
                    current_sf_start += (aligned_k + 127) / 128;
                    current_group += 1;
                    current_k_start = (current_k_end + 256 - 1) / 256 * 256;
                    current_k_end = grouped_layout[current_group];
                    current_shape_k = current_k_end - current_k_start;
                }
                int _max_0 = ((1) > ((current_shape_k + 256 - 1) / 256) ? (1) : ((current_shape_k + 256 - 1) / 256));
                int K_tiles = _max_0;
                unsigned int block_idx = bid_1 - (unsigned int)(current_group * 22 * 18);
                unsigned int blocks_per_group = 352;
                unsigned int group = block_idx / blocks_per_group;
                unsigned int first_n = group * 16;
                unsigned int in_group = block_idx % blocks_per_group;
                int _min_0 = ((16) < (18 - first_n) ? (16) : (18 - first_n));
                unsigned int n_in_group = _min_0;
                unsigned int off_m = in_group / n_in_group * 240;
                unsigned int off_n = (first_n + in_group % n_in_group) * 128;
                for (int k = 0; k < K_tiles; k++) {
                    mbarrier_wait(empty_addr + (stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem(smem_sfa_addr + stage * 2048, (&SFA), off_m, current_sf_start + k * 2, sf_full_addr + (stage) * 8);
                        tma_2d_gmem2smem(smem_sfb_addr + stage * 1024, (&SFB), off_n, current_sf_start + k * 2, sf_full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(sf_full_addr + (stage) * 8, 3072);
                        tma_2d_gmem2smem_cta2(smem_a_addr + stage * 15360, (&A), current_k_start + k * 256, off_m + (unsigned int)(cta_rank * 120), ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_b_addr + stage * 16384, (&B), current_k_start + k * 256, off_n, ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 63488);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 6) { stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int accum_stage = 0;
            unsigned int completed = 0;
            unsigned int _phase_tmem_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                int current_group_1 = 0;
                int current_k_start_1 = 0;
                int current_sf_start_1 = 0;
                int current_shape_k_1 = grouped_layout[0];
                int current_k_end_1 = current_shape_k_1;
                #pragma unroll 1
                for (unsigned int bid_2 = bid; bid_2 < 1188; bid_2 += 152) {
                    while (bid_2 >= (unsigned int)((current_group_1 + 1) * 396)) {
                        int aligned_k_1 = (current_shape_k_1 + 256 - 1) / 256 * 256;
                        current_sf_start_1 += (aligned_k_1 + 127) / 128;
                        current_group_1 += 1;
                        current_k_start_1 = (current_k_end_1 + 256 - 1) / 256 * 256;
                        current_k_end_1 = grouped_layout[current_group_1];
                        current_shape_k_1 = current_k_end_1 - current_k_start_1;
                    }
                    int _max_2 = ((1) > ((current_shape_k_1 + 256 - 1) / 256) ? (1) : ((current_shape_k_1 + 256 - 1) / 256));
                    int K_tiles_1 = _max_2;
                    mbarrier_wait(tmem_empty_addr + (accum_stage) * 8, _phase_tmem_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 4
                    for (int k_1 = 0; k_1 < K_tiles_1; k_1++) {
                        mbarrier_wait(full_addr + (stage_1) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_a, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (stage_1) * 128)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_a + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (stage_1) * 128 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_a + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (stage_1) * 128 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_a + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (stage_1) * 128 + 96)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_b, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (stage_1) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_scale_b + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (stage_1) * 64 + 32)));
                        }
                        __syncwarp();
                        int init_flag = ((k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            int _mma_a_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 960;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 240)), a_desc + 0, b_desc + 0,
                                    0x10bc0480U, tmem_scale_b + 0, tmem_scale_a + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                                tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 240)), a_desc + 2, b_desc + 2,
                                    0x50bc04a0U, tmem_scale_b + 0, tmem_scale_a + 0, 1);
                            }
                            int _mma_a_lo_1 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_1 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_1) * 960;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 240)), a_desc + 0, b_desc + 0,
                                    0x10bc0480U, tmem_scale_b + 4 + 0, tmem_scale_a + 8 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                                tcgen05_mma_mxf4_bs_cta2((tmem_accum + (accum_stage * 240)), a_desc + 2, b_desc + 2,
                                    0x50bc04a0U, tmem_scale_b + 4 + 0, tmem_scale_a + 8 + 0, 1);
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (stage_1) * 8, (uint16_t)(3));
                        if (k_1 == K_tiles_1 - 1) {
                            elect_commit_cg2_multicast(tmem_full_addr + (accum_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        stage_1 += 1;
                        if (stage_1 == 6) { stage_1 = 0; _phase_full ^= 1; }
                    }
                    accum_stage += 1;
                    if (accum_stage == 2) { accum_stage = 0; _phase_tmem_empty ^= 1; }
                    completed = completed + 1;
                }
                if (completed > 0) {
                    mbarrier_wait(tmem_empty_addr + ((completed - 1) % 2) * 8, (completed - 1) / 2 % 2);
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp >= 2 && warp <= 3) {
        { // transpose_main
            unsigned int stage_2 = 0;
            unsigned int sf_subblock = warp - 2;
            int current_group_2 = 0;
            int current_k_start_2 = 0;
            int current_sf_start_2 = 0;
            int current_shape_k_2 = grouped_layout[0];
            int current_k_end_2 = current_shape_k_2;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (unsigned int bid_3 = bid; bid_3 < 1188; bid_3 += 152) {
                while (bid_3 >= (unsigned int)((current_group_2 + 1) * 396)) {
                    int aligned_k_2 = (current_shape_k_2 + 256 - 1) / 256 * 256;
                    current_sf_start_2 += (aligned_k_2 + 127) / 128;
                    current_group_2 += 1;
                    current_k_start_2 = (current_k_end_2 + 256 - 1) / 256 * 256;
                    current_k_end_2 = grouped_layout[current_group_2];
                    current_shape_k_2 = current_k_end_2 - current_k_start_2;
                }
                int _max_1 = ((1) > ((current_shape_k_2 + 256 - 1) / 256) ? (1) : ((current_shape_k_2 + 256 - 1) / 256));
                int K_tiles_2 = _max_1;
                for (int k_2 = 0; k_2 < K_tiles_2; k_2++) {
                    mbarrier_wait(sf_full_addr + (stage_2) * 8, _phase_sf_full);
                    #pragma unroll
                    for (int sf_block = 0; sf_block < 2; sf_block++) {
                        unsigned int words[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[i])) : "r"(smem_sfa_addr + stage_2 * 2048 + sf_subblock * 256 * 4 + (unsigned int)(sf_block * 512) + ((unsigned int)(i * 32) + lane) * 4));
                        }
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_sfa_addr + stage_2 * 2048 + sf_subblock * 256 * 4 + (unsigned int)(sf_block * 512) + lane * 16), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                    }
                    unsigned int words_1[4];
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_1[i_1])) : "r"(smem_sfb_addr + stage_2 * 1024 + sf_subblock * 128 * 4 + ((unsigned int)(i_1 * 32) + lane) * 4));
                    }
                    __syncwarp();
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_sfb_addr + stage_2 * 1024 + sf_subblock * 128 * 4 + lane * 16), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((full_addr + (stage_2) * 8) & 0xFEFFFFFF) : "memory");
                    stage_2 += 1;
                    if (stage_2 == 6) { stage_2 = 0; _phase_sf_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int accum_stage_1 = 0;
            unsigned int store_stage = 0;
            unsigned int warp_0 = warp - 4;
            int current_group_3 = 0;
            int current_k_start_3 = 0;
            int current_sf_start_3 = 0;
            int current_shape_k_3 = grouped_layout[0];
            int current_k_end_3 = current_shape_k_3;
            unsigned int _phase_tmem_full = 0;
            #pragma unroll 1
            for (unsigned int bid_4 = bid; bid_4 < 1188; bid_4 += 152) {
                while (bid_4 >= (unsigned int)((current_group_3 + 1) * 396)) {
                    int aligned_k_3 = (current_shape_k_3 + 256 - 1) / 256 * 256;
                    current_sf_start_3 += (aligned_k_3 + 127) / 128;
                    current_group_3 += 1;
                    current_k_start_3 = (current_k_end_3 + 256 - 1) / 256 * 256;
                    current_k_end_3 = grouped_layout[current_group_3];
                    current_shape_k_3 = current_k_end_3 - current_k_start_3;
                }
                int _max_3 = ((1) > ((current_shape_k_3 + 256 - 1) / 256) ? (1) : ((current_shape_k_3 + 256 - 1) / 256));
                int K_tiles_3 = _max_3;
                unsigned int block_idx_1 = bid_4 - (unsigned int)(current_group_3 * 22 * 18);
                unsigned int blocks_per_group_1 = 352;
                unsigned int group_1 = block_idx_1 / blocks_per_group_1;
                unsigned int first_n_1 = group_1 * 16;
                unsigned int in_group_1 = block_idx_1 % blocks_per_group_1;
                int _min_1 = ((16) < (18 - first_n_1) ? (16) : (18 - first_n_1));
                unsigned int n_in_group_1 = _min_1;
                unsigned int off_m_1 = in_group_1 / n_in_group_1 * 240;
                unsigned int off_n_1 = (first_n_1 + in_group_1 % n_in_group_1) * 128;
                mbarrier_wait(tmem_full_addr + (accum_stage_1) * 8, _phase_tmem_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                for (int s = 0; s < 15; s++) {
                    if (warp_0 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    unsigned int stage_base = epi_addr + store_stage * 8192;
                    #pragma unroll
                    for (int i_2 = 0; i_2 < 2; i_2++) {
                        unsigned int address = taddr + accum_stage_1 * 240 + (unsigned int)(s * 16) + (unsigned int)(i_2 * 8);
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], address);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int element = 0; element < 8; element++) {
                            _tmem_load_0[element] = ((current_shape_k_3 == 0) ? 0.0f : _tmem_load_0[element]);
                        }
                        if (s == 14 && i_2 == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        #pragma unroll
                        for (int row = 0; row < 8; row++) {
                            unsigned int write_addr = stage_base + warp_0 * 16 * 128 + (unsigned int)(i_2 * 8 * 128) + (unsigned int)(row * 128) + (lane / 4 ^ (unsigned int)row) * 16 + lane % 4 * 4;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(write_addr), "r"((__as_u32(_tmem_load_0[row]))));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int atom = 0; atom < 4; atom++) {
                                unsigned int store_addr = stage_base + (unsigned int)(atom * 16 * 128);
                                tma_store_3d((&C_tma), off_n_1 + (unsigned int)(atom * 32), off_m_1 + (unsigned int)(s * 16), current_group_3, store_addr);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    __syncwarp();
                    store_stage = (store_stage + 1) % 2;
                }
                accum_stage_1 += 1;
                if (accum_stage_1 == 2) { accum_stage_1 = 0; _phase_tmem_full ^= 1; }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
    }

    // Cleanup
}

} // extern "C"
