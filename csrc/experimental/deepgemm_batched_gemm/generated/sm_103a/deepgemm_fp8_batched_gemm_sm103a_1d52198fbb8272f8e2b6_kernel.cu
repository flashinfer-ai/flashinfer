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
#define TMEM_NCOLS 64
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 32
#define TMEM_SF_B_OFFSET 36
#define NUM_MAIN_PIPE_STAGES 12
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_CD_OFF 0
#define SMEM_CD_STAGE_BYTES 2048
#define SMEM_CD_STRIDE 2048
#define SMEM_A_OFF 4096
#define SMEM_A_STAGE_BYTES 1024
#define SMEM_A_STRIDE 1024
#define SMEM_B_OFF 16384
#define SMEM_B_STAGE_BYTES 16384
#define SMEM_B_STRIDE 16384
#define SMEM_SFA_OFF 212992
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 219136
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_TOTAL 225664
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
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


__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_deepgemm_fp8_batched_gemm_sm103a_1d52198fbb8272f8e2b6(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, unsigned int* __restrict__ SFD, unsigned int M, unsigned int grid_m, unsigned long long sfd_stride, float alpha)
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
    #define sf_full_addr (mbar_base + 96)
    #define empty_addr (mbar_base + 192)
    #define acc_full_addr (mbar_base + 288)
    #define acc_empty_addr (mbar_base + 304)
    #define overlap_empty_addr (mbar_base + 320)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* cd = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int cd_addr = smem + 0;
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 4096);
    const int a_addr = smem + 4096;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 16384);
    const int b_addr = smem + 16384;
    unsigned int* sfa = reinterpret_cast<unsigned int*>(smem_raw + 212992);
    const int sfa_addr = smem + 212992;
    unsigned int* sfb = reinterpret_cast<unsigned int*>(smem_raw + 219136);
    const int sfb_addr = smem + 219136;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 42 barriers)
    // Mbarriers at smem_raw[225280..225616)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // full: 12 barriers, init_count=66
            mbarrier_init(smem + 225280, 66);
            mbarrier_init(smem + 225288, 66);
            mbarrier_init(smem + 225296, 66);
            mbarrier_init(smem + 225304, 66);
            mbarrier_init(smem + 225312, 66);
            mbarrier_init(smem + 225320, 66);
            mbarrier_init(smem + 225328, 66);
            mbarrier_init(smem + 225336, 66);
            mbarrier_init(smem + 225344, 66);
            mbarrier_init(smem + 225352, 66);
            mbarrier_init(smem + 225360, 66);
            mbarrier_init(smem + 225368, 66);
            // sf_full: 12 barriers, init_count=1
            mbarrier_init(smem + 225376, 1);
            mbarrier_init(smem + 225384, 1);
            mbarrier_init(smem + 225392, 1);
            mbarrier_init(smem + 225400, 1);
            mbarrier_init(smem + 225408, 1);
            mbarrier_init(smem + 225416, 1);
            mbarrier_init(smem + 225424, 1);
            mbarrier_init(smem + 225432, 1);
            mbarrier_init(smem + 225440, 1);
            mbarrier_init(smem + 225448, 1);
            mbarrier_init(smem + 225456, 1);
            mbarrier_init(smem + 225464, 1);
            // empty: 12 barriers, init_count=1
            mbarrier_init(smem + 225472, 1);
            mbarrier_init(smem + 225480, 1);
            mbarrier_init(smem + 225488, 1);
            mbarrier_init(smem + 225496, 1);
            mbarrier_init(smem + 225504, 1);
            mbarrier_init(smem + 225512, 1);
            mbarrier_init(smem + 225520, 1);
            mbarrier_init(smem + 225528, 1);
            mbarrier_init(smem + 225536, 1);
            mbarrier_init(smem + 225544, 1);
            mbarrier_init(smem + 225552, 1);
            mbarrier_init(smem + 225560, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 225568, 1);
            mbarrier_init(smem + 225576, 1);
            // acc_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 225584, 256);
            mbarrier_init(smem + 225592, 256);
            // overlap_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 225600, 256);
            mbarrier_init(smem + 225608, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 40 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 225616);
    if (warp == 2) {
        int _tmem_hold = smem + 225616;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 32;
    const int tmem_sf_b = taddr + 36;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (int tile = bid; tile < 8 * grid_m * 8; tile += num_bids) {
                unsigned int head = (unsigned int)tile / (grid_m * 8);
                unsigned int local = (unsigned int)tile % (grid_m * 8);
                unsigned int off_m = local / 8 * 16;
                unsigned int off_n = local % 8 * 128;
                #pragma unroll 4
                for (int kt = 0; kt < 32; kt++) {
                    mbarrier_wait(empty_addr + (stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        unsigned int sf_bytes = 0;
                        if (kt % 4 == 0) {
                            tma_2d_gmem2smem(sfa_addr + stage * 512, (&SFA), off_m, head * 8 + (unsigned int)(kt / 4), sf_full_addr + (stage) * 8);
                            tma_2d_gmem2smem(sfb_addr + stage * 512, (&SFB), off_n, head * 8 + (unsigned int)(kt / 4), sf_full_addr + (stage) * 8);
                            sf_bytes = 1024;
                        }
                        mbarrier_arrive_expect_tx(sf_full_addr + (stage) * 8, sf_bytes);
                        tma_3d_gmem2smem_cta2(a_addr + stage * 1024, (&A), kt * 128, off_m + (unsigned int)(cta_rank * 8), head, ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(b_addr + stage * 16384, (&B), kt * 128, off_n, head, ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 34816);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 12) { stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int acc_stage = 0;
            unsigned int completed = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int tile_1 = bid; tile_1 < 8 * grid_m * 8; tile_1 += num_bids) {
                    mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 4
                    for (int kt_1 = 0; kt_1 < 32; kt_1++) {
                        mbarrier_wait(full_addr + (stage_1) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (kt_1 % 4 == 0) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (stage_1) * 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (stage_1) * 32)));
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            int _mma_a_lo_0 = (((b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_0 = (((a_addr) >> 4) & 0x3FFF) + (stage_1) * 64;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 16)), a_desc + 0, b_desc + 0,
                                    (0x10840000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, ((kt_1 == 0) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 16)), a_desc + 2, b_desc + 2,
                                    (0x10840000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 16)), a_desc + 4, b_desc + 4,
                                    (0x10840000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 16)), a_desc + 6, b_desc + 6,
                                    (0x10840000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_b, tmem_sf_a, 1);
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (stage_1) * 8, (uint16_t)(3));
                        if (kt_1 == 31) {
                            elect_commit_cg2_multicast(acc_full_addr + (acc_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        stage_1 += 1;
                        if (stage_1 == 12) { stage_1 = 0; _phase_full ^= 1; }
                    }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_acc_empty ^= 1; }
                    completed += 1;
                }
                if (completed > 0) {
                    mbarrier_wait(acc_empty_addr + ((completed - 1) % 2) * 8, (completed - 1) / 2 & 1);
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp == 2) {
        { // transpose_main
            unsigned int stage_2 = 0;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (int tile_2 = bid; tile_2 < 8 * grid_m * 8; tile_2 += num_bids) {
                #pragma unroll 1
                for (int kt_2 = 0; kt_2 < 32; kt_2++) {
                    mbarrier_wait(sf_full_addr + (stage_2) * 8, _phase_sf_full);
                    if (kt_2 % 4 == 0) {
                        unsigned int _sf_v[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(sfa_addr + stage_2 * 512 + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(sfa_addr + stage_2 * 512 + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(sfa_addr + stage_2 * 512 + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(sfa_addr + stage_2 * 512 + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfa_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v[3])));
                        unsigned int _sf_v_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"(sfb_addr + stage_2 * 512 + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"(sfb_addr + stage_2 * 512 + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"(sfb_addr + stage_2 * 512 + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"(sfb_addr + stage_2 * 512 + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v_0[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v_0[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v_0[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(sfb_addr + stage_2 * 512 + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v_0[3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((full_addr + (stage_2) * 8) & 0xFEFFFFFF) : "memory");
                    stage_2 += 1;
                    if (stage_2 == 12) { stage_2 = 0; _phase_sf_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: store ----
    if (warp >= 4 && warp <= 7) {
        { // store_main
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            unsigned int warp_0 = warp - 4;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int tile_3 = bid; tile_3 < 8 * grid_m * 8; tile_3 += num_bids) {
                unsigned int head_1 = (unsigned int)tile_3 / (grid_m * 8);
                unsigned int local_1 = (unsigned int)tile_3 % (grid_m * 8);
                unsigned int off_m_1 = local_1 / 8 * 16;
                unsigned int off_n_1 = local_1 % 8 * 128;
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int store_idx = 0; store_idx < 1; store_idx++) {
                    if (warp_0 == 0) {
                        asm volatile("cp.async.bulk.wait_group 1;");
                    }
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    #pragma unroll
                    for (int load_idx = 0; load_idx < 2; load_idx++) {
                        unsigned int address = taddr + acc_stage_1 * 16 + (unsigned int)(store_idx * 16) + (unsigned int)(load_idx * 8);
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
                        if (store_idx == 0 && load_idx == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((acc_empty_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        unsigned int packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        unsigned int second_packed[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                            second_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed[2] = second_packed[0];
                        packed[3] = second_packed[1];
                        uint32_t _bf16x2_abs_0;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(packed[0]));
                        uint32_t _bf16x2_abs_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(packed[1]));
                        uint32_t _bf16x2_max_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_1));
                        uint32_t _bf16x2_abs_2;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(packed[2]));
                        uint32_t _bf16x2_abs_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(packed[3]));
                        uint32_t _bf16x2_max_1;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_abs_2), "r"(_bf16x2_abs_3));
                        uint32_t _bf16x2_max_2;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_max_0), "r"(_bf16x2_max_1));
                        unsigned int pair_max = _bf16x2_max_2;
                        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair_max, 4);
                        uint32_t _bf16x2_max_3;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(pair_max), "r"(_shfl_xor_0));
                        pair_max = _bf16x2_max_3;
                        unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max, 8);
                        uint32_t _bf16x2_max_4;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(pair_max), "r"(_shfl_xor_1));
                        pair_max = _bf16x2_max_4;
                        unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, pair_max, 16);
                        uint32_t _bf16x2_max_5;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(pair_max), "r"(_shfl_xor_2));
                        pair_max = _bf16x2_max_5;
                        unsigned int lower = (pair_max & 65535) + 31 >> 7;
                        unsigned int upper = (pair_max >> 16) + 31 >> 7;
                        lower = ((lower > 113) ? lower : (unsigned int)113) - 8;
                        upper = ((upper > 113) ? upper : (unsigned int)113) - 8;
                        unsigned int inverse = 254 - lower << 7 | 254 - upper << 23;
                        float quantized[8];
                        #pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            uint32_t _bf16x2_mul_0;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(packed[pair]), "r"(inverse));
                            quantized[pair * 2] = __uint_as_float(_bf16x2_mul_0 << 16);
                            quantized[pair * 2 + 1] = __uint_as_float(_bf16x2_mul_0 & 4294901760);
                        }
                        unsigned int out[2];
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
                            out[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(quantized[4]), "f"(quantized[5]),
                                                   "f"(quantized[6]), "f"(quantized[7]));
                            out[1] = _packed;
                        }
                        unsigned int write_addr = cd_addr + store_stage * 2048 + (unsigned int)(load_idx * 8 * 128) + lane % 8 * 128 + (warp_0 * 2 + lane / 8 ^ lane % 8) * 16;
                        uint32_t _stmatrix_b8_addr_0 = static_cast<uint32_t>(write_addr);
                        asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_b8_addr_0), "r"(out[0]), "r"(out[1])
                            : "memory");
                        if (lane < 8) {
                            unsigned int row = off_m_1 + (unsigned int)(store_idx * 16) + (unsigned int)(load_idx * 8) + 2 * (lane % 4) + lane / 4;
                            if (row < M) {
                                unsigned long long sf_index = ((unsigned long long)head_1 * 1024 + (unsigned long long)off_n_1 + (unsigned long long)(warp_0 * 32)) / 32;
                                unsigned long long word_offset = sf_index / 4 * sfd_stride + (unsigned long long)row;
                                unsigned int exponent = ((lane < 4) ? lower : upper);
                                *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(SFD) + (word_offset * 4 + sf_index % 4)) + (0)) = (unsigned char)(exponent);
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_3d((&D), off_n_1, off_m_1 + (unsigned int)(store_idx * 16), head_1, cd_addr + store_stage * 2048);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    store_stage = (store_stage + 1) % 2;
                    __syncwarp();
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_full ^= 1; }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(64));
    }

    // Cleanup
}

} // extern "C"
