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
#define TMEM_NCOLS 192
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_EPI_PIPE_STAGES 2
#define SMEM_SX_OFF 1024
#define SMEM_SX_STAGE_BYTES 12288
#define SMEM_SX_STRIDE 12288
#define SMEM_SW_OFF 87040
#define SMEM_SW_STAGE_BYTES 16384
#define SMEM_SW_STRIDE 16384
#define SMEM_METADATA_OFF 201728
#define SMEM_METADATA_STAGE_BYTES 3072
#define SMEM_METADATA_STRIDE 3072
#define SMEM_CACHED_COUNTS_OFF 203264
#define SMEM_CACHED_COUNTS_STAGE_BYTES 1536
#define SMEM_CACHED_COUNTS_STRIDE 1536
#define SMEM_TOTAL 204800

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


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

extern "C" {

__global__ __launch_bounds__(896, 1) void
kernel_deepgemm_mega_gate_sm103a_acbb10091c0ba56d92ee(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap W, float* __restrict__ bias, float* __restrict__ image_bias, uint8_t* __restrict__ image_mask, uint8_t* __restrict__ mask, int* __restrict__ physical_map, int* __restrict__ logical_count, long long* __restrict__ topk_idx, long long* __restrict__ unmapped_idx, float* __restrict__ topk_weights, float* __restrict__ scratch, unsigned long long* __restrict__ score_barriers, uint8_t* __restrict__ fixed_mask, uint8_t* __restrict__ random_mask, int num_tokens, int num_shared, int map_width, unsigned int ep_rank, float routed_scale, long long unmapped_stride)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 56)
    #define tmem_full_addr (mbar_base + 112)
    #define tmem_empty_addr (mbar_base + 128)
    #define metadata_ready_addr (mbar_base + 144)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sx_addr = smem + 1024;
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 87040);
    const int sw_addr = smem + 87040;
    float* metadata = reinterpret_cast<float*>(smem_raw + 201728);
    const int metadata_addr = smem + 201728;
    int* cached_counts = reinterpret_cast<int*>(smem_raw + 203264);
    const int cached_counts_addr = smem + 203264;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (0, 4), init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 32, 1);
            // empty: stages (0, 4), init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'epi_pipe' ---
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // tmem_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // metadata_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (1, 5), init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 40, 1);
            // empty: stages (1, 5), init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 96, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (2, 6), init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 48, 1);
            // empty: stages (2, 6), init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 3) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (3,), init_count=1
            mbarrier_init(smem + 24, 1);
            // empty: stages (3,), init_count=1
            mbarrier_init(smem + 80, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 152);
    if (warp == 2) {
        int _tmem_hold = smem + 152;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int ls = 0;
            int logical = bid % 24;
            int expert_group = logical % 3;
            int split = logical / 3;
            unsigned int _phase_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int tile = bid / 24; tile < 6; tile += 6) {
                    int effective_n = ((tile == 5) ? 32 : 96);
                    #pragma unroll 4
                    for (int kb = 0; kb < 10; kb++) {
                        mbarrier_wait(empty_addr + (ls) * 8, _phase_empty);
                        int ko = split * 10 * 64 + kb * 64;
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sx_addr + ls * 12288), "l"((&X)), "r"(0), "r"(tile * 96 + cta_rank * effective_n), "r"(ko / 64),
                               "r"(full_addr + (ls) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        tma_3d_gmem2smem(sw_addr + ls * 16384, (&W), 0, expert_group * 128 + cta_rank * 128, ko / 64, full_addr + (ls) * 8);
                        mbarrier_arrive_expect_tx(full_addr + (ls) * 8, 28672);
                        ls += 1;
                        if (ls == 7) { ls = 0; _phase_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int ms = 0;
            unsigned int me = 0;
            unsigned int _phase_tmem_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int tile_1 = bid / 24; tile_1 < 6; tile_1 += 6) {
                    mbarrier_wait(tmem_empty_addr + (me) * 8, _phase_tmem_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kb_1 = 0; kb_1 < 10; kb_1++) {
                        mbarrier_wait(full_addr + (ms) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init = ((kb_1 == 0) ? 1 : 0);
                        if (tile_1 == 5) {
                            int _mma_a_lo_0 = (((sw_addr) >> 4) & 0x3FFF) + (ms) * 1024;
                            int _mma_b_lo_0 = (((sx_addr) >> 4) & 0x3FFF) + (ms) * 768;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (me * 96))), "r"(((init) ? 0 : 1)));
                        } else {
                            int _mma_a_lo_1 = (((sw_addr) >> 4) & 0x3FFF) + (ms) * 1024;
                            int _mma_b_lo_1 = (((sx_addr) >> 4) & 0x3FFF) + (ms) * 768;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135791760;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (me * 96))), "r"(((init) ? 0 : 1)));
                        }
                        __syncwarp();
                        if (kb_1 == 9) {
                            elect_commit(tmem_full_addr + (me) * 8);
                        }
                        __syncwarp();
                        elect_commit(empty_addr + (ms) * 8);
                        ms += 1;
                        if (ms == 7) { ms = 0; _phase_full ^= 1; }
                    }
                    me += 1;
                    if (me == 2) { me = 0; _phase_tmem_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: cache ----
    if (warp == 3) {
        { // cache_main
            #pragma unroll
            for (int wave = 0; wave < 12; wave++) {
                int expert = (unsigned int)(wave * 32) + lane;
                metadata[expert] = ((expert < 384) ? bias[expert] : 0.0f);
                cached_counts[expert] = ((expert < 384) ? logical_count[expert] : 0);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            mbarrier_arrive(metadata_ready_addr);
        }
    }
    // ---- Role: gate ----
    if (warp >= 4 && warp <= 27) {
        { // gate_main
            unsigned int ge = 0;
            int gw = warp - 4;
            int logical_1 = bid % 24;
            int expert_group_1 = logical_1 % 3;
            int split_1 = logical_1 / 3;
            uint64_t _grid_id_0;
            asm volatile("mov.u64 %0, %%gridid;" : "=l"(_grid_id_0));
            unsigned long long epoch = (_grid_id_0 + 1) * 64;
            mbarrier_wait(metadata_ready_addr, 0);
            unsigned int _phase_tmem_full = 0;
            #pragma unroll 1
            for (int tile_2 = bid / 24; tile_2 < 6; tile_2 += 6) {
                int effective_n_1 = ((tile_2 == 5) ? 32 : 96);
                int _min_0 = ((512) < ((tile_2 + 1) * 96) ? (512) : ((tile_2 + 1) * 96));
                int valid_tokens = _min_0 - tile_2 * 96;
                if (gw == 23 && lane == 31) {
                    if (logical_1 == 0) {
                        asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))), "l"(static_cast<unsigned long long>(epoch)) : "memory");
                    } else {
                        {
                        unsigned long long _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))) : "memory");
                        } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(epoch)) >= static_cast<unsigned long long>(24));
                        }
                    }
                }
                __syncwarp();
                mbarrier_wait(tmem_full_addr + (ge) * 8, _phase_tmem_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int subpartition = gw % 4;
                int num_columns = ((0) ? effective_n_1 / 2 : effective_n_1);
                int token_base = ((0) ? subpartition / 2 * num_columns : 0);
                int expert_atom = ((0) ? subpartition % 2 : subpartition);
                int expert_1 = (unsigned int)(expert_group_1 * 128 + cta_rank * 128 + expert_atom * 32) + lane;
                #pragma unroll 1
                for (int col = gw / 4 * 8; col < num_columns; col += 48) {
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + ge * 96 + (unsigned int)(subpartition * 32 << 16) + (unsigned int)col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int item = 0; item < 8; item++) {
                        float stored = _tmem_load_0[item];
                        scratch[((tile_2 * 8 + split_1) * 96 + token_base + col + item) * 384 + expert_1] = stored;
                    }
                }
                asm volatile("bar.sync 8, 768;" ::: "memory");
                if (gw == 23 && lane == 31) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tmem_empty_addr + (ge) * 8);
                    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                    asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
                    #elif defined(__CUDA_ARCH__)
                    #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
                    #endif
                }
                if (gw == 0 && lane == 0) {
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(epoch + 24)) >= static_cast<unsigned long long>(1));
                    }
                }
                asm volatile("bar.sync 8, 768;" ::: "memory");
                #pragma unroll 1
                for (int token_local = gw * 24 + logical_1; token_local < valid_tokens; token_local += 576) {
                    int token = tile_2 * 96 + token_local;
                    int handled = 0;
                    if (handled == 0) {
                        float rankings[12];
                        float unbiased_values[12];
                        unsigned int permutations[3];
                        int is_image = 0;
                        #pragma unroll
                        for (int wave_1 = 0; wave_1 < 3; wave_1++) {
                            int expert_base = (unsigned int)(wave_1 * 128) + lane * 4;
                            float _vec_load_0[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(scratch + ((tile_2 * 8 * 96 + token_local) * 384 + expert_base) + 0);
                                _vec_load_0[0 + 0] = _v4.x;
                                _vec_load_0[0 + 1] = _v4.y;
                                _vec_load_0[0 + 2] = _v4.z;
                                _vec_load_0[0 + 3] = _v4.w;
                            }
                            #pragma unroll
                            for (int sp = 1; sp < 8; sp++) {
                                float _vec_load_1[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (((tile_2 * 8 + sp) * 96 + token_local) * 384 + expert_base) + 0);
                                    _vec_load_1[0 + 0] = _v4.x;
                                    _vec_load_1[0 + 1] = _v4.y;
                                    _vec_load_1[0 + 2] = _v4.z;
                                    _vec_load_1[0 + 3] = _v4.w;
                                }
                                #pragma unroll
                                for (int value = 0; value < 4; value++) {
                                    _vec_load_0[value] = _vec_load_0[value] + _vec_load_1[value];
                                }
                            }
                            #pragma unroll
                            for (int value_1 = 0; value_1 < 4; value_1++) {
                                rankings[wave_1 * 4 + value_1] = _vec_load_0[value_1];
                            }
                        }
                        #pragma unroll
                        for (int wave_2 = 0; wave_2 < 3; wave_2++) {
                            float exponentials[4];
                            float softplus_values[4];
                            #pragma unroll
                            for (int value_2 = 0; value_2 < 4; value_2++) {
                                float _exp_0 = expf(rankings[wave_2 * 4 + value_2]);
                                exponentials[value_2] = _exp_0;
                            }
                            #pragma unroll
                            for (int value_3 = 0; value_3 < 4; value_3++) {
                                float _log1p_0 = log1pf(exponentials[value_3]);
                                softplus_values[value_3] = _log1p_0;
                            }
                            #pragma unroll
                            for (int value_4 = 0; value_4 < 4; value_4++) {
                                float raw_score = rankings[wave_2 * 4 + value_4];
                                float _sqrt_0;
                                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(((raw_score > 20.0f) ? raw_score : softplus_values[value_4])));
                                rankings[wave_2 * 4 + value_4] = _sqrt_0;
                            }
                            #pragma unroll
                            for (int value_5 = 0; value_5 < 4; value_5++) {
                                int expert_id = (unsigned int)(wave_2 * 128) + lane * 4 + (unsigned int)value_5;
                                float raw = rankings[wave_2 * 4 + value_5];
                                unbiased_values[wave_2 * 4 + value_5] = raw;
                                float ranking_bias = 0.0f;
                                ranking_bias = metadata[expert_id];
                                rankings[wave_2 * 4 + value_5] = ((expert_id < 384) ? raw + ranking_bias : -CUDART_INF_F);
                            }
                        }
                        #pragma unroll
                        for (int wave_3 = 0; wave_3 < 3; wave_3++) {
                            int local_indices[4];
                            #pragma unroll
                            for (int value_6 = 0; value_6 < 4; value_6++) {
                                local_indices[value_6] = value_6;
                            }
                            if (rankings[wave_3 * 4 + 1] > rankings[wave_3 * 4] || rankings[wave_3 * 4 + 1] == rankings[wave_3 * 4] && local_indices[1] < local_indices[0]) {
                                float saved_score = rankings[wave_3 * 4];
                                int saved_idx = local_indices[0];
                                rankings[wave_3 * 4] = rankings[wave_3 * 4 + 1];
                                local_indices[0] = local_indices[1];
                                rankings[wave_3 * 4 + 1] = saved_score;
                                local_indices[1] = saved_idx;
                            }
                            if (rankings[wave_3 * 4 + 3] > rankings[wave_3 * 4 + 2] || rankings[wave_3 * 4 + 3] == rankings[wave_3 * 4 + 2] && local_indices[3] < local_indices[2]) {
                                float saved_score_1 = rankings[wave_3 * 4 + 2];
                                int saved_idx_1 = local_indices[2];
                                rankings[wave_3 * 4 + 2] = rankings[wave_3 * 4 + 3];
                                local_indices[2] = local_indices[3];
                                rankings[wave_3 * 4 + 3] = saved_score_1;
                                local_indices[3] = saved_idx_1;
                            }
                            if (rankings[wave_3 * 4 + 2] > rankings[wave_3 * 4] || rankings[wave_3 * 4 + 2] == rankings[wave_3 * 4] && local_indices[2] < local_indices[0]) {
                                float saved_score_2 = rankings[wave_3 * 4];
                                int saved_idx_2 = local_indices[0];
                                rankings[wave_3 * 4] = rankings[wave_3 * 4 + 2];
                                local_indices[0] = local_indices[2];
                                rankings[wave_3 * 4 + 2] = saved_score_2;
                                local_indices[2] = saved_idx_2;
                            }
                            if (rankings[wave_3 * 4 + 3] > rankings[wave_3 * 4 + 1] || rankings[wave_3 * 4 + 3] == rankings[wave_3 * 4 + 1] && local_indices[3] < local_indices[1]) {
                                float saved_score_3 = rankings[wave_3 * 4 + 1];
                                int saved_idx_3 = local_indices[1];
                                rankings[wave_3 * 4 + 1] = rankings[wave_3 * 4 + 3];
                                local_indices[1] = local_indices[3];
                                rankings[wave_3 * 4 + 3] = saved_score_3;
                                local_indices[3] = saved_idx_3;
                            }
                            if (rankings[wave_3 * 4 + 2] > rankings[wave_3 * 4 + 1] || rankings[wave_3 * 4 + 2] == rankings[wave_3 * 4 + 1] && local_indices[2] < local_indices[1]) {
                                float saved_score_4 = rankings[wave_3 * 4 + 1];
                                int saved_idx_4 = local_indices[1];
                                rankings[wave_3 * 4 + 1] = rankings[wave_3 * 4 + 2];
                                local_indices[1] = local_indices[2];
                                rankings[wave_3 * 4 + 2] = saved_score_4;
                                local_indices[2] = saved_idx_4;
                            }
                            permutations[wave_3] = (unsigned int)(local_indices[0] | local_indices[1] << 2 | local_indices[2] << 4 | local_indices[3] << 6);
                        }
                        unsigned int cursors = 0;
                        int selected = -1;
                        #pragma unroll
                        for (int oi = 0; oi < 6; oi++) {
                            float best_score = -CUDART_INF_F;
                            int best_expert = -1;
                            unsigned int best_wave = 0;
                            #pragma unroll
                            for (int wave_4 = 0; wave_4 < 3; wave_4++) {
                                unsigned int cursor = cursors >> (unsigned int)(wave_4 * 3) & 7;
                                float candidate_score = ((cursor == 4) ? -CUDART_INF_F : (((cursor & 2) != 0) ? (((cursor & 1) != 0) ? rankings[wave_4 * 4 + 3] : rankings[wave_4 * 4 + 2]) : (((cursor & 1) != 0) ? rankings[wave_4 * 4 + 1] : rankings[wave_4 * 4])));
                                unsigned int candidate_offset = permutations[wave_4] >> cursor * 2 & 3;
                                if (candidate_score > best_score) {
                                    best_score = candidate_score;
                                    best_expert = (unsigned int)(wave_4 * 128) + lane * 4 + (unsigned int)(int)candidate_offset;
                                    best_wave = wave_4;
                                }
                            }
                            float _warp_redux_f32_0;
                            asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_0) : "f"(best_score));
                            unsigned int tied = ((best_score == _warp_redux_f32_0 && best_expert >= 0) ? (unsigned int)best_expert : (unsigned int)4294967295);
                            unsigned int _warp_redux_u32_0;
                            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(tied));
                            if ((unsigned int)best_expert == _warp_redux_u32_0) {
                                cursors = cursors + (1 << best_wave * 3);
                            }
                            if (lane == (unsigned int)oi) {
                                selected = (int)_warp_redux_u32_0;
                            }
                        }
                        float selected_score = 0.0f;
                        int selected_value = ((selected >= 0) ? selected / 128 * 4 + selected % 4 : -1);
                        int source_lane = ((selected >= 0) ? selected % 128 / 4 : 0);
                        #pragma unroll
                        for (int vi = 0; vi < 12; vi++) {
                            float _shfl_0;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(unbiased_values[vi]), "r"(source_lane));
                            if (selected_value == vi) {
                                selected_score = _shfl_0;
                            }
                        }
                        int chosen = selected;
                        if (lane >= 6 && lane < (unsigned int)(6 + num_shared)) {
                            chosen = lane + 384 - 6;
                        }
                        int physical = chosen;
                        if (lane < (unsigned int)(6 + num_shared)) {
                            unsigned int duplicates = 0;
                            if (lane < 6) {
                                duplicates = (unsigned int)cached_counts[chosen];
                            } else {
                                duplicates = (unsigned int)logical_count[chosen];
                            }
                            unsigned int duplicate = (ep_rank + (unsigned int)token * 23333) % duplicates;
                            physical = physical_map[(unsigned int)(chosen * map_width) + duplicate];
                        }
                        float total = selected_score;
                        #pragma unroll
                        for (int delta = 0; delta < 5; delta++) {
                            if (16 >> delta < 8) {
                                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, total, 16 >> delta);
                                total = total + _shfl_xor_0;
                            }
                        }
                        float result = selected_score;
                        if (lane < 6) {
                            result = selected_score / (total + 1e-20f) * routed_scale;
                        } else if (lane < (unsigned int)(6 + num_shared)) {
                            result = 1.0f;
                        }
                        if (lane < (unsigned int)(6 + num_shared)) {
                            topk_idx[(unsigned int)(token * (6 + num_shared)) + lane] = (long long)physical;
                            topk_weights[(unsigned int)(token * (6 + num_shared)) + lane] = result;
                        }
                    }
                }
                ge += 1;
                if (ge == 2) { ge = 0; _phase_tmem_full ^= 1; }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    __syncwarp();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(256));
    }

    // Cleanup
}

} // extern "C"
