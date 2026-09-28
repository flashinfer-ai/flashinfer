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

#define DEEPGEMM_INF CUDART_INF_F
#define TMEM_NCOLS 396
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SF_Q_OFFSET 384
#define TMEM_TMEM_SF_KV_OFFSET 388
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 10
#define NUM_TMEM_PIPE_STAGES 3
#define SMEM_NEGINF_SCRATCH_OFF 201728
#define SMEM_NEGINF_SCRATCH_STAGE_BYTES 4096
#define SMEM_NEGINF_SCRATCH_STRIDE 4096
#define SMEM_SMEM_Q_FP4_OFF 0
#define SMEM_SMEM_Q_FP4_STAGE_BYTES 8192
#define SMEM_SMEM_Q_FP4_STRIDE 8192
#define SMEM_SMEM_KV_FP4_OFF 24576
#define SMEM_SMEM_KV_FP4_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP4_STRIDE 16384
#define SMEM_SMEM_SF_Q_OFF 188416
#define SMEM_SMEM_SF_Q_STAGE_BYTES 512
#define SMEM_SMEM_SF_Q_STRIDE 512
#define SMEM_SMEM_SF_KV_OFF 189952
#define SMEM_SMEM_SF_KV_STAGE_BYTES 1024
#define SMEM_SMEM_SF_KV_STRIDE 1024
#define SMEM_SMEM_WEIGHTS_OFF 200192
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 512
#define SMEM_SMEM_WEIGHTS_STRIDE 512
#define SMEM_TOTAL 206336

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


__device__ __forceinline__ void tcgen05_mma_mxf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_deepgemm_dense_mqa_sm103a_ffe7cb3c68384cf2be65(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap Weights, const __grid_constant__ CUtensorMap SF_Q, const __grid_constant__ CUtensorMap SF_KV, float* __restrict__ Logits, int* __restrict__ cu_seq_len_k_start, int* __restrict__ cu_seq_len_k_end, int seq_len, int seq_len_kv, int stride_logits, int num_q_blocks, unsigned int* __restrict__ ScheduleMeta)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 205824;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 24)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 128)
    #define umma_full_addr (mbar_base + 208)
    #define umma_empty_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* neginf_scratch = reinterpret_cast<float*>(smem_raw + 201728);
    const int neginf_scratch_addr = smem + 201728;
    uint8_t* smem_q_fp4 = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int smem_q_fp4_addr = smem + 0;
    uint8_t* smem_kv_fp4 = reinterpret_cast<uint8_t*>(smem_raw + 24576);
    const int smem_kv_fp4_addr = smem + 24576;
    uint8_t* smem_sf_q = reinterpret_cast<uint8_t*>(smem_raw + 188416);
    const int smem_sf_q_addr = smem + 188416;
    uint8_t* smem_sf_kv = reinterpret_cast<uint8_t*>(smem_raw + 189952);
    const int smem_sf_kv_addr = smem + 189952;
    uint8_t* smem_weights = reinterpret_cast<uint8_t*>(smem_raw + 200192);
    const int smem_weights_addr = smem + 200192;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SF_Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SF_KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[205824..206080)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 205824, 1);
            mbarrier_init(smem + 205832, 1);
            mbarrier_init(smem + 205840, 1);
            // q_empty: 3 barriers, init_count=288
            mbarrier_init(smem + 205848, 288);
            mbarrier_init(smem + 205856, 288);
            mbarrier_init(smem + 205864, 288);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 10 barriers, init_count=1
            mbarrier_init(smem + 205872, 1);
            mbarrier_init(smem + 205880, 1);
            mbarrier_init(smem + 205888, 1);
            mbarrier_init(smem + 205896, 1);
            mbarrier_init(smem + 205904, 1);
            mbarrier_init(smem + 205912, 1);
            mbarrier_init(smem + 205920, 1);
            mbarrier_init(smem + 205928, 1);
            mbarrier_init(smem + 205936, 1);
            mbarrier_init(smem + 205944, 1);
            // kv_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 205952, 1);
            mbarrier_init(smem + 205960, 1);
            mbarrier_init(smem + 205968, 1);
            mbarrier_init(smem + 205976, 1);
            mbarrier_init(smem + 205984, 1);
            mbarrier_init(smem + 205992, 1);
            mbarrier_init(smem + 206000, 1);
            mbarrier_init(smem + 206008, 1);
            mbarrier_init(smem + 206016, 1);
            mbarrier_init(smem + 206024, 1);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 206032, 1);
            mbarrier_init(smem + 206040, 1);
            mbarrier_init(smem + 206048, 1);
            // umma_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 206056, 128);
            mbarrier_init(smem + 206064, 128);
            mbarrier_init(smem + 206072, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 396 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 206080);
    if (warp == 10) {
        int _tmem_hold = smem + 206080;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sf_q = taddr + 384;
    const int tmem_tmem_sf_kv = taddr + 388;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: math ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // math_main
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            unsigned int math_q_stage = 0;
            unsigned int math_tmem_stage = wg_idx;
            unsigned int first_q = bid;
            unsigned int q_step = num_bids;
            unsigned int split_offset = 0;
            unsigned int remaining = 0;
            first_q = ScheduleMeta[bid * 2];
            split_offset = ScheduleMeta[bid * 2 + 1];
            remaining = ScheduleMeta[2 * num_bids + bid];
            q_step = 1;
            unsigned int first_q_0 = first_q;
            unsigned int q_step_1 = q_step;
            unsigned int split_offset_2 = split_offset;
            unsigned int remaining_3 = remaining;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_umma_full = 0;
            #pragma unroll 1
            for (unsigned int q_block_idx = first_q_0; q_block_idx < num_q_blocks; q_block_idx += q_step_1) {
                if (remaining_3 == 0) {
                    break;
                }
                int q_start = q_block_idx * 4;
                int kv_start = 0;
                unsigned int num_kv_blocks = 0;
                unsigned int span_offset = (3 * num_bids + 1) / 2 * 2;
                kv_start = ScheduleMeta[span_offset + q_block_idx * 2] + split_offset_2 * 256;
                unsigned int span_splits = ScheduleMeta[span_offset + q_block_idx * 2 + 1];
                if (split_offset_2 < span_splits) {
                    unsigned int _min_7 = ((span_splits - split_offset_2) < (remaining_3) ? (span_splits - split_offset_2) : (remaining_3));
                    num_kv_blocks = _min_7;
                }
                remaining_3 -= num_kv_blocks;
                split_offset_2 = 0;
                if (num_kv_blocks > 0) {
                    mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full);
                    float weights_reg[128];
                    int wsmem = smem_weights_addr + math_q_stage * 512;
                    #pragma unroll
                    for (int wi = 0; wi < 32; wi++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[wi * 4])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&weights_reg[(wi * 4) + 3]))
                            : "r"(wsmem + wi * 16));
                    }
                    int seq_start[4];
                    int seq_end[4];
                    {
                        #pragma unroll
                        for (int qi_bound = 0; qi_bound < 4; qi_bound++) {
                            int bound_row = ((q_start + qi_bound < seq_len) ? q_start + qi_bound : seq_len - 1);
                            int raw_start = cu_seq_len_k_start[bound_row];
                            int raw_end = cu_seq_len_k_end[bound_row];
                            seq_start[qi_bound] = ((raw_start < seq_len_kv) ? raw_start : seq_len_kv);
                            seq_end[qi_bound] = ((raw_end < seq_len_kv) ? raw_end : seq_len_kv);
                        }
                    }
                    #pragma unroll 1
                    for (unsigned int kv_iter = 0; kv_iter < num_kv_blocks; kv_iter++) {
                        mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, _phase_umma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int kv_pos = (unsigned int)kv_start + kv_iter * 256 + wg_idx * 128 + (warp % 4 * 32 + lane);
                        #pragma unroll
                        for (int qi = 0; qi < 4; qi++) {
                            int acc_base = taddr + math_tmem_stage * 128 + (unsigned int)(qi * 32) + (warp % 4 * 32 << 16);
                            float _tmem_load_0[32];
                            tmem_ld_x16(&_tmem_load_0[0], acc_base);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            tmem_ld_x16(&_tmem_load_0[16], acc_base + 16);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (qi == 3) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(umma_empty_addr + (math_tmem_stage) * 8);
                            }
                            float _relu_wsum_0;
                            {
                                float2 _sum0 = make_float2(0.0f, 0.0f);
                                float2 _sum1 = make_float2(0.0f, 0.0f);
                                #pragma unroll
                                for (int _j = 0; _j < 32; _j += 4) {
                                    float2 _a0_raw = make_float2(_tmem_load_0[0 + _j], _tmem_load_0[0 + _j + 1]);
                                    float2 _a0_abs = make_float2(fabsf(_tmem_load_0[0 + _j]), fabsf(_tmem_load_0[0 + _j + 1]));
                                    float2 _a0;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a0) : "l"(*(const unsigned long long*)&_a0_raw), "l"(*(const unsigned long long*)&_a0_abs));
                                    float2 _b0 = make_float2(weights_reg[qi * 32 + _j], weights_reg[qi * 32 + _j + 1]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                    float2 _a1_raw = make_float2(_tmem_load_0[0 + _j + 2], _tmem_load_0[0 + _j + 3]);
                                    float2 _a1_abs = make_float2(fabsf(_tmem_load_0[0 + _j + 2]), fabsf(_tmem_load_0[0 + _j + 3]));
                                    float2 _a1;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                    float2 _b1 = make_float2(weights_reg[qi * 32 + _j + 2], weights_reg[qi * 32 + _j + 3]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                }
                                float2 _sum;
                                asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                            }
                            float weighted_sum = _relu_wsum_0;
                            int q_row = q_start + qi;
                            unsigned long long q_offset = (unsigned long long)q_row * (unsigned long long)stride_logits;
                            unsigned long long out_elem = q_offset + (unsigned long long)kv_pos;
                            {
                                weighted_sum = ((kv_pos >= seq_start[qi] && kv_pos < seq_end[qi]) ? weighted_sum : -CUDART_INF_F);
                            }
                            *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = weighted_sum;
                        }
                        math_tmem_stage += 1;
                        if (math_tmem_stage == 3) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                        math_tmem_stage += 1;
                        if (math_tmem_stage == 3) { math_tmem_stage = 0; _phase_umma_full ^= 1; }
                    }
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(q_empty_addr + (math_q_stage) * 8);
                    math_q_stage += 1;
                    if (math_q_stage == 3) { math_q_stage = 0; _phase_q_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_q ----
    if (warp == 8) {
        { // load_q_main
            unsigned int load_q_stage = 0;
            unsigned int _phase_q_empty = 1;
            if (elect_sync()) {
                unsigned int first_q_1 = bid;
                unsigned int q_step_2 = num_bids;
                unsigned int split_offset_1 = 0;
                unsigned int remaining_1 = 0;
                first_q_1 = ScheduleMeta[bid * 2];
                split_offset_1 = ScheduleMeta[bid * 2 + 1];
                remaining_1 = ScheduleMeta[2 * num_bids + bid];
                q_step_2 = 1;
                unsigned int first_q_0_1 = first_q_1;
                unsigned int q_step_1_1 = q_step_2;
                unsigned int split_offset_2_1 = split_offset_1;
                unsigned int remaining_3_1 = remaining_1;
                #pragma unroll 1
                for (unsigned int q_block_idx_1 = first_q_0_1; q_block_idx_1 < num_q_blocks; q_block_idx_1 += q_step_1_1) {
                    if (remaining_3_1 == 0) {
                        break;
                    }
                    int q_start_1 = q_block_idx_1 * 4;
                    int kv_start_1 = 0;
                    unsigned int num_kv_blocks_1 = 0;
                    unsigned int span_offset_1 = (3 * num_bids + 1) / 2 * 2;
                    kv_start_1 = ScheduleMeta[span_offset_1 + q_block_idx_1 * 2] + split_offset_2_1 * 256;
                    unsigned int span_splits_1 = ScheduleMeta[span_offset_1 + q_block_idx_1 * 2 + 1];
                    if (split_offset_2_1 < span_splits_1) {
                        unsigned int _min_4 = ((span_splits_1 - split_offset_2_1) < (remaining_3_1) ? (span_splits_1 - split_offset_2_1) : (remaining_3_1));
                        num_kv_blocks_1 = _min_4;
                    }
                    remaining_3_1 -= num_kv_blocks_1;
                    split_offset_2_1 = 0;
                    if (num_kv_blocks_1 > 0) {
                        mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                        tma_2d_gmem2smem(smem_q_fp4_addr + load_q_stage * 8192, (&Q), 0, q_block_idx_1 * 128, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_sf_q_addr + load_q_stage * 512, (&SF_Q), 0, q_block_idx_1 * 32, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + load_q_stage * 512, (&Weights), 0, q_block_idx_1 * 4, q_full_addr + (load_q_stage) * 8);
                        mbarrier_arrive_expect_tx(q_full_addr + (load_q_stage) * 8, 9216);
                        load_q_stage += 1;
                        if (load_q_stage == 3) { load_q_stage = 0; _phase_q_empty ^= 1; }
                    }
                }
            }
            __syncwarp();
        }
    }
    // ---- Role: load_kv ----
    if (warp == 9) {
        { // load_kv_main
            unsigned int load_kv_stage = 0;
            unsigned int first_q_2 = bid;
            unsigned int q_step_3 = num_bids;
            unsigned int split_offset_3 = 0;
            unsigned int remaining_2 = 0;
            first_q_2 = ScheduleMeta[bid * 2];
            split_offset_3 = ScheduleMeta[bid * 2 + 1];
            remaining_2 = ScheduleMeta[2 * num_bids + bid];
            q_step_3 = 1;
            unsigned int first_q_0_2 = first_q_2;
            unsigned int q_step_1_2 = q_step_3;
            unsigned int split_offset_2_2 = split_offset_3;
            unsigned int remaining_3_2 = remaining_2;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_2 = first_q_0_2; q_block_idx_2 < num_q_blocks; q_block_idx_2 += q_step_1_2) {
                if (remaining_3_2 == 0) {
                    break;
                }
                int q_start_2 = q_block_idx_2 * 4;
                int kv_start_2 = 0;
                unsigned int num_kv_blocks_2 = 0;
                unsigned int span_offset_2 = (3 * num_bids + 1) / 2 * 2;
                kv_start_2 = ScheduleMeta[span_offset_2 + q_block_idx_2 * 2] + split_offset_2_2 * 256;
                unsigned int span_splits_2 = ScheduleMeta[span_offset_2 + q_block_idx_2 * 2 + 1];
                if (split_offset_2_2 < span_splits_2) {
                    unsigned int _min_5 = ((span_splits_2 - split_offset_2_2) < (remaining_3_2) ? (span_splits_2 - split_offset_2_2) : (remaining_3_2));
                    num_kv_blocks_2 = _min_5;
                }
                remaining_3_2 -= num_kv_blocks_2;
                split_offset_2_2 = 0;
                if (num_kv_blocks_2 > 0) {
                    #pragma unroll 1
                    for (unsigned int kv_iter_1 = 0; kv_iter_1 < num_kv_blocks_2; kv_iter_1++) {
                        int kv_row = (unsigned int)kv_start_2 + kv_iter_1 * 256;
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            tma_2d_gmem2smem(smem_kv_fp4_addr + load_kv_stage * 16384, (&KV), 0, kv_row, kv_full_addr + (load_kv_stage) * 8);
                            tma_2d_gmem2smem(smem_sf_kv_addr + load_kv_stage * 1024, (&SF_KV), 0, kv_row / 4, kv_full_addr + (load_kv_stage) * 8);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 17408);
                            load_kv_stage += 1;
                            if (load_kv_stage == 10) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        __syncwarp();
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 10) {
        { // mma_main
            unsigned int mma_q_stage = 0;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_tmem_stage = 0;
            unsigned int first_q_3 = bid;
            unsigned int q_step_4 = num_bids;
            unsigned int split_offset_4 = 0;
            unsigned int remaining_4 = 0;
            first_q_3 = ScheduleMeta[bid * 2];
            split_offset_4 = ScheduleMeta[bid * 2 + 1];
            remaining_4 = ScheduleMeta[2 * num_bids + bid];
            q_step_4 = 1;
            unsigned int first_q_0_3 = first_q_3;
            unsigned int q_step_1_3 = q_step_4;
            unsigned int split_offset_2_3 = split_offset_4;
            unsigned int remaining_3_3 = remaining_4;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_3 = first_q_0_3; q_block_idx_3 < num_q_blocks; q_block_idx_3 += q_step_1_3) {
                if (remaining_3_3 == 0) {
                    break;
                }
                int q_start_3 = q_block_idx_3 * 4;
                int kv_start_3 = 0;
                unsigned int num_kv_blocks_3 = 0;
                unsigned int span_offset_3 = (3 * num_bids + 1) / 2 * 2;
                kv_start_3 = ScheduleMeta[span_offset_3 + q_block_idx_3 * 2] + split_offset_2_3 * 256;
                unsigned int span_splits_3 = ScheduleMeta[span_offset_3 + q_block_idx_3 * 2 + 1];
                if (split_offset_2_3 < span_splits_3) {
                    unsigned int _min_6 = ((span_splits_3 - split_offset_2_3) < (remaining_3_3) ? (span_splits_3 - split_offset_2_3) : (remaining_3_3));
                    num_kv_blocks_3 = _min_6;
                }
                remaining_3_3 -= num_kv_blocks_3;
                split_offset_2_3 = 0;
                if (num_kv_blocks_3 > 0) {
                    mbarrier_wait(q_full_addr + (mma_q_stage) * 8, _phase_q_full_1);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    unsigned int _sf_v[4];
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(smem_sf_q_addr + mma_q_stage * 512 + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                    __syncwarp();
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v[0])));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v[1])));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v[2])));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sf_q_addr + mma_q_stage * 512 + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v[3])));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sf_q, make_sf_cp_desc_lo_sbo128((((smem_sf_q_addr) >> 4) + (mma_q_stage) * 32)));
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (unsigned int kv_iter_2 = 0; kv_iter_2 < num_kv_blocks_3; kv_iter_2++) {
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        int sfkv_smem = smem_sf_kv_addr + mma_kv_stage * 1024;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        unsigned int _sf_v_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"((unsigned int)sfkv_smem + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"((unsigned int)sfkv_smem + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"((unsigned int)sfkv_smem + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"((unsigned int)sfkv_smem + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfkv_smem + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v_0[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfkv_smem + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v_0[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfkv_smem + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v_0[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfkv_smem + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v_0[3])));
                        unsigned int _sf_v_1[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[0])) : "r"((unsigned int)(sfkv_smem + 512) + ((lane >> 3 ^ 0) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[1])) : "r"((unsigned int)(sfkv_smem + 512) + ((lane >> 3 ^ 1) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[2])) : "r"((unsigned int)(sfkv_smem + 512) + ((lane >> 3 ^ 2) * 32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_1[3])) : "r"((unsigned int)(sfkv_smem + 512) + ((lane >> 3 ^ 3) * 32 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfkv_smem + 512) + (lane * 4 + (lane >> 3 ^ 0)) * 4), "r"((_sf_v_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfkv_smem + 512) + (lane * 4 + (lane >> 3 ^ 1)) * 4), "r"((_sf_v_1[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfkv_smem + 512) + (lane * 4 + (lane >> 3 ^ 2)) * 4), "r"((_sf_v_1[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfkv_smem + 512) + (lane * 4 + (lane >> 3 ^ 3)) * 4), "r"((_sf_v_1[3])));
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_tmem_sf_kv, make_sf_cp_desc_lo_sbo128((((smem_sf_kv_addr) >> 4) + (mma_kv_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4((tmem_tmem_sf_kv + 4), make_sf_cp_desc_lo_sbo128((((smem_sf_kv_addr) >> 4) + (mma_kv_stage) * 64 + 32)));
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_kv_fp4_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
                            int _mma_b_lo_0 = (((smem_q_fp4_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 512;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x8a00480U, tmem_tmem_sf_kv + 0, tmem_tmem_sf_q + 0, 0);
                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x48a004a0U, tmem_tmem_sf_kv + 0, tmem_tmem_sf_q + 0, 1);
                            }
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = (((smem_kv_fp4_addr + 8192) >> 4) & 0x3FFF) + (mma_kv_stage) * 1024;
                            int _mma_b_lo_1 = (((smem_q_fp4_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 512;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 0, b_desc + 0,
                                    0x8a00480U, tmem_tmem_sf_kv + 4 + 0, tmem_tmem_sf_q + 0, 0);
                                tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mma_tmem_stage * 128)), a_desc + 2, b_desc + 2,
                                    0x48a004a0U, tmem_tmem_sf_kv + 4 + 0, tmem_tmem_sf_q + 0, 1);
                            }
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                        }
                        __syncwarp();
                        elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 10) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                    mbarrier_arrive(q_empty_addr + (mma_q_stage) * 8);
                    mma_q_stage += 1;
                    if (mma_q_stage == 3) { mma_q_stage = 0; _phase_q_full_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 11) {
        { // idle_main
            {
                float neg_inf[4];
                #pragma unroll
                for (int component = 0; component < 4; component++) {
                    neg_inf[component] = -CUDART_INF_F;
                }
                for (int scratch_j = lane; scratch_j < 1024; scratch_j += 32) {
                    neginf_scratch[scratch_j] = -CUDART_INF_F;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                unsigned int clean_tiles_per_row = (stride_logits + 4095) / 4096;
                unsigned int clean_tasks = (unsigned int)num_q_blocks * clean_tiles_per_row;
                #pragma unroll 1
                for (unsigned int clean_task = bid; clean_task < clean_tasks; clean_task += num_bids) {
                    unsigned int q_block_idx_4 = clean_task / clean_tiles_per_row;
                    int clean_begin = clean_task % clean_tiles_per_row * 4096;
                    int clean_end_raw = clean_begin + 4096;
                    int _min_0 = ((clean_end_raw) < (stride_logits) ? (clean_end_raw) : (stride_logits));
                    int clean_end = _min_0;
                    int q_start_4 = q_block_idx_4 * 4;
                    int ks0 = cu_seq_len_k_start[q_start_4];
                    int ke0 = cu_seq_len_k_end[q_start_4];
                    int kv_start_acc = ((ks0 < seq_len_kv) ? ks0 : seq_len_kv);
                    int kv_end_acc = ((ke0 < seq_len_kv) ? ke0 : seq_len_kv);
                    int row_idx = ((q_start_4 + 1 < seq_len) ? q_start_4 + 1 : seq_len - 1);
                    int ks = cu_seq_len_k_start[row_idx];
                    int ke = cu_seq_len_k_end[row_idx];
                    int sk = ((ks < seq_len_kv) ? ks : seq_len_kv);
                    int se = ((ke < seq_len_kv) ? ke : seq_len_kv);
                    kv_start_acc = ((sk < kv_start_acc) ? sk : kv_start_acc);
                    kv_end_acc = ((se > kv_end_acc) ? se : kv_end_acc);
                    int row_idx_0 = ((q_start_4 + 2 < seq_len) ? q_start_4 + 2 : seq_len - 1);
                    int ks_1 = cu_seq_len_k_start[row_idx_0];
                    int ke_2 = cu_seq_len_k_end[row_idx_0];
                    int sk_3 = ((ks_1 < seq_len_kv) ? ks_1 : seq_len_kv);
                    int se_4 = ((ke_2 < seq_len_kv) ? ke_2 : seq_len_kv);
                    kv_start_acc = ((sk_3 < kv_start_acc) ? sk_3 : kv_start_acc);
                    kv_end_acc = ((se_4 > kv_end_acc) ? se_4 : kv_end_acc);
                    int row_idx_5 = ((q_start_4 + 3 < seq_len) ? q_start_4 + 3 : seq_len - 1);
                    int ks_6 = cu_seq_len_k_start[row_idx_5];
                    int ke_7 = cu_seq_len_k_end[row_idx_5];
                    int sk_8 = ((ks_6 < seq_len_kv) ? ks_6 : seq_len_kv);
                    int se_9 = ((ke_7 < seq_len_kv) ? ke_7 : seq_len_kv);
                    kv_start_acc = ((sk_8 < kv_start_acc) ? sk_8 : kv_start_acc);
                    kv_end_acc = ((se_9 > kv_end_acc) ? se_9 : kv_end_acc);
                    int kv_start_4 = kv_start_acc / 4 * 4;
                    unsigned int num_kv_blocks_4 = (kv_end_acc - kv_start_4 + 256 - 1) / 256;
                    int raw_end_1 = (unsigned int)kv_start_4 + num_kv_blocks_4 * 256;
                    int coverage_end = ((raw_end_1 < stride_logits) ? raw_end_1 : stride_logits);
                    #pragma unroll 1
                    for (int qi_1 = 0; qi_1 < 4; qi_1++) {
                        unsigned long long row_base = (unsigned long long)(q_start_4 + qi_1) * (unsigned long long)stride_logits;
                        int _min_1 = ((kv_start_4) < (clean_end) ? (kv_start_4) : (clean_end));
                        int aligned_start = (clean_begin + 3) / 4 * 4;
                        int aligned_end = _min_1 / 4 * 4;
                        if (aligned_start >= aligned_end) {
                            for (int j = (unsigned int)clean_begin + lane; j < _min_1; j += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                        } else {
                            for (int j_1 = (unsigned int)clean_begin + lane; j_1 < aligned_start; j_1 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_1)) + (0)) = -CUDART_INF_F;
                            }
                            for (int j_2 = (unsigned int)aligned_end + lane; j_2 < _min_1; j_2 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_2)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                            if (elect_sync()) {
                                for (int j_3 = aligned_start; j_3 < aligned_end; j_3 += 1024) {
                                    int _min_2 = ((aligned_end - j_3) < (1024) ? (aligned_end - j_3) : (1024));
                                    int bulk_elems = _min_2;
                                    {
                                        void* _cpbulk_dst_0 = reinterpret_cast<void*>(Logits + (row_base + (unsigned long long)j_3));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_0), "r"(neginf_scratch_addr), "r"((uint32_t)(bulk_elems * 4))
                                            : "memory");
                                    }
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            __syncwarp();
                        }
                        int _max_0 = ((coverage_end) > (clean_begin) ? (coverage_end) : (clean_begin));
                        int aligned_start_0 = (_max_0 + 3) / 4 * 4;
                        int aligned_end_1 = clean_end / 4 * 4;
                        if (aligned_start_0 >= aligned_end_1) {
                            for (int j_4 = (unsigned int)_max_0 + lane; j_4 < clean_end; j_4 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_4)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                        } else {
                            for (int j_5 = (unsigned int)_max_0 + lane; j_5 < aligned_start_0; j_5 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_5)) + (0)) = -CUDART_INF_F;
                            }
                            for (int j_6 = (unsigned int)aligned_end_1 + lane; j_6 < clean_end; j_6 += 32) {
                                *(reinterpret_cast<float*>(Logits + (row_base + (unsigned long long)j_6)) + (0)) = -CUDART_INF_F;
                            }
                            __syncwarp();
                            if (elect_sync()) {
                                for (int j_7 = aligned_start_0; j_7 < aligned_end_1; j_7 += 1024) {
                                    int _min_3 = ((aligned_end_1 - j_7) < (1024) ? (aligned_end_1 - j_7) : (1024));
                                    int bulk_elems_1 = _min_3;
                                    {
                                        void* _cpbulk_dst_1 = reinterpret_cast<void*>(Logits + (row_base + (unsigned long long)j_7));
                                        asm volatile(
                                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                            :: "l"(_cpbulk_dst_1), "r"(neginf_scratch_addr), "r"((uint32_t)(bulk_elems_1 * 4))
                                            : "memory");
                                    }
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            __syncwarp();
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 10) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
