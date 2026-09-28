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
#define TMEM_NCOLS 384
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 5
#define NUM_TMEM_PIPE_STAGES 3
#define SMEM_NEGINF_SCRATCH_OFF 221200
#define SMEM_NEGINF_SCRATCH_STAGE_BYTES 4096
#define SMEM_NEGINF_SCRATCH_STRIDE 4096
#define SMEM_LOCAL_WORK_OFF 220672
#define SMEM_LOCAL_WORK_STAGE_BYTES 128
#define SMEM_LOCAL_WORK_STRIDE 128
#define SMEM_LOCAL_COST_OFF 220800
#define SMEM_LOCAL_COST_STAGE_BYTES 128
#define SMEM_LOCAL_COST_STRIDE 128
#define SMEM_LOCAL_BASE_OFF 220928
#define SMEM_LOCAL_BASE_STAGE_BYTES 128
#define SMEM_LOCAL_BASE_STRIDE 128
#define SMEM_LOCAL_SPLITS_OFF 221056
#define SMEM_LOCAL_SPLITS_STAGE_BYTES 128
#define SMEM_LOCAL_SPLITS_STRIDE 128
#define SMEM_LOCAL_HEADER_OFF 221184
#define SMEM_LOCAL_HEADER_STAGE_BYTES 12
#define SMEM_LOCAL_HEADER_STRIDE 12
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_WEIGHTS_OFF 50176
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 512
#define SMEM_SMEM_WEIGHTS_STRIDE 512
#define SMEM_SMEM_KV_OFF 51712
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 33792
#define SMEM_SMEM_KV_SCALES_OFF 84480
#define SMEM_SMEM_KV_SCALES_STAGE_BYTES 1024
#define SMEM_SMEM_KV_SCALES_STRIDE 33792
#define SMEM_CANDIDATE_QUEUE_VALUES_OFF 153088
#define SMEM_CANDIDATE_QUEUE_VALUES_STAGE_BYTES 8192
#define SMEM_CANDIDATE_QUEUE_VALUES_STRIDE 8192
#define SMEM_CANDIDATE_QUEUE_INDICES_OFF 161280
#define SMEM_CANDIDATE_QUEUE_INDICES_STAGE_BYTES 8192
#define SMEM_CANDIDATE_QUEUE_INDICES_STRIDE 8192
#define SMEM_TOTAL 225408
#define CANDIDATE_MODE 0
#define FULL_Q_BLOCKS 1

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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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
kernel_deepgemm_dense_mqa_sm100a_2252a896cf4f3d30acf4(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap Q_scales_alias, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, unsigned int* __restrict__ ScheduleMeta, float* __restrict__ CandidateValues, int* __restrict__ CandidateIndices, int* __restrict__ CandidateCounts, float* __restrict__ ScoreThresholds, int* __restrict__ cu_seq_len_k_start, int* __restrict__ cu_seq_len_k_end, unsigned int seq_len, unsigned int seq_len_kv, unsigned int stride_logits, unsigned int num_q_blocks, unsigned int num_kv_splits, unsigned int candidate_capacity)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 24)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 88)
    #define umma_full_addr (mbar_base + 128)
    #define umma_empty_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* neginf_scratch = reinterpret_cast<float*>(smem_raw + 221200);
    const int neginf_scratch_addr = smem + 221200;
    unsigned int* local_work = reinterpret_cast<unsigned int*>(smem_raw + 220672);
    const int local_work_addr = smem + 220672;
    unsigned int* local_cost = reinterpret_cast<unsigned int*>(smem_raw + 220800);
    const int local_cost_addr = smem + 220800;
    unsigned int* local_base = reinterpret_cast<unsigned int*>(smem_raw + 220928);
    const int local_base_addr = smem + 220928;
    unsigned int* local_splits = reinterpret_cast<unsigned int*>(smem_raw + 221056);
    const int local_splits_addr = smem + 221056;
    unsigned int* local_header = reinterpret_cast<unsigned int*>(smem_raw + 221184);
    const int local_header_addr = smem + 221184;
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    float* smem_weights = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_weights_addr = smem + 50176;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 51712);
    const int smem_kv_addr = smem + 51712;
    float* smem_kv_scales = reinterpret_cast<float*>(smem_raw + 84480);
    const int smem_kv_scales_addr = smem + 84480;
    float* candidate_queue_values = reinterpret_cast<float*>(smem_raw + 153088);
    const int candidate_queue_values_addr = smem + 153088;
    int* candidate_queue_indices = reinterpret_cast<int*>(smem_raw + 161280);
    const int candidate_queue_indices_addr = smem + 161280;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q_scales_alias))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV_scales))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // q_empty: 3 barriers, init_count=288
            mbarrier_init(smem + 24, 288);
            mbarrier_init(smem + 32, 288);
            mbarrier_init(smem + 40, 288);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 5 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // kv_empty: 5 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            mbarrier_init(smem + 112, 256);
            mbarrier_init(smem + 120, 256);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // umma_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 10) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid < 32) {
        unsigned int qidx = tid;
        unsigned int start = 4294967295;
        unsigned int end = 0;
        #pragma unroll
        for (int qi = 0; qi < 4; qi++) {
            unsigned int row = qidx * 4 + (unsigned int)qi;
            unsigned int _min_0 = (((unsigned int)cu_seq_len_k_start[row]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_start[row]) : (seq_len_kv));
            unsigned int _min_1 = ((start) < (_min_0) ? (start) : (_min_0));
            start = _min_1;
            unsigned int _min_2 = (((unsigned int)cu_seq_len_k_end[row]) < (seq_len_kv) ? ((unsigned int)cu_seq_len_k_end[row]) : (seq_len_kv));
            unsigned int _max_0 = ((end) > (_min_2) ? (end) : (_min_2));
            end = _max_0;
        }
        unsigned int base = start / 4 * 4;
        unsigned int splits = (end - base + 255) / 256;
        unsigned int cost = splits;
        if (splits != 0) {
            cost += 1;
        }
        local_base[qidx] = base;
        local_splits[qidx] = splits;
        if (bid == 0) {
            ScheduleMeta[444 + 2 * qidx] = base;
            ScheduleMeta[444 + 2 * qidx + 1] = splits;
        }
        unsigned long long scanned = ((unsigned long long)splits << 32) + (unsigned long long)cost;
        unsigned long long _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, scanned, 1, 32);
        unsigned long long preceding = _shfl_up_0;
        if (lane >= 1) {
            scanned += preceding;
        }
        unsigned long long _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, scanned, 2, 32);
        unsigned long long preceding_0 = _shfl_up_1;
        if (lane >= 2) {
            scanned += preceding_0;
        }
        unsigned long long _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, scanned, 4, 32);
        unsigned long long preceding_1 = _shfl_up_2;
        if (lane >= 4) {
            scanned += preceding_1;
        }
        unsigned long long _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, scanned, 8, 32);
        unsigned long long preceding_2 = _shfl_up_3;
        if (lane >= 8) {
            scanned += preceding_2;
        }
        unsigned long long _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, scanned, 16, 32);
        unsigned long long preceding_3 = _shfl_up_4;
        if (lane >= 16) {
            scanned += preceding_3;
        }
        local_work[qidx] = (unsigned int)(scanned >> 32);
        local_cost[qidx] = (unsigned int)scanned;
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int total_work = local_work[31];
        unsigned int total_cost = local_cost[31];
        unsigned int per_cta = total_cost / 148;
        unsigned int remainder = total_cost % 148;
        int _min_3 = ((bid) < (remainder) ? (bid) : (remainder));
        unsigned int target = (unsigned int)bid * per_cta + (unsigned int)_min_3;
        unsigned int block = 32;
        unsigned int split = 0;
        unsigned int coordinate = total_work;
        if (target != total_cost) {
            block = 0;
            unsigned int probe = block + 32;
            if (probe <= 32) {
                if (target >= local_cost[probe - 1]) {
                    block = probe;
                }
            }
            unsigned int probe_0 = block + 16;
            if (probe_0 <= 32) {
                if (target >= local_cost[probe_0 - 1]) {
                    block = probe_0;
                }
            }
            unsigned int probe_1 = block + 8;
            if (probe_1 <= 32) {
                if (target >= local_cost[probe_1 - 1]) {
                    block = probe_1;
                }
            }
            unsigned int probe_2 = block + 4;
            if (probe_2 <= 32) {
                if (target >= local_cost[probe_2 - 1]) {
                    block = probe_2;
                }
            }
            unsigned int probe_3 = block + 2;
            if (probe_3 <= 32) {
                if (target >= local_cost[probe_3 - 1]) {
                    block = probe_3;
                }
            }
            unsigned int probe_4 = block + 1;
            if (probe_4 <= 32) {
                if (target >= local_cost[probe_4 - 1]) {
                    block = probe_4;
                }
            }
            unsigned int cost_before = 0;
            unsigned int work_before = 0;
            if (block > 0) {
                cost_before = local_cost[block - 1];
                work_before = local_work[block - 1];
            }
            unsigned int _max_1 = ((target - cost_before) > (1) ? (target - cost_before) : (1));
            unsigned int _min_4 = ((_max_1 - 1) < (local_work[block] - work_before - 1) ? (_max_1 - 1) : (local_work[block] - work_before - 1));
            split = _min_4;
            coordinate = work_before + split;
        }
        int _min_5 = ((bid + 1) < (remainder) ? (bid + 1) : (remainder));
        unsigned int target_0 = (unsigned int)(bid + 1) * per_cta + (unsigned int)_min_5;
        unsigned int block_1 = 32;
        unsigned int split_2 = 0;
        unsigned int coordinate_3 = total_work;
        if (target_0 != total_cost) {
            block_1 = 0;
            unsigned int probe_5 = block_1 + 32;
            if (probe_5 <= 32) {
                if (target_0 >= local_cost[probe_5 - 1]) {
                    block_1 = probe_5;
                }
            }
            unsigned int probe_0_1 = block_1 + 16;
            if (probe_0_1 <= 32) {
                if (target_0 >= local_cost[probe_0_1 - 1]) {
                    block_1 = probe_0_1;
                }
            }
            unsigned int probe_1_1 = block_1 + 8;
            if (probe_1_1 <= 32) {
                if (target_0 >= local_cost[probe_1_1 - 1]) {
                    block_1 = probe_1_1;
                }
            }
            unsigned int probe_2_1 = block_1 + 4;
            if (probe_2_1 <= 32) {
                if (target_0 >= local_cost[probe_2_1 - 1]) {
                    block_1 = probe_2_1;
                }
            }
            unsigned int probe_3_1 = block_1 + 2;
            if (probe_3_1 <= 32) {
                if (target_0 >= local_cost[probe_3_1 - 1]) {
                    block_1 = probe_3_1;
                }
            }
            unsigned int probe_4_1 = block_1 + 1;
            if (probe_4_1 <= 32) {
                if (target_0 >= local_cost[probe_4_1 - 1]) {
                    block_1 = probe_4_1;
                }
            }
            unsigned int cost_before_1 = 0;
            unsigned int work_before_1 = 0;
            if (block_1 > 0) {
                cost_before_1 = local_cost[block_1 - 1];
                work_before_1 = local_work[block_1 - 1];
            }
            unsigned int _max_2 = ((target_0 - cost_before_1) > (1) ? (target_0 - cost_before_1) : (1));
            unsigned int _min_6 = ((_max_2 - 1) < (local_work[block_1] - work_before_1 - 1) ? (_max_2 - 1) : (local_work[block_1] - work_before_1 - 1));
            split_2 = _min_6;
            coordinate_3 = work_before_1 + split_2;
        }
        local_header[0] = block;
        local_header[1] = split;
        local_header[2] = coordinate_3 - coordinate;
        ScheduleMeta[2 * bid] = block;
        ScheduleMeta[2 * bid + 1] = split;
        ScheduleMeta[296 + bid] = coordinate_3 - coordinate;
    }
    __syncthreads();

    // ---- Role: load_q ----
    if (warp == 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // load_q_main
            unsigned int load_q_stage = 0;
            unsigned int load_q_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int _phase_q_empty = 1;
            if (elect_sync()) {
                unsigned int first_q = bid;
                unsigned int first_split = 0;
                unsigned int remaining = 0;
                {
                    first_q = local_header[0];
                    first_split = local_header[1];
                    remaining = local_header[2];
                }
                unsigned int first_q_0 = first_q;
                unsigned int first_split_1 = first_split;
                unsigned int remaining_2 = remaining;
                #pragma unroll 1
                for (unsigned int q_block_idx = first_q_0; q_block_idx < load_q_num_blocks; q_block_idx++) {
                    unsigned int scheduled_kv_start = 0;
                    unsigned int scheduled_num_splits = 1;
                    {
                        if (remaining_2 == 0) {
                            break;
                        }
                        unsigned int span_word = 444 + q_block_idx * 2;
                        unsigned int base_1 = local_base[q_block_idx];
                        unsigned int span_splits = local_splits[q_block_idx];
                        unsigned int splits_1 = 0;
                        if (first_split_1 < span_splits) {
                            unsigned int available = span_splits - first_split_1;
                            splits_1 = ((available < remaining_2) ? available : remaining_2);
                        }
                        scheduled_kv_start = base_1 + first_split_1 * 256;
                        scheduled_num_splits = splits_1;
                        remaining_2 -= scheduled_num_splits;
                        first_split_1 = 0;
                    }
                    if (scheduled_num_splits > 0) {
                        mbarrier_wait(q_empty_addr + (load_q_stage) * 8, _phase_q_empty);
                        tma_2d_gmem2smem(smem_q_addr + load_q_stage * 16384, (&Q), 0, q_block_idx * 128, q_full_addr + (load_q_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + load_q_stage * 512, (&Weights), 0, q_block_idx * 4, q_full_addr + (load_q_stage) * 8);
                        mbarrier_arrive_expect_tx(q_full_addr + (load_q_stage) * 8, 16896);
                        load_q_stage += 1;
                        if (load_q_stage == 3) { load_q_stage = 0; _phase_q_empty ^= 1; }
                    }
                }
            }
            __syncwarp();
        }
    // ---- Role: load_kv ----
    } else if (warp == 9) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // load_kv_main
            unsigned int load_kv_stage = 0;
            unsigned int load_kv_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int first_q_1 = bid;
            unsigned int first_split_2 = 0;
            unsigned int remaining_1 = 0;
            {
                first_q_1 = local_header[0];
                first_split_2 = local_header[1];
                remaining_1 = local_header[2];
            }
            unsigned int first_q_0_1 = first_q_1;
            unsigned int first_split_1_1 = first_split_2;
            unsigned int remaining_2_1 = remaining_1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_1 = first_q_0_1; q_block_idx_1 < load_kv_num_blocks; q_block_idx_1++) {
                unsigned int scheduled_kv_start_1 = 0;
                unsigned int scheduled_num_splits_1 = 1;
                {
                    if (remaining_2_1 == 0) {
                        break;
                    }
                    unsigned int span_word_1 = 444 + q_block_idx_1 * 2;
                    unsigned int base_2 = local_base[q_block_idx_1];
                    unsigned int span_splits_1 = local_splits[q_block_idx_1];
                    unsigned int splits_2 = 0;
                    if (first_split_1_1 < span_splits_1) {
                        unsigned int available_1 = span_splits_1 - first_split_1_1;
                        splits_2 = ((available_1 < remaining_2_1) ? available_1 : remaining_2_1);
                    }
                    scheduled_kv_start_1 = base_2 + first_split_1_1 * 256;
                    scheduled_num_splits_1 = splits_2;
                    remaining_2_1 -= scheduled_num_splits_1;
                    first_split_1_1 = 0;
                }
                if (scheduled_num_splits_1 > 0) {
                    unsigned int kv_start = scheduled_kv_start_1;
                    unsigned int num_kv_blocks = scheduled_num_splits_1;
                    #pragma unroll 1
                    for (unsigned int kv_iter = 0; kv_iter < num_kv_blocks; kv_iter++) {
                        int kv_row = kv_start + kv_iter * 256;
                        if (elect_sync()) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            tma_2d_gmem2smem(smem_kv_addr + load_kv_stage * 33792, (&KV), 0, kv_row, kv_full_addr + (load_kv_stage) * 8);
                            tma_2d_gmem2smem(smem_kv_scales_addr + load_kv_stage * 33792, (&KV_scales), kv_row, 0, kv_full_addr + (load_kv_stage) * 8);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 33792);
                            load_kv_stage += 1;
                            if (load_kv_stage == 5) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        __syncwarp();
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 10) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // mma_main
            unsigned int mma_q_stage = 0;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_tmem_stage = 0;
            unsigned int mma_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int first_q_2 = bid;
            unsigned int first_split_3 = 0;
            unsigned int remaining_3 = 0;
            {
                first_q_2 = local_header[0];
                first_split_3 = local_header[1];
                remaining_3 = local_header[2];
            }
            unsigned int first_q_0_2 = first_q_2;
            unsigned int first_split_1_2 = first_split_3;
            unsigned int remaining_2_2 = remaining_3;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (unsigned int q_block_idx_2 = first_q_0_2; q_block_idx_2 < mma_num_blocks; q_block_idx_2++) {
                unsigned int scheduled_kv_start_2 = 0;
                unsigned int scheduled_num_splits_2 = 1;
                {
                    if (remaining_2_2 == 0) {
                        break;
                    }
                    unsigned int span_word_2 = 444 + q_block_idx_2 * 2;
                    unsigned int base_3 = local_base[q_block_idx_2];
                    unsigned int span_splits_2 = local_splits[q_block_idx_2];
                    unsigned int splits_3 = 0;
                    if (first_split_1_2 < span_splits_2) {
                        unsigned int available_2 = span_splits_2 - first_split_1_2;
                        splits_3 = ((available_2 < remaining_2_2) ? available_2 : remaining_2_2);
                    }
                    scheduled_kv_start_2 = base_3 + first_split_1_2 * 256;
                    scheduled_num_splits_2 = splits_3;
                    remaining_2_2 -= scheduled_num_splits_2;
                    first_split_1_2 = 0;
                }
                if (scheduled_num_splits_2 > 0) {
                    unsigned int kv_start_1 = scheduled_kv_start_2;
                    unsigned int num_kv_blocks_1 = scheduled_num_splits_2;
                    mbarrier_wait(q_full_addr + (mma_q_stage) * 8, _phase_q_full);
                    #pragma unroll 1
                    for (unsigned int kv_iter_1 = 0; kv_iter_1 < num_kv_blocks_1; kv_iter_1++) {
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        if (elect_sync()) {
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2112;
                            int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 1024;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (mma_tmem_stage * 128))), "r"(0));
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                            mbarrier_wait(umma_empty_addr + (mma_tmem_stage) * 8, _phase_umma_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = (((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (mma_kv_stage) * 2112;
                            int _mma_b_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (mma_q_stage) * 1024;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_acc + (mma_tmem_stage * 128))), "r"(0));
                            tcgen05_commit(umma_full_addr + (mma_tmem_stage) * 8);
                            mma_tmem_stage += 1;
                            if (mma_tmem_stage == 3) { mma_tmem_stage = 0; _phase_umma_empty ^= 1; }
                        }
                        __syncwarp();
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 5) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                    }
                    mbarrier_arrive(q_empty_addr + (mma_q_stage) * 8);
                    mma_q_stage += 1;
                    if (mma_q_stage == 3) { mma_q_stage = 0; _phase_q_full ^= 1; }
                }
            }
        }
    // ---- Role: clean ----
    } else if (warp == 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // clean_main
            {
                {
                    float neg_inf[4];
                    #pragma unroll
                    for (int component = 0; component < 4; component++) {
                        neg_inf[component] = -CUDART_INF_F;
                    }
                    for (unsigned int scratch_j = lane; scratch_j < 1024; scratch_j += 32) {
                        neginf_scratch[scratch_j] = -CUDART_INF_F;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    unsigned int clean_num_q_blocks = (seq_len + 4 - 1) / 4;
                    unsigned int clean_tiles_per_row = (stride_logits + 4095) / 4096;
                    unsigned int clean_tasks = clean_num_q_blocks * clean_tiles_per_row;
                    #pragma unroll 1
                    for (unsigned int clean_task = bid; clean_task < clean_tasks; clean_task += num_bids) {
                        unsigned int q_block_idx_3 = clean_task / clean_tiles_per_row;
                        unsigned int clean_begin = clean_task % clean_tiles_per_row * 4096;
                        unsigned int clean_end_raw = clean_begin + 4096;
                        unsigned int _min_7 = ((clean_end_raw) < (stride_logits) ? (clean_end_raw) : (stride_logits));
                        unsigned int clean_end = _min_7;
                        unsigned int q_start = q_block_idx_3 * 4;
                        unsigned int start_v = 4294967295;
                        unsigned int end_v = 0;
                        #pragma unroll 8
                        for (unsigned int token_idx = 0; token_idx < 4; token_idx++) {
                            unsigned int row_unclamped = q_start + token_idx;
                            unsigned int row_idx = ((row_unclamped < seq_len - 1) ? row_unclamped : seq_len - 1);
                            unsigned int k_start_raw = cu_seq_len_k_start[row_idx];
                            unsigned int k_end_raw = cu_seq_len_k_end[row_idx];
                            unsigned int k_start = ((k_start_raw < seq_len_kv) ? k_start_raw : seq_len_kv);
                            unsigned int k_end = ((k_end_raw < seq_len_kv) ? k_end_raw : seq_len_kv);
                            start_v = ((start_v < k_start) ? start_v : k_start);
                            end_v = ((end_v > k_end) ? end_v : k_end);
                        }
                        unsigned int kv_start_2 = start_v / 4 * 4;
                        unsigned int num_kv_blocks_2 = (end_v - kv_start_2 + 256 - 1) / 256;
                        unsigned int raw_end = kv_start_2 + num_kv_blocks_2 * 256;
                        unsigned int coverage_end = ((raw_end < stride_logits) ? raw_end : stride_logits);
                        #pragma unroll 1
                        for (unsigned int qi_1 = 0; qi_1 < 4; qi_1++) {
                            unsigned long long row_base = (unsigned long long)(q_start + qi_1) * (unsigned long long)stride_logits;
                            {
                                float* row_ptr = Logits + row_base;
                                unsigned int _min_8 = ((kv_start_2) < (clean_end) ? (kv_start_2) : (clean_end));
                                unsigned int aligned_start = (clean_begin + 3) / 4 * 4;
                                unsigned int aligned_end = _min_8 / 4 * 4;
                                if (aligned_start >= aligned_end) {
                                    for (unsigned int j = clean_begin + lane; j < _min_8; j += 32) {
                                        *(reinterpret_cast<float*>(row_ptr + j) + (0)) = -CUDART_INF_F;
                                    }
                                    __syncwarp();
                                } else {
                                    for (unsigned int j_1 = clean_begin + lane; j_1 < aligned_start; j_1 += 32) {
                                        *(reinterpret_cast<float*>(row_ptr + j_1) + (0)) = -CUDART_INF_F;
                                    }
                                    for (unsigned int j_2 = aligned_end + lane; j_2 < _min_8; j_2 += 32) {
                                        *(reinterpret_cast<float*>(row_ptr + j_2) + (0)) = -CUDART_INF_F;
                                    }
                                    __syncwarp();
                                    if (elect_sync()) {
                                        for (unsigned int j_3 = aligned_start; j_3 < aligned_end; j_3 += 1024) {
                                            unsigned int _min_9 = ((aligned_end - j_3) < (1024) ? (aligned_end - j_3) : (1024));
                                            unsigned int bulk_elems = _min_9;
                                            {
                                                void* _cpbulk_dst_0 = reinterpret_cast<void*>(row_ptr + j_3);
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
                                unsigned int _max_3 = ((coverage_end) > (clean_begin) ? (coverage_end) : (clean_begin));
                                unsigned int aligned_start_0 = (_max_3 + 3) / 4 * 4;
                                unsigned int aligned_end_1 = clean_end / 4 * 4;
                                if (aligned_start_0 >= aligned_end_1) {
                                    for (unsigned int j_4 = _max_3 + lane; j_4 < clean_end; j_4 += 32) {
                                        *(reinterpret_cast<float*>(row_ptr + j_4) + (0)) = -CUDART_INF_F;
                                    }
                                    __syncwarp();
                                } else {
                                    for (unsigned int j_5 = _max_3 + lane; j_5 < aligned_start_0; j_5 += 32) {
                                        *(reinterpret_cast<float*>(row_ptr + j_5) + (0)) = -CUDART_INF_F;
                                    }
                                    for (unsigned int j_6 = aligned_end_1 + lane; j_6 < clean_end; j_6 += 32) {
                                        *(reinterpret_cast<float*>(row_ptr + j_6) + (0)) = -CUDART_INF_F;
                                    }
                                    __syncwarp();
                                    if (elect_sync()) {
                                        for (unsigned int j_7 = aligned_start_0; j_7 < aligned_end_1; j_7 += 1024) {
                                            unsigned int _min_10 = ((aligned_end_1 - j_7) < (1024) ? (aligned_end_1 - j_7) : (1024));
                                            unsigned int bulk_elems_1 = _min_10;
                                            {
                                                void* _cpbulk_dst_1 = reinterpret_cast<void*>(row_ptr + j_7);
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
                    }
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    __syncwarp();
                }
            }
        }
    // ---- Role: math ----
    } else if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // math_main
            int local_thread_idx = warp % 4 * 32 + lane;
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            int math_thread_idx = wg_idx * 128 + (unsigned int)local_thread_idx;
            unsigned int math_q_stage = 0;
            unsigned int math_kv_stage = 0;
            unsigned int math_tmem_stage = wg_idx;
            unsigned int math_tmem_phase = 0;
            unsigned int math_num_blocks = (seq_len + 4 - 1) / 4;
            unsigned int first_q_3 = bid;
            unsigned int first_split_4 = 0;
            unsigned int remaining_4 = 0;
            {
                first_q_3 = local_header[0];
                first_split_4 = local_header[1];
                remaining_4 = local_header[2];
            }
            unsigned int first_q_0_3 = first_q_3;
            unsigned int first_split_1_3 = first_split_4;
            unsigned int remaining_2_3 = remaining_4;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_1 = 0;
            #pragma unroll 1
            for (unsigned int q_block_idx_4 = first_q_0_3; q_block_idx_4 < math_num_blocks; q_block_idx_4++) {
                unsigned int scheduled_kv_start_3 = 0;
                unsigned int scheduled_num_splits_3 = 1;
                {
                    if (remaining_2_3 == 0) {
                        break;
                    }
                    unsigned int span_word_3 = 444 + q_block_idx_4 * 2;
                    unsigned int base_4 = local_base[q_block_idx_4];
                    unsigned int span_splits_3 = local_splits[q_block_idx_4];
                    unsigned int splits_4 = 0;
                    if (first_split_1_3 < span_splits_3) {
                        unsigned int available_3 = span_splits_3 - first_split_1_3;
                        splits_4 = ((available_3 < remaining_2_3) ? available_3 : remaining_2_3);
                    }
                    scheduled_kv_start_3 = base_4 + first_split_1_3 * 256;
                    scheduled_num_splits_3 = splits_4;
                    remaining_2_3 -= scheduled_num_splits_3;
                    first_split_1_3 = 0;
                }
                if (scheduled_num_splits_3 > 0) {
                    unsigned int q_start_1 = q_block_idx_4 * 4;
                    unsigned int last = seq_len - 1;
                    unsigned int q0 = ((last > q_start_1) ? q_start_1 : last);
                    unsigned int q1 = ((last > q_start_1 + 1) ? q_start_1 + 1 : last);
                    unsigned int q2 = ((last > q_start_1 + 2) ? q_start_1 + 2 : last);
                    unsigned int q3 = ((last > q_start_1 + 3) ? q_start_1 + 3 : last);
                    unsigned int ks_l0 = cu_seq_len_k_start[q0];
                    unsigned int ks_l1 = cu_seq_len_k_start[q1];
                    unsigned int ks_l2 = cu_seq_len_k_start[q2];
                    unsigned int ks_l3 = cu_seq_len_k_start[q3];
                    unsigned int ks0 = ((ks_l0 < seq_len_kv) ? ks_l0 : seq_len_kv);
                    unsigned int ks1 = ((ks_l1 < seq_len_kv) ? ks_l1 : seq_len_kv);
                    unsigned int ks2 = ((ks_l2 < seq_len_kv) ? ks_l2 : seq_len_kv);
                    unsigned int ks3 = ((ks_l3 < seq_len_kv) ? ks_l3 : seq_len_kv);
                    unsigned int ke_l0 = cu_seq_len_k_end[q0];
                    unsigned int ke_l1 = cu_seq_len_k_end[q1];
                    unsigned int ke_l2 = cu_seq_len_k_end[q2];
                    unsigned int ke_l3 = cu_seq_len_k_end[q3];
                    unsigned int ke0 = ((ke_l0 < seq_len_kv) ? ke_l0 : seq_len_kv);
                    unsigned int ke1 = ((ke_l1 < seq_len_kv) ? ke_l1 : seq_len_kv);
                    unsigned int ke2 = ((ke_l2 < seq_len_kv) ? ke_l2 : seq_len_kv);
                    unsigned int ke3 = ((ke_l3 < seq_len_kv) ? ke_l3 : seq_len_kv);
                    unsigned int start_01 = ((ks0 < ks1) ? ks0 : ks1);
                    unsigned int start_012 = ((start_01 < ks2) ? start_01 : ks2);
                    unsigned int start_v_1 = ((start_012 < ks3) ? start_012 : ks3);
                    unsigned int end_01 = ((ke0 > ke1) ? ke0 : ke1);
                    unsigned int end_012 = ((end_01 > ke2) ? end_01 : ke2);
                    unsigned int end_v_1 = ((end_012 > ke3) ? end_012 : ke3);
                    unsigned int aligned_start_1 = start_v_1 / 4 * 4;
                    unsigned int kv_start_3 = aligned_start_1;
                    unsigned int kv_end = end_v_1;
                    unsigned int kv_span = kv_end - kv_start_3;
                    unsigned int num_kv_blocks_3 = (kv_span + 256 - 1) / 256;
                    unsigned int q_start_0 = q_start_1;
                    unsigned int kv_start_1_1 = kv_start_3;
                    unsigned int num_kv_blocks_2_1 = num_kv_blocks_3;
                    {
                        kv_start_1_1 = scheduled_kv_start_3;
                        num_kv_blocks_2_1 = scheduled_num_splits_3;
                    }
                    unsigned int dense_start[4];
                    unsigned int dense_end[4];
                    {
                        dense_start[0] = ks0;
                        dense_end[0] = ke0;
                        dense_start[1] = ks1;
                        dense_end[1] = ke1;
                        dense_start[2] = ks2;
                        dense_end[2] = ke2;
                        dense_start[3] = ks3;
                        dense_end[3] = ke3;
                    }
                    mbarrier_wait(q_full_addr + (math_q_stage) * 8, _phase_q_full_1);
                    float weights_reg[128];
                    int weight_stage_base = math_q_stage * 128;
                    #pragma unroll
                    for (int wi = 0; wi < 128; wi++) {
                        weights_reg[wi] = smem_weights[weight_stage_base + wi];
                    }
                    int q_row_valid[4];
                    int ks_qi[4];
                    int ke_qi[4];
                    float threshold_qi[4];
                    int queue_counts[4];
                    #pragma unroll
                    for (int qi_pre = 0; qi_pre < 4; qi_pre++) {
                        int q_row_pre = q_start_0 + (unsigned int)qi_pre;
                        {
                            q_row_valid[qi_pre] = 1;
                        }
                    }
                    #pragma unroll 1
                    for (unsigned int kv_iter_2 = 0; kv_iter_2 < num_kv_blocks_2_1; kv_iter_2++) {
                        mbarrier_wait(kv_full_addr + (math_kv_stage) * 8, _phase_kv_full_1);
                        int kv_scale_stage_base = math_kv_stage * 8448;
                        float scale_kv = smem_kv_scales[kv_scale_stage_base + math_thread_idx];
                        mbarrier_wait(umma_full_addr + (math_tmem_stage) * 8, math_tmem_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(kv_empty_addr + (math_kv_stage) * 8);
                        math_kv_stage += 1;
                        if (math_kv_stage == 5) { math_kv_stage = 0; _phase_kv_full_1 ^= 1; }
                        int kv_pos = kv_start_1_1 + kv_iter_2 * 256 + (unsigned int)math_thread_idx;
                        #pragma unroll
                        for (int qi_2 = 0; qi_2 < 4; qi_2++) {
                            int acc_base = taddr + math_tmem_stage * 128 + (unsigned int)(qi_2 * 32) + (warp % 4 * 32 << 16);
                            float _tmem_load_0[32];
                            tmem_ld_x16(&_tmem_load_0[0], acc_base);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            tmem_ld_x16(&_tmem_load_0[16], acc_base + 16);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (qi_2 == 3) {
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
                                    float2 _b0 = make_float2(weights_reg[qi_2 * 32 + _j], weights_reg[qi_2 * 32 + _j + 1]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum0) : "l"(*(const unsigned long long*)&_a0), "l"(*(const unsigned long long*)&_b0));
                                    float2 _a1_raw = make_float2(_tmem_load_0[0 + _j + 2], _tmem_load_0[0 + _j + 3]);
                                    float2 _a1_abs = make_float2(fabsf(_tmem_load_0[0 + _j + 2]), fabsf(_tmem_load_0[0 + _j + 3]));
                                    float2 _a1;
                                    asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_a1) : "l"(*(const unsigned long long*)&_a1_raw), "l"(*(const unsigned long long*)&_a1_abs));
                                    float2 _b1 = make_float2(weights_reg[qi_2 * 32 + _j + 2], weights_reg[qi_2 * 32 + _j + 3]);
                                    asm volatile("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(*(unsigned long long*)&_sum1) : "l"(*(const unsigned long long*)&_a1), "l"(*(const unsigned long long*)&_b1));
                                }
                                float2 _sum;
                                asm volatile("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                            }
                            int q_row = q_start_0 + (unsigned int)qi_2;
                            float in_range_result = scale_kv * _relu_wsum_0;
                            {
                                float materialized_result = in_range_result;
                                {
                                    unsigned int rel_kv = (unsigned int)kv_pos - dense_start[qi_2];
                                    unsigned int row_len = dense_end[qi_2] - dense_start[qi_2];
                                    materialized_result = ((rel_kv < row_len) ? in_range_result : -CUDART_INF_F);
                                }
                                {
                                    int out_elem = (unsigned int)q_row * stride_logits + (unsigned int)kv_pos;
                                    *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = materialized_result;
                                }
                            }
                        }
                        math_tmem_stage += 2;
                        if (math_tmem_stage >= 3) {
                            math_tmem_stage -= 3;
                            math_tmem_phase ^= 1;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(q_empty_addr + (math_q_stage) * 8);
                    math_q_stage += 1;
                    if (math_q_stage == 3) { math_q_stage = 0; _phase_q_full_1 ^= 1; }
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    }

    // Cleanup
}

} // extern "C"
