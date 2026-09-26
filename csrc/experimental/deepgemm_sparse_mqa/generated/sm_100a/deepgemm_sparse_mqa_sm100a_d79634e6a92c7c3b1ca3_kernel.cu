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
#define TMEM_NCOLS 428
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFQ_OFFSET 320
#define TMEM_TMEM_SFKV_OFFSET 328
#define NUM_QPIPE_STAGES 2
#define NUM_KVPIPE_STAGES 5
#define NUM_TPIPE_STAGES 5
#define SMEM_Q_OFF 0
#define SMEM_Q_STAGE_BYTES 4096
#define SMEM_Q_STRIDE 4096
#define SMEM_KV_OFF 8192
#define SMEM_KV_STAGE_BYTES 40960
#define SMEM_KV_STRIDE 40960
#define SMEM_SFQ_OFF 212992
#define SMEM_SFQ_STAGE_BYTES 512
#define SMEM_SFQ_STRIDE 512
#define SMEM_SFKV_OFF 214016
#define SMEM_SFKV_STAGE_BYTES 2560
#define SMEM_SFKV_STRIDE 2560
#define SMEM_WEIGHTS_OFF 226816
#define SMEM_WEIGHTS_STAGE_BYTES 128
#define SMEM_WEIGHTS_STRIDE 128
#define SMEM_INFOS_OFF 227072
#define SMEM_INFOS_STAGE_BYTES 3200
#define SMEM_INFOS_STRIDE 3200
#define SMEM_QBLOCKS_OFF 230272
#define SMEM_QBLOCKS_STAGE_BYTES 24
#define SMEM_QBLOCKS_STRIDE 24
#define SMEM_HEADERS_OFF 230304
#define SMEM_HEADERS_STAGE_BYTES 80
#define SMEM_HEADERS_STRIDE 80
#define SMEM_SFQ_WORDS_OFF 212992
#define SMEM_SFQ_WORDS_STAGE_BYTES 1024
#define SMEM_SFQ_WORDS_STRIDE 1024
#define SMEM_TOTAL 230912
#define THREADS 896

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

extern "C" {

__global__ __launch_bounds__(896, 1) void
kernel_deepgemm_sparse_mqa_sm100a_d79634e6a92c7c3b1ca3(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap SF_Q, const __grid_constant__ CUtensorMap Weights, const __grid_constant__ CUtensorMap KV_TMA, const __grid_constant__ CUtensorMap SF_KV_TMA, uint8_t* __restrict__ KV, unsigned int* __restrict__ SF_KV, unsigned int* __restrict__ Metadata, __nv_bfloat16* __restrict__ Logits, unsigned int logits_stride, unsigned int kv_page_stride_bytes, unsigned int num_sms)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 230384;
    #define qfull_addr (mbar_base + 0)
    #define sfqfull_addr (mbar_base + 16)
    #define qempty_addr (mbar_base + 32)
    #define metafull_addr (mbar_base + 48)
    #define sfcopied_addr (mbar_base + 88)
    #define kvfull_addr (mbar_base + 128)
    #define kvempty_addr (mbar_base + 168)
    #define tfull_addr (mbar_base + 208)
    #define tempty_addr (mbar_base + 248)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* q = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int q_addr = smem + 0;
    uint8_t* kv = reinterpret_cast<uint8_t*>(smem_raw + 8192);
    const int kv_addr = smem + 8192;
    uint8_t* sfq = reinterpret_cast<uint8_t*>(smem_raw + 212992);
    const int sfq_addr = smem + 212992;
    uint8_t* sfkv = reinterpret_cast<uint8_t*>(smem_raw + 214016);
    const int sfkv_addr = smem + 214016;
    uint8_t* weights = reinterpret_cast<uint8_t*>(smem_raw + 226816);
    const int weights_addr = smem + 226816;
    unsigned int* infos = reinterpret_cast<unsigned int*>(smem_raw + 227072);
    const int infos_addr = smem + 227072;
    unsigned int* qblocks = reinterpret_cast<unsigned int*>(smem_raw + 230272);
    const int qblocks_addr = smem + 230272;
    unsigned int* headers = reinterpret_cast<unsigned int*>(smem_raw + 230304);
    const int headers_addr = smem + 230304;
    unsigned int* sfq_words = reinterpret_cast<unsigned int*>(smem_raw + 212992);
    const int sfq_words_addr = smem + 212992;
    if (warp == 20) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 20) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SF_Q))) : "memory"); }
    if (warp == 20) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[230384..230672)

    if (warp == 20) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'qpipe' ---
            // qfull: 2 barriers, init_count=1
            mbarrier_init(smem + 230384, 1);
            mbarrier_init(smem + 230392, 1);
            // sfqfull: 2 barriers, init_count=1
            mbarrier_init(smem + 230400, 1);
            mbarrier_init(smem + 230408, 1);
            // qempty: 2 barriers, init_count=704
            mbarrier_init(smem + 230416, 704);
            mbarrier_init(smem + 230424, 704);
            // --- pipeline 'kvpipe' ---
            // metafull: 5 barriers, init_count=32
            mbarrier_init(smem + 230432, 32);
            mbarrier_init(smem + 230440, 32);
            mbarrier_init(smem + 230448, 32);
            mbarrier_init(smem + 230456, 32);
            mbarrier_init(smem + 230464, 32);
            // sfcopied: 5 barriers, init_count=160
            mbarrier_init(smem + 230472, 160);
            mbarrier_init(smem + 230480, 160);
            mbarrier_init(smem + 230488, 160);
            mbarrier_init(smem + 230496, 160);
            mbarrier_init(smem + 230504, 160);
            // kvfull: 5 barriers, init_count=161
            mbarrier_init(smem + 230512, 161);
            mbarrier_init(smem + 230520, 161);
            mbarrier_init(smem + 230528, 161);
            mbarrier_init(smem + 230536, 161);
            mbarrier_init(smem + 230544, 161);
            // kvempty: 5 barriers, init_count=641
            mbarrier_init(smem + 230552, 641);
            mbarrier_init(smem + 230560, 641);
            mbarrier_init(smem + 230568, 641);
            mbarrier_init(smem + 230576, 641);
            mbarrier_init(smem + 230584, 641);
            // --- pipeline 'tpipe' ---
            // tfull: 5 barriers, init_count=1
            mbarrier_init(smem + 230592, 1);
            mbarrier_init(smem + 230600, 1);
            mbarrier_init(smem + 230608, 1);
            mbarrier_init(smem + 230616, 1);
            mbarrier_init(smem + 230624, 1);
            // tempty: 5 barriers, init_count=128
            mbarrier_init(smem + 230632, 128);
            mbarrier_init(smem + 230640, 128);
            mbarrier_init(smem + 230648, 128);
            mbarrier_init(smem + 230656, 128);
            mbarrier_init(smem + 230664, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 428 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 230672);
    if (warp == 21) {
        int _tmem_hold = smem + 230672;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sfq = taddr + 320;
    const int tmem_tmem_sfkv = taddr + 328;
    #pragma unroll 1
    for (unsigned int pad = tid; pad < 128; pad += 896) {
        sfq_words[pad / 64 * 128 + 64 + pad % 64] = 0;
    }
    __syncthreads();
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: math ----
    if (warp <= 19) {
        { // math_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 72;");
            unsigned int asq = 0;
            unsigned int ask = 0;
            unsigned int ast = warp / 4;
            unsigned int ast_phase = 0;
            uint16_t* logits_bits = reinterpret_cast<uint16_t*>(Logits);
            unsigned int row = warp % 4 * 32 + lane;
            unsigned int token = warp / 4 * 128 + row;
            unsigned int row_tmem = warp % 4 * 32 << 16;
            unsigned int wreg[32];
            unsigned int _phase_qfull = 0;
            #pragma unroll 1
            for (unsigned int qblock = 0; qblock < Metadata[1] + 1; qblock++) {
                mbarrier_wait(qfull_addr + (asq) * 8, _phase_qfull);
                unsigned int qbase = qblocks[asq * 3];
                unsigned int nq = qblocks[asq * 3 + 1];
                unsigned int nsplits = qblocks[asq * 3 + 2];
                if (nq == 0) {
                    break;
                }
                #pragma unroll
                for (int qi = 0; qi < 2; qi++) {
                    if (nq > (unsigned int)qi) {
                        #pragma unroll
                        for (int wi = 0; wi < 16; wi++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wreg[qi * 16 + wi])) : "r"(weights_addr + asq * 128 + (unsigned int)(qi * 64) + (unsigned int)(wi * 4)));
                        }
                    }
                }
                #pragma unroll 1
                for (unsigned int split = 0; split < nsplits; split++) {
                    mbarrier_wait(tfull_addr + (ast) * 8, ast_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int q0base = headers[ask * 4 + 2];
                    unsigned int q1base = headers[ask * 4 + 3];
                    unsigned int slots = infos[(ask * 80 + token / 8) * 2 + 1];
                    mbarrier_arrive(kvempty_addr + (ask) * 8);
                    #pragma unroll
                    for (int qi_1 = 0; qi_1 < 2; qi_1++) {
                        if (nq > (unsigned int)qi_1) {
                            unsigned int sum0 = 0;
                            unsigned int sum1 = 0;
                            #pragma unroll
                            for (int hb = 0; hb < 32; hb += 16) {
                                float _tmem_load_0[16];
                                tmem_ld_x16(&_tmem_load_0[0], taddr + ast * 64 + (unsigned int)(qi_1 * 32) + (unsigned int)hb + row_tmem);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                if ((unsigned int)(qi_1 + 1) == nq && hb == 16) {
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    mbarrier_arrive(tempty_addr + (ast) * 8);
                                }
                                #pragma unroll
                                for (int ho = 0; ho < 16; ho += 4) {
                                    uint32_t _bf16x2_relu_0;
                                    asm("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_relu_0) : "f"(_tmem_load_0[ho + 1]), "f"(_tmem_load_0[ho]));
                                    uint32_t _bf16x2_relu_1;
                                    asm("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_relu_1) : "f"(_tmem_load_0[ho + 3]), "f"(_tmem_load_0[ho + 2]));
                                    uint32_t _bf16x2_fma_0;
                                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_0) : "r"(_bf16x2_relu_0), "r"(wreg[qi_1 * 16 + (hb + ho) / 2]), "r"(sum0));
                                    sum0 = _bf16x2_fma_0;
                                    uint32_t _bf16x2_fma_1;
                                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(_bf16x2_fma_1) : "r"(_bf16x2_relu_1), "r"(wreg[qi_1 * 16 + (hb + ho + 2) / 2]), "r"(sum1));
                                    sum1 = _bf16x2_fma_1;
                                }
                            }
                            uint32_t _bf16x2_add_0;
                            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(sum0), "r"(sum1));
                            uint16_t _bf16_add_0;
                            asm("add.rn.bf16 %0, %1, %2;" : "=h"(_bf16_add_0) : "h"(static_cast<uint16_t>(_bf16x2_add_0 & 65535)), "h"(static_cast<uint16_t>(_bf16x2_add_0 >> 16)));
                            unsigned int slot = slots >> (unsigned int)(qi_1 * 16) & 65535;
                            if (slot != 65535) {
                                unsigned int base = ((qi_1 == 0) ? q0base : q1base);
                                unsigned int col = (base + slot) * 8 + token % 8;
                                unsigned long long out = (unsigned long long)(qbase + (unsigned int)qi_1) * (unsigned long long)logits_stride + (unsigned long long)col;
                                logits_bits[out] = _bf16_add_0;
                            }
                        }
                    }
                    ask += 1;
                    if (ask == 5) { ask = 0; }
                    ast += 5;
                    if (ast >= 5) {
                        ast -= 5;
                        ast_phase ^= 1;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(qempty_addr + (asq) * 8);
                asq += 1;
                if (asq == 2) { asq = 0; _phase_qfull ^= 1; }
            }
            asm volatile("barrier.sync 0, 640;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    // ---- Role: producer ----
    } else if (warp == 20) {
        { // producer_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
            unsigned int psq = 0;
            unsigned int psk = 0;
            unsigned int nwaves = Metadata[1];
            unsigned int sched = 4 + Metadata[0] * 164;
            unsigned int _phase_qempty = 1;
            unsigned int _phase_kvempty = 1;
            #pragma unroll 1
            for (unsigned int wave = 0; wave < nwaves; wave++) {
                unsigned int ep = sched + (wave * num_sms + (unsigned int)bid) * 4;
                unsigned int begin = Metadata[ep];
                unsigned int end = Metadata[ep + 1];
                if (begin != end) {
                    unsigned int qbase_1 = Metadata[ep + 2];
                    unsigned int nq_1 = Metadata[ep + 3];
                    mbarrier_wait(qempty_addr + (psq) * 8, _phase_qempty);
                    if (elect_sync()) {
                        qblocks[psq * 3] = qbase_1;
                        qblocks[psq * 3 + 1] = nq_1;
                        qblocks[psq * 3 + 2] = end - begin;
                        tma_2d_gmem2smem(q_addr + psq * 4096, (&Q), 0, qbase_1 * 32, qfull_addr + (psq) * 8);
                        tma_2d_gmem2smem(sfq_addr + psq * 512, (&SF_Q), 0, qbase_1, qfull_addr + (psq) * 8);
                        tma_2d_gmem2smem(weights_addr + psq * 128, (&Weights), 0, qbase_1, qfull_addr + (psq) * 8);
                        mbarrier_arrive_expect_tx(qfull_addr + (psq) * 8, 4480);
                    }
                    #pragma unroll 1
                    for (unsigned int split_1 = begin; split_1 < end; split_1++) {
                        mbarrier_wait(kvempty_addr + (psk) * 8, _phase_kvempty);
                        if (elect_sync()) {
                            asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16;"
                                :: "r"(headers_addr + psk * 16), "l"(Metadata + (4 + split_1 * 164)));
                        }
                        #pragma unroll 1
                        for (unsigned int chunk = lane; chunk < 40; chunk += 32) {
                            asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16;"
                                :: "r"(infos_addr + psk * 80 * 8 + chunk * 16), "l"(Metadata + (8 + split_1 * 164 + chunk * 4)));
                        }
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(metafull_addr + (psk) * 8) : "memory");
                        psk += 1;
                        if (psk == 5) { psk = 0; _phase_kvempty ^= 1; }
                    }
                    mbarrier_arrive(qempty_addr + (psq) * 8);
                    psq += 1;
                    if (psq == 2) { psq = 0; _phase_qempty ^= 1; }
                }
            }
            mbarrier_wait(qempty_addr + (psq) * 8, _phase_qempty);
            mbarrier_wait(kvempty_addr + (psk) * 8, _phase_kvempty);
            if (elect_sync()) {
                qblocks[psq * 3 + 1] = 0;
                headers[psk * 4 + 1] = 0;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(metafull_addr + (psk) * 8), "r"((uint32_t)(32)) : "memory");
                mbarrier_arrive(qfull_addr + (psq) * 8);
            }
        }
    // ---- Role: transpose ----
    } else if (warp == 21) {
        { // transpose_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
            unsigned int tsq = 0;
            unsigned int tsk = 0;
            unsigned int _phase_qfull_1 = 0;
            unsigned int _phase_sfcopied = 0;
            #pragma unroll 1
            for (unsigned int qblock_1 = 0; qblock_1 < Metadata[1] + 1; qblock_1++) {
                mbarrier_wait(qfull_addr + (tsq) * 8, _phase_qfull_1);
                if (qblocks[tsq * 3 + 1] == 0) {
                    break;
                }
                unsigned int nsplits_1 = qblocks[tsq * 3 + 2];
                unsigned int words[4];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(sfq_addr + tsq * 512 + lane * 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(sfq_addr + tsq * 512 + (32 + lane) * 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[2])) : "r"(sfq_addr + tsq * 512 + (64 + lane) * 4));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[3])) : "r"(sfq_addr + tsq * 512 + (96 + lane) * 4));
                __syncwarp();
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(sfq_addr + tsq * 512 + lane * 16), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfq + tsq * 4, make_sf_cp_desc_lo_sbo128((((sfq_addr) >> 4) + (tsq) * 32)));
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(sfqfull_addr + (tsq) * 8);
                }
                #pragma unroll 1
                for (unsigned int split_2 = 0; split_2 < nsplits_1; split_2++) {
                    mbarrier_wait(sfcopied_addr + (tsk) * 8, _phase_sfcopied);
                    #pragma unroll
                    for (int block = 0; block < 5; block++) {
                        unsigned int words_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_0[0])) : "r"(sfkv_addr + tsk * 2560 + (unsigned int)(block * 512) + lane * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_0[1])) : "r"(sfkv_addr + tsk * 2560 + (unsigned int)(block * 512) + (32 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_0[2])) : "r"(sfkv_addr + tsk * 2560 + (unsigned int)(block * 512) + (64 + lane) * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_0[3])) : "r"(sfkv_addr + tsk * 2560 + (unsigned int)(block * 512) + (96 + lane) * 4));
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(sfkv_addr + tsk * 2560 + (unsigned int)(block * 512) + lane * 16), "r"(*reinterpret_cast<uint32_t*>(&words_0[0])), "r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfkv + tsk * 20, make_sf_cp_desc_lo_sbo128((((sfkv_addr) >> 4) + (tsk) * 160)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfkv + tsk * 20 + 4), make_sf_cp_desc_lo_sbo128((((sfkv_addr) >> 4) + (tsk) * 160 + 32)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfkv + tsk * 20 + 8), make_sf_cp_desc_lo_sbo128((((sfkv_addr) >> 4) + (tsk) * 160 + 64)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfkv + tsk * 20 + 12), make_sf_cp_desc_lo_sbo128((((sfkv_addr) >> 4) + (tsk) * 160 + 96)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfkv + tsk * 20 + 16), make_sf_cp_desc_lo_sbo128((((sfkv_addr) >> 4) + (tsk) * 160 + 128)));
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(kvfull_addr + (tsk) * 8);
                    }
                    tsk += 1;
                    if (tsk == 5) { tsk = 0; _phase_sfcopied ^= 1; }
                }
                tsq += 1;
                if (tsq == 2) { tsq = 0; _phase_qfull_1 ^= 1; }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 22) {
        { // mma_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
            unsigned int msq = 0;
            unsigned int msk = 0;
            unsigned int mst = 0;
            unsigned int _phase_qfull_2 = 0;
            unsigned int _phase_sfqfull = 0;
            unsigned int _phase_kvfull = 0;
            unsigned int _phase_tempty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int qblock_2 = 0; qblock_2 < Metadata[1] + 1; qblock_2++) {
                    mbarrier_wait(qfull_addr + (msq) * 8, _phase_qfull_2);
                    if (qblocks[msq * 3 + 1] == 0) {
                        break;
                    }
                    unsigned int nsplits_2 = qblocks[msq * 3 + 2];
                    mbarrier_wait(sfqfull_addr + (msq) * 8, _phase_sfqfull);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (unsigned int split_3 = 0; split_3 < nsplits_2; split_3++) {
                        mbarrier_wait(kvfull_addr + (msk) * 8, _phase_kvfull);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(tempty_addr + (mst) * 8, _phase_tempty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((kv_addr) >> 4) & 0x3FFF) + (msk) * 2560;
                        int _mma_b_lo_0 = (((q_addr) >> 4) & 0x3FFF) + (msq) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 0, b_desc + 0,
                                0x8900480U, (unsigned int)tmem_tmem_sfkv + msk * 5 * 4 + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 0);
                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 2, b_desc + 2,
                                0x489004a0U, (unsigned int)tmem_tmem_sfkv + msk * 5 * 4 + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 1);
                        }
                        tcgen05_commit(tfull_addr + (mst) * 8);
                        mst += 1;
                        if (mst == 5) { mst = 0; _phase_tempty ^= 1; }
                        mbarrier_wait(tempty_addr + (mst) * 8, _phase_tempty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = (((kv_addr + 8192) >> 4) & 0x3FFF) + (msk) * 2560;
                        int _mma_b_lo_1 = (((q_addr) >> 4) & 0x3FFF) + (msq) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 0, b_desc + 0,
                                0x8900480U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 4) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 0);
                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 2, b_desc + 2,
                                0x489004a0U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 4) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 1);
                        }
                        tcgen05_commit(tfull_addr + (mst) * 8);
                        mst += 1;
                        if (mst == 5) { mst = 0; _phase_tempty ^= 1; }
                        mbarrier_wait(tempty_addr + (mst) * 8, _phase_tempty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = (((kv_addr + 16384) >> 4) & 0x3FFF) + (msk) * 2560;
                        int _mma_b_lo_2 = (((q_addr) >> 4) & 0x3FFF) + (msq) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 0, b_desc + 0,
                                0x8900480U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 8) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 0);
                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 2, b_desc + 2,
                                0x489004a0U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 8) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 1);
                        }
                        tcgen05_commit(tfull_addr + (mst) * 8);
                        mst += 1;
                        if (mst == 5) { mst = 0; _phase_tempty ^= 1; }
                        mbarrier_wait(tempty_addr + (mst) * 8, _phase_tempty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_3 = (((kv_addr + 24576) >> 4) & 0x3FFF) + (msk) * 2560;
                        int _mma_b_lo_3 = (((q_addr) >> 4) & 0x3FFF) + (msq) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 0, b_desc + 0,
                                0x8900480U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 12) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 0);
                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 2, b_desc + 2,
                                0x489004a0U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 12) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 1);
                        }
                        tcgen05_commit(tfull_addr + (mst) * 8);
                        mst += 1;
                        if (mst == 5) { mst = 0; _phase_tempty ^= 1; }
                        mbarrier_wait(tempty_addr + (mst) * 8, _phase_tempty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_4 = (((kv_addr + 32768) >> 4) & 0x3FFF) + (msk) * 2560;
                        int _mma_b_lo_4 = (((q_addr) >> 4) & 0x3FFF) + (msq) * 256;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 0, b_desc + 0,
                                0x8900480U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 16) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 0);
                            tcgen05_mma_mxf4_bs((tmem_tmem_acc + (mst * 64)), a_desc + 2, b_desc + 2,
                                0x489004a0U, (unsigned int)tmem_tmem_sfkv + (msk * 5 * 4 + 16) + 0, (unsigned int)tmem_tmem_sfq + msq * 4 + 0, 1);
                        }
                        tcgen05_commit(tfull_addr + (mst) * 8);
                        mst += 1;
                        if (mst == 5) { mst = 0; _phase_tempty ^= 1; }
                        tcgen05_commit(kvempty_addr + (msk) * 8);
                        msk += 1;
                        if (msk == 5) { msk = 0; _phase_kvfull ^= 1; }
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                        :: "r"(qempty_addr + (msq) * 8), "r"((uint32_t)(32)) : "memory");
                    msq += 1;
                    if (msq == 2) { msq = 0; _phase_qfull_2 ^= 1; _phase_sfqfull ^= 1; }
                }
            }
        }
    // ---- Role: copy ----
    } else if (warp >= 23 && warp <= 27) {
        { // copy_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
            unsigned int csk = 0;
            unsigned int cwarp = warp - 23;
            unsigned int _phase_metafull = 0;
            #pragma unroll 1
            for (unsigned int metadata_split = 0; metadata_split < Metadata[0] + 1; metadata_split++) {
                mbarrier_wait(metafull_addr + (csk) * 8, _phase_metafull);
                unsigned int packed = headers[csk * 4 + 1];
                unsigned int nblocks = packed & 2147483647;
                if (nblocks == 0) {
                    break;
                }
                unsigned int gathered = 1;
                if (gathered != 0) {
                    unsigned int block_base = cwarp * 16;
                    if (block_base < nblocks) {
                        unsigned int physical = 0;
                        if (lane < 16) {
                            physical = infos[(csk * 80 + block_base + lane) * 2];
                        }
                        unsigned int page = physical / 8;
                        unsigned int token_1 = physical % 8 * 8;
                        unsigned long long data_offset = (unsigned long long)page * (unsigned long long)kv_page_stride_bytes + (unsigned long long)(token_1 * 64);
                        unsigned long long sf_offset = (unsigned long long)page * (unsigned long long)kv_page_stride_bytes + 4096 + (unsigned long long)(token_1 * 4);
                        unsigned int bo = lane / 2;
                        unsigned int chunk_1 = lane % 2;
                        unsigned long long _shfl_0 = __shfl_sync(0xFFFFFFFF, sf_offset, bo);
                        unsigned long long src_sf = _shfl_0;
                        asm volatile("cp.async.cg.shared::cta.global.L2::64B [%0], [%1], 16;"
                            :: "r"(sfkv_addr + csk * 2560 + (block_base + bo) * 8 * 4 + chunk_1 * 16), "l"(KV + (src_sf + (unsigned long long)(chunk_1 * 16))));
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(sfcopied_addr + (csk) * 8) : "memory");
                        #pragma unroll
                        for (int pair = 0; pair < 16; pair += 2) {
                            unsigned int bo_0 = (unsigned int)pair + lane / 16;
                            unsigned long long _shfl_1 = __shfl_sync(0xFFFFFFFF, data_offset, bo_0);
                            unsigned long long src_kv = _shfl_1;
                            #pragma unroll
                            for (int chunk_base = 0; chunk_base < 32; chunk_base += 16) {
                                unsigned int chunk_0 = (unsigned int)chunk_base + lane % 16;
                                unsigned int swizzled = chunk_0 & 4294967292 | chunk_0 & 3 ^ chunk_0 >> 3 & 3;
                                asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16;"
                                    :: "r"(kv_addr + csk * 40960 + ((block_base + bo_0) * 8 * 64 / 16 + swizzled) * 16), "l"(KV + (src_kv + (unsigned long long)(chunk_0 * 16))));
                            }
                        }
                    } else {
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(sfcopied_addr + (csk) * 8) : "memory");
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(kvfull_addr + (csk) * 8) : "memory");
                }
                csk += 1;
                if (csk == 5) { csk = 0; _phase_metafull ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
