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
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 256
#define TMEM_SF_B_OFFSET 260
#define NUM_MAIN_PIPE_STAGES 5
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_CD_OFF 0
#define SMEM_CD_STAGE_BYTES 16384
#define SMEM_CD_STRIDE 16384
#define SMEM_A_OFF 32768
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 114688
#define SMEM_B_STAGE_BYTES 16384
#define SMEM_B_STRIDE 16384
#define SMEM_SFA_OFF 196608
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 199168
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_TOTAL 201984
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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(256, 1) void
kernel_deepgemm_fp8_batched_gemm_sm100a_1c85c7b53b8618287b52(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, unsigned int* __restrict__ SFD, unsigned int M, unsigned int grid_m, unsigned long long sfd_stride, float alpha)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 201728;
    #define full_addr (mbar_base + 0)
    #define sf_full_addr (mbar_base + 40)
    #define empty_addr (mbar_base + 80)
    #define acc_full_addr (mbar_base + 120)
    #define acc_empty_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* cd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int cd_addr = smem + 0;
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 32768);
    const int a_addr = smem + 32768;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 114688);
    const int b_addr = smem + 114688;
    unsigned int* sfa = reinterpret_cast<unsigned int*>(smem_raw + 196608);
    const int sfa_addr = smem + 196608;
    unsigned int* sfb = reinterpret_cast<unsigned int*>(smem_raw + 199168);
    const int sfb_addr = smem + 199168;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory"); }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[201728..201880)

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'main_pipe' ---
            // full: 5 barriers, init_count=33
            mbarrier_init(smem + 201728, 33);
            mbarrier_init(smem + 201736, 33);
            mbarrier_init(smem + 201744, 33);
            mbarrier_init(smem + 201752, 33);
            mbarrier_init(smem + 201760, 33);
            // sf_full: 5 barriers, init_count=1
            mbarrier_init(smem + 201768, 1);
            mbarrier_init(smem + 201776, 1);
            mbarrier_init(smem + 201784, 1);
            mbarrier_init(smem + 201792, 1);
            mbarrier_init(smem + 201800, 1);
            // empty: 5 barriers, init_count=1
            mbarrier_init(smem + 201808, 1);
            mbarrier_init(smem + 201816, 1);
            mbarrier_init(smem + 201824, 1);
            mbarrier_init(smem + 201832, 1);
            mbarrier_init(smem + 201840, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 201848, 1);
            mbarrier_init(smem + 201856, 1);
            // acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 201864, 128);
            mbarrier_init(smem + 201872, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 264 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 201880);
    if (warp == 2) {
        int _tmem_hold = smem + 201880;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 256;
    const int tmem_sf_b = taddr + 260;
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
                unsigned int off_m = local / 8 * 128;
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
                        tma_3d_gmem2smem(a_addr + stage * 16384, (&A), kt * 128, off_m, head, full_addr + (stage) * 8);
                        tma_3d_gmem2smem(b_addr + stage * 16384, (&B), kt * 128, off_n, head, full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 32768);
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int acc_stage = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_full = 0;
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
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (stage_1) * 32)));
                            tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (stage_1) * 32)));
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        int _mma_a_lo_0 = (((a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _mma_b_lo_0 = (((b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, ((kt_1 == 0) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 2, b_desc + 2,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 4, b_desc + 4,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs((tmem_acc + (acc_stage * 128)), a_desc + 6, b_desc + 6,
                                (0x8a00000U | ((((uint32_t)(kt_1) % 4U)) << 29) | ((((uint32_t)(kt_1) % 4U)) << 4)), tmem_sf_a, tmem_sf_b, 1);
                        }
                    }
                    __syncwarp();
                    elect_commit(empty_addr + (stage_1) * 8);
                    if (kt_1 == 31) {
                        elect_commit(acc_full_addr + (acc_stage) * 8);
                    }
                    __syncwarp();
                    stage_1 += 1;
                    if (stage_1 == 5) { stage_1 = 0; _phase_full ^= 1; }
                }
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_empty ^= 1; }
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
                    mbarrier_arrive(full_addr + (stage_2) * 8);
                    stage_2 += 1;
                    if (stage_2 == 5) { stage_2 = 0; _phase_sf_full ^= 1; }
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
                unsigned int off_m_1 = local_1 / 8 * 128;
                unsigned int off_n_1 = local_1 % 8 * 128;
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int atom = 0; atom < 2; atom++) {
                    if (warp == 4) {
                        asm volatile("cp.async.bulk.wait_group 1;");
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    unsigned int row = off_m_1 + warp_0 * 32 + lane;
                    unsigned int sf_word = 0;
                    #pragma unroll
                    for (int chunk = 0; chunk < 8; chunk++) {
                        unsigned int col = atom * 64 + chunk * 8;
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], (unsigned int)tmem_acc + (acc_stage_1 * 128 + col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            float2 _f2_0 = make_float2(_tmem_load_0[pair * 2], _tmem_load_0[pair * 2 + 1]);
                            float2 _f2_1 = make_float2(alpha, alpha);
                            float2 _mul_f32x2_0;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
                            float2 scaled = _mul_f32x2_0;
                            _tmem_load_0[pair * 2] = scaled.x;
                            _tmem_load_0[pair * 2 + 1] = scaled.y;
                        }
                        if (atom == 1 && chunk == 7) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(acc_empty_addr + (acc_stage_1) * 8);
                        }
                        unsigned int smem_row = store_stage * 128 + warp_0 * 32 + lane;
                        unsigned int packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((cd_addr + (smem_row * 128 + (unsigned int)(chunk * 16) ^ (smem_row * 128 + (unsigned int)(chunk * 16) >> 7 & 7) << 4))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_3d((&D), off_n_1 + (unsigned int)(atom * 64), off_m_1, head_1, cd_addr + store_stage * 16384);
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
    asm volatile("barrier.sync 0, 256;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
    }

    // Cleanup
}

} // extern "C"
