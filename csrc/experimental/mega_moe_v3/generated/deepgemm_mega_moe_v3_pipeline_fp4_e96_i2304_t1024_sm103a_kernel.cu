// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt.

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 72
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SCALE_A_OFFSET 64
#define TMEM_SCALE_B_OFFSET 68
#define NUM_PIPE_STAGES 10
#define SMEM_COMBINE_SMEM_OFF 0
#define SMEM_COMBINE_SMEM_STAGE_BYTES 143360
#define SMEM_COMBINE_SMEM_STRIDE 143360
#define SMEM_SMEM_HIST_OFF 0
#define SMEM_SMEM_HIST_STAGE_BYTES 384
#define SMEM_SMEM_HIST_STRIDE 384
#define SMEM_SMEM_DST_ROW_OFF 384
#define SMEM_SMEM_DST_ROW_STAGE_BYTES 16
#define SMEM_SMEM_DST_ROW_STRIDE 16
#define SMEM_SEND_BUFFER_OFF 1024
#define SMEM_SEND_BUFFER_STAGE_BYTES 20480
#define SMEM_SEND_BUFFER_STRIDE 20480
#define SMEM_SMEM_OUTPUT_OFF 21504
#define SMEM_SMEM_OUTPUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUTPUT_STRIDE 8192
#define SMEM_SMEM_A_OFF 29696
#define SMEM_SMEM_A_STAGE_BYTES 2048
#define SMEM_SMEM_A_STRIDE 2048
#define SMEM_SMEM_B_OFF 50176
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 214016
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 512
#define SMEM_SMEM_SFB_OFF 219136
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 512
#define SMEM_AMAX_REDUCTION_OFF 224256
#define SMEM_AMAX_REDUCTION_STAGE_BYTES 512
#define SMEM_AMAX_REDUCTION_STRIDE 512
#define SMEM_TASK_INFOS_OFF 224768
#define SMEM_TASK_INFOS_STAGE_BYTES 128
#define SMEM_TASK_INFOS_STRIDE 128
#define SMEM_TOTAL 225664
#define THREADS 512
#ifndef NUM_CTAS
#error "NUM_CTAS is a downstream specialization of this program; define it on the compile line"
#endif
#define LAUNCH_MIN_BLOCKS 1

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



__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


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


__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
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




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}



__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
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

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_deepgemm_mega_moe_v3_pipeline_fp4_e96_i2304_t1024_sm103a(int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8, unsigned int* __restrict__ x_sf, const __grid_constant__ CUtensorMap A, int* __restrict__ pool_fp8, unsigned int* __restrict__ pool_sf, float* __restrict__ routing_weight_pool, int* __restrict__ token_to_permuted, int* __restrict__ meta_token, int* __restrict__ meta_slot, int* __restrict__ expert_counts, int* __restrict__ expert_row_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ tile_expert, int* __restrict__ tile_m_local, int* __restrict__ total_m_tiles_out, const __grid_constant__ CUtensorMap B1, const __grid_constant__ CUtensorMap SFB1, const __grid_constant__ CUtensorMap I_fp8_w, uint8_t* __restrict__ SF_I_w, __nv_bfloat16* __restrict__ l1_bf16_capture, const __grid_constant__ CUtensorMap A2, unsigned int* __restrict__ SFA2, const __grid_constant__ CUtensorMap B2, const __grid_constant__ CUtensorMap SFB2, __nv_bfloat16* __restrict__ expert_output, __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ histogram_done, unsigned int* __restrict__ prefix_done, unsigned int* __restrict__ dispatch_done, unsigned int* __restrict__ l1_arrival, unsigned int* __restrict__ l2_done, int num_tokens, int top_k, int num_experts, int N1, int K1, int grid_n1, int K1_tiles, int N2, int K2, int grid_n2, int K2_tiles, int total_m_tiles, int M_total, float activation_clamp, unsigned int* __restrict__ PrivateCounters, unsigned long long* __restrict__ PrivateMasks, unsigned int* __restrict__ PublicExpertOutput, unsigned int* __restrict__ PrivateSFBlockOffsets, const __grid_constant__ CUtensorMap PrivateSF1, const __grid_constant__ CUtensorMap PrivateSF2, unsigned int* __restrict__ PrivateSF1Words, uint8_t* __restrict__ PrivateSF2Bytes, int local_expert_offset)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem + 224896;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 80)
    #define tmem_full_addr (mbar_base + 160)
    #define tmem_empty_addr (mbar_base + 176)
    #define task_full_addr (mbar_base + 192)
    #define task_empty_addr (mbar_base + 224)
    #define combine_barriers_addr (mbar_base + 256)
    #define pull_barriers_addr (mbar_base + 640)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* combine_smem = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int combine_smem_addr = smem + 0;
    int* smem_hist = reinterpret_cast<int*>(smem_raw + 0);
    const int smem_hist_addr = smem + 0;
    int* smem_dst_row = reinterpret_cast<int*>(smem_raw + 384);
    const int smem_dst_row_addr = smem + 384;
    uint8_t* send_buffer = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int send_buffer_addr = smem + 1024;
    uint8_t* smem_output = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_output_addr = smem + 21504;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_a_addr = smem + 29696;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_b_addr = smem + 50176;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int smem_sfa_addr = smem + 214016;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 219136);
    const int smem_sfb_addr = smem + 219136;
    float* amax_reduction = reinterpret_cast<float*>(smem_raw + 224256);
    const int amax_reduction_addr = smem + 224256;
    unsigned int* task_infos = reinterpret_cast<unsigned int*>(smem_raw + 224768);
    const int task_infos_addr = smem + 224768;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B1))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB1))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&I_fp8_w))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A2))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B2))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB2))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&PrivateSF1))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&PrivateSF2))) : "memory"); }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 84 barriers)
    // Mbarriers at smem_raw[224896..225568)

    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 10 barriers, init_count=4
            mbarrier_init(smem + 224896, 4);
            mbarrier_init(smem + 224904, 4);
            mbarrier_init(smem + 224912, 4);
            mbarrier_init(smem + 224920, 4);
            mbarrier_init(smem + 224928, 4);
            mbarrier_init(smem + 224936, 4);
            mbarrier_init(smem + 224944, 4);
            mbarrier_init(smem + 224952, 4);
            mbarrier_init(smem + 224960, 4);
            mbarrier_init(smem + 224968, 4);
            // empty: 10 barriers, init_count=1
            mbarrier_init(smem + 224976, 1);
            mbarrier_init(smem + 224984, 1);
            mbarrier_init(smem + 224992, 1);
            mbarrier_init(smem + 225000, 1);
            mbarrier_init(smem + 225008, 1);
            mbarrier_init(smem + 225016, 1);
            mbarrier_init(smem + 225024, 1);
            mbarrier_init(smem + 225032, 1);
            mbarrier_init(smem + 225040, 1);
            mbarrier_init(smem + 225048, 1);
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 225056, 1);
            mbarrier_init(smem + 225064, 1);
            // tmem_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 225072, 512);
            mbarrier_init(smem + 225080, 512);
            // task_full: 4 barriers, init_count=1
            mbarrier_init(smem + 225088, 1);
            mbarrier_init(smem + 225096, 1);
            mbarrier_init(smem + 225104, 1);
            mbarrier_init(smem + 225112, 1);
            // task_empty: 4 barriers, init_count=512
            mbarrier_init(smem + 225120, 512);
            mbarrier_init(smem + 225128, 512);
            mbarrier_init(smem + 225136, 512);
            mbarrier_init(smem + 225144, 512);
            // combine_barriers: 48 barriers, init_count=1
            mbarrier_init(smem + 225152, 1);
            mbarrier_init(smem + 225160, 1);
            mbarrier_init(smem + 225168, 1);
            mbarrier_init(smem + 225176, 1);
            mbarrier_init(smem + 225184, 1);
            mbarrier_init(smem + 225192, 1);
            mbarrier_init(smem + 225200, 1);
            mbarrier_init(smem + 225208, 1);
            mbarrier_init(smem + 225216, 1);
            mbarrier_init(smem + 225224, 1);
            mbarrier_init(smem + 225232, 1);
            mbarrier_init(smem + 225240, 1);
            mbarrier_init(smem + 225248, 1);
            mbarrier_init(smem + 225256, 1);
            mbarrier_init(smem + 225264, 1);
            mbarrier_init(smem + 225272, 1);
            mbarrier_init(smem + 225280, 1);
            mbarrier_init(smem + 225288, 1);
            mbarrier_init(smem + 225296, 1);
            mbarrier_init(smem + 225304, 1);
            mbarrier_init(smem + 225312, 1);
            mbarrier_init(smem + 225320, 1);
            mbarrier_init(smem + 225328, 1);
            mbarrier_init(smem + 225336, 1);
            mbarrier_init(smem + 225344, 1);
            mbarrier_init(smem + 225352, 1);
            mbarrier_init(smem + 225360, 1);
            mbarrier_init(smem + 225368, 1);
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
            mbarrier_init(smem + 225472, 1);
            mbarrier_init(smem + 225480, 1);
            mbarrier_init(smem + 225488, 1);
            mbarrier_init(smem + 225496, 1);
            mbarrier_init(smem + 225504, 1);
            mbarrier_init(smem + 225512, 1);
            mbarrier_init(smem + 225520, 1);
            mbarrier_init(smem + 225528, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // pull_barriers: 4 barriers, init_count=1
            mbarrier_init(smem + 225536, 1);
            mbarrier_init(smem + 225544, 1);
            mbarrier_init(smem + 225552, 1);
            mbarrier_init(smem + 225560, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 72 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 225568);
    if (warp == 3) {
        int _tmem_hold = smem + 225568;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_scale_a = taddr + 64;
    const int tmem_scale_b = taddr + 68;

    // ---- Role: dispatch ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // dispatch_main
            int disp_warp = warp;
            int disp_tid = disp_warp * 32 + lane;
            int total_pairs = num_tokens * top_k;
            int hidden_u32 = K1 / 4;
            int sf_words_x = K1 / 128;
            unsigned int sync_tag = (unsigned int)1 << 31;
            unsigned int hist_addend = ((bid == 0) ? sync_tag - (unsigned int)(NUM_CTAS - 1) : (unsigned int)1);
            unsigned int prefix_participants = (unsigned int)NUM_CTAS;
            prefix_participants = prefix_participants + (unsigned int)NUM_CTAS / 2;
            unsigned int prefix_addend = ((bid == 0) ? sync_tag - (prefix_participants - 1) : (unsigned int)1);
            unsigned int done_participants = (unsigned int)NUM_CTAS;
            unsigned int done_addend = ((bid == 0) ? sync_tag - (done_participants - 1) : (unsigned int)1);
            int grid_stride = num_bids * 128;
            int cta_base = bid * 128;
            #pragma unroll 1
            for (int idx = cta_base + disp_tid; idx < total_pairs; idx += grid_stride) {
                int e_h = topk_idx_i32[idx * 2] - local_expert_offset;
                if ((unsigned int)e_h < 96) {
                    int slot_h = atomicAdd(&expert_counts[e_h], 1);
                    expert_scatter_offsets[e_h * 1024 + slot_h] = idx;
                } else {
                    token_to_permuted[idx] = -1;
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            unsigned int hist_ticket = 0;
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_0;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_add_0) : "l"(histogram_done), "r"(static_cast<uint32_t>(hist_addend)) : "memory");
                    hist_ticket = _atomic_fetch_add_0;
                    unsigned int _wait_acquire_mask_0;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_0) : "l"(reinterpret_cast<unsigned int*>(histogram_done)) : "memory");
                    } while (((_wait_acquire_mask_0 ^ static_cast<unsigned int>(hist_ticket ^ sync_tag)) & static_cast<unsigned int>(sync_tag)) != 0);
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (bid == 0) {
                if (warp == 0) {
                    unsigned int public_prior = 0;
                    unsigned int source_prior = 0;
                    unsigned int prefetched[3];
                    prefetched[0] = (unsigned int)expert_counts[lane];
                    prefetched[1] = (unsigned int)expert_counts[32 + lane];
                    prefetched[2] = (unsigned int)expert_counts[64 + lane];
                    #pragma unroll
                    for (int stripe = 0; stripe < 3; stripe++) {
                        unsigned int expert = stripe * 32 + lane;
                        unsigned int prefix_count = 0;
                        prefix_count = prefetched[stripe];
                        unsigned int public_tiles = (prefix_count + 256 - 1) / 256 * 2;
                        unsigned int source_blocks = (prefix_count + 32 - 1) / 32;
                        uint32_t public_inclusive = public_tiles;
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 1, 32); if (lane >= 1) { public_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 2, 32); if (lane >= 2) { public_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 4, 32); if (lane >= 4) { public_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 8, 32); if (lane >= 8) { public_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, public_inclusive, 16, 32); if (lane >= 16) { public_inclusive += peer; } }
                        uint32_t source_inclusive = source_blocks;
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 1, 32); if (lane >= 1) { source_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 2, 32); if (lane >= 2) { source_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 4, 32); if (lane >= 4) { source_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 8, 32); if (lane >= 8) { source_inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, source_inclusive, 16, 32); if (lane >= 16) { source_inclusive += peer; } }
                        unsigned int public_offset = public_prior + public_inclusive - public_tiles;
                        unsigned int source_offset = source_prior + source_inclusive - source_blocks;
                        if (expert < (unsigned int)num_experts) {
                            expert_row_offsets[expert] = (int)(public_offset * 128);
                            PrivateSFBlockOffsets[expert] = source_offset;
                            #pragma unroll 1
                            for (int tt = 0; tt < public_tiles; tt++) {
                                tile_expert[public_offset + (unsigned int)tt] = (int)expert;
                                tile_m_local[public_offset + (unsigned int)tt] = (int)(tt * 128);
                            }
                        }
                        unsigned int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, public_inclusive, 31);
                        public_prior += _shfl_sync_1;
                        unsigned int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, source_inclusive, 31);
                        source_prior += _shfl_sync_2;
                    }
                    if (lane == 0) {
                        total_m_tiles_out[0] = (int)public_prior;
                    }
                }
                if (warp == 0) {
                    __syncwarp();
                }
            }
            unsigned int prefix_ticket = 0;
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_3;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_add_3) : "l"(prefix_done), "r"(static_cast<uint32_t>(prefix_addend)) : "memory");
                    prefix_ticket = _atomic_fetch_add_3;
                    unsigned int _wait_acquire_mask_1;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_1) : "l"(reinterpret_cast<unsigned int*>(prefix_done)) : "memory");
                    } while (((_wait_acquire_mask_1 ^ static_cast<unsigned int>(prefix_ticket ^ sync_tag)) & static_cast<unsigned int>(sync_tag)) != 0);
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            unsigned int pull_phase = 0;
            int global_warp_idx = bid * 4 + disp_warp;
            int warps_per_grid = NUM_CTAS * 4;
            int pull_expert = -1;
            int expert_start = 0;
            int expert_end = 0;
            #pragma unroll 1
            for (int pair_idx = global_warp_idx; pair_idx < total_pairs; pair_idx += warps_per_grid) {
                while (expert_end <= pair_idx && pull_expert < 95) {
                    pull_expert += 1;
                    expert_start = expert_end;
                    expert_end += expert_counts[pull_expert];
                }
                if (expert_end <= pair_idx) {
                    break;
                }
                int e_s = pull_expert;
                int dst_row = expert_row_offsets[e_s] + pair_idx - expert_start;
                int source_pair = expert_scatter_offsets[e_s * 1024 + pair_idx - expert_start];
                int t_s = source_pair / 6;
                int k_s = source_pair - t_s * 6;
                if (lane == 0) {
                    token_to_permuted[source_pair] = dst_row;
                    meta_token[dst_row] = t_s;
                    meta_slot[dst_row] = k_s;
                    routing_weight_pool[dst_row] = topk_weights[source_pair];
                }
                __syncwarp();
                unsigned long long src_u32_off = (unsigned long long)t_s * (unsigned long long)hidden_u32;
                unsigned long long dst_u32_off = (unsigned long long)dst_row * (unsigned long long)hidden_u32;
                if (elect_sync()) {
                    cp_async_bulk_gmem2smem(send_buffer_addr + (unsigned int)(disp_warp * 5120), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x_fp8) + ((unsigned long long)src_u32_off * (unsigned long long)4)), 5120, pull_barriers_addr + (disp_warp) * 8);
                    mbarrier_arrive_expect_tx(pull_barriers_addr + (disp_warp) * 8, 5120);
                }
                __syncwarp();
                unsigned long long src_sf_off = (unsigned long long)t_s * (unsigned long long)sf_words_x;
                unsigned long long dst_sf_off = (unsigned long long)dst_row * (unsigned long long)sf_words_x;
                unsigned int in_expert = (unsigned int)(dst_row - expert_row_offsets[e_s]);
                unsigned int sf_block = PrivateSFBlockOffsets[e_s] + in_expert / 32;
                unsigned int source_sf_token = sf_block * 128 + in_expert % 32 * 4;
                #pragma unroll 1
                for (int w = lane; w < sf_words_x; w += 32) {
                    unsigned int sf_word = x_sf[src_sf_off + (unsigned long long)w];
                    pool_sf[dst_sf_off + (unsigned long long)w] = sf_word;
                    PrivateSF1Words[(unsigned int)(w * 36864) + source_sf_token] = sf_word;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_wait(pull_barriers_addr + (disp_warp) * 8, pull_phase);
                    pull_phase ^= 1;
                    {
                        void* _cpbulk_dst_4 = reinterpret_cast<void*>(pool_fp8 + dst_u32_off);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_4), "r"(send_buffer_addr + (unsigned int)(disp_warp * 5120)), "r"((uint32_t)(5120))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                __syncwarp();
                if (elect_sync()) {
                    unsigned int arrivals = 1;
                    if (in_expert + 1 == (unsigned int)expert_counts[e_s]) {
                        arrivals = 32 - in_expert % 32;
                    }
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (578 + sf_block))), "r"(static_cast<unsigned int>(arrivals)) : "memory");
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            __threadfence_system();
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_old_0;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_0) : "l"(dispatch_done), "r"(static_cast<uint32_t>(done_addend)) : "memory");
                }
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (bid == 0) {
                #pragma unroll 1
                for (int i = disp_tid; i < num_experts; i += 128) {
                    expert_counts[i] = 0;
                }
                if (disp_tid == 0) {
                    PrivateCounters[0] = (unsigned int)0;
                    PrivateCounters[1] = (unsigned int)0;
                }
            } else {
                #pragma unroll 1
                for (int block = (bid - 1) * 128 + disp_tid; block < 288; block += (NUM_CTAS - 1) * 128) {
                    PrivateMasks[block] = (unsigned long long)0;
                    PrivateCounters[578 + block] = (unsigned int)0;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // load_a_main
            unsigned int task[8];
            unsigned int a_stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (int task_iteration = 0; task_iteration < 10945; task_iteration++) {
                mbarrier_wait(task_full_addr + (task_iteration % 4) * 8, task_iteration / 4 & 1);
                task[0] = task_infos[task_iteration % 4 * 8];
                task[1] = task_infos[task_iteration % 4 * 8 + 1];
                task[2] = task_infos[task_iteration % 4 * 8 + 2];
                task[3] = task_infos[task_iteration % 4 * 8 + 3];
                task[4] = task_infos[task_iteration % 4 * 8 + 4];
                task[5] = task_infos[task_iteration % 4 * 8 + 5];
                task[6] = task_infos[task_iteration % 4 * 8 + 6];
                task[7] = task_infos[task_iteration % 4 * 8 + 7];
                if (task[0] == 0) {
                    break;
                }
                unsigned int public_m = (unsigned int)expert_row_offsets[task[1]] + task[2] * 32;
                unsigned int aligned_m = (task[5] + 15) / 16 * 16;
                unsigned int token_offset = public_m + (unsigned int)cta_rank * (aligned_m / 2);
                unsigned long long pending = 68719476735;
                if (task[0] == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (578 + task[4]))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(32)) >= static_cast<unsigned int>(1));
                    }
                }
                #pragma unroll 2
                for (int k = 0; k < task[7] / 128; k++) {
                    if (task[0] == 2) {
                        unsigned long long k_mask = (unsigned long long)3 << (unsigned long long)(k * 2);
                        if ((pending & k_mask) != 0) {
                            unsigned long long _wait_acquire_mask_3;
                            do {
                            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_wait_acquire_mask_3) : "l"((reinterpret_cast<unsigned long long*>(PrivateMasks) + (task[4]))) : "memory");
                            } while (((_wait_acquire_mask_3 ^ static_cast<unsigned long long>((unsigned long long)68719476735)) & static_cast<unsigned long long>(k_mask)) != 0);
                            unsigned long long observed = _wait_acquire_mask_3;
                            pending = observed ^ 68719476735;
                        }
                    }
                    mbarrier_wait(empty_addr + (a_stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        if (task[0] == 1) {
                            tma_2d_gmem2smem_cta2(smem_a_addr + a_stage * 2048, (&A), k * 128, token_offset, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 512, (&PrivateSF1), task[4] * 128, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        } else {
                            tma_2d_gmem2smem_cta2(smem_a_addr + a_stage * 2048, (&A2), k * 128, token_offset, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 512, (&PrivateSF2), task[4] * 128, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                        }
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (a_stage) * 8, 5120);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (a_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    __syncwarp();
                    a_stage += 1;
                    if (a_stage == 10) { a_stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 5) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // load_b_main
            unsigned int task_1[8];
            unsigned int b_stage = 0;
            unsigned int _phase_empty_1 = 1;
            #pragma unroll 1
            for (int task_iteration_1 = 0; task_iteration_1 < 10945; task_iteration_1++) {
                mbarrier_wait(task_full_addr + (task_iteration_1 % 4) * 8, task_iteration_1 / 4 & 1);
                task_1[0] = task_infos[task_iteration_1 % 4 * 8];
                task_1[1] = task_infos[task_iteration_1 % 4 * 8 + 1];
                task_1[2] = task_infos[task_iteration_1 % 4 * 8 + 2];
                task_1[3] = task_infos[task_iteration_1 % 4 * 8 + 3];
                task_1[4] = task_infos[task_iteration_1 % 4 * 8 + 4];
                task_1[5] = task_infos[task_iteration_1 % 4 * 8 + 5];
                task_1[6] = task_infos[task_iteration_1 % 4 * 8 + 6];
                task_1[7] = task_infos[task_iteration_1 % 4 * 8 + 7];
                if (task_1[0] == 0) {
                    break;
                }
                unsigned int n_offset = (task_1[3] * 2 + (unsigned int)cta_rank) * 128;
                #pragma unroll 2
                for (int k_1 = 0; k_1 < task_1[7] / 128; k_1++) {
                    mbarrier_wait(empty_addr + (b_stage) * 8, _phase_empty_1);
                    unsigned int sf_k_offset = task_1[1] * (task_1[7] / 128) + (unsigned int)k_1;
                    if (elect_sync()) {
                        if (task_1[0] == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_b_addr + b_stage * 16384), "l"((&B1)), "r"(k_1 * 128), "r"(task_1[1] * task_1[6] + n_offset),
                                   "r"(((full_addr + (b_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_sfb_addr + b_stage * 512), "l"((&SFB1)), "r"(n_offset), "r"(sf_k_offset),
                                   "r"(((full_addr + (b_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_b_addr + b_stage * 16384), "l"((&B2)), "r"(k_1 * 128), "r"(task_1[1] * task_1[6] + n_offset),
                                   "r"(((full_addr + (b_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_sfb_addr + b_stage * 512), "l"((&SFB2)), "r"(n_offset), "r"(sf_k_offset),
                                   "r"(((full_addr + (b_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (b_stage) * 8, 17408);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (b_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    __syncwarp();
                    b_stage += 1;
                    if (b_stage == 10) { b_stage = 0; _phase_empty_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // mma_main
            unsigned int task_2[8];
            unsigned int mma_stage = 0;
            unsigned int completed_tasks = 0;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int task_iteration_2 = 0; task_iteration_2 < 10945; task_iteration_2++) {
                    mbarrier_wait(task_full_addr + (task_iteration_2 % 4) * 8, task_iteration_2 / 4 & 1);
                    task_2[0] = task_infos[task_iteration_2 % 4 * 8];
                    task_2[1] = task_infos[task_iteration_2 % 4 * 8 + 1];
                    task_2[2] = task_infos[task_iteration_2 % 4 * 8 + 2];
                    task_2[3] = task_infos[task_iteration_2 % 4 * 8 + 3];
                    task_2[4] = task_infos[task_iteration_2 % 4 * 8 + 4];
                    task_2[5] = task_infos[task_iteration_2 % 4 * 8 + 5];
                    task_2[6] = task_infos[task_iteration_2 % 4 * 8 + 6];
                    task_2[7] = task_infos[task_iteration_2 % 4 * 8 + 7];
                    if (task_2[0] == 0) {
                        break;
                    }
                    unsigned int accum_stage = task_iteration_2 % 2;
                    unsigned int aligned_m_1 = (task_2[5] + 15) / 16 * 16;
                    int _stage_tmem_empty = (int)((unsigned int)task_iteration_2 % 2);
                    int _parity_tmem_empty = (int)((unsigned int)task_iteration_2 / 2 % 2) ^ 1;
                    mbarrier_wait(tmem_empty_addr + (_stage_tmem_empty) * 8, _parity_tmem_empty);
                    #pragma unroll 2
                    for (int k_2 = 0; k_2 < task_2[7] / 128; k_2++) {
                        mbarrier_wait(full_addr + (mma_stage) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((k_2 == 0) ? 1 : 0);
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_scale_a, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_scale_b, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_stage) * 32)), _elected_0);
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024;
                        int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 128;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 32)), a_desc + 0, b_desc + 0,
                                (((0x10880280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((0) << 29) | ((0) << 4)), tmem_scale_b, tmem_scale_a, ((init_flag) ? 0 : 1), _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 32)), a_desc + 2, b_desc + 2,
                                (((0x10880280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((1) << 29) | ((1) << 4)), tmem_scale_b, tmem_scale_a, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 32)), a_desc + 4, b_desc + 4,
                                (((0x10880280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((2) << 29) | ((2) << 4)), tmem_scale_b, tmem_scale_a, 1, _elected_0);
                            tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 32)), a_desc + 6, b_desc + 6,
                                (((0x10880280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((3) << 29) | ((3) << 4)), tmem_scale_b, tmem_scale_a, 1, _elected_0);
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        if ((unsigned int)k_2 == task_2[7] / 128 - 1) {
                            elect_commit_cg2_multicast(tmem_full_addr + (accum_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        mma_stage += 1;
                        if (mma_stage == 10) { mma_stage = 0; _phase_full ^= 1; }
                    }
                    completed_tasks += 1;
                }
                if (completed_tasks > 0) {
                    unsigned int _stage_tmem_empty_1 = (completed_tasks - 1) % 2;
                    unsigned int _parity_tmem_empty_1 = (completed_tasks - 1) / 2 % 2;
                    mbarrier_wait(tmem_empty_addr + (_stage_tmem_empty_1) * 8, _parity_tmem_empty_1);
                }
            }
        }
    }
    // ---- Role: schedule ----
    if (warp == 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // schedule_main
            unsigned int counts[3];
            unsigned int state[2];
            unsigned int task_3[8];
            unsigned int schedule_iteration = 0;
            unsigned int sched_tag = (unsigned int)1 << 31;
            unsigned int sched_ticket = 0;
            if (cta_rank == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_fetch_add_0_1;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_add_0_1) : "l"(prefix_done), "r"(static_cast<uint32_t>(1)) : "memory");
                    sched_ticket = _atomic_fetch_add_0_1;
                    unsigned int _wait_acquire_mask_2;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_2) : "l"(reinterpret_cast<unsigned int*>(prefix_done)) : "memory");
                    } while (((_wait_acquire_mask_2 ^ static_cast<unsigned int>(sched_ticket ^ sched_tag)) & static_cast<unsigned int>(sched_tag)) != 0);
                }
                __syncwarp();
                unsigned int expert_1 = lane;
                counts[0] = 0;
                if (expert_1 < 96) {
                    counts[0] = (unsigned int)expert_counts[expert_1];
                }
                unsigned int expert_0 = 32 + lane;
                counts[1] = 0;
                if (expert_0 < 96) {
                    counts[1] = (unsigned int)expert_counts[expert_0];
                }
                unsigned int expert_1_1 = 64 + lane;
                counts[2] = 0;
                if (expert_1_1 < 96) {
                    counts[2] = (unsigned int)expert_counts[expert_1_1];
                }
                __syncwarp();
                unsigned int num_blocks = 0;
                if (lane < 96) {
                    num_blocks += (counts[0] + 32 - 1) / 32;
                }
                if (32 + lane < 96) {
                    num_blocks += (counts[1] + 32 - 1) / 32;
                }
                if (64 + lane < 96) {
                    num_blocks += (counts[2] + 32 - 1) / 32;
                }
                unsigned int total;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(total) : "r"(num_blocks));
                unsigned int waves = (total * 18 + (unsigned int)NUM_CTAS / 2 - 1) / ((unsigned int)NUM_CTAS / 2);
                unsigned int interleave_waves = 2;
                state[0] = total;
                unsigned int _max_2 = ((1) > (interleave_waves) ? (1) : (interleave_waves));
                unsigned int _min_1 = ((_max_2) < (waves) ? (_max_2) : (waves));
                state[1] = _min_1;
                #pragma unroll 1
                for (int iteration = 0; iteration < 10945; iteration++) {
                    unsigned int _stage_task_empty = schedule_iteration % 4;
                    unsigned int _parity_task_empty = schedule_iteration / 4 % 2 ^ 1;
                    mbarrier_wait(task_empty_addr + (_stage_task_empty) * 8, _parity_task_empty);
                    unsigned int phase = 0;
                    unsigned int claimed = 0;
                    unsigned int clusters = 18;
                    unsigned int shape_n = 4608;
                    unsigned int shape_k = 5120;
                    if (state[1] != 4294967295u && state[1] != 0) {
                        state[1] = state[1] - 1;
                        unsigned int claimed_0 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_fetch_add_3_1 = atomicAdd(PrivateCounters, 1);
                            claimed_0 = _atomic_fetch_add_3_1;
                        }
                        unsigned int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, claimed_0, 0);
                        claimed = _shfl_sync_4;
                        if (claimed >= state[0] * 18) {
                            state[1] = 4294967295;
                        } else {
                            phase = 1;
                        }
                    }
                    if (phase == 0) {
                        unsigned int claimed_0_1 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_fetch_add_5 = atomicAdd(PrivateCounters + 1, 1);
                            claimed_0_1 = _atomic_fetch_add_5;
                        }
                        unsigned int _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, claimed_0_1, 0);
                        claimed = _shfl_sync_6;
                        if (claimed < state[0] * 20) {
                            if (state[1] != 4294967295u) {
                                state[1] = 1;
                            }
                            phase = 2;
                            clusters = 20;
                            shape_n = 5120;
                            shape_k = 2304;
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
                    if (phase != 0) {
                        unsigned int pool_block = claimed / clusters;
                        task_3[0] = phase;
                        task_3[1] = 0;
                        task_3[2] = 0;
                        task_3[3] = claimed % clusters;
                        task_3[4] = pool_block;
                        task_3[5] = 0;
                        task_3[6] = shape_n;
                        task_3[7] = shape_k;
                        unsigned int block_offset = 0;
                        unsigned int expert_2 = lane;
                        unsigned int tokens = counts[0];
                        unsigned int blocks = (tokens + 32 - 1) / 32;
                        uint32_t inclusive = blocks;
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32); if (lane >= 1) { inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32); if (lane >= 2) { inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32); if (lane >= 4) { inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32); if (lane >= 8) { inclusive += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32); if (lane >= 16) { inclusive += peer; } }
                        unsigned int lane_offset = block_offset + inclusive - blocks;
                        unsigned int owner_mask = __ballot_sync(0xFFFFFFFF, expert_2 < 96 && (pool_block >= lane_offset && pool_block < lane_offset + blocks));
                        if (owner_mask != 0) {
                            int _find_first_set_7 = __ffs(owner_mask);
                            unsigned int owner_lane = _find_first_set_7 - 1;
                            unsigned int local_block = pool_block - lane_offset;
                            unsigned int valid = ((tokens - local_block * 32) < (32) ? (tokens - local_block * 32) : (32));
                            unsigned int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, expert_2, owner_lane);
                            task_3[1] = _shfl_sync_8;
                            unsigned int _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, local_block, owner_lane);
                            task_3[2] = _shfl_sync_9;
                            unsigned int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, valid, owner_lane);
                            task_3[5] = _shfl_sync_10;
                        }
                        unsigned int _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, inclusive, 31);
                        block_offset += _shfl_sync_11;
                        unsigned int expert_3 = 32 + lane;
                        unsigned int tokens_4 = counts[1];
                        unsigned int blocks_5 = (tokens_4 + 32 - 1) / 32;
                        uint32_t inclusive_6 = blocks_5;
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_6, 1, 32); if (lane >= 1) { inclusive_6 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_6, 2, 32); if (lane >= 2) { inclusive_6 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_6, 4, 32); if (lane >= 4) { inclusive_6 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_6, 8, 32); if (lane >= 8) { inclusive_6 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_6, 16, 32); if (lane >= 16) { inclusive_6 += peer; } }
                        unsigned int lane_offset_7 = block_offset + inclusive_6 - blocks_5;
                        unsigned int owner_mask_8 = __ballot_sync(0xFFFFFFFF, expert_3 < 96 && (pool_block >= lane_offset_7 && pool_block < lane_offset_7 + blocks_5));
                        if (owner_mask_8 != 0) {
                            int _find_first_set_12 = __ffs(owner_mask_8);
                            unsigned int owner_lane_1 = _find_first_set_12 - 1;
                            unsigned int local_block_1 = pool_block - lane_offset_7;
                            unsigned int valid_1 = ((tokens_4 - local_block_1 * 32) < (32) ? (tokens_4 - local_block_1 * 32) : (32));
                            unsigned int _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, expert_3, owner_lane_1);
                            task_3[1] = _shfl_sync_13;
                            unsigned int _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, local_block_1, owner_lane_1);
                            task_3[2] = _shfl_sync_14;
                            unsigned int _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, valid_1, owner_lane_1);
                            task_3[5] = _shfl_sync_15;
                        }
                        unsigned int _shfl_sync_16 = __shfl_sync(0xFFFFFFFF, inclusive_6, 31);
                        block_offset += _shfl_sync_16;
                        unsigned int expert_9 = 64 + lane;
                        unsigned int tokens_10 = counts[2];
                        unsigned int blocks_11 = (tokens_10 + 32 - 1) / 32;
                        uint32_t inclusive_12 = blocks_11;
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_12, 1, 32); if (lane >= 1) { inclusive_12 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_12, 2, 32); if (lane >= 2) { inclusive_12 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_12, 4, 32); if (lane >= 4) { inclusive_12 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_12, 8, 32); if (lane >= 8) { inclusive_12 += peer; } }
                        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_12, 16, 32); if (lane >= 16) { inclusive_12 += peer; } }
                        unsigned int lane_offset_13 = block_offset + inclusive_12 - blocks_11;
                        unsigned int owner_mask_14 = __ballot_sync(0xFFFFFFFF, expert_9 < 96 && (pool_block >= lane_offset_13 && pool_block < lane_offset_13 + blocks_11));
                        if (owner_mask_14 != 0) {
                            int _find_first_set_17 = __ffs(owner_mask_14);
                            unsigned int owner_lane_2 = _find_first_set_17 - 1;
                            unsigned int local_block_2 = pool_block - lane_offset_13;
                            unsigned int valid_2 = ((tokens_10 - local_block_2 * 32) < (32) ? (tokens_10 - local_block_2 * 32) : (32));
                            unsigned int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, expert_9, owner_lane_2);
                            task_3[1] = _shfl_sync_18;
                            unsigned int _shfl_sync_19 = __shfl_sync(0xFFFFFFFF, local_block_2, owner_lane_2);
                            task_3[2] = _shfl_sync_19;
                            unsigned int _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, valid_2, owner_lane_2);
                            task_3[5] = _shfl_sync_20;
                        }
                        unsigned int _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, inclusive_12, 31);
                        block_offset += _shfl_sync_21;
                        if (phase == 2) {
                            unsigned int required = (task_3[4] + 1) * 18;
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"(reinterpret_cast<unsigned int*>(PrivateCounters)) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(required)) >= static_cast<unsigned int>((unsigned int)0 - required));
                            }
                        }
                    }
                    if (task_3[0] == 0) {
                        break;
                    }
                    if (lane < 2) {
                        uint32_t destination;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(destination) : "r"(task_infos_addr + schedule_iteration % 4 * 32), "r"(lane));
                        uint32_t barrier;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(barrier) : "r"(task_full_addr + (unsigned int)(schedule_iteration % 4 * 8)), "r"(lane));
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
                    schedule_iteration += 1;
                }
                unsigned int _stage_task_empty_1 = schedule_iteration % 4;
                unsigned int _parity_task_empty_1 = schedule_iteration / 4 % 2 ^ 1;
                mbarrier_wait(task_empty_addr + (_stage_task_empty_1) * 8, _parity_task_empty_1);
                task_3[0] = 0;
                task_3[1] = 0;
                task_3[2] = 0;
                task_3[3] = 0;
                task_3[4] = 0;
                task_3[5] = 0;
                task_3[6] = 0;
                task_3[7] = 0;
                if (lane < 2) {
                    uint32_t destination_1;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(destination_1) : "r"(task_infos_addr + schedule_iteration % 4 * 32), "r"(lane));
                    uint32_t barrier_1;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(barrier_1) : "r"(task_full_addr + (unsigned int)(schedule_iteration % 4 * 8)), "r"(lane));
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
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            unsigned int task_4[8];
            #pragma unroll 1
            for (int task_iteration_3 = 0; task_iteration_3 < 10945; task_iteration_3++) {
                mbarrier_wait(task_full_addr + (task_iteration_3 % 4) * 8, task_iteration_3 / 4 & 1);
                task_4[0] = task_infos[task_iteration_3 % 4 * 8];
                task_4[1] = task_infos[task_iteration_3 % 4 * 8 + 1];
                task_4[2] = task_infos[task_iteration_3 % 4 * 8 + 2];
                task_4[3] = task_infos[task_iteration_3 % 4 * 8 + 3];
                task_4[4] = task_infos[task_iteration_3 % 4 * 8 + 4];
                task_4[5] = task_infos[task_iteration_3 % 4 * 8 + 5];
                task_4[6] = task_infos[task_iteration_3 % 4 * 8 + 6];
                task_4[7] = task_infos[task_iteration_3 % 4 * 8 + 7];
                if (task_4[0] == 0) {
                    break;
                }
                unsigned int accum_stage_1 = task_iteration_3 % 2;
                int _stage_tmem_full = (int)((unsigned int)task_iteration_3 % 2);
                int _parity_tmem_full = (int)((unsigned int)task_iteration_3 / 2 % 2);
                mbarrier_wait(tmem_full_addr + (_stage_tmem_full) * 8, _parity_tmem_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((task_empty_addr + (task_iteration_3 % 4) * 8) & 0xFEFFFFFF) : "memory");
                if (task_4[0] == 1) {
                    unsigned int epi_warp = warp - 8;
                    unsigned int epi_wg = epi_warp / 4;
                    unsigned int warp_in_wg = epi_warp % 4;
                    unsigned int valid_m = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int pool_block_1 = task_4[4];
                    unsigned int ring_block = pool_block_1 % 288;
                    unsigned int block_1 = ((unsigned int)expert_row_offsets[task_4[1]] + task_4[2] * 32) / 32;
                    unsigned int sf_stride = 147456;
                    if (task_4[0] == 3) {
                        block_1 = pool_block_1;
                        sf_stride = 0;
                    }
                    unsigned int n_block = task_4[3] * 2 + (unsigned int)cta_rank;
                    float cached_weight = 1.0f;
                    float activation[8];
                    float amax[4];
                    float quantized[4];
                    #pragma unroll
                    for (int s = 0; s < 1; s++) {
                        if (valid_m <= epi_wg * 16 + (unsigned int)(s * 16)) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            break;
                        }
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 2; i_1++) {
                            unsigned int j = s * 2 + i_1;
                            if (task_4[0] == 1 && j * 8 % 32 == 0) {
                                if (j * 8 + (unsigned int)lane < 16) {
                                    cached_weight = routing_weight_pool[block_1 * 32 + epi_wg * 16 + j * 8 + (unsigned int)lane];
                                }
                            }
                            float first_weight = __shfl_sync(0xFFFFFFFF, cached_weight, j * 8 % 32 + (unsigned int)(lane % 4 * 2));
                            float second_weight = __shfl_sync(0xFFFFFFFF, cached_weight, j * 8 % 32 + (unsigned int)(lane % 4 * 2) + 1);
                            unsigned int address = accum_stage_1 * 32 + epi_wg * 16 + j * 8;
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
                            if (j == 1) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                            __nv_bfloat162 _min_0 = __hmin2(_bf16x2_0, _bf16x2_2);
                            __nv_bfloat162 _min_1_1 = __hmin2(__hmax2(_bf16x2_1, _bf16x2_3), _bf16x2_2);
                            float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(_min_0);
                            float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(_min_1_1);
                            float denominator[2];
                            float2 activated_gate;
                            float _exp_0 = __expf(-_cvt_f32_0.x);
                            denominator[0] = _exp_0;
                            float _exp_1 = __expf(-_cvt_f32_0.y);
                            denominator[1] = _exp_1;
                            const float2 _add2_2 = {1.0f, 1.0f};
                            #pragma unroll
                            for (int _la = 0; _la < 1; _la++) {
                                reinterpret_cast<float2*>(denominator)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(denominator)[_la], _add2_2);
                            }
                            float _rcp_3 = approx_rcp(denominator[0]);
                            float _rcp_4 = approx_rcp(denominator[1]);
                            activated_gate = mul_f32x2_noftz(_cvt_f32_0, make_float2(_rcp_3, _rcp_4));
                            float2 gated_up;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gated_up) : "l"(*(const unsigned long long*)&activated_gate), "l"(*(const unsigned long long*)&_cvt_f32_1));
                            float2 _f32x2_mul_b_5 = make_float2(first_weight, second_weight);
                            float2 activated;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&activated) : "l"(*(const unsigned long long*)&gated_up), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
                            activation[i_1 * 4] = activated.x;
                            activation[i_1 * 4 + 1] = activated.y;
                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                            __nv_bfloat162 _min_2 = __hmin2(_bf16x2_4, _bf16x2_6);
                            __nv_bfloat162 _min_3 = __hmin2(__hmax2(_bf16x2_5, _bf16x2_7), _bf16x2_6);
                            float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(_min_2);
                            float2 _cvt_f32_3 = cake_bf16x2_to_f32x2(_min_3);
                            float denominator_0[2];
                            float2 activated_gate_1;
                            float _exp_6 = __expf(-_cvt_f32_2.x);
                            denominator_0[0] = _exp_6;
                            float _exp_7 = __expf(-_cvt_f32_2.y);
                            denominator_0[1] = _exp_7;
                            const float2 _add2_8 = {1.0f, 1.0f};
                            #pragma unroll
                            for (int _la = 0; _la < 1; _la++) {
                                reinterpret_cast<float2*>(denominator_0)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(denominator_0)[_la], _add2_8);
                            }
                            float _rcp_9 = approx_rcp(denominator_0[0]);
                            float _rcp_10 = approx_rcp(denominator_0[1]);
                            activated_gate_1 = mul_f32x2_noftz(_cvt_f32_2, make_float2(_rcp_9, _rcp_10));
                            float2 gated_up_2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gated_up_2) : "l"(*(const unsigned long long*)&activated_gate_1), "l"(*(const unsigned long long*)&_cvt_f32_3));
                            float2 _f32x2_mul_b_11 = make_float2(first_weight, second_weight);
                            float2 activated_3;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&activated_3) : "l"(*(const unsigned long long*)&gated_up_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
                            activation[i_1 * 4 + 2] = activated_3.x;
                            activation[i_1 * 4 + 3] = activated_3.y;
                            float first_max;
                            asm("max.f32 %0, %1, %2;" : "=f"(first_max) : "f"(fabsf(activation[i_1 * 4])), "f"(fabsf(activation[i_1 * 4 + 2])));
                            float second_max;
                            asm("max.f32 %0, %1, %2;" : "=f"(second_max) : "f"(fabsf(activation[i_1 * 4 + 1])), "f"(fabsf(activation[i_1 * 4 + 3])));
                            float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, first_max, 4);
                            first_max = max_noftz(first_max, _shfl_xor_sync_12);
                            float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, second_max, 4);
                            second_max = max_noftz(second_max, _shfl_xor_sync_13);
                            float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, first_max, 8);
                            first_max = max_noftz(first_max, _shfl_xor_sync_14);
                            float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, second_max, 8);
                            second_max = max_noftz(second_max, _shfl_xor_sync_15);
                            float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, first_max, 16);
                            first_max = max_noftz(first_max, _shfl_xor_sync_16);
                            float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, second_max, 16);
                            second_max = max_noftz(second_max, _shfl_xor_sync_17);
                            amax[i_1 * 2] = first_max;
                            amax[i_1 * 2 + 1] = second_max;
                            if (lane < 4) {
                                amax_reduction[epi_warp * 16 + (unsigned int)(i_1 * 8) + (unsigned int)(lane * 2)] = first_max;
                                amax_reduction[epi_warp * 16 + (unsigned int)(i_1 * 8) + (unsigned int)(lane * 2) + 1] = second_max;
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            __syncwarp();
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        unsigned int tma_stage = s % 2;
                        unsigned int store_address = smem_output_addr + (epi_wg * 2 + tma_stage) * 16 * 64;
                        asm volatile("cp.async.bulk.wait_group 1;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 2; i_2++) {
                            unsigned int paired_index = (epi_warp ^ 1) * 16 + (unsigned int)(i_2 * 8) + (unsigned int)(lane % 4 * 2);
                            float first_max_1;
                            asm("max.f32 %0, %1, %2;" : "=f"(first_max_1) : "f"(amax[i_2 * 2]), "f"(amax_reduction[paired_index]));
                            float second_max_1;
                            asm("max.f32 %0, %1, %2;" : "=f"(second_max_1) : "f"(amax[i_2 * 2 + 1]), "f"(amax_reduction[paired_index + 1]));
                            unsigned int first_bits = __as_u32(first_max_1);
                            unsigned int second_bits = __as_u32(second_max_1);
                            unsigned int _max_18 = ((first_bits + 2097151 >> 23) > (113) ? (first_bits + 2097151 >> 23) : (113));
                            unsigned int first_exp = _max_18 - 8;
                            unsigned int _max_19 = ((second_bits + 2097151 >> 23) > (113) ? (second_bits + 2097151 >> 23) : (113));
                            unsigned int second_exp = _max_19 - 8;
                            unsigned int first_inverse_bits = 254 - first_exp << 23;
                            unsigned int second_inverse_bits = 254 - second_exp << 23;
                            float first_inverse = __uint_as_float(first_inverse_bits);
                            float second_inverse = __uint_as_float(second_inverse_bits);
                            quantized[0] = activation[i_2 * 4] * first_inverse;
                            quantized[1] = activation[i_2 * 4 + 1] * second_inverse;
                            quantized[2] = activation[i_2 * 4 + 2] * first_inverse;
                            quantized[3] = activation[i_2 * 4 + 3] * second_inverse;
                            unsigned int _fp8_0[1];
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
                            unsigned int stsm_address = store_address + (unsigned int)(i_2 * 8 * 64) + (unsigned int)(lane * 64) + (warp_in_wg ^ (unsigned int)(lane / 2)) * 16;
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                            uint32_t _stmatrix_b8_x1_addr_20 = static_cast<uint32_t>(stsm_address);
                            asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                                :: "r"(_stmatrix_b8_x1_addr_20), "r"(_fp8_0[0]) : "memory");
                            #elif defined(__CUDA_ARCH__)
                            #error "StmatrixTransB8X1 requires SM100 or newer"
                            #endif
                            if (warp_in_wg % 2 == 0 && lane < 4) {
                                unsigned int sf_k = n_block * 2 + warp_in_wg / 2;
                                unsigned int token_base = epi_wg * 16 + (unsigned int)(s * 16) + (unsigned int)(i_2 * 8);
                                unsigned int sf_address = (block_1 * 32 + token_base + (unsigned int)(lane * 2)) * 72 + sf_k;
                                SF_I_w[sf_address] = (uint8_t)first_exp;
                                SF_I_w[sf_address + 72] = (uint8_t)second_exp;
                                unsigned int source_token = pool_block_1 * 128 + token_base * 4 + (unsigned int)(lane * 8);
                                unsigned int source_address = sf_k / 4 * 147456 + source_token * 4 + sf_k % 4;
                                PrivateSF2Bytes[source_address] = (uint8_t)first_exp;
                                PrivateSF2Bytes[source_address + 16] = (uint8_t)second_exp;
                            }
                            __syncwarp();
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                        if (warp_in_wg == 0) {
                            if (elect_sync()) {
                                if (task_4[0] == 3) {
                                    tma_store_2d((&I_fp8_w), n_block * 64, block_1 * 32 + epi_wg * 16 + (unsigned int)(s * 16), store_address);
                                } else {
                                    tma_store_2d((&I_fp8_w), n_block * 64, block_1 * 32 + epi_wg * 16 + (unsigned int)(s * 16), store_address);
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            if (task_4[0] == 3) {
                                asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (pool_block_1))), "r"(static_cast<unsigned int>(1)) : "memory");
                            } else {
                                asm volatile("red.release.gpu.global.xor.b64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(PrivateMasks) + (ring_block))), "l"(static_cast<unsigned long long>((unsigned long long)1 << (unsigned long long)n_block)) : "memory");
                            }
                        }
                    }
                    __syncwarp();
                } else {
                    unsigned int epi_warp_1 = warp - 8;
                    unsigned int epi_wg_1 = epi_warp_1 / 4;
                    unsigned int warp_in_wg_1 = epi_warp_1 % 4;
                    unsigned int valid_m_1 = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                    unsigned int pool_m = (unsigned int)expert_row_offsets[task_4[1]] + task_4[2] * 32;
                    unsigned int n_offset_1 = (task_4[3] * 2 + (unsigned int)cta_rank) * 128;
                    unsigned int output_base = smem_output_addr + epi_wg_1 * 16 * 128 * 2;
                    unsigned int cached_token[2];
                    unsigned int cached_topk[2];
                    unsigned int packed[4];
                    #pragma unroll
                    for (int s_1 = 0; s_1 < 1; s_1++) {
                        if (valid_m_1 <= epi_wg_1 * 16 + (unsigned int)(s_1 * 16)) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            break;
                        }
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 2; j_1++) {
                            unsigned int row = epi_wg_1 * 16 + (unsigned int)(s_1 * 16) + (unsigned int)(j_1 * 8) + warp_in_wg_1 * 2 + (unsigned int)(lane / 16);
                            if (row < valid_m_1) {
                                if (task_4[0] == 4) {
                                    cached_token[j_1] = pool_m + row;
                                    cached_topk[j_1] = 6;
                                } else {
                                    cached_token[j_1] = pool_m + row;
                                    cached_topk[j_1] = 0;
                                }
                            }
                        }
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 2; i_3++) {
                            unsigned int address_1 = accum_stage_1 * 32 + epi_wg_1 * 16 + (unsigned int)(s_1 * 16) + (unsigned int)(i_3 * 8);
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
                            if (i_3 == 0 && s_1 > 0) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            if (s_1 == 0 && i_3 == 1) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            unsigned int _tmem_load_2_bf16[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                                _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int _tmem_load_3_bf16[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                                _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int row_1 = lane % 8;
                            unsigned int col = epi_warp_1 % 2 * 4 + (unsigned int)(lane / 8);
                            unsigned int write_address = output_base + warp_in_wg_1 / 2 * 16 * 128 + (unsigned int)(i_3 * 8 * 128) + row_1 * 128 + (col ^ row_1) * 16;
                            uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(write_address);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1]))
                                : "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 2; j_2++) {
                            unsigned int row_in_store = (unsigned int)(j_2 * 8) + warp_in_wg_1 * 2 + (unsigned int)(lane / 16);
                            unsigned int row_2 = epi_wg_1 * 16 + (unsigned int)(s_1 * 16) + row_in_store;
                            if (row_2 >= valid_m_1) {
                                break;
                            }
                            unsigned int row_in_atom = (warp_in_wg_1 * 2 + (unsigned int)(lane / 16)) % 8;
                            unsigned int read_address = output_base + (unsigned int)(lane % 16 / 8 * 16 * 128) + row_in_store * 128 + ((unsigned int)(lane % 8) ^ row_in_atom) * 16;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                : "r"(read_address));
                            unsigned long long dst_index = (unsigned long long)(pool_m + row_2) * 2560 + (unsigned long long)(n_offset_1 / 2) + (unsigned long long)(lane % 16 * 4);
                            reinterpret_cast<int4*>(PublicExpertOutput + dst_index)[0] = reinterpret_cast<int4*>(packed)[0];
                        }
                    }
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                }
            }
            if (warp == 8) {
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(128));
            }
            __threadfence();
            asm volatile("barrier.sync 15, 256;" ::: "memory");
            unsigned int l2_tag = (unsigned int)1 << 31;
            unsigned int l2_addend = 1;
            if (bid == 0) {
                l2_addend = l2_tag - (unsigned int)(NUM_CTAS - 1);
            }
            unsigned int l2_ticket = 0;
            if (warp == 8) {
                if (lane == 0) {
                    unsigned int _atomic_fetch_add_22;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_add_22) : "l"(l2_done), "r"(static_cast<uint32_t>(l2_addend)) : "memory");
                    l2_ticket = _atomic_fetch_add_22;
                }
            }
            if (warp == 8) {
                if (lane == 0) {
                    unsigned int _wait_acquire_mask_4;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_4) : "l"(reinterpret_cast<unsigned int*>(l2_done)) : "memory");
                    } while (((_wait_acquire_mask_4 ^ static_cast<unsigned int>(l2_ticket ^ l2_tag)) & static_cast<unsigned int>(l2_tag)) != 0);
                }
            }
            asm volatile("barrier.sync 15, 256;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int epi_warp_2 = warp - 8;
            unsigned int phase_1 = 0;
            unsigned int values[4];
            float reduced[40];
            unsigned int store_values[4];
            #pragma unroll 1
            for (unsigned int token_chunk = epi_warp_2 * (unsigned int)NUM_CTAS + (unsigned int)bid; token_chunk < num_tokens * 4; token_chunk += NUM_CTAS * 8) {
                unsigned int token = token_chunk / 4;
                unsigned int chunk = token_chunk % 4;
                int stored_row = -1;
                if (lane < 6) {
                    stored_row = token_to_permuted[token * 6 + (unsigned int)lane];
                }
                unsigned int routed = __ballot_sync(0xFFFFFFFF, stored_row >= 0);
                #pragma unroll 1
                for (unsigned int i_4 = 0; i_4 < 6; i_4++) {
                    unsigned int row_3 = __shfl_sync(0xFFFFFFFF, (unsigned int)stored_row, i_4);
                    if (elect_sync()) {
                        if ((routed >> i_4 & 1) != 0) {
                            cp_async_bulk_gmem2smem(combine_smem_addr + (epi_warp_2 * 6 + i_4) * 2560, expert_output + ((unsigned long long)row_3 * 5120 + (unsigned long long)(chunk * 1280)), 2560, combine_barriers_addr + (epi_warp_2 * 6 + i_4) * 8);
                            mbarrier_arrive_expect_tx(combine_barriers_addr + (epi_warp_2 * 6 + i_4) * 8, 2560);
                        } else {
                            mbarrier_arrive(combine_barriers_addr + (epi_warp_2 * 6 + i_4) * 8);
                        }
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
                reduced[16] = 0.0f;
                reduced[17] = 0.0f;
                reduced[18] = 0.0f;
                reduced[19] = 0.0f;
                reduced[20] = 0.0f;
                reduced[21] = 0.0f;
                reduced[22] = 0.0f;
                reduced[23] = 0.0f;
                reduced[24] = 0.0f;
                reduced[25] = 0.0f;
                reduced[26] = 0.0f;
                reduced[27] = 0.0f;
                reduced[28] = 0.0f;
                reduced[29] = 0.0f;
                reduced[30] = 0.0f;
                reduced[31] = 0.0f;
                reduced[32] = 0.0f;
                reduced[33] = 0.0f;
                reduced[34] = 0.0f;
                reduced[35] = 0.0f;
                reduced[36] = 0.0f;
                reduced[37] = 0.0f;
                reduced[38] = 0.0f;
                reduced[39] = 0.0f;
                #pragma unroll 1
                for (unsigned int i_5 = 0; i_5 < 6; i_5++) {
                    mbarrier_wait(combine_barriers_addr + (epi_warp_2 * 6 + i_5) * 8, phase_1);
                    if ((routed >> i_5 & 1) != 0) {
                        #pragma unroll
                        for (unsigned int j_3 = 0; j_3 < 5; j_3++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&values[0])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 3]))
                                : "r"(combine_smem_addr + (epi_warp_2 * 6 + i_5) * 2560 + (j_3 * 32 + (unsigned int)lane) * 16));
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
                    }
                }
                phase_1 ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                #pragma unroll
                for (unsigned int j_4 = 0; j_4 < 5; j_4++) {
                    store_values[0] = __as_u32(__float22bfloat162_rn(make_float2(reduced[j_4 * 8], reduced[j_4 * 8 + 1])));
                    store_values[1] = __as_u32(__float22bfloat162_rn(make_float2(reduced[j_4 * 8 + 2], reduced[j_4 * 8 + 2 + 1])));
                    store_values[2] = __as_u32(__float22bfloat162_rn(make_float2(reduced[j_4 * 8 + 4], reduced[j_4 * 8 + 4 + 1])));
                    store_values[3] = __as_u32(__float22bfloat162_rn(make_float2(reduced[j_4 * 8 + 6], reduced[j_4 * 8 + 6 + 1])));
                    if (j_4 == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        __syncwarp();
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(combine_smem_addr + (48 + epi_warp_2) * 2560 + (j_4 * 32 + (unsigned int)lane) * 16), "r"(store_values[0]), "r"(store_values[1]), "r"(store_values[2]), "r"(store_values[3]));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    {
                        void* _cpbulk_dst_23 = reinterpret_cast<void*>(y + ((unsigned long long)token * 5120 + (unsigned long long)(chunk * 1280)));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_23), "r"(combine_smem_addr + (48 + epi_warp_2) * 2560), "r"((uint32_t)(2560))
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
