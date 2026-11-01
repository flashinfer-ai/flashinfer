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
#define TMEM_NCOLS 40
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SCALE_A_OFFSET 32
#define TMEM_SCALE_B_OFFSET 36
#define NUM_PIPE_STAGES 11
#define NUM_ACCUM_PIPE_STAGES 2
#define NUM_TASK_PIPE_STAGES 4
#define SMEM_COMBINE_SMEM_OFF 0
#define SMEM_COMBINE_SMEM_STAGE_BYTES 143360
#define SMEM_COMBINE_SMEM_STRIDE 143360
#define SMEM_SMEM_HIST_OFF 0
#define SMEM_SMEM_HIST_STAGE_BYTES 1536
#define SMEM_SMEM_HIST_STRIDE 1536
#define SMEM_SMEM_DST_ROW_OFF 2048
#define SMEM_SMEM_DST_ROW_STAGE_BYTES 16
#define SMEM_SMEM_DST_ROW_STRIDE 16
#define SMEM_SMEM_ERO_OFF 2064
#define SMEM_SMEM_ERO_STAGE_BYTES 1536
#define SMEM_SMEM_ERO_STRIDE 1536
#define SMEM_SMEM_OUTPUT_OFF 22528
#define SMEM_SMEM_OUTPUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUTPUT_STRIDE 4096
#define SMEM_SMEM_A_OFF 26624
#define SMEM_SMEM_A_STAGE_BYTES 1024
#define SMEM_SMEM_A_STRIDE 1024
#define SMEM_SMEM_B_OFF 37888
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 218112
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 512
#define SMEM_SMEM_SFB_OFF 223744
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 512
#define SMEM_AMAX_REDUCTION_OFF 229376
#define SMEM_AMAX_REDUCTION_STAGE_BYTES 256
#define SMEM_AMAX_REDUCTION_STRIDE 256
#define SMEM_TASK_INFOS_OFF 229632
#define SMEM_TASK_INFOS_STAGE_BYTES 128
#define SMEM_TASK_INFOS_STRIDE 128
#define SMEM_TOTAL 230528
#define THREADS 512
#define NUM_CTAS 152
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



__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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
kernel_deepgemm_mega_moe_v3_pipeline_fp4_e384_i384_t16_sm103a(int* __restrict__ topk_idx_i32, float* __restrict__ topk_weights, int* __restrict__ x_fp8, unsigned int* __restrict__ x_sf, const __grid_constant__ CUtensorMap A, int* __restrict__ pool_fp8, unsigned int* __restrict__ pool_sf, float* __restrict__ routing_weight_pool, int* __restrict__ token_to_permuted, int* __restrict__ meta_token, int* __restrict__ meta_slot, int* __restrict__ expert_counts, int* __restrict__ expert_row_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ tile_expert, int* __restrict__ tile_m_local, int* __restrict__ total_m_tiles_out, const __grid_constant__ CUtensorMap B1, const __grid_constant__ CUtensorMap SFB1, const __grid_constant__ CUtensorMap I_fp8_w, uint8_t* __restrict__ SF_I_w, __nv_bfloat16* __restrict__ l1_bf16_capture, const __grid_constant__ CUtensorMap A2, unsigned int* __restrict__ SFA2, const __grid_constant__ CUtensorMap B2, const __grid_constant__ CUtensorMap SFB2, __nv_bfloat16* __restrict__ expert_output, __nv_bfloat16* __restrict__ y, unsigned int* __restrict__ histogram_done, unsigned int* __restrict__ prefix_done, unsigned int* __restrict__ dispatch_done, unsigned int* __restrict__ l1_arrival, unsigned int* __restrict__ l2_done, int num_tokens, int top_k, int num_experts, int N1, int K1, int grid_n1, int K1_tiles, int N2, int K2, int grid_n2, int K2_tiles, int total_m_tiles, int M_total, float activation_clamp, unsigned int* __restrict__ PrivateCounters, unsigned long long* __restrict__ PrivateMasks, unsigned int* __restrict__ PublicExpertOutput, unsigned int* __restrict__ PrivateSFBlockOffsets, const __grid_constant__ CUtensorMap PrivateSF1, const __grid_constant__ CUtensorMap PrivateSF2, unsigned int* __restrict__ PrivateSF1Words, uint8_t* __restrict__ PrivateSF2Bytes, int local_expert_offset)
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

    const int mbar_base = smem + 229760;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 88)
    #define tmem_full_addr (mbar_base + 176)
    #define tmem_empty_addr (mbar_base + 192)
    #define task_full_addr (mbar_base + 208)
    #define task_empty_addr (mbar_base + 240)
    #define combine_barriers_addr (mbar_base + 272)

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
    int* smem_dst_row = reinterpret_cast<int*>(smem_raw + 2048);
    const int smem_dst_row_addr = smem + 2048;
    int* smem_ero = reinterpret_cast<int*>(smem_raw + 2064);
    const int smem_ero_addr = smem + 2064;
    uint8_t* smem_output = reinterpret_cast<uint8_t*>(smem_raw + 22528);
    const int smem_output_addr = smem + 22528;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 26624);
    const int smem_a_addr = smem + 26624;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_b_addr = smem + 37888;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 218112);
    const int smem_sfa_addr = smem + 218112;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 223744);
    const int smem_sfb_addr = smem + 223744;
    float* amax_reduction = reinterpret_cast<float*>(smem_raw + 229376);
    const int amax_reduction_addr = smem + 229376;
    unsigned int* task_infos = reinterpret_cast<unsigned int*>(smem_raw + 229632);
    const int task_infos_addr = smem + 229632;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B1))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB1))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&I_fp8_w))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A2))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B2))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB2))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&PrivateSF1))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&PrivateSF2))) : "memory"); }
    if (warp == 0) {
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(task_infos_addr + (unsigned int)lane * 4), "r"(0) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 82 barriers)
    // Mbarriers at smem_raw[229760..230416)

    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 11 barriers, init_count=4
            mbarrier_init(smem + 229760, 4);
            mbarrier_init(smem + 229768, 4);
            mbarrier_init(smem + 229776, 4);
            mbarrier_init(smem + 229784, 4);
            mbarrier_init(smem + 229792, 4);
            mbarrier_init(smem + 229800, 4);
            mbarrier_init(smem + 229808, 4);
            mbarrier_init(smem + 229816, 4);
            mbarrier_init(smem + 229824, 4);
            mbarrier_init(smem + 229832, 4);
            mbarrier_init(smem + 229840, 4);
            // empty: 11 barriers, init_count=1
            mbarrier_init(smem + 229848, 1);
            mbarrier_init(smem + 229856, 1);
            mbarrier_init(smem + 229864, 1);
            mbarrier_init(smem + 229872, 1);
            mbarrier_init(smem + 229880, 1);
            mbarrier_init(smem + 229888, 1);
            mbarrier_init(smem + 229896, 1);
            mbarrier_init(smem + 229904, 1);
            mbarrier_init(smem + 229912, 1);
            mbarrier_init(smem + 229920, 1);
            mbarrier_init(smem + 229928, 1);
            // --- pipeline 'accum_pipe' ---
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 229936, 1);
            mbarrier_init(smem + 229944, 1);
            // tmem_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 229952, 512);
            mbarrier_init(smem + 229960, 512);
            // --- pipeline 'task_pipe' ---
            // task_full: 4 barriers, init_count=1
            mbarrier_init(smem + 229968, 1);
            mbarrier_init(smem + 229976, 1);
            mbarrier_init(smem + 229984, 1);
            mbarrier_init(smem + 229992, 1);
            // task_empty: 4 barriers, init_count=512
            mbarrier_init(smem + 230000, 512);
            mbarrier_init(smem + 230008, 512);
            mbarrier_init(smem + 230016, 512);
            mbarrier_init(smem + 230024, 512);
            // combine_barriers: 48 barriers, init_count=1
            mbarrier_init(smem + 230032, 1);
            mbarrier_init(smem + 230040, 1);
            mbarrier_init(smem + 230048, 1);
            mbarrier_init(smem + 230056, 1);
            mbarrier_init(smem + 230064, 1);
            mbarrier_init(smem + 230072, 1);
            mbarrier_init(smem + 230080, 1);
            mbarrier_init(smem + 230088, 1);
            mbarrier_init(smem + 230096, 1);
            mbarrier_init(smem + 230104, 1);
            mbarrier_init(smem + 230112, 1);
            mbarrier_init(smem + 230120, 1);
            mbarrier_init(smem + 230128, 1);
            mbarrier_init(smem + 230136, 1);
            mbarrier_init(smem + 230144, 1);
            mbarrier_init(smem + 230152, 1);
            mbarrier_init(smem + 230160, 1);
            mbarrier_init(smem + 230168, 1);
            mbarrier_init(smem + 230176, 1);
            mbarrier_init(smem + 230184, 1);
            mbarrier_init(smem + 230192, 1);
            mbarrier_init(smem + 230200, 1);
            mbarrier_init(smem + 230208, 1);
            mbarrier_init(smem + 230216, 1);
            mbarrier_init(smem + 230224, 1);
            mbarrier_init(smem + 230232, 1);
            mbarrier_init(smem + 230240, 1);
            mbarrier_init(smem + 230248, 1);
            mbarrier_init(smem + 230256, 1);
            mbarrier_init(smem + 230264, 1);
            mbarrier_init(smem + 230272, 1);
            mbarrier_init(smem + 230280, 1);
            mbarrier_init(smem + 230288, 1);
            mbarrier_init(smem + 230296, 1);
            mbarrier_init(smem + 230304, 1);
            mbarrier_init(smem + 230312, 1);
            mbarrier_init(smem + 230320, 1);
            mbarrier_init(smem + 230328, 1);
            mbarrier_init(smem + 230336, 1);
            mbarrier_init(smem + 230344, 1);
            mbarrier_init(smem + 230352, 1);
            mbarrier_init(smem + 230360, 1);
            mbarrier_init(smem + 230368, 1);
            mbarrier_init(smem + 230376, 1);
            mbarrier_init(smem + 230384, 1);
            mbarrier_init(smem + 230392, 1);
            mbarrier_init(smem + 230400, 1);
            mbarrier_init(smem + 230408, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 40 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 230416);
    if (warp == 3) {
        int _tmem_hold = smem + 230416;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_scale_a = taddr + 32;
    const int tmem_scale_b = taddr + 36;

    // ---- Role: dispatch ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // dispatch_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
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
            #pragma unroll 1
            for (int i = disp_tid; i < num_experts; i += 128) {
                smem_hist[i] = 0;
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            #pragma unroll 1
            for (int idx = disp_tid; idx < total_pairs; idx += 128) {
                int e_h = topk_idx_i32[idx * 2] - local_expert_offset;
                if ((unsigned int)e_h < 384) {
                    atomicAdd(&smem_hist[e_h], 1);
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (bid == 0) {
                #pragma unroll 1
                for (int i_1 = disp_tid; i_1 < num_experts; i_1 += 128) {
                    int c_h = smem_hist[i_1];
                    if (c_h > 0) {
                        atomicAdd(&expert_counts[i_1], c_h);
                    }
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (warp == 0) {
                unsigned int fold_public_prior = 0;
                unsigned int fold_source_prior = 0;
                unsigned int fold_prefetched[12];
                int fold_count_11_0 = smem_hist[lane];
                fold_prefetched[0] = (unsigned int)fold_count_11_0;
                int fold_count_11_1 = smem_hist[32 + lane];
                fold_prefetched[1] = (unsigned int)fold_count_11_1;
                int fold_count_11_2 = smem_hist[64 + lane];
                fold_prefetched[2] = (unsigned int)fold_count_11_2;
                int fold_count_11_3 = smem_hist[96 + lane];
                fold_prefetched[3] = (unsigned int)fold_count_11_3;
                int fold_count_11_4 = smem_hist[128 + lane];
                fold_prefetched[4] = (unsigned int)fold_count_11_4;
                int fold_count_11_5 = smem_hist[160 + lane];
                fold_prefetched[5] = (unsigned int)fold_count_11_5;
                int fold_count_11_6 = smem_hist[192 + lane];
                fold_prefetched[6] = (unsigned int)fold_count_11_6;
                int fold_count_11_7 = smem_hist[224 + lane];
                fold_prefetched[7] = (unsigned int)fold_count_11_7;
                int fold_count_11_8 = smem_hist[256 + lane];
                fold_prefetched[8] = (unsigned int)fold_count_11_8;
                int fold_count_11_9 = smem_hist[288 + lane];
                fold_prefetched[9] = (unsigned int)fold_count_11_9;
                int fold_count_11_10 = smem_hist[320 + lane];
                fold_prefetched[10] = (unsigned int)fold_count_11_10;
                int fold_count_11_11 = smem_hist[352 + lane];
                fold_prefetched[11] = (unsigned int)fold_count_11_11;
                unsigned int fold_public_offsets[12];
                unsigned int fold_source_offsets[12];
                #pragma unroll
                for (int stripe = 0; stripe < 12; stripe++) {
                    unsigned int fold_prefix_count = fold_prefetched[stripe];
                    unsigned int fold_public_tiles = (fold_prefix_count + 256 - 1) / 256 * 2;
                    unsigned int fold_source_blocks = (fold_prefix_count + 16 - 1) / 16;
                    uint32_t fold_public_inclusive = fold_public_tiles;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_public_inclusive, 1, 32); if (lane >= 1) { fold_public_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_public_inclusive, 2, 32); if (lane >= 2) { fold_public_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_public_inclusive, 4, 32); if (lane >= 4) { fold_public_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_public_inclusive, 8, 32); if (lane >= 8) { fold_public_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_public_inclusive, 16, 32); if (lane >= 16) { fold_public_inclusive += peer; } }
                    uint32_t fold_source_inclusive = fold_source_blocks;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_source_inclusive, 1, 32); if (lane >= 1) { fold_source_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_source_inclusive, 2, 32); if (lane >= 2) { fold_source_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_source_inclusive, 4, 32); if (lane >= 4) { fold_source_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_source_inclusive, 8, 32); if (lane >= 8) { fold_source_inclusive += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, fold_source_inclusive, 16, 32); if (lane >= 16) { fold_source_inclusive += peer; } }
                    fold_public_offsets[stripe] = fold_public_prior + fold_public_inclusive - fold_public_tiles;
                    fold_source_offsets[stripe] = fold_source_prior + fold_source_inclusive - fold_source_blocks;
                    unsigned int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, fold_public_inclusive, 31);
                    fold_public_prior += _shfl_sync_0;
                    unsigned int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, fold_source_inclusive, 31);
                    fold_source_prior += _shfl_sync_1;
                }
                #pragma unroll
                for (int stripe_1 = 0; stripe_1 < 12; stripe_1++) {
                    unsigned int fold_expert = stripe_1 * 32 + lane;
                    unsigned int fold_prefix_count_1 = fold_prefetched[stripe_1];
                    unsigned int fold_public_tiles_1 = (fold_prefix_count_1 + 256 - 1) / 256 * 2;
                    unsigned int fold_public_offset = fold_public_offsets[stripe_1];
                    unsigned int fold_source_offset = fold_source_offsets[stripe_1];
                    if (fold_expert < (unsigned int)num_experts) {
                        expert_row_offsets[fold_expert] = (int)(fold_public_offset * 128);
                        smem_ero[fold_expert] = (int)(fold_public_offset * 128);
                        PrivateSFBlockOffsets[fold_expert] = fold_source_offset;
                    }
                }
                if (bid == 0) {
                    if (lane == 0) {
                        total_m_tiles_out[0] = (int)fold_public_prior;
                    }
                }
            }
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            if (bid == 0) {
                if (warp == 0) {
                    #pragma unroll
                    for (int tl_stripe = 0; tl_stripe < 12; tl_stripe++) {
                        unsigned int tl_expert = tl_stripe * 32 + lane;
                        if (tl_expert < (unsigned int)num_experts) {
                            int tl_count = smem_hist[tl_expert];
                            unsigned int tl_tiles = ((unsigned int)tl_count + 256 - 1) / 256 * 2;
                            unsigned int tl_offset = (unsigned int)smem_ero[tl_expert] / 128;
                            #pragma unroll 1
                            for (int tt = 0; tt < tl_tiles; tt++) {
                                tile_expert[tl_offset + (unsigned int)tt] = (int)tl_expert;
                                tile_m_local[tl_offset + (unsigned int)tt] = (int)(tt * 128);
                            }
                        }
                    }
                }
            }
            int global_warp_idx = bid * 4 + disp_warp;
            int warps_per_grid = NUM_CTAS * 4;
            #pragma unroll 1
            for (int pair_idx = global_warp_idx; pair_idx < total_pairs; pair_idx += warps_per_grid) {
                int e_s = topk_idx_i32[pair_idx * 2] - local_expert_offset;
                if ((unsigned int)e_s >= 384) {
                    if (lane == 0) {
                        token_to_permuted[pair_idx] = -1;
                    }
                    continue;
                }
                int t_s = pair_idx / top_k;
                int k_s = pair_idx - t_s * top_k;
                int rank = 0;
                #pragma unroll
                for (int rank_round = 0; rank_round < 3; rank_round++) {
                    int rank_pair = rank_round * 32 + lane;
                    int rank_match = 0;
                    if (rank_pair < pair_idx) {
                        int rank_route = topk_idx_i32[rank_pair * 2] - local_expert_offset;
                        if (rank_route == e_s) {
                            rank_match = 1;
                        }
                    }
                    unsigned int rank_ballot = __ballot_sync(0xFFFFFFFF, rank_match != 0);
                    rank += (int)(unsigned int)__popc(rank_ballot);
                }
                int dst_row = smem_ero[e_s] + rank;
                if (lane == 0) {
                    float cached_rw = topk_weights[pair_idx];
                    token_to_permuted[pair_idx] = dst_row;
                    meta_token[dst_row] = t_s;
                    meta_slot[dst_row] = k_s;
                    routing_weight_pool[dst_row] = cached_rw;
                }
                __syncwarp();
                unsigned long long src_u32_off = (unsigned long long)t_s * (unsigned long long)hidden_u32;
                unsigned long long dst_u32_off = (unsigned long long)dst_row * (unsigned long long)hidden_u32;
                int pull_lane_off = lane * 4;
                int _vec_load_0[4];
                {
                    const int4* _ivptr_2 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off) + 0);
                    int4 _ivld_2;
                    _ivld_2 = *_ivptr_2;
                    _vec_load_0[0 + 0] = _ivld_2.x;
                    _vec_load_0[0 + 1] = _ivld_2.y;
                    _vec_load_0[0 + 2] = _ivld_2.z;
                    _vec_load_0[0 + 3] = _ivld_2.w;
                }
                int _vec_load_1[4];
                {
                    const int4* _ivptr_3 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 128) + 0);
                    int4 _ivld_3;
                    _ivld_3 = *_ivptr_3;
                    _vec_load_1[0 + 0] = _ivld_3.x;
                    _vec_load_1[0 + 1] = _ivld_3.y;
                    _vec_load_1[0 + 2] = _ivld_3.z;
                    _vec_load_1[0 + 3] = _ivld_3.w;
                }
                int _vec_load_2[4];
                {
                    const int4* _ivptr_4 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 256) + 0);
                    int4 _ivld_4;
                    _ivld_4 = *_ivptr_4;
                    _vec_load_2[0 + 0] = _ivld_4.x;
                    _vec_load_2[0 + 1] = _ivld_4.y;
                    _vec_load_2[0 + 2] = _ivld_4.z;
                    _vec_load_2[0 + 3] = _ivld_4.w;
                }
                int _vec_load_3[4];
                {
                    const int4* _ivptr_5 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 384) + 0);
                    int4 _ivld_5;
                    _ivld_5 = *_ivptr_5;
                    _vec_load_3[0 + 0] = _ivld_5.x;
                    _vec_load_3[0 + 1] = _ivld_5.y;
                    _vec_load_3[0 + 2] = _ivld_5.z;
                    _vec_load_3[0 + 3] = _ivld_5.w;
                }
                int _vec_load_4[4];
                {
                    const int4* _ivptr_6 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 512) + 0);
                    int4 _ivld_6;
                    _ivld_6 = *_ivptr_6;
                    _vec_load_4[0 + 0] = _ivld_6.x;
                    _vec_load_4[0 + 1] = _ivld_6.y;
                    _vec_load_4[0 + 2] = _ivld_6.z;
                    _vec_load_4[0 + 3] = _ivld_6.w;
                }
                int _vec_load_5[4];
                {
                    const int4* _ivptr_7 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 640) + 0);
                    int4 _ivld_7;
                    _ivld_7 = *_ivptr_7;
                    _vec_load_5[0 + 0] = _ivld_7.x;
                    _vec_load_5[0 + 1] = _ivld_7.y;
                    _vec_load_5[0 + 2] = _ivld_7.z;
                    _vec_load_5[0 + 3] = _ivld_7.w;
                }
                int _vec_load_6[4];
                {
                    const int4* _ivptr_8 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 768) + 0);
                    int4 _ivld_8;
                    _ivld_8 = *_ivptr_8;
                    _vec_load_6[0 + 0] = _ivld_8.x;
                    _vec_load_6[0 + 1] = _ivld_8.y;
                    _vec_load_6[0 + 2] = _ivld_8.z;
                    _vec_load_6[0 + 3] = _ivld_8.w;
                }
                int _vec_load_7[4];
                {
                    const int4* _ivptr_9 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 896) + 0);
                    int4 _ivld_9;
                    _ivld_9 = *_ivptr_9;
                    _vec_load_7[0 + 0] = _ivld_9.x;
                    _vec_load_7[0 + 1] = _ivld_9.y;
                    _vec_load_7[0 + 2] = _ivld_9.z;
                    _vec_load_7[0 + 3] = _ivld_9.w;
                }
                int _vec_load_8[4];
                {
                    const int4* _ivptr_10 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 1024) + 0);
                    int4 _ivld_10;
                    _ivld_10 = *_ivptr_10;
                    _vec_load_8[0 + 0] = _ivld_10.x;
                    _vec_load_8[0 + 1] = _ivld_10.y;
                    _vec_load_8[0 + 2] = _ivld_10.z;
                    _vec_load_8[0 + 3] = _ivld_10.w;
                }
                int _vec_load_9[4];
                {
                    const int4* _ivptr_11 = reinterpret_cast<const int4*>(x_fp8 + (src_u32_off + (unsigned long long)pull_lane_off + 1152) + 0);
                    int4 _ivld_11;
                    _ivld_11 = *_ivptr_11;
                    _vec_load_9[0 + 0] = _ivld_11.x;
                    _vec_load_9[0 + 1] = _ivld_11.y;
                    _vec_load_9[0 + 2] = _ivld_11.z;
                    _vec_load_9[0 + 3] = _ivld_11.w;
                }
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off))[0] = reinterpret_cast<int4*>(_vec_load_0)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 128))[0] = reinterpret_cast<int4*>(_vec_load_1)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 256))[0] = reinterpret_cast<int4*>(_vec_load_2)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 384))[0] = reinterpret_cast<int4*>(_vec_load_3)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 512))[0] = reinterpret_cast<int4*>(_vec_load_4)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 640))[0] = reinterpret_cast<int4*>(_vec_load_5)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 768))[0] = reinterpret_cast<int4*>(_vec_load_6)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 896))[0] = reinterpret_cast<int4*>(_vec_load_7)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 1024))[0] = reinterpret_cast<int4*>(_vec_load_8)[0];
                reinterpret_cast<int4*>(pool_fp8 + (dst_u32_off + (unsigned long long)pull_lane_off + 1152))[0] = reinterpret_cast<int4*>(_vec_load_9)[0];
                unsigned long long src_sf_off = (unsigned long long)t_s * (unsigned long long)sf_words_x;
                unsigned long long dst_sf_off = (unsigned long long)dst_row * (unsigned long long)sf_words_x;
                unsigned int in_expert = (unsigned int)(dst_row - smem_ero[e_s]);
                unsigned int sf_block = PrivateSFBlockOffsets[e_s] + in_expert / 16;
                unsigned int source_sf_token = sf_block * 128 + in_expert % 16 * 4;
                unsigned int sfw0 = x_sf[src_sf_off + (unsigned long long)lane];
                unsigned int sfw1 = 0;
                if (sf_words_x > lane + 32) {
                    sfw1 = x_sf[src_sf_off + (unsigned long long)lane + 32];
                }
                pool_sf[dst_sf_off + (unsigned long long)lane] = sfw0;
                PrivateSF1Words[(unsigned int)(lane * 49920) + source_sf_token] = sfw0;
                if (sf_words_x > lane + 32) {
                    pool_sf[dst_sf_off + (unsigned long long)lane + 32] = sfw1;
                    PrivateSF1Words[(unsigned int)((lane + 32) * 49920) + source_sf_token] = sfw1;
                }
                __syncwarp();
                if (elect_sync()) {
                    unsigned int arrivals = 1;
                    int fold_expert_count = smem_hist[e_s];
                    if (in_expert + 1 == (unsigned int)fold_expert_count) {
                        arrivals = 16 - in_expert % 16;
                    }
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (782 + sf_block))), "r"(static_cast<unsigned int>(arrivals)) : "memory");
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int _atomic_old_0;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_0) : "l"(dispatch_done), "r"(static_cast<uint32_t>(done_addend)) : "memory");
                }
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (bid == 0) {
                #pragma unroll 1
                for (int i_2 = disp_tid; i_2 < num_experts; i_2 += 128) {
                    expert_counts[i_2] = 0;
                    expert_scatter_offsets[i_2] = 0;
                }
                if (disp_tid == 0) {
                    PrivateCounters[0] = (unsigned int)0;
                    PrivateCounters[1] = (unsigned int)0;
                }
            } else {
                #pragma unroll 1
                for (int block = (bid - 1) * 128 + disp_tid; block < 390; block += (NUM_CTAS - 1) * 128) {
                    PrivateMasks[block] = (unsigned long long)0;
                    PrivateCounters[782 + block] = (unsigned int)0;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            unsigned int task[8];
            unsigned int a_stage = 0;
            unsigned int group_block = 4294967295;
            unsigned int group_token_offset = 0;
            unsigned int group_ready = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (int task_iteration = 0; task_iteration < 8971; task_iteration++) {
                int _stage_task_full = (int)((unsigned int)task_iteration % 4);
                int _parity_task_full = (int)((unsigned int)task_iteration / 4 % 2);
                mbarrier_wait(task_full_addr + (_stage_task_full) * 8, _parity_task_full);
                unsigned int slot = task_infos_addr + (unsigned int)task_iteration % 4 * 32;
                unsigned int e24 = ((unsigned int)task_iteration / 4 & 127 | 128) << 24;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&task[0])), "=r"(*reinterpret_cast<uint32_t*>(&task[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task[(0) + 3]))
                    : "r"(slot));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&task[4])), "=r"(*reinterpret_cast<uint32_t*>(&task[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task[(4) + 3]))
                    : "r"(slot + 16));
                unsigned int x0 = task[0] ^ e24;
                unsigned int x4 = task[4] ^ e24;
                int _warp_vote_0 = __all_sync(0xFFFFFFFF, (x0 | x4) < 16777216);
                if (_warp_vote_0 == 0) {
                    while (1) {
                        uint32_t word_0;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_0) : "r"(static_cast<uint32_t>(slot)));
                        task[0] = word_0;
                        uint32_t word_1;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_1) : "r"(static_cast<uint32_t>(slot + 4)));
                        task[1] = word_1;
                        uint32_t word_2;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_2) : "r"(static_cast<uint32_t>(slot + 8)));
                        task[2] = word_2;
                        uint32_t word_3;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_3) : "r"(static_cast<uint32_t>(slot + 12)));
                        task[3] = word_3;
                        uint32_t word_4;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_4) : "r"(static_cast<uint32_t>(slot + 16)));
                        task[4] = word_4;
                        uint32_t word_5;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_5) : "r"(static_cast<uint32_t>(slot + 20)));
                        task[5] = word_5;
                        uint32_t word_6;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_6) : "r"(static_cast<uint32_t>(slot + 24)));
                        task[6] = word_6;
                        uint32_t word_7;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_7) : "r"(static_cast<uint32_t>(slot + 28)));
                        task[7] = word_7;
                        x0 = task[0] ^ e24;
                        x4 = task[4] ^ e24;
                        int _warp_vote_1 = __all_sync(0xFFFFFFFF, (x0 | x4) < 16777216);
                        if (_warp_vote_1 != 0) {
                            break;
                        }
                    }
                    __syncwarp();
                }
                task[0] = x0;
                task[4] = x4;
                if (task[0] == 0) {
                    break;
                }
                unsigned int group_hit = 0;
                if (task[0] == 2) {
                    if (task[4] == group_block) {
                        group_hit = 1;
                    }
                }
                unsigned int token_offset = group_token_offset;
                unsigned long long pending = 63;
                if (group_hit == 0) {
                    unsigned int public_m = (unsigned int)smem_ero[task[1]] + task[2] * 16;
                    unsigned int aligned_m = (task[5] + 15) / 16 * 16;
                    token_offset = public_m + (unsigned int)cta_rank * (aligned_m / 2);
                } else if (group_ready != 0) {
                    pending = (unsigned long long)0;
                }
                group_block = 4294967295;
                if (task[0] == 2) {
                    group_block = task[4];
                }
                group_token_offset = token_offset;
                if (task[0] == 1) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(PrivateCounters) + (782 + task[4]))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(16)) >= static_cast<unsigned int>(1));
                    }
                }
                unsigned int w2t_subs = ((task[0] == 2) ? 2 : 1);
                #pragma unroll 1
                for (int _sub = 0; _sub < w2t_subs; _sub++) {
                    #pragma unroll 2
                    for (int k = 0; k < task[7] / 128; k++) {
                        if (task[0] == 2) {
                            unsigned long long k_mask = (unsigned long long)3 << (unsigned long long)(k * 2);
                            if ((pending & k_mask) != 0) {
                                unsigned long long _wait_acquire_mask_0;
                                do {
                                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_wait_acquire_mask_0) : "l"((reinterpret_cast<unsigned long long*>(PrivateMasks) + (task[4]))) : "memory");
                                } while (((_wait_acquire_mask_0 ^ static_cast<unsigned long long>((unsigned long long)63)) & static_cast<unsigned long long>(k_mask)) != 0);
                                unsigned long long observed = _wait_acquire_mask_0;
                                pending = observed ^ 63;
                            }
                        }
                        mbarrier_wait(empty_addr + (a_stage) * 8, _phase_empty);
                        if (elect_sync()) {
                            if (task[0] == 1) {
                                tma_2d_gmem2smem_cta2(smem_a_addr + a_stage * 1024, (&A), k * 128, token_offset, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 512, (&PrivateSF1), task[4] * 128, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            } else {
                                tma_2d_gmem2smem_cta2(smem_a_addr + a_stage * 1024, (&A2), k * 128, token_offset, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(smem_sfa_addr + a_stage * 512, (&PrivateSF2), task[4] * 128, k, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            }
                            if (cta_rank == 0) {
                                mbarrier_arrive_expect_tx(full_addr + (a_stage) * 8, 3072);
                            } else {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((full_addr + (a_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        __syncwarp();
                        a_stage += 1;
                        if (a_stage == 11) { a_stage = 0; _phase_empty ^= 1; }
                    }
                }
                group_ready = 0;
                if (pending == 0) {
                    group_ready = 1;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 5) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            unsigned int task_1[8];
            unsigned int b_stage = 0;
            unsigned int _phase_empty_1 = 1;
            #pragma unroll 1
            for (int task_iteration_1 = 0; task_iteration_1 < 8971; task_iteration_1++) {
                int _stage_task_full_1 = (int)((unsigned int)task_iteration_1 % 4);
                int _parity_task_full_1 = (int)((unsigned int)task_iteration_1 / 4 % 2);
                mbarrier_wait(task_full_addr + (_stage_task_full_1) * 8, _parity_task_full_1);
                unsigned int slot_1 = task_infos_addr + (unsigned int)task_iteration_1 % 4 * 32;
                unsigned int e24_1 = ((unsigned int)task_iteration_1 / 4 & 127 | 128) << 24;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&task_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&task_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task_1[(0) + 3]))
                    : "r"(slot_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&task_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&task_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task_1[(4) + 3]))
                    : "r"(slot_1 + 16));
                unsigned int x0_1 = task_1[0] ^ e24_1;
                unsigned int x4_1 = task_1[4] ^ e24_1;
                int _warp_vote_0_1 = __all_sync(0xFFFFFFFF, (x0_1 | x4_1) < 16777216);
                if (_warp_vote_0_1 == 0) {
                    while (1) {
                        uint32_t word_8;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_8) : "r"(static_cast<uint32_t>(slot_1)));
                        task_1[0] = word_8;
                        uint32_t word_9;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_9) : "r"(static_cast<uint32_t>(slot_1 + 4)));
                        task_1[1] = word_9;
                        uint32_t word_10;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_10) : "r"(static_cast<uint32_t>(slot_1 + 8)));
                        task_1[2] = word_10;
                        uint32_t word_11;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_11) : "r"(static_cast<uint32_t>(slot_1 + 12)));
                        task_1[3] = word_11;
                        uint32_t word_12;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_12) : "r"(static_cast<uint32_t>(slot_1 + 16)));
                        task_1[4] = word_12;
                        uint32_t word_13;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_13) : "r"(static_cast<uint32_t>(slot_1 + 20)));
                        task_1[5] = word_13;
                        uint32_t word_14;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_14) : "r"(static_cast<uint32_t>(slot_1 + 24)));
                        task_1[6] = word_14;
                        uint32_t word_15;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_15) : "r"(static_cast<uint32_t>(slot_1 + 28)));
                        task_1[7] = word_15;
                        x0_1 = task_1[0] ^ e24_1;
                        x4_1 = task_1[4] ^ e24_1;
                        int _warp_vote_1_1 = __all_sync(0xFFFFFFFF, (x0_1 | x4_1) < 16777216);
                        if (_warp_vote_1_1 != 0) {
                            break;
                        }
                    }
                    __syncwarp();
                }
                task_1[0] = x0_1;
                task_1[4] = x4_1;
                if (task_1[0] == 0) {
                    break;
                }
                unsigned int w2t_subs_1 = ((task_1[0] == 2) ? 2 : 1);
                #pragma unroll 1
                for (int _sub_1 = 0; _sub_1 < w2t_subs_1; _sub_1++) {
                    unsigned int n_offset = ((task_1[3] + (unsigned int)_sub_1) * 2 + (unsigned int)cta_rank) * 128;
                    #pragma unroll 2
                    for (int k_1 = 0; k_1 < task_1[7] / 128; k_1++) {
                        mbarrier_wait(empty_addr + (b_stage) * 8, _phase_empty_1);
                        unsigned int sf_k_offset = task_1[1] * (task_1[7] / 128) + (unsigned int)k_1;
                        if (elect_sync()) {
                            if (task_1[0] == 1) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + b_stage * 16384, (&B1), k_1 * 128, n_offset, task_1[1], ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(smem_sfb_addr + b_stage * 512, (&SFB1), n_offset, sf_k_offset, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                            } else {
                                tma_2d_gmem2smem_cta2(smem_b_addr + b_stage * 16384, (&B2), k_1 * 128, task_1[1] * task_1[6] + n_offset, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(smem_sfb_addr + b_stage * 512, (&SFB2), n_offset, sf_k_offset, ((full_addr + (b_stage) * 8) & 0xFEFFFFFF));
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
                        if (b_stage == 11) { b_stage = 0; _phase_empty_1 ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // mma_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            unsigned int task_2[8];
            unsigned int mma_stage = 0;
            int completed_tasks = 0;
            unsigned int accum_cursor = 0;
            unsigned int _phase_tmem_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int task_iteration_2 = 0; task_iteration_2 < 8971; task_iteration_2++) {
                    int _stage_task_full_2 = (int)((unsigned int)task_iteration_2 % 4);
                    int _parity_task_full_2 = (int)((unsigned int)task_iteration_2 / 4 % 2);
                    mbarrier_wait(task_full_addr + (_stage_task_full_2) * 8, _parity_task_full_2);
                    unsigned int slot_2 = task_infos_addr + (unsigned int)task_iteration_2 % 4 * 32;
                    unsigned int e24_2 = ((unsigned int)task_iteration_2 / 4 & 127 | 128) << 24;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&task_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&task_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task_2[(0) + 3]))
                        : "r"(slot_2));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&task_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&task_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task_2[(4) + 3]))
                        : "r"(slot_2 + 16));
                    unsigned int x0_2 = task_2[0] ^ e24_2;
                    unsigned int x4_2 = task_2[4] ^ e24_2;
                    int _warp_vote_0_2 = __all_sync(0xFFFFFFFF, (x0_2 | x4_2) < 16777216);
                    if (_warp_vote_0_2 == 0) {
                        while (1) {
                            uint32_t word_16;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_16) : "r"(static_cast<uint32_t>(slot_2)));
                            task_2[0] = word_16;
                            uint32_t word_17;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_17) : "r"(static_cast<uint32_t>(slot_2 + 4)));
                            task_2[1] = word_17;
                            uint32_t word_18;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_18) : "r"(static_cast<uint32_t>(slot_2 + 8)));
                            task_2[2] = word_18;
                            uint32_t word_19;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_19) : "r"(static_cast<uint32_t>(slot_2 + 12)));
                            task_2[3] = word_19;
                            uint32_t word_20;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_20) : "r"(static_cast<uint32_t>(slot_2 + 16)));
                            task_2[4] = word_20;
                            uint32_t word_21;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_21) : "r"(static_cast<uint32_t>(slot_2 + 20)));
                            task_2[5] = word_21;
                            uint32_t word_22;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_22) : "r"(static_cast<uint32_t>(slot_2 + 24)));
                            task_2[6] = word_22;
                            uint32_t word_23;
                            asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_23) : "r"(static_cast<uint32_t>(slot_2 + 28)));
                            task_2[7] = word_23;
                            x0_2 = task_2[0] ^ e24_2;
                            x4_2 = task_2[4] ^ e24_2;
                            int _warp_vote_1_2 = __all_sync(0xFFFFFFFF, (x0_2 | x4_2) < 16777216);
                            if (_warp_vote_1_2 != 0) {
                                break;
                            }
                        }
                        __syncwarp();
                    }
                    task_2[0] = x0_2;
                    task_2[4] = x4_2;
                    if (task_2[0] == 0) {
                        break;
                    }
                    unsigned int w2t_subs_2 = ((task_2[0] == 2) ? 2 : 1);
                    #pragma unroll 1
                    for (int _sub_2 = 0; _sub_2 < w2t_subs_2; _sub_2++) {
                        unsigned int accum_stage = (unsigned int)completed_tasks % 2;
                        unsigned int aligned_m_1 = (task_2[5] + 15) / 16 * 16;
                        mbarrier_wait(tmem_empty_addr + (accum_cursor) * 8, _phase_tmem_empty);
                        #pragma unroll 2
                        for (int k_2 = 0; k_2 < task_2[7] / 128; k_2++) {
                            mbarrier_wait(full_addr + (mma_stage) * 8, _phase_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((k_2 == 0) ? 1 : 0);
                            uint32_t _elected_2 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_scale_a, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 32)), _elected_2);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_scale_b, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (mma_stage) * 32)), _elected_2);
                            int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024;
                            int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 64;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 16)), a_desc + 0, b_desc + 0,
                                    (((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((0) << 29) | ((0) << 4)), tmem_scale_b, tmem_scale_a, ((init_flag) ? 0 : 1), _elected_2);
                                tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 16)), a_desc + 2, b_desc + 2,
                                    (((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((1) << 29) | ((1) << 4)), tmem_scale_b, tmem_scale_a, 1, _elected_2);
                                tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 16)), a_desc + 4, b_desc + 4,
                                    (((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((2) << 29) | ((2) << 4)), tmem_scale_b, tmem_scale_a, 1, _elected_2);
                                tcgen05_mma_mxf8_bs_cta2_leader((tmem_accum + (accum_stage * 16)), a_desc + 6, b_desc + 6,
                                    (((0x10840280U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(aligned_m_1) >> 3) << 17)) | ((3) << 29) | ((3) << 4)), tmem_scale_b, tmem_scale_a, 1, _elected_2);
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            __syncwarp();
                            elect_commit_cg2_multicast(empty_addr + (mma_stage) * 8, (uint16_t)(3));
                            if ((unsigned int)k_2 == task_2[7] / 128 - 1) {
                                elect_commit_cg2_multicast(tmem_full_addr + (accum_cursor) * 8, (uint16_t)(3));
                            }
                            __syncwarp();
                            mma_stage += 1;
                            if (mma_stage == 11) { mma_stage = 0; _phase_full ^= 1; }
                        }
                        completed_tasks += 1;
                        accum_cursor += 1;
                        if (accum_cursor == 2) { accum_cursor = 0; _phase_tmem_empty ^= 1; }
                    }
                }
                if (completed_tasks > 0) {
                    mbarrier_wait(tmem_empty_addr + (((accum_cursor + 1 < 2) ? accum_cursor + 1 : (unsigned int)0)) * 8, ((accum_cursor + 1 < 2) ? _phase_tmem_empty : _phase_tmem_empty ^ 1));
                }
            }
        }
    }
    // ---- Role: schedule ----
    if (warp == 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // schedule_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            unsigned int counts[12];
            unsigned int offsets[12];
            unsigned int state[5];
            unsigned int task_3[8];
            unsigned int schedule_stage = 0;
            unsigned int pub_count = 0;
            unsigned int sched_tag = (unsigned int)1 << 31;
            unsigned int sched_ticket = 0;
            unsigned int _phase_task_empty = 1;
            if (cta_rank == 0) {
                unsigned int expert_11_0 = lane;
                counts[0] = 0;
                if (expert_11_0 < 384) {
                    int fold_c_1_0 = smem_hist[expert_11_0];
                    counts[0] = (unsigned int)fold_c_1_0;
                }
                unsigned int expert_11_1 = 32 + lane;
                counts[1] = 0;
                if (expert_11_1 < 384) {
                    int fold_c_1_1 = smem_hist[expert_11_1];
                    counts[1] = (unsigned int)fold_c_1_1;
                }
                unsigned int expert_11_2 = 64 + lane;
                counts[2] = 0;
                if (expert_11_2 < 384) {
                    int fold_c_1_2 = smem_hist[expert_11_2];
                    counts[2] = (unsigned int)fold_c_1_2;
                }
                unsigned int expert_11_3 = 96 + lane;
                counts[3] = 0;
                if (expert_11_3 < 384) {
                    int fold_c_1_3 = smem_hist[expert_11_3];
                    counts[3] = (unsigned int)fold_c_1_3;
                }
                unsigned int expert_11_4 = 128 + lane;
                counts[4] = 0;
                if (expert_11_4 < 384) {
                    int fold_c_1_4 = smem_hist[expert_11_4];
                    counts[4] = (unsigned int)fold_c_1_4;
                }
                unsigned int expert_11_5 = 160 + lane;
                counts[5] = 0;
                if (expert_11_5 < 384) {
                    int fold_c_1_5 = smem_hist[expert_11_5];
                    counts[5] = (unsigned int)fold_c_1_5;
                }
                unsigned int expert_11_6 = 192 + lane;
                counts[6] = 0;
                if (expert_11_6 < 384) {
                    int fold_c_1_6 = smem_hist[expert_11_6];
                    counts[6] = (unsigned int)fold_c_1_6;
                }
                unsigned int expert_11_7 = 224 + lane;
                counts[7] = 0;
                if (expert_11_7 < 384) {
                    int fold_c_1_7 = smem_hist[expert_11_7];
                    counts[7] = (unsigned int)fold_c_1_7;
                }
                unsigned int expert_11_8 = 256 + lane;
                counts[8] = 0;
                if (expert_11_8 < 384) {
                    int fold_c_1_8 = smem_hist[expert_11_8];
                    counts[8] = (unsigned int)fold_c_1_8;
                }
                unsigned int expert_11_9 = 288 + lane;
                counts[9] = 0;
                if (expert_11_9 < 384) {
                    int fold_c_1_9 = smem_hist[expert_11_9];
                    counts[9] = (unsigned int)fold_c_1_9;
                }
                unsigned int expert_11_10 = 320 + lane;
                counts[10] = 0;
                if (expert_11_10 < 384) {
                    int fold_c_1_10 = smem_hist[expert_11_10];
                    counts[10] = (unsigned int)fold_c_1_10;
                }
                unsigned int expert_11_11 = 352 + lane;
                counts[11] = 0;
                if (expert_11_11 < 384) {
                    int fold_c_1_11 = smem_hist[expert_11_11];
                    counts[11] = (unsigned int)fold_c_1_11;
                }
                __syncwarp();
                unsigned int num_blocks = 0;
                if (lane < 384) {
                    num_blocks += (counts[0] + 16 - 1) / 16;
                }
                if (32 + lane < 384) {
                    num_blocks += (counts[1] + 16 - 1) / 16;
                }
                if (64 + lane < 384) {
                    num_blocks += (counts[2] + 16 - 1) / 16;
                }
                if (96 + lane < 384) {
                    num_blocks += (counts[3] + 16 - 1) / 16;
                }
                if (128 + lane < 384) {
                    num_blocks += (counts[4] + 16 - 1) / 16;
                }
                if (160 + lane < 384) {
                    num_blocks += (counts[5] + 16 - 1) / 16;
                }
                if (192 + lane < 384) {
                    num_blocks += (counts[6] + 16 - 1) / 16;
                }
                if (224 + lane < 384) {
                    num_blocks += (counts[7] + 16 - 1) / 16;
                }
                if (256 + lane < 384) {
                    num_blocks += (counts[8] + 16 - 1) / 16;
                }
                if (288 + lane < 384) {
                    num_blocks += (counts[9] + 16 - 1) / 16;
                }
                if (320 + lane < 384) {
                    num_blocks += (counts[10] + 16 - 1) / 16;
                }
                if (352 + lane < 384) {
                    num_blocks += (counts[11] + 16 - 1) / 16;
                }
                unsigned int total;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(total) : "r"(num_blocks));
                unsigned int waves = (total * 3 + (unsigned int)NUM_CTAS / 2 - 1) / ((unsigned int)NUM_CTAS / 2);
                unsigned int interleave_waves = 2;
                state[0] = total;
                unsigned int _max_1 = ((1) > (interleave_waves) ? (1) : (interleave_waves));
                unsigned int _min_0 = ((_max_1) < (waves) ? (_max_1) : (waves));
                state[1] = _min_0;
                state[2] = 0;
                state[3] = 0;
                state[4] = 0;
                unsigned int block_offset = 0;
                unsigned int tokens_1_0 = counts[0];
                unsigned int blocks_1_0 = (tokens_1_0 + 16 - 1) / 16;
                uint32_t inclusive_1_0 = blocks_1_0;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_0, 1, 32); if (lane >= 1) { inclusive_1_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_0, 2, 32); if (lane >= 2) { inclusive_1_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_0, 4, 32); if (lane >= 4) { inclusive_1_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_0, 8, 32); if (lane >= 8) { inclusive_1_0 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_0, 16, 32); if (lane >= 16) { inclusive_1_0 += peer; } }
                offsets[0] = block_offset + inclusive_1_0 - blocks_1_0;
                unsigned int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, inclusive_1_0, 31);
                block_offset += _shfl_sync_2;
                unsigned int tokens_1_1 = counts[1];
                unsigned int blocks_1_1 = (tokens_1_1 + 16 - 1) / 16;
                uint32_t inclusive_1_1 = blocks_1_1;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_1, 1, 32); if (lane >= 1) { inclusive_1_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_1, 2, 32); if (lane >= 2) { inclusive_1_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_1, 4, 32); if (lane >= 4) { inclusive_1_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_1, 8, 32); if (lane >= 8) { inclusive_1_1 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_1, 16, 32); if (lane >= 16) { inclusive_1_1 += peer; } }
                offsets[1] = block_offset + inclusive_1_1 - blocks_1_1;
                unsigned int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, inclusive_1_1, 31);
                block_offset += _shfl_sync_3;
                unsigned int tokens_1_2 = counts[2];
                unsigned int blocks_1_2 = (tokens_1_2 + 16 - 1) / 16;
                uint32_t inclusive_1_2 = blocks_1_2;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_2, 1, 32); if (lane >= 1) { inclusive_1_2 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_2, 2, 32); if (lane >= 2) { inclusive_1_2 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_2, 4, 32); if (lane >= 4) { inclusive_1_2 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_2, 8, 32); if (lane >= 8) { inclusive_1_2 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_2, 16, 32); if (lane >= 16) { inclusive_1_2 += peer; } }
                offsets[2] = block_offset + inclusive_1_2 - blocks_1_2;
                unsigned int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, inclusive_1_2, 31);
                block_offset += _shfl_sync_4;
                unsigned int tokens_1_3 = counts[3];
                unsigned int blocks_1_3 = (tokens_1_3 + 16 - 1) / 16;
                uint32_t inclusive_1_3 = blocks_1_3;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_3, 1, 32); if (lane >= 1) { inclusive_1_3 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_3, 2, 32); if (lane >= 2) { inclusive_1_3 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_3, 4, 32); if (lane >= 4) { inclusive_1_3 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_3, 8, 32); if (lane >= 8) { inclusive_1_3 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_3, 16, 32); if (lane >= 16) { inclusive_1_3 += peer; } }
                offsets[3] = block_offset + inclusive_1_3 - blocks_1_3;
                unsigned int _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, inclusive_1_3, 31);
                block_offset += _shfl_sync_5;
                unsigned int tokens_1_4 = counts[4];
                unsigned int blocks_1_4 = (tokens_1_4 + 16 - 1) / 16;
                uint32_t inclusive_1_4 = blocks_1_4;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_4, 1, 32); if (lane >= 1) { inclusive_1_4 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_4, 2, 32); if (lane >= 2) { inclusive_1_4 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_4, 4, 32); if (lane >= 4) { inclusive_1_4 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_4, 8, 32); if (lane >= 8) { inclusive_1_4 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_4, 16, 32); if (lane >= 16) { inclusive_1_4 += peer; } }
                offsets[4] = block_offset + inclusive_1_4 - blocks_1_4;
                unsigned int _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, inclusive_1_4, 31);
                block_offset += _shfl_sync_6;
                unsigned int tokens_1_5 = counts[5];
                unsigned int blocks_1_5 = (tokens_1_5 + 16 - 1) / 16;
                uint32_t inclusive_1_5 = blocks_1_5;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_5, 1, 32); if (lane >= 1) { inclusive_1_5 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_5, 2, 32); if (lane >= 2) { inclusive_1_5 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_5, 4, 32); if (lane >= 4) { inclusive_1_5 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_5, 8, 32); if (lane >= 8) { inclusive_1_5 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_5, 16, 32); if (lane >= 16) { inclusive_1_5 += peer; } }
                offsets[5] = block_offset + inclusive_1_5 - blocks_1_5;
                unsigned int _shfl_sync_7 = __shfl_sync(0xFFFFFFFF, inclusive_1_5, 31);
                block_offset += _shfl_sync_7;
                unsigned int tokens_1_6 = counts[6];
                unsigned int blocks_1_6 = (tokens_1_6 + 16 - 1) / 16;
                uint32_t inclusive_1_6 = blocks_1_6;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_6, 1, 32); if (lane >= 1) { inclusive_1_6 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_6, 2, 32); if (lane >= 2) { inclusive_1_6 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_6, 4, 32); if (lane >= 4) { inclusive_1_6 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_6, 8, 32); if (lane >= 8) { inclusive_1_6 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_6, 16, 32); if (lane >= 16) { inclusive_1_6 += peer; } }
                offsets[6] = block_offset + inclusive_1_6 - blocks_1_6;
                unsigned int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, inclusive_1_6, 31);
                block_offset += _shfl_sync_8;
                unsigned int tokens_1_7 = counts[7];
                unsigned int blocks_1_7 = (tokens_1_7 + 16 - 1) / 16;
                uint32_t inclusive_1_7 = blocks_1_7;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_7, 1, 32); if (lane >= 1) { inclusive_1_7 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_7, 2, 32); if (lane >= 2) { inclusive_1_7 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_7, 4, 32); if (lane >= 4) { inclusive_1_7 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_7, 8, 32); if (lane >= 8) { inclusive_1_7 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_7, 16, 32); if (lane >= 16) { inclusive_1_7 += peer; } }
                offsets[7] = block_offset + inclusive_1_7 - blocks_1_7;
                unsigned int _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, inclusive_1_7, 31);
                block_offset += _shfl_sync_9;
                unsigned int tokens_1_8 = counts[8];
                unsigned int blocks_1_8 = (tokens_1_8 + 16 - 1) / 16;
                uint32_t inclusive_1_8 = blocks_1_8;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_8, 1, 32); if (lane >= 1) { inclusive_1_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_8, 2, 32); if (lane >= 2) { inclusive_1_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_8, 4, 32); if (lane >= 4) { inclusive_1_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_8, 8, 32); if (lane >= 8) { inclusive_1_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_8, 16, 32); if (lane >= 16) { inclusive_1_8 += peer; } }
                offsets[8] = block_offset + inclusive_1_8 - blocks_1_8;
                unsigned int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, inclusive_1_8, 31);
                block_offset += _shfl_sync_10;
                unsigned int tokens_1_9 = counts[9];
                unsigned int blocks_1_9 = (tokens_1_9 + 16 - 1) / 16;
                uint32_t inclusive_1_9 = blocks_1_9;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_9, 1, 32); if (lane >= 1) { inclusive_1_9 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_9, 2, 32); if (lane >= 2) { inclusive_1_9 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_9, 4, 32); if (lane >= 4) { inclusive_1_9 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_9, 8, 32); if (lane >= 8) { inclusive_1_9 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_9, 16, 32); if (lane >= 16) { inclusive_1_9 += peer; } }
                offsets[9] = block_offset + inclusive_1_9 - blocks_1_9;
                unsigned int _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, inclusive_1_9, 31);
                block_offset += _shfl_sync_11;
                unsigned int tokens_1_10 = counts[10];
                unsigned int blocks_1_10 = (tokens_1_10 + 16 - 1) / 16;
                uint32_t inclusive_1_10 = blocks_1_10;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_10, 1, 32); if (lane >= 1) { inclusive_1_10 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_10, 2, 32); if (lane >= 2) { inclusive_1_10 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_10, 4, 32); if (lane >= 4) { inclusive_1_10 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_10, 8, 32); if (lane >= 8) { inclusive_1_10 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_10, 16, 32); if (lane >= 16) { inclusive_1_10 += peer; } }
                offsets[10] = block_offset + inclusive_1_10 - blocks_1_10;
                unsigned int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, inclusive_1_10, 31);
                block_offset += _shfl_sync_12;
                unsigned int tokens_1_11 = counts[11];
                unsigned int blocks_1_11 = (tokens_1_11 + 16 - 1) / 16;
                uint32_t inclusive_1_11 = blocks_1_11;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_11, 1, 32); if (lane >= 1) { inclusive_1_11 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_11, 2, 32); if (lane >= 2) { inclusive_1_11 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_11, 4, 32); if (lane >= 4) { inclusive_1_11 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_11, 8, 32); if (lane >= 8) { inclusive_1_11 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, inclusive_1_11, 16, 32); if (lane >= 16) { inclusive_1_11 += peer; } }
                offsets[11] = block_offset + inclusive_1_11 - blocks_1_11;
                unsigned int _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, inclusive_1_11, 31);
                block_offset += _shfl_sync_13;
                #pragma unroll 1
                for (int iteration = 0; iteration < 8971; iteration++) {
                    mbarrier_wait(task_empty_addr + (schedule_stage) * 8, _phase_task_empty);
                    unsigned int phase = 0;
                    unsigned int claimed = 0;
                    unsigned int clusters = 3;
                    unsigned int shape_n = 768;
                    unsigned int shape_k = 5120;
                    if (state[1] != 4294967295u && state[1] != 0) {
                        state[1] = state[1] - 1;
                        unsigned int claimed_0 = 0;
                        if (lane == 0) {
                            unsigned int _atomic_fetch_add_14 = atomicAdd(PrivateCounters, 1);
                            claimed_0 = _atomic_fetch_add_14;
                        }
                        unsigned int _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, claimed_0, 0);
                        claimed = _shfl_sync_15;
                        if (claimed >= state[0] * 3) {
                            state[1] = 4294967295;
                            state[2] = state[0];
                        } else {
                            phase = 1;
                        }
                    }
                    if (phase == 0) {
                        if (state[3] != 0) {
                            unsigned int _shfl_sync_16 = __shfl_sync(0xFFFFFFFF, state[4], 0);
                            claimed = _shfl_sync_16;
                            state[3] = 0;
                        } else {
                            unsigned int claimed_0_1 = 0;
                            if (lane == 0) {
                                unsigned int _atomic_fetch_add_17 = atomicAdd(PrivateCounters + 1, 1);
                                claimed_0_1 = _atomic_fetch_add_17;
                            }
                            unsigned int _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, claimed_0_1, 0);
                            claimed = _shfl_sync_18;
                        }
                        if (claimed < state[0] * 10) {
                            if (state[1] != 4294967295u) {
                                state[1] = 2;
                            } else {
                                if (lane == 0) {
                                    unsigned int _atomic_fetch_add_19 = atomicAdd(PrivateCounters + 1, 1);
                                    state[4] = _atomic_fetch_add_19;
                                }
                                state[3] = 1;
                            }
                            phase = 2;
                            clusters = 20;
                            shape_n = 5120;
                            shape_k = 384;
                            claimed = claimed * 2;
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
                        unsigned int expert_0 = lane;
                        unsigned int tokens_0 = counts[0];
                        unsigned int blocks_0 = (tokens_0 + 16 - 1) / 16;
                        unsigned int lane_offset_0 = offsets[0];
                        unsigned int owner_mask_0 = __ballot_sync(0xFFFFFFFF, expert_0 < 384 && (pool_block >= lane_offset_0 && pool_block < lane_offset_0 + blocks_0));
                        if (owner_mask_0 != 0) {
                            int _find_first_set_20 = __ffs(owner_mask_0);
                            unsigned int owner_lane = _find_first_set_20 - 1;
                            unsigned int local_block = pool_block - lane_offset_0;
                            unsigned int valid = ((tokens_0 - local_block * 16) < (16) ? (tokens_0 - local_block * 16) : (16));
                            unsigned int _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, expert_0, owner_lane);
                            task_3[1] = _shfl_sync_21;
                            unsigned int _shfl_sync_22 = __shfl_sync(0xFFFFFFFF, local_block, owner_lane);
                            task_3[2] = _shfl_sync_22;
                            unsigned int _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, valid, owner_lane);
                            task_3[5] = _shfl_sync_23;
                        }
                        unsigned int expert_3 = 32 + lane;
                        unsigned int tokens_2 = counts[1];
                        unsigned int blocks_2 = (tokens_2 + 16 - 1) / 16;
                        unsigned int lane_offset_1 = offsets[1];
                        unsigned int owner_mask_1 = __ballot_sync(0xFFFFFFFF, expert_3 < 384 && (pool_block >= lane_offset_1 && pool_block < lane_offset_1 + blocks_2));
                        if (owner_mask_1 != 0) {
                            int _find_first_set_24 = __ffs(owner_mask_1);
                            unsigned int owner_lane_1 = _find_first_set_24 - 1;
                            unsigned int local_block_1 = pool_block - lane_offset_1;
                            unsigned int valid_1 = ((tokens_2 - local_block_1 * 16) < (16) ? (tokens_2 - local_block_1 * 16) : (16));
                            unsigned int _shfl_sync_25 = __shfl_sync(0xFFFFFFFF, expert_3, owner_lane_1);
                            task_3[1] = _shfl_sync_25;
                            unsigned int _shfl_sync_26 = __shfl_sync(0xFFFFFFFF, local_block_1, owner_lane_1);
                            task_3[2] = _shfl_sync_26;
                            unsigned int _shfl_sync_27 = __shfl_sync(0xFFFFFFFF, valid_1, owner_lane_1);
                            task_3[5] = _shfl_sync_27;
                        }
                        unsigned int expert_4 = 64 + lane;
                        unsigned int tokens_3 = counts[2];
                        unsigned int blocks_3 = (tokens_3 + 16 - 1) / 16;
                        unsigned int lane_offset_2 = offsets[2];
                        unsigned int owner_mask_2 = __ballot_sync(0xFFFFFFFF, expert_4 < 384 && (pool_block >= lane_offset_2 && pool_block < lane_offset_2 + blocks_3));
                        if (owner_mask_2 != 0) {
                            int _find_first_set_28 = __ffs(owner_mask_2);
                            unsigned int owner_lane_2 = _find_first_set_28 - 1;
                            unsigned int local_block_2 = pool_block - lane_offset_2;
                            unsigned int valid_2 = ((tokens_3 - local_block_2 * 16) < (16) ? (tokens_3 - local_block_2 * 16) : (16));
                            unsigned int _shfl_sync_29 = __shfl_sync(0xFFFFFFFF, expert_4, owner_lane_2);
                            task_3[1] = _shfl_sync_29;
                            unsigned int _shfl_sync_30 = __shfl_sync(0xFFFFFFFF, local_block_2, owner_lane_2);
                            task_3[2] = _shfl_sync_30;
                            unsigned int _shfl_sync_31 = __shfl_sync(0xFFFFFFFF, valid_2, owner_lane_2);
                            task_3[5] = _shfl_sync_31;
                        }
                        unsigned int expert_5 = 96 + lane;
                        unsigned int tokens_4 = counts[3];
                        unsigned int blocks_4 = (tokens_4 + 16 - 1) / 16;
                        unsigned int lane_offset_3 = offsets[3];
                        unsigned int owner_mask_3 = __ballot_sync(0xFFFFFFFF, expert_5 < 384 && (pool_block >= lane_offset_3 && pool_block < lane_offset_3 + blocks_4));
                        if (owner_mask_3 != 0) {
                            int _find_first_set_32 = __ffs(owner_mask_3);
                            unsigned int owner_lane_3 = _find_first_set_32 - 1;
                            unsigned int local_block_3 = pool_block - lane_offset_3;
                            unsigned int valid_3 = ((tokens_4 - local_block_3 * 16) < (16) ? (tokens_4 - local_block_3 * 16) : (16));
                            unsigned int _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, expert_5, owner_lane_3);
                            task_3[1] = _shfl_sync_33;
                            unsigned int _shfl_sync_34 = __shfl_sync(0xFFFFFFFF, local_block_3, owner_lane_3);
                            task_3[2] = _shfl_sync_34;
                            unsigned int _shfl_sync_35 = __shfl_sync(0xFFFFFFFF, valid_3, owner_lane_3);
                            task_3[5] = _shfl_sync_35;
                        }
                        unsigned int expert_6 = 128 + lane;
                        unsigned int tokens_5 = counts[4];
                        unsigned int blocks_5 = (tokens_5 + 16 - 1) / 16;
                        unsigned int lane_offset_4 = offsets[4];
                        unsigned int owner_mask_4 = __ballot_sync(0xFFFFFFFF, expert_6 < 384 && (pool_block >= lane_offset_4 && pool_block < lane_offset_4 + blocks_5));
                        if (owner_mask_4 != 0) {
                            int _find_first_set_36 = __ffs(owner_mask_4);
                            unsigned int owner_lane_4 = _find_first_set_36 - 1;
                            unsigned int local_block_4 = pool_block - lane_offset_4;
                            unsigned int valid_4 = ((tokens_5 - local_block_4 * 16) < (16) ? (tokens_5 - local_block_4 * 16) : (16));
                            unsigned int _shfl_sync_37 = __shfl_sync(0xFFFFFFFF, expert_6, owner_lane_4);
                            task_3[1] = _shfl_sync_37;
                            unsigned int _shfl_sync_38 = __shfl_sync(0xFFFFFFFF, local_block_4, owner_lane_4);
                            task_3[2] = _shfl_sync_38;
                            unsigned int _shfl_sync_39 = __shfl_sync(0xFFFFFFFF, valid_4, owner_lane_4);
                            task_3[5] = _shfl_sync_39;
                        }
                        unsigned int expert_7 = 160 + lane;
                        unsigned int tokens_6 = counts[5];
                        unsigned int blocks_6 = (tokens_6 + 16 - 1) / 16;
                        unsigned int lane_offset_5 = offsets[5];
                        unsigned int owner_mask_5 = __ballot_sync(0xFFFFFFFF, expert_7 < 384 && (pool_block >= lane_offset_5 && pool_block < lane_offset_5 + blocks_6));
                        if (owner_mask_5 != 0) {
                            int _find_first_set_40 = __ffs(owner_mask_5);
                            unsigned int owner_lane_5 = _find_first_set_40 - 1;
                            unsigned int local_block_5 = pool_block - lane_offset_5;
                            unsigned int valid_5 = ((tokens_6 - local_block_5 * 16) < (16) ? (tokens_6 - local_block_5 * 16) : (16));
                            unsigned int _shfl_sync_41 = __shfl_sync(0xFFFFFFFF, expert_7, owner_lane_5);
                            task_3[1] = _shfl_sync_41;
                            unsigned int _shfl_sync_42 = __shfl_sync(0xFFFFFFFF, local_block_5, owner_lane_5);
                            task_3[2] = _shfl_sync_42;
                            unsigned int _shfl_sync_43 = __shfl_sync(0xFFFFFFFF, valid_5, owner_lane_5);
                            task_3[5] = _shfl_sync_43;
                        }
                        unsigned int expert_8 = 192 + lane;
                        unsigned int tokens_7 = counts[6];
                        unsigned int blocks_7 = (tokens_7 + 16 - 1) / 16;
                        unsigned int lane_offset_6 = offsets[6];
                        unsigned int owner_mask_6 = __ballot_sync(0xFFFFFFFF, expert_8 < 384 && (pool_block >= lane_offset_6 && pool_block < lane_offset_6 + blocks_7));
                        if (owner_mask_6 != 0) {
                            int _find_first_set_44 = __ffs(owner_mask_6);
                            unsigned int owner_lane_6 = _find_first_set_44 - 1;
                            unsigned int local_block_6 = pool_block - lane_offset_6;
                            unsigned int valid_6 = ((tokens_7 - local_block_6 * 16) < (16) ? (tokens_7 - local_block_6 * 16) : (16));
                            unsigned int _shfl_sync_45 = __shfl_sync(0xFFFFFFFF, expert_8, owner_lane_6);
                            task_3[1] = _shfl_sync_45;
                            unsigned int _shfl_sync_46 = __shfl_sync(0xFFFFFFFF, local_block_6, owner_lane_6);
                            task_3[2] = _shfl_sync_46;
                            unsigned int _shfl_sync_47 = __shfl_sync(0xFFFFFFFF, valid_6, owner_lane_6);
                            task_3[5] = _shfl_sync_47;
                        }
                        unsigned int expert_9 = 224 + lane;
                        unsigned int tokens_8 = counts[7];
                        unsigned int blocks_8 = (tokens_8 + 16 - 1) / 16;
                        unsigned int lane_offset_7 = offsets[7];
                        unsigned int owner_mask_7 = __ballot_sync(0xFFFFFFFF, expert_9 < 384 && (pool_block >= lane_offset_7 && pool_block < lane_offset_7 + blocks_8));
                        if (owner_mask_7 != 0) {
                            int _find_first_set_48 = __ffs(owner_mask_7);
                            unsigned int owner_lane_7 = _find_first_set_48 - 1;
                            unsigned int local_block_7 = pool_block - lane_offset_7;
                            unsigned int valid_7 = ((tokens_8 - local_block_7 * 16) < (16) ? (tokens_8 - local_block_7 * 16) : (16));
                            unsigned int _shfl_sync_49 = __shfl_sync(0xFFFFFFFF, expert_9, owner_lane_7);
                            task_3[1] = _shfl_sync_49;
                            unsigned int _shfl_sync_50 = __shfl_sync(0xFFFFFFFF, local_block_7, owner_lane_7);
                            task_3[2] = _shfl_sync_50;
                            unsigned int _shfl_sync_51 = __shfl_sync(0xFFFFFFFF, valid_7, owner_lane_7);
                            task_3[5] = _shfl_sync_51;
                        }
                        unsigned int expert_10 = 256 + lane;
                        unsigned int tokens_9 = counts[8];
                        unsigned int blocks_9 = (tokens_9 + 16 - 1) / 16;
                        unsigned int lane_offset_8 = offsets[8];
                        unsigned int owner_mask_8 = __ballot_sync(0xFFFFFFFF, expert_10 < 384 && (pool_block >= lane_offset_8 && pool_block < lane_offset_8 + blocks_9));
                        if (owner_mask_8 != 0) {
                            int _find_first_set_52 = __ffs(owner_mask_8);
                            unsigned int owner_lane_8 = _find_first_set_52 - 1;
                            unsigned int local_block_8 = pool_block - lane_offset_8;
                            unsigned int valid_8 = ((tokens_9 - local_block_8 * 16) < (16) ? (tokens_9 - local_block_8 * 16) : (16));
                            unsigned int _shfl_sync_53 = __shfl_sync(0xFFFFFFFF, expert_10, owner_lane_8);
                            task_3[1] = _shfl_sync_53;
                            unsigned int _shfl_sync_54 = __shfl_sync(0xFFFFFFFF, local_block_8, owner_lane_8);
                            task_3[2] = _shfl_sync_54;
                            unsigned int _shfl_sync_55 = __shfl_sync(0xFFFFFFFF, valid_8, owner_lane_8);
                            task_3[5] = _shfl_sync_55;
                        }
                        unsigned int expert_12 = 288 + lane;
                        unsigned int tokens_10 = counts[9];
                        unsigned int blocks_10 = (tokens_10 + 16 - 1) / 16;
                        unsigned int lane_offset_9 = offsets[9];
                        unsigned int owner_mask_9 = __ballot_sync(0xFFFFFFFF, expert_12 < 384 && (pool_block >= lane_offset_9 && pool_block < lane_offset_9 + blocks_10));
                        if (owner_mask_9 != 0) {
                            int _find_first_set_56 = __ffs(owner_mask_9);
                            unsigned int owner_lane_9 = _find_first_set_56 - 1;
                            unsigned int local_block_9 = pool_block - lane_offset_9;
                            unsigned int valid_9 = ((tokens_10 - local_block_9 * 16) < (16) ? (tokens_10 - local_block_9 * 16) : (16));
                            unsigned int _shfl_sync_57 = __shfl_sync(0xFFFFFFFF, expert_12, owner_lane_9);
                            task_3[1] = _shfl_sync_57;
                            unsigned int _shfl_sync_58 = __shfl_sync(0xFFFFFFFF, local_block_9, owner_lane_9);
                            task_3[2] = _shfl_sync_58;
                            unsigned int _shfl_sync_59 = __shfl_sync(0xFFFFFFFF, valid_9, owner_lane_9);
                            task_3[5] = _shfl_sync_59;
                        }
                        unsigned int expert_13 = 320 + lane;
                        unsigned int tokens_11 = counts[10];
                        unsigned int blocks_11 = (tokens_11 + 16 - 1) / 16;
                        unsigned int lane_offset_10 = offsets[10];
                        unsigned int owner_mask_10 = __ballot_sync(0xFFFFFFFF, expert_13 < 384 && (pool_block >= lane_offset_10 && pool_block < lane_offset_10 + blocks_11));
                        if (owner_mask_10 != 0) {
                            int _find_first_set_60 = __ffs(owner_mask_10);
                            unsigned int owner_lane_10 = _find_first_set_60 - 1;
                            unsigned int local_block_10 = pool_block - lane_offset_10;
                            unsigned int valid_10 = ((tokens_11 - local_block_10 * 16) < (16) ? (tokens_11 - local_block_10 * 16) : (16));
                            unsigned int _shfl_sync_61 = __shfl_sync(0xFFFFFFFF, expert_13, owner_lane_10);
                            task_3[1] = _shfl_sync_61;
                            unsigned int _shfl_sync_62 = __shfl_sync(0xFFFFFFFF, local_block_10, owner_lane_10);
                            task_3[2] = _shfl_sync_62;
                            unsigned int _shfl_sync_63 = __shfl_sync(0xFFFFFFFF, valid_10, owner_lane_10);
                            task_3[5] = _shfl_sync_63;
                        }
                        unsigned int expert_14 = 352 + lane;
                        unsigned int tokens_12 = counts[11];
                        unsigned int blocks_12 = (tokens_12 + 16 - 1) / 16;
                        unsigned int lane_offset_11 = offsets[11];
                        unsigned int owner_mask_11 = __ballot_sync(0xFFFFFFFF, expert_14 < 384 && (pool_block >= lane_offset_11 && pool_block < lane_offset_11 + blocks_12));
                        if (owner_mask_11 != 0) {
                            int _find_first_set_64 = __ffs(owner_mask_11);
                            unsigned int owner_lane_11 = _find_first_set_64 - 1;
                            unsigned int local_block_11 = pool_block - lane_offset_11;
                            unsigned int valid_11 = ((tokens_12 - local_block_11 * 16) < (16) ? (tokens_12 - local_block_11 * 16) : (16));
                            unsigned int _shfl_sync_65 = __shfl_sync(0xFFFFFFFF, expert_14, owner_lane_11);
                            task_3[1] = _shfl_sync_65;
                            unsigned int _shfl_sync_66 = __shfl_sync(0xFFFFFFFF, local_block_11, owner_lane_11);
                            task_3[2] = _shfl_sync_66;
                            unsigned int _shfl_sync_67 = __shfl_sync(0xFFFFFFFF, valid_11, owner_lane_11);
                            task_3[5] = _shfl_sync_67;
                        }
                        if (phase == 2) {
                            if (task_3[4] >= state[2]) {
                                unsigned int required = (task_3[4] + 1) * 3;
                                {
                                unsigned int _acquire_observed;
                                do {
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"(reinterpret_cast<unsigned int*>(PrivateCounters)) : "memory");
                                } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(required)) >= static_cast<unsigned int>((unsigned int)0 - required));
                                }
                                state[2] = task_3[4] + 1;
                            }
                        }
                    }
                    if (task_3[0] == 0) {
                        break;
                    }
                    if (lane < 2) {
                        unsigned int e24_3 = (pub_count / 4 & 127 | 128) << 24;
                        uint32_t destination;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(destination) : "r"(task_infos_addr + schedule_stage * 32), "r"(lane));
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(task_full_addr + (schedule_stage) * 8), "r"(lane), "r"((uint32_t)(32)) : "memory");
                        uint32_t _mbar_peer_68;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_68) : "r"(task_full_addr + (schedule_stage) * 8), "r"(lane));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(destination), "r"(task_3[0] | e24_3), "r"(task_3[1]), "r"(task_3[2]), "r"(task_3[3]), "r"(_mbar_peer_68) : "memory");
                        uint32_t _mbar_peer_69;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_69) : "r"(task_full_addr + (schedule_stage) * 8), "r"(lane));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(destination + 16), "r"(task_3[4] | e24_3), "r"(task_3[5]), "r"(task_3[6]), "r"(task_3[7]), "r"(_mbar_peer_69) : "memory");
                    }
                    __syncwarp();
                    schedule_stage += 1;
                    if (schedule_stage == 4) { schedule_stage = 0; _phase_task_empty ^= 1; }
                    pub_count += 1;
                }
                mbarrier_wait(task_empty_addr + (schedule_stage) * 8, _phase_task_empty);
                task_3[0] = 0;
                task_3[1] = 0;
                task_3[2] = 0;
                task_3[3] = 0;
                task_3[4] = 0;
                task_3[5] = 0;
                task_3[6] = 0;
                task_3[7] = 0;
                if (lane < 2) {
                    unsigned int e24_4 = (pub_count / 4 & 127 | 128) << 24;
                    uint32_t destination_1;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(destination_1) : "r"(task_infos_addr + schedule_stage * 32), "r"(lane));
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                        "}"
                        :: "r"(task_full_addr + (schedule_stage) * 8), "r"(lane), "r"((uint32_t)(32)) : "memory");
                    uint32_t _mbar_peer_70;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_70) : "r"(task_full_addr + (schedule_stage) * 8), "r"(lane));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(destination_1), "r"(task_3[0] | e24_4), "r"(task_3[1]), "r"(task_3[2]), "r"(task_3[3]), "r"(_mbar_peer_70) : "memory");
                    uint32_t _mbar_peer_71;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_71) : "r"(task_full_addr + (schedule_stage) * 8), "r"(lane));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(destination_1 + 16), "r"(task_3[4] | e24_4), "r"(task_3[5]), "r"(task_3[6]), "r"(task_3[7]), "r"(_mbar_peer_71) : "memory");
                }
                __syncwarp();
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("barrier.sync 12, 512;" ::: "memory");
            unsigned int task_4[8];
            int epi_uses = 0;
            #pragma unroll 1
            for (int task_iteration_3 = 0; task_iteration_3 < 8971; task_iteration_3++) {
                int _stage_task_full_3 = (int)((unsigned int)task_iteration_3 % 4);
                int _parity_task_full_3 = (int)((unsigned int)task_iteration_3 / 4 % 2);
                mbarrier_wait(task_full_addr + (_stage_task_full_3) * 8, _parity_task_full_3);
                unsigned int slot_3 = task_infos_addr + (unsigned int)task_iteration_3 % 4 * 32;
                unsigned int e24_5 = ((unsigned int)task_iteration_3 / 4 & 127 | 128) << 24;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&task_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&task_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task_4[(0) + 3]))
                    : "r"(slot_3));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&task_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&task_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&task_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&task_4[(4) + 3]))
                    : "r"(slot_3 + 16));
                unsigned int x0_3 = task_4[0] ^ e24_5;
                unsigned int x4_3 = task_4[4] ^ e24_5;
                int _warp_vote_0_3 = __all_sync(0xFFFFFFFF, (x0_3 | x4_3) < 16777216);
                if (_warp_vote_0_3 == 0) {
                    while (1) {
                        uint32_t word_24;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_24) : "r"(static_cast<uint32_t>(slot_3)));
                        task_4[0] = word_24;
                        uint32_t word_25;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_25) : "r"(static_cast<uint32_t>(slot_3 + 4)));
                        task_4[1] = word_25;
                        uint32_t word_26;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_26) : "r"(static_cast<uint32_t>(slot_3 + 8)));
                        task_4[2] = word_26;
                        uint32_t word_27;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_27) : "r"(static_cast<uint32_t>(slot_3 + 12)));
                        task_4[3] = word_27;
                        uint32_t word_28;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_28) : "r"(static_cast<uint32_t>(slot_3 + 16)));
                        task_4[4] = word_28;
                        uint32_t word_29;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_29) : "r"(static_cast<uint32_t>(slot_3 + 20)));
                        task_4[5] = word_29;
                        uint32_t word_30;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_30) : "r"(static_cast<uint32_t>(slot_3 + 24)));
                        task_4[6] = word_30;
                        uint32_t word_31;
                        asm volatile("ld.relaxed.cluster.shared::cta.b32 %0, [%1];" : "=r"(word_31) : "r"(static_cast<uint32_t>(slot_3 + 28)));
                        task_4[7] = word_31;
                        x0_3 = task_4[0] ^ e24_5;
                        x4_3 = task_4[4] ^ e24_5;
                        int _warp_vote_1_3 = __all_sync(0xFFFFFFFF, (x0_3 | x4_3) < 16777216);
                        if (_warp_vote_1_3 != 0) {
                            break;
                        }
                    }
                    __syncwarp();
                }
                task_4[0] = x0_3;
                task_4[4] = x4_3;
                if (task_4[0] == 0) {
                    break;
                }
                unsigned int w2t_subs_3 = ((task_4[0] == 2) ? 2 : 1);
                #pragma unroll 1
                for (int _sub_3 = 0; _sub_3 < w2t_subs_3; _sub_3++) {
                    unsigned int accum_stage_1 = (unsigned int)epi_uses % 2;
                    int _parity_tmem_full = (int)((unsigned int)epi_uses / 2 % 2);
                    mbarrier_wait(tmem_full_addr + (accum_stage_1) * 8, _parity_tmem_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (_sub_3 == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((task_empty_addr + ((int)((unsigned int)task_iteration_3 % 4)) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    if (task_4[0] == 1) {
                        unsigned int epi_warp = warp - 8;
                        unsigned int epi_wg = epi_warp / 4;
                        unsigned int warp_in_wg = epi_warp % 4;
                        unsigned int valid_m = __shfl_sync(0xFFFFFFFF, task_4[5], 0);
                        unsigned int pool_block_1 = task_4[4];
                        unsigned int ring_block = pool_block_1 % 390;
                        unsigned int block_1 = ((unsigned int)smem_ero[task_4[1]] + task_4[2] * 16) / 16;
                        unsigned int sf_stride = 199680;
                        if (task_4[0] == 3) {
                            block_1 = pool_block_1;
                            sf_stride = 0;
                        }
                        unsigned int n_block = task_4[3] * 2 + (unsigned int)cta_rank;
                        float cached_weight = 1.0f;
                        float activation[4];
                        float amax[2];
                        float quantized[4];
                        #pragma unroll
                        for (int s = 0; s < 1; s++) {
                            if (valid_m <= epi_wg * 8 + (unsigned int)(s * 8)) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                                break;
                            }
                            #pragma unroll
                            for (int i_3 = 0; i_3 < 1; i_3++) {
                                unsigned int j = s + i_3;
                                if (task_4[0] == 1 && j * 8 % 32 == 0) {
                                    if (j * 8 + (unsigned int)lane < 8) {
                                        cached_weight = routing_weight_pool[block_1 * 16 + epi_wg * 8 + j * 8 + (unsigned int)lane];
                                    }
                                }
                                float first_weight = __shfl_sync(0xFFFFFFFF, cached_weight, j * 8 % 32 + (unsigned int)(lane % 4 * 2));
                                float second_weight = __shfl_sync(0xFFFFFFFF, cached_weight, j * 8 % 32 + (unsigned int)(lane % 4 * 2) + 1);
                                unsigned int address = accum_stage_1 * 16 + epi_wg * 8 + j * 8;
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
                                if (j == 0) {
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                                }
                                __nv_bfloat162 bf16x2 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                                __nv_bfloat162 bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                                __nv_bfloat162 bf16x2_2 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                                __nv_bfloat162 bf16x2_3 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                                __nv_bfloat162 min_1 = __hmin2(bf16x2, bf16x2_2);
                                __nv_bfloat162 min_2 = __hmin2(__hmax2(bf16x2_1, bf16x2_3), bf16x2_2);
                                float2 cvt_f32 = cake_bf16x2_to_f32x2(min_1);
                                float2 cvt_f32_1 = cake_bf16x2_to_f32x2(min_2);
                                float denominator[2];
                                float2 activated_gate;
                                float _exp_2 = __expf(-cvt_f32.x);
                                denominator[0] = _exp_2;
                                float _exp_3 = __expf(-cvt_f32.y);
                                denominator[1] = _exp_3;
                                const float2 _add2_4 = {1.0f, 1.0f};
                                #pragma unroll
                                for (int _la = 0; _la < 1; _la++) {
                                    reinterpret_cast<float2*>(denominator)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(denominator)[_la], _add2_4);
                                }
                                float _rcp_5 = approx_rcp(denominator[0]);
                                float _rcp_6 = approx_rcp(denominator[1]);
                                activated_gate = mul_f32x2_noftz(cvt_f32, make_float2(_rcp_5, _rcp_6));
                                float2 gated_up;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gated_up) : "l"(*(const unsigned long long*)&activated_gate), "l"(*(const unsigned long long*)&cvt_f32_1));
                                float2 _f32x2_mul_b_7 = make_float2(first_weight, second_weight);
                                float2 activated;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&activated) : "l"(*(const unsigned long long*)&gated_up), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
                                activation[i_3 * 4] = activated.x;
                                activation[i_3 * 4 + 1] = activated.y;
                                __nv_bfloat162 bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                                __nv_bfloat162 bf16x2_1_0 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                                __nv_bfloat162 bf16x2_2_0 = __float22bfloat162_rn(make_float2(10.0f, 10.0f));
                                __nv_bfloat162 bf16x2_3_0 = __float22bfloat162_rn(make_float2(-10.0f, -10.0f));
                                __nv_bfloat162 min_1_0 = __hmin2(bf16x2_0, bf16x2_2_0);
                                __nv_bfloat162 min_2_0 = __hmin2(__hmax2(bf16x2_1_0, bf16x2_3_0), bf16x2_2_0);
                                float2 cvt_f32_0 = cake_bf16x2_to_f32x2(min_1_0);
                                float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(min_2_0);
                                float denominator_0[2];
                                float2 activated_gate_0;
                                float _exp_8 = __expf(-cvt_f32_0.x);
                                denominator_0[0] = _exp_8;
                                float _exp_9 = __expf(-cvt_f32_0.y);
                                denominator_0[1] = _exp_9;
                                const float2 _add2_10 = {1.0f, 1.0f};
                                #pragma unroll
                                for (int _la = 0; _la < 1; _la++) {
                                    reinterpret_cast<float2*>(denominator_0)[_la] = add_f32x2_rn_noftz(reinterpret_cast<float2*>(denominator_0)[_la], _add2_10);
                                }
                                float _rcp_11 = approx_rcp(denominator_0[0]);
                                float _rcp_12 = approx_rcp(denominator_0[1]);
                                activated_gate_0 = mul_f32x2_noftz(cvt_f32_0, make_float2(_rcp_11, _rcp_12));
                                float2 gated_up_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gated_up_0) : "l"(*(const unsigned long long*)&activated_gate_0), "l"(*(const unsigned long long*)&cvt_f32_1_0));
                                float2 _f32x2_mul_b_13 = make_float2(first_weight, second_weight);
                                float2 activated_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&activated_0) : "l"(*(const unsigned long long*)&gated_up_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
                                activation[i_3 * 4 + 2] = activated_0.x;
                                activation[i_3 * 4 + 3] = activated_0.y;
                                float first_max;
                                asm("max.f32 %0, %1, %2;" : "=f"(first_max) : "f"(fabsf(activation[i_3 * 4])), "f"(fabsf(activation[i_3 * 4 + 2])));
                                float second_max;
                                asm("max.f32 %0, %1, %2;" : "=f"(second_max) : "f"(fabsf(activation[i_3 * 4 + 1])), "f"(fabsf(activation[i_3 * 4 + 3])));
                                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, first_max, 4);
                                first_max = max_noftz(first_max, _shfl_xor_sync_14);
                                float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, second_max, 4);
                                second_max = max_noftz(second_max, _shfl_xor_sync_15);
                                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, first_max, 8);
                                first_max = max_noftz(first_max, _shfl_xor_sync_16);
                                float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, second_max, 8);
                                second_max = max_noftz(second_max, _shfl_xor_sync_17);
                                float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, first_max, 16);
                                first_max = max_noftz(first_max, _shfl_xor_sync_18);
                                float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, second_max, 16);
                                second_max = max_noftz(second_max, _shfl_xor_sync_19);
                                amax[i_3 * 2] = first_max;
                                amax[i_3 * 2 + 1] = second_max;
                                if (lane < 4) {
                                    amax_reduction[epi_warp * 8 + (unsigned int)(i_3 * 8) + (unsigned int)(lane * 2)] = first_max;
                                    amax_reduction[epi_warp * 8 + (unsigned int)(i_3 * 8) + (unsigned int)(lane * 2) + 1] = second_max;
                                }
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                __syncwarp();
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            unsigned int tma_stage = s % 2;
                            unsigned int store_address = smem_output_addr + (epi_wg * 2 + tma_stage) * 8 * 64;
                            asm volatile("cp.async.bulk.wait_group 1;");
                            asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg) : "memory");
                            #pragma unroll
                            for (int i_4 = 0; i_4 < 1; i_4++) {
                                unsigned int paired_index = (epi_warp ^ 1) * 8 + (unsigned int)(i_4 * 8) + (unsigned int)(lane % 4 * 2);
                                float first_max_1;
                                asm("max.f32 %0, %1, %2;" : "=f"(first_max_1) : "f"(amax[i_4 * 2]), "f"(amax_reduction[paired_index]));
                                float second_max_1;
                                asm("max.f32 %0, %1, %2;" : "=f"(second_max_1) : "f"(amax[i_4 * 2 + 1]), "f"(amax_reduction[paired_index + 1]));
                                unsigned int first_bits = __as_u32(first_max_1);
                                unsigned int second_bits = __as_u32(second_max_1);
                                unsigned int _max_20 = ((first_bits + 2097151 >> 23) > (113) ? (first_bits + 2097151 >> 23) : (113));
                                unsigned int first_exp = _max_20 - 8;
                                unsigned int _max_21 = ((second_bits + 2097151 >> 23) > (113) ? (second_bits + 2097151 >> 23) : (113));
                                unsigned int second_exp = _max_21 - 8;
                                unsigned int first_inverse_bits = 254 - first_exp << 23;
                                unsigned int second_inverse_bits = 254 - second_exp << 23;
                                float first_inverse = __uint_as_float(first_inverse_bits);
                                float second_inverse = __uint_as_float(second_inverse_bits);
                                quantized[0] = activation[i_4 * 4] * first_inverse;
                                quantized[1] = activation[i_4 * 4 + 1] * second_inverse;
                                quantized[2] = activation[i_4 * 4 + 2] * first_inverse;
                                quantized[3] = activation[i_4 * 4 + 3] * second_inverse;
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
                                unsigned int stsm_address = store_address + (unsigned int)(i_4 * 8 * 64) + (unsigned int)(lane * 64) + (warp_in_wg ^ (unsigned int)(lane / 2)) * 16;
                                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                                uint32_t _stmatrix_b8_x1_addr_22 = static_cast<uint32_t>(stsm_address);
                                asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                                    :: "r"(_stmatrix_b8_x1_addr_22), "r"(_fp8_0[0]) : "memory");
                                #elif defined(__CUDA_ARCH__)
                                #error "StmatrixTransB8X1 requires SM100 or newer"
                                #endif
                                if (warp_in_wg % 2 == 0 && lane < 4) {
                                    unsigned int sf_k = n_block * 2 + warp_in_wg / 2;
                                    unsigned int token_base = epi_wg * 8 + (unsigned int)(s * 8) + (unsigned int)(i_4 * 8);
                                    unsigned int sf_address = (block_1 * 16 + token_base + (unsigned int)(lane * 2)) * 12 + sf_k;
                                    SF_I_w[sf_address] = (uint8_t)first_exp;
                                    SF_I_w[sf_address + 12] = (uint8_t)second_exp;
                                    unsigned int source_token = pool_block_1 * 128 + token_base * 4 + (unsigned int)(lane * 8);
                                    unsigned int source_address = sf_k / 4 * 199680 + source_token * 4 + sf_k % 4;
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
                                        tma_store_2d((&I_fp8_w), n_block * 64, block_1 * 16 + epi_wg * 8 + (unsigned int)(s * 8), store_address);
                                    } else {
                                        tma_store_2d((&I_fp8_w), n_block * 64, block_1 * 16 + epi_wg * 8 + (unsigned int)(s * 8), store_address);
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
                        unsigned int pool_m = (unsigned int)smem_ero[task_4[1]] + task_4[2] * 16;
                        unsigned int n_offset_1 = (task_4[3] * 2 + (unsigned int)cta_rank) * 128;
                        unsigned int output_base = smem_output_addr + epi_wg_1 * 8 * 128 * 2;
                        unsigned int cached_token[1];
                        unsigned int cached_topk[1];
                        unsigned int packed[4];
                        #pragma unroll
                        for (int s_1 = 0; s_1 < 1; s_1++) {
                            if (valid_m_1 <= epi_wg_1 * 8 + (unsigned int)(s_1 * 8)) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                                break;
                            }
                            #pragma unroll
                            for (int j_1 = 0; j_1 < 1; j_1++) {
                                unsigned int row = epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + (unsigned int)(j_1 * 8) + warp_in_wg_1 * 2 + (unsigned int)(lane / 16);
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
                            for (int i_5 = 0; i_5 < 1; i_5++) {
                                unsigned int address_1 = accum_stage_1 * 16 + epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + (unsigned int)(i_5 * 8);
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
                                if (i_5 == 0 && s_1 > 0) {
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                                    asm volatile("tcgen05.fence::after_thread_sync;");
                                }
                                if (s_1 == 0 && i_5 == 0) {
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
                                unsigned int write_address = output_base + warp_in_wg_1 / 2 * 8 * 128 + (unsigned int)(i_5 * 8 * 128) + row_1 * 128 + (col ^ row_1) * 16;
                                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(write_address);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1]))
                                    : "memory");
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync %0, 128;" :: "r"(3 + epi_wg_1) : "memory");
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 1; j_2++) {
                                unsigned int row_in_store = (unsigned int)(j_2 * 8) + warp_in_wg_1 * 2 + (unsigned int)(lane / 16);
                                unsigned int row_2 = epi_wg_1 * 8 + (unsigned int)(s_1 * 8) + row_in_store;
                                if (row_2 >= valid_m_1) {
                                    break;
                                }
                                unsigned int row_in_atom = (warp_in_wg_1 * 2 + (unsigned int)(lane / 16)) % 8;
                                unsigned int read_address = output_base + (unsigned int)(lane % 16 / 8 * 8 * 128) + row_in_store * 128 + ((unsigned int)(lane % 8) ^ row_in_atom) * 16;
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                    : "r"(read_address));
                                unsigned long long dst_index = (unsigned long long)(pool_m + row_2) * 2560 + (unsigned long long)(n_offset_1 / 2) + (unsigned long long)(lane % 16 * 4);
                                reinterpret_cast<int4*>(PublicExpertOutput + dst_index)[0] = reinterpret_cast<int4*>(packed)[0];
                            }
                        }
                        asm volatile("barrier.sync 2, 256;" ::: "memory");
                    }
                    task_4[3] = task_4[3] + 1;
                    epi_uses += 1;
                }
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int l2_tag = (unsigned int)1 << 31;
            unsigned int l2_addend = 1;
            if (bid == 0) {
                l2_addend = l2_tag - (unsigned int)(NUM_CTAS - 1);
            }
            unsigned int l2_ticket = 0;
            if (warp == 8) {
                if (lane == 0) {
                    unsigned int _atomic_fetch_add_24;
                    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_add_24) : "l"(l2_done), "r"(static_cast<uint32_t>(l2_addend)) : "memory");
                    l2_ticket = _atomic_fetch_add_24;
                }
            }
            if (warp == 8) {
                __syncwarp();
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(64));
            }
            if (warp == 8) {
                if (lane == 0) {
                    unsigned int _wait_acquire_mask_1;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_1) : "l"(reinterpret_cast<unsigned int*>(l2_done)) : "memory");
                    } while (((_wait_acquire_mask_1 ^ static_cast<unsigned int>(l2_ticket ^ l2_tag)) & static_cast<unsigned int>(l2_tag)) != 0);
                }
            }
            asm volatile("barrier.sync 15, 256;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int epi_warp_2 = warp - 8;
            unsigned int values[4];
            float reduced[40];
            unsigned int store_values[4];
            unsigned int buffers = epi_warp_2 * 6;
            unsigned int _phase_combine_barriers = 0;
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
                for (unsigned int i_6 = 0; i_6 < 6; i_6++) {
                    unsigned int row_3 = __shfl_sync(0xFFFFFFFF, (unsigned int)stored_row, i_6);
                    if (elect_sync()) {
                        if ((routed >> i_6 & 1) != 0) {
                            cp_async_bulk_gmem2smem(combine_smem_addr + (epi_warp_2 * 6 + i_6) * 2560, expert_output + ((unsigned long long)row_3 * 5120 + (unsigned long long)(chunk * 1280)), 2560, combine_barriers_addr + (((buffers + i_6 >= epi_warp_2 * 6 + 6) ? buffers + i_6 - 6 : buffers + i_6)) * 8);
                            mbarrier_arrive_expect_tx(combine_barriers_addr + (((buffers + i_6 >= epi_warp_2 * 6 + 6) ? buffers + i_6 - 6 : buffers + i_6)) * 8, 2560);
                        } else {
                            mbarrier_arrive(combine_barriers_addr + (((buffers + i_6 >= epi_warp_2 * 6 + 6) ? buffers + i_6 - 6 : buffers + i_6)) * 8);
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
                for (unsigned int i_7 = 0; i_7 < 6; i_7++) {
                    mbarrier_wait(combine_barriers_addr + (((buffers + i_7 >= epi_warp_2 * 6 + 6) ? buffers + i_7 - 6 : buffers + i_7)) * 8, _phase_combine_barriers ^ (unsigned int)(buffers + i_7 >= epi_warp_2 * 6 + 6));
                    if ((routed >> i_7 & 1) != 0) {
                        #pragma unroll
                        for (unsigned int j_3 = 0; j_3 < 5; j_3++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&values[0])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values[(0) + 3]))
                                : "r"(combine_smem_addr + (epi_warp_2 * 6 + i_7) * 2560 + (j_3 * 32 + (unsigned int)lane) * 16));
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
                        void* _cpbulk_dst_25 = reinterpret_cast<void*>(y + ((unsigned long long)token * 5120 + (unsigned long long)(chunk * 1280)));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_25), "r"(combine_smem_addr + (48 + epi_warp_2) * 2560), "r"((uint32_t)(2560))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                __syncwarp();
                _phase_combine_barriers ^= 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"
