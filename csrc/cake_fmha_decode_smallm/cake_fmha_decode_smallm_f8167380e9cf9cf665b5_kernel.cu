/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_HI_OFFSET 128
#define TMEM_TMEM_O_LO_OFFSET 192
#define NUM_KV_PIPE_STAGES 2
#define NUM_SM_PIPE_STAGES 2
#define SMEM_SMEM_ST_A_OFF 191488
#define SMEM_SMEM_ST_A_STAGE_BYTES 16512
#define SMEM_SMEM_ST_A_STRIDE 16512
#define SMEM_SMEM_ST_B_OFF 208000
#define SMEM_SMEM_ST_B_STAGE_BYTES 16512
#define SMEM_SMEM_ST_B_STRIDE 16512
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 512
#define SMEM_SMEM_SCALE_STRIDE 512
#define SMEM_SMEM_SUM_OFF 2304
#define SMEM_SMEM_SUM_STAGE_BYTES 256
#define SMEM_SMEM_SUM_STRIDE 256
#define SMEM_SMEM_INV_OFF 2048
#define SMEM_SMEM_INV_STAGE_BYTES 256
#define SMEM_SMEM_INV_STRIDE 256
#define SMEM_SMEM_MAX_OFF 2560
#define SMEM_SMEM_MAX_STAGE_BYTES 256
#define SMEM_SMEM_MAX_STRIDE 256
#define SMEM_SMEM_RED_OFF 3072
#define SMEM_SMEM_RED_STAGE_BYTES 48
#define SMEM_SMEM_RED_STRIDE 48
#define SMEM_SMEM_W_OFF 4096
#define SMEM_SMEM_W_STAGE_BYTES 1024
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_RECV_HI_OFF 76800
#define SMEM_SMEM_RECV_HI_STAGE_BYTES 32768
#define SMEM_SMEM_RECV_HI_STRIDE 32768
#define SMEM_SMEM_RECV_LO_OFF 142336
#define SMEM_SMEM_RECV_LO_STAGE_BYTES 32768
#define SMEM_SMEM_RECV_LO_STRIDE 32768
#define SMEM_SMEM_RECV_LSE_OFF 4096
#define SMEM_SMEM_RECV_LSE_STAGE_BYTES 512
#define SMEM_SMEM_RECV_LSE_STRIDE 512
#define SMEM_SMEM_RECV_W_OFF 4608
#define SMEM_SMEM_RECV_W_STAGE_BYTES 512
#define SMEM_SMEM_RECV_W_STRIDE 512
#define SMEM_SMEM_RECV_INV_OFF 3072
#define SMEM_SMEM_RECV_INV_STAGE_BYTES 128
#define SMEM_SMEM_RECV_INV_STRIDE 128
#define SMEM_SMEM_Q_HI_OFF 11264
#define SMEM_SMEM_Q_HI_STAGE_BYTES 16384
#define SMEM_SMEM_Q_HI_STRIDE 16384
#define SMEM_SMEM_Q_LO_OFF 27648
#define SMEM_SMEM_Q_LO_STAGE_BYTES 16384
#define SMEM_SMEM_Q_LO_STRIDE 16384
#define SMEM_SMEM_KV_HI_OFF 44032
#define SMEM_SMEM_KV_HI_STAGE_BYTES 32768
#define SMEM_SMEM_KV_HI_STRIDE 32768
#define SMEM_SMEM_KV_LO_OFF 109568
#define SMEM_SMEM_KV_LO_STAGE_BYTES 32768
#define SMEM_SMEM_KV_LO_STRIDE 32768
#define SMEM_SMEM_V_HI_OFF 44032
#define SMEM_SMEM_V_HI_STAGE_BYTES 32768
#define SMEM_SMEM_V_HI_STRIDE 32768
#define SMEM_SMEM_V_LO_OFF 109568
#define SMEM_SMEM_V_LO_STAGE_BYTES 32768
#define SMEM_SMEM_V_LO_STRIDE 32768
#define SMEM_SMEM_P_OFF 175104
#define SMEM_SMEM_P_STAGE_BYTES 16384
#define SMEM_SMEM_P_STRIDE 16384
#define SMEM_TOTAL 224512
#define THREADS 512
#define Q_LEN 8
#define GROUP 8
#define Q_BOX_ROWS 8
#define NUM_SPLIT 9
#define BLOCK_N 128
#define HEAD_DIM 256
#define HEAD_DIM_HALF 128
#define NUM_KV_STAGES 2
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
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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



__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}




__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(9,1,1) void
kernel_cake_fmha_decode_smallm_f8167380e9cf9cf665b5(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ partial_O_ptr, float* __restrict__ partial_LSE_ptr, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, unsigned int* __restrict__ counters, int* __restrict__ page_table, int* __restrict__ seq_lens, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads)
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

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 24)
    #define s_full_addr (mbar_base + 40)
    #define corr_scale_addr (mbar_base + 56)
    #define p_full_addr (mbar_base + 72)
    #define o_ready_addr (mbar_base + 80)
    #define o_final_addr (mbar_base + 88)
    #define p_empty_addr (mbar_base + 96)
    #define stats_full_addr (mbar_base + 104)
    #define tmem_dealloc_addr (mbar_base + 112)
    #define recv_full_addr (mbar_base + 120)
    #define lse_full_addr (mbar_base + 128)
    #define recv_free_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 9;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 9;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* smem_st_a = reinterpret_cast<float*>(smem_raw + 191488);
    const int smem_st_a_addr = smem + 191488;
    float* smem_st_b = reinterpret_cast<float*>(smem_raw + 208000);
    const int smem_st_b_addr = smem + 208000;
    float* smem_scale = reinterpret_cast<float*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 2304);
    const int smem_sum_addr = smem + 2304;
    float* smem_inv = reinterpret_cast<float*>(smem_raw + 2048);
    const int smem_inv_addr = smem + 2048;
    float* smem_max = reinterpret_cast<float*>(smem_raw + 2560);
    const int smem_max_addr = smem + 2560;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 3072);
    const int smem_red_addr = smem + 3072;
    float* smem_w = reinterpret_cast<float*>(smem_raw + 4096);
    const int smem_w_addr = smem + 4096;
    __nv_bfloat16* smem_recv_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 76800);
    const int smem_recv_hi_addr = smem + 76800;
    __nv_bfloat16* smem_recv_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 142336);
    const int smem_recv_lo_addr = smem + 142336;
    float* smem_recv_lse = reinterpret_cast<float*>(smem_raw + 4096);
    const int smem_recv_lse_addr = smem + 4096;
    float* smem_recv_w = reinterpret_cast<float*>(smem_raw + 4608);
    const int smem_recv_w_addr = smem + 4608;
    float* smem_recv_inv = reinterpret_cast<float*>(smem_raw + 3072);
    const int smem_recv_inv_addr = smem + 3072;
    __nv_bfloat16* smem_q_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 11264);
    const int smem_q_hi_addr = smem + 11264;
    __nv_bfloat16* smem_q_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 27648);
    const int smem_q_lo_addr = smem + 27648;
    __nv_bfloat16* smem_kv_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 44032);
    const int smem_kv_hi_addr = smem + 44032;
    __nv_bfloat16* smem_kv_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 109568);
    const int smem_kv_lo_addr = smem + 109568;
    __nv_bfloat16* smem_v_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 44032);
    const int smem_v_hi_addr = smem + 44032;
    __nv_bfloat16* smem_v_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 109568);
    const int smem_v_lo_addr = smem + 109568;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 175104);
    const int smem_p_addr = smem + 175104;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // corr_scale: 2 barriers, init_count=256
            mbarrier_init(smem + 56, 256);
            mbarrier_init(smem + 64, 256);
            // p_full: 1 barriers, init_count=384
            mbarrier_init(smem + 72, 384);
            // o_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // o_final: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // p_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // stats_full: 1 barriers, init_count=256
            mbarrier_init(smem + 104, 256);
            // tmem_dealloc: 1 barriers, init_count=384
            mbarrier_init(smem + 112, 384);
            // recv_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // lse_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // recv_free: 1 barriers, init_count=8
            mbarrier_init(smem + 136, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o_hi = taddr + 128;
    const int tmem_tmem_o_lo = taddr + 192;
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int work = blockIdx.x;
            int tile_idx = work / NUM_SPLIT;
            int split_idx = work % NUM_SPLIT;
            int batch_idx = tile_idx / num_kv_heads;
            int kv_head_idx = tile_idx % num_kv_heads;
            int kv_len = seq_lens[batch_idx];
            int num_blocks = (kv_len + BLOCK_N - 1) / BLOCK_N;
            int base_blocks = num_blocks / NUM_SPLIT;
            int extra_blocks = num_blocks % NUM_SPLIT;
            int cnt = base_blocks;
            int start = extra_blocks * (base_blocks + 1) + (split_idx - extra_blocks) * base_blocks;
            if (split_idx < extra_blocks) {
                cnt = base_blocks + 1;
                start = split_idx * (base_blocks + 1);
            }
            int prefix = kv_len - Q_LEN;
            int wg = warp / 4;
            const int warp_in_wg = warp % 4;
            const int lane_base = warp_in_wg * 32;
            int wg_tid = warp_in_wg * 32 + lane;
            int col_local = wg_tid / 4;
            int quarter = wg_tid % 4;
            int col_base = wg * 32;
            int my_col = col_base + col_local;
            int tok_base = quarter * 8;
            int my_s_base = taddr + (unsigned int)(lane_base << 16) + (unsigned int)col_base;
            float* st_ptr = ((wg != 0) ? (smem_st_b) : (smem_st_a));
            int vis_col = prefix + 1 + my_col / GROUP;
            float row_max = -CUDART_INF_F;
            float psum = 0.0f;
            int sm_stage = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_p_empty_0 = 1;
            #pragma unroll 1
            for (int n = 0; n < cnt; n++) {
                mbarrier_wait(s_full_addr + (sm_stage) * 8, _phase_s_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(my_s_base + sm_stage * 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int tok_w = lane_base + lane;
                #pragma unroll
                for (int c = 0; c < 32; c++) {
                    st_ptr[c * 129 + tok_w] = _tmem_load_0[c];
                }
                if (wg != 0) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int my_block = start + cnt - 1 - n;
                int blk_pos = my_block * BLOCK_N + tok_base;
                float svals[32];
                float lmax = -CUDART_INF_F;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float s_ji = st_ptr[col_local * 129 + tok_base + 32 * j + i];
                        if (vis_col <= blk_pos + 32 * j + i) {
                            s_ji = -CUDART_INF_F;
                        }
                        svals[j * 8 + i] = s_ji;
                        lmax = max_noftz(lmax, s_ji);
                    }
                }
                float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, lmax, 1);
                lmax = max_noftz(lmax, _shfl_xor_sync_0);
                float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, lmax, 2);
                lmax = max_noftz(lmax, _shfl_xor_sync_1);
                float new_max;
                asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(row_max), "f"(lmax));
                float acc_scale = 1.0f;
                if (row_max > -CUDART_INF_F) {
                    float _exp2_2 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                    acc_scale = _exp2_2;
                }
                if (quarter == 0) {
                    smem_scale[sm_stage * 64 + my_col] = acc_scale;
                }
                mbarrier_arrive(corr_scale_addr + (sm_stage) * 8);
                float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                float p_vals[32];
                float lsum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    float p_k = approx_exp2((svals[k] - safe_max) * softmax_scale_log2);
                    p_vals[k] = p_k;
                    lsum = lsum + p_k;
                }
                psum = __fmaf_rn(psum, acc_scale, lsum);
                row_max = new_max;
                mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                _phase_p_empty_0 ^= 1;
                int k_run = 0;
                unsigned int regs_p[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 0], p_vals[_lp*2+1 + 0]));
                    regs_p[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run / 64 * 64 + my_col) * 128 + (k_run % 64 + tok_base) * 2 ^ ((k_run / 64 * 64 + my_col) * 128 + (k_run % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p[0]), "r"(regs_p[1]), "r"(regs_p[2]), "r"(regs_p[3]) : "memory");
                int k_run_0 = 32;
                unsigned int regs_p_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 8], p_vals[_lp*2+1 + 8]));
                    regs_p_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_0 / 64 * 64 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 ^ ((k_run_0 / 64 * 64 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_1[0]), "r"(regs_p_1[1]), "r"(regs_p_1[2]), "r"(regs_p_1[3]) : "memory");
                int k_run_2 = 64;
                unsigned int regs_p_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 16], p_vals[_lp*2+1 + 16]));
                    regs_p_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_2 / 64 * 64 + my_col) * 128 + (k_run_2 % 64 + tok_base) * 2 ^ ((k_run_2 / 64 * 64 + my_col) * 128 + (k_run_2 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_3[0]), "r"(regs_p_3[1]), "r"(regs_p_3[2]), "r"(regs_p_3[3]) : "memory");
                int k_run_4 = 96;
                unsigned int regs_p_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 24], p_vals[_lp*2+1 + 24]));
                    regs_p_5[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_4 / 64 * 64 + my_col) * 128 + (k_run_4 % 64 + tok_base) * 2 ^ ((k_run_4 / 64 * 64 + my_col) * 128 + (k_run_4 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_5[0]), "r"(regs_p_5[1]), "r"(regs_p_5[2]), "r"(regs_p_5[3]) : "memory");
                asm volatile("fence.proxy.async;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                if (wg != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                sm_stage += 1;
                if (sm_stage == 2) { sm_stage = 0; _phase_s_full ^= 1; }
            }
            float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, psum, 1);
            float total = psum + _shfl_xor_sync_3;
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_4;
            if (quarter == 0) {
                smem_sum[my_col] = total;
                smem_max[my_col] = row_max;
                float _rcp_5 = approx_rcp(total);
                float inv_total = ((total > 0.0f) ? _rcp_5 : 0.0f);
                smem_inv[my_col] = inv_total;
            }
            mbarrier_arrive(stats_full_addr);
            if (quarter == 0) {
                float lse_c = -CUDART_INF_F;
                if (total > 0.0f && row_max > -CUDART_INF_F) {
                    float _log2_6;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(total));
                    lse_c = __fmaf_rn(row_max, softmax_scale_log2, _log2_6);
                }
                int g_r = my_col / 8;
                int owner_r = g_r % 9;
                int lse_slot = g_r / 9 * 9 + split_idx;
                uint32_t _cluster_mapa_7;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_7) : "r"(smem_recv_lse_addr + (unsigned int)((lse_slot * 8 + my_col % 8) * 4)), "r"(owner_r));
                uint32_t _mbar_peer_8;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_8) : "r"(lse_full_addr), "r"(owner_r));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_7), "f"(lse_c), "r"(_mbar_peer_8) : "memory");
            }
            if (wg != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (cnt > 0) {
                mbarrier_wait(o_final_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            float inv_s[32];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(0) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(4) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4 + 16)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[8])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(8) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4 + 32)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[12])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(12) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4 + 48)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[16])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(16) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4 + 64)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[20])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(20) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4 + 80)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[24])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(24) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4 + 96)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&inv_s[28])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_s[(28) + 3]))
                : "r"(smem_inv_addr + (unsigned int)(col_base * 4 + 112)));
            float o_sm[32];
            if (cnt > 0) {
                int o_lo_base = taddr + 192 + (unsigned int)(lane_base << 16) + (unsigned int)col_base;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_sm[0]), "=f"(o_sm[1]), "=f"(o_sm[2]), "=f"(o_sm[3]), "=f"(o_sm[4]), "=f"(o_sm[5]), "=f"(o_sm[6]), "=f"(o_sm[7]), "=f"(o_sm[8]), "=f"(o_sm[9]), "=f"(o_sm[10]), "=f"(o_sm[11]), "=f"(o_sm[12]), "=f"(o_sm[13]), "=f"(o_sm[14]), "=f"(o_sm[15]), "=f"(o_sm[16]), "=f"(o_sm[17]), "=f"(o_sm[18]), "=f"(o_sm[19]), "=f"(o_sm[20]), "=f"(o_sm[21]), "=f"(o_sm[22]), "=f"(o_sm[23]), "=f"(o_sm[24]), "=f"(o_sm[25]), "=f"(o_sm[26]), "=f"(o_sm[27]), "=f"(o_sm[28]), "=f"(o_sm[29]), "=f"(o_sm[30]), "=f"(o_sm[31])
                    : "r"(o_lo_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            } else {
                o_sm[0] = 0.0f;
                o_sm[1] = 0.0f;
                o_sm[2] = 0.0f;
                o_sm[3] = 0.0f;
                o_sm[4] = 0.0f;
                o_sm[5] = 0.0f;
                o_sm[6] = 0.0f;
                o_sm[7] = 0.0f;
                o_sm[8] = 0.0f;
                o_sm[9] = 0.0f;
                o_sm[10] = 0.0f;
                o_sm[11] = 0.0f;
                o_sm[12] = 0.0f;
                o_sm[13] = 0.0f;
                o_sm[14] = 0.0f;
                o_sm[15] = 0.0f;
                o_sm[16] = 0.0f;
                o_sm[17] = 0.0f;
                o_sm[18] = 0.0f;
                o_sm[19] = 0.0f;
                o_sm[20] = 0.0f;
                o_sm[21] = 0.0f;
                o_sm[22] = 0.0f;
                o_sm[23] = 0.0f;
                o_sm[24] = 0.0f;
                o_sm[25] = 0.0f;
                o_sm[26] = 0.0f;
                o_sm[27] = 0.0f;
                o_sm[28] = 0.0f;
                o_sm[29] = 0.0f;
                o_sm[30] = 0.0f;
                o_sm[31] = 0.0f;
            }
            int d_lo = HEAD_DIM_HALF + lane_base + lane;
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            float vals_s[32];
            #pragma unroll
            for (int c_1 = 0; c_1 < 32; c_1++) {
                vals_s[c_1] = o_sm[c_1] * inv_s[c_1];
                if (cnt == 0) {
                    vals_s[c_1] = 0.0f;
                }
            }
            mbarrier_wait_cluster_hint(recv_free_addr, 0, 10000000);
            int owner = col_base / 8 % 9;
            int slot = col_base / 8 / 9 * 9 + split_idx;
            unsigned int words[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_s[_lp*2 + 0], vals_s[_lp*2+1 + 0]));
                words[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int slot_addr = ((slot < 8) ? smem_recv_hi_addr + (unsigned int)(slot * 4096) : smem_recv_lo_addr + (unsigned int)((slot - 8) * 4096));
            uint32_t _cluster_mapa_9;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_9) : "r"(slot_addr + (unsigned int)(d_lo * 16)), "r"(owner));
            uint32_t _mbar_peer_10;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_10) : "r"(recv_full_addr), "r"(owner));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_cluster_mapa_9), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]), "r"(_mbar_peer_10) : "memory");
            int owner_0 = (col_base / 8 + 1) % 9;
            int slot_1 = (col_base / 8 + 1) / 9 * 9 + split_idx;
            unsigned int words_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_s[_lp*2 + 8], vals_s[_lp*2+1 + 8]));
                words_2[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int slot_addr_3 = ((slot_1 < 8) ? smem_recv_hi_addr + (unsigned int)(slot_1 * 4096) : smem_recv_lo_addr + (unsigned int)((slot_1 - 8) * 4096));
            uint32_t _cluster_mapa_11;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_11) : "r"(slot_addr_3 + (unsigned int)(d_lo * 16)), "r"(owner_0));
            uint32_t _mbar_peer_12;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_12) : "r"(recv_full_addr), "r"(owner_0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_cluster_mapa_11), "r"(words_2[0]), "r"(words_2[1]), "r"(words_2[2]), "r"(words_2[3]), "r"(_mbar_peer_12) : "memory");
            int owner_4 = (col_base / 8 + 2) % 9;
            int slot_5 = (col_base / 8 + 2) / 9 * 9 + split_idx;
            unsigned int words_6[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_s[_lp*2 + 16], vals_s[_lp*2+1 + 16]));
                words_6[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int slot_addr_7 = ((slot_5 < 8) ? smem_recv_hi_addr + (unsigned int)(slot_5 * 4096) : smem_recv_lo_addr + (unsigned int)((slot_5 - 8) * 4096));
            uint32_t _cluster_mapa_13;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_13) : "r"(slot_addr_7 + (unsigned int)(d_lo * 16)), "r"(owner_4));
            uint32_t _mbar_peer_14;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_14) : "r"(recv_full_addr), "r"(owner_4));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_cluster_mapa_13), "r"(words_6[0]), "r"(words_6[1]), "r"(words_6[2]), "r"(words_6[3]), "r"(_mbar_peer_14) : "memory");
            int owner_8 = (col_base / 8 + 3) % 9;
            int slot_9 = (col_base / 8 + 3) / 9 * 9 + split_idx;
            unsigned int words_10[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_s[_lp*2 + 24], vals_s[_lp*2+1 + 24]));
                words_10[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int slot_addr_11 = ((slot_9 < 8) ? smem_recv_hi_addr + (unsigned int)(slot_9 * 4096) : smem_recv_lo_addr + (unsigned int)((slot_9 - 8) * 4096));
            uint32_t _cluster_mapa_15;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_15) : "r"(slot_addr_11 + (unsigned int)(d_lo * 16)), "r"(owner_8));
            uint32_t _mbar_peer_16;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_16) : "r"(recv_full_addr), "r"(owner_8));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_cluster_mapa_15), "r"(words_10[0]), "r"(words_10[1]), "r"(words_10[2]), "r"(words_10[3]), "r"(_mbar_peer_16) : "memory");
            int flat_tid_s = warp * 32 + lane;
            if (split_idx < 8) {
                mbarrier_wait_cluster_hint(lse_full_addr, 0, 10000000);
                int warp_q = flat_tid_s / 32;
                int lane_m = flat_tid_s % 32;
                int group = split_idx;
                if (group < 8) {
                    float lse_s = -CUDART_INF_F;
                    if (lane_m < 9) {
                        lse_s = smem_recv_lse[lane_m * 8 + warp_q];
                    }
                    float merged_max = lse_s;
                    float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, merged_max, 1);
                    merged_max = max_noftz(merged_max, _shfl_xor_sync_17);
                    float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, merged_max, 2);
                    merged_max = max_noftz(merged_max, _shfl_xor_sync_18);
                    float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, merged_max, 4);
                    merged_max = max_noftz(merged_max, _shfl_xor_sync_19);
                    float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, merged_max, 8);
                    merged_max = max_noftz(merged_max, _shfl_xor_sync_20);
                    float w_s = 0.0f;
                    if (lse_s != -CUDART_INF_F) {
                        float _exp2_21 = approx_exp2(lse_s - merged_max);
                        w_s = _exp2_21;
                    }
                    if (lane_m < 9) {
                        smem_recv_w[lane_m * 8 + warp_q] = w_s;
                    }
                    float weight_sum = 0.0f;
                    float _shfl_sync_22 = __shfl_sync(0xFFFFFFFF, w_s, 0);
                    weight_sum = weight_sum + _shfl_sync_22;
                    float _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, w_s, 1);
                    weight_sum = weight_sum + _shfl_sync_23;
                    float _shfl_sync_24 = __shfl_sync(0xFFFFFFFF, w_s, 2);
                    weight_sum = weight_sum + _shfl_sync_24;
                    float _shfl_sync_25 = __shfl_sync(0xFFFFFFFF, w_s, 3);
                    weight_sum = weight_sum + _shfl_sync_25;
                    float _shfl_sync_26 = __shfl_sync(0xFFFFFFFF, w_s, 4);
                    weight_sum = weight_sum + _shfl_sync_26;
                    float _shfl_sync_27 = __shfl_sync(0xFFFFFFFF, w_s, 5);
                    weight_sum = weight_sum + _shfl_sync_27;
                    float _shfl_sync_28 = __shfl_sync(0xFFFFFFFF, w_s, 6);
                    weight_sum = weight_sum + _shfl_sync_28;
                    float _shfl_sync_29 = __shfl_sync(0xFFFFFFFF, w_s, 7);
                    weight_sum = weight_sum + _shfl_sync_29;
                    float _shfl_sync_30 = __shfl_sync(0xFFFFFFFF, w_s, 8);
                    weight_sum = weight_sum + _shfl_sync_30;
                    int r = group * 8 + warp_q;
                    if (lane_m == 0) {
                        float _rcp_31 = approx_rcp(weight_sum);
                        float inv_w = ((weight_sum > 0.0f) ? _rcp_31 : 0.0f);
                        smem_recv_inv[warp_q] = inv_w;
                        if (r < Q_LEN * GROUP) {
                            float merged_lse = -CUDART_INF_F;
                            if (weight_sum > 0.0f) {
                                float _log2_32;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_32) : "f"(weight_sum));
                                merged_lse = merged_max + _log2_32;
                            }
                            int q_row_l = batch_idx * Q_LEN + r / GROUP;
                            int q_head_l = kv_head_idx * GROUP + r % GROUP;
                            *(reinterpret_cast<float*>(LSE_ptr + (q_row_l * num_q_heads + q_head_l)) + (0)) = merged_lse;
                        }
                    }
                }
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                mbarrier_wait_cluster_hint(recv_full_addr, 0, 10000000);
                int group_0 = split_idx;
                if (group_0 < 8) {
                    float acc[8];
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    unsigned int words_0[4];
                    unsigned int static_addr = smem_recv_hi_addr;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 3]))
                        : "r"(static_addr + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_0[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) << 16);
                        (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0[1 + _pair]) << 16);
                        (&_unpack_bf16_0[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0[2 + _pair]) << 16);
                        (&_unpack_bf16_0[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0[3 + _pair]) << 16);
                        (&_unpack_bf16_0[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r[(0) + 3]))
                        : "r"(smem_recv_w_addr));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r[(4) + 3]))
                        : "r"(smem_recv_w_addr + 16));
                    #pragma unroll
                    for (int q = 0; q < 8; q++) {
                        acc[q] = __fmaf_rn(_unpack_bf16_0[q], w_r[q], acc[q]);
                    }
                    unsigned int words_1[4];
                    unsigned int static_addr_2 = smem_recv_hi_addr + 4096;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                        : "r"(static_addr_2 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_1[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) << 16);
                        (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[1 + _pair]) << 16);
                        (&_unpack_bf16_1[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[2 + _pair]) << 16);
                        (&_unpack_bf16_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[3 + _pair]) << 16);
                        (&_unpack_bf16_1[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_3[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[(0) + 3]))
                        : "r"(smem_recv_w_addr + 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_3[(4) + 3]))
                        : "r"(smem_recv_w_addr + 48));
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 8; q_1++) {
                        acc[q_1] = __fmaf_rn(_unpack_bf16_1[q_1], w_r_3[q_1], acc[q_1]);
                    }
                    unsigned int words_4[4];
                    unsigned int static_addr_5 = smem_recv_hi_addr + 8192;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 3]))
                        : "r"(static_addr_5 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_2[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) << 16);
                        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_4[1 + _pair]) << 16);
                        (&_unpack_bf16_2[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_4[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_4[2 + _pair]) << 16);
                        (&_unpack_bf16_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_4[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_4[3 + _pair]) << 16);
                        (&_unpack_bf16_2[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_4[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_6[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[(0) + 3]))
                        : "r"(smem_recv_w_addr + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_6[(4) + 3]))
                        : "r"(smem_recv_w_addr + 80));
                    #pragma unroll
                    for (int q_2 = 0; q_2 < 8; q_2++) {
                        acc[q_2] = __fmaf_rn(_unpack_bf16_2[q_2], w_r_6[q_2], acc[q_2]);
                    }
                    unsigned int words_7[4];
                    unsigned int static_addr_8 = smem_recv_hi_addr + 12288;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_7[(0) + 3]))
                        : "r"(static_addr_8 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_3[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_7[_pair]) << 16);
                        (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_7[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_7[1 + _pair]) << 16);
                        (&_unpack_bf16_3[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_7[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_7[2 + _pair]) << 16);
                        (&_unpack_bf16_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_7[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_7[3 + _pair]) << 16);
                        (&_unpack_bf16_3[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_7[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_9[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[(0) + 3]))
                        : "r"(smem_recv_w_addr + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_9[(4) + 3]))
                        : "r"(smem_recv_w_addr + 112));
                    #pragma unroll
                    for (int q_3 = 0; q_3 < 8; q_3++) {
                        acc[q_3] = __fmaf_rn(_unpack_bf16_3[q_3], w_r_9[q_3], acc[q_3]);
                    }
                    unsigned int words_11[4];
                    unsigned int static_addr_12 = smem_recv_hi_addr + 16384;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_11[(0) + 3]))
                        : "r"(static_addr_12 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_4[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_11[_pair]) << 16);
                        (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_11[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_11[1 + _pair]) << 16);
                        (&_unpack_bf16_4[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_11[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_11[2 + _pair]) << 16);
                        (&_unpack_bf16_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_11[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_11[3 + _pair]) << 16);
                        (&_unpack_bf16_4[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_11[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_13[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[(0) + 3]))
                        : "r"(smem_recv_w_addr + 128));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_13[(4) + 3]))
                        : "r"(smem_recv_w_addr + 144));
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 8; q_4++) {
                        acc[q_4] = __fmaf_rn(_unpack_bf16_4[q_4], w_r_13[q_4], acc[q_4]);
                    }
                    unsigned int words_14[4];
                    unsigned int static_addr_15 = smem_recv_hi_addr + 20480;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_14[(0) + 3]))
                        : "r"(static_addr_15 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_5[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_14[_pair]) << 16);
                        (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_14[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_14[1 + _pair]) << 16);
                        (&_unpack_bf16_5[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_14[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_14[2 + _pair]) << 16);
                        (&_unpack_bf16_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_14[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_14[3 + _pair]) << 16);
                        (&_unpack_bf16_5[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_14[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_16[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[(0) + 3]))
                        : "r"(smem_recv_w_addr + 160));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_16[(4) + 3]))
                        : "r"(smem_recv_w_addr + 176));
                    #pragma unroll
                    for (int q_5 = 0; q_5 < 8; q_5++) {
                        acc[q_5] = __fmaf_rn(_unpack_bf16_5[q_5], w_r_16[q_5], acc[q_5]);
                    }
                    unsigned int words_17[4];
                    unsigned int static_addr_18 = smem_recv_hi_addr + 24576;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_17[(0) + 3]))
                        : "r"(static_addr_18 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_6[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_17[_pair]) << 16);
                        (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_17[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_17[1 + _pair]) << 16);
                        (&_unpack_bf16_6[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_17[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_17[2 + _pair]) << 16);
                        (&_unpack_bf16_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_17[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_17[3 + _pair]) << 16);
                        (&_unpack_bf16_6[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_17[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_19[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[(0) + 3]))
                        : "r"(smem_recv_w_addr + 192));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_19[(4) + 3]))
                        : "r"(smem_recv_w_addr + 208));
                    #pragma unroll
                    for (int q_6 = 0; q_6 < 8; q_6++) {
                        acc[q_6] = __fmaf_rn(_unpack_bf16_6[q_6], w_r_19[q_6], acc[q_6]);
                    }
                    unsigned int words_20[4];
                    unsigned int static_addr_21 = smem_recv_hi_addr + 28672;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_20[(0) + 3]))
                        : "r"(static_addr_21 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_7[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_20[_pair]) << 16);
                        (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_20[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_20[1 + _pair]) << 16);
                        (&_unpack_bf16_7[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_20[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_20[2 + _pair]) << 16);
                        (&_unpack_bf16_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_20[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_20[3 + _pair]) << 16);
                        (&_unpack_bf16_7[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_20[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_22[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[(0) + 3]))
                        : "r"(smem_recv_w_addr + 224));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_22[(4) + 3]))
                        : "r"(smem_recv_w_addr + 240));
                    #pragma unroll
                    for (int q_7 = 0; q_7 < 8; q_7++) {
                        acc[q_7] = __fmaf_rn(_unpack_bf16_7[q_7], w_r_22[q_7], acc[q_7]);
                    }
                    unsigned int words_23[4];
                    unsigned int static_addr_24 = smem_recv_lo_addr;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&words_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_23[(0) + 3]))
                        : "r"(static_addr_24 + (unsigned int)(flat_tid_s * 16)));
                    float _unpack_bf16_8[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_23[_pair]) << 16);
                        (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_23[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_23[1 + _pair]) << 16);
                        (&_unpack_bf16_8[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_23[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_23[2 + _pair]) << 16);
                        (&_unpack_bf16_8[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_23[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_23[3 + _pair]) << 16);
                        (&_unpack_bf16_8[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_23[3 + _pair]) & 0xffff0000u);
                    }
                    float w_r_25[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[(0) + 3]))
                        : "r"(smem_recv_w_addr + 256));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_r_25[(4) + 3]))
                        : "r"(smem_recv_w_addr + 272));
                    #pragma unroll
                    for (int q_8 = 0; q_8 < 8; q_8++) {
                        acc[q_8] = __fmaf_rn(_unpack_bf16_8[q_8], w_r_25[q_8], acc[q_8]);
                    }
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 8; q_9++) {
                        int r_1 = group_0 * 8 + q_9;
                        if (r_1 < Q_LEN * GROUP) {
                            float out_q = acc[q_9] * smem_recv_inv[q_9];
                            int q_row = batch_idx * Q_LEN + r_1 / GROUP;
                            int q_head = kv_head_idx * GROUP + r_1 % GROUP;
                            *(reinterpret_cast<__nv_bfloat16*>(O_ptr + ((q_row * num_q_heads + q_head) * HEAD_DIM + flat_tid_s)) + (0)) = (__nv_bfloat16)out_q;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            int work_1 = blockIdx.x;
            int tile_idx_1 = work_1 / NUM_SPLIT;
            int split_idx_1 = work_1 % NUM_SPLIT;
            int batch_idx_1 = tile_idx_1 / num_kv_heads;
            int kv_head_idx_1 = tile_idx_1 % num_kv_heads;
            int kv_len_1 = seq_lens[batch_idx_1];
            int num_blocks_1 = (kv_len_1 + BLOCK_N - 1) / BLOCK_N;
            int base_blocks_1 = num_blocks_1 / NUM_SPLIT;
            int extra_blocks_1 = num_blocks_1 % NUM_SPLIT;
            int cnt_1 = base_blocks_1;
            int start_1 = extra_blocks_1 * (base_blocks_1 + 1) + (split_idx_1 - extra_blocks_1) * base_blocks_1;
            if (split_idx_1 < extra_blocks_1) {
                cnt_1 = base_blocks_1 + 1;
                start_1 = split_idx_1 * (base_blocks_1 + 1);
            }
            int prefix_1 = kv_len_1 - Q_LEN;
            const int warp_in_wg_c = warp % 4;
            const int corr_row = warp_in_wg_c * 32 << 16;
            int d_idx = warp_in_wg_c * 32 + lane;
            unsigned int corr_scale_cursor = 0;
            unsigned int _phase_corr_scale = 0;
            unsigned int _phase_o_ready_0 = 0;
            #pragma unroll 1
            for (int n_1 = 0; n_1 < cnt_1; n_1++) {
                mbarrier_wait(corr_scale_addr + (corr_scale_cursor) * 8, _phase_corr_scale);
                corr_scale_cursor += 1;
                if (corr_scale_cursor == 2) { corr_scale_cursor = 0; _phase_corr_scale ^= 1; }
                if (n_1 > 0) {
                    mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                    _phase_o_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sc_off = n_1 % NUM_KV_STAGES * 64;
                    int need = 0;
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 64; c_2++) {
                        need = need | ((smem_scale[sc_off + c_2] != 1.0f) ? 1 : 0);
                    }
                    int _warp_vote_0 = __any_sync(0xFFFFFFFF, need != 0);
                    if (_warp_vote_0 != 0) {
                        #pragma unroll
                        for (int half = 0; half < 2; half++) {
                            int o_base = taddr + (unsigned int)(((half == 0) ? 128 : 192)) + (unsigned int)corr_row;
                            #pragma unroll
                            for (int chunk = 0; chunk < 2; chunk++) {
                                float _tmem_load_1[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                                    : "r"(o_base + chunk * 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int c_3 = 0; c_3 < 32; c_3++) {
                                    _tmem_load_1[c_3] = _tmem_load_1[c_3] * smem_scale[sc_off + chunk * 32 + c_3];
                                }
                                tmem_st_x32_f32(o_base + chunk * 32, _tmem_load_1);
                            }
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
            }
            if (cnt_1 > 0) {
                mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                _phase_o_ready_0 ^= 1;
            }
            if (warp_in_wg_c == 0) {
                if (elect_sync()) {
                    if (cnt_1 > 1) {
                        if (split_idx_1 < 8) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(0) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(1) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(2) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(3) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(4) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(5) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(6) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(7) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(recv_free_addr), "r"(8) : "memory");
                        }
                    }
                }
            }
            unsigned int _phase_stats_full_0 = 0;
            mbarrier_wait(stats_full_addr, _phase_stats_full_0);
            _phase_stats_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait_cluster_hint(recv_free_addr, 0, 10000000);
            int slot_2 = work_1;
            int o_hi_base = taddr + 128 + (unsigned int)corr_row;
            #pragma unroll
            for (int chunk_1 = 0; chunk_1 < 2; chunk_1++) {
                float inv[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[0])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(0) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[4])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(4) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128 + 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[8])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(8) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128 + 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[12])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(12) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128 + 48)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[16])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(16) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128 + 64)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[20])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(20) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128 + 80)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[24])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(24) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128 + 96)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&inv[28])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv[(28) + 3]))
                    : "r"(smem_inv_addr + (unsigned int)(chunk_1 * 128 + 112)));
                float o_epi[32];
                if (cnt_1 > 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_epi[0]), "=f"(o_epi[1]), "=f"(o_epi[2]), "=f"(o_epi[3]), "=f"(o_epi[4]), "=f"(o_epi[5]), "=f"(o_epi[6]), "=f"(o_epi[7]), "=f"(o_epi[8]), "=f"(o_epi[9]), "=f"(o_epi[10]), "=f"(o_epi[11]), "=f"(o_epi[12]), "=f"(o_epi[13]), "=f"(o_epi[14]), "=f"(o_epi[15]), "=f"(o_epi[16]), "=f"(o_epi[17]), "=f"(o_epi[18]), "=f"(o_epi[19]), "=f"(o_epi[20]), "=f"(o_epi[21]), "=f"(o_epi[22]), "=f"(o_epi[23]), "=f"(o_epi[24]), "=f"(o_epi[25]), "=f"(o_epi[26]), "=f"(o_epi[27]), "=f"(o_epi[28]), "=f"(o_epi[29]), "=f"(o_epi[30]), "=f"(o_epi[31])
                        : "r"(o_hi_base + chunk_1 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                } else {
                    o_epi[0] = 0.0f;
                    o_epi[1] = 0.0f;
                    o_epi[2] = 0.0f;
                    o_epi[3] = 0.0f;
                    o_epi[4] = 0.0f;
                    o_epi[5] = 0.0f;
                    o_epi[6] = 0.0f;
                    o_epi[7] = 0.0f;
                    o_epi[8] = 0.0f;
                    o_epi[9] = 0.0f;
                    o_epi[10] = 0.0f;
                    o_epi[11] = 0.0f;
                    o_epi[12] = 0.0f;
                    o_epi[13] = 0.0f;
                    o_epi[14] = 0.0f;
                    o_epi[15] = 0.0f;
                    o_epi[16] = 0.0f;
                    o_epi[17] = 0.0f;
                    o_epi[18] = 0.0f;
                    o_epi[19] = 0.0f;
                    o_epi[20] = 0.0f;
                    o_epi[21] = 0.0f;
                    o_epi[22] = 0.0f;
                    o_epi[23] = 0.0f;
                    o_epi[24] = 0.0f;
                    o_epi[25] = 0.0f;
                    o_epi[26] = 0.0f;
                    o_epi[27] = 0.0f;
                    o_epi[28] = 0.0f;
                    o_epi[29] = 0.0f;
                    o_epi[30] = 0.0f;
                    o_epi[31] = 0.0f;
                }
                #pragma unroll
                for (int c_4 = 0; c_4 < 32; c_4++) {
                    o_epi[c_4] = o_epi[c_4] * inv[c_4];
                    if (cnt_1 == 0) {
                        o_epi[c_4] = 0.0f;
                    }
                }
                int owner_1 = chunk_1 * 4 % 9;
                int slot_0 = chunk_1 * 4 / 9 * 9 + split_idx_1;
                unsigned int words_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_epi[_lp*2 + 0], o_epi[_lp*2+1 + 0]));
                    words_3[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_addr_1 = ((slot_0 < 8) ? smem_recv_hi_addr + (unsigned int)(slot_0 * 4096) : smem_recv_lo_addr + (unsigned int)((slot_0 - 8) * 4096));
                uint32_t _cluster_mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_1) : "r"(slot_addr_1 + (unsigned int)(d_idx * 16)), "r"(owner_1));
                uint32_t _mbar_peer_2;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_2) : "r"(recv_full_addr), "r"(owner_1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_1), "r"(words_3[0]), "r"(words_3[1]), "r"(words_3[2]), "r"(words_3[3]), "r"(_mbar_peer_2) : "memory");
                int owner_1_1 = (chunk_1 * 4 + 1) % 9;
                int slot_2_1 = (chunk_1 * 4 + 1) / 9 * 9 + split_idx_1;
                unsigned int words_3_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_epi[_lp*2 + 8], o_epi[_lp*2+1 + 8]));
                    words_3_1[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_addr_4 = ((slot_2_1 < 8) ? smem_recv_hi_addr + (unsigned int)(slot_2_1 * 4096) : smem_recv_lo_addr + (unsigned int)((slot_2_1 - 8) * 4096));
                uint32_t _cluster_mapa_3;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_3) : "r"(slot_addr_4 + (unsigned int)(d_idx * 16)), "r"(owner_1_1));
                uint32_t _mbar_peer_4;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_4) : "r"(recv_full_addr), "r"(owner_1_1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_3), "r"(words_3_1[0]), "r"(words_3_1[1]), "r"(words_3_1[2]), "r"(words_3_1[3]), "r"(_mbar_peer_4) : "memory");
                int owner_5 = (chunk_1 * 4 + 2) % 9;
                int slot_6 = (chunk_1 * 4 + 2) / 9 * 9 + split_idx_1;
                unsigned int words_7_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_epi[_lp*2 + 16], o_epi[_lp*2+1 + 16]));
                    words_7_1[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_addr_8 = ((slot_6 < 8) ? smem_recv_hi_addr + (unsigned int)(slot_6 * 4096) : smem_recv_lo_addr + (unsigned int)((slot_6 - 8) * 4096));
                uint32_t _cluster_mapa_5;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_5) : "r"(slot_addr_8 + (unsigned int)(d_idx * 16)), "r"(owner_5));
                uint32_t _mbar_peer_6;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_6) : "r"(recv_full_addr), "r"(owner_5));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_5), "r"(words_7_1[0]), "r"(words_7_1[1]), "r"(words_7_1[2]), "r"(words_7_1[3]), "r"(_mbar_peer_6) : "memory");
                int owner_9 = (chunk_1 * 4 + 3) % 9;
                int slot_10 = (chunk_1 * 4 + 3) / 9 * 9 + split_idx_1;
                unsigned int words_11_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_epi[_lp*2 + 24], o_epi[_lp*2+1 + 24]));
                    words_11_1[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int slot_addr_12 = ((slot_10 < 8) ? smem_recv_hi_addr + (unsigned int)(slot_10 * 4096) : smem_recv_lo_addr + (unsigned int)((slot_10 - 8) * 4096));
                uint32_t _cluster_mapa_7_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_7_1) : "r"(slot_addr_12 + (unsigned int)(d_idx * 16)), "r"(owner_9));
                uint32_t _mbar_peer_8;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_8) : "r"(recv_full_addr), "r"(owner_9));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_7_1), "r"(words_11_1[0]), "r"(words_11_1[1]), "r"(words_11_1[2]), "r"(words_11_1[3]), "r"(_mbar_peer_8) : "memory");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            int flat_tid_c = 256 + (warp - 8) * 32 + lane;
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int work_2 = blockIdx.x;
            int tile_idx_2 = work_2 / NUM_SPLIT;
            int split_idx_2 = work_2 % NUM_SPLIT;
            int batch_idx_2 = tile_idx_2 / num_kv_heads;
            int kv_head_idx_2 = tile_idx_2 % num_kv_heads;
            int kv_len_2 = seq_lens[batch_idx_2];
            int num_blocks_2 = (kv_len_2 + BLOCK_N - 1) / BLOCK_N;
            int base_blocks_2 = num_blocks_2 / NUM_SPLIT;
            int extra_blocks_2 = num_blocks_2 % NUM_SPLIT;
            int cnt_2 = base_blocks_2;
            int start_2 = extra_blocks_2 * (base_blocks_2 + 1) + (split_idx_2 - extra_blocks_2) * base_blocks_2;
            if (split_idx_2 < extra_blocks_2) {
                cnt_2 = base_blocks_2 + 1;
                start_2 = split_idx_2 * (base_blocks_2 + 1);
            }
            int prefix_2 = kv_len_2 - Q_LEN;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_p_full_0 = 0;
            if (cnt_2 > 0) {
                mbarrier_wait(kv_full_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_0 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 506U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_1 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 506U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                }
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr);
                int first_pv = 1;
                #pragma unroll 1
                for (int n_2 = 0; n_2 < cnt_2; n_2++) {
                    int stage = n_2 % 2;
                    if (cnt_2 > n_2 + 1) {
                        int nstage = (n_2 + 1) % 2;
                        int _stage_kv_full = (int)((unsigned int)(n_2 + 1) % 2);
                        mbarrier_wait(kv_full_addr + (_stage_kv_full) * 8, 0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (nstage) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 506U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135267472, 1);
                            }
                        }
                        int _mma_a_lo_3 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (nstage) * 2048);
                        int _mma_b_lo_3 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                            uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 506U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135267472, 1);
                            }
                        }
                        elect_commit(s_full_addr + (nstage) * 8);
                        elect_commit(kv_empty_addr + (nstage) * 8);
                    }
                    int _stage_kv_full_1 = (int)((unsigned int)n_2 % 2);
                    mbarrier_wait(kv_full_addr + (_stage_kv_full_1) * 8, 1);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int first_pv_flag = first_pv;
                    int _mma_a_lo_4 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, ((first_pv_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135300240, 1);
                        }
                    }
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, ((first_pv_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135300240, 1);
                        }
                    }
                    elect_commit2(kv_empty_addr + ((int)((unsigned int)n_2 % 2)) * 8, o_ready_addr);
                    elect_commit(p_empty_addr);
                    first_pv = 0;
                }
                elect_commit(o_final_addr);
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 13) {
        { // load_warp_main
            int split_l = blockIdx.x % NUM_SPLIT;
            int tile_l = blockIdx.x / NUM_SPLIT;
            int pt_base = tile_l / num_kv_heads * max_pages_per_seq;
            int spec_pages[2];
            int spec_p = split_l * 2;
            if (spec_p > max_pages_per_seq - 1) {
                spec_p = max_pages_per_seq - 1;
            }
            spec_pages[0] = page_table[pt_base + spec_p];
            int spec_p_0 = split_l * 2 + 1;
            if (spec_p_0 > max_pages_per_seq - 1) {
                spec_p_0 = max_pages_per_seq - 1;
            }
            spec_pages[1] = page_table[pt_base + spec_p_0];
            if (elect_sync()) {
                int q_head0 = tile_l % num_kv_heads * GROUP;
                int q_row0 = tile_l / num_kv_heads * Q_LEN;
                mbarrier_arrive_expect_tx(q_full_addr, 64 * HEAD_DIM * 2);
                tma_4d_gmem2smem(smem_q_hi_addr, (&Q), 0, q_head0, q_row0, 0, q_full_addr);
                tma_4d_gmem2smem(smem_q_lo_addr, (&Q), 0, q_head0, q_row0, 2, q_full_addr);
            }
            int work_3 = blockIdx.x;
            int tile_idx_3 = work_3 / NUM_SPLIT;
            int split_idx_3 = work_3 % NUM_SPLIT;
            int batch_idx_3 = tile_idx_3 / num_kv_heads;
            int kv_head_idx_3 = tile_idx_3 % num_kv_heads;
            int kv_len_3 = seq_lens[batch_idx_3];
            int num_blocks_3 = (kv_len_3 + BLOCK_N - 1) / BLOCK_N;
            int base_blocks_3 = num_blocks_3 / NUM_SPLIT;
            int extra_blocks_3 = num_blocks_3 % NUM_SPLIT;
            int cnt_3 = base_blocks_3;
            int start_3 = extra_blocks_3 * (base_blocks_3 + 1) + (split_idx_3 - extra_blocks_3) * base_blocks_3;
            if (split_idx_3 < extra_blocks_3) {
                cnt_3 = base_blocks_3 + 1;
                start_3 = split_idx_3 * (base_blocks_3 + 1);
            }
            int prefix_3 = kv_len_3 - Q_LEN;
            int max_pg = (kv_len_3 + 64 - 1) / 64 - 1;
            const int kv_tx = BLOCK_N * HEAD_DIM * 2;
            if (cnt_3 > 0) {
                if (elect_sync()) {
                    int prefill = ((cnt_3 < NUM_KV_STAGES) ? cnt_3 : NUM_KV_STAGES);
                    #pragma unroll 1
                    for (int ni = 0; ni < prefill; ni++) {
                        int n_block = start_3 + cnt_3 - 1 - ni;
                        int kv_stage = ni % 2;
                        int _stage_kv_empty = (int)((unsigned int)ni % 2);
                        mbarrier_wait(kv_empty_addr + (_stage_kv_empty) * 8, 1);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, kv_tx);
                        int hdst = smem_kv_hi_addr + (unsigned int)(kv_stage * 32768);
                        int ldst = smem_kv_lo_addr + (unsigned int)(kv_stage * 32768);
                        int raw_p = n_block * 2;
                        int clamp_p = ((raw_p > max_pg) ? max_pg : raw_p);
                        int pg0 = spec_pages[0];
                        if (n_block != split_l || raw_p > max_pg && max_pg != max_pages_per_seq - 1) {
                            pg0 = page_table[pt_base + clamp_p];
                        }
                        int toff = 0;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst + toff), "l"((&K)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx_3), "r"(pg0),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_0 = 16384;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst + toff_0), "l"((&K)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx_3), "r"(pg0),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_1 = 0;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst + toff_1), "l"((&K)), "r"(0), "r"(0), "r"(2), "r"(kv_head_idx_3), "r"(pg0),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_2 = 16384;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst + toff_2), "l"((&K)), "r"(0), "r"(0), "r"(3), "r"(kv_head_idx_3), "r"(pg0),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int raw_p_3 = n_block * 2 + 1;
                        int clamp_p_4 = ((raw_p_3 > max_pg) ? max_pg : raw_p_3);
                        int pg0_5 = spec_pages[1];
                        if (n_block != split_l || raw_p_3 > max_pg && max_pg != max_pages_per_seq - 1) {
                            pg0_5 = page_table[pt_base + clamp_p_4];
                        }
                        int toff_6 = 8192;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst + toff_6), "l"((&K)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx_3), "r"(pg0_5),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_7 = 24576;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst + toff_7), "l"((&K)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx_3), "r"(pg0_5),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_8 = 8192;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst + toff_8), "l"((&K)), "r"(0), "r"(0), "r"(2), "r"(kv_head_idx_3), "r"(pg0_5),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_9 = 24576;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst + toff_9), "l"((&K)), "r"(0), "r"(0), "r"(3), "r"(kv_head_idx_3), "r"(pg0_5),
                               "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < cnt_3; ni_1++) {
                        int stage_1 = ni_1 % 2;
                        int n_block_1 = start_3 + cnt_3 - 1 - ni_1;
                        int _stage_kv_empty_1 = (int)((unsigned int)ni_1 % 2);
                        mbarrier_wait(kv_empty_addr + (_stage_kv_empty_1) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage_1) * 8, kv_tx);
                        int hdst_1 = smem_kv_hi_addr + (unsigned int)(stage_1 * 32768);
                        int ldst_1 = smem_kv_lo_addr + (unsigned int)(stage_1 * 32768);
                        int raw_p_1 = n_block_1 * 2;
                        int clamp_p_1 = ((raw_p_1 > max_pg) ? max_pg : raw_p_1);
                        int pg0_1 = spec_pages[0];
                        if (n_block_1 != split_l || raw_p_1 > max_pg && max_pg != max_pages_per_seq - 1) {
                            pg0_1 = page_table[pt_base + clamp_p_1];
                        }
                        int toff_3 = 0;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst_1 + toff_3), "l"((&V)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx_3), "r"(pg0_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_0_1 = 16384;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst_1 + toff_0_1), "l"((&V)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx_3), "r"(pg0_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_1_1 = 0;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst_1 + toff_1_1), "l"((&V)), "r"(0), "r"(0), "r"(2), "r"(kv_head_idx_3), "r"(pg0_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_2_1 = 16384;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst_1 + toff_2_1), "l"((&V)), "r"(0), "r"(0), "r"(3), "r"(kv_head_idx_3), "r"(pg0_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int raw_p_3_1 = n_block_1 * 2 + 1;
                        int clamp_p_4_1 = ((raw_p_3_1 > max_pg) ? max_pg : raw_p_3_1);
                        int pg0_5_1 = spec_pages[1];
                        if (n_block_1 != split_l || raw_p_3_1 > max_pg && max_pg != max_pages_per_seq - 1) {
                            pg0_5_1 = page_table[pt_base + clamp_p_4_1];
                        }
                        int toff_6_1 = 8192;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst_1 + toff_6_1), "l"((&V)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx_3), "r"(pg0_5_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_7_1 = 24576;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(hdst_1 + toff_7_1), "l"((&V)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx_3), "r"(pg0_5_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_8_1 = 8192;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst_1 + toff_8_1), "l"((&V)), "r"(0), "r"(0), "r"(2), "r"(kv_head_idx_3), "r"(pg0_5_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int toff_9_1 = 24576;
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(ldst_1 + toff_9_1), "l"((&V)), "r"(0), "r"(0), "r"(3), "r"(kv_head_idx_3), "r"(pg0_5_1),
                               "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        int next_ni = ni_1 + NUM_KV_STAGES;
                        if (next_ni < cnt_3) {
                            int next_block = start_3 + cnt_3 - 1 - next_ni;
                            int _stage_kv_empty_2 = (int)((unsigned int)ni_1 % 2);
                            mbarrier_wait(kv_empty_addr + (_stage_kv_empty_2) * 8, 1);
                            int kstage = next_ni % 2;
                            mbarrier_arrive_expect_tx(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8, kv_tx);
                            int hdst_0 = smem_kv_hi_addr + (unsigned int)((ni_1 + NUM_KV_STAGES) % 2 * 32768);
                            int ldst_1_1 = smem_kv_lo_addr + (unsigned int)((ni_1 + NUM_KV_STAGES) % 2 * 32768);
                            int raw_p_2 = next_block * 2;
                            int clamp_p_3 = ((raw_p_2 > max_pg) ? max_pg : raw_p_2);
                            int pg0_4 = spec_pages[0];
                            if (next_block != split_l || raw_p_2 > max_pg && max_pg != max_pages_per_seq - 1) {
                                pg0_4 = page_table[pt_base + clamp_p_3];
                            }
                            int toff_5 = 0;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(hdst_0 + toff_5), "l"((&K)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx_3), "r"(pg0_4),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            int toff_10 = 16384;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(hdst_0 + toff_10), "l"((&K)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx_3), "r"(pg0_4),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            int toff_11 = 0;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ldst_1_1 + toff_11), "l"((&K)), "r"(0), "r"(0), "r"(2), "r"(kv_head_idx_3), "r"(pg0_4),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            int toff_12 = 16384;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ldst_1_1 + toff_12), "l"((&K)), "r"(0), "r"(0), "r"(3), "r"(kv_head_idx_3), "r"(pg0_4),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            int raw_p_13 = next_block * 2 + 1;
                            int clamp_p_14 = ((raw_p_13 > max_pg) ? max_pg : raw_p_13);
                            int pg0_15 = spec_pages[1];
                            if (next_block != split_l || raw_p_13 > max_pg && max_pg != max_pages_per_seq - 1) {
                                pg0_15 = page_table[pt_base + clamp_p_14];
                            }
                            int toff_16 = 8192;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(hdst_0 + toff_16), "l"((&K)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx_3), "r"(pg0_15),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            int toff_17 = 24576;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(hdst_0 + toff_17), "l"((&K)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx_3), "r"(pg0_15),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            int toff_18 = 8192;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ldst_1_1 + toff_18), "l"((&K)), "r"(0), "r"(0), "r"(2), "r"(kv_head_idx_3), "r"(pg0_15),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            int toff_19 = 24576;
                            asm volatile(
                                "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ldst_1_1 + toff_19), "l"((&K)), "r"(0), "r"(0), "r"(3), "r"(kv_head_idx_3), "r"(pg0_15),
                                   "r"(kv_full_addr + ((int)((unsigned int)(ni_1 + NUM_KV_STAGES) % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
            }
        }
    }
    // ---- Role: pad ----
    if (warp >= 14 && warp <= 15) {
        { // pad_main
            int work_4 = blockIdx.x;
            int tile_idx_4 = work_4 / NUM_SPLIT;
            int split_idx_4 = work_4 % NUM_SPLIT;
            int batch_idx_4 = tile_idx_4 / num_kv_heads;
            int kv_head_idx_4 = tile_idx_4 % num_kv_heads;
            int kv_len_4 = seq_lens[batch_idx_4];
            int num_blocks_4 = (kv_len_4 + BLOCK_N - 1) / BLOCK_N;
            int base_blocks_4 = num_blocks_4 / NUM_SPLIT;
            int extra_blocks_4 = num_blocks_4 % NUM_SPLIT;
            int cnt_4 = base_blocks_4;
            int start_4 = extra_blocks_4 * (base_blocks_4 + 1) + (split_idx_4 - extra_blocks_4) * base_blocks_4;
            if (split_idx_4 < extra_blocks_4) {
                cnt_4 = base_blocks_4 + 1;
                start_4 = split_idx_4 * (base_blocks_4 + 1);
            }
            int prefix_4 = kv_len_4 - Q_LEN;
            if (warp == 14) {
                if (elect_sync()) {
                    if (split_idx_4 < 8) {
                        int owned = (7 - split_idx_4) / 9 + 1;
                        mbarrier_arrive_expect_tx(lse_full_addr, owned * 9 * 8 * 4);
                        mbarrier_arrive_expect_tx(recv_full_addr, owned * 9 * 4096);
                        if (cnt_4 <= 1) {
                            if (split_idx_4 < 8) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(0) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(1) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(2) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(3) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(4) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(5) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(6) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(7) : "memory");
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(recv_free_addr), "r"(8) : "memory");
                            }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
