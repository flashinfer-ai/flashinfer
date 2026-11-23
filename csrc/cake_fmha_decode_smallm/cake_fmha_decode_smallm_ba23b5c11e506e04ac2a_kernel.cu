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
#define SMEM_SMEM_ST_B_STAGE_BYTES 8256
#define SMEM_SMEM_ST_B_STRIDE 8256
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
#define SMEM_SMEM_Q_HI_STAGE_BYTES 8192
#define SMEM_SMEM_Q_HI_STRIDE 8192
#define SMEM_SMEM_Q_LO_OFF 27648
#define SMEM_SMEM_Q_LO_STAGE_BYTES 8192
#define SMEM_SMEM_Q_LO_STRIDE 8192
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
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_TOTAL 224512
#define THREADS 512
#define Q_LEN 4
#define GROUP 8
#define Q_BOX_ROWS 4
#define NUM_SPLIT 152
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_decode_smallm_ba23b5c11e506e04ac2a(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ partial_O_ptr, float* __restrict__ partial_LSE_ptr, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, unsigned int* __restrict__ counters, int* __restrict__ page_table, int* __restrict__ seq_lens, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partial_O_ptr));
    __builtin_assume(__isGlobal(partial_O_ptr));
    asm volatile("" : "+l"(partial_LSE_ptr));
    __builtin_assume(__isGlobal(partial_LSE_ptr));

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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

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

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 15 barriers)
    // Mbarriers at smem_raw[0..120)

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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 224 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 0) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
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
            int col_local = wg_tid / 8;
            int quarter = wg_tid % 8;
            int col_base = wg * 16;
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
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], my_s_base + sm_stage * 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int tok_w = lane_base + lane;
                #pragma unroll
                for (int c = 0; c < 16; c++) {
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
                float svals[16];
                float lmax = -CUDART_INF_F;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float s_ji = st_ptr[col_local * 129 + tok_base + 64 * j + i];
                        if (vis_col <= blk_pos + 64 * j + i) {
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
                float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, lmax, 4);
                lmax = max_noftz(lmax, _shfl_xor_sync_2);
                float new_max;
                asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(row_max), "f"(lmax));
                float acc_scale = 1.0f;
                if (row_max > -CUDART_INF_F) {
                    float _exp2_3 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                    acc_scale = _exp2_3;
                }
                if (quarter == 0) {
                    smem_scale[sm_stage * 64 + my_col] = acc_scale;
                }
                mbarrier_arrive(corr_scale_addr + (sm_stage) * 8);
                float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                float p_vals[16];
                float lsum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 16; k++) {
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
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run / 64 * 32 + my_col) * 128 + (k_run % 64 + tok_base) * 2 ^ ((k_run / 64 * 32 + my_col) * 128 + (k_run % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p[0]), "r"(regs_p[1]), "r"(regs_p[2]), "r"(regs_p[3]) : "memory");
                int k_run_0 = 64;
                unsigned int regs_p_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 8], p_vals[_lp*2+1 + 8]));
                    regs_p_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_0 / 64 * 32 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 ^ ((k_run_0 / 64 * 32 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_1[0]), "r"(regs_p_1[1]), "r"(regs_p_1[2]), "r"(regs_p_1[3]) : "memory");
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
            float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, psum, 1);
            float total = psum + _shfl_xor_sync_4;
            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_sync_5;
            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total = total + _shfl_xor_sync_6;
            if (quarter == 0) {
                smem_sum[my_col] = total;
                smem_max[my_col] = row_max;
                float _rcp_7 = approx_rcp(total);
                float inv_total = ((total > 0.0f) ? _rcp_7 : 0.0f);
                smem_inv[my_col] = inv_total;
            }
            mbarrier_arrive(stats_full_addr);
            if (wg != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (cnt > 0) {
                mbarrier_wait(o_final_addr, 0);
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            float inv_s[16];
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
            float o_sm[16];
            if (cnt > 0) {
                int o_lo_base = taddr + 192 + (unsigned int)(lane_base << 16) + (unsigned int)col_base;
                tmem_ld_x16(&o_sm[0], o_lo_base);
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
            }
            int d_lo = HEAD_DIM_HALF + lane_base + lane;
            #pragma unroll
            for (int c_1 = 0; c_1 < 16; c_1++) {
                float val_s = o_sm[c_1] * inv_s[c_1];
                if (cnt == 0) {
                    val_s = 0.0f;
                }
                *(reinterpret_cast<__nv_bfloat16*>(partial_O_ptr + ((blockIdx.x * 32 + col_base + c_1) * HEAD_DIM + d_lo)) + (0)) = (__nv_bfloat16)val_s;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            int flat_tid_s = warp * 32 + lane;
            asm volatile("fence.release.gpu;" ::: "memory");
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            if (flat_tid_s == 0) {
                unsigned int arrived;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(arrived) : "l"(&counters[tile_idx * 2]), "r"(static_cast<uint32_t>(1)) : "memory");
                if (arrived + 1 != (unsigned int)NUM_SPLIT) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(counters) + (tile_idx * 2))) : "memory");
                    } while (_acquire_observed < static_cast<unsigned int>(NUM_SPLIT));
                    }
                }
            }
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            asm volatile("fence.acquire.gpu;" ::: "memory");
            int slot_base = tile_idx * NUM_SPLIT;
            int fold_warp = flat_tid_s / 32;
            int lane_f = flat_tid_s % 32;
            #pragma unroll 1
            for (int r = split_idx; r < 32; r += NUM_SPLIT) {
                if (r < Q_LEN * GROUP) {
                    float lses[19];
                    unsigned int words[76];
                    int slot_k = slot_base + fold_warp;
                    lses[0] = -CUDART_INF_F;
                    words[0] = 0;
                    words[1] = 0;
                    words[2] = 0;
                    words[3] = 0;
                    if (fold_warp < NUM_SPLIT) {
                        lses[0] = partial_LSE_ptr[slot_k * 32 + r];
                        {
                            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[0 + 0] = _uv4_8.x;
                            words[0 + 1] = _uv4_8.y;
                            words[0 + 2] = _uv4_8.z;
                            words[0 + 3] = _uv4_8.w;
                        }
                    }
                    int slot_k_0 = slot_base + fold_warp + 8;
                    lses[1] = -CUDART_INF_F;
                    words[4] = 0;
                    words[5] = 0;
                    words[6] = 0;
                    words[7] = 0;
                    if (fold_warp + 8 < NUM_SPLIT) {
                        lses[1] = partial_LSE_ptr[slot_k_0 * 32 + r];
                        {
                            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_0 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[4 + 0] = _uv4_9.x;
                            words[4 + 1] = _uv4_9.y;
                            words[4 + 2] = _uv4_9.z;
                            words[4 + 3] = _uv4_9.w;
                        }
                    }
                    int slot_k_1 = slot_base + fold_warp + 16;
                    lses[2] = -CUDART_INF_F;
                    words[8] = 0;
                    words[9] = 0;
                    words[10] = 0;
                    words[11] = 0;
                    if (fold_warp + 16 < NUM_SPLIT) {
                        lses[2] = partial_LSE_ptr[slot_k_1 * 32 + r];
                        {
                            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_1 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[8 + 0] = _uv4_10.x;
                            words[8 + 1] = _uv4_10.y;
                            words[8 + 2] = _uv4_10.z;
                            words[8 + 3] = _uv4_10.w;
                        }
                    }
                    int slot_k_2 = slot_base + fold_warp + 24;
                    lses[3] = -CUDART_INF_F;
                    words[12] = 0;
                    words[13] = 0;
                    words[14] = 0;
                    words[15] = 0;
                    if (fold_warp + 24 < NUM_SPLIT) {
                        lses[3] = partial_LSE_ptr[slot_k_2 * 32 + r];
                        {
                            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_2 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[12 + 0] = _uv4_11.x;
                            words[12 + 1] = _uv4_11.y;
                            words[12 + 2] = _uv4_11.z;
                            words[12 + 3] = _uv4_11.w;
                        }
                    }
                    int slot_k_3 = slot_base + fold_warp + 32;
                    lses[4] = -CUDART_INF_F;
                    words[16] = 0;
                    words[17] = 0;
                    words[18] = 0;
                    words[19] = 0;
                    if (fold_warp + 32 < NUM_SPLIT) {
                        lses[4] = partial_LSE_ptr[slot_k_3 * 32 + r];
                        {
                            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_3 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[16 + 0] = _uv4_12.x;
                            words[16 + 1] = _uv4_12.y;
                            words[16 + 2] = _uv4_12.z;
                            words[16 + 3] = _uv4_12.w;
                        }
                    }
                    int slot_k_4 = slot_base + fold_warp + 40;
                    lses[5] = -CUDART_INF_F;
                    words[20] = 0;
                    words[21] = 0;
                    words[22] = 0;
                    words[23] = 0;
                    if (fold_warp + 40 < NUM_SPLIT) {
                        lses[5] = partial_LSE_ptr[slot_k_4 * 32 + r];
                        {
                            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_4 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[20 + 0] = _uv4_13.x;
                            words[20 + 1] = _uv4_13.y;
                            words[20 + 2] = _uv4_13.z;
                            words[20 + 3] = _uv4_13.w;
                        }
                    }
                    int slot_k_5 = slot_base + fold_warp + 48;
                    lses[6] = -CUDART_INF_F;
                    words[24] = 0;
                    words[25] = 0;
                    words[26] = 0;
                    words[27] = 0;
                    if (fold_warp + 48 < NUM_SPLIT) {
                        lses[6] = partial_LSE_ptr[slot_k_5 * 32 + r];
                        {
                            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_5 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[24 + 0] = _uv4_14.x;
                            words[24 + 1] = _uv4_14.y;
                            words[24 + 2] = _uv4_14.z;
                            words[24 + 3] = _uv4_14.w;
                        }
                    }
                    int slot_k_6 = slot_base + fold_warp + 56;
                    lses[7] = -CUDART_INF_F;
                    words[28] = 0;
                    words[29] = 0;
                    words[30] = 0;
                    words[31] = 0;
                    if (fold_warp + 56 < NUM_SPLIT) {
                        lses[7] = partial_LSE_ptr[slot_k_6 * 32 + r];
                        {
                            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_6 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[28 + 0] = _uv4_15.x;
                            words[28 + 1] = _uv4_15.y;
                            words[28 + 2] = _uv4_15.z;
                            words[28 + 3] = _uv4_15.w;
                        }
                    }
                    int slot_k_7 = slot_base + fold_warp + 64;
                    lses[8] = -CUDART_INF_F;
                    words[32] = 0;
                    words[33] = 0;
                    words[34] = 0;
                    words[35] = 0;
                    if (fold_warp + 64 < NUM_SPLIT) {
                        lses[8] = partial_LSE_ptr[slot_k_7 * 32 + r];
                        {
                            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_7 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[32 + 0] = _uv4_16.x;
                            words[32 + 1] = _uv4_16.y;
                            words[32 + 2] = _uv4_16.z;
                            words[32 + 3] = _uv4_16.w;
                        }
                    }
                    int slot_k_8 = slot_base + fold_warp + 72;
                    lses[9] = -CUDART_INF_F;
                    words[36] = 0;
                    words[37] = 0;
                    words[38] = 0;
                    words[39] = 0;
                    if (fold_warp + 72 < NUM_SPLIT) {
                        lses[9] = partial_LSE_ptr[slot_k_8 * 32 + r];
                        {
                            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_8 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[36 + 0] = _uv4_17.x;
                            words[36 + 1] = _uv4_17.y;
                            words[36 + 2] = _uv4_17.z;
                            words[36 + 3] = _uv4_17.w;
                        }
                    }
                    int slot_k_9 = slot_base + fold_warp + 80;
                    lses[10] = -CUDART_INF_F;
                    words[40] = 0;
                    words[41] = 0;
                    words[42] = 0;
                    words[43] = 0;
                    if (fold_warp + 80 < NUM_SPLIT) {
                        lses[10] = partial_LSE_ptr[slot_k_9 * 32 + r];
                        {
                            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_9 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[40 + 0] = _uv4_18.x;
                            words[40 + 1] = _uv4_18.y;
                            words[40 + 2] = _uv4_18.z;
                            words[40 + 3] = _uv4_18.w;
                        }
                    }
                    int slot_k_10 = slot_base + fold_warp + 88;
                    lses[11] = -CUDART_INF_F;
                    words[44] = 0;
                    words[45] = 0;
                    words[46] = 0;
                    words[47] = 0;
                    if (fold_warp + 88 < NUM_SPLIT) {
                        lses[11] = partial_LSE_ptr[slot_k_10 * 32 + r];
                        {
                            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_10 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[44 + 0] = _uv4_19.x;
                            words[44 + 1] = _uv4_19.y;
                            words[44 + 2] = _uv4_19.z;
                            words[44 + 3] = _uv4_19.w;
                        }
                    }
                    int slot_k_11 = slot_base + fold_warp + 96;
                    lses[12] = -CUDART_INF_F;
                    words[48] = 0;
                    words[49] = 0;
                    words[50] = 0;
                    words[51] = 0;
                    if (fold_warp + 96 < NUM_SPLIT) {
                        lses[12] = partial_LSE_ptr[slot_k_11 * 32 + r];
                        {
                            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_11 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[48 + 0] = _uv4_20.x;
                            words[48 + 1] = _uv4_20.y;
                            words[48 + 2] = _uv4_20.z;
                            words[48 + 3] = _uv4_20.w;
                        }
                    }
                    int slot_k_12 = slot_base + fold_warp + 104;
                    lses[13] = -CUDART_INF_F;
                    words[52] = 0;
                    words[53] = 0;
                    words[54] = 0;
                    words[55] = 0;
                    if (fold_warp + 104 < NUM_SPLIT) {
                        lses[13] = partial_LSE_ptr[slot_k_12 * 32 + r];
                        {
                            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_12 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[52 + 0] = _uv4_21.x;
                            words[52 + 1] = _uv4_21.y;
                            words[52 + 2] = _uv4_21.z;
                            words[52 + 3] = _uv4_21.w;
                        }
                    }
                    int slot_k_13 = slot_base + fold_warp + 112;
                    lses[14] = -CUDART_INF_F;
                    words[56] = 0;
                    words[57] = 0;
                    words[58] = 0;
                    words[59] = 0;
                    if (fold_warp + 112 < NUM_SPLIT) {
                        lses[14] = partial_LSE_ptr[slot_k_13 * 32 + r];
                        {
                            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_13 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[56 + 0] = _uv4_22.x;
                            words[56 + 1] = _uv4_22.y;
                            words[56 + 2] = _uv4_22.z;
                            words[56 + 3] = _uv4_22.w;
                        }
                    }
                    int slot_k_14 = slot_base + fold_warp + 120;
                    lses[15] = -CUDART_INF_F;
                    words[60] = 0;
                    words[61] = 0;
                    words[62] = 0;
                    words[63] = 0;
                    if (fold_warp + 120 < NUM_SPLIT) {
                        lses[15] = partial_LSE_ptr[slot_k_14 * 32 + r];
                        {
                            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_14 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[60 + 0] = _uv4_23.x;
                            words[60 + 1] = _uv4_23.y;
                            words[60 + 2] = _uv4_23.z;
                            words[60 + 3] = _uv4_23.w;
                        }
                    }
                    int slot_k_15 = slot_base + fold_warp + 128;
                    lses[16] = -CUDART_INF_F;
                    words[64] = 0;
                    words[65] = 0;
                    words[66] = 0;
                    words[67] = 0;
                    if (fold_warp + 128 < NUM_SPLIT) {
                        lses[16] = partial_LSE_ptr[slot_k_15 * 32 + r];
                        {
                            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_15 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[64 + 0] = _uv4_24.x;
                            words[64 + 1] = _uv4_24.y;
                            words[64 + 2] = _uv4_24.z;
                            words[64 + 3] = _uv4_24.w;
                        }
                    }
                    int slot_k_16 = slot_base + fold_warp + 136;
                    lses[17] = -CUDART_INF_F;
                    words[68] = 0;
                    words[69] = 0;
                    words[70] = 0;
                    words[71] = 0;
                    if (fold_warp + 136 < NUM_SPLIT) {
                        lses[17] = partial_LSE_ptr[slot_k_16 * 32 + r];
                        {
                            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_16 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[68 + 0] = _uv4_25.x;
                            words[68 + 1] = _uv4_25.y;
                            words[68 + 2] = _uv4_25.z;
                            words[68 + 3] = _uv4_25.w;
                        }
                    }
                    int slot_k_17 = slot_base + fold_warp + 144;
                    lses[18] = -CUDART_INF_F;
                    words[72] = 0;
                    words[73] = 0;
                    words[74] = 0;
                    words[75] = 0;
                    if (fold_warp + 144 < NUM_SPLIT) {
                        lses[18] = partial_LSE_ptr[slot_k_17 * 32 + r];
                        {
                            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(partial_O_ptr + ((slot_k_17 * 32 + r) * HEAD_DIM + lane_f * 8) + 0);
                            words[72 + 0] = _uv4_26.x;
                            words[72 + 1] = _uv4_26.y;
                            words[72 + 2] = _uv4_26.z;
                            words[72 + 3] = _uv4_26.w;
                        }
                    }
                    float warp_max = lses[0];
                    warp_max = max_noftz(warp_max, lses[1]);
                    warp_max = max_noftz(warp_max, lses[2]);
                    warp_max = max_noftz(warp_max, lses[3]);
                    warp_max = max_noftz(warp_max, lses[4]);
                    warp_max = max_noftz(warp_max, lses[5]);
                    warp_max = max_noftz(warp_max, lses[6]);
                    warp_max = max_noftz(warp_max, lses[7]);
                    warp_max = max_noftz(warp_max, lses[8]);
                    warp_max = max_noftz(warp_max, lses[9]);
                    warp_max = max_noftz(warp_max, lses[10]);
                    warp_max = max_noftz(warp_max, lses[11]);
                    warp_max = max_noftz(warp_max, lses[12]);
                    warp_max = max_noftz(warp_max, lses[13]);
                    warp_max = max_noftz(warp_max, lses[14]);
                    warp_max = max_noftz(warp_max, lses[15]);
                    warp_max = max_noftz(warp_max, lses[16]);
                    warp_max = max_noftz(warp_max, lses[17]);
                    warp_max = max_noftz(warp_max, lses[18]);
                    if (lane_f == 0) {
                        smem_red[fold_warp] = warp_max;
                    }
                    asm volatile("barrier.sync 11, 256;" ::: "memory");
                    float merged_max = smem_red[0];
                    #pragma unroll
                    for (int w = 1; w < 8; w++) {
                        merged_max = max_noftz(merged_max, smem_red[w]);
                    }
                    float warp_sum = 0.0f;
                    float acc[8];
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    float w_k = 0.0f;
                    if (lses[0] != -CUDART_INF_F) {
                        float _exp2_27 = approx_exp2(lses[0] - merged_max);
                        w_k = _exp2_27;
                    }
                    warp_sum = warp_sum + w_k;
                    float _unpack_bf16_0[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[_pair]) << 16);
                        (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[1 + _pair]) << 16);
                        (&_unpack_bf16_0[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[1 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) << 16);
                        (&_unpack_bf16_0[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_0[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[3 + _pair]) << 16);
                        (&_unpack_bf16_0[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[3 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        acc[j_1] = __fmaf_rn(_unpack_bf16_0[j_1], w_k, acc[j_1]);
                    }
                    float w_k_18 = 0.0f;
                    if (lses[1] != -CUDART_INF_F) {
                        float _exp2_28 = approx_exp2(lses[1] - merged_max);
                        w_k_18 = _exp2_28;
                    }
                    warp_sum = warp_sum + w_k_18;
                    float _unpack_bf16_1[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[4 + _pair]) << 16);
                        (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[4 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[5 + _pair]) << 16);
                        (&_unpack_bf16_1[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[5 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[6 + _pair]) << 16);
                        (&_unpack_bf16_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[6 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_1[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[7 + _pair]) << 16);
                        (&_unpack_bf16_1[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[7 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        acc[j_2] = __fmaf_rn(_unpack_bf16_1[j_2], w_k_18, acc[j_2]);
                    }
                    float w_k_19 = 0.0f;
                    if (lses[2] != -CUDART_INF_F) {
                        float _exp2_29 = approx_exp2(lses[2] - merged_max);
                        w_k_19 = _exp2_29;
                    }
                    warp_sum = warp_sum + w_k_19;
                    float _unpack_bf16_2[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[8 + _pair]) << 16);
                        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[8 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[9 + _pair]) << 16);
                        (&_unpack_bf16_2[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[9 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[10 + _pair]) << 16);
                        (&_unpack_bf16_2[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[10 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_2[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[11 + _pair]) << 16);
                        (&_unpack_bf16_2[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[11 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 8; j_3++) {
                        acc[j_3] = __fmaf_rn(_unpack_bf16_2[j_3], w_k_19, acc[j_3]);
                    }
                    float w_k_20 = 0.0f;
                    if (lses[3] != -CUDART_INF_F) {
                        float _exp2_30 = approx_exp2(lses[3] - merged_max);
                        w_k_20 = _exp2_30;
                    }
                    warp_sum = warp_sum + w_k_20;
                    float _unpack_bf16_3[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[12 + _pair]) << 16);
                        (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[12 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[13 + _pair]) << 16);
                        (&_unpack_bf16_3[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[13 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[14 + _pair]) << 16);
                        (&_unpack_bf16_3[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[14 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_3[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[15 + _pair]) << 16);
                        (&_unpack_bf16_3[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[15 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 8; j_4++) {
                        acc[j_4] = __fmaf_rn(_unpack_bf16_3[j_4], w_k_20, acc[j_4]);
                    }
                    float w_k_21 = 0.0f;
                    if (lses[4] != -CUDART_INF_F) {
                        float _exp2_31 = approx_exp2(lses[4] - merged_max);
                        w_k_21 = _exp2_31;
                    }
                    warp_sum = warp_sum + w_k_21;
                    float _unpack_bf16_4[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[16 + _pair]) << 16);
                        (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[16 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[17 + _pair]) << 16);
                        (&_unpack_bf16_4[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[17 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[18 + _pair]) << 16);
                        (&_unpack_bf16_4[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[18 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_4[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[19 + _pair]) << 16);
                        (&_unpack_bf16_4[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[19 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 8; j_5++) {
                        acc[j_5] = __fmaf_rn(_unpack_bf16_4[j_5], w_k_21, acc[j_5]);
                    }
                    float w_k_22 = 0.0f;
                    if (lses[5] != -CUDART_INF_F) {
                        float _exp2_32 = approx_exp2(lses[5] - merged_max);
                        w_k_22 = _exp2_32;
                    }
                    warp_sum = warp_sum + w_k_22;
                    float _unpack_bf16_5[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[20 + _pair]) << 16);
                        (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[20 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[21 + _pair]) << 16);
                        (&_unpack_bf16_5[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[21 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[22 + _pair]) << 16);
                        (&_unpack_bf16_5[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[22 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_5[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[23 + _pair]) << 16);
                        (&_unpack_bf16_5[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[23 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        acc[j_6] = __fmaf_rn(_unpack_bf16_5[j_6], w_k_22, acc[j_6]);
                    }
                    float w_k_23 = 0.0f;
                    if (lses[6] != -CUDART_INF_F) {
                        float _exp2_33 = approx_exp2(lses[6] - merged_max);
                        w_k_23 = _exp2_33;
                    }
                    warp_sum = warp_sum + w_k_23;
                    float _unpack_bf16_6[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[24 + _pair]) << 16);
                        (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[24 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[25 + _pair]) << 16);
                        (&_unpack_bf16_6[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[25 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[26 + _pair]) << 16);
                        (&_unpack_bf16_6[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[26 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_6[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[27 + _pair]) << 16);
                        (&_unpack_bf16_6[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[27 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 8; j_7++) {
                        acc[j_7] = __fmaf_rn(_unpack_bf16_6[j_7], w_k_23, acc[j_7]);
                    }
                    float w_k_24 = 0.0f;
                    if (lses[7] != -CUDART_INF_F) {
                        float _exp2_34 = approx_exp2(lses[7] - merged_max);
                        w_k_24 = _exp2_34;
                    }
                    warp_sum = warp_sum + w_k_24;
                    float _unpack_bf16_7[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[28 + _pair]) << 16);
                        (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[28 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[29 + _pair]) << 16);
                        (&_unpack_bf16_7[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[29 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[30 + _pair]) << 16);
                        (&_unpack_bf16_7[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[30 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_7[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[31 + _pair]) << 16);
                        (&_unpack_bf16_7[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[31 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 8; j_8++) {
                        acc[j_8] = __fmaf_rn(_unpack_bf16_7[j_8], w_k_24, acc[j_8]);
                    }
                    float w_k_25 = 0.0f;
                    if (lses[8] != -CUDART_INF_F) {
                        float _exp2_35 = approx_exp2(lses[8] - merged_max);
                        w_k_25 = _exp2_35;
                    }
                    warp_sum = warp_sum + w_k_25;
                    float _unpack_bf16_8[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[32 + _pair]) << 16);
                        (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[32 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[33 + _pair]) << 16);
                        (&_unpack_bf16_8[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[33 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[34 + _pair]) << 16);
                        (&_unpack_bf16_8[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[34 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_8[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[35 + _pair]) << 16);
                        (&_unpack_bf16_8[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[35 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 8; j_9++) {
                        acc[j_9] = __fmaf_rn(_unpack_bf16_8[j_9], w_k_25, acc[j_9]);
                    }
                    float w_k_26 = 0.0f;
                    if (lses[9] != -CUDART_INF_F) {
                        float _exp2_36 = approx_exp2(lses[9] - merged_max);
                        w_k_26 = _exp2_36;
                    }
                    warp_sum = warp_sum + w_k_26;
                    float _unpack_bf16_9[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[36 + _pair]) << 16);
                        (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[36 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_9[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[37 + _pair]) << 16);
                        (&_unpack_bf16_9[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[37 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_9[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[38 + _pair]) << 16);
                        (&_unpack_bf16_9[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[38 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_9[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[39 + _pair]) << 16);
                        (&_unpack_bf16_9[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[39 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 8; j_10++) {
                        acc[j_10] = __fmaf_rn(_unpack_bf16_9[j_10], w_k_26, acc[j_10]);
                    }
                    float w_k_27 = 0.0f;
                    if (lses[10] != -CUDART_INF_F) {
                        float _exp2_37 = approx_exp2(lses[10] - merged_max);
                        w_k_27 = _exp2_37;
                    }
                    warp_sum = warp_sum + w_k_27;
                    float _unpack_bf16_10[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[40 + _pair]) << 16);
                        (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[40 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_10[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[41 + _pair]) << 16);
                        (&_unpack_bf16_10[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[41 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_10[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[42 + _pair]) << 16);
                        (&_unpack_bf16_10[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[42 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_10[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[43 + _pair]) << 16);
                        (&_unpack_bf16_10[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[43 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 8; j_11++) {
                        acc[j_11] = __fmaf_rn(_unpack_bf16_10[j_11], w_k_27, acc[j_11]);
                    }
                    float w_k_28 = 0.0f;
                    if (lses[11] != -CUDART_INF_F) {
                        float _exp2_38 = approx_exp2(lses[11] - merged_max);
                        w_k_28 = _exp2_38;
                    }
                    warp_sum = warp_sum + w_k_28;
                    float _unpack_bf16_11[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[44 + _pair]) << 16);
                        (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[44 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_11[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[45 + _pair]) << 16);
                        (&_unpack_bf16_11[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[45 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_11[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[46 + _pair]) << 16);
                        (&_unpack_bf16_11[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[46 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_11[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[47 + _pair]) << 16);
                        (&_unpack_bf16_11[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[47 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_12 = 0; j_12 < 8; j_12++) {
                        acc[j_12] = __fmaf_rn(_unpack_bf16_11[j_12], w_k_28, acc[j_12]);
                    }
                    float w_k_29 = 0.0f;
                    if (lses[12] != -CUDART_INF_F) {
                        float _exp2_39 = approx_exp2(lses[12] - merged_max);
                        w_k_29 = _exp2_39;
                    }
                    warp_sum = warp_sum + w_k_29;
                    float _unpack_bf16_12[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[48 + _pair]) << 16);
                        (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[48 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_12[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[49 + _pair]) << 16);
                        (&_unpack_bf16_12[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[49 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_12[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[50 + _pair]) << 16);
                        (&_unpack_bf16_12[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[50 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_12[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[51 + _pair]) << 16);
                        (&_unpack_bf16_12[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[51 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 8; j_13++) {
                        acc[j_13] = __fmaf_rn(_unpack_bf16_12[j_13], w_k_29, acc[j_13]);
                    }
                    float w_k_30 = 0.0f;
                    if (lses[13] != -CUDART_INF_F) {
                        float _exp2_40 = approx_exp2(lses[13] - merged_max);
                        w_k_30 = _exp2_40;
                    }
                    warp_sum = warp_sum + w_k_30;
                    float _unpack_bf16_13[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[52 + _pair]) << 16);
                        (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[52 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_13[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[53 + _pair]) << 16);
                        (&_unpack_bf16_13[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[53 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_13[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[54 + _pair]) << 16);
                        (&_unpack_bf16_13[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[54 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_13[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[55 + _pair]) << 16);
                        (&_unpack_bf16_13[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[55 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_14 = 0; j_14 < 8; j_14++) {
                        acc[j_14] = __fmaf_rn(_unpack_bf16_13[j_14], w_k_30, acc[j_14]);
                    }
                    float w_k_31 = 0.0f;
                    if (lses[14] != -CUDART_INF_F) {
                        float _exp2_41 = approx_exp2(lses[14] - merged_max);
                        w_k_31 = _exp2_41;
                    }
                    warp_sum = warp_sum + w_k_31;
                    float _unpack_bf16_14[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[56 + _pair]) << 16);
                        (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[56 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_14[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[57 + _pair]) << 16);
                        (&_unpack_bf16_14[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[57 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_14[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[58 + _pair]) << 16);
                        (&_unpack_bf16_14[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[58 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_14[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[59 + _pair]) << 16);
                        (&_unpack_bf16_14[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[59 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 8; j_15++) {
                        acc[j_15] = __fmaf_rn(_unpack_bf16_14[j_15], w_k_31, acc[j_15]);
                    }
                    float w_k_32 = 0.0f;
                    if (lses[15] != -CUDART_INF_F) {
                        float _exp2_42 = approx_exp2(lses[15] - merged_max);
                        w_k_32 = _exp2_42;
                    }
                    warp_sum = warp_sum + w_k_32;
                    float _unpack_bf16_15[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[60 + _pair]) << 16);
                        (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[60 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_15[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[61 + _pair]) << 16);
                        (&_unpack_bf16_15[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[61 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_15[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[62 + _pair]) << 16);
                        (&_unpack_bf16_15[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[62 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_15[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[63 + _pair]) << 16);
                        (&_unpack_bf16_15[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[63 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 8; j_16++) {
                        acc[j_16] = __fmaf_rn(_unpack_bf16_15[j_16], w_k_32, acc[j_16]);
                    }
                    float w_k_33 = 0.0f;
                    if (lses[16] != -CUDART_INF_F) {
                        float _exp2_43 = approx_exp2(lses[16] - merged_max);
                        w_k_33 = _exp2_43;
                    }
                    warp_sum = warp_sum + w_k_33;
                    float _unpack_bf16_16[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[64 + _pair]) << 16);
                        (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[64 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_16[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[65 + _pair]) << 16);
                        (&_unpack_bf16_16[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[65 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_16[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[66 + _pair]) << 16);
                        (&_unpack_bf16_16[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[66 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_16[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[67 + _pair]) << 16);
                        (&_unpack_bf16_16[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[67 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 8; j_17++) {
                        acc[j_17] = __fmaf_rn(_unpack_bf16_16[j_17], w_k_33, acc[j_17]);
                    }
                    float w_k_34 = 0.0f;
                    if (lses[17] != -CUDART_INF_F) {
                        float _exp2_44 = approx_exp2(lses[17] - merged_max);
                        w_k_34 = _exp2_44;
                    }
                    warp_sum = warp_sum + w_k_34;
                    float _unpack_bf16_17[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[68 + _pair]) << 16);
                        (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[68 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_17[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[69 + _pair]) << 16);
                        (&_unpack_bf16_17[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[69 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_17[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[70 + _pair]) << 16);
                        (&_unpack_bf16_17[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[70 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_17[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[71 + _pair]) << 16);
                        (&_unpack_bf16_17[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[71 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 8; j_18++) {
                        acc[j_18] = __fmaf_rn(_unpack_bf16_17[j_18], w_k_34, acc[j_18]);
                    }
                    float w_k_35 = 0.0f;
                    if (lses[18] != -CUDART_INF_F) {
                        float _exp2_45 = approx_exp2(lses[18] - merged_max);
                        w_k_35 = _exp2_45;
                    }
                    warp_sum = warp_sum + w_k_35;
                    float _unpack_bf16_18[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[72 + _pair]) << 16);
                        (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[72 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_18[2 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[73 + _pair]) << 16);
                        (&_unpack_bf16_18[2 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[73 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_18[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[74 + _pair]) << 16);
                        (&_unpack_bf16_18[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[74 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 1; _pair++) {
                        (&_unpack_bf16_18[6 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[75 + _pair]) << 16);
                        (&_unpack_bf16_18[6 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[75 + _pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 8; j_19++) {
                        acc[j_19] = __fmaf_rn(_unpack_bf16_18[j_19], w_k_35, acc[j_19]);
                    }
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 8; j_20++) {
                        smem_st_a[(fold_warp * 32 + lane_f) * 8 + j_20] = acc[j_20];
                    }
                    if (lane_f == 0) {
                        smem_w[fold_warp] = warp_sum;
                    }
                    asm volatile("barrier.sync 11, 256;" ::: "memory");
                    if (fold_warp == 0) {
                        float weight_sum = smem_w[0];
                        #pragma unroll
                        for (int w_1 = 1; w_1 < 8; w_1++) {
                            weight_sum = weight_sum + smem_w[w_1];
                        }
                        float _rcp_46 = approx_rcp(weight_sum);
                        float inv_sum = ((weight_sum > 0.0f) ? _rcp_46 : 0.0f);
                        float out[8];
                        #pragma unroll
                        for (int j_21 = 0; j_21 < 8; j_21++) {
                            float total_j = smem_st_a[lane_f * 8 + j_21];
                            #pragma unroll
                            for (int g = 1; g < 8; g++) {
                                total_j = total_j + smem_st_a[(g * 32 + lane_f) * 8 + j_21];
                            }
                            out[j_21] = total_j * inv_sum;
                        }
                        int j_row = r / GROUP;
                        int h_row = r % GROUP;
                        int q_row = batch_idx * Q_LEN + j_row;
                        int q_head = kv_head_idx * GROUP + h_row;
                        int o_idx = (q_row * num_q_heads + q_head) * HEAD_DIM + lane_f * 8;
                        unsigned int _packed_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O_ptr + o_idx)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                        if (lane_f == 0) {
                            float merged_lse = -CUDART_INF_F;
                            if (weight_sum > 0.0f) {
                                float _log2_47;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_47) : "f"(weight_sum));
                                merged_lse = merged_max + _log2_47;
                            }
                            *(reinterpret_cast<float*>(LSE_ptr + (q_row * num_q_heads + q_head)) + (0)) = merged_lse;
                        }
                    }
                    asm volatile("barrier.sync 11, 256;" ::: "memory");
                }
            }
            if (flat_tid_s == 0) {
                unsigned int old_done = atomicAdd(&counters[tile_idx * 2 + 1], 1);
                if (old_done + 1 == (unsigned int)NUM_SPLIT) {
                    *(reinterpret_cast<unsigned int*>(counters + (tile_idx * 2)) + (0)) = 0;
                    *(reinterpret_cast<unsigned int*>(counters + (tile_idx * 2 + 1)) + (0)) = 0;
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
                    for (int c_2 = 0; c_2 < 32; c_2++) {
                        need = need | ((smem_scale[sc_off + c_2] != 1.0f) ? 1 : 0);
                    }
                    int _warp_vote_0 = __any_sync(0xFFFFFFFF, need != 0);
                    if (_warp_vote_0 != 0) {
                        #pragma unroll
                        for (int half = 0; half < 2; half++) {
                            int o_base = taddr + (unsigned int)(((half == 0) ? 128 : 192)) + (unsigned int)corr_row;
                            #pragma unroll
                            for (int chunk = 0; chunk < 1; chunk++) {
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
            unsigned int _phase_stats_full_0 = 0;
            mbarrier_wait(stats_full_addr, _phase_stats_full_0);
            _phase_stats_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int slot = work_1;
            int o_hi_base = taddr + 128 + (unsigned int)corr_row;
            #pragma unroll
            for (int chunk_1 = 0; chunk_1 < 1; chunk_1++) {
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
                    const int r_c = chunk_1 * 32 + c_4;
                    float val_c = o_epi[c_4] * inv[c_4];
                    if (cnt_1 == 0) {
                        val_c = 0.0f;
                    }
                    *(reinterpret_cast<__nv_bfloat16*>(partial_O_ptr + ((slot * 32 + r_c) * HEAD_DIM + d_idx)) + (0)) = (__nv_bfloat16)val_c;
                }
            }
            if (warp_in_wg_c == 0) {
                #pragma unroll
                for (int chunk_2 = 0; chunk_2 < 1; chunk_2++) {
                    int r_l = chunk_2 * 32 + lane;
                    float m_l = smem_max[r_l];
                    float s_l = smem_sum[r_l];
                    float lse_l = -CUDART_INF_F;
                    if (s_l > 0.0f && m_l > -CUDART_INF_F) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(s_l));
                        lse_l = __fmaf_rn(m_l, softmax_scale_log2, _log2_1);
                    }
                    *(reinterpret_cast<float*>(partial_LSE_ptr + (slot * 32 + r_l)) + (0)) = lse_l;
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            int flat_tid_c = 256 + (warp - 8) * 32 + lane;
            asm volatile("fence.release.gpu;" ::: "memory");
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            if (flat_tid_c == 0) {
                unsigned int arrived_1;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(arrived_1) : "l"(&counters[tile_idx_1 * 2]), "r"(static_cast<uint32_t>(1)) : "memory");
                if (arrived_1 + 1 != (unsigned int)NUM_SPLIT) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(counters) + (tile_idx_1 * 2))) : "memory");
                    } while (_acquire_observed < static_cast<unsigned int>(NUM_SPLIT));
                    }
                }
            }
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            asm volatile("fence.acquire.gpu;" ::: "memory");
            if (flat_tid_c == 0) {
                unsigned int old_done_1 = atomicAdd(&counters[tile_idx_1 * 2 + 1], 1);
                if (old_done_1 + 1 == (unsigned int)NUM_SPLIT) {
                    *(reinterpret_cast<unsigned int*>(counters + (tile_idx_1 * 2)) + (0)) = 0;
                    *(reinterpret_cast<unsigned int*>(counters + (tile_idx_1 * 2 + 1)) + (0)) = 0;
                }
            }
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
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_1 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
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
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 250U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                        }
                        int _mma_a_lo_3 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (nstage) * 2048);
                        int _mma_b_lo_3 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                            uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 250U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
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
                    int _mma_b_lo_4 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, ((first_pv_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                    }
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, ((first_pv_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
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
                mbarrier_arrive_expect_tx(q_full_addr, 32 * HEAD_DIM * 2);
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
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
