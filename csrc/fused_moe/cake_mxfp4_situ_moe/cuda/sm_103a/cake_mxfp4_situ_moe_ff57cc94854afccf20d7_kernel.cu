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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 76
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 64
#define TMEM_ACC1_OFFSET 32
#define TMEM_SF_A1_OFFSET 68
#define TMEM_SF_B_OFFSET 72
#define NUM_PAB_STAGES 5
#define NUM_PB_STAGES 5
#define NUM_PACC_STAGES 2
#define NUM_PTILE_STAGES 8
#define SMEM_A_OFF 1024
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 164864
#define SMEM_B_STAGE_BYTES 2048
#define SMEM_B_STRIDE 2048
#define SMEM_SFA_OFF 175104
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 180224
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_SINFO_OFF 182784
#define SMEM_SINFO_STAGE_BYTES 224
#define SMEM_SINFO_STRIDE 224
#define SMEM_STOK_OFF 183008
#define SMEM_STOK_STAGE_BYTES 512
#define SMEM_STOK_STRIDE 512
#define SMEM_SSCALE_OFF 183520
#define SMEM_SSCALE_STAGE_BYTES 544
#define SMEM_SSCALE_STRIDE 544
#define SMEM_SEXCH_OFF 184064
#define SMEM_SEXCH_STAGE_BYTES 4352
#define SMEM_SEXCH_STRIDE 4352
#define SMEM_TOTAL 188416
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}



__device__ __forceinline__ void tcgen05_mma_mxf8_bs_elect(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


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


extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mxfp4_situ_moe_ff57cc94854afccf20d7(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ B, uint8_t* __restrict__ SFB, __nv_bfloat16* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, int* __restrict__ tile_idx_to_row_group, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int num_m_tiles, int group_capacity, int k_tiles, int k_cols, int sf_cols, int out_cols, int top_k, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, uint8_t* __restrict__ act_sf, float* __restrict__ zero_buf, int zero_words, int num_rows_b, int act_cols, int act_sf_cols, int* __restrict__ dbg)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 40)
    #define b_full_addr (mbar_base + 80)
    #define b_free_addr (mbar_base + 120)
    #define acc_full_addr (mbar_base + 160)
    #define acc_free_addr (mbar_base + 176)
    #define tile_full_addr (mbar_base + 192)
    #define tile_free_addr (mbar_base + 256)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a_addr = smem + 1024;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int b_addr = smem + 164864;
    uint8_t* sfa = reinterpret_cast<uint8_t*>(smem_raw + 175104);
    const int sfa_addr = smem + 175104;
    uint8_t* sfb = reinterpret_cast<uint8_t*>(smem_raw + 180224);
    const int sfb_addr = smem + 180224;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 182784);
    const int sinfo_addr = smem + 182784;
    int* stok = reinterpret_cast<int*>(smem_raw + 183008);
    const int stok_addr = smem + 183008;
    float* sscale = reinterpret_cast<float*>(smem_raw + 183520);
    const int sscale_addr = smem + 183520;
    float* sexch = reinterpret_cast<float*>(smem_raw + 184064);
    const int sexch_addr = smem + 184064;
    int _mma_base_lo_0 = ((a_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_1 = ((b_addr) >> 4) & 0x3FFF;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 40 barriers)
    // Mbarriers at smem_raw[0..320)

    if (warp == 0) {
        // --- pipeline 'pab' ---
        // ab_full: 5 barriers, init_count=1
        // ab_free: 5 barriers, init_count=1
        // --- pipeline 'pb' ---
        // b_full: 5 barriers, init_count=32
        // b_free: 5 barriers, init_count=1
        // --- pipeline 'pacc' ---
        // acc_full: 2 barriers, init_count=1
        // acc_free: 2 barriers, init_count=128
        // --- pipeline 'ptile' ---
        // tile_full: 8 barriers, init_count=32
        // tile_free: 8 barriers, init_count=224
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(24), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(22), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(15), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        if (lane < 8) {
            mbarrier_init(smem + 256 + lane * 8, 224);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (128 columns, 76 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 320);
    if (warp == 0) {
        int _tmem_hold = smem + 320;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 64;
    const int tmem_acc1 = taddr + 32;
    const int tmem_sf_a1 = taddr + 68;
    const int tmem_sf_b = taddr + 72;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int nrec_epilogue = 0;
            int nst_epilogue = 0;
            int epi_tidx = tid;
            int lane_0 = lane;
            int epi_warp = warp;
            unsigned int acc_stage = 0;
            unsigned int tile_stage = 0;
            int info[7];
            int cur_tok[16];
            float cur_scale[16];
            float meta_alpha = 0.0f;
            int rec_m_tile = 0;
            float vals[16];
            unsigned int _phase_tile_full = 0;
            mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
            info[0] = sinfo[tile_stage * 7];
            info[1] = sinfo[tile_stage * 7 + 1];
            info[2] = sinfo[tile_stage * 7 + 2];
            info[3] = sinfo[tile_stage * 7 + 3];
            info[4] = sinfo[tile_stage * 7 + 4];
            info[5] = sinfo[tile_stage * 7 + 5];
            info[6] = sinfo[tile_stage * 7 + 6];
            rec_m_tile = sinfo[tile_stage * 7];
            meta_alpha = sscale[tile_stage * 17 + 16];
            cur_tok[0] = stok[tile_stage * 16];
            cur_scale[0] = sscale[tile_stage * 17];
            cur_tok[1] = stok[tile_stage * 16 + 1];
            cur_scale[1] = sscale[tile_stage * 17 + 1];
            cur_tok[2] = stok[tile_stage * 16 + 2];
            cur_scale[2] = sscale[tile_stage * 17 + 2];
            cur_tok[3] = stok[tile_stage * 16 + 3];
            cur_scale[3] = sscale[tile_stage * 17 + 3];
            cur_tok[4] = stok[tile_stage * 16 + 4];
            cur_scale[4] = sscale[tile_stage * 17 + 4];
            cur_tok[5] = stok[tile_stage * 16 + 5];
            cur_scale[5] = sscale[tile_stage * 17 + 5];
            cur_tok[6] = stok[tile_stage * 16 + 6];
            cur_scale[6] = sscale[tile_stage * 17 + 6];
            cur_tok[7] = stok[tile_stage * 16 + 7];
            cur_scale[7] = sscale[tile_stage * 17 + 7];
            cur_tok[8] = stok[tile_stage * 16 + 8];
            cur_scale[8] = sscale[tile_stage * 17 + 8];
            cur_tok[9] = stok[tile_stage * 16 + 9];
            cur_scale[9] = sscale[tile_stage * 17 + 9];
            cur_tok[10] = stok[tile_stage * 16 + 10];
            cur_scale[10] = sscale[tile_stage * 17 + 10];
            cur_tok[11] = stok[tile_stage * 16 + 11];
            cur_scale[11] = sscale[tile_stage * 17 + 11];
            cur_tok[12] = stok[tile_stage * 16 + 12];
            cur_scale[12] = sscale[tile_stage * 17 + 12];
            cur_tok[13] = stok[tile_stage * 16 + 13];
            cur_scale[13] = sscale[tile_stage * 17 + 13];
            cur_tok[14] = stok[tile_stage * 16 + 14];
            cur_scale[14] = sscale[tile_stage * 17 + 14];
            cur_tok[15] = stok[tile_stage * 16 + 15];
            cur_scale[15] = sscale[tile_stage * 17 + 15];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
            tile_stage += 1;
            if (tile_stage == 8) { tile_stage = 0; _phase_tile_full ^= 1; }
            nrec_epilogue = nrec_epilogue + 1;
            bool is_even_lane = lane_0 % 2 == 0;
            int is_gate_lane = (int)(epi_warp >= 2);
            int exch_row = epi_tidx & 63;
            float inv_fp8_max = 0.002232142857142857f;
            float zero_f32 = 0.0f;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _tile = 0; _tile < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile++) {
                if (info[3] == 0) {
                    break;
                }
                int row_base = info[1] * 16;
                int mn_limit = info[4];
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int m_tile_e = rec_m_tile * 2;
                int h = m_tile_e * 128 + epi_tidx;
                if (m_tile_e < num_m_tiles) {
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[0] = _tmem_load_0[0] * meta_alpha;
                    vals[1] = _tmem_load_0[1] * meta_alpha;
                    vals[2] = _tmem_load_0[2] * meta_alpha;
                    vals[3] = _tmem_load_0[3] * meta_alpha;
                    vals[4] = _tmem_load_0[4] * meta_alpha;
                    vals[5] = _tmem_load_0[5] * meta_alpha;
                    vals[6] = _tmem_load_0[6] * meta_alpha;
                    vals[7] = _tmem_load_0[7] * meta_alpha;
                    vals[8] = _tmem_load_0[8] * meta_alpha;
                    vals[9] = _tmem_load_0[9] * meta_alpha;
                    vals[10] = _tmem_load_0[10] * meta_alpha;
                    vals[11] = _tmem_load_0[11] * meta_alpha;
                    vals[12] = _tmem_load_0[12] * meta_alpha;
                    vals[13] = _tmem_load_0[13] * meta_alpha;
                    vals[14] = _tmem_load_0[14] * meta_alpha;
                    vals[15] = _tmem_load_0[15] * meta_alpha;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    {
                        float v = vals[0] * cur_scale[0];
                        int tok = cur_tok[0];
                        bool col_ok = mn_limit > row_base;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok * out_cols + h])), "f"(v), "r"((unsigned int)(col_ok)) : "memory");
                        float v_0 = vals[1] * cur_scale[1];
                        int tok_1 = cur_tok[1];
                        bool col_ok_2 = mn_limit > row_base + 1;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_1 * out_cols + h])), "f"(v_0), "r"((unsigned int)(col_ok_2)) : "memory");
                        float v_3 = vals[2] * cur_scale[2];
                        int tok_4 = cur_tok[2];
                        bool col_ok_5 = mn_limit > row_base + 2;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_4 * out_cols + h])), "f"(v_3), "r"((unsigned int)(col_ok_5)) : "memory");
                        float v_6 = vals[3] * cur_scale[3];
                        int tok_7 = cur_tok[3];
                        bool col_ok_8 = mn_limit > row_base + 3;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_7 * out_cols + h])), "f"(v_6), "r"((unsigned int)(col_ok_8)) : "memory");
                        float v_9 = vals[4] * cur_scale[4];
                        int tok_10 = cur_tok[4];
                        bool col_ok_11 = mn_limit > row_base + 4;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_10 * out_cols + h])), "f"(v_9), "r"((unsigned int)(col_ok_11)) : "memory");
                        float v_12 = vals[5] * cur_scale[5];
                        int tok_13 = cur_tok[5];
                        bool col_ok_14 = mn_limit > row_base + 5;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_13 * out_cols + h])), "f"(v_12), "r"((unsigned int)(col_ok_14)) : "memory");
                        float v_15 = vals[6] * cur_scale[6];
                        int tok_16 = cur_tok[6];
                        bool col_ok_17 = mn_limit > row_base + 6;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_16 * out_cols + h])), "f"(v_15), "r"((unsigned int)(col_ok_17)) : "memory");
                        float v_18 = vals[7] * cur_scale[7];
                        int tok_19 = cur_tok[7];
                        bool col_ok_20 = mn_limit > row_base + 7;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_19 * out_cols + h])), "f"(v_18), "r"((unsigned int)(col_ok_20)) : "memory");
                        float v_21 = vals[8] * cur_scale[8];
                        int tok_22 = cur_tok[8];
                        bool col_ok_23 = mn_limit > row_base + 8;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_22 * out_cols + h])), "f"(v_21), "r"((unsigned int)(col_ok_23)) : "memory");
                        float v_24 = vals[9] * cur_scale[9];
                        int tok_25 = cur_tok[9];
                        bool col_ok_26 = mn_limit > row_base + 9;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_25 * out_cols + h])), "f"(v_24), "r"((unsigned int)(col_ok_26)) : "memory");
                        float v_27 = vals[10] * cur_scale[10];
                        int tok_28 = cur_tok[10];
                        bool col_ok_29 = mn_limit > row_base + 10;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_28 * out_cols + h])), "f"(v_27), "r"((unsigned int)(col_ok_29)) : "memory");
                        float v_30 = vals[11] * cur_scale[11];
                        int tok_31 = cur_tok[11];
                        bool col_ok_32 = mn_limit > row_base + 11;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_31 * out_cols + h])), "f"(v_30), "r"((unsigned int)(col_ok_32)) : "memory");
                        float v_33 = vals[12] * cur_scale[12];
                        int tok_34 = cur_tok[12];
                        bool col_ok_35 = mn_limit > row_base + 12;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_34 * out_cols + h])), "f"(v_33), "r"((unsigned int)(col_ok_35)) : "memory");
                        float v_36 = vals[13] * cur_scale[13];
                        int tok_37 = cur_tok[13];
                        bool col_ok_38 = mn_limit > row_base + 13;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_37 * out_cols + h])), "f"(v_36), "r"((unsigned int)(col_ok_38)) : "memory");
                        float v_39 = vals[14] * cur_scale[14];
                        int tok_40 = cur_tok[14];
                        bool col_ok_41 = mn_limit > row_base + 14;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_40 * out_cols + h])), "f"(v_39), "r"((unsigned int)(col_ok_41)) : "memory");
                        float v_42 = vals[15] * cur_scale[15];
                        int tok_43 = cur_tok[15];
                        bool col_ok_44 = mn_limit > row_base + 15;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_43 * out_cols + h])), "f"(v_42), "r"((unsigned int)(col_ok_44)) : "memory");
                    }
                }
                int m_tile_e_0 = rec_m_tile * 2 + 1;
                int h_1 = m_tile_e_0 * 128 + epi_tidx;
                if (m_tile_e_0 < num_m_tiles) {
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], taddr + 32 + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[0] = _tmem_load_1[0] * meta_alpha;
                    vals[1] = _tmem_load_1[1] * meta_alpha;
                    vals[2] = _tmem_load_1[2] * meta_alpha;
                    vals[3] = _tmem_load_1[3] * meta_alpha;
                    vals[4] = _tmem_load_1[4] * meta_alpha;
                    vals[5] = _tmem_load_1[5] * meta_alpha;
                    vals[6] = _tmem_load_1[6] * meta_alpha;
                    vals[7] = _tmem_load_1[7] * meta_alpha;
                    vals[8] = _tmem_load_1[8] * meta_alpha;
                    vals[9] = _tmem_load_1[9] * meta_alpha;
                    vals[10] = _tmem_load_1[10] * meta_alpha;
                    vals[11] = _tmem_load_1[11] * meta_alpha;
                    vals[12] = _tmem_load_1[12] * meta_alpha;
                    vals[13] = _tmem_load_1[13] * meta_alpha;
                    vals[14] = _tmem_load_1[14] * meta_alpha;
                    vals[15] = _tmem_load_1[15] * meta_alpha;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    {
                        float v_1 = vals[0] * cur_scale[0];
                        int tok_2 = cur_tok[0];
                        bool col_ok_1 = mn_limit > row_base;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_2 * out_cols + h_1])), "f"(v_1), "r"((unsigned int)(col_ok_1)) : "memory");
                        float v_0_1 = vals[1] * cur_scale[1];
                        int tok_1_1 = cur_tok[1];
                        bool col_ok_2_1 = mn_limit > row_base + 1;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_1_1 * out_cols + h_1])), "f"(v_0_1), "r"((unsigned int)(col_ok_2_1)) : "memory");
                        float v_3_1 = vals[2] * cur_scale[2];
                        int tok_4_1 = cur_tok[2];
                        bool col_ok_5_1 = mn_limit > row_base + 2;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_4_1 * out_cols + h_1])), "f"(v_3_1), "r"((unsigned int)(col_ok_5_1)) : "memory");
                        float v_6_1 = vals[3] * cur_scale[3];
                        int tok_7_1 = cur_tok[3];
                        bool col_ok_8_1 = mn_limit > row_base + 3;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_7_1 * out_cols + h_1])), "f"(v_6_1), "r"((unsigned int)(col_ok_8_1)) : "memory");
                        float v_9_1 = vals[4] * cur_scale[4];
                        int tok_10_1 = cur_tok[4];
                        bool col_ok_11_1 = mn_limit > row_base + 4;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_10_1 * out_cols + h_1])), "f"(v_9_1), "r"((unsigned int)(col_ok_11_1)) : "memory");
                        float v_12_1 = vals[5] * cur_scale[5];
                        int tok_13_1 = cur_tok[5];
                        bool col_ok_14_1 = mn_limit > row_base + 5;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_13_1 * out_cols + h_1])), "f"(v_12_1), "r"((unsigned int)(col_ok_14_1)) : "memory");
                        float v_15_1 = vals[6] * cur_scale[6];
                        int tok_16_1 = cur_tok[6];
                        bool col_ok_17_1 = mn_limit > row_base + 6;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_16_1 * out_cols + h_1])), "f"(v_15_1), "r"((unsigned int)(col_ok_17_1)) : "memory");
                        float v_18_1 = vals[7] * cur_scale[7];
                        int tok_19_1 = cur_tok[7];
                        bool col_ok_20_1 = mn_limit > row_base + 7;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_19_1 * out_cols + h_1])), "f"(v_18_1), "r"((unsigned int)(col_ok_20_1)) : "memory");
                        float v_21_1 = vals[8] * cur_scale[8];
                        int tok_22_1 = cur_tok[8];
                        bool col_ok_23_1 = mn_limit > row_base + 8;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_22_1 * out_cols + h_1])), "f"(v_21_1), "r"((unsigned int)(col_ok_23_1)) : "memory");
                        float v_24_1 = vals[9] * cur_scale[9];
                        int tok_25_1 = cur_tok[9];
                        bool col_ok_26_1 = mn_limit > row_base + 9;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_25_1 * out_cols + h_1])), "f"(v_24_1), "r"((unsigned int)(col_ok_26_1)) : "memory");
                        float v_27_1 = vals[10] * cur_scale[10];
                        int tok_28_1 = cur_tok[10];
                        bool col_ok_29_1 = mn_limit > row_base + 10;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_28_1 * out_cols + h_1])), "f"(v_27_1), "r"((unsigned int)(col_ok_29_1)) : "memory");
                        float v_30_1 = vals[11] * cur_scale[11];
                        int tok_31_1 = cur_tok[11];
                        bool col_ok_32_1 = mn_limit > row_base + 11;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_31_1 * out_cols + h_1])), "f"(v_30_1), "r"((unsigned int)(col_ok_32_1)) : "memory");
                        float v_33_1 = vals[12] * cur_scale[12];
                        int tok_34_1 = cur_tok[12];
                        bool col_ok_35_1 = mn_limit > row_base + 12;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_34_1 * out_cols + h_1])), "f"(v_33_1), "r"((unsigned int)(col_ok_35_1)) : "memory");
                        float v_36_1 = vals[13] * cur_scale[13];
                        int tok_37_1 = cur_tok[13];
                        bool col_ok_38_1 = mn_limit > row_base + 13;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_37_1 * out_cols + h_1])), "f"(v_36_1), "r"((unsigned int)(col_ok_38_1)) : "memory");
                        float v_39_1 = vals[14] * cur_scale[14];
                        int tok_40_1 = cur_tok[14];
                        bool col_ok_41_1 = mn_limit > row_base + 14;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_40_1 * out_cols + h_1])), "f"(v_39_1), "r"((unsigned int)(col_ok_41_1)) : "memory");
                        float v_42_1 = vals[15] * cur_scale[15];
                        int tok_43_1 = cur_tok[15];
                        bool col_ok_44_1 = mn_limit > row_base + 15;
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_43_1 * out_cols + h_1])), "f"(v_42_1), "r"((unsigned int)(col_ok_44_1)) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_free_addr + (acc_stage) * 8);
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_full ^= 1; }
                nst_epilogue = nst_epilogue + 1;
                mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
                info[0] = sinfo[tile_stage * 7];
                info[1] = sinfo[tile_stage * 7 + 1];
                info[2] = sinfo[tile_stage * 7 + 2];
                info[3] = sinfo[tile_stage * 7 + 3];
                info[4] = sinfo[tile_stage * 7 + 4];
                info[5] = sinfo[tile_stage * 7 + 5];
                info[6] = sinfo[tile_stage * 7 + 6];
                rec_m_tile = sinfo[tile_stage * 7];
                meta_alpha = sscale[tile_stage * 17 + 16];
                cur_tok[0] = stok[tile_stage * 16];
                cur_scale[0] = sscale[tile_stage * 17];
                cur_tok[1] = stok[tile_stage * 16 + 1];
                cur_scale[1] = sscale[tile_stage * 17 + 1];
                cur_tok[2] = stok[tile_stage * 16 + 2];
                cur_scale[2] = sscale[tile_stage * 17 + 2];
                cur_tok[3] = stok[tile_stage * 16 + 3];
                cur_scale[3] = sscale[tile_stage * 17 + 3];
                cur_tok[4] = stok[tile_stage * 16 + 4];
                cur_scale[4] = sscale[tile_stage * 17 + 4];
                cur_tok[5] = stok[tile_stage * 16 + 5];
                cur_scale[5] = sscale[tile_stage * 17 + 5];
                cur_tok[6] = stok[tile_stage * 16 + 6];
                cur_scale[6] = sscale[tile_stage * 17 + 6];
                cur_tok[7] = stok[tile_stage * 16 + 7];
                cur_scale[7] = sscale[tile_stage * 17 + 7];
                cur_tok[8] = stok[tile_stage * 16 + 8];
                cur_scale[8] = sscale[tile_stage * 17 + 8];
                cur_tok[9] = stok[tile_stage * 16 + 9];
                cur_scale[9] = sscale[tile_stage * 17 + 9];
                cur_tok[10] = stok[tile_stage * 16 + 10];
                cur_scale[10] = sscale[tile_stage * 17 + 10];
                cur_tok[11] = stok[tile_stage * 16 + 11];
                cur_scale[11] = sscale[tile_stage * 17 + 11];
                cur_tok[12] = stok[tile_stage * 16 + 12];
                cur_scale[12] = sscale[tile_stage * 17 + 12];
                cur_tok[13] = stok[tile_stage * 16 + 13];
                cur_scale[13] = sscale[tile_stage * 17 + 13];
                cur_tok[14] = stok[tile_stage * 16 + 14];
                cur_scale[14] = sscale[tile_stage * 17 + 14];
                cur_tok[15] = stok[tile_stage * 16 + 15];
                cur_scale[15] = sscale[tile_stage * 17 + 15];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
                tile_stage += 1;
                if (tile_stage == 8) { tile_stage = 0; _phase_tile_full ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int nrec_mma = 0;
            int nst_mma = 0;
            unsigned int sa = 0;
            unsigned int sb = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int tile_stage_1 = 0;
            unsigned int pha = 0;
            unsigned int ab_tok = 1;
            int info_1[7];
            unsigned int _phase_tile_full_1 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
            info_1[0] = sinfo[tile_stage_1 * 7];
            info_1[1] = sinfo[tile_stage_1 * 7 + 1];
            info_1[2] = sinfo[tile_stage_1 * 7 + 2];
            info_1[3] = sinfo[tile_stage_1 * 7 + 3];
            info_1[4] = sinfo[tile_stage_1 * 7 + 4];
            info_1[5] = sinfo[tile_stage_1 * 7 + 5];
            info_1[6] = sinfo[tile_stage_1 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
            tile_stage_1 += 1;
            if (tile_stage_1 == 8) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
            nrec_mma = nrec_mma + 1;
            unsigned int _phase_acc_free = 1;
            unsigned int _phase_b_full = 0;
            #pragma unroll 1
            for (int _tile_1 = 0; _tile_1 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_1++) {
                if (info_1[3] == 0) {
                    break;
                }
                uint32_t _mbar_token_0 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                ab_tok = _mbar_token_0;
                mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 32)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2) * 32)));
                    }
                    int _mma_a_lo_0 = (_mma_base_lo_0) + (sa * 2) * 1024;
                    int _mma_b_lo_0 = (_mma_base_lo_1) + (sb) * 128;
                    {
                        uint32_t a_desc_lo = (uint32_t)_mma_a_lo_0;
                        uint32_t b_desc_lo = (uint32_t)_mma_b_lo_0;

                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                            0x8840280U, tmem_sf_a, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                            0x28840290U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                            0x488402a0U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                            0x688402b0U, tmem_sf_a, tmem_sf_b, 1);
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a1, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2 + 1) * 32)));
                    }
                    int _mma_a_lo_1 = (_mma_base_lo_0) + (sa * 2 + 1) * 1024;
                    int _mma_b_lo_1 = (_mma_base_lo_1) + (sb) * 128;
                    {
                        uint32_t a_desc_lo = (uint32_t)_mma_a_lo_1;
                        uint32_t b_desc_lo = (uint32_t)_mma_b_lo_1;

                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                            0x8840280U, tmem_sf_a1, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                            0x28840290U, tmem_sf_a1, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                            0x488402a0U, tmem_sf_a1, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                            0x688402b0U, tmem_sf_a1, tmem_sf_b, 1);
                    }
                }
                elect_commit(ab_free_addr + (sa) * 8);
                elect_commit(b_free_addr + (sb) * 8);
                sa += 1;
                if (sa == 5) { sa = 0; pha ^= 1; }
                sb += 1;
                if (sb == 5) { sb = 0; _phase_b_full ^= 1; }
                nst_mma = nst_mma + 1;
                ab_tok = 1;
                if (k_tiles > 1) {
                    uint32_t _mbar_token_1 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                    ab_tok = _mbar_token_1;
                }
                #pragma unroll 1
                for (int k = 1; k < k_tiles; k++) {
                    mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                    mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 32)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2) * 32)));
                        }
                        int _mma_a_lo_2 = (_mma_base_lo_0) + (sa * 2) * 1024;
                        int _mma_b_lo_2 = (_mma_base_lo_1) + (sb) * 128;
                        {
                            uint32_t a_desc_lo = (uint32_t)_mma_a_lo_2;
                            uint32_t b_desc_lo = (uint32_t)_mma_b_lo_2;

                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                                0x8840280U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                                0x28840290U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                                0x488402a0U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                                0x688402b0U, tmem_sf_a, tmem_sf_b, 1);
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a1, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2 + 1) * 32)));
                        }
                        int _mma_a_lo_3 = (_mma_base_lo_0) + (sa * 2 + 1) * 1024;
                        int _mma_b_lo_3 = (_mma_base_lo_1) + (sb) * 128;
                        {
                            uint32_t a_desc_lo = (uint32_t)_mma_a_lo_3;
                            uint32_t b_desc_lo = (uint32_t)_mma_b_lo_3;

                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                                0x8840280U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                                0x28840290U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                                0x488402a0U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 16)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                                0x688402b0U, tmem_sf_a1, tmem_sf_b, 1);
                        }
                    }
                    elect_commit(ab_free_addr + (sa) * 8);
                    elect_commit(b_free_addr + (sb) * 8);
                    sa += 1;
                    if (sa == 5) { sa = 0; pha ^= 1; }
                    sb += 1;
                    if (sb == 5) { sb = 0; _phase_b_full ^= 1; }
                    nst_mma = nst_mma + 1;
                    ab_tok = 1;
                    if (k + 1 < k_tiles) {
                        uint32_t _mbar_token_2 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                        ab_tok = _mbar_token_2;
                    }
                }
                elect_commit(acc_full_addr + (acc_stage_1) * 8);
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_free ^= 1; }
                nst_mma = nst_mma + 1;
                mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
                info_1[0] = sinfo[tile_stage_1 * 7];
                info_1[1] = sinfo[tile_stage_1 * 7 + 1];
                info_1[2] = sinfo[tile_stage_1 * 7 + 2];
                info_1[3] = sinfo[tile_stage_1 * 7 + 3];
                info_1[4] = sinfo[tile_stage_1 * 7 + 4];
                info_1[5] = sinfo[tile_stage_1 * 7 + 5];
                info_1[6] = sinfo[tile_stage_1 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
                tile_stage_1 += 1;
                if (tile_stage_1 == 8) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
                nrec_mma = nrec_mma + 1;
            }
        }
    }
    // ---- Role: tma ----
    if (warp == 5) {
        { // tma_main
            int nrec_tma = 0;
            int nst_tma = 0;
            unsigned int stage = 0;
            unsigned int tile_stage_2 = 0;
            int info_2[7];
            unsigned int _phase_tile_full_2 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
            info_2[0] = sinfo[tile_stage_2 * 7];
            info_2[1] = sinfo[tile_stage_2 * 7 + 1];
            info_2[2] = sinfo[tile_stage_2 * 7 + 2];
            info_2[3] = sinfo[tile_stage_2 * 7 + 3];
            info_2[4] = sinfo[tile_stage_2 * 7 + 4];
            info_2[5] = sinfo[tile_stage_2 * 7 + 5];
            info_2[6] = sinfo[tile_stage_2 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
            tile_stage_2 += 1;
            if (tile_stage_2 == 8) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
            nrec_tma = nrec_tma + 1;
            int batch[2];
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                int _min_0 = ((info_2[0] * 2) < (num_m_tiles - 1) ? (info_2[0] * 2) : (num_m_tiles - 1));
                batch[0] = info_2[2] * num_m_tiles + _min_0;
                int _min_1 = ((info_2[0] * 2 + 1) < (num_m_tiles - 1) ? (info_2[0] * 2 + 1) : (num_m_tiles - 1));
                batch[1] = info_2[2] * num_m_tiles + _min_1;
                int row_base_tma = info_2[1] * 16;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(ab_free_addr + (stage) * 8, _phase_ab_free);
                    if (elect_sync()) {
                        {
                            mbarrier_arrive_expect_tx(ab_full_addr + (stage) * 8, 17408);
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(a_addr + stage * 2 * 16384), "l"((&A)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(sfa_addr + stage * 2 * 512), "l"((&SFA)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(a_addr + (stage * 2 + 1) * 16384), "l"((&A)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[1]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(sfa_addr + (stage * 2 + 1) * 512), "l"((&SFA)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[1]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_ab_free ^= 1; }
                    nst_tma = nst_tma + 1;
                }
                mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
                info_2[0] = sinfo[tile_stage_2 * 7];
                info_2[1] = sinfo[tile_stage_2 * 7 + 1];
                info_2[2] = sinfo[tile_stage_2 * 7 + 2];
                info_2[3] = sinfo[tile_stage_2 * 7 + 3];
                info_2[4] = sinfo[tile_stage_2 * 7 + 4];
                info_2[5] = sinfo[tile_stage_2 * 7 + 5];
                info_2[6] = sinfo[tile_stage_2 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
                tile_stage_2 += 1;
                if (tile_stage_2 == 8) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
                nrec_tma = nrec_tma + 1;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 6) {
        { // scheduler_main
            int nrec_scheduler = 0;
            int nst_scheduler = 0;
            unsigned int tile_stage_3 = 0;
            int num_valid = num_non_exiting_tiles[0];
            int m_chunks = (num_m_tiles + 1) / 2;
            int total_items = m_chunks * group_capacity;
            int sched_first = bid;
            int sched_step = num_bids;
            int sb_rec = lane / 16;
            int sb_col = lane - sb_rec * 16;
            int sb_stride = sched_step * 2;
            int sbi[4];
            float sbf[2];
            int sbni[4];
            float sbnf[2];
            int sb_nbase = sched_first;
            int sb_it = 0;
            int sb_rg = 0;
            int sb_ok = 0;
            int sb_rgs = 0;
            int sb_lk = 0;
            int sb_lkl = 0;
            int sb_prow = 0;
            int sb_mv = 0;
            int sb_exp = 0;
            int sb_safe = 0;
            int sb_tok = 0;
            int sb_tk = 0;
            sb_it = sb_nbase + sb_rec * sched_step;
            sb_rg = sb_it / m_chunks;
            sb_ok = (int)(sb_rg < num_valid);
            sb_rgs = sb_rg * sb_ok;
            sbni[3] = sb_rgs;
            sb_lk = sb_rgs;
            sb_lkl = sb_rgs;
            sbni[0] = tile_idx_to_expert_idx[sb_lk];
            sbni[1] = tile_idx_to_mn_limit[sb_lkl];
            int _max_0 = ((sbni[0]) > (0) ? (sbni[0]) : (0));
            sbnf[0] = alpha[_max_0 * sb_ok];
            sb_prow = sbni[3] * 16 + sb_col;
            sbnf[1] = 1.0f;
            sbni[2] = sb_prow;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 1
            for (int sb_base = sched_first; sb_base < total_items; sb_base += sb_stride) {
                sbi[0] = sbni[0];
                sbi[1] = sbni[1];
                sbi[2] = sbni[2];
                sbi[3] = sbni[3];
                sbf[0] = sbnf[0];
                sbf[1] = sbnf[1];
                sb_nbase = sb_base + sb_stride;
                sb_it = sb_nbase + sb_rec * sched_step;
                sb_rg = sb_it / m_chunks;
                sb_ok = (int)(sb_rg < num_valid);
                sb_rgs = sb_rg * sb_ok;
                sbni[3] = sb_rgs;
                sb_lk = sb_rgs;
                sb_lkl = sb_rgs;
                sbni[0] = tile_idx_to_expert_idx[sb_lk];
                sbni[1] = tile_idx_to_mn_limit[sb_lkl];
                int _max_1 = ((sbni[0]) > (0) ? (sbni[0]) : (0));
                sbnf[0] = alpha[_max_1 * sb_ok];
                sb_prow = sbni[3] * 16 + sb_col;
                sbnf[1] = 1.0f;
                sbni[2] = sb_prow;
                int sb_item = sb_base;
                int row_group = sb_item / m_chunks;
                int m_tile = sb_item - row_group * m_chunks;
                if (row_group >= num_valid) {
                    break;
                }
                mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
                if (sb_rec == 0) {
                    sscale[tile_stage_3 * 17 + (unsigned int)sb_col] = sbf[1];
                    stok[tile_stage_3 * 16 + (unsigned int)sb_col] = sbi[2];
                    if (sb_col == 0) {
                        sscale[tile_stage_3 * 17 + 16] = sbf[0];
                        sinfo[tile_stage_3 * 7] = m_tile;
                        sinfo[tile_stage_3 * 7 + 1] = sbi[3];
                        sinfo[tile_stage_3 * 7 + 2] = sbi[0];
                        sinfo[tile_stage_3 * 7 + 3] = 1;
                        sinfo[tile_stage_3 * 7 + 4] = sbi[1];
                        sinfo[tile_stage_3 * 7 + 5] = 0;
                        sinfo[tile_stage_3 * 7 + 6] = k_tiles;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 32;" ::: "memory");
                mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
                tile_stage_3 += 1;
                if (tile_stage_3 == 8) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
                int sb_item_0 = sb_base + sched_step;
                int row_group_1 = sb_item_0 / m_chunks;
                int m_tile_2 = sb_item_0 - row_group_1 * m_chunks;
                if (row_group_1 >= num_valid) {
                    break;
                }
                mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
                if (sb_rec == 1) {
                    sscale[tile_stage_3 * 17 + (unsigned int)sb_col] = sbf[1];
                    stok[tile_stage_3 * 16 + (unsigned int)sb_col] = sbi[2];
                    if (sb_col == 0) {
                        sscale[tile_stage_3 * 17 + 16] = sbf[0];
                        sinfo[tile_stage_3 * 7] = m_tile_2;
                        sinfo[tile_stage_3 * 7 + 1] = sbi[3];
                        sinfo[tile_stage_3 * 7 + 2] = sbi[0];
                        sinfo[tile_stage_3 * 7 + 3] = 1;
                        sinfo[tile_stage_3 * 7 + 4] = sbi[1];
                        sinfo[tile_stage_3 * 7 + 5] = 0;
                        sinfo[tile_stage_3 * 7 + 6] = k_tiles;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 32;" ::: "memory");
                mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
                tile_stage_3 += 1;
                if (tile_stage_3 == 8) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
            }
            mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
            if (elect_sync()) {
                sinfo[tile_stage_3 * 7] = 0;
                sinfo[tile_stage_3 * 7 + 1] = 0;
                sinfo[tile_stage_3 * 7 + 2] = -1;
                sinfo[tile_stage_3 * 7 + 3] = 0;
                sinfo[tile_stage_3 * 7 + 4] = 0;
                sinfo[tile_stage_3 * 7 + 5] = 0;
                sinfo[tile_stage_3 * 7 + 6] = 0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 4, 32;" ::: "memory");
            mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
            tile_stage_3 += 1;
            if (tile_stage_3 == 8) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
        }
    }
    // ---- Role: gather ----
    if (warp == 7) {
        { // gather_main
            int nrec_gather = 0;
            int nst_gather = 0;
            unsigned int stage_1 = 0;
            unsigned int tile_stage_4 = 0;
            int info_3[7];
            unsigned int _phase_tile_full_3 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
            info_3[0] = sinfo[tile_stage_4 * 7];
            info_3[1] = sinfo[tile_stage_4 * 7 + 1];
            info_3[2] = sinfo[tile_stage_4 * 7 + 2];
            info_3[3] = sinfo[tile_stage_4 * 7 + 3];
            info_3[4] = sinfo[tile_stage_4 * 7 + 4];
            info_3[5] = sinfo[tile_stage_4 * 7 + 5];
            info_3[6] = sinfo[tile_stage_4 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
            tile_stage_4 += 1;
            if (tile_stage_4 == 8) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
            nrec_gather = nrec_gather + 1;
            int gather_sub = warp - 7;
            int lane_0_1 = lane;
            int chunk = lane_0_1 % 8;
            int row_in_pass = lane_0_1 / 8;
            int row_src[4];
            int row_ok[4];
            int sf_src[1];
            int sf_ok[1];
            int cta_row0 = 0;
            unsigned int _phase_b_free = 1;
            #pragma unroll 1
            for (int _tile_3 = 0; _tile_3 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                int row_base_1 = info_3[1] * 16;
                int mn_limit_1 = info_3[4];
                int row = gather_sub * 4 + row_in_pass;
                int prow = row_base_1 + cta_row0 + row;
                int ok = (int)(prow < mn_limit_1);
                row_src[0] = prow * ok;
                row_ok[0] = ok;
                int row_0 = (gather_sub + 1) * 4 + row_in_pass;
                int prow_1 = row_base_1 + cta_row0 + row_0;
                int ok_2 = (int)(prow_1 < mn_limit_1);
                row_src[1] = prow_1 * ok_2;
                row_ok[1] = ok_2;
                int row_3 = (gather_sub + 2) * 4 + row_in_pass;
                int prow_4 = row_base_1 + cta_row0 + row_3;
                int ok_5 = (int)(prow_4 < mn_limit_1);
                row_src[2] = prow_4 * ok_5;
                row_ok[2] = ok_5;
                int row_6 = (gather_sub + 3) * 4 + row_in_pass;
                int prow_7 = row_base_1 + cta_row0 + row_6;
                int ok_8 = (int)(prow_7 < mn_limit_1);
                row_src[3] = prow_7 * ok_8;
                row_ok[3] = ok_8;
                int srow = gather_sub * 32 + lane_0_1;
                int sprow = row_base_1 + srow;
                int sok = (int)(sprow < mn_limit_1 && srow < 16);
                sf_src[0] = sprow * sok;
                sf_ok[0] = sok;
                #pragma unroll 1
                for (int k_2 = 0; k_2 < k_tiles; k_2++) {
                    mbarrier_wait(b_free_addr + (stage_1) * 8, _phase_b_free);
                    int k0 = (info_3[5] + k_2) * 128;
                    int dst_off = (gather_sub * 4 + row_in_pass) * 128 + (chunk ^ (gather_sub * 4 + row_in_pass) % 8) * 16;
                    int src_off = row_src[0] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off), "l"(B + src_off), "r"((row_ok[0] != 0) ? 16 : 0));
                    }
                    int dst_off_0 = ((gather_sub + 1) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 1) * 4 + row_in_pass) % 8) * 16;
                    int src_off_1 = row_src[1] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off_0), "l"(B + src_off_1), "r"((row_ok[1] != 0) ? 16 : 0));
                    }
                    int dst_off_2 = ((gather_sub + 2) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 2) * 4 + row_in_pass) % 8) * 16;
                    int src_off_3 = row_src[2] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off_2), "l"(B + src_off_3), "r"((row_ok[2] != 0) ? 16 : 0));
                    }
                    int dst_off_4 = ((gather_sub + 3) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 3) * 4 + row_in_pass) % 8) * 16;
                    int src_off_5 = row_src[3] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off_4), "l"(B + src_off_5), "r"((row_ok[3] != 0) ? 16 : 0));
                    }
                    int sf_src_off = sf_src[0] * sf_cols + (info_3[5] + k_2) * 4;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 512 + (unsigned int)(gather_sub / 4 * 512) + (unsigned int)(lane_0_1 * 16) + (unsigned int)(gather_sub % 4 * 4)), "l"(SFB + sf_src_off), "r"((sf_ok[0] != 0) ? 4 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(b_full_addr + (stage_1) * 8) : "memory");
                    stage_1 += 1;
                    if (stage_1 == 5) { stage_1 = 0; _phase_b_free ^= 1; }
                    nst_gather = nst_gather + 1;
                }
                mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
                info_3[0] = sinfo[tile_stage_4 * 7];
                info_3[1] = sinfo[tile_stage_4 * 7 + 1];
                info_3[2] = sinfo[tile_stage_4 * 7 + 2];
                info_3[3] = sinfo[tile_stage_4 * 7 + 3];
                info_3[4] = sinfo[tile_stage_4 * 7 + 4];
                info_3[5] = sinfo[tile_stage_4 * 7 + 5];
                info_3[6] = sinfo[tile_stage_4 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
                tile_stage_4 += 1;
                if (tile_stage_4 == 8) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
                nrec_gather = nrec_gather + 1;
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }

    // Kernel epilogue ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
