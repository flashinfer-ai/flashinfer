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
#define TMEM_NCOLS 448
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 128
#define TMEM_ACC_WT_OFFSET 256
#define TMEM_ACC_U_OFFSET 320
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define NUM_TWO_STAGES 2
#define NUM_RING2_STAGES 1
#define SMEM_KG_PANEL_OFF 1024
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_KG_MN_OFF 1024
#define SMEM_KG_MN_STAGE_BYTES 16384
#define SMEM_KG_MN_STRIDE 16384
#define SMEM_W_PHYS_OFF 33792
#define SMEM_W_PHYS_STAGE_BYTES 16384
#define SMEM_W_PHYS_STRIDE 16384
#define SMEM_W_MN_OFF 33792
#define SMEM_W_MN_STAGE_BYTES 16384
#define SMEM_W_MN_STRIDE 16384
#define SMEM_HS_OFF 50176
#define SMEM_HS_STAGE_BYTES 32768
#define SMEM_HS_STRIDE 32768
#define SMEM_HS_MN_OFF 50176
#define SMEM_HS_MN_STAGE_BYTES 32768
#define SMEM_HS_MN_STRIDE 32768
#define SMEM_BV_OFF 82944
#define SMEM_BV_STAGE_BYTES 16384
#define SMEM_BV_STRIDE 16384
#define SMEM_BV_MN_OFF 82944
#define SMEM_BV_MN_STAGE_BYTES 16384
#define SMEM_BV_MN_STRIDE 16384
#define SMEM_AKK_OP_OFF 99328
#define SMEM_AKK_OP_STAGE_BYTES 8192
#define SMEM_AKK_OP_STRIDE 8192
#define SMEM_KB_PANEL_OFF 107520
#define SMEM_KB_PANEL_STAGE_BYTES 8192
#define SMEM_KB_PANEL_STRIDE 8192
#define SMEM_KB_MN_OFF 107520
#define SMEM_KB_MN_STAGE_BYTES 16384
#define SMEM_KB_MN_STRIDE 16384
#define SMEM_VB_PANEL_OFF 123904
#define SMEM_VB_PANEL_STAGE_BYTES 8192
#define SMEM_VB_PANEL_STRIDE 8192
#define SMEM_VB_RAW_OFF 123904
#define SMEM_VB_RAW_STAGE_BYTES 16384
#define SMEM_VB_RAW_STRIDE 16384
#define SMEM_VB_MN_OFF 123904
#define SMEM_VB_MN_STAGE_BYTES 16384
#define SMEM_VB_MN_STRIDE 16384
#define SMEM_U_TILE_OFF 140288
#define SMEM_U_TILE_STAGE_BYTES 16384
#define SMEM_U_TILE_STRIDE 16384
#define SMEM_TOTAL 173056
#define THREADS 448
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__global__ __launch_bounds__(448, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_945a87fa84a265fc9f94(const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap kb_tma, const __grid_constant__ CUtensorMap v_tma, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, const __grid_constant__ CUtensorMap kg_tma, float* __restrict__ gk, const __grid_constant__ CUtensorMap h_out, const __grid_constant__ CUtensorMap v_new, __nv_bfloat16* __restrict__ w_t_out, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int segments, float* __restrict__ h0, float* __restrict__ ht_out, int use_h0, int write_tapes, int write_ht)
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
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define hs_ready_addr (mbar_base + 32)
    #define vp_ready_addr (mbar_base + 40)
    #define bv_ready_addr (mbar_base + 48)
    #define dec_done_addr (mbar_base + 56)
    #define h_ready_addr (mbar_base + 64)
    #define h_tape_read_addr (mbar_base + 72)
    #define bv_tape_read_addr (mbar_base + 80)
    #define wy_full_addr (mbar_base + 88)
    #define vb_ready_addr (mbar_base + 96)
    #define wy_empty_addr (mbar_base + 104)
    #define w_full_addr (mbar_base + 112)
    #define w_ready_addr (mbar_base + 120)
    #define u_ready_addr (mbar_base + 128)
    #define u_read_addr (mbar_base + 144)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_panel_addr = smem + 1024;
    __nv_bfloat16* kg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_mn_addr = smem + 1024;
    __nv_bfloat16* w_phys = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_phys_addr = smem + 33792;
    __nv_bfloat16* w_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_mn_addr = smem + 33792;
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int hs_addr = smem + 50176;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int hs_mn_addr = smem + 50176;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_addr = smem + 82944;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_mn_addr = smem + 82944;
    __nv_bfloat16* akk_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int akk_op_addr = smem + 99328;
    __nv_bfloat16* kb_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int kb_panel_addr = smem + 107520;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int kb_mn_addr = smem + 107520;
    __nv_bfloat16* vb_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 123904);
    const int vb_panel_addr = smem + 123904;
    __nv_bfloat16* vb_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 123904);
    const int vb_raw_addr = smem + 123904;
    __nv_bfloat16* vb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 123904);
    const int vb_mn_addr = smem + 123904;
    __nv_bfloat16* u_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int u_tile_addr = smem + 140288;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'one' ---
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dec_done: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // h_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 64, 2);
            // h_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // bv_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // --- pipeline 'ring2' ---
            // wy_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // vb_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            // wy_empty: 1 barriers, init_count=129
            mbarrier_init(smem + 104, 129);
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // w_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            // --- pipeline 'two' ---
            // u_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // u_read: 2 barriers, init_count=128
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 448 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 128;
    const int tmem_acc_wt = taddr + 256;
    const int tmem_acc_u = taddr + 320;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int seq = blockIdx.x / num_heads % num_seqs;
            int walk = blockIdx.x / (num_heads * num_seqs);
            int chain = 0;
            int nchain = 0;
            if (segments > 0) {
                chain = seq / segments;
                nchain = num_seqs / segments;
            } else {
                nchain = -segments;
                for (int j = 1; j < nchain; j++) {
                    if (seq >= seq_chunk_start[j]) {
                        chain = chain + 1;
                    }
                }
            }
            int tblk = nchain + 1 + seq + chain;
            int c0 = seq_chunk_start[tblk];
            int nc = seq_chunk_start[tblk + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            int tapes_on = 0;
            if (walk == 0) {
                if (write_tapes == 1) {
                    tapes_on = 1;
                }
            }
            unsigned int ust_s = 0;
            float init_row[16];
            if (walk == 1) {
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    init_row[j_1] = ((k_row == j_1) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    init_row[j_2] = ((k_row == 16 + j_2) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    init_row[j_3] = ((k_row == 32 + j_3) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    init_row[j_4] = ((k_row == 48 + j_4) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    init_row[j_5] = ((k_row == 64 + j_5) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    init_row[j_6] = ((k_row == 80 + j_6) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    init_row[j_7] = ((k_row == 96 + j_7) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    init_row[j_8] = ((k_row == 112 + j_8) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            } else if (use_h0 == 1) {
                long long h0_base = ((long long)seq * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(q * 4));
                        init_row[q * 4 + 0] = _v4.x;
                        init_row[q * 4 + 1] = _v4.y;
                        init_row[q * 4 + 2] = _v4.z;
                        init_row[q * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                #pragma unroll
                for (int q_1 = 0; q_1 < 4; q_1++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(16 + q_1 * 4));
                        init_row[q_1 * 4 + 0] = _v4.x;
                        init_row[q_1 * 4 + 1] = _v4.y;
                        init_row[q_1 * 4 + 2] = _v4.z;
                        init_row[q_1 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                #pragma unroll
                for (int q_2 = 0; q_2 < 4; q_2++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(32 + q_2 * 4));
                        init_row[q_2 * 4 + 0] = _v4.x;
                        init_row[q_2 * 4 + 1] = _v4.y;
                        init_row[q_2 * 4 + 2] = _v4.z;
                        init_row[q_2 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                #pragma unroll
                for (int q_3 = 0; q_3 < 4; q_3++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(48 + q_3 * 4));
                        init_row[q_3 * 4 + 0] = _v4.x;
                        init_row[q_3 * 4 + 1] = _v4.y;
                        init_row[q_3 * 4 + 2] = _v4.z;
                        init_row[q_3 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                #pragma unroll
                for (int q_4 = 0; q_4 < 4; q_4++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(64 + q_4 * 4));
                        init_row[q_4 * 4 + 0] = _v4.x;
                        init_row[q_4 * 4 + 1] = _v4.y;
                        init_row[q_4 * 4 + 2] = _v4.z;
                        init_row[q_4 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                #pragma unroll
                for (int q_5 = 0; q_5 < 4; q_5++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(80 + q_5 * 4));
                        init_row[q_5 * 4 + 0] = _v4.x;
                        init_row[q_5 * 4 + 1] = _v4.y;
                        init_row[q_5 * 4 + 2] = _v4.z;
                        init_row[q_5 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                #pragma unroll
                for (int q_6 = 0; q_6 < 4; q_6++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(96 + q_6 * 4));
                        init_row[q_6 * 4 + 0] = _v4.x;
                        init_row[q_6 * 4 + 1] = _v4.y;
                        init_row[q_6 * 4 + 2] = _v4.z;
                        init_row[q_6 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                #pragma unroll
                for (int q_7 = 0; q_7 < 4; q_7++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(112 + q_7 * 4));
                        init_row[q_7 * 4 + 0] = _v4.x;
                        init_row[q_7 * 4 + 1] = _v4.y;
                        init_row[q_7 * 4 + 2] = _v4.z;
                        init_row[q_7 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            } else {
                init_row[0] = 0.0f;
                init_row[1] = 0.0f;
                init_row[2] = 0.0f;
                init_row[3] = 0.0f;
                init_row[4] = 0.0f;
                init_row[5] = 0.0f;
                init_row[6] = 0.0f;
                init_row[7] = 0.0f;
                init_row[8] = 0.0f;
                init_row[9] = 0.0f;
                init_row[10] = 0.0f;
                init_row[11] = 0.0f;
                init_row[12] = 0.0f;
                init_row[13] = 0.0f;
                init_row[14] = 0.0f;
                init_row[15] = 0.0f;
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int u_pk[32];
            u_pk[0] = 0;
            u_pk[1] = 0;
            u_pk[2] = 0;
            u_pk[3] = 0;
            u_pk[4] = 0;
            u_pk[5] = 0;
            u_pk[6] = 0;
            u_pk[7] = 0;
            u_pk[8] = 0;
            u_pk[9] = 0;
            u_pk[10] = 0;
            u_pk[11] = 0;
            u_pk[12] = 0;
            u_pk[13] = 0;
            u_pk[14] = 0;
            u_pk[15] = 0;
            u_pk[16] = 0;
            u_pk[17] = 0;
            u_pk[18] = 0;
            u_pk[19] = 0;
            u_pk[20] = 0;
            u_pk[21] = 0;
            u_pk[22] = 0;
            u_pk[23] = 0;
            u_pk[24] = 0;
            u_pk[25] = 0;
            u_pk[26] = 0;
            u_pk[27] = 0;
            u_pk[28] = 0;
            u_pk[29] = 0;
            u_pk[30] = 0;
            u_pk[31] = 0;
            unsigned int _phase_bv_tape_read_0 = 1;
            unsigned int _phase_u_ready = 0;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_tape_read_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                float _tmem_load_16[16];
                tmem_ld_x16(&_tmem_load_16[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_16_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 0], _tmem_load_16[_lp*2+1 + 0]));
                    _tmem_load_16_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_8 * 8 / 64 * 16384 + k_row * 128 + q_8 * 8 % 64 * 2 ^ (q_8 * 8 / 64 * 16384 + k_row * 128 + q_8 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_16_bf16[q_8 * 4]), "r"(_tmem_load_16_bf16[q_8 * 4 + 1]), "r"(_tmem_load_16_bf16[q_8 * 4 + 2]), "r"(_tmem_load_16_bf16[q_8 * 4 + 3]) : "memory");
                }
                float _tmem_load_17[16];
                tmem_ld_x16(&_tmem_load_17[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_17_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 0], _tmem_load_17[_lp*2+1 + 0]));
                    _tmem_load_17_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_9 = 0; q_9 < 2; q_9++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_9 * 8) / 64 * 16384 + k_row * 128 + (16 + q_9 * 8) % 64 * 2 ^ ((16 + q_9 * 8) / 64 * 16384 + k_row * 128 + (16 + q_9 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_17_bf16[q_9 * 4]), "r"(_tmem_load_17_bf16[q_9 * 4 + 1]), "r"(_tmem_load_17_bf16[q_9 * 4 + 2]), "r"(_tmem_load_17_bf16[q_9 * 4 + 3]) : "memory");
                }
                float _tmem_load_18[16];
                tmem_ld_x16(&_tmem_load_18[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_18_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 0], _tmem_load_18[_lp*2+1 + 0]));
                    _tmem_load_18_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_10 = 0; q_10 < 2; q_10++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_10 * 8) / 64 * 16384 + k_row * 128 + (32 + q_10 * 8) % 64 * 2 ^ ((32 + q_10 * 8) / 64 * 16384 + k_row * 128 + (32 + q_10 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[q_10 * 4]), "r"(_tmem_load_18_bf16[q_10 * 4 + 1]), "r"(_tmem_load_18_bf16[q_10 * 4 + 2]), "r"(_tmem_load_18_bf16[q_10 * 4 + 3]) : "memory");
                }
                float _tmem_load_19[16];
                tmem_ld_x16(&_tmem_load_19[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_19_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 0], _tmem_load_19[_lp*2+1 + 0]));
                    _tmem_load_19_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_11 = 0; q_11 < 2; q_11++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_11 * 8) / 64 * 16384 + k_row * 128 + (48 + q_11 * 8) % 64 * 2 ^ ((48 + q_11 * 8) / 64 * 16384 + k_row * 128 + (48 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[q_11 * 4]), "r"(_tmem_load_19_bf16[q_11 * 4 + 1]), "r"(_tmem_load_19_bf16[q_11 * 4 + 2]), "r"(_tmem_load_19_bf16[q_11 * 4 + 3]) : "memory");
                }
                float _tmem_load_20[16];
                tmem_ld_x16(&_tmem_load_20[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_20_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_20[_lp*2 + 0], _tmem_load_20[_lp*2+1 + 0]));
                    _tmem_load_20_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_12 = 0; q_12 < 2; q_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((64 + q_12 * 8) / 64 * 16384 + k_row * 128 + (64 + q_12 * 8) % 64 * 2 ^ ((64 + q_12 * 8) / 64 * 16384 + k_row * 128 + (64 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_20_bf16[q_12 * 4]), "r"(_tmem_load_20_bf16[q_12 * 4 + 1]), "r"(_tmem_load_20_bf16[q_12 * 4 + 2]), "r"(_tmem_load_20_bf16[q_12 * 4 + 3]) : "memory");
                }
                float _tmem_load_21[16];
                tmem_ld_x16(&_tmem_load_21[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_21_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_21[_lp*2 + 0], _tmem_load_21[_lp*2+1 + 0]));
                    _tmem_load_21_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_13 = 0; q_13 < 2; q_13++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((80 + q_13 * 8) / 64 * 16384 + k_row * 128 + (80 + q_13 * 8) % 64 * 2 ^ ((80 + q_13 * 8) / 64 * 16384 + k_row * 128 + (80 + q_13 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_21_bf16[q_13 * 4]), "r"(_tmem_load_21_bf16[q_13 * 4 + 1]), "r"(_tmem_load_21_bf16[q_13 * 4 + 2]), "r"(_tmem_load_21_bf16[q_13 * 4 + 3]) : "memory");
                }
                float _tmem_load_22[16];
                tmem_ld_x16(&_tmem_load_22[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_22_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_22[_lp*2 + 0], _tmem_load_22[_lp*2+1 + 0]));
                    _tmem_load_22_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_14 = 0; q_14 < 2; q_14++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((96 + q_14 * 8) / 64 * 16384 + k_row * 128 + (96 + q_14 * 8) % 64 * 2 ^ ((96 + q_14 * 8) / 64 * 16384 + k_row * 128 + (96 + q_14 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_22_bf16[q_14 * 4]), "r"(_tmem_load_22_bf16[q_14 * 4 + 1]), "r"(_tmem_load_22_bf16[q_14 * 4 + 2]), "r"(_tmem_load_22_bf16[q_14 * 4 + 3]) : "memory");
                }
                float _tmem_load_23[16];
                tmem_ld_x16(&_tmem_load_23[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_23_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_23[_lp*2 + 0], _tmem_load_23[_lp*2+1 + 0]));
                    _tmem_load_23_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_15 = 0; q_15 < 2; q_15++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((112 + q_15 * 8) / 64 * 16384 + k_row * 128 + (112 + q_15 * 8) % 64 * 2 ^ ((112 + q_15 * 8) / 64 * 16384 + k_row * 128 + (112 + q_15 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_23_bf16[q_15 * 4]), "r"(_tmem_load_23_bf16[q_15 * 4 + 1]), "r"(_tmem_load_23_bf16[q_15 * 4 + 2]), "r"(_tmem_load_23_bf16[q_15 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                if (tapes_on == 1) {
                    mbarrier_wait(bv_tape_read_addr, _phase_bv_tape_read_0);
                    _phase_bv_tape_read_0 ^= 1;
                }
                mbarrier_wait(u_ready_addr + (ust_s) * 8, _phase_u_ready);
                if (walk == 0) {
                    #pragma unroll
                    for (int q_16 = 0; q_16 < 2; q_16++) {
                        unsigned int uaddr = (u_tile_addr + ust_s * 16384 + (unsigned int)((lane_half * 16 + q_16 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_16 * 8) % 64 * 2 ^ ((lane_half * 16 + q_16 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_16 * 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[q_16 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(q_16 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(q_16 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(q_16 * 4) + 3]))
                            : "r"(uaddr));
                    }
                    #pragma unroll
                    for (int q_17 = 0; q_17 < 2; q_17++) {
                        unsigned int uaddr_1 = (u_tile_addr + ust_s * 16384 + (unsigned int)((32 + lane_half * 16 + q_17 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_17 * 8) % 64 * 2 ^ ((32 + lane_half * 16 + q_17 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_17 * 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[8 + q_17 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(8 + q_17 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(8 + q_17 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(8 + q_17 * 4) + 3]))
                            : "r"(uaddr_1));
                    }
                    #pragma unroll
                    for (int q_18 = 0; q_18 < 2; q_18++) {
                        unsigned int uaddr_2 = (u_tile_addr + ust_s * 16384 + (unsigned int)((64 + lane_half * 16 + q_18 * 8) / 64 * 8192 + t_row * 128 + (64 + lane_half * 16 + q_18 * 8) % 64 * 2 ^ ((64 + lane_half * 16 + q_18 * 8) / 64 * 8192 + t_row * 128 + (64 + lane_half * 16 + q_18 * 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[16 + q_18 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(16 + q_18 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(16 + q_18 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(16 + q_18 * 4) + 3]))
                            : "r"(uaddr_2));
                    }
                    #pragma unroll
                    for (int q_19 = 0; q_19 < 2; q_19++) {
                        unsigned int uaddr_3 = (u_tile_addr + ust_s * 16384 + (unsigned int)((96 + lane_half * 16 + q_19 * 8) / 64 * 8192 + t_row * 128 + (96 + lane_half * 16 + q_19 * 8) % 64 * 2 ^ ((96 + lane_half * 16 + q_19 * 8) / 64 * 8192 + t_row * 128 + (96 + lane_half * 16 + q_19 * 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&u_pk[24 + q_19 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(24 + q_19 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(24 + q_19 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&u_pk[(24 + q_19 * 4) + 3]))
                            : "r"(uaddr_3));
                    }
                }
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_24[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                float _unpack_bf16_8[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) << 16);
                    (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) & 0xffff0000u);
                }
                float bvv[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    bvv[j_9] = _unpack_bf16_8[j_9] - _tmem_load_24[j_9];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_20 = 0; q_20 < 2; q_20++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col + q_20 * 8) / 64 * 8192 + t_row * 128 + (col + q_20 * 8) % 64 * 2 ^ ((col + q_20 * 8) / 64 * 8192 + t_row * 128 + (col + q_20 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[q_20 * 4]), "r"(bvv_bf16[q_20 * 4 + 1]), "r"(bvv_bf16[q_20 * 4 + 2]), "r"(bvv_bf16[q_20 * 4 + 3]) : "memory");
                }
                float _tmem_load_25[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_0 = 32 + lane_half * 16;
                float _unpack_bf16_9[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) << 16);
                    (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) & 0xffff0000u);
                }
                float bvv_1[16];
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    bvv_1[j_10] = _unpack_bf16_9[j_10] - _tmem_load_25[j_10];
                }
                unsigned int bvv_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    bvv_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_21 = 0; q_21 < 2; q_21++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_0 + q_21 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_21 * 8) % 64 * 2 ^ ((col_0 + q_21 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_21 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_1_bf16[q_21 * 4]), "r"(bvv_1_bf16[q_21 * 4 + 1]), "r"(bvv_1_bf16[q_21 * 4 + 2]), "r"(bvv_1_bf16[q_21 * 4 + 3]) : "memory");
                }
                float _tmem_load_26[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_2 = 64 + lane_half * 16;
                float _unpack_bf16_10[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) << 16);
                    (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) & 0xffff0000u);
                }
                float bvv_3[16];
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    bvv_3[j_11] = _unpack_bf16_10[j_11] - _tmem_load_26[j_11];
                }
                unsigned int bvv_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_3[_lp*2 + 0], bvv_3[_lp*2+1 + 0]));
                    bvv_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_22 = 0; q_22 < 2; q_22++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_2 + q_22 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_22 * 8) % 64 * 2 ^ ((col_2 + q_22 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_22 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_3_bf16[q_22 * 4]), "r"(bvv_3_bf16[q_22 * 4 + 1]), "r"(bvv_3_bf16[q_22 * 4 + 2]), "r"(bvv_3_bf16[q_22 * 4 + 3]) : "memory");
                }
                float _tmem_load_27[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_4 = 96 + lane_half * 16;
                float _unpack_bf16_11[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) << 16);
                    (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) & 0xffff0000u);
                }
                float bvv_5[16];
                #pragma unroll
                for (int j_12 = 0; j_12 < 16; j_12++) {
                    bvv_5[j_12] = _unpack_bf16_11[j_12] - _tmem_load_27[j_12];
                }
                unsigned int bvv_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_5[_lp*2 + 0], bvv_5[_lp*2+1 + 0]));
                    bvv_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_23 = 0; q_23 < 2; q_23++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_4 + q_23 * 8) / 64 * 8192 + t_row * 128 + (col_4 + q_23 * 8) % 64 * 2 ^ ((col_4 + q_23 * 8) / 64 * 8192 + t_row * 128 + (col_4 + q_23 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_5_bf16[q_23 * 4]), "r"(bvv_5_bf16[q_23 * 4 + 1]), "r"(bvv_5_bf16[q_23 * 4 + 2]), "r"(bvv_5_bf16[q_23 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                mbarrier_arrive(u_read_addr + (ust_s) * 8);
                ust_s += 1;
                if (ust_s == 2) { ust_s = 0; _phase_u_ready ^= 1; }
                if (tapes_on == 1) {
                    mbarrier_wait(h_tape_read_addr, _phase_h_tape_read_0);
                    _phase_h_tape_read_0 ^= 1;
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            if (write_ht == 1) {
                long long ht_base = (((long long)walk * (long long)num_seqs + (long long)seq) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                float _tmem_load_28[16];
                tmem_ld_x16(&_tmem_load_28[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_24 = 0; q_24 < 4; q_24++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_28[q_24 * 4 + 0], _tmem_load_28[q_24 * 4 + 1], _tmem_load_28[q_24 * 4 + 2], _tmem_load_28[q_24 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(q_24 * 4)) = _v4;
                    }
                }
                float _tmem_load_29[16];
                tmem_ld_x16(&_tmem_load_29[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_25 = 0; q_25 < 4; q_25++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_29[q_25 * 4 + 0], _tmem_load_29[q_25 * 4 + 1], _tmem_load_29[q_25 * 4 + 2], _tmem_load_29[q_25 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(16 + q_25 * 4)) = _v4;
                    }
                }
                float _tmem_load_30[16];
                tmem_ld_x16(&_tmem_load_30[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_26 = 0; q_26 < 4; q_26++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_30[q_26 * 4 + 0], _tmem_load_30[q_26 * 4 + 1], _tmem_load_30[q_26 * 4 + 2], _tmem_load_30[q_26 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(32 + q_26 * 4)) = _v4;
                    }
                }
                float _tmem_load_31[16];
                tmem_ld_x16(&_tmem_load_31[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_27 = 0; q_27 < 4; q_27++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_31[q_27 * 4 + 0], _tmem_load_31[q_27 * 4 + 1], _tmem_load_31[q_27 * 4 + 2], _tmem_load_31[q_27 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(48 + q_27 * 4)) = _v4;
                    }
                }
                float _tmem_load_32[16];
                tmem_ld_x16(&_tmem_load_32[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_28 = 0; q_28 < 4; q_28++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_32[q_28 * 4 + 0], _tmem_load_32[q_28 * 4 + 1], _tmem_load_32[q_28 * 4 + 2], _tmem_load_32[q_28 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(64 + q_28 * 4)) = _v4;
                    }
                }
                float _tmem_load_33[16];
                tmem_ld_x16(&_tmem_load_33[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_29 = 0; q_29 < 4; q_29++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_33[q_29 * 4 + 0], _tmem_load_33[q_29 * 4 + 1], _tmem_load_33[q_29 * 4 + 2], _tmem_load_33[q_29 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(80 + q_29 * 4)) = _v4;
                    }
                }
                float _tmem_load_34[16];
                tmem_ld_x16(&_tmem_load_34[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_30 = 0; q_30 < 4; q_30++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_34[q_30 * 4 + 0], _tmem_load_34[q_30 * 4 + 1], _tmem_load_34[q_30 * 4 + 2], _tmem_load_34[q_30 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(96 + q_30 * 4)) = _v4;
                    }
                }
                float _tmem_load_35[16];
                tmem_ld_x16(&_tmem_load_35[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_31 = 0; q_31 < 4; q_31++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_35[q_31 * 4 + 0], _tmem_load_35[q_31 * 4 + 1], _tmem_load_35[q_31 * 4 + 2], _tmem_load_35[q_31 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(112 + q_31 * 4)) = _v4;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: decayw ----
    if (warp >= 4 && warp <= 7) {
        { // decayw_main
            int head_1 = blockIdx.x % num_heads;
            int seq_1 = blockIdx.x / num_heads % num_seqs;
            int walk_1 = blockIdx.x / (num_heads * num_seqs);
            int chain_1 = 0;
            int nchain_1 = 0;
            if (segments > 0) {
                chain_1 = seq_1 / segments;
                nchain_1 = num_seqs / segments;
            } else {
                nchain_1 = -segments;
                for (int j_13 = 1; j_13 < nchain_1; j_13++) {
                    if (seq_1 >= seq_chunk_start[j_13]) {
                        chain_1 = chain_1 + 1;
                    }
                }
            }
            int tblk_1 = nchain_1 + 1 + seq_1 + chain_1;
            int c0_1 = seq_chunk_start[tblk_1];
            int nc_1 = seq_chunk_start[tblk_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int warp_in_wg_1 = warp % 4;
            int k_row_1 = warp_in_wg_1 * 32 + lane;
            int h_row_base_1 = warp_in_wg_1 * 32 << 16;
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_next = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                gk_pre = gk[gn_next];
            }
            unsigned int _phase_hs_ready_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float decay = approx_exp2(gk_pre);
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], taddr + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2[16];
                #pragma unroll
                for (int j_14 = 0; j_14 < 16; j_14++) {
                    hs2[j_14] = _tmem_load_8[j_14] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1, hs2);
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_0[16];
                #pragma unroll
                for (int j_15 = 0; j_15 < 16; j_15++) {
                    hs2_0[j_15] = _tmem_load_9[j_15] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hs2_0);
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_1[16];
                #pragma unroll
                for (int j_16 = 0; j_16 < 16; j_16++) {
                    hs2_1[j_16] = _tmem_load_10[j_16] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hs2_1);
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_2[16];
                #pragma unroll
                for (int j_17 = 0; j_17 < 16; j_17++) {
                    hs2_2[j_17] = _tmem_load_11[j_17] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hs2_2);
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], taddr + (unsigned int)h_row_base_1 + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_3[16];
                #pragma unroll
                for (int j_18 = 0; j_18 < 16; j_18++) {
                    hs2_3[j_18] = _tmem_load_12[j_18] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 64, hs2_3);
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], taddr + (unsigned int)h_row_base_1 + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_4[16];
                #pragma unroll
                for (int j_19 = 0; j_19 < 16; j_19++) {
                    hs2_4[j_19] = _tmem_load_13[j_19] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 80, hs2_4);
                float _tmem_load_14[16];
                tmem_ld_x16(&_tmem_load_14[0], taddr + (unsigned int)h_row_base_1 + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_5[16];
                #pragma unroll
                for (int j_20 = 0; j_20 < 16; j_20++) {
                    hs2_5[j_20] = _tmem_load_14[j_20] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 96, hs2_5);
                float _tmem_load_15[16];
                tmem_ld_x16(&_tmem_load_15[0], taddr + (unsigned int)h_row_base_1 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_6[16];
                #pragma unroll
                for (int j_21 = 0; j_21 < 16; j_21++) {
                    hs2_6[j_21] = _tmem_load_15[j_21] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 112, hs2_6);
                if (nc_1 > c_1 + 1) {
                    long long gn_next_1 = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                    gk_pre = gk[gn_next_1];
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int head_2 = blockIdx.x % num_heads;
            int seq_2 = blockIdx.x / num_heads % num_seqs;
            int walk_2 = blockIdx.x / (num_heads * num_seqs);
            int chain_2 = 0;
            int nchain_2 = 0;
            if (segments > 0) {
                chain_2 = seq_2 / segments;
                nchain_2 = num_seqs / segments;
            } else {
                nchain_2 = -segments;
                for (int j_22 = 1; j_22 < nchain_2; j_22++) {
                    if (seq_2 >= seq_chunk_start[j_22]) {
                        chain_2 = chain_2 + 1;
                    }
                }
            }
            int tblk_2 = nchain_2 + 1 + seq_2 + chain_2;
            int c0_2 = seq_chunk_start[tblk_2];
            int nc_2 = seq_chunk_start[tblk_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int tapes_on_1 = 0;
            if (walk_2 == 0) {
                if (write_tapes == 1) {
                    tapes_on_1 = 1;
                }
            }
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                if (nc_2 > 0) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 16384);
                    int trow = row0_2;
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&kg_tma), seg * 64, head_2, trow, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 16384);
                        int trow_1 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                            tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg_1) * 8192, (&kg_tma), seg_1 * 64, head_2, trow_1, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    if (tapes_on_1 == 1) {
                        int chunk = (c0_2 + c_2) * num_heads + head_2;
                        mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                        _phase_hs_ready_0_1 ^= 1;
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
                            tma_store_3d((&h_out), p * 64, 0, chunk, hs_addr + (unsigned int)(p * 16384));
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        int trow_2 = row0_2 + c_2 * 64;
                        mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                        _phase_bv_ready_0 ^= 1;
                        #pragma unroll
                        for (int p_1 = 0; p_1 < 2; p_1++) {
                            tma_store_3d((&v_new), p_1 * 64, head_2, trow_2, bv_addr + (unsigned int)(p_1 * 8192));
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        mbarrier_arrive(h_tape_read_addr);
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(bv_tape_read_addr);
                    }
                }
                if (tapes_on_1 == 1) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int head_3 = blockIdx.x % num_heads;
            int seq_3 = blockIdx.x / num_heads % num_seqs;
            int walk_3 = blockIdx.x / (num_heads * num_seqs);
            int chain_3 = 0;
            int nchain_3 = 0;
            if (segments > 0) {
                chain_3 = seq_3 / segments;
                nchain_3 = num_seqs / segments;
            } else {
                nchain_3 = -segments;
                for (int j_23 = 1; j_23 < nchain_3; j_23++) {
                    if (seq_3 >= seq_chunk_start[j_23]) {
                        chain_3 = chain_3 + 1;
                    }
                }
            }
            int tblk_3 = nchain_3 + 1 + seq_3 + chain_3;
            int c0_3 = seq_chunk_start[tblk_3];
            int nc_3 = seq_chunk_start[tblk_3 + 1] - c0_3;
            int row0_3 = c0_3 * 64;
            unsigned int stage_1 = 0;
            unsigned int wst_m = 0;
            unsigned int mst = 0;
            unsigned int _phase_wy_full_0 = 0;
            unsigned int _phase_vb_ready_0 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_w_ready_0 = 0;
            unsigned int _phase_hs_ready_0_2 = 0;
            unsigned int _phase_bv_ready_0_1 = 0;
            unsigned int _phase_dec_done_0 = 0;
            if (elect_sync()) {
                if (nc_3 > 0) {
                    mbarrier_wait(wy_full_addr, _phase_wy_full_0);
                    _phase_wy_full_0 ^= 1;
                    mbarrier_wait(vb_ready_addr, _phase_vb_ready_0);
                    _phase_vb_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((akk_op_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((akk_op_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_wt), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((akk_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_3 = (((vb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = ((akk_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_3 = (((vb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_u), "r"(0) : "memory");
                    tcgen05_commit(w_full_addr);
                    tcgen05_commit(wy_empty_addr);
                }
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(w_ready_addr, _phase_w_ready_0);
                    _phase_w_ready_0 ^= 1;
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_2);
                    _phase_hs_ready_0_2 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((w_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
                    int _smem_desc_lo_5 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_4 = (((w_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
                    int _smem_desc_lo_5 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
                    "mov.b32 id, 69305488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    if (nc_3 > c_3 + 1) {
                        mbarrier_wait(wy_full_addr, _phase_wy_full_0);
                        _phase_wy_full_0 ^= 1;
                        mbarrier_wait(vb_ready_addr, _phase_vb_ready_0);
                        _phase_vb_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_7 = (((akk_op_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_6 = (((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_7 = (((akk_op_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_wt), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_8 = ((akk_op_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_9 = (((vb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_8 = ((akk_op_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_9 = (((vb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_acc_u), "r"(0) : "memory");
                        tcgen05_commit(w_full_addr);
                        tcgen05_commit(wy_empty_addr);
                    }
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0_1);
                    _phase_bv_ready_0_1 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                    _phase_dec_done_0 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(h_ready_addr);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: cvt ----
    if (warp >= 10 && warp <= 13) {
        { // cvt_main
            int head_4 = blockIdx.x % num_heads;
            int seq_4 = blockIdx.x / num_heads % num_seqs;
            int walk_4 = blockIdx.x / (num_heads * num_seqs);
            int chain_4 = 0;
            int nchain_4 = 0;
            if (segments > 0) {
                chain_4 = seq_4 / segments;
                nchain_4 = num_seqs / segments;
            } else {
                nchain_4 = -segments;
                for (int j_24 = 1; j_24 < nchain_4; j_24++) {
                    if (seq_4 >= seq_chunk_start[j_24]) {
                        chain_4 = chain_4 + 1;
                    }
                }
            }
            int tblk_4 = nchain_4 + 1 + seq_4 + chain_4;
            int c0_4 = seq_chunk_start[tblk_4];
            int nc_4 = seq_chunk_start[tblk_4 + 1] - c0_4;
            int row0_4 = c0_4 * 64;
            const int cvt_tid = (warp - 10) * 32 + lane;
            int warp_in_wg_2 = warp % 4;
            int k_row_2 = warp_in_wg_2 * 32 + lane;
            int row_base = warp_in_wg_2 * 32 << 16;
            int lane_half_1 = lane >> 4;
            int t_row_1 = warp_in_wg_2 * 16 + (lane & 15);
            const int v_row = cvt_tid >> 1;
            const int ch0 = (cvt_tid & 1) * 8;
            unsigned int wst_c = 0;
            unsigned int ust = 0;
            unsigned int wst_i = 0;
            int bos_a = 0;
            int bos_b = 0;
            if (nc_4 > 0) {
                bos_a = chunk_bos[c0_4];
            }
            if (nc_4 > 1) {
                bos_b = chunk_bos[c0_4 + 1];
            }
            unsigned int _phase_wy_empty_0 = 1;
            if (warp == 10) {
                if (elect_sync()) {
                    if (nc_4 > 0) {
                        mbarrier_wait(wy_empty_addr, _phase_wy_empty_0);
                        _phase_wy_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(wy_full_addr, 40960);
                        int trow_3 = row0_4;
                        int arow = bos_a;
                        tma_3d_gmem2smem(akk_op_addr, (&akk_tma), 0, head_4, arow, wy_full_addr);
                        #pragma unroll
                        for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                            tma_3d_gmem2smem(kb_panel_addr + (unsigned int)seg_2 * 8192, (&kb_tma), seg_2 * 64, head_4, trow_3, wy_full_addr);
                            tma_3d_gmem2smem(vb_panel_addr + (unsigned int)seg_2 * 8192, (&v_tma), seg_2 * 64, head_4, bos_a, wy_full_addr);
                        }
                        if (nc_4 > 1) {
                            int trow2 = row0_4 + 64;
                            int arow2 = bos_b;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_4)), "r"((int)(arow2)) : "memory");
                            #pragma unroll
                            for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kb_tma))), "r"((int)(seg_3 * 64)), "r"((int)(head_4)), "r"((int)(trow2)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(seg_3 * 64)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                            }
                        }
                        bos_a = bos_b;
                        if (nc_4 > 2) {
                            bos_b = chunk_bos[c0_4 + 2];
                        }
                    }
                }
            }
            int bos_n = 0;
            int clen_n = 0;
            if (nc_4 > 0) {
                bos_n = chunk_bos[c0_4];
                clen_n = chunk_len[c0_4];
            }
            float bt_n = 0.0f;
            int clast = clen_n - 1;
            if (clast < 0) {
                clast = 0;
            }
            int t_c = v_row;
            if (t_c > clast) {
                t_c = clast;
            }
            if (nc_4 > 0) {
                long long bidx = (long long)(bos_n + t_c) * (long long)num_heads + (long long)head_4;
                bt_n = (float)beta[bidx];
            }
            unsigned int zeros[4];
            zeros[0] = 0;
            zeros[1] = 0;
            zeros[2] = 0;
            zeros[3] = 0;
            unsigned int _phase_wy_full_0_1 = 0;
            unsigned int _phase_w_full_0 = 0;
            unsigned int _phase_u_read = 1;
            #pragma unroll 1
            for (int j_25 = 0; j_25 < nc_4; j_25++) {
                int clen_j = clen_n;
                float bt_j = bt_n;
                if (nc_4 > j_25 + 1) {
                    bos_n = chunk_bos[c0_4 + j_25 + 1];
                    clen_n = chunk_len[c0_4 + j_25 + 1];
                    clast = clen_n - 1;
                    if (clast < 0) {
                        clast = 0;
                    }
                    t_c = v_row;
                    if (t_c > clast) {
                        t_c = clast;
                    }
                    long long bidx_1 = (long long)(bos_n + t_c) * (long long)num_heads + (long long)head_4;
                    bt_n = (float)beta[bidx_1];
                }
                mbarrier_wait(wy_full_addr, _phase_wy_full_0_1);
                _phase_wy_full_0_1 ^= 1;
                if (clen_j < 64) {
                    int npad = (64 - clen_j) * 8;
                    for (int idx = cvt_tid; idx < npad; idx += 128) {
                        int prow = clen_j + (idx >> 3);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)((idx & 7) * 8 / 64 * 8192 + prow * 128 + (idx & 7) * 8 % 64 * 2 ^ ((idx & 7) * 8 / 64 * 8192 + prow * 128 + (idx & 7) * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                    }
                }
                int valid = 0;
                if (v_row < clen_j) {
                    valid = 1;
                }
                float bt = 0.0f;
                if (valid == 1) {
                    bt = bt_j;
                }
                unsigned int mask = (unsigned int)(-valid);
                unsigned int addr = (vb_raw_addr + (unsigned int)(ch0 * 8 / 64 * 8192 + v_row * 128 + ch0 * 8 % 64 * 2 ^ (ch0 * 8 / 64 * 8192 + v_row * 128 + ch0 * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                    : "r"(addr));
                unsigned int sel[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    sel[i] = raw[i] & mask;
                }
                float _unpack_bf16_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel[_pair]) & 0xffff0000u);
                }
                float prod[8];
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    prod[e] = _unpack_bf16_0[e] * bt;
                }
                unsigned int packed[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod[_lp*2 + 0], prod[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                unsigned int addr_0 = (vb_raw_addr + (unsigned int)((ch0 + 1) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 1) * 8 % 64 * 2 ^ ((ch0 + 1) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 1) * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 3]))
                    : "r"(addr_0));
                unsigned int sel_2[4];
                #pragma unroll
                for (int i_1 = 0; i_1 < 4; i_1++) {
                    sel_2[i_1] = raw_1[i_1] & mask;
                }
                float _unpack_bf16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_2[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_2[_pair]) & 0xffff0000u);
                }
                float prod_3[8];
                #pragma unroll
                for (int e_1 = 0; e_1 < 8; e_1++) {
                    prod_3[e_1] = _unpack_bf16_1[e_1] * bt;
                }
                unsigned int packed_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_3[_lp*2 + 0], prod_3[_lp*2+1 + 0]));
                    packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_0), "r"(packed_4[0]), "r"(packed_4[1]), "r"(packed_4[2]), "r"(packed_4[3]) : "memory");
                unsigned int addr_5 = (vb_raw_addr + (unsigned int)((ch0 + 2) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 2) * 8 % 64 * 2 ^ ((ch0 + 2) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 2) * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw_6[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_6[(0) + 3]))
                    : "r"(addr_5));
                unsigned int sel_7[4];
                #pragma unroll
                for (int i_2 = 0; i_2 < 4; i_2++) {
                    sel_7[i_2] = raw_6[i_2] & mask;
                }
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_7[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_7[_pair]) & 0xffff0000u);
                }
                float prod_8[8];
                #pragma unroll
                for (int e_2 = 0; e_2 < 8; e_2++) {
                    prod_8[e_2] = _unpack_bf16_2[e_2] * bt;
                }
                unsigned int packed_9[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_8[_lp*2 + 0], prod_8[_lp*2+1 + 0]));
                    packed_9[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_5), "r"(packed_9[0]), "r"(packed_9[1]), "r"(packed_9[2]), "r"(packed_9[3]) : "memory");
                unsigned int addr_10 = (vb_raw_addr + (unsigned int)((ch0 + 3) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 3) * 8 % 64 * 2 ^ ((ch0 + 3) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 3) * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw_11[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_11[(0) + 3]))
                    : "r"(addr_10));
                unsigned int sel_12[4];
                #pragma unroll
                for (int i_3 = 0; i_3 < 4; i_3++) {
                    sel_12[i_3] = raw_11[i_3] & mask;
                }
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_12[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_12[_pair]) & 0xffff0000u);
                }
                float prod_13[8];
                #pragma unroll
                for (int e_3 = 0; e_3 < 8; e_3++) {
                    prod_13[e_3] = _unpack_bf16_3[e_3] * bt;
                }
                unsigned int packed_14[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_13[_lp*2 + 0], prod_13[_lp*2+1 + 0]));
                    packed_14[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_10), "r"(packed_14[0]), "r"(packed_14[1]), "r"(packed_14[2]), "r"(packed_14[3]) : "memory");
                unsigned int addr_15 = (vb_raw_addr + (unsigned int)((ch0 + 4) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 4) * 8 % 64 * 2 ^ ((ch0 + 4) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 4) * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw_16[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_16[(0) + 3]))
                    : "r"(addr_15));
                unsigned int sel_17[4];
                #pragma unroll
                for (int i_4 = 0; i_4 < 4; i_4++) {
                    sel_17[i_4] = raw_16[i_4] & mask;
                }
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_17[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_17[_pair]) & 0xffff0000u);
                }
                float prod_18[8];
                #pragma unroll
                for (int e_4 = 0; e_4 < 8; e_4++) {
                    prod_18[e_4] = _unpack_bf16_4[e_4] * bt;
                }
                unsigned int packed_19[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_18[_lp*2 + 0], prod_18[_lp*2+1 + 0]));
                    packed_19[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_15), "r"(packed_19[0]), "r"(packed_19[1]), "r"(packed_19[2]), "r"(packed_19[3]) : "memory");
                unsigned int addr_20 = (vb_raw_addr + (unsigned int)((ch0 + 5) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 5) * 8 % 64 * 2 ^ ((ch0 + 5) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 5) * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw_21[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_21[(0) + 3]))
                    : "r"(addr_20));
                unsigned int sel_22[4];
                #pragma unroll
                for (int i_5 = 0; i_5 < 4; i_5++) {
                    sel_22[i_5] = raw_21[i_5] & mask;
                }
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_22[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_22[_pair]) & 0xffff0000u);
                }
                float prod_23[8];
                #pragma unroll
                for (int e_5 = 0; e_5 < 8; e_5++) {
                    prod_23[e_5] = _unpack_bf16_5[e_5] * bt;
                }
                unsigned int packed_24[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_23[_lp*2 + 0], prod_23[_lp*2+1 + 0]));
                    packed_24[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_20), "r"(packed_24[0]), "r"(packed_24[1]), "r"(packed_24[2]), "r"(packed_24[3]) : "memory");
                unsigned int addr_25 = (vb_raw_addr + (unsigned int)((ch0 + 6) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 6) * 8 % 64 * 2 ^ ((ch0 + 6) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 6) * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw_26[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_26[(0) + 3]))
                    : "r"(addr_25));
                unsigned int sel_27[4];
                #pragma unroll
                for (int i_6 = 0; i_6 < 4; i_6++) {
                    sel_27[i_6] = raw_26[i_6] & mask;
                }
                float _unpack_bf16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_27[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_27[_pair]) & 0xffff0000u);
                }
                float prod_28[8];
                #pragma unroll
                for (int e_6 = 0; e_6 < 8; e_6++) {
                    prod_28[e_6] = _unpack_bf16_6[e_6] * bt;
                }
                unsigned int packed_29[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_28[_lp*2 + 0], prod_28[_lp*2+1 + 0]));
                    packed_29[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_25), "r"(packed_29[0]), "r"(packed_29[1]), "r"(packed_29[2]), "r"(packed_29[3]) : "memory");
                unsigned int addr_30 = (vb_raw_addr + (unsigned int)((ch0 + 7) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 7) * 8 % 64 * 2 ^ ((ch0 + 7) * 8 / 64 * 8192 + v_row * 128 + (ch0 + 7) * 8 % 64 * 2 >> 7 & 7) << 4));
                unsigned int raw_31[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_31[(0) + 3]))
                    : "r"(addr_30));
                unsigned int sel_32[4];
                #pragma unroll
                for (int i_7 = 0; i_7 < 4; i_7++) {
                    sel_32[i_7] = raw_31[i_7] & mask;
                }
                float _unpack_bf16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_32[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_32[_pair]) & 0xffff0000u);
                }
                float prod_33[8];
                #pragma unroll
                for (int e_7 = 0; e_7 < 8; e_7++) {
                    prod_33[e_7] = _unpack_bf16_7[e_7] * bt;
                }
                unsigned int packed_34[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_33[_lp*2 + 0], prod_33[_lp*2+1 + 0]));
                    packed_34[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_30), "r"(packed_34[0]), "r"(packed_34[1]), "r"(packed_34[2]), "r"(packed_34[3]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(vb_ready_addr);
                mbarrier_arrive(wy_empty_addr);
                __syncwarp();
                if (warp == 10) {
                    if (elect_sync()) {
                        if (nc_4 > j_25 + 1) {
                            mbarrier_wait(wy_empty_addr, _phase_wy_empty_0);
                            _phase_wy_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(wy_full_addr, 40960);
                            int trow_4 = row0_4 + (j_25 + 1) * 64;
                            int arow_1 = bos_a;
                            tma_3d_gmem2smem(akk_op_addr, (&akk_tma), 0, head_4, arow_1, wy_full_addr);
                            #pragma unroll
                            for (int seg_4 = 0; seg_4 < 2; seg_4++) {
                                tma_3d_gmem2smem(kb_panel_addr + (unsigned int)seg_4 * 8192, (&kb_tma), seg_4 * 64, head_4, trow_4, wy_full_addr);
                                tma_3d_gmem2smem(vb_panel_addr + (unsigned int)seg_4 * 8192, (&v_tma), seg_4 * 64, head_4, bos_a, wy_full_addr);
                            }
                            if (nc_4 > j_25 + 1 + 1) {
                                int trow2_1 = row0_4 + (j_25 + 1 + 1) * 64;
                                int arow2_1 = bos_b;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_4)), "r"((int)(arow2_1)) : "memory");
                                #pragma unroll
                                for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kb_tma))), "r"((int)(seg_5 * 64)), "r"((int)(head_4)), "r"((int)(trow2_1)) : "memory");
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(seg_5 * 64)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                                }
                            }
                            bos_a = bos_b;
                            if (nc_4 > j_25 + 1 + 2) {
                                bos_b = chunk_bos[c0_4 + (j_25 + 1) + 2];
                            }
                        }
                    }
                }
                mbarrier_wait(w_full_addr, _phase_w_full_0);
                _phase_w_full_0 ^= 1;
                mbarrier_wait(u_read_addr + (ust) * 8, _phase_u_read);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int wcol = (int)wst_c * 64;
                long long widx = (((long long)(c0_4 + j_25) * 128 + (long long)k_row_2) * (long long)num_heads + (long long)head_4) * 64;
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + 256 + (unsigned int)row_base + (unsigned int)wcol);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_32 = 0; q_32 < 2; q_32++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)(q_32 * 8 / 64 * 16384 + k_row_2 * 128 + q_32 * 8 % 64 * 2 ^ (q_32 * 8 / 64 * 16384 + k_row_2 * 128 + q_32 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[q_32 * 4]), "r"(_tmem_load_0_bf16[q_32 * 4 + 1]), "r"(_tmem_load_0_bf16[q_32 * 4 + 2]), "r"(_tmem_load_0_bf16[q_32 * 4 + 3]) : "memory");
                }
                if (walk_4 == 0) {
                    unsigned int _packed_0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 2], _tmem_load_0[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 4], _tmem_load_0[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 6], _tmem_load_0[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 8], _tmem_load_0[_lp*2+1 + 8]));
                        _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 10], _tmem_load_0[_lp*2+1 + 10]));
                        _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 12], _tmem_load_0[_lp*2+1 + 12]));
                        _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 14], _tmem_load_0[_lp*2+1 + 14]));
                        _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(w_t_out + widx)), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                    }
                }
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + 256 + (unsigned int)row_base + (unsigned int)(wcol + 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_33 = 0; q_33 < 2; q_33++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((16 + q_33 * 8) / 64 * 16384 + k_row_2 * 128 + (16 + q_33 * 8) % 64 * 2 ^ ((16 + q_33 * 8) / 64 * 16384 + k_row_2 * 128 + (16 + q_33 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[q_33 * 4]), "r"(_tmem_load_1_bf16[q_33 * 4 + 1]), "r"(_tmem_load_1_bf16[q_33 * 4 + 2]), "r"(_tmem_load_1_bf16[q_33 * 4 + 3]) : "memory");
                }
                if (walk_4 == 0) {
                    unsigned int _packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 2], _tmem_load_1[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 4], _tmem_load_1[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 6], _tmem_load_1[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 8], _tmem_load_1[_lp*2+1 + 8]));
                        _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 10], _tmem_load_1[_lp*2+1 + 10]));
                        _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 12], _tmem_load_1[_lp*2+1 + 12]));
                        _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 14], _tmem_load_1[_lp*2+1 + 14]));
                        _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(w_t_out + (widx + 16))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                    }
                }
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + 256 + (unsigned int)row_base + (unsigned int)(wcol + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_34 = 0; q_34 < 2; q_34++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((32 + q_34 * 8) / 64 * 16384 + k_row_2 * 128 + (32 + q_34 * 8) % 64 * 2 ^ ((32 + q_34 * 8) / 64 * 16384 + k_row_2 * 128 + (32 + q_34 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[q_34 * 4]), "r"(_tmem_load_2_bf16[q_34 * 4 + 1]), "r"(_tmem_load_2_bf16[q_34 * 4 + 2]), "r"(_tmem_load_2_bf16[q_34 * 4 + 3]) : "memory");
                }
                if (walk_4 == 0) {
                    unsigned int _packed_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                        _packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 2], _tmem_load_2[_lp*2+1 + 2]));
                        _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 4], _tmem_load_2[_lp*2+1 + 4]));
                        _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 6], _tmem_load_2[_lp*2+1 + 6]));
                        _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 8], _tmem_load_2[_lp*2+1 + 8]));
                        _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 10], _tmem_load_2[_lp*2+1 + 10]));
                        _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 12], _tmem_load_2[_lp*2+1 + 12]));
                        _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 14], _tmem_load_2[_lp*2+1 + 14]));
                        _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(_packed_2);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(w_t_out + (widx + 32))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                    }
                }
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 256 + (unsigned int)row_base + (unsigned int)(wcol + 48));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_35 = 0; q_35 < 2; q_35++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((48 + q_35 * 8) / 64 * 16384 + k_row_2 * 128 + (48 + q_35 * 8) % 64 * 2 ^ ((48 + q_35 * 8) / 64 * 16384 + k_row_2 * 128 + (48 + q_35 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[q_35 * 4]), "r"(_tmem_load_3_bf16[q_35 * 4 + 1]), "r"(_tmem_load_3_bf16[q_35 * 4 + 2]), "r"(_tmem_load_3_bf16[q_35 * 4 + 3]) : "memory");
                }
                if (walk_4 == 0) {
                    unsigned int _packed_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                        _packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 2], _tmem_load_3[_lp*2+1 + 2]));
                        _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 4], _tmem_load_3[_lp*2+1 + 4]));
                        _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 6], _tmem_load_3[_lp*2+1 + 6]));
                        _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 8], _tmem_load_3[_lp*2+1 + 8]));
                        _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 10], _tmem_load_3[_lp*2+1 + 10]));
                        _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 12], _tmem_load_3[_lp*2+1 + 12]));
                        _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 14], _tmem_load_3[_lp*2+1 + 14]));
                        _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_3);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(w_t_out + (widx + 48))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                    }
                }
                float _tmem_load_4[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                    : "r"(taddr + 320 + (unsigned int)row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int ucol = lane_half_1 * 16;
                unsigned int _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_36 = 0; q_36 < 2; q_36++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((u_tile_addr + ust * 16384 + (unsigned int)((ucol + q_36 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol + q_36 * 8) % 64 * 2 ^ ((ucol + q_36 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol + q_36 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[q_36 * 4]), "r"(_tmem_load_4_bf16[q_36 * 4 + 1]), "r"(_tmem_load_4_bf16[q_36 * 4 + 2]), "r"(_tmem_load_4_bf16[q_36 * 4 + 3]) : "memory");
                }
                float _tmem_load_5[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                    : "r"(taddr + 320 + (unsigned int)row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int ucol_35 = 32 + lane_half_1 * 16;
                unsigned int _tmem_load_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_37 = 0; q_37 < 2; q_37++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((u_tile_addr + ust * 16384 + (unsigned int)((ucol_35 + q_37 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol_35 + q_37 * 8) % 64 * 2 ^ ((ucol_35 + q_37 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol_35 + q_37 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[q_37 * 4]), "r"(_tmem_load_5_bf16[q_37 * 4 + 1]), "r"(_tmem_load_5_bf16[q_37 * 4 + 2]), "r"(_tmem_load_5_bf16[q_37 * 4 + 3]) : "memory");
                }
                float _tmem_load_6[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                    : "r"(taddr + 320 + (unsigned int)row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int ucol_36 = 64 + lane_half_1 * 16;
                unsigned int _tmem_load_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_38 = 0; q_38 < 2; q_38++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((u_tile_addr + ust * 16384 + (unsigned int)((ucol_36 + q_38 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol_36 + q_38 * 8) % 64 * 2 ^ ((ucol_36 + q_38 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol_36 + q_38 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[q_38 * 4]), "r"(_tmem_load_6_bf16[q_38 * 4 + 1]), "r"(_tmem_load_6_bf16[q_38 * 4 + 2]), "r"(_tmem_load_6_bf16[q_38 * 4 + 3]) : "memory");
                }
                float _tmem_load_7[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                    : "r"(taddr + 320 + (unsigned int)row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int ucol_37 = 96 + lane_half_1 * 16;
                unsigned int _tmem_load_7_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                    _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_39 = 0; q_39 < 2; q_39++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((u_tile_addr + ust * 16384 + (unsigned int)((ucol_37 + q_39 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol_37 + q_39 * 8) % 64 * 2 ^ ((ucol_37 + q_39 * 8) / 64 * 8192 + t_row_1 * 128 + (ucol_37 + q_39 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[q_39 * 4]), "r"(_tmem_load_7_bf16[q_39 * 4 + 1]), "r"(_tmem_load_7_bf16[q_39 * 4 + 2]), "r"(_tmem_load_7_bf16[q_39 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(w_ready_addr);
                mbarrier_arrive(u_ready_addr + (ust) * 8);
                ust += 1;
                if (ust == 2) { ust = 0; _phase_u_read ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
