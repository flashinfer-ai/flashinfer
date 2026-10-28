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
#define TMEM_NCOLS 512
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 64
#define TMEM_ACC_QH_OFFSET 128
#define TMEM_ACC_AV_OFFSET 192
#define TMEM_ACC_WT_OFFSET 256
#define TMEM_ACC_U_OFFSET 384
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define NUM_ACCS_STAGES 2
#define NUM_RING2_STAGES 2
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
#define SMEM_HS_OFF 66560
#define SMEM_HS_STAGE_BYTES 16384
#define SMEM_HS_STRIDE 16384
#define SMEM_HS_MN_OFF 66560
#define SMEM_HS_MN_STAGE_BYTES 16384
#define SMEM_HS_MN_STRIDE 16384
#define SMEM_BV_OFF 82944
#define SMEM_BV_STAGE_BYTES 8192
#define SMEM_BV_STRIDE 8192
#define SMEM_BV_MN_OFF 82944
#define SMEM_BV_MN_STAGE_BYTES 8192
#define SMEM_BV_MN_STRIDE 8192
#define SMEM_QG_PANEL_OFF 91136
#define SMEM_QG_PANEL_STAGE_BYTES 8192
#define SMEM_QG_PANEL_STRIDE 8192
#define SMEM_QG_OP_OFF 91136
#define SMEM_QG_OP_STAGE_BYTES 16384
#define SMEM_QG_OP_STRIDE 16384
#define SMEM_AQK_OP_OFF 107520
#define SMEM_AQK_OP_STAGE_BYTES 8192
#define SMEM_AQK_OP_STRIDE 8192
#define SMEM_AKK_OP_OFF 115712
#define SMEM_AKK_OP_STAGE_BYTES 8192
#define SMEM_AKK_OP_STRIDE 8192
#define SMEM_KB_PANEL_OFF 132096
#define SMEM_KB_PANEL_STAGE_BYTES 8192
#define SMEM_KB_PANEL_STRIDE 8192
#define SMEM_KB_MN_OFF 132096
#define SMEM_KB_MN_STAGE_BYTES 16384
#define SMEM_KB_MN_STRIDE 16384
#define SMEM_VB_RAW_OFF 164864
#define SMEM_VB_RAW_STAGE_BYTES 8192
#define SMEM_VB_RAW_STRIDE 8192
#define SMEM_VB_MN_OFF 164864
#define SMEM_VB_MN_STAGE_BYTES 8192
#define SMEM_VB_MN_STRIDE 8192
#define SMEM_TOTAL 181248
#define THREADS 384
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


extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_898a1eb972fbaf55789a(const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap kb_tma, float* __restrict__ gk, const __grid_constant__ CUtensorMap h_out, const __grid_constant__ CUtensorMap v_new, int num_heads, int* __restrict__ seq_chunk_start, int* __restrict__ seq_order, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap aqk_tma, __nv_bfloat16* __restrict__ o_out, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float scale, int write_tapes, const __grid_constant__ CUtensorMap v_tma, __nv_bfloat16* __restrict__ beta)
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
    #define qg_full_addr (mbar_base + 88)
    #define aqk_full_addr (mbar_base + 96)
    #define qg_empty_addr (mbar_base + 104)
    #define aqk_empty_addr (mbar_base + 112)
    #define o_done_addr (mbar_base + 120)
    #define o_drained_addr (mbar_base + 128)
    #define wy_full_addr (mbar_base + 136)
    #define vb_ready_addr (mbar_base + 152)
    #define wy_empty_addr (mbar_base + 168)
    #define w_full_addr (mbar_base + 184)
    #define w_ready_addr (mbar_base + 200)

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
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_addr = smem + 66560;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn_addr = smem + 66560;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_addr = smem + 82944;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int bv_mn_addr = smem + 82944;
    __nv_bfloat16* qg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int qg_panel_addr = smem + 91136;
    __nv_bfloat16* qg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int qg_op_addr = smem + 91136;
    __nv_bfloat16* aqk_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int aqk_op_addr = smem + 107520;
    __nv_bfloat16* akk_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int akk_op_addr = smem + 115712;
    __nv_bfloat16* kb_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int kb_panel_addr = smem + 132096;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int kb_mn_addr = smem + 132096;
    __nv_bfloat16* vb_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int vb_raw_addr = smem + 164864;
    __nv_bfloat16* vb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int vb_mn_addr = smem + 164864;

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 27 barriers)
    // Mbarriers at smem_raw[0..216)

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
            // h_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // h_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // bv_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // qg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // aqk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // qg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // aqk_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // o_drained: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // --- pipeline 'ring2' ---
            // wy_full: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // vb_ready: 2 barriers, init_count=64
            mbarrier_init(smem + 152, 64);
            mbarrier_init(smem + 160, 64);
            // wy_empty: 2 barriers, init_count=65
            mbarrier_init(smem + 168, 65);
            mbarrier_init(smem + 176, 65);
            // w_full: 2 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // w_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216);
    if (warp == 0) {
        int _tmem_hold = smem + 216;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 64;
    const int tmem_acc_qh = taddr + 128;
    const int tmem_acc_av = taddr + 192;
    const int tmem_acc_wt = taddr + 256;
    const int tmem_acc_u = taddr + 384;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int half = blockIdx.x % 2;
            int head = blockIdx.x / 2 % num_heads;
            int rank = blockIdx.x / (2 * num_heads);
            int seq = seq_order[rank];
            int c0 = seq_chunk_start[seq];
            int nc = seq_chunk_start[seq + 1] - c0;
            int row0 = c0 * 64;
            int vbase = half * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            unsigned int ubuf = 0;
            unsigned int wst_s = 0;
            float zeros16[16];
            zeros16[0] = 0.0f;
            zeros16[1] = 0.0f;
            zeros16[2] = 0.0f;
            zeros16[3] = 0.0f;
            zeros16[4] = 0.0f;
            zeros16[5] = 0.0f;
            zeros16[6] = 0.0f;
            zeros16[7] = 0.0f;
            zeros16[8] = 0.0f;
            zeros16[9] = 0.0f;
            zeros16[10] = 0.0f;
            zeros16[11] = 0.0f;
            zeros16[12] = 0.0f;
            zeros16[13] = 0.0f;
            zeros16[14] = 0.0f;
            zeros16[15] = 0.0f;
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, zeros16);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int _phase_bv_tape_read_0 = 1;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_tape_read_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            unsigned int _phase_dec_done_0 = 0;
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
                for (int q = 0; q < 2; q++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q * 8 / 64 * 16384 + k_row * 128 + q * 8 % 64 * 2 ^ (q * 8 / 64 * 16384 + k_row * 128 + q * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_16_bf16[q * 4]), "r"(_tmem_load_16_bf16[q * 4 + 1]), "r"(_tmem_load_16_bf16[q * 4 + 2]), "r"(_tmem_load_16_bf16[q * 4 + 3]) : "memory");
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
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_1 * 8) / 64 * 16384 + k_row * 128 + (16 + q_1 * 8) % 64 * 2 ^ ((16 + q_1 * 8) / 64 * 16384 + k_row * 128 + (16 + q_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_17_bf16[q_1 * 4]), "r"(_tmem_load_17_bf16[q_1 * 4 + 1]), "r"(_tmem_load_17_bf16[q_1 * 4 + 2]), "r"(_tmem_load_17_bf16[q_1 * 4 + 3]) : "memory");
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
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_2 * 8) / 64 * 16384 + k_row * 128 + (32 + q_2 * 8) % 64 * 2 ^ ((32 + q_2 * 8) / 64 * 16384 + k_row * 128 + (32 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[q_2 * 4]), "r"(_tmem_load_18_bf16[q_2 * 4 + 1]), "r"(_tmem_load_18_bf16[q_2 * 4 + 2]), "r"(_tmem_load_18_bf16[q_2 * 4 + 3]) : "memory");
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
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_3 * 8) / 64 * 16384 + k_row * 128 + (48 + q_3 * 8) % 64 * 2 ^ ((48 + q_3 * 8) / 64 * 16384 + k_row * 128 + (48 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[q_3 * 4]), "r"(_tmem_load_19_bf16[q_3 * 4 + 1]), "r"(_tmem_load_19_bf16[q_3 * 4 + 2]), "r"(_tmem_load_19_bf16[q_3 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                if (write_tapes == 1) {
                    mbarrier_wait(bv_tape_read_addr, _phase_bv_tape_read_0);
                    _phase_bv_tape_read_0 ^= 1;
                }
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int ucol = (int)ubuf * 64;
                float _tmem_load_20[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_20[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_21[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[15]))
                    : "r"(taddr + 384 + (unsigned int)vp_row_base + (unsigned int)ucol));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                unsigned int _tmem_load_21_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_21[_lp*2 + 0], _tmem_load_21[_lp*2+1 + 0]));
                    _tmem_load_21_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_8[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_tmem_load_21_bf16[_pair]) << 16);
                    (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_tmem_load_21_bf16[_pair]) & 0xffff0000u);
                }
                float bvv[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    bvv[j] = _unpack_bf16_8[j] - _tmem_load_20[j];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 ^ ((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[q_4 * 4]), "r"(bvv_bf16[q_4 * 4 + 1]), "r"(bvv_bf16[q_4 * 4 + 2]), "r"(bvv_bf16[q_4 * 4 + 3]) : "memory");
                }
                float _tmem_load_22[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_23[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[15]))
                    : "r"(taddr + 384 + (unsigned int)vp_row_base + (unsigned int)(ucol + 32)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_0 = 32 + lane_half * 16;
                unsigned int _tmem_load_23_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_23[_lp*2 + 0], _tmem_load_23[_lp*2+1 + 0]));
                    _tmem_load_23_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_9[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_tmem_load_23_bf16[_pair]) << 16);
                    (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_tmem_load_23_bf16[_pair]) & 0xffff0000u);
                }
                float bvv_1[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    bvv_1[j_1] = _unpack_bf16_9[j_1] - _tmem_load_22[j_1];
                }
                unsigned int bvv_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    bvv_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_5 = 0; q_5 < 2; q_5++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_0 + q_5 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_5 * 8) % 64 * 2 ^ ((col_0 + q_5 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_1_bf16[q_5 * 4]), "r"(bvv_1_bf16[q_5 * 4 + 1]), "r"(bvv_1_bf16[q_5 * 4 + 2]), "r"(bvv_1_bf16[q_5 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                ubuf += 1;
                if (ubuf == 2) { ubuf = 0; }
                if (write_tapes == 1) {
                    mbarrier_wait(h_tape_read_addr, _phase_h_tape_read_0);
                    _phase_h_tape_read_0 ^= 1;
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                _phase_dec_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
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
            int half_1 = blockIdx.x % 2;
            int head_1 = blockIdx.x / 2 % num_heads;
            int rank_1 = blockIdx.x / (2 * num_heads);
            int seq_1 = seq_order[rank_1];
            int c0_1 = seq_chunk_start[seq_1];
            int nc_1 = seq_chunk_start[seq_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int warp_in_wg_1 = warp % 4;
            int k_row_1 = warp_in_wg_1 * 32 + lane;
            int h_row_base_1 = warp_in_wg_1 * 32 << 16;
            int vbase_1 = half_1 * 64;
            int lane_half_1 = lane >> 4;
            int t_row_1 = warp_in_wg_1 * 16 + (lane & 15);
            int o_row_base = warp_in_wg_1 * 32 << 16;
            int ext_next = 0;
            int clen_next = 0;
            if (nc_1 > 0) {
                ext_next = chunk_bos[c0_1];
                clen_next = chunk_len[c0_1];
            }
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_next = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                gk_pre = gk[gn_next];
            }
            unsigned int wst_d = 0;
            unsigned int _phase_w_full = 0;
            if (nc_1 > 0) {
                mbarrier_wait(w_full_addr + (wst_d) * 8, _phase_w_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int wcol = (int)wst_d * 64;
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)wcol);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_6 = 0; q_6 < 2; q_6++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)(q_6 * 8 / 64 * 16384 + k_row_1 * 128 + q_6 * 8 % 64 * 2 ^ (q_6 * 8 / 64 * 16384 + k_row_1 * 128 + q_6 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[q_6 * 4]), "r"(_tmem_load_0_bf16[q_6 * 4 + 1]), "r"(_tmem_load_0_bf16[q_6 * 4 + 2]), "r"(_tmem_load_0_bf16[q_6 * 4 + 3]) : "memory");
                }
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)(wcol + 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_7 = 0; q_7 < 2; q_7++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)((16 + q_7 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_7 * 8) % 64 * 2 ^ ((16 + q_7 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[q_7 * 4]), "r"(_tmem_load_1_bf16[q_7 * 4 + 1]), "r"(_tmem_load_1_bf16[q_7 * 4 + 2]), "r"(_tmem_load_1_bf16[q_7 * 4 + 3]) : "memory");
                }
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)(wcol + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)((32 + q_8 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_8 * 8) % 64 * 2 ^ ((32 + q_8 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[q_8 * 4]), "r"(_tmem_load_2_bf16[q_8 * 4 + 1]), "r"(_tmem_load_2_bf16[q_8 * 4 + 2]), "r"(_tmem_load_2_bf16[q_8 * 4 + 3]) : "memory");
                }
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)(wcol + 48));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_9 = 0; q_9 < 2; q_9++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)((48 + q_9 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_9 * 8) % 64 * 2 ^ ((48 + q_9 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_9 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[q_9 * 4]), "r"(_tmem_load_3_bf16[q_9 * 4 + 1]), "r"(_tmem_load_3_bf16[q_9 * 4 + 2]), "r"(_tmem_load_3_bf16[q_9 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(w_ready_addr + (wst_d) * 8);
                wst_d += 1;
                if (wst_d == 2) { wst_d = 0; _phase_w_full ^= 1; }
            }
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_o_done_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float decay = approx_exp2(gk_pre);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    hs2[j_2] = _tmem_load_4[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1, hs2);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_0[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    hs2_0[j_3] = _tmem_load_5[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hs2_0);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_1[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    hs2_1[j_4] = _tmem_load_6[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hs2_1);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_2[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    hs2_2[j_5] = _tmem_load_7[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hs2_2);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
                if (nc_1 > c_1 + 1) {
                    long long gn_next_1 = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                    gk_pre = gk[gn_next_1];
                }
                int ext0 = ext_next;
                int clen = clen_next;
                if (nc_1 > c_1 + 1) {
                    ext_next = chunk_bos[c0_1 + c_1 + 1];
                    clen_next = chunk_len[c0_1 + c_1 + 1];
                }
                if (nc_1 > c_1 + 1) {
                    mbarrier_wait(w_full_addr + (wst_d) * 8, _phase_w_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int wcol_1 = (int)wst_d * 64;
                    float _tmem_load_8[16];
                    tmem_ld_x16(&_tmem_load_8[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)wcol_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int _tmem_load_8_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                        _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_10 = 0; q_10 < 2; q_10++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)(q_10 * 8 / 64 * 16384 + k_row_1 * 128 + q_10 * 8 % 64 * 2 ^ (q_10 * 8 / 64 * 16384 + k_row_1 * 128 + q_10 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[q_10 * 4]), "r"(_tmem_load_8_bf16[q_10 * 4 + 1]), "r"(_tmem_load_8_bf16[q_10 * 4 + 2]), "r"(_tmem_load_8_bf16[q_10 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_9[16];
                    tmem_ld_x16(&_tmem_load_9[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)(wcol_1 + 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int _tmem_load_9_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                        _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_11 = 0; q_11 < 2; q_11++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)((16 + q_11 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_11 * 8) % 64 * 2 ^ ((16 + q_11 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[q_11 * 4]), "r"(_tmem_load_9_bf16[q_11 * 4 + 1]), "r"(_tmem_load_9_bf16[q_11 * 4 + 2]), "r"(_tmem_load_9_bf16[q_11 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_10[16];
                    tmem_ld_x16(&_tmem_load_10[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)(wcol_1 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int _tmem_load_10_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                        _tmem_load_10_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_12 = 0; q_12 < 2; q_12++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)((32 + q_12 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_12 * 8) % 64 * 2 ^ ((32 + q_12 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_10_bf16[q_12 * 4]), "r"(_tmem_load_10_bf16[q_12 * 4 + 1]), "r"(_tmem_load_10_bf16[q_12 * 4 + 2]), "r"(_tmem_load_10_bf16[q_12 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_11[16];
                    tmem_ld_x16(&_tmem_load_11[0], taddr + 256 + (unsigned int)h_row_base_1 + (unsigned int)(wcol_1 + 48));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int _tmem_load_11_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                        _tmem_load_11_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_13 = 0; q_13 < 2; q_13++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + wst_d * 16384 + (unsigned int)((48 + q_13 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_13 * 8) % 64 * 2 ^ ((48 + q_13 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_13 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_11_bf16[q_13 * 4]), "r"(_tmem_load_11_bf16[q_13 * 4 + 1]), "r"(_tmem_load_11_bf16[q_13 * 4 + 2]), "r"(_tmem_load_11_bf16[q_13 * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(w_ready_addr + (wst_d) * 8);
                    wst_d += 1;
                    if (wst_d == 2) { wst_d = 0; _phase_w_full ^= 1; }
                }
                long long obase = ((long long)(ext0 + t_row_1) * (long long)num_heads + (long long)head_1) * 128 + (long long)vbase_1;
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int col_1 = lane_half_1 * 16;
                float _tmem_load_12[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                    : "r"(taddr + 128 + (unsigned int)o_row_base));
                float _tmem_load_13[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                    : "r"(taddr + 192 + (unsigned int)o_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov[16];
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    ov[j_6] = __fmaf_rn(_tmem_load_12[j_6], scale, _tmem_load_13[j_6]);
                }
                if (t_row_1 < clen) {
                    unsigned int _packed_0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 0], ov[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 2], ov[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 4], ov[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 6], ov[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 8], ov[_lp*2+1 + 8]));
                        _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 10], ov[_lp*2+1 + 10]));
                        _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 12], ov[_lp*2+1 + 12]));
                        _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 14], ov[_lp*2+1 + 14]));
                        _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_1))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                    }
                }
                int col_3 = 32 + lane_half_1 * 16;
                float _tmem_load_14[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                    : "r"(taddr + 128 + (unsigned int)o_row_base + 32));
                float _tmem_load_15[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                    : "r"(taddr + 192 + (unsigned int)o_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_4[16];
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    ov_4[j_7] = __fmaf_rn(_tmem_load_14[j_7], scale, _tmem_load_15[j_7]);
                }
                if (t_row_1 < clen) {
                    unsigned int _packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 0], ov_4[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 2], ov_4[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 4], ov_4[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 6], ov_4[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 8], ov_4[_lp*2+1 + 8]));
                        _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 10], ov_4[_lp*2+1 + 10]));
                        _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 12], ov_4[_lp*2+1 + 12]));
                        _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_4[_lp*2 + 14], ov_4[_lp*2+1 + 14]));
                        _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_3))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_drained_addr);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int half_2 = blockIdx.x % 2;
            int head_2 = blockIdx.x / 2 % num_heads;
            int rank_2 = blockIdx.x / (2 * num_heads);
            int seq_2 = seq_order[rank_2];
            int c0_2 = seq_chunk_start[seq_2];
            int nc_2 = seq_chunk_start[seq_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int vbase_2 = half_2 * 64;
            unsigned int stage = 0;
            int aq_tok0 = 0;
            if (nc_2 > 0) {
                aq_tok0 = chunk_bos[c0_2];
            }
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_qg_empty_0 = 1;
            unsigned int _phase_aqk_empty_0 = 1;
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
                    mbarrier_wait(qg_empty_addr, _phase_qg_empty_0);
                    _phase_qg_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                    int trow_0 = row0_2;
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)seg_1 * 8192, (&qg_tma), seg_1 * 64, head_2, trow_0, qg_full_addr);
                    }
                    if (nc_2 > 1) {
                        int trow2 = row0_2 + 64;
                        #pragma unroll
                        for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&qg_tma))), "r"((int)(seg_2 * 64)), "r"((int)(head_2)), "r"((int)(trow2)) : "memory");
                        }
                    }
                    mbarrier_wait(aqk_empty_addr, _phase_aqk_empty_0);
                    _phase_aqk_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                    int trow_1 = aq_tok0;
                    tma_3d_gmem2smem(aqk_op_addr, (&aqk_tma), 0, head_2, trow_1, aqk_full_addr);
                    if (nc_2 > 1) {
                        int trow2_1 = aq_tok0 + 64;
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&aqk_tma))), "r"((int)(0)), "r"((int)(head_2)), "r"((int)(trow2_1)) : "memory");
                    }
                }
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 16384);
                        int trow_2 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                            tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg_3) * 8192, (&kg_tma), seg_3 * 64, head_2, trow_2, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    if (write_tapes == 1) {
                        int chunk = (c0_2 + c_2) * num_heads + head_2;
                        mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                        _phase_hs_ready_0_1 ^= 1;
                        tma_store_3d((&h_out), vbase_2, 0, chunk, hs_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(qg_empty_addr, _phase_qg_empty_0);
                        _phase_qg_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                        int trow_3 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_4 = 0; seg_4 < 2; seg_4++) {
                            tma_3d_gmem2smem(qg_panel_addr + (unsigned int)seg_4 * 8192, (&qg_tma), seg_4 * 64, head_2, trow_3, qg_full_addr);
                        }
                        if (nc_2 > c_2 + 1 + 1) {
                            int trow2_2 = row0_2 + (c_2 + 1 + 1) * 64;
                            #pragma unroll
                            for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&qg_tma))), "r"((int)(seg_5 * 64)), "r"((int)(head_2)), "r"((int)(trow2_2)) : "memory");
                            }
                        }
                    }
                    if (write_tapes == 1) {
                        int trow_4 = row0_2 + c_2 * 64;
                        mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                        _phase_bv_ready_0 ^= 1;
                        tma_store_3d((&v_new), vbase_2, head_2, trow_4, bv_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        mbarrier_arrive(h_tape_read_addr);
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(bv_tape_read_addr);
                    }
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(aqk_empty_addr, _phase_aqk_empty_0);
                        _phase_aqk_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                        int trow_5 = aq_tok0 + (c_2 + 1) * 64;
                        tma_3d_gmem2smem(aqk_op_addr, (&aqk_tma), 0, head_2, trow_5, aqk_full_addr);
                        if (nc_2 > c_2 + 1 + 1) {
                            int trow2_3 = aq_tok0 + (c_2 + 1 + 1) * 64;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&aqk_tma))), "r"((int)(0)), "r"((int)(head_2)), "r"((int)(trow2_3)) : "memory");
                        }
                    }
                }
                if (write_tapes == 1) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int half_3 = blockIdx.x % 2;
            int head_3 = blockIdx.x / 2 % num_heads;
            int rank_3 = blockIdx.x / (2 * num_heads);
            int seq_3 = seq_order[rank_3];
            int c0_3 = seq_chunk_start[seq_3];
            int nc_3 = seq_chunk_start[seq_3 + 1] - c0_3;
            int row0_3 = c0_3 * 64;
            unsigned int stage_1 = 0;
            unsigned int ubuf_1 = 0;
            unsigned int wst_m = 0;
            unsigned int mst = 0;
            unsigned int _phase_wy_full = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_w_ready = 0;
            unsigned int _phase_hs_ready_0_2 = 0;
            unsigned int _phase_qg_full_0 = 0;
            unsigned int _phase_o_drained_0 = 1;
            unsigned int _phase_bv_ready_0_1 = 0;
            unsigned int _phase_dec_done_0_1 = 0;
            unsigned int _phase_aqk_full_0 = 0;
            if (elect_sync()) {
                if (nc_3 > 0) {
                    mbarrier_wait(wy_full_addr + (wst_m) * 8, _phase_wy_full);
                    mbarrier_wait(vb_ready_addr + (wst_m) * 8, _phase_wy_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (wst_m) * 1024;
                    int _smem_desc_lo_1 = ((((akk_op_addr) >> 4) & 0x7FFF) | 0x2000000) + (wst_m) * 512;
#else
                    int _smem_desc_lo_0 = ((((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (wst_m) * 1024;
                    int _smem_desc_lo_1 = ((((akk_op_addr) >> 4) & 0x3FFF) | 0x2000000) + (wst_m) * 512;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc_wt + (wst_m * 64))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((akk_op_addr) >> 4) & 0x7FFF) + (wst_m) * 512;
                    int _smem_desc_lo_3 = ((((vb_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (wst_m) * 512;
#else
                    int _smem_desc_lo_2 = (((akk_op_addr) >> 4) & 0x3FFF) + (wst_m) * 512;
                    int _smem_desc_lo_3 = ((((vb_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (wst_m) * 512;
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
                    "mov.b32 id, 68224144;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc_u + (ubuf_1 * 64))), "r"(0) : "memory");
                    tcgen05_commit(w_full_addr + (wst_m) * 8);
                    tcgen05_commit(wy_empty_addr + (wst_m) * 8);
                    ubuf_1 += 1;
                    if (ubuf_1 == 2) { ubuf_1 = 0; }
                    wst_m += 1;
                    if (wst_m == 2) { wst_m = 0; _phase_wy_full ^= 1; }
                }
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(w_ready_addr + (mst) * 8, _phase_w_ready);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_2);
                    _phase_hs_ready_0_2 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = ((((w_mn_addr) >> 4) & 0x7FFF) | 0x4000000) + (mst) * 1024;
                    int _mma_b_lo_0 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _mma_a_lo_0 = ((((w_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mst) * 1024;
                    int _mma_b_lo_0 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 68256912;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mst += 1;
                    if (mst == 2) { mst = 0; _phase_w_ready ^= 1; }
                    if (nc_3 > c_3 + 1) {
                        mbarrier_wait(wy_full_addr + (wst_m) * 8, _phase_wy_full);
                        mbarrier_wait(vb_ready_addr + (wst_m) * 8, _phase_wy_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_4 = ((((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (wst_m) * 1024;
                        int _smem_desc_lo_5 = ((((akk_op_addr) >> 4) & 0x7FFF) | 0x2000000) + (wst_m) * 512;
#else
                        int _smem_desc_lo_4 = ((((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (wst_m) * 1024;
                        int _smem_desc_lo_5 = ((((akk_op_addr) >> 4) & 0x3FFF) | 0x2000000) + (wst_m) * 512;
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"((tmem_acc_wt + (wst_m * 64))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((akk_op_addr) >> 4) & 0x7FFF) + (wst_m) * 512;
                        int _smem_desc_lo_7 = ((((vb_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (wst_m) * 512;
#else
                        int _smem_desc_lo_6 = (((akk_op_addr) >> 4) & 0x3FFF) + (wst_m) * 512;
                        int _smem_desc_lo_7 = ((((vb_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (wst_m) * 512;
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
                    "mov.b32 id, 68224144;\n\t"
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_acc_u + (ubuf_1 * 64))), "r"(0) : "memory");
                        tcgen05_commit(w_full_addr + (wst_m) * 8);
                        tcgen05_commit(wy_empty_addr + (wst_m) * 8);
                        ubuf_1 += 1;
                        if (ubuf_1 == 2) { ubuf_1 = 0; }
                        wst_m += 1;
                        if (wst_m == 2) { wst_m = 0; _phase_wy_full ^= 1; }
                    }
                    mbarrier_wait(qg_full_addr, _phase_qg_full_0);
                    _phase_qg_full_0 ^= 1;
                    mbarrier_wait(o_drained_addr, _phase_o_drained_0);
                    _phase_o_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = ((qg_op_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_8 = ((qg_op_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68224144;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_mma_b_lo_0), "r"(tmem_acc_qh), "r"(0) : "memory");
                    tcgen05_commit(qg_empty_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0_1);
                    _phase_bv_ready_0_1 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0_1);
                    _phase_dec_done_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_1 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _mma_b_lo_1 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _mma_a_lo_1 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _mma_b_lo_1 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135365776;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    mbarrier_wait(aqk_full_addr, _phase_aqk_full_0);
                    _phase_aqk_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_9 = ((aqk_op_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_9 = ((aqk_op_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68224144;\n\t"
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
                    :: "r"(_smem_desc_lo_9), "r"(_mma_b_lo_1), "r"(tmem_acc_av), "r"(0) : "memory");
                    tcgen05_commit(aqk_empty_addr);
                    tcgen05_commit(o_done_addr);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: cvt ----
    if (warp >= 10 && warp <= 11) {
        { // cvt_main
            int half_4 = blockIdx.x % 2;
            int head_4 = blockIdx.x / 2 % num_heads;
            int rank_4 = blockIdx.x / (2 * num_heads);
            int seq_4 = seq_order[rank_4];
            int c0_4 = seq_chunk_start[seq_4];
            int nc_4 = seq_chunk_start[seq_4 + 1] - c0_4;
            int row0_4 = c0_4 * 64;
            const int cvt_tid = (warp - 10) * 32 + lane;
            unsigned int wst_c = 0;
            int vbase_3 = half_4 * 64;
            unsigned int wst_i = 0;
            int bos_a = 0;
            int bos_b = 0;
            if (nc_4 > 0) {
                bos_a = chunk_bos[c0_4];
            }
            if (nc_4 > 1) {
                bos_b = chunk_bos[c0_4 + 1];
            }
            unsigned int _phase_wy_empty = 1;
            if (warp == 10) {
                if (elect_sync()) {
                    if (nc_4 > 0) {
                        mbarrier_wait(wy_empty_addr + (wst_i) * 8, _phase_wy_empty);
                        mbarrier_arrive_expect_tx(wy_full_addr + (wst_i) * 8, 32768);
                        int trow_6 = row0_4;
                        tma_3d_gmem2smem(akk_op_addr + wst_i * 8192, (&akk_tma), 0, head_4, bos_a, wy_full_addr + (wst_i) * 8);
                        #pragma unroll
                        for (int seg_6 = 0; seg_6 < 2; seg_6++) {
                            tma_3d_gmem2smem(kb_panel_addr + (wst_i * 2 + (unsigned int)seg_6) * 8192, (&kb_tma), seg_6 * 64, head_4, trow_6, wy_full_addr + (wst_i) * 8);
                        }
                        tma_3d_gmem2smem(vb_raw_addr + wst_i * 8192, (&v_tma), vbase_3, head_4, bos_a, wy_full_addr + (wst_i) * 8);
                        if (nc_4 > 1) {
                            int trow2_4 = row0_4 + 64;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                            #pragma unroll
                            for (int seg_7 = 0; seg_7 < 2; seg_7++) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kb_tma))), "r"((int)(seg_7 * 64)), "r"((int)(head_4)), "r"((int)(trow2_4)) : "memory");
                            }
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(vbase_3)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                        }
                        bos_a = bos_b;
                        if (nc_4 > 2) {
                            bos_b = chunk_bos[c0_4 + 2];
                        }
                        wst_i += 1;
                        if (wst_i == 2) { wst_i = 0; _phase_wy_empty ^= 1; }
                    }
                    if (nc_4 > 1) {
                        mbarrier_wait(wy_empty_addr + (wst_i) * 8, _phase_wy_empty);
                        mbarrier_arrive_expect_tx(wy_full_addr + (wst_i) * 8, 32768);
                        int trow_7 = row0_4 + 64;
                        tma_3d_gmem2smem(akk_op_addr + wst_i * 8192, (&akk_tma), 0, head_4, bos_a, wy_full_addr + (wst_i) * 8);
                        #pragma unroll
                        for (int seg_8 = 0; seg_8 < 2; seg_8++) {
                            tma_3d_gmem2smem(kb_panel_addr + (wst_i * 2 + (unsigned int)seg_8) * 8192, (&kb_tma), seg_8 * 64, head_4, trow_7, wy_full_addr + (wst_i) * 8);
                        }
                        tma_3d_gmem2smem(vb_raw_addr + wst_i * 8192, (&v_tma), vbase_3, head_4, bos_a, wy_full_addr + (wst_i) * 8);
                        if (nc_4 > 2) {
                            int trow2_5 = row0_4 + 128;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                            #pragma unroll
                            for (int seg_9 = 0; seg_9 < 2; seg_9++) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kb_tma))), "r"((int)(seg_9 * 64)), "r"((int)(head_4)), "r"((int)(trow2_5)) : "memory");
                            }
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(vbase_3)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                        }
                        bos_a = bos_b;
                        if (nc_4 > 3) {
                            bos_b = chunk_bos[c0_4 + 1 + 2];
                        }
                        wst_i += 1;
                        if (wst_i == 2) { wst_i = 0; _phase_wy_empty ^= 1; }
                    }
                }
            }
            int bos_n = 0;
            int clen_n = 0;
            if (nc_4 > 0) {
                bos_n = chunk_bos[c0_4];
                clen_n = chunk_len[c0_4];
            }
            float bt_n[1];
            int t_r = cvt_tid;
            int clast = clen_n - 1;
            if (clast < 0) {
                clast = 0;
            }
            int t_c = t_r;
            if (t_c > clast) {
                t_c = clast;
            }
            if (nc_4 > 0) {
                long long bidx = (long long)(bos_n + t_c) * (long long)num_heads + (long long)head_4;
                bt_n[0] = (float)beta[bidx];
            }
            unsigned int _phase_wy_full_1 = 0;
            #pragma unroll 1
            for (int j_8 = 0; j_8 < nc_4; j_8++) {
                int clen_j = clen_n;
                float bt_j[1];
                bt_j[0] = bt_n[0];
                if (nc_4 > j_8 + 1) {
                    bos_n = chunk_bos[c0_4 + j_8 + 1];
                    clen_n = chunk_len[c0_4 + j_8 + 1];
                    int t_r_0 = cvt_tid;
                    int clast_1 = clen_n - 1;
                    if (clast_1 < 0) {
                        clast_1 = 0;
                    }
                    int t_c_2 = t_r_0;
                    if (t_c_2 > clast_1) {
                        t_c_2 = clast_1;
                    }
                    long long bidx_1 = (long long)(bos_n + t_c_2) * (long long)num_heads + (long long)head_4;
                    bt_n[0] = (float)beta[bidx_1];
                }
                mbarrier_wait(wy_full_addr + (wst_c) * 8, _phase_wy_full_1);
                int t_r_0_1 = cvt_tid;
                int valid = 0;
                if (t_r_0_1 < clen_j) {
                    valid = 1;
                }
                float bt = 0.0f;
                if (valid == 1) {
                    bt = bt_j[0];
                }
                unsigned int mask = (unsigned int)(-valid);
                unsigned int addr = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 ^ (t_r_0_1 * 128 >> 7 & 7) << 4));
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
                unsigned int addr_1 = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 + 16 ^ (t_r_0_1 * 128 + 16 >> 7 & 7) << 4));
                unsigned int raw_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 3]))
                    : "r"(addr_1));
                unsigned int sel_3[4];
                #pragma unroll
                for (int i_1 = 0; i_1 < 4; i_1++) {
                    sel_3[i_1] = raw_2[i_1] & mask;
                }
                float _unpack_bf16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_3[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_3[_pair]) & 0xffff0000u);
                }
                float prod_4[8];
                #pragma unroll
                for (int e_1 = 0; e_1 < 8; e_1++) {
                    prod_4[e_1] = _unpack_bf16_1[e_1] * bt;
                }
                unsigned int packed_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_4[_lp*2 + 0], prod_4[_lp*2+1 + 0]));
                    packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1), "r"(packed_5[0]), "r"(packed_5[1]), "r"(packed_5[2]), "r"(packed_5[3]) : "memory");
                unsigned int addr_6 = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 + 32 ^ (t_r_0_1 * 128 + 32 >> 7 & 7) << 4));
                unsigned int raw_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_7[(0) + 3]))
                    : "r"(addr_6));
                unsigned int sel_8[4];
                #pragma unroll
                for (int i_2 = 0; i_2 < 4; i_2++) {
                    sel_8[i_2] = raw_7[i_2] & mask;
                }
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_8[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_8[_pair]) & 0xffff0000u);
                }
                float prod_9[8];
                #pragma unroll
                for (int e_2 = 0; e_2 < 8; e_2++) {
                    prod_9[e_2] = _unpack_bf16_2[e_2] * bt;
                }
                unsigned int packed_10[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_9[_lp*2 + 0], prod_9[_lp*2+1 + 0]));
                    packed_10[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_6), "r"(packed_10[0]), "r"(packed_10[1]), "r"(packed_10[2]), "r"(packed_10[3]) : "memory");
                unsigned int addr_11 = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 + 48 ^ (t_r_0_1 * 128 + 48 >> 7 & 7) << 4));
                unsigned int raw_12[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_12[(0) + 3]))
                    : "r"(addr_11));
                unsigned int sel_13[4];
                #pragma unroll
                for (int i_3 = 0; i_3 < 4; i_3++) {
                    sel_13[i_3] = raw_12[i_3] & mask;
                }
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_13[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_13[_pair]) & 0xffff0000u);
                }
                float prod_14[8];
                #pragma unroll
                for (int e_3 = 0; e_3 < 8; e_3++) {
                    prod_14[e_3] = _unpack_bf16_3[e_3] * bt;
                }
                unsigned int packed_15[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_14[_lp*2 + 0], prod_14[_lp*2+1 + 0]));
                    packed_15[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_11), "r"(packed_15[0]), "r"(packed_15[1]), "r"(packed_15[2]), "r"(packed_15[3]) : "memory");
                unsigned int addr_16 = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 + 64 ^ (t_r_0_1 * 128 + 64 >> 7 & 7) << 4));
                unsigned int raw_17[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_17[(0) + 3]))
                    : "r"(addr_16));
                unsigned int sel_18[4];
                #pragma unroll
                for (int i_4 = 0; i_4 < 4; i_4++) {
                    sel_18[i_4] = raw_17[i_4] & mask;
                }
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_18[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_18[_pair]) & 0xffff0000u);
                }
                float prod_19[8];
                #pragma unroll
                for (int e_4 = 0; e_4 < 8; e_4++) {
                    prod_19[e_4] = _unpack_bf16_4[e_4] * bt;
                }
                unsigned int packed_20[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_19[_lp*2 + 0], prod_19[_lp*2+1 + 0]));
                    packed_20[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_16), "r"(packed_20[0]), "r"(packed_20[1]), "r"(packed_20[2]), "r"(packed_20[3]) : "memory");
                unsigned int addr_21 = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 + 80 ^ (t_r_0_1 * 128 + 80 >> 7 & 7) << 4));
                unsigned int raw_22[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_22[(0) + 3]))
                    : "r"(addr_21));
                unsigned int sel_23[4];
                #pragma unroll
                for (int i_5 = 0; i_5 < 4; i_5++) {
                    sel_23[i_5] = raw_22[i_5] & mask;
                }
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_23[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_23[_pair]) & 0xffff0000u);
                }
                float prod_24[8];
                #pragma unroll
                for (int e_5 = 0; e_5 < 8; e_5++) {
                    prod_24[e_5] = _unpack_bf16_5[e_5] * bt;
                }
                unsigned int packed_25[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_24[_lp*2 + 0], prod_24[_lp*2+1 + 0]));
                    packed_25[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_21), "r"(packed_25[0]), "r"(packed_25[1]), "r"(packed_25[2]), "r"(packed_25[3]) : "memory");
                unsigned int addr_26 = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 + 96 ^ (t_r_0_1 * 128 + 96 >> 7 & 7) << 4));
                unsigned int raw_27[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_27[(0) + 3]))
                    : "r"(addr_26));
                unsigned int sel_28[4];
                #pragma unroll
                for (int i_6 = 0; i_6 < 4; i_6++) {
                    sel_28[i_6] = raw_27[i_6] & mask;
                }
                float _unpack_bf16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_28[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_28[_pair]) & 0xffff0000u);
                }
                float prod_29[8];
                #pragma unroll
                for (int e_6 = 0; e_6 < 8; e_6++) {
                    prod_29[e_6] = _unpack_bf16_6[e_6] * bt;
                }
                unsigned int packed_30[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_29[_lp*2 + 0], prod_29[_lp*2+1 + 0]));
                    packed_30[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_26), "r"(packed_30[0]), "r"(packed_30[1]), "r"(packed_30[2]), "r"(packed_30[3]) : "memory");
                unsigned int addr_31 = (vb_raw_addr + wst_c * 8192 + (unsigned int)(t_r_0_1 * 128 + 112 ^ (t_r_0_1 * 128 + 112 >> 7 & 7) << 4));
                unsigned int raw_32[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_32[(0) + 3]))
                    : "r"(addr_31));
                unsigned int sel_33[4];
                #pragma unroll
                for (int i_7 = 0; i_7 < 4; i_7++) {
                    sel_33[i_7] = raw_32[i_7] & mask;
                }
                float _unpack_bf16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_33[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_33[_pair]) & 0xffff0000u);
                }
                float prod_34[8];
                #pragma unroll
                for (int e_7 = 0; e_7 < 8; e_7++) {
                    prod_34[e_7] = _unpack_bf16_7[e_7] * bt;
                }
                unsigned int packed_35[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_34[_lp*2 + 0], prod_34[_lp*2+1 + 0]));
                    packed_35[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_31), "r"(packed_35[0]), "r"(packed_35[1]), "r"(packed_35[2]), "r"(packed_35[3]) : "memory");
                if (clen_j < 64) {
                    unsigned int zeros4[4];
                    zeros4[0] = 0;
                    zeros4[1] = 0;
                    zeros4[2] = 0;
                    zeros4[3] = 0;
                    unsigned int akk_base = akk_op_addr + wst_c * 8192;
                    int npad16 = (64 - clen_j) * 8;
                    #pragma unroll 1
                    for (int idx = cvt_tid; idx < npad16; idx += 64) {
                        int off = (clen_j + (idx >> 3)) * 128 + (idx & 7) * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(akk_base + (unsigned int)off), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(vb_ready_addr + (wst_c) * 8);
                mbarrier_arrive(wy_empty_addr + (wst_c) * 8);
                wst_c += 1;
                if (wst_c == 2) { wst_c = 0; _phase_wy_full_1 ^= 1; }
                __syncwarp();
                if (warp == 10) {
                    if (elect_sync()) {
                        if (nc_4 > j_8 + 2) {
                            mbarrier_wait(wy_empty_addr + (wst_i) * 8, _phase_wy_empty);
                            mbarrier_arrive_expect_tx(wy_full_addr + (wst_i) * 8, 32768);
                            int trow_8 = row0_4 + (j_8 + 2) * 64;
                            tma_3d_gmem2smem(akk_op_addr + wst_i * 8192, (&akk_tma), 0, head_4, bos_a, wy_full_addr + (wst_i) * 8);
                            #pragma unroll
                            for (int seg_10 = 0; seg_10 < 2; seg_10++) {
                                tma_3d_gmem2smem(kb_panel_addr + (wst_i * 2 + (unsigned int)seg_10) * 8192, (&kb_tma), seg_10 * 64, head_4, trow_8, wy_full_addr + (wst_i) * 8);
                            }
                            tma_3d_gmem2smem(vb_raw_addr + wst_i * 8192, (&v_tma), vbase_3, head_4, bos_a, wy_full_addr + (wst_i) * 8);
                            if (nc_4 > j_8 + 2 + 1) {
                                int trow2_6 = row0_4 + (j_8 + 2 + 1) * 64;
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                                #pragma unroll
                                for (int seg_11 = 0; seg_11 < 2; seg_11++) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kb_tma))), "r"((int)(seg_11 * 64)), "r"((int)(head_4)), "r"((int)(trow2_6)) : "memory");
                                }
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(vbase_3)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                            }
                            bos_a = bos_b;
                            if (nc_4 > j_8 + 2 + 2) {
                                bos_b = chunk_bos[c0_4 + (j_8 + 2) + 2];
                            }
                            wst_i += 1;
                            if (wst_i == 2) { wst_i = 0; _phase_wy_empty ^= 1; }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
