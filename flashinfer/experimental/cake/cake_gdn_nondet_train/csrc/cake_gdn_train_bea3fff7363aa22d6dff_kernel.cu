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


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_bea3fff7363aa22d6dff(const __grid_constant__ CUtensorMap kn_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap a_tma, float* __restrict__ g_cs, float* __restrict__ beta, float* __restrict__ b_out, float* __restrict__ m_out, int* __restrict__ seg_flag, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ walk, int* __restrict__ m_slot, int num_heads, int num_v_heads, int b_head, int b_row, int m_head, int m_row, int m_col)
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

    const int mbar_base = smem + 184320;
    #define kq_full_addr (mbar_base + 0)
    #define kq_fixed_addr (mbar_base + 24)
    #define kq_free_addr (mbar_base + 48)
    #define va_full_addr (mbar_base + 72)
    #define v_fixed_addr (mbar_base + 88)
    #define va_free_addr (mbar_base + 104)
    #define bk_ready_addr (mbar_base + 120)
    #define bk_free_addr (mbar_base + 128)
    #define gates_ready_addr (mbar_base + 136)
    #define gates_free_addr (mbar_base + 168)
    #define hs_ready_b_addr (mbar_base + 200)
    #define ws_full_b_addr (mbar_base + 208)
    #define vp_ready_b_addr (mbar_base + 216)
    #define s_full_b_addr (mbar_base + 224)
    #define u_full_addr (mbar_base + 232)
    #define hs_ready_m_addr (mbar_base + 240)
    #define ws_full_m_addr (mbar_base + 248)
    #define vp_ready_m_addr (mbar_base + 256)
    #define s_full_m_addr (mbar_base + 264)
    #define w_full_addr (mbar_base + 272)
    #define wt_ready_addr (mbar_base + 280)
    #define u_taken_addr (mbar_base + 288)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* s_kT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kT_addr = smem + 0;
    __nv_bfloat16* s_vT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_vT_addr = smem + 98304;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 114688);
    const int s_a_addr = smem + 114688;
    __nv_bfloat16* s_bk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 147456);
    const int s_bk_addr = smem + 147456;
    __nv_bfloat16* s_wn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int s_wn_addr = smem + 163840;
    float* s_gt = reinterpret_cast<float*>(smem_raw + 180224);
    const int s_gt_addr = smem + 180224;

    // Mbarrier init (37 barriers at smem_raw[184320..184616))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kqring' ---
            // kq_full: 3 barriers, init_count=1
            mbarrier_init(smem + 184320, 1);
            mbarrier_init(smem + 184328, 1);
            mbarrier_init(smem + 184336, 1);
            // kq_fixed: 3 barriers, init_count=128
            mbarrier_init(smem + 184344, 128);
            mbarrier_init(smem + 184352, 128);
            mbarrier_init(smem + 184360, 128);
            // kq_free: 3 barriers, init_count=1
            mbarrier_init(smem + 184368, 1);
            mbarrier_init(smem + 184376, 1);
            mbarrier_init(smem + 184384, 1);
            // --- pipeline 'varing' ---
            // va_full: 2 barriers, init_count=1
            mbarrier_init(smem + 184392, 1);
            mbarrier_init(smem + 184400, 1);
            // v_fixed: 2 barriers, init_count=128
            mbarrier_init(smem + 184408, 128);
            mbarrier_init(smem + 184416, 128);
            // va_free: 2 barriers, init_count=1
            mbarrier_init(smem + 184424, 1);
            mbarrier_init(smem + 184432, 1);
            // --- pipeline 'one' ---
            // bk_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 184440, 128);
            // bk_free: 1 barriers, init_count=1
            mbarrier_init(smem + 184448, 1);
            // --- pipeline 'gring' ---
            // gates_ready: 4 barriers, init_count=32
            mbarrier_init(smem + 184456, 32);
            mbarrier_init(smem + 184464, 32);
            mbarrier_init(smem + 184472, 32);
            mbarrier_init(smem + 184480, 32);
            // gates_free: 4 barriers, init_count=384
            mbarrier_init(smem + 184488, 384);
            mbarrier_init(smem + 184496, 384);
            mbarrier_init(smem + 184504, 384);
            mbarrier_init(smem + 184512, 384);
            // --- pipeline 'one' ---
            // hs_ready_b: 1 barriers, init_count=128
            mbarrier_init(smem + 184520, 128);
            // ws_full_b: 1 barriers, init_count=1
            mbarrier_init(smem + 184528, 1);
            // vp_ready_b: 1 barriers, init_count=128
            mbarrier_init(smem + 184536, 128);
            // s_full_b: 1 barriers, init_count=1
            mbarrier_init(smem + 184544, 1);
            // u_full: 1 barriers, init_count=1
            mbarrier_init(smem + 184552, 1);
            // hs_ready_m: 1 barriers, init_count=128
            mbarrier_init(smem + 184560, 128);
            // ws_full_m: 1 barriers, init_count=1
            mbarrier_init(smem + 184568, 1);
            // vp_ready_m: 1 barriers, init_count=128
            mbarrier_init(smem + 184576, 128);
            // s_full_m: 1 barriers, init_count=1
            mbarrier_init(smem + 184584, 1);
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 184592, 1);
            // wt_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 184600, 128);
            // u_taken: 1 barriers, init_count=128
            mbarrier_init(smem + 184608, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184616);
    if (warp == 0) {
        int _tmem_hold = smem + 184616;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_sb = taddr;
    const unsigned int tmem_sm = taddr + 128;
    const unsigned int tmem_hb = taddr + 256;
    const unsigned int tmem_hm = taddr + 320;
    const unsigned int tmem_vb = taddr + 256;
    const unsigned int tmem_vm = taddr + 320;
    const unsigned int tmem_wb = taddr + 384;
    const unsigned int tmem_wm = taddr + 448;

    // Kernel post-init ops
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 120;");
    }

    // ---- Role: chb ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // chb_main
            int blk = blockIdx.x;
            int seg = walk[blk / num_v_heads];
            int hv = blk % num_v_heads;
            int c_begin = seg_chunk_start[seg];
            int nc = seg_chunk_start[seg + 1] - c_begin;
            int vrow = warp % 4 * 32 + lane;
            int q4 = warp % 4;
            int row_off = q4 * 32 << 16;
            float blk_0[32];
            blk_0[0] = 0.0f;
            blk_0[1] = 0.0f;
            blk_0[2] = 0.0f;
            blk_0[3] = 0.0f;
            blk_0[4] = 0.0f;
            blk_0[5] = 0.0f;
            blk_0[6] = 0.0f;
            blk_0[7] = 0.0f;
            blk_0[8] = 0.0f;
            blk_0[9] = 0.0f;
            blk_0[10] = 0.0f;
            blk_0[11] = 0.0f;
            blk_0[12] = 0.0f;
            blk_0[13] = 0.0f;
            blk_0[14] = 0.0f;
            blk_0[15] = 0.0f;
            blk_0[16] = 0.0f;
            blk_0[17] = 0.0f;
            blk_0[18] = 0.0f;
            blk_0[19] = 0.0f;
            blk_0[20] = 0.0f;
            blk_0[21] = 0.0f;
            blk_0[22] = 0.0f;
            blk_0[23] = 0.0f;
            blk_0[24] = 0.0f;
            blk_0[25] = 0.0f;
            blk_0[26] = 0.0f;
            blk_0[27] = 0.0f;
            blk_0[28] = 0.0f;
            blk_0[29] = 0.0f;
            blk_0[30] = 0.0f;
            blk_0[31] = 0.0f;
            tmem_st_x32_f32(tmem_sb + row_off + 0, blk_0);
            float blk_1[32];
            blk_1[0] = 0.0f;
            blk_1[1] = 0.0f;
            blk_1[2] = 0.0f;
            blk_1[3] = 0.0f;
            blk_1[4] = 0.0f;
            blk_1[5] = 0.0f;
            blk_1[6] = 0.0f;
            blk_1[7] = 0.0f;
            blk_1[8] = 0.0f;
            blk_1[9] = 0.0f;
            blk_1[10] = 0.0f;
            blk_1[11] = 0.0f;
            blk_1[12] = 0.0f;
            blk_1[13] = 0.0f;
            blk_1[14] = 0.0f;
            blk_1[15] = 0.0f;
            blk_1[16] = 0.0f;
            blk_1[17] = 0.0f;
            blk_1[18] = 0.0f;
            blk_1[19] = 0.0f;
            blk_1[20] = 0.0f;
            blk_1[21] = 0.0f;
            blk_1[22] = 0.0f;
            blk_1[23] = 0.0f;
            blk_1[24] = 0.0f;
            blk_1[25] = 0.0f;
            blk_1[26] = 0.0f;
            blk_1[27] = 0.0f;
            blk_1[28] = 0.0f;
            blk_1[29] = 0.0f;
            blk_1[30] = 0.0f;
            blk_1[31] = 0.0f;
            tmem_st_x32_f32(tmem_sb + row_off + 32, blk_1);
            float blk_2[32];
            blk_2[0] = 0.0f;
            blk_2[1] = 0.0f;
            blk_2[2] = 0.0f;
            blk_2[3] = 0.0f;
            blk_2[4] = 0.0f;
            blk_2[5] = 0.0f;
            blk_2[6] = 0.0f;
            blk_2[7] = 0.0f;
            blk_2[8] = 0.0f;
            blk_2[9] = 0.0f;
            blk_2[10] = 0.0f;
            blk_2[11] = 0.0f;
            blk_2[12] = 0.0f;
            blk_2[13] = 0.0f;
            blk_2[14] = 0.0f;
            blk_2[15] = 0.0f;
            blk_2[16] = 0.0f;
            blk_2[17] = 0.0f;
            blk_2[18] = 0.0f;
            blk_2[19] = 0.0f;
            blk_2[20] = 0.0f;
            blk_2[21] = 0.0f;
            blk_2[22] = 0.0f;
            blk_2[23] = 0.0f;
            blk_2[24] = 0.0f;
            blk_2[25] = 0.0f;
            blk_2[26] = 0.0f;
            blk_2[27] = 0.0f;
            blk_2[28] = 0.0f;
            blk_2[29] = 0.0f;
            blk_2[30] = 0.0f;
            blk_2[31] = 0.0f;
            tmem_st_x32_f32(tmem_sb + row_off + 64, blk_2);
            float blk_3[32];
            blk_3[0] = 0.0f;
            blk_3[1] = 0.0f;
            blk_3[2] = 0.0f;
            blk_3[3] = 0.0f;
            blk_3[4] = 0.0f;
            blk_3[5] = 0.0f;
            blk_3[6] = 0.0f;
            blk_3[7] = 0.0f;
            blk_3[8] = 0.0f;
            blk_3[9] = 0.0f;
            blk_3[10] = 0.0f;
            blk_3[11] = 0.0f;
            blk_3[12] = 0.0f;
            blk_3[13] = 0.0f;
            blk_3[14] = 0.0f;
            blk_3[15] = 0.0f;
            blk_3[16] = 0.0f;
            blk_3[17] = 0.0f;
            blk_3[18] = 0.0f;
            blk_3[19] = 0.0f;
            blk_3[20] = 0.0f;
            blk_3[21] = 0.0f;
            blk_3[22] = 0.0f;
            blk_3[23] = 0.0f;
            blk_3[24] = 0.0f;
            blk_3[25] = 0.0f;
            blk_3[26] = 0.0f;
            blk_3[27] = 0.0f;
            blk_3[28] = 0.0f;
            blk_3[29] = 0.0f;
            blk_3[30] = 0.0f;
            blk_3[31] = 0.0f;
            tmem_st_x32_f32(tmem_sb + row_off + 96, blk_3);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int up[32];
            unsigned int gs_c = 0;
            unsigned int _phase_u_full_0 = 0;
            if (nc > 0) {
                mbarrier_wait(u_full_addr, _phase_u_full_0);
                _phase_u_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float uv[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(uv[0]), "=f"(uv[1]), "=f"(uv[2]), "=f"(uv[3]), "=f"(uv[4]), "=f"(uv[5]), "=f"(uv[6]), "=f"(uv[7]), "=f"(uv[8]), "=f"(uv[9]), "=f"(uv[10]), "=f"(uv[11]), "=f"(uv[12]), "=f"(uv[13]), "=f"(uv[14]), "=f"(uv[15]), "=f"(uv[16]), "=f"(uv[17]), "=f"(uv[18]), "=f"(uv[19]), "=f"(uv[20]), "=f"(uv[21]), "=f"(uv[22]), "=f"(uv[23]), "=f"(uv[24]), "=f"(uv[25]), "=f"(uv[26]), "=f"(uv[27]), "=f"(uv[28]), "=f"(uv[29]), "=f"(uv[30]), "=f"(uv[31])
                    : "r"(tmem_wb + row_off));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(uv[32]), "=f"(uv[33]), "=f"(uv[34]), "=f"(uv[35]), "=f"(uv[36]), "=f"(uv[37]), "=f"(uv[38]), "=f"(uv[39]), "=f"(uv[40]), "=f"(uv[41]), "=f"(uv[42]), "=f"(uv[43]), "=f"(uv[44]), "=f"(uv[45]), "=f"(uv[46]), "=f"(uv[47]), "=f"(uv[48]), "=f"(uv[49]), "=f"(uv[50]), "=f"(uv[51]), "=f"(uv[52]), "=f"(uv[53]), "=f"(uv[54]), "=f"(uv[55]), "=f"(uv[56]), "=f"(uv[57]), "=f"(uv[58]), "=f"(uv[59]), "=f"(uv[60]), "=f"(uv[61]), "=f"(uv[62]), "=f"(uv[63])
                    : "r"(tmem_wb + row_off + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uv[_lp*2 + 0], uv[_lp*2+1 + 0]));
                    up[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(u_taken_addr);
            }
            unsigned int _phase_gates_ready = 0;
            unsigned int _phase_ws_full_b_0 = 0;
            unsigned int _phase_s_full_b_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                int n = chunk_len[c_begin + c];
                mbarrier_wait(gates_ready_addr + (gs_c) * 8, _phase_gates_ready);
                int g_off = c % 4 * 256;
                float e2gl = s_gt[g_off + 192];
                float sv[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                    : "r"(tmem_sb + row_off + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                    hp[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hb + row_off + 0), "r"(*reinterpret_cast<const uint32_t*>(&hp[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp[15])));
                const float2 _scale2_0 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv)[_ls], _scale2_0);
                tmem_st_x32_f32(tmem_sb + row_off + 0, sv);
                float sv_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_0[0]), "=f"(sv_0[1]), "=f"(sv_0[2]), "=f"(sv_0[3]), "=f"(sv_0[4]), "=f"(sv_0[5]), "=f"(sv_0[6]), "=f"(sv_0[7]), "=f"(sv_0[8]), "=f"(sv_0[9]), "=f"(sv_0[10]), "=f"(sv_0[11]), "=f"(sv_0[12]), "=f"(sv_0[13]), "=f"(sv_0[14]), "=f"(sv_0[15]), "=f"(sv_0[16]), "=f"(sv_0[17]), "=f"(sv_0[18]), "=f"(sv_0[19]), "=f"(sv_0[20]), "=f"(sv_0[21]), "=f"(sv_0[22]), "=f"(sv_0[23]), "=f"(sv_0[24]), "=f"(sv_0[25]), "=f"(sv_0[26]), "=f"(sv_0[27]), "=f"(sv_0[28]), "=f"(sv_0[29]), "=f"(sv_0[30]), "=f"(sv_0[31])
                    : "r"(tmem_sb + row_off + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_0[_lp*2 + 0], sv_0[_lp*2+1 + 0]));
                    hp_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hb + row_off + 16), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[15])));
                const float2 _scale2_1 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_0)[_ls], _scale2_1);
                tmem_st_x32_f32(tmem_sb + row_off + 32, sv_0);
                float sv_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31])
                    : "r"(tmem_sb + row_off + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_1[_lp*2 + 0], sv_1[_lp*2+1 + 0]));
                    hp_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hb + row_off + 32), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[15])));
                const float2 _scale2_2 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_1)[_ls], _scale2_2);
                tmem_st_x32_f32(tmem_sb + row_off + 64, sv_1);
                float sv_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_2[0]), "=f"(sv_2[1]), "=f"(sv_2[2]), "=f"(sv_2[3]), "=f"(sv_2[4]), "=f"(sv_2[5]), "=f"(sv_2[6]), "=f"(sv_2[7]), "=f"(sv_2[8]), "=f"(sv_2[9]), "=f"(sv_2[10]), "=f"(sv_2[11]), "=f"(sv_2[12]), "=f"(sv_2[13]), "=f"(sv_2[14]), "=f"(sv_2[15]), "=f"(sv_2[16]), "=f"(sv_2[17]), "=f"(sv_2[18]), "=f"(sv_2[19]), "=f"(sv_2[20]), "=f"(sv_2[21]), "=f"(sv_2[22]), "=f"(sv_2[23]), "=f"(sv_2[24]), "=f"(sv_2[25]), "=f"(sv_2[26]), "=f"(sv_2[27]), "=f"(sv_2[28]), "=f"(sv_2[29]), "=f"(sv_2[30]), "=f"(sv_2[31])
                    : "r"(tmem_sb + row_off + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp_2[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_2[_lp*2 + 0], sv_2[_lp*2+1 + 0]));
                    hp_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hb + row_off + 48), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[15])));
                const float2 _scale2_3 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_2)[_ls], _scale2_3);
                tmem_st_x32_f32(tmem_sb + row_off + 96, sv_2);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_b_addr);
                mbarrier_wait(ws_full_b_addr, _phase_ws_full_b_0);
                _phase_ws_full_b_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float vv[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(vv[0]), "=f"(vv[1]), "=f"(vv[2]), "=f"(vv[3]), "=f"(vv[4]), "=f"(vv[5]), "=f"(vv[6]), "=f"(vv[7]), "=f"(vv[8]), "=f"(vv[9]), "=f"(vv[10]), "=f"(vv[11]), "=f"(vv[12]), "=f"(vv[13]), "=f"(vv[14]), "=f"(vv[15]), "=f"(vv[16]), "=f"(vv[17]), "=f"(vv[18]), "=f"(vv[19]), "=f"(vv[20]), "=f"(vv[21]), "=f"(vv[22]), "=f"(vv[23]), "=f"(vv[24]), "=f"(vv[25]), "=f"(vv[26]), "=f"(vv[27]), "=f"(vv[28]), "=f"(vv[29]), "=f"(vv[30]), "=f"(vv[31])
                    : "r"(tmem_wb + row_off + 0));
                float dks[32];
                dks[0] = s_gt[g_off];
                dks[1] = s_gt[g_off + 1];
                dks[2] = s_gt[g_off + 2];
                dks[3] = s_gt[g_off + 3];
                dks[4] = s_gt[g_off + 4];
                dks[5] = s_gt[g_off + 5];
                dks[6] = s_gt[g_off + 6];
                dks[7] = s_gt[g_off + 7];
                dks[8] = s_gt[g_off + 8];
                dks[9] = s_gt[g_off + 9];
                dks[10] = s_gt[g_off + 10];
                dks[11] = s_gt[g_off + 11];
                dks[12] = s_gt[g_off + 12];
                dks[13] = s_gt[g_off + 13];
                dks[14] = s_gt[g_off + 14];
                dks[15] = s_gt[g_off + 15];
                dks[16] = s_gt[g_off + 16];
                dks[17] = s_gt[g_off + 17];
                dks[18] = s_gt[g_off + 18];
                dks[19] = s_gt[g_off + 19];
                dks[20] = s_gt[g_off + 20];
                dks[21] = s_gt[g_off + 21];
                dks[22] = s_gt[g_off + 22];
                dks[23] = s_gt[g_off + 23];
                dks[24] = s_gt[g_off + 24];
                dks[25] = s_gt[g_off + 25];
                dks[26] = s_gt[g_off + 26];
                dks[27] = s_gt[g_off + 27];
                dks[28] = s_gt[g_off + 28];
                dks[29] = s_gt[g_off + 29];
                dks[30] = s_gt[g_off + 30];
                dks[31] = s_gt[g_off + 31];
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                vv[0] = __uint_as_float(up[0] << 16) - vv[0];
                vv[1] = __uint_as_float(up[0] & 4294901760u) - vv[1];
                vv[2] = __uint_as_float(up[1] << 16) - vv[2];
                vv[3] = __uint_as_float(up[1] & 4294901760u) - vv[3];
                vv[4] = __uint_as_float(up[2] << 16) - vv[4];
                vv[5] = __uint_as_float(up[2] & 4294901760u) - vv[5];
                vv[6] = __uint_as_float(up[3] << 16) - vv[6];
                vv[7] = __uint_as_float(up[3] & 4294901760u) - vv[7];
                vv[8] = __uint_as_float(up[4] << 16) - vv[8];
                vv[9] = __uint_as_float(up[4] & 4294901760u) - vv[9];
                vv[10] = __uint_as_float(up[5] << 16) - vv[10];
                vv[11] = __uint_as_float(up[5] & 4294901760u) - vv[11];
                vv[12] = __uint_as_float(up[6] << 16) - vv[12];
                vv[13] = __uint_as_float(up[6] & 4294901760u) - vv[13];
                vv[14] = __uint_as_float(up[7] << 16) - vv[14];
                vv[15] = __uint_as_float(up[7] & 4294901760u) - vv[15];
                vv[16] = __uint_as_float(up[8] << 16) - vv[16];
                vv[17] = __uint_as_float(up[8] & 4294901760u) - vv[17];
                vv[18] = __uint_as_float(up[9] << 16) - vv[18];
                vv[19] = __uint_as_float(up[9] & 4294901760u) - vv[19];
                vv[20] = __uint_as_float(up[10] << 16) - vv[20];
                vv[21] = __uint_as_float(up[10] & 4294901760u) - vv[21];
                vv[22] = __uint_as_float(up[11] << 16) - vv[22];
                vv[23] = __uint_as_float(up[11] & 4294901760u) - vv[23];
                vv[24] = __uint_as_float(up[12] << 16) - vv[24];
                vv[25] = __uint_as_float(up[12] & 4294901760u) - vv[25];
                vv[26] = __uint_as_float(up[13] << 16) - vv[26];
                vv[27] = __uint_as_float(up[13] & 4294901760u) - vv[27];
                vv[28] = __uint_as_float(up[14] << 16) - vv[28];
                vv[29] = __uint_as_float(up[14] & 4294901760u) - vv[29];
                vv[30] = __uint_as_float(up[15] << 16) - vv[30];
                vv[31] = __uint_as_float(up[15] & 4294901760u) - vv[31];
                if (n < 64) {
                    vv[0] = ((0 < n) ? vv[0] : 0.0f);
                    vv[1] = ((1 < n) ? vv[1] : 0.0f);
                    vv[2] = ((2 < n) ? vv[2] : 0.0f);
                    vv[3] = ((3 < n) ? vv[3] : 0.0f);
                    vv[4] = ((4 < n) ? vv[4] : 0.0f);
                    vv[5] = ((5 < n) ? vv[5] : 0.0f);
                    vv[6] = ((6 < n) ? vv[6] : 0.0f);
                    vv[7] = ((7 < n) ? vv[7] : 0.0f);
                    vv[8] = ((8 < n) ? vv[8] : 0.0f);
                    vv[9] = ((9 < n) ? vv[9] : 0.0f);
                    vv[10] = ((10 < n) ? vv[10] : 0.0f);
                    vv[11] = ((11 < n) ? vv[11] : 0.0f);
                    vv[12] = ((12 < n) ? vv[12] : 0.0f);
                    vv[13] = ((13 < n) ? vv[13] : 0.0f);
                    vv[14] = ((14 < n) ? vv[14] : 0.0f);
                    vv[15] = ((15 < n) ? vv[15] : 0.0f);
                    vv[16] = ((16 < n) ? vv[16] : 0.0f);
                    vv[17] = ((17 < n) ? vv[17] : 0.0f);
                    vv[18] = ((18 < n) ? vv[18] : 0.0f);
                    vv[19] = ((19 < n) ? vv[19] : 0.0f);
                    vv[20] = ((20 < n) ? vv[20] : 0.0f);
                    vv[21] = ((21 < n) ? vv[21] : 0.0f);
                    vv[22] = ((22 < n) ? vv[22] : 0.0f);
                    vv[23] = ((23 < n) ? vv[23] : 0.0f);
                    vv[24] = ((24 < n) ? vv[24] : 0.0f);
                    vv[25] = ((25 < n) ? vv[25] : 0.0f);
                    vv[26] = ((26 < n) ? vv[26] : 0.0f);
                    vv[27] = ((27 < n) ? vv[27] : 0.0f);
                    vv[28] = ((28 < n) ? vv[28] : 0.0f);
                    vv[29] = ((29 < n) ? vv[29] : 0.0f);
                    vv[30] = ((30 < n) ? vv[30] : 0.0f);
                    vv[31] = ((31 < n) ? vv[31] : 0.0f);
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv)[_ls], reinterpret_cast<const float2*>(dks)[_ls]);
                unsigned int vp[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv[_lp*2 + 0], vv[_lp*2+1 + 0]));
                    vp[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_vb + row_off + 0), "r"(*reinterpret_cast<const uint32_t*>(&vp[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp[7])), "r"(*reinterpret_cast<const uint32_t*>(&vp[8])), "r"(*reinterpret_cast<const uint32_t*>(&vp[9])), "r"(*reinterpret_cast<const uint32_t*>(&vp[10])), "r"(*reinterpret_cast<const uint32_t*>(&vp[11])), "r"(*reinterpret_cast<const uint32_t*>(&vp[12])), "r"(*reinterpret_cast<const uint32_t*>(&vp[13])), "r"(*reinterpret_cast<const uint32_t*>(&vp[14])), "r"(*reinterpret_cast<const uint32_t*>(&vp[15])));
                float vv_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(vv_0[0]), "=f"(vv_0[1]), "=f"(vv_0[2]), "=f"(vv_0[3]), "=f"(vv_0[4]), "=f"(vv_0[5]), "=f"(vv_0[6]), "=f"(vv_0[7]), "=f"(vv_0[8]), "=f"(vv_0[9]), "=f"(vv_0[10]), "=f"(vv_0[11]), "=f"(vv_0[12]), "=f"(vv_0[13]), "=f"(vv_0[14]), "=f"(vv_0[15]), "=f"(vv_0[16]), "=f"(vv_0[17]), "=f"(vv_0[18]), "=f"(vv_0[19]), "=f"(vv_0[20]), "=f"(vv_0[21]), "=f"(vv_0[22]), "=f"(vv_0[23]), "=f"(vv_0[24]), "=f"(vv_0[25]), "=f"(vv_0[26]), "=f"(vv_0[27]), "=f"(vv_0[28]), "=f"(vv_0[29]), "=f"(vv_0[30]), "=f"(vv_0[31])
                    : "r"(tmem_wb + row_off + 32));
                float dks_0[32];
                dks_0[0] = s_gt[g_off + 32];
                dks_0[1] = s_gt[g_off + 32 + 1];
                dks_0[2] = s_gt[g_off + 32 + 2];
                dks_0[3] = s_gt[g_off + 32 + 3];
                dks_0[4] = s_gt[g_off + 32 + 4];
                dks_0[5] = s_gt[g_off + 32 + 5];
                dks_0[6] = s_gt[g_off + 32 + 6];
                dks_0[7] = s_gt[g_off + 32 + 7];
                dks_0[8] = s_gt[g_off + 32 + 8];
                dks_0[9] = s_gt[g_off + 32 + 9];
                dks_0[10] = s_gt[g_off + 32 + 10];
                dks_0[11] = s_gt[g_off + 32 + 11];
                dks_0[12] = s_gt[g_off + 32 + 12];
                dks_0[13] = s_gt[g_off + 32 + 13];
                dks_0[14] = s_gt[g_off + 32 + 14];
                dks_0[15] = s_gt[g_off + 32 + 15];
                dks_0[16] = s_gt[g_off + 32 + 16];
                dks_0[17] = s_gt[g_off + 32 + 17];
                dks_0[18] = s_gt[g_off + 32 + 18];
                dks_0[19] = s_gt[g_off + 32 + 19];
                dks_0[20] = s_gt[g_off + 32 + 20];
                dks_0[21] = s_gt[g_off + 32 + 21];
                dks_0[22] = s_gt[g_off + 32 + 22];
                dks_0[23] = s_gt[g_off + 32 + 23];
                dks_0[24] = s_gt[g_off + 32 + 24];
                dks_0[25] = s_gt[g_off + 32 + 25];
                dks_0[26] = s_gt[g_off + 32 + 26];
                dks_0[27] = s_gt[g_off + 32 + 27];
                dks_0[28] = s_gt[g_off + 32 + 28];
                dks_0[29] = s_gt[g_off + 32 + 29];
                dks_0[30] = s_gt[g_off + 32 + 30];
                dks_0[31] = s_gt[g_off + 32 + 31];
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                vv_0[0] = __uint_as_float(up[16] << 16) - vv_0[0];
                vv_0[1] = __uint_as_float(up[16] & 4294901760u) - vv_0[1];
                vv_0[2] = __uint_as_float(up[17] << 16) - vv_0[2];
                vv_0[3] = __uint_as_float(up[17] & 4294901760u) - vv_0[3];
                vv_0[4] = __uint_as_float(up[18] << 16) - vv_0[4];
                vv_0[5] = __uint_as_float(up[18] & 4294901760u) - vv_0[5];
                vv_0[6] = __uint_as_float(up[19] << 16) - vv_0[6];
                vv_0[7] = __uint_as_float(up[19] & 4294901760u) - vv_0[7];
                vv_0[8] = __uint_as_float(up[20] << 16) - vv_0[8];
                vv_0[9] = __uint_as_float(up[20] & 4294901760u) - vv_0[9];
                vv_0[10] = __uint_as_float(up[21] << 16) - vv_0[10];
                vv_0[11] = __uint_as_float(up[21] & 4294901760u) - vv_0[11];
                vv_0[12] = __uint_as_float(up[22] << 16) - vv_0[12];
                vv_0[13] = __uint_as_float(up[22] & 4294901760u) - vv_0[13];
                vv_0[14] = __uint_as_float(up[23] << 16) - vv_0[14];
                vv_0[15] = __uint_as_float(up[23] & 4294901760u) - vv_0[15];
                vv_0[16] = __uint_as_float(up[24] << 16) - vv_0[16];
                vv_0[17] = __uint_as_float(up[24] & 4294901760u) - vv_0[17];
                vv_0[18] = __uint_as_float(up[25] << 16) - vv_0[18];
                vv_0[19] = __uint_as_float(up[25] & 4294901760u) - vv_0[19];
                vv_0[20] = __uint_as_float(up[26] << 16) - vv_0[20];
                vv_0[21] = __uint_as_float(up[26] & 4294901760u) - vv_0[21];
                vv_0[22] = __uint_as_float(up[27] << 16) - vv_0[22];
                vv_0[23] = __uint_as_float(up[27] & 4294901760u) - vv_0[23];
                vv_0[24] = __uint_as_float(up[28] << 16) - vv_0[24];
                vv_0[25] = __uint_as_float(up[28] & 4294901760u) - vv_0[25];
                vv_0[26] = __uint_as_float(up[29] << 16) - vv_0[26];
                vv_0[27] = __uint_as_float(up[29] & 4294901760u) - vv_0[27];
                vv_0[28] = __uint_as_float(up[30] << 16) - vv_0[28];
                vv_0[29] = __uint_as_float(up[30] & 4294901760u) - vv_0[29];
                vv_0[30] = __uint_as_float(up[31] << 16) - vv_0[30];
                vv_0[31] = __uint_as_float(up[31] & 4294901760u) - vv_0[31];
                if (n < 64) {
                    vv_0[0] = ((32 < n) ? vv_0[0] : 0.0f);
                    vv_0[1] = ((33 < n) ? vv_0[1] : 0.0f);
                    vv_0[2] = ((34 < n) ? vv_0[2] : 0.0f);
                    vv_0[3] = ((35 < n) ? vv_0[3] : 0.0f);
                    vv_0[4] = ((36 < n) ? vv_0[4] : 0.0f);
                    vv_0[5] = ((37 < n) ? vv_0[5] : 0.0f);
                    vv_0[6] = ((38 < n) ? vv_0[6] : 0.0f);
                    vv_0[7] = ((39 < n) ? vv_0[7] : 0.0f);
                    vv_0[8] = ((40 < n) ? vv_0[8] : 0.0f);
                    vv_0[9] = ((41 < n) ? vv_0[9] : 0.0f);
                    vv_0[10] = ((42 < n) ? vv_0[10] : 0.0f);
                    vv_0[11] = ((43 < n) ? vv_0[11] : 0.0f);
                    vv_0[12] = ((44 < n) ? vv_0[12] : 0.0f);
                    vv_0[13] = ((45 < n) ? vv_0[13] : 0.0f);
                    vv_0[14] = ((46 < n) ? vv_0[14] : 0.0f);
                    vv_0[15] = ((47 < n) ? vv_0[15] : 0.0f);
                    vv_0[16] = ((48 < n) ? vv_0[16] : 0.0f);
                    vv_0[17] = ((49 < n) ? vv_0[17] : 0.0f);
                    vv_0[18] = ((50 < n) ? vv_0[18] : 0.0f);
                    vv_0[19] = ((51 < n) ? vv_0[19] : 0.0f);
                    vv_0[20] = ((52 < n) ? vv_0[20] : 0.0f);
                    vv_0[21] = ((53 < n) ? vv_0[21] : 0.0f);
                    vv_0[22] = ((54 < n) ? vv_0[22] : 0.0f);
                    vv_0[23] = ((55 < n) ? vv_0[23] : 0.0f);
                    vv_0[24] = ((56 < n) ? vv_0[24] : 0.0f);
                    vv_0[25] = ((57 < n) ? vv_0[25] : 0.0f);
                    vv_0[26] = ((58 < n) ? vv_0[26] : 0.0f);
                    vv_0[27] = ((59 < n) ? vv_0[27] : 0.0f);
                    vv_0[28] = ((60 < n) ? vv_0[28] : 0.0f);
                    vv_0[29] = ((61 < n) ? vv_0[29] : 0.0f);
                    vv_0[30] = ((62 < n) ? vv_0[30] : 0.0f);
                    vv_0[31] = ((63 < n) ? vv_0[31] : 0.0f);
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv_0)[_ls], reinterpret_cast<const float2*>(dks_0)[_ls]);
                unsigned int vp_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_0[_lp*2 + 0], vv_0[_lp*2+1 + 0]));
                    vp_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_vb + row_off + 16), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[15])));
                mbarrier_arrive(gates_free_addr + (gs_c) * 8);
                gs_c += 1;
                if (gs_c == 4) { gs_c = 0; _phase_gates_ready ^= 1; }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(vp_ready_b_addr);
                if (c + 1 < nc) {
                    mbarrier_wait(u_full_addr, _phase_u_full_0);
                    _phase_u_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float uv_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(uv_1[0]), "=f"(uv_1[1]), "=f"(uv_1[2]), "=f"(uv_1[3]), "=f"(uv_1[4]), "=f"(uv_1[5]), "=f"(uv_1[6]), "=f"(uv_1[7]), "=f"(uv_1[8]), "=f"(uv_1[9]), "=f"(uv_1[10]), "=f"(uv_1[11]), "=f"(uv_1[12]), "=f"(uv_1[13]), "=f"(uv_1[14]), "=f"(uv_1[15]), "=f"(uv_1[16]), "=f"(uv_1[17]), "=f"(uv_1[18]), "=f"(uv_1[19]), "=f"(uv_1[20]), "=f"(uv_1[21]), "=f"(uv_1[22]), "=f"(uv_1[23]), "=f"(uv_1[24]), "=f"(uv_1[25]), "=f"(uv_1[26]), "=f"(uv_1[27]), "=f"(uv_1[28]), "=f"(uv_1[29]), "=f"(uv_1[30]), "=f"(uv_1[31])
                        : "r"(tmem_wb + row_off));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(uv_1[32]), "=f"(uv_1[33]), "=f"(uv_1[34]), "=f"(uv_1[35]), "=f"(uv_1[36]), "=f"(uv_1[37]), "=f"(uv_1[38]), "=f"(uv_1[39]), "=f"(uv_1[40]), "=f"(uv_1[41]), "=f"(uv_1[42]), "=f"(uv_1[43]), "=f"(uv_1[44]), "=f"(uv_1[45]), "=f"(uv_1[46]), "=f"(uv_1[47]), "=f"(uv_1[48]), "=f"(uv_1[49]), "=f"(uv_1[50]), "=f"(uv_1[51]), "=f"(uv_1[52]), "=f"(uv_1[53]), "=f"(uv_1[54]), "=f"(uv_1[55]), "=f"(uv_1[56]), "=f"(uv_1[57]), "=f"(uv_1[58]), "=f"(uv_1[59]), "=f"(uv_1[60]), "=f"(uv_1[61]), "=f"(uv_1[62]), "=f"(uv_1[63])
                        : "r"(tmem_wb + row_off + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uv_1[_lp*2 + 0], uv_1[_lp*2+1 + 0]));
                        up[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(u_taken_addr);
                }
                mbarrier_wait(s_full_b_addr, _phase_s_full_b_0);
                _phase_s_full_b_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            long long b_base = ((long long)seg * (long long)num_v_heads + (long long)hv) * (long long)b_head + (long long)vrow;
            float blk_4[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_4[0]), "=f"(blk_4[1]), "=f"(blk_4[2]), "=f"(blk_4[3]), "=f"(blk_4[4]), "=f"(blk_4[5]), "=f"(blk_4[6]), "=f"(blk_4[7]), "=f"(blk_4[8]), "=f"(blk_4[9]), "=f"(blk_4[10]), "=f"(blk_4[11]), "=f"(blk_4[12]), "=f"(blk_4[13]), "=f"(blk_4[14]), "=f"(blk_4[15]), "=f"(blk_4[16]), "=f"(blk_4[17]), "=f"(blk_4[18]), "=f"(blk_4[19]), "=f"(blk_4[20]), "=f"(blk_4[21]), "=f"(blk_4[22]), "=f"(blk_4[23]), "=f"(blk_4[24]), "=f"(blk_4[25]), "=f"(blk_4[26]), "=f"(blk_4[27]), "=f"(blk_4[28]), "=f"(blk_4[29]), "=f"(blk_4[30]), "=f"(blk_4[31])
                : "r"(tmem_sb + row_off + 0));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            b_out[b_base + (long long)b_row * 0ll] = blk_4[0];
            b_out[b_base + (long long)b_row * 1ll] = blk_4[1];
            b_out[b_base + (long long)b_row * 2ll] = blk_4[2];
            b_out[b_base + (long long)b_row * 3ll] = blk_4[3];
            b_out[b_base + (long long)b_row * 4ll] = blk_4[4];
            b_out[b_base + (long long)b_row * 5ll] = blk_4[5];
            b_out[b_base + (long long)b_row * 6ll] = blk_4[6];
            b_out[b_base + (long long)b_row * 7ll] = blk_4[7];
            b_out[b_base + (long long)b_row * 8ll] = blk_4[8];
            b_out[b_base + (long long)b_row * 9ll] = blk_4[9];
            b_out[b_base + (long long)b_row * 10ll] = blk_4[10];
            b_out[b_base + (long long)b_row * 11ll] = blk_4[11];
            b_out[b_base + (long long)b_row * 12ll] = blk_4[12];
            b_out[b_base + (long long)b_row * 13ll] = blk_4[13];
            b_out[b_base + (long long)b_row * 14ll] = blk_4[14];
            b_out[b_base + (long long)b_row * 15ll] = blk_4[15];
            b_out[b_base + (long long)b_row * 16ll] = blk_4[16];
            b_out[b_base + (long long)b_row * 17ll] = blk_4[17];
            b_out[b_base + (long long)b_row * 18ll] = blk_4[18];
            b_out[b_base + (long long)b_row * 19ll] = blk_4[19];
            b_out[b_base + (long long)b_row * 20ll] = blk_4[20];
            b_out[b_base + (long long)b_row * 21ll] = blk_4[21];
            b_out[b_base + (long long)b_row * 22ll] = blk_4[22];
            b_out[b_base + (long long)b_row * 23ll] = blk_4[23];
            b_out[b_base + (long long)b_row * 24ll] = blk_4[24];
            b_out[b_base + (long long)b_row * 25ll] = blk_4[25];
            b_out[b_base + (long long)b_row * 26ll] = blk_4[26];
            b_out[b_base + (long long)b_row * 27ll] = blk_4[27];
            b_out[b_base + (long long)b_row * 28ll] = blk_4[28];
            b_out[b_base + (long long)b_row * 29ll] = blk_4[29];
            b_out[b_base + (long long)b_row * 30ll] = blk_4[30];
            b_out[b_base + (long long)b_row * 31ll] = blk_4[31];
            float blk_5[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_5[0]), "=f"(blk_5[1]), "=f"(blk_5[2]), "=f"(blk_5[3]), "=f"(blk_5[4]), "=f"(blk_5[5]), "=f"(blk_5[6]), "=f"(blk_5[7]), "=f"(blk_5[8]), "=f"(blk_5[9]), "=f"(blk_5[10]), "=f"(blk_5[11]), "=f"(blk_5[12]), "=f"(blk_5[13]), "=f"(blk_5[14]), "=f"(blk_5[15]), "=f"(blk_5[16]), "=f"(blk_5[17]), "=f"(blk_5[18]), "=f"(blk_5[19]), "=f"(blk_5[20]), "=f"(blk_5[21]), "=f"(blk_5[22]), "=f"(blk_5[23]), "=f"(blk_5[24]), "=f"(blk_5[25]), "=f"(blk_5[26]), "=f"(blk_5[27]), "=f"(blk_5[28]), "=f"(blk_5[29]), "=f"(blk_5[30]), "=f"(blk_5[31])
                : "r"(tmem_sb + row_off + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            b_out[b_base + (long long)b_row * 32ll] = blk_5[0];
            b_out[b_base + (long long)b_row * 33ll] = blk_5[1];
            b_out[b_base + (long long)b_row * 34ll] = blk_5[2];
            b_out[b_base + (long long)b_row * 35ll] = blk_5[3];
            b_out[b_base + (long long)b_row * 36ll] = blk_5[4];
            b_out[b_base + (long long)b_row * 37ll] = blk_5[5];
            b_out[b_base + (long long)b_row * 38ll] = blk_5[6];
            b_out[b_base + (long long)b_row * 39ll] = blk_5[7];
            b_out[b_base + (long long)b_row * 40ll] = blk_5[8];
            b_out[b_base + (long long)b_row * 41ll] = blk_5[9];
            b_out[b_base + (long long)b_row * 42ll] = blk_5[10];
            b_out[b_base + (long long)b_row * 43ll] = blk_5[11];
            b_out[b_base + (long long)b_row * 44ll] = blk_5[12];
            b_out[b_base + (long long)b_row * 45ll] = blk_5[13];
            b_out[b_base + (long long)b_row * 46ll] = blk_5[14];
            b_out[b_base + (long long)b_row * 47ll] = blk_5[15];
            b_out[b_base + (long long)b_row * 48ll] = blk_5[16];
            b_out[b_base + (long long)b_row * 49ll] = blk_5[17];
            b_out[b_base + (long long)b_row * 50ll] = blk_5[18];
            b_out[b_base + (long long)b_row * 51ll] = blk_5[19];
            b_out[b_base + (long long)b_row * 52ll] = blk_5[20];
            b_out[b_base + (long long)b_row * 53ll] = blk_5[21];
            b_out[b_base + (long long)b_row * 54ll] = blk_5[22];
            b_out[b_base + (long long)b_row * 55ll] = blk_5[23];
            b_out[b_base + (long long)b_row * 56ll] = blk_5[24];
            b_out[b_base + (long long)b_row * 57ll] = blk_5[25];
            b_out[b_base + (long long)b_row * 58ll] = blk_5[26];
            b_out[b_base + (long long)b_row * 59ll] = blk_5[27];
            b_out[b_base + (long long)b_row * 60ll] = blk_5[28];
            b_out[b_base + (long long)b_row * 61ll] = blk_5[29];
            b_out[b_base + (long long)b_row * 62ll] = blk_5[30];
            b_out[b_base + (long long)b_row * 63ll] = blk_5[31];
            float blk_6[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_6[0]), "=f"(blk_6[1]), "=f"(blk_6[2]), "=f"(blk_6[3]), "=f"(blk_6[4]), "=f"(blk_6[5]), "=f"(blk_6[6]), "=f"(blk_6[7]), "=f"(blk_6[8]), "=f"(blk_6[9]), "=f"(blk_6[10]), "=f"(blk_6[11]), "=f"(blk_6[12]), "=f"(blk_6[13]), "=f"(blk_6[14]), "=f"(blk_6[15]), "=f"(blk_6[16]), "=f"(blk_6[17]), "=f"(blk_6[18]), "=f"(blk_6[19]), "=f"(blk_6[20]), "=f"(blk_6[21]), "=f"(blk_6[22]), "=f"(blk_6[23]), "=f"(blk_6[24]), "=f"(blk_6[25]), "=f"(blk_6[26]), "=f"(blk_6[27]), "=f"(blk_6[28]), "=f"(blk_6[29]), "=f"(blk_6[30]), "=f"(blk_6[31])
                : "r"(tmem_sb + row_off + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            b_out[b_base + (long long)b_row * 64ll] = blk_6[0];
            b_out[b_base + (long long)b_row * 65ll] = blk_6[1];
            b_out[b_base + (long long)b_row * 66ll] = blk_6[2];
            b_out[b_base + (long long)b_row * 67ll] = blk_6[3];
            b_out[b_base + (long long)b_row * 68ll] = blk_6[4];
            b_out[b_base + (long long)b_row * 69ll] = blk_6[5];
            b_out[b_base + (long long)b_row * 70ll] = blk_6[6];
            b_out[b_base + (long long)b_row * 71ll] = blk_6[7];
            b_out[b_base + (long long)b_row * 72ll] = blk_6[8];
            b_out[b_base + (long long)b_row * 73ll] = blk_6[9];
            b_out[b_base + (long long)b_row * 74ll] = blk_6[10];
            b_out[b_base + (long long)b_row * 75ll] = blk_6[11];
            b_out[b_base + (long long)b_row * 76ll] = blk_6[12];
            b_out[b_base + (long long)b_row * 77ll] = blk_6[13];
            b_out[b_base + (long long)b_row * 78ll] = blk_6[14];
            b_out[b_base + (long long)b_row * 79ll] = blk_6[15];
            b_out[b_base + (long long)b_row * 80ll] = blk_6[16];
            b_out[b_base + (long long)b_row * 81ll] = blk_6[17];
            b_out[b_base + (long long)b_row * 82ll] = blk_6[18];
            b_out[b_base + (long long)b_row * 83ll] = blk_6[19];
            b_out[b_base + (long long)b_row * 84ll] = blk_6[20];
            b_out[b_base + (long long)b_row * 85ll] = blk_6[21];
            b_out[b_base + (long long)b_row * 86ll] = blk_6[22];
            b_out[b_base + (long long)b_row * 87ll] = blk_6[23];
            b_out[b_base + (long long)b_row * 88ll] = blk_6[24];
            b_out[b_base + (long long)b_row * 89ll] = blk_6[25];
            b_out[b_base + (long long)b_row * 90ll] = blk_6[26];
            b_out[b_base + (long long)b_row * 91ll] = blk_6[27];
            b_out[b_base + (long long)b_row * 92ll] = blk_6[28];
            b_out[b_base + (long long)b_row * 93ll] = blk_6[29];
            b_out[b_base + (long long)b_row * 94ll] = blk_6[30];
            b_out[b_base + (long long)b_row * 95ll] = blk_6[31];
            float blk_7[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_7[0]), "=f"(blk_7[1]), "=f"(blk_7[2]), "=f"(blk_7[3]), "=f"(blk_7[4]), "=f"(blk_7[5]), "=f"(blk_7[6]), "=f"(blk_7[7]), "=f"(blk_7[8]), "=f"(blk_7[9]), "=f"(blk_7[10]), "=f"(blk_7[11]), "=f"(blk_7[12]), "=f"(blk_7[13]), "=f"(blk_7[14]), "=f"(blk_7[15]), "=f"(blk_7[16]), "=f"(blk_7[17]), "=f"(blk_7[18]), "=f"(blk_7[19]), "=f"(blk_7[20]), "=f"(blk_7[21]), "=f"(blk_7[22]), "=f"(blk_7[23]), "=f"(blk_7[24]), "=f"(blk_7[25]), "=f"(blk_7[26]), "=f"(blk_7[27]), "=f"(blk_7[28]), "=f"(blk_7[29]), "=f"(blk_7[30]), "=f"(blk_7[31])
                : "r"(tmem_sb + row_off + 96));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            b_out[b_base + (long long)b_row * 96ll] = blk_7[0];
            b_out[b_base + (long long)b_row * 97ll] = blk_7[1];
            b_out[b_base + (long long)b_row * 98ll] = blk_7[2];
            b_out[b_base + (long long)b_row * 99ll] = blk_7[3];
            b_out[b_base + (long long)b_row * 100ll] = blk_7[4];
            b_out[b_base + (long long)b_row * 101ll] = blk_7[5];
            b_out[b_base + (long long)b_row * 102ll] = blk_7[6];
            b_out[b_base + (long long)b_row * 103ll] = blk_7[7];
            b_out[b_base + (long long)b_row * 104ll] = blk_7[8];
            b_out[b_base + (long long)b_row * 105ll] = blk_7[9];
            b_out[b_base + (long long)b_row * 106ll] = blk_7[10];
            b_out[b_base + (long long)b_row * 107ll] = blk_7[11];
            b_out[b_base + (long long)b_row * 108ll] = blk_7[12];
            b_out[b_base + (long long)b_row * 109ll] = blk_7[13];
            b_out[b_base + (long long)b_row * 110ll] = blk_7[14];
            b_out[b_base + (long long)b_row * 111ll] = blk_7[15];
            b_out[b_base + (long long)b_row * 112ll] = blk_7[16];
            b_out[b_base + (long long)b_row * 113ll] = blk_7[17];
            b_out[b_base + (long long)b_row * 114ll] = blk_7[18];
            b_out[b_base + (long long)b_row * 115ll] = blk_7[19];
            b_out[b_base + (long long)b_row * 116ll] = blk_7[20];
            b_out[b_base + (long long)b_row * 117ll] = blk_7[21];
            b_out[b_base + (long long)b_row * 118ll] = blk_7[22];
            b_out[b_base + (long long)b_row * 119ll] = blk_7[23];
            b_out[b_base + (long long)b_row * 120ll] = blk_7[24];
            b_out[b_base + (long long)b_row * 121ll] = blk_7[25];
            b_out[b_base + (long long)b_row * 122ll] = blk_7[26];
            b_out[b_base + (long long)b_row * 123ll] = blk_7[27];
            b_out[b_base + (long long)b_row * 124ll] = blk_7[28];
            b_out[b_base + (long long)b_row * 125ll] = blk_7[29];
            b_out[b_base + (long long)b_row * 126ll] = blk_7[30];
            b_out[b_base + (long long)b_row * 127ll] = blk_7[31];
        }
    }
    // ---- Role: chm ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // chm_main
            int blk_8 = blockIdx.x;
            int seg_1 = walk[blk_8 / num_v_heads];
            int hv_1 = blk_8 % num_v_heads;
            int c_begin_1 = seg_chunk_start[seg_1];
            int nc_1 = seg_chunk_start[seg_1 + 1] - c_begin_1;
            int vrow_1 = warp % 4 * 32 + lane;
            int q4_1 = warp % 4;
            int row_off_1 = q4_1 * 32 << 16;
            float blk_8_1[32];
            blk_8_1[0] = ((0 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[1] = ((1 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[2] = ((2 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[3] = ((3 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[4] = ((4 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[5] = ((5 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[6] = ((6 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[7] = ((7 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[8] = ((8 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[9] = ((9 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[10] = ((10 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[11] = ((11 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[12] = ((12 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[13] = ((13 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[14] = ((14 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[15] = ((15 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[16] = ((16 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[17] = ((17 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[18] = ((18 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[19] = ((19 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[20] = ((20 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[21] = ((21 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[22] = ((22 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[23] = ((23 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[24] = ((24 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[25] = ((25 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[26] = ((26 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[27] = ((27 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[28] = ((28 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[29] = ((29 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[30] = ((30 == vrow_1) ? 1.0f : 0.0f);
            blk_8_1[31] = ((31 == vrow_1) ? 1.0f : 0.0f);
            tmem_st_x32_f32(tmem_sm + row_off_1 + 0, blk_8_1);
            float blk_9[32];
            blk_9[0] = ((32 == vrow_1) ? 1.0f : 0.0f);
            blk_9[1] = ((33 == vrow_1) ? 1.0f : 0.0f);
            blk_9[2] = ((34 == vrow_1) ? 1.0f : 0.0f);
            blk_9[3] = ((35 == vrow_1) ? 1.0f : 0.0f);
            blk_9[4] = ((36 == vrow_1) ? 1.0f : 0.0f);
            blk_9[5] = ((37 == vrow_1) ? 1.0f : 0.0f);
            blk_9[6] = ((38 == vrow_1) ? 1.0f : 0.0f);
            blk_9[7] = ((39 == vrow_1) ? 1.0f : 0.0f);
            blk_9[8] = ((40 == vrow_1) ? 1.0f : 0.0f);
            blk_9[9] = ((41 == vrow_1) ? 1.0f : 0.0f);
            blk_9[10] = ((42 == vrow_1) ? 1.0f : 0.0f);
            blk_9[11] = ((43 == vrow_1) ? 1.0f : 0.0f);
            blk_9[12] = ((44 == vrow_1) ? 1.0f : 0.0f);
            blk_9[13] = ((45 == vrow_1) ? 1.0f : 0.0f);
            blk_9[14] = ((46 == vrow_1) ? 1.0f : 0.0f);
            blk_9[15] = ((47 == vrow_1) ? 1.0f : 0.0f);
            blk_9[16] = ((48 == vrow_1) ? 1.0f : 0.0f);
            blk_9[17] = ((49 == vrow_1) ? 1.0f : 0.0f);
            blk_9[18] = ((50 == vrow_1) ? 1.0f : 0.0f);
            blk_9[19] = ((51 == vrow_1) ? 1.0f : 0.0f);
            blk_9[20] = ((52 == vrow_1) ? 1.0f : 0.0f);
            blk_9[21] = ((53 == vrow_1) ? 1.0f : 0.0f);
            blk_9[22] = ((54 == vrow_1) ? 1.0f : 0.0f);
            blk_9[23] = ((55 == vrow_1) ? 1.0f : 0.0f);
            blk_9[24] = ((56 == vrow_1) ? 1.0f : 0.0f);
            blk_9[25] = ((57 == vrow_1) ? 1.0f : 0.0f);
            blk_9[26] = ((58 == vrow_1) ? 1.0f : 0.0f);
            blk_9[27] = ((59 == vrow_1) ? 1.0f : 0.0f);
            blk_9[28] = ((60 == vrow_1) ? 1.0f : 0.0f);
            blk_9[29] = ((61 == vrow_1) ? 1.0f : 0.0f);
            blk_9[30] = ((62 == vrow_1) ? 1.0f : 0.0f);
            blk_9[31] = ((63 == vrow_1) ? 1.0f : 0.0f);
            tmem_st_x32_f32(tmem_sm + row_off_1 + 32, blk_9);
            float blk_10[32];
            blk_10[0] = ((64 == vrow_1) ? 1.0f : 0.0f);
            blk_10[1] = ((65 == vrow_1) ? 1.0f : 0.0f);
            blk_10[2] = ((66 == vrow_1) ? 1.0f : 0.0f);
            blk_10[3] = ((67 == vrow_1) ? 1.0f : 0.0f);
            blk_10[4] = ((68 == vrow_1) ? 1.0f : 0.0f);
            blk_10[5] = ((69 == vrow_1) ? 1.0f : 0.0f);
            blk_10[6] = ((70 == vrow_1) ? 1.0f : 0.0f);
            blk_10[7] = ((71 == vrow_1) ? 1.0f : 0.0f);
            blk_10[8] = ((72 == vrow_1) ? 1.0f : 0.0f);
            blk_10[9] = ((73 == vrow_1) ? 1.0f : 0.0f);
            blk_10[10] = ((74 == vrow_1) ? 1.0f : 0.0f);
            blk_10[11] = ((75 == vrow_1) ? 1.0f : 0.0f);
            blk_10[12] = ((76 == vrow_1) ? 1.0f : 0.0f);
            blk_10[13] = ((77 == vrow_1) ? 1.0f : 0.0f);
            blk_10[14] = ((78 == vrow_1) ? 1.0f : 0.0f);
            blk_10[15] = ((79 == vrow_1) ? 1.0f : 0.0f);
            blk_10[16] = ((80 == vrow_1) ? 1.0f : 0.0f);
            blk_10[17] = ((81 == vrow_1) ? 1.0f : 0.0f);
            blk_10[18] = ((82 == vrow_1) ? 1.0f : 0.0f);
            blk_10[19] = ((83 == vrow_1) ? 1.0f : 0.0f);
            blk_10[20] = ((84 == vrow_1) ? 1.0f : 0.0f);
            blk_10[21] = ((85 == vrow_1) ? 1.0f : 0.0f);
            blk_10[22] = ((86 == vrow_1) ? 1.0f : 0.0f);
            blk_10[23] = ((87 == vrow_1) ? 1.0f : 0.0f);
            blk_10[24] = ((88 == vrow_1) ? 1.0f : 0.0f);
            blk_10[25] = ((89 == vrow_1) ? 1.0f : 0.0f);
            blk_10[26] = ((90 == vrow_1) ? 1.0f : 0.0f);
            blk_10[27] = ((91 == vrow_1) ? 1.0f : 0.0f);
            blk_10[28] = ((92 == vrow_1) ? 1.0f : 0.0f);
            blk_10[29] = ((93 == vrow_1) ? 1.0f : 0.0f);
            blk_10[30] = ((94 == vrow_1) ? 1.0f : 0.0f);
            blk_10[31] = ((95 == vrow_1) ? 1.0f : 0.0f);
            tmem_st_x32_f32(tmem_sm + row_off_1 + 64, blk_10);
            float blk_11[32];
            blk_11[0] = ((96 == vrow_1) ? 1.0f : 0.0f);
            blk_11[1] = ((97 == vrow_1) ? 1.0f : 0.0f);
            blk_11[2] = ((98 == vrow_1) ? 1.0f : 0.0f);
            blk_11[3] = ((99 == vrow_1) ? 1.0f : 0.0f);
            blk_11[4] = ((100 == vrow_1) ? 1.0f : 0.0f);
            blk_11[5] = ((101 == vrow_1) ? 1.0f : 0.0f);
            blk_11[6] = ((102 == vrow_1) ? 1.0f : 0.0f);
            blk_11[7] = ((103 == vrow_1) ? 1.0f : 0.0f);
            blk_11[8] = ((104 == vrow_1) ? 1.0f : 0.0f);
            blk_11[9] = ((105 == vrow_1) ? 1.0f : 0.0f);
            blk_11[10] = ((106 == vrow_1) ? 1.0f : 0.0f);
            blk_11[11] = ((107 == vrow_1) ? 1.0f : 0.0f);
            blk_11[12] = ((108 == vrow_1) ? 1.0f : 0.0f);
            blk_11[13] = ((109 == vrow_1) ? 1.0f : 0.0f);
            blk_11[14] = ((110 == vrow_1) ? 1.0f : 0.0f);
            blk_11[15] = ((111 == vrow_1) ? 1.0f : 0.0f);
            blk_11[16] = ((112 == vrow_1) ? 1.0f : 0.0f);
            blk_11[17] = ((113 == vrow_1) ? 1.0f : 0.0f);
            blk_11[18] = ((114 == vrow_1) ? 1.0f : 0.0f);
            blk_11[19] = ((115 == vrow_1) ? 1.0f : 0.0f);
            blk_11[20] = ((116 == vrow_1) ? 1.0f : 0.0f);
            blk_11[21] = ((117 == vrow_1) ? 1.0f : 0.0f);
            blk_11[22] = ((118 == vrow_1) ? 1.0f : 0.0f);
            blk_11[23] = ((119 == vrow_1) ? 1.0f : 0.0f);
            blk_11[24] = ((120 == vrow_1) ? 1.0f : 0.0f);
            blk_11[25] = ((121 == vrow_1) ? 1.0f : 0.0f);
            blk_11[26] = ((122 == vrow_1) ? 1.0f : 0.0f);
            blk_11[27] = ((123 == vrow_1) ? 1.0f : 0.0f);
            blk_11[28] = ((124 == vrow_1) ? 1.0f : 0.0f);
            blk_11[29] = ((125 == vrow_1) ? 1.0f : 0.0f);
            blk_11[30] = ((126 == vrow_1) ? 1.0f : 0.0f);
            blk_11[31] = ((127 == vrow_1) ? 1.0f : 0.0f);
            tmem_st_x32_f32(tmem_sm + row_off_1 + 96, blk_11);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int up_1[32];
            unsigned int gs_c_1 = 0;
            unsigned int _phase_w_full_0 = 0;
            if (nc_1 > 0) {
                mbarrier_wait(w_full_addr, _phase_w_full_0);
                _phase_w_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float wv[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])), "=r"(*reinterpret_cast<uint32_t*>(&wv[1])), "=r"(*reinterpret_cast<uint32_t*>(&wv[2])), "=r"(*reinterpret_cast<uint32_t*>(&wv[3])), "=r"(*reinterpret_cast<uint32_t*>(&wv[4])), "=r"(*reinterpret_cast<uint32_t*>(&wv[5])), "=r"(*reinterpret_cast<uint32_t*>(&wv[6])), "=r"(*reinterpret_cast<uint32_t*>(&wv[7])), "=r"(*reinterpret_cast<uint32_t*>(&wv[8])), "=r"(*reinterpret_cast<uint32_t*>(&wv[9])), "=r"(*reinterpret_cast<uint32_t*>(&wv[10])), "=r"(*reinterpret_cast<uint32_t*>(&wv[11])), "=r"(*reinterpret_cast<uint32_t*>(&wv[12])), "=r"(*reinterpret_cast<uint32_t*>(&wv[13])), "=r"(*reinterpret_cast<uint32_t*>(&wv[14])), "=r"(*reinterpret_cast<uint32_t*>(&wv[15])), "=r"(*reinterpret_cast<uint32_t*>(&wv[16])), "=r"(*reinterpret_cast<uint32_t*>(&wv[17])), "=r"(*reinterpret_cast<uint32_t*>(&wv[18])), "=r"(*reinterpret_cast<uint32_t*>(&wv[19])), "=r"(*reinterpret_cast<uint32_t*>(&wv[20])), "=r"(*reinterpret_cast<uint32_t*>(&wv[21])), "=r"(*reinterpret_cast<uint32_t*>(&wv[22])), "=r"(*reinterpret_cast<uint32_t*>(&wv[23])), "=r"(*reinterpret_cast<uint32_t*>(&wv[24])), "=r"(*reinterpret_cast<uint32_t*>(&wv[25])), "=r"(*reinterpret_cast<uint32_t*>(&wv[26])), "=r"(*reinterpret_cast<uint32_t*>(&wv[27])), "=r"(*reinterpret_cast<uint32_t*>(&wv[28])), "=r"(*reinterpret_cast<uint32_t*>(&wv[29])), "=r"(*reinterpret_cast<uint32_t*>(&wv[30])), "=r"(*reinterpret_cast<uint32_t*>(&wv[31]))
                    : "r"(tmem_wm + row_off_1 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wv[32])), "=r"(*reinterpret_cast<uint32_t*>(&wv[33])), "=r"(*reinterpret_cast<uint32_t*>(&wv[34])), "=r"(*reinterpret_cast<uint32_t*>(&wv[35])), "=r"(*reinterpret_cast<uint32_t*>(&wv[36])), "=r"(*reinterpret_cast<uint32_t*>(&wv[37])), "=r"(*reinterpret_cast<uint32_t*>(&wv[38])), "=r"(*reinterpret_cast<uint32_t*>(&wv[39])), "=r"(*reinterpret_cast<uint32_t*>(&wv[40])), "=r"(*reinterpret_cast<uint32_t*>(&wv[41])), "=r"(*reinterpret_cast<uint32_t*>(&wv[42])), "=r"(*reinterpret_cast<uint32_t*>(&wv[43])), "=r"(*reinterpret_cast<uint32_t*>(&wv[44])), "=r"(*reinterpret_cast<uint32_t*>(&wv[45])), "=r"(*reinterpret_cast<uint32_t*>(&wv[46])), "=r"(*reinterpret_cast<uint32_t*>(&wv[47])), "=r"(*reinterpret_cast<uint32_t*>(&wv[48])), "=r"(*reinterpret_cast<uint32_t*>(&wv[49])), "=r"(*reinterpret_cast<uint32_t*>(&wv[50])), "=r"(*reinterpret_cast<uint32_t*>(&wv[51])), "=r"(*reinterpret_cast<uint32_t*>(&wv[52])), "=r"(*reinterpret_cast<uint32_t*>(&wv[53])), "=r"(*reinterpret_cast<uint32_t*>(&wv[54])), "=r"(*reinterpret_cast<uint32_t*>(&wv[55])), "=r"(*reinterpret_cast<uint32_t*>(&wv[56])), "=r"(*reinterpret_cast<uint32_t*>(&wv[57])), "=r"(*reinterpret_cast<uint32_t*>(&wv[58])), "=r"(*reinterpret_cast<uint32_t*>(&wv[59])), "=r"(*reinterpret_cast<uint32_t*>(&wv[60])), "=r"(*reinterpret_cast<uint32_t*>(&wv[61])), "=r"(*reinterpret_cast<uint32_t*>(&wv[62])), "=r"(*reinterpret_cast<uint32_t*>(&wv[63]))
                    : "r"(tmem_wm + row_off_1 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int wp[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wv[_lp*2 + 0], wv[_lp*2+1 + 0]));
                    wp[_lp] = *(uint32_t*)&_bf2;
                }
                int mi = lane / 8;
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi % 2)) / 64 * 8192 + (8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi % 2)) % 64 / 8 ^ (8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&wp[0])), "r"(*reinterpret_cast<const uint32_t*>(&wp[1])), "r"(*reinterpret_cast<const uint32_t*>(&wp[2])), "r"(*reinterpret_cast<const uint32_t*>(&wp[3]))
                    : "memory");
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi % 2)) / 64 * 8192 + (16 + 8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi % 2)) % 64 / 8 ^ (16 + 8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&wp[4])), "r"(*reinterpret_cast<const uint32_t*>(&wp[5])), "r"(*reinterpret_cast<const uint32_t*>(&wp[6])), "r"(*reinterpret_cast<const uint32_t*>(&wp[7]))
                    : "memory");
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi % 2)) / 64 * 8192 + (32 + 8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi % 2)) % 64 / 8 ^ (32 + 8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&wp[8])), "r"(*reinterpret_cast<const uint32_t*>(&wp[9])), "r"(*reinterpret_cast<const uint32_t*>(&wp[10])), "r"(*reinterpret_cast<const uint32_t*>(&wp[11]))
                    : "memory");
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi % 2)) / 64 * 8192 + (48 + 8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi % 2)) % 64 / 8 ^ (48 + 8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&wp[12])), "r"(*reinterpret_cast<const uint32_t*>(&wp[13])), "r"(*reinterpret_cast<const uint32_t*>(&wp[14])), "r"(*reinterpret_cast<const uint32_t*>(&wp[15]))
                    : "memory");
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi % 2)) / 64 * 8192 + (8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi % 2)) % 64 / 8 ^ (8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&wp[16])), "r"(*reinterpret_cast<const uint32_t*>(&wp[17])), "r"(*reinterpret_cast<const uint32_t*>(&wp[18])), "r"(*reinterpret_cast<const uint32_t*>(&wp[19]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi % 2)) / 64 * 8192 + (16 + 8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi % 2)) % 64 / 8 ^ (16 + 8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&wp[20])), "r"(*reinterpret_cast<const uint32_t*>(&wp[21])), "r"(*reinterpret_cast<const uint32_t*>(&wp[22])), "r"(*reinterpret_cast<const uint32_t*>(&wp[23]))
                    : "memory");
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi % 2)) / 64 * 8192 + (32 + 8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi % 2)) % 64 / 8 ^ (32 + 8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&wp[24])), "r"(*reinterpret_cast<const uint32_t*>(&wp[25])), "r"(*reinterpret_cast<const uint32_t*>(&wp[26])), "r"(*reinterpret_cast<const uint32_t*>(&wp[27]))
                    : "memory");
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi % 2)) / 64 * 8192 + (48 + 8 * (mi / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi % 2)) % 64 / 8 ^ (48 + 8 * (mi / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&wp[28])), "r"(*reinterpret_cast<const uint32_t*>(&wp[29])), "r"(*reinterpret_cast<const uint32_t*>(&wp[30])), "r"(*reinterpret_cast<const uint32_t*>(&wp[31]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(wt_ready_addr);
            }
            unsigned int _phase_gates_ready_1 = 0;
            unsigned int _phase_ws_full_m_0 = 0;
            unsigned int _phase_s_full_m_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                int n_1 = chunk_len[c_begin_1 + c_1];
                mbarrier_wait(gates_ready_addr + (gs_c_1) * 8, _phase_gates_ready_1);
                int g_off_1 = c_1 % 4 * 256;
                float e2gl_1 = s_gt[g_off_1 + 192];
                float sv_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_3[0]), "=f"(sv_3[1]), "=f"(sv_3[2]), "=f"(sv_3[3]), "=f"(sv_3[4]), "=f"(sv_3[5]), "=f"(sv_3[6]), "=f"(sv_3[7]), "=f"(sv_3[8]), "=f"(sv_3[9]), "=f"(sv_3[10]), "=f"(sv_3[11]), "=f"(sv_3[12]), "=f"(sv_3[13]), "=f"(sv_3[14]), "=f"(sv_3[15]), "=f"(sv_3[16]), "=f"(sv_3[17]), "=f"(sv_3[18]), "=f"(sv_3[19]), "=f"(sv_3[20]), "=f"(sv_3[21]), "=f"(sv_3[22]), "=f"(sv_3[23]), "=f"(sv_3[24]), "=f"(sv_3[25]), "=f"(sv_3[26]), "=f"(sv_3[27]), "=f"(sv_3[28]), "=f"(sv_3[29]), "=f"(sv_3[30]), "=f"(sv_3[31])
                    : "r"(tmem_sm + row_off_1 + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp_3[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_3[_lp*2 + 0], sv_3[_lp*2+1 + 0]));
                    hp_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hm + row_off_1 + 0), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3[15])));
                const float2 _scale2_8 = {e2gl_1, e2gl_1};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_3)[_ls], _scale2_8);
                tmem_st_x32_f32(tmem_sm + row_off_1 + 0, sv_3);
                float sv_3_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_3_1[0]), "=f"(sv_3_1[1]), "=f"(sv_3_1[2]), "=f"(sv_3_1[3]), "=f"(sv_3_1[4]), "=f"(sv_3_1[5]), "=f"(sv_3_1[6]), "=f"(sv_3_1[7]), "=f"(sv_3_1[8]), "=f"(sv_3_1[9]), "=f"(sv_3_1[10]), "=f"(sv_3_1[11]), "=f"(sv_3_1[12]), "=f"(sv_3_1[13]), "=f"(sv_3_1[14]), "=f"(sv_3_1[15]), "=f"(sv_3_1[16]), "=f"(sv_3_1[17]), "=f"(sv_3_1[18]), "=f"(sv_3_1[19]), "=f"(sv_3_1[20]), "=f"(sv_3_1[21]), "=f"(sv_3_1[22]), "=f"(sv_3_1[23]), "=f"(sv_3_1[24]), "=f"(sv_3_1[25]), "=f"(sv_3_1[26]), "=f"(sv_3_1[27]), "=f"(sv_3_1[28]), "=f"(sv_3_1[29]), "=f"(sv_3_1[30]), "=f"(sv_3_1[31])
                    : "r"(tmem_sm + row_off_1 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp_3_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_3_1[_lp*2 + 0], sv_3_1[_lp*2+1 + 0]));
                    hp_3_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hm + row_off_1 + 16), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_3_1[15])));
                const float2 _scale2_9 = {e2gl_1, e2gl_1};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_3_1)[_ls], _scale2_9);
                tmem_st_x32_f32(tmem_sm + row_off_1 + 32, sv_3_1);
                float sv_4[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_4[0]), "=f"(sv_4[1]), "=f"(sv_4[2]), "=f"(sv_4[3]), "=f"(sv_4[4]), "=f"(sv_4[5]), "=f"(sv_4[6]), "=f"(sv_4[7]), "=f"(sv_4[8]), "=f"(sv_4[9]), "=f"(sv_4[10]), "=f"(sv_4[11]), "=f"(sv_4[12]), "=f"(sv_4[13]), "=f"(sv_4[14]), "=f"(sv_4[15]), "=f"(sv_4[16]), "=f"(sv_4[17]), "=f"(sv_4[18]), "=f"(sv_4[19]), "=f"(sv_4[20]), "=f"(sv_4[21]), "=f"(sv_4[22]), "=f"(sv_4[23]), "=f"(sv_4[24]), "=f"(sv_4[25]), "=f"(sv_4[26]), "=f"(sv_4[27]), "=f"(sv_4[28]), "=f"(sv_4[29]), "=f"(sv_4[30]), "=f"(sv_4[31])
                    : "r"(tmem_sm + row_off_1 + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp_4[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_4[_lp*2 + 0], sv_4[_lp*2+1 + 0]));
                    hp_4[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hm + row_off_1 + 32), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_4[15])));
                const float2 _scale2_10 = {e2gl_1, e2gl_1};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_4)[_ls], _scale2_10);
                tmem_st_x32_f32(tmem_sm + row_off_1 + 64, sv_4);
                float sv_5[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_5[0]), "=f"(sv_5[1]), "=f"(sv_5[2]), "=f"(sv_5[3]), "=f"(sv_5[4]), "=f"(sv_5[5]), "=f"(sv_5[6]), "=f"(sv_5[7]), "=f"(sv_5[8]), "=f"(sv_5[9]), "=f"(sv_5[10]), "=f"(sv_5[11]), "=f"(sv_5[12]), "=f"(sv_5[13]), "=f"(sv_5[14]), "=f"(sv_5[15]), "=f"(sv_5[16]), "=f"(sv_5[17]), "=f"(sv_5[18]), "=f"(sv_5[19]), "=f"(sv_5[20]), "=f"(sv_5[21]), "=f"(sv_5[22]), "=f"(sv_5[23]), "=f"(sv_5[24]), "=f"(sv_5[25]), "=f"(sv_5[26]), "=f"(sv_5[27]), "=f"(sv_5[28]), "=f"(sv_5[29]), "=f"(sv_5[30]), "=f"(sv_5[31])
                    : "r"(tmem_sm + row_off_1 + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int hp_5[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_5[_lp*2 + 0], sv_5[_lp*2+1 + 0]));
                    hp_5[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hm + row_off_1 + 48), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_5[15])));
                const float2 _scale2_11 = {e2gl_1, e2gl_1};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_5)[_ls], _scale2_11);
                tmem_st_x32_f32(tmem_sm + row_off_1 + 96, sv_5);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_m_addr);
                mbarrier_wait(ws_full_m_addr, _phase_ws_full_m_0);
                _phase_ws_full_m_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float vv_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(vv_1[0]), "=f"(vv_1[1]), "=f"(vv_1[2]), "=f"(vv_1[3]), "=f"(vv_1[4]), "=f"(vv_1[5]), "=f"(vv_1[6]), "=f"(vv_1[7]), "=f"(vv_1[8]), "=f"(vv_1[9]), "=f"(vv_1[10]), "=f"(vv_1[11]), "=f"(vv_1[12]), "=f"(vv_1[13]), "=f"(vv_1[14]), "=f"(vv_1[15]), "=f"(vv_1[16]), "=f"(vv_1[17]), "=f"(vv_1[18]), "=f"(vv_1[19]), "=f"(vv_1[20]), "=f"(vv_1[21]), "=f"(vv_1[22]), "=f"(vv_1[23]), "=f"(vv_1[24]), "=f"(vv_1[25]), "=f"(vv_1[26]), "=f"(vv_1[27]), "=f"(vv_1[28]), "=f"(vv_1[29]), "=f"(vv_1[30]), "=f"(vv_1[31])
                    : "r"(tmem_wm + row_off_1 + 0));
                float dks_1[32];
                dks_1[0] = s_gt[g_off_1];
                dks_1[1] = s_gt[g_off_1 + 1];
                dks_1[2] = s_gt[g_off_1 + 2];
                dks_1[3] = s_gt[g_off_1 + 3];
                dks_1[4] = s_gt[g_off_1 + 4];
                dks_1[5] = s_gt[g_off_1 + 5];
                dks_1[6] = s_gt[g_off_1 + 6];
                dks_1[7] = s_gt[g_off_1 + 7];
                dks_1[8] = s_gt[g_off_1 + 8];
                dks_1[9] = s_gt[g_off_1 + 9];
                dks_1[10] = s_gt[g_off_1 + 10];
                dks_1[11] = s_gt[g_off_1 + 11];
                dks_1[12] = s_gt[g_off_1 + 12];
                dks_1[13] = s_gt[g_off_1 + 13];
                dks_1[14] = s_gt[g_off_1 + 14];
                dks_1[15] = s_gt[g_off_1 + 15];
                dks_1[16] = s_gt[g_off_1 + 16];
                dks_1[17] = s_gt[g_off_1 + 17];
                dks_1[18] = s_gt[g_off_1 + 18];
                dks_1[19] = s_gt[g_off_1 + 19];
                dks_1[20] = s_gt[g_off_1 + 20];
                dks_1[21] = s_gt[g_off_1 + 21];
                dks_1[22] = s_gt[g_off_1 + 22];
                dks_1[23] = s_gt[g_off_1 + 23];
                dks_1[24] = s_gt[g_off_1 + 24];
                dks_1[25] = s_gt[g_off_1 + 25];
                dks_1[26] = s_gt[g_off_1 + 26];
                dks_1[27] = s_gt[g_off_1 + 27];
                dks_1[28] = s_gt[g_off_1 + 28];
                dks_1[29] = s_gt[g_off_1 + 29];
                dks_1[30] = s_gt[g_off_1 + 30];
                dks_1[31] = s_gt[g_off_1 + 31];
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float zf = 0.0f;
                vv_1[0] = zf - vv_1[0];
                vv_1[1] = zf - vv_1[1];
                vv_1[2] = zf - vv_1[2];
                vv_1[3] = zf - vv_1[3];
                vv_1[4] = zf - vv_1[4];
                vv_1[5] = zf - vv_1[5];
                vv_1[6] = zf - vv_1[6];
                vv_1[7] = zf - vv_1[7];
                vv_1[8] = zf - vv_1[8];
                vv_1[9] = zf - vv_1[9];
                vv_1[10] = zf - vv_1[10];
                vv_1[11] = zf - vv_1[11];
                vv_1[12] = zf - vv_1[12];
                vv_1[13] = zf - vv_1[13];
                vv_1[14] = zf - vv_1[14];
                vv_1[15] = zf - vv_1[15];
                vv_1[16] = zf - vv_1[16];
                vv_1[17] = zf - vv_1[17];
                vv_1[18] = zf - vv_1[18];
                vv_1[19] = zf - vv_1[19];
                vv_1[20] = zf - vv_1[20];
                vv_1[21] = zf - vv_1[21];
                vv_1[22] = zf - vv_1[22];
                vv_1[23] = zf - vv_1[23];
                vv_1[24] = zf - vv_1[24];
                vv_1[25] = zf - vv_1[25];
                vv_1[26] = zf - vv_1[26];
                vv_1[27] = zf - vv_1[27];
                vv_1[28] = zf - vv_1[28];
                vv_1[29] = zf - vv_1[29];
                vv_1[30] = zf - vv_1[30];
                vv_1[31] = zf - vv_1[31];
                if (n_1 < 64) {
                    vv_1[0] = ((0 < n_1) ? vv_1[0] : 0.0f);
                    vv_1[1] = ((1 < n_1) ? vv_1[1] : 0.0f);
                    vv_1[2] = ((2 < n_1) ? vv_1[2] : 0.0f);
                    vv_1[3] = ((3 < n_1) ? vv_1[3] : 0.0f);
                    vv_1[4] = ((4 < n_1) ? vv_1[4] : 0.0f);
                    vv_1[5] = ((5 < n_1) ? vv_1[5] : 0.0f);
                    vv_1[6] = ((6 < n_1) ? vv_1[6] : 0.0f);
                    vv_1[7] = ((7 < n_1) ? vv_1[7] : 0.0f);
                    vv_1[8] = ((8 < n_1) ? vv_1[8] : 0.0f);
                    vv_1[9] = ((9 < n_1) ? vv_1[9] : 0.0f);
                    vv_1[10] = ((10 < n_1) ? vv_1[10] : 0.0f);
                    vv_1[11] = ((11 < n_1) ? vv_1[11] : 0.0f);
                    vv_1[12] = ((12 < n_1) ? vv_1[12] : 0.0f);
                    vv_1[13] = ((13 < n_1) ? vv_1[13] : 0.0f);
                    vv_1[14] = ((14 < n_1) ? vv_1[14] : 0.0f);
                    vv_1[15] = ((15 < n_1) ? vv_1[15] : 0.0f);
                    vv_1[16] = ((16 < n_1) ? vv_1[16] : 0.0f);
                    vv_1[17] = ((17 < n_1) ? vv_1[17] : 0.0f);
                    vv_1[18] = ((18 < n_1) ? vv_1[18] : 0.0f);
                    vv_1[19] = ((19 < n_1) ? vv_1[19] : 0.0f);
                    vv_1[20] = ((20 < n_1) ? vv_1[20] : 0.0f);
                    vv_1[21] = ((21 < n_1) ? vv_1[21] : 0.0f);
                    vv_1[22] = ((22 < n_1) ? vv_1[22] : 0.0f);
                    vv_1[23] = ((23 < n_1) ? vv_1[23] : 0.0f);
                    vv_1[24] = ((24 < n_1) ? vv_1[24] : 0.0f);
                    vv_1[25] = ((25 < n_1) ? vv_1[25] : 0.0f);
                    vv_1[26] = ((26 < n_1) ? vv_1[26] : 0.0f);
                    vv_1[27] = ((27 < n_1) ? vv_1[27] : 0.0f);
                    vv_1[28] = ((28 < n_1) ? vv_1[28] : 0.0f);
                    vv_1[29] = ((29 < n_1) ? vv_1[29] : 0.0f);
                    vv_1[30] = ((30 < n_1) ? vv_1[30] : 0.0f);
                    vv_1[31] = ((31 < n_1) ? vv_1[31] : 0.0f);
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv_1)[_ls], reinterpret_cast<const float2*>(dks_1)[_ls]);
                unsigned int vp_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_1[_lp*2 + 0], vv_1[_lp*2+1 + 0]));
                    vp_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_vm + row_off_1 + 0), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[15])));
                float vv_1_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(vv_1_1[0]), "=f"(vv_1_1[1]), "=f"(vv_1_1[2]), "=f"(vv_1_1[3]), "=f"(vv_1_1[4]), "=f"(vv_1_1[5]), "=f"(vv_1_1[6]), "=f"(vv_1_1[7]), "=f"(vv_1_1[8]), "=f"(vv_1_1[9]), "=f"(vv_1_1[10]), "=f"(vv_1_1[11]), "=f"(vv_1_1[12]), "=f"(vv_1_1[13]), "=f"(vv_1_1[14]), "=f"(vv_1_1[15]), "=f"(vv_1_1[16]), "=f"(vv_1_1[17]), "=f"(vv_1_1[18]), "=f"(vv_1_1[19]), "=f"(vv_1_1[20]), "=f"(vv_1_1[21]), "=f"(vv_1_1[22]), "=f"(vv_1_1[23]), "=f"(vv_1_1[24]), "=f"(vv_1_1[25]), "=f"(vv_1_1[26]), "=f"(vv_1_1[27]), "=f"(vv_1_1[28]), "=f"(vv_1_1[29]), "=f"(vv_1_1[30]), "=f"(vv_1_1[31])
                    : "r"(tmem_wm + row_off_1 + 32));
                float dks_1_1[32];
                dks_1_1[0] = s_gt[g_off_1 + 32];
                dks_1_1[1] = s_gt[g_off_1 + 32 + 1];
                dks_1_1[2] = s_gt[g_off_1 + 32 + 2];
                dks_1_1[3] = s_gt[g_off_1 + 32 + 3];
                dks_1_1[4] = s_gt[g_off_1 + 32 + 4];
                dks_1_1[5] = s_gt[g_off_1 + 32 + 5];
                dks_1_1[6] = s_gt[g_off_1 + 32 + 6];
                dks_1_1[7] = s_gt[g_off_1 + 32 + 7];
                dks_1_1[8] = s_gt[g_off_1 + 32 + 8];
                dks_1_1[9] = s_gt[g_off_1 + 32 + 9];
                dks_1_1[10] = s_gt[g_off_1 + 32 + 10];
                dks_1_1[11] = s_gt[g_off_1 + 32 + 11];
                dks_1_1[12] = s_gt[g_off_1 + 32 + 12];
                dks_1_1[13] = s_gt[g_off_1 + 32 + 13];
                dks_1_1[14] = s_gt[g_off_1 + 32 + 14];
                dks_1_1[15] = s_gt[g_off_1 + 32 + 15];
                dks_1_1[16] = s_gt[g_off_1 + 32 + 16];
                dks_1_1[17] = s_gt[g_off_1 + 32 + 17];
                dks_1_1[18] = s_gt[g_off_1 + 32 + 18];
                dks_1_1[19] = s_gt[g_off_1 + 32 + 19];
                dks_1_1[20] = s_gt[g_off_1 + 32 + 20];
                dks_1_1[21] = s_gt[g_off_1 + 32 + 21];
                dks_1_1[22] = s_gt[g_off_1 + 32 + 22];
                dks_1_1[23] = s_gt[g_off_1 + 32 + 23];
                dks_1_1[24] = s_gt[g_off_1 + 32 + 24];
                dks_1_1[25] = s_gt[g_off_1 + 32 + 25];
                dks_1_1[26] = s_gt[g_off_1 + 32 + 26];
                dks_1_1[27] = s_gt[g_off_1 + 32 + 27];
                dks_1_1[28] = s_gt[g_off_1 + 32 + 28];
                dks_1_1[29] = s_gt[g_off_1 + 32 + 29];
                dks_1_1[30] = s_gt[g_off_1 + 32 + 30];
                dks_1_1[31] = s_gt[g_off_1 + 32 + 31];
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float zf_0 = 0.0f;
                vv_1_1[0] = zf_0 - vv_1_1[0];
                vv_1_1[1] = zf_0 - vv_1_1[1];
                vv_1_1[2] = zf_0 - vv_1_1[2];
                vv_1_1[3] = zf_0 - vv_1_1[3];
                vv_1_1[4] = zf_0 - vv_1_1[4];
                vv_1_1[5] = zf_0 - vv_1_1[5];
                vv_1_1[6] = zf_0 - vv_1_1[6];
                vv_1_1[7] = zf_0 - vv_1_1[7];
                vv_1_1[8] = zf_0 - vv_1_1[8];
                vv_1_1[9] = zf_0 - vv_1_1[9];
                vv_1_1[10] = zf_0 - vv_1_1[10];
                vv_1_1[11] = zf_0 - vv_1_1[11];
                vv_1_1[12] = zf_0 - vv_1_1[12];
                vv_1_1[13] = zf_0 - vv_1_1[13];
                vv_1_1[14] = zf_0 - vv_1_1[14];
                vv_1_1[15] = zf_0 - vv_1_1[15];
                vv_1_1[16] = zf_0 - vv_1_1[16];
                vv_1_1[17] = zf_0 - vv_1_1[17];
                vv_1_1[18] = zf_0 - vv_1_1[18];
                vv_1_1[19] = zf_0 - vv_1_1[19];
                vv_1_1[20] = zf_0 - vv_1_1[20];
                vv_1_1[21] = zf_0 - vv_1_1[21];
                vv_1_1[22] = zf_0 - vv_1_1[22];
                vv_1_1[23] = zf_0 - vv_1_1[23];
                vv_1_1[24] = zf_0 - vv_1_1[24];
                vv_1_1[25] = zf_0 - vv_1_1[25];
                vv_1_1[26] = zf_0 - vv_1_1[26];
                vv_1_1[27] = zf_0 - vv_1_1[27];
                vv_1_1[28] = zf_0 - vv_1_1[28];
                vv_1_1[29] = zf_0 - vv_1_1[29];
                vv_1_1[30] = zf_0 - vv_1_1[30];
                vv_1_1[31] = zf_0 - vv_1_1[31];
                if (n_1 < 64) {
                    vv_1_1[0] = ((32 < n_1) ? vv_1_1[0] : 0.0f);
                    vv_1_1[1] = ((33 < n_1) ? vv_1_1[1] : 0.0f);
                    vv_1_1[2] = ((34 < n_1) ? vv_1_1[2] : 0.0f);
                    vv_1_1[3] = ((35 < n_1) ? vv_1_1[3] : 0.0f);
                    vv_1_1[4] = ((36 < n_1) ? vv_1_1[4] : 0.0f);
                    vv_1_1[5] = ((37 < n_1) ? vv_1_1[5] : 0.0f);
                    vv_1_1[6] = ((38 < n_1) ? vv_1_1[6] : 0.0f);
                    vv_1_1[7] = ((39 < n_1) ? vv_1_1[7] : 0.0f);
                    vv_1_1[8] = ((40 < n_1) ? vv_1_1[8] : 0.0f);
                    vv_1_1[9] = ((41 < n_1) ? vv_1_1[9] : 0.0f);
                    vv_1_1[10] = ((42 < n_1) ? vv_1_1[10] : 0.0f);
                    vv_1_1[11] = ((43 < n_1) ? vv_1_1[11] : 0.0f);
                    vv_1_1[12] = ((44 < n_1) ? vv_1_1[12] : 0.0f);
                    vv_1_1[13] = ((45 < n_1) ? vv_1_1[13] : 0.0f);
                    vv_1_1[14] = ((46 < n_1) ? vv_1_1[14] : 0.0f);
                    vv_1_1[15] = ((47 < n_1) ? vv_1_1[15] : 0.0f);
                    vv_1_1[16] = ((48 < n_1) ? vv_1_1[16] : 0.0f);
                    vv_1_1[17] = ((49 < n_1) ? vv_1_1[17] : 0.0f);
                    vv_1_1[18] = ((50 < n_1) ? vv_1_1[18] : 0.0f);
                    vv_1_1[19] = ((51 < n_1) ? vv_1_1[19] : 0.0f);
                    vv_1_1[20] = ((52 < n_1) ? vv_1_1[20] : 0.0f);
                    vv_1_1[21] = ((53 < n_1) ? vv_1_1[21] : 0.0f);
                    vv_1_1[22] = ((54 < n_1) ? vv_1_1[22] : 0.0f);
                    vv_1_1[23] = ((55 < n_1) ? vv_1_1[23] : 0.0f);
                    vv_1_1[24] = ((56 < n_1) ? vv_1_1[24] : 0.0f);
                    vv_1_1[25] = ((57 < n_1) ? vv_1_1[25] : 0.0f);
                    vv_1_1[26] = ((58 < n_1) ? vv_1_1[26] : 0.0f);
                    vv_1_1[27] = ((59 < n_1) ? vv_1_1[27] : 0.0f);
                    vv_1_1[28] = ((60 < n_1) ? vv_1_1[28] : 0.0f);
                    vv_1_1[29] = ((61 < n_1) ? vv_1_1[29] : 0.0f);
                    vv_1_1[30] = ((62 < n_1) ? vv_1_1[30] : 0.0f);
                    vv_1_1[31] = ((63 < n_1) ? vv_1_1[31] : 0.0f);
                }
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv_1_1)[_ls], reinterpret_cast<const float2*>(dks_1_1)[_ls]);
                unsigned int vp_1_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_1_1[_lp*2 + 0], vv_1_1[_lp*2+1 + 0]));
                    vp_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_vm + row_off_1 + 16), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1_1[15])));
                mbarrier_arrive(gates_free_addr + (gs_c_1) * 8);
                gs_c_1 += 1;
                if (gs_c_1 == 4) { gs_c_1 = 0; _phase_gates_ready_1 ^= 1; }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(vp_ready_m_addr);
                if (c_1 + 1 < nc_1) {
                    mbarrier_wait(w_full_addr, _phase_w_full_0);
                    _phase_w_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float wv_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[31]))
                        : "r"(tmem_wm + row_off_1 + 0));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wv_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[63]))
                        : "r"(tmem_wm + row_off_1 + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int wp_1[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wv_1[_lp*2 + 0], wv_1[_lp*2+1 + 0]));
                        wp_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int mi_1 = lane / 8;
                    uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi_1 % 2)) / 64 * 8192 + (8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi_1 % 2)) % 64 / 8 ^ (8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi_1 % 2)) / 64 * 8192 + (16 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi_1 % 2)) % 64 / 8 ^ (16 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[7]))
                        : "memory");
                    uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi_1 % 2)) / 64 * 8192 + (32 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi_1 % 2)) % 64 / 8 ^ (32 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 8 * (mi_1 % 2)) / 64 * 8192 + (48 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 8 * (mi_1 % 2)) % 64 / 8 ^ (48 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[15]))
                        : "memory");
                    uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[19]))
                        : "memory");
                    uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (16 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (16 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[23]))
                        : "memory");
                    uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (32 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (32 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[27]))
                        : "memory");
                    uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (48 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (48 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4_1 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[31]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(wt_ready_addr);
                }
                mbarrier_wait(s_full_m_addr, _phase_s_full_m_0);
                _phase_s_full_m_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int slot = m_slot[seg_1];
            long long m_base = ((long long)slot * (long long)num_v_heads + (long long)hv_1) * (long long)m_head + (long long)(m_col + vrow_1);
            float blk_12[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_12[0]), "=f"(blk_12[1]), "=f"(blk_12[2]), "=f"(blk_12[3]), "=f"(blk_12[4]), "=f"(blk_12[5]), "=f"(blk_12[6]), "=f"(blk_12[7]), "=f"(blk_12[8]), "=f"(blk_12[9]), "=f"(blk_12[10]), "=f"(blk_12[11]), "=f"(blk_12[12]), "=f"(blk_12[13]), "=f"(blk_12[14]), "=f"(blk_12[15]), "=f"(blk_12[16]), "=f"(blk_12[17]), "=f"(blk_12[18]), "=f"(blk_12[19]), "=f"(blk_12[20]), "=f"(blk_12[21]), "=f"(blk_12[22]), "=f"(blk_12[23]), "=f"(blk_12[24]), "=f"(blk_12[25]), "=f"(blk_12[26]), "=f"(blk_12[27]), "=f"(blk_12[28]), "=f"(blk_12[29]), "=f"(blk_12[30]), "=f"(blk_12[31])
                : "r"(tmem_sm + row_off_1 + 0));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            m_out[m_base + (long long)m_row * 0ll] = blk_12[0];
            m_out[m_base + (long long)m_row * 1ll] = blk_12[1];
            m_out[m_base + (long long)m_row * 2ll] = blk_12[2];
            m_out[m_base + (long long)m_row * 3ll] = blk_12[3];
            m_out[m_base + (long long)m_row * 4ll] = blk_12[4];
            m_out[m_base + (long long)m_row * 5ll] = blk_12[5];
            m_out[m_base + (long long)m_row * 6ll] = blk_12[6];
            m_out[m_base + (long long)m_row * 7ll] = blk_12[7];
            m_out[m_base + (long long)m_row * 8ll] = blk_12[8];
            m_out[m_base + (long long)m_row * 9ll] = blk_12[9];
            m_out[m_base + (long long)m_row * 10ll] = blk_12[10];
            m_out[m_base + (long long)m_row * 11ll] = blk_12[11];
            m_out[m_base + (long long)m_row * 12ll] = blk_12[12];
            m_out[m_base + (long long)m_row * 13ll] = blk_12[13];
            m_out[m_base + (long long)m_row * 14ll] = blk_12[14];
            m_out[m_base + (long long)m_row * 15ll] = blk_12[15];
            m_out[m_base + (long long)m_row * 16ll] = blk_12[16];
            m_out[m_base + (long long)m_row * 17ll] = blk_12[17];
            m_out[m_base + (long long)m_row * 18ll] = blk_12[18];
            m_out[m_base + (long long)m_row * 19ll] = blk_12[19];
            m_out[m_base + (long long)m_row * 20ll] = blk_12[20];
            m_out[m_base + (long long)m_row * 21ll] = blk_12[21];
            m_out[m_base + (long long)m_row * 22ll] = blk_12[22];
            m_out[m_base + (long long)m_row * 23ll] = blk_12[23];
            m_out[m_base + (long long)m_row * 24ll] = blk_12[24];
            m_out[m_base + (long long)m_row * 25ll] = blk_12[25];
            m_out[m_base + (long long)m_row * 26ll] = blk_12[26];
            m_out[m_base + (long long)m_row * 27ll] = blk_12[27];
            m_out[m_base + (long long)m_row * 28ll] = blk_12[28];
            m_out[m_base + (long long)m_row * 29ll] = blk_12[29];
            m_out[m_base + (long long)m_row * 30ll] = blk_12[30];
            m_out[m_base + (long long)m_row * 31ll] = blk_12[31];
            float blk_13[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_13[0]), "=f"(blk_13[1]), "=f"(blk_13[2]), "=f"(blk_13[3]), "=f"(blk_13[4]), "=f"(blk_13[5]), "=f"(blk_13[6]), "=f"(blk_13[7]), "=f"(blk_13[8]), "=f"(blk_13[9]), "=f"(blk_13[10]), "=f"(blk_13[11]), "=f"(blk_13[12]), "=f"(blk_13[13]), "=f"(blk_13[14]), "=f"(blk_13[15]), "=f"(blk_13[16]), "=f"(blk_13[17]), "=f"(blk_13[18]), "=f"(blk_13[19]), "=f"(blk_13[20]), "=f"(blk_13[21]), "=f"(blk_13[22]), "=f"(blk_13[23]), "=f"(blk_13[24]), "=f"(blk_13[25]), "=f"(blk_13[26]), "=f"(blk_13[27]), "=f"(blk_13[28]), "=f"(blk_13[29]), "=f"(blk_13[30]), "=f"(blk_13[31])
                : "r"(tmem_sm + row_off_1 + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            m_out[m_base + (long long)m_row * 32ll] = blk_13[0];
            m_out[m_base + (long long)m_row * 33ll] = blk_13[1];
            m_out[m_base + (long long)m_row * 34ll] = blk_13[2];
            m_out[m_base + (long long)m_row * 35ll] = blk_13[3];
            m_out[m_base + (long long)m_row * 36ll] = blk_13[4];
            m_out[m_base + (long long)m_row * 37ll] = blk_13[5];
            m_out[m_base + (long long)m_row * 38ll] = blk_13[6];
            m_out[m_base + (long long)m_row * 39ll] = blk_13[7];
            m_out[m_base + (long long)m_row * 40ll] = blk_13[8];
            m_out[m_base + (long long)m_row * 41ll] = blk_13[9];
            m_out[m_base + (long long)m_row * 42ll] = blk_13[10];
            m_out[m_base + (long long)m_row * 43ll] = blk_13[11];
            m_out[m_base + (long long)m_row * 44ll] = blk_13[12];
            m_out[m_base + (long long)m_row * 45ll] = blk_13[13];
            m_out[m_base + (long long)m_row * 46ll] = blk_13[14];
            m_out[m_base + (long long)m_row * 47ll] = blk_13[15];
            m_out[m_base + (long long)m_row * 48ll] = blk_13[16];
            m_out[m_base + (long long)m_row * 49ll] = blk_13[17];
            m_out[m_base + (long long)m_row * 50ll] = blk_13[18];
            m_out[m_base + (long long)m_row * 51ll] = blk_13[19];
            m_out[m_base + (long long)m_row * 52ll] = blk_13[20];
            m_out[m_base + (long long)m_row * 53ll] = blk_13[21];
            m_out[m_base + (long long)m_row * 54ll] = blk_13[22];
            m_out[m_base + (long long)m_row * 55ll] = blk_13[23];
            m_out[m_base + (long long)m_row * 56ll] = blk_13[24];
            m_out[m_base + (long long)m_row * 57ll] = blk_13[25];
            m_out[m_base + (long long)m_row * 58ll] = blk_13[26];
            m_out[m_base + (long long)m_row * 59ll] = blk_13[27];
            m_out[m_base + (long long)m_row * 60ll] = blk_13[28];
            m_out[m_base + (long long)m_row * 61ll] = blk_13[29];
            m_out[m_base + (long long)m_row * 62ll] = blk_13[30];
            m_out[m_base + (long long)m_row * 63ll] = blk_13[31];
            float blk_14[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_14[0]), "=f"(blk_14[1]), "=f"(blk_14[2]), "=f"(blk_14[3]), "=f"(blk_14[4]), "=f"(blk_14[5]), "=f"(blk_14[6]), "=f"(blk_14[7]), "=f"(blk_14[8]), "=f"(blk_14[9]), "=f"(blk_14[10]), "=f"(blk_14[11]), "=f"(blk_14[12]), "=f"(blk_14[13]), "=f"(blk_14[14]), "=f"(blk_14[15]), "=f"(blk_14[16]), "=f"(blk_14[17]), "=f"(blk_14[18]), "=f"(blk_14[19]), "=f"(blk_14[20]), "=f"(blk_14[21]), "=f"(blk_14[22]), "=f"(blk_14[23]), "=f"(blk_14[24]), "=f"(blk_14[25]), "=f"(blk_14[26]), "=f"(blk_14[27]), "=f"(blk_14[28]), "=f"(blk_14[29]), "=f"(blk_14[30]), "=f"(blk_14[31])
                : "r"(tmem_sm + row_off_1 + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            m_out[m_base + (long long)m_row * 64ll] = blk_14[0];
            m_out[m_base + (long long)m_row * 65ll] = blk_14[1];
            m_out[m_base + (long long)m_row * 66ll] = blk_14[2];
            m_out[m_base + (long long)m_row * 67ll] = blk_14[3];
            m_out[m_base + (long long)m_row * 68ll] = blk_14[4];
            m_out[m_base + (long long)m_row * 69ll] = blk_14[5];
            m_out[m_base + (long long)m_row * 70ll] = blk_14[6];
            m_out[m_base + (long long)m_row * 71ll] = blk_14[7];
            m_out[m_base + (long long)m_row * 72ll] = blk_14[8];
            m_out[m_base + (long long)m_row * 73ll] = blk_14[9];
            m_out[m_base + (long long)m_row * 74ll] = blk_14[10];
            m_out[m_base + (long long)m_row * 75ll] = blk_14[11];
            m_out[m_base + (long long)m_row * 76ll] = blk_14[12];
            m_out[m_base + (long long)m_row * 77ll] = blk_14[13];
            m_out[m_base + (long long)m_row * 78ll] = blk_14[14];
            m_out[m_base + (long long)m_row * 79ll] = blk_14[15];
            m_out[m_base + (long long)m_row * 80ll] = blk_14[16];
            m_out[m_base + (long long)m_row * 81ll] = blk_14[17];
            m_out[m_base + (long long)m_row * 82ll] = blk_14[18];
            m_out[m_base + (long long)m_row * 83ll] = blk_14[19];
            m_out[m_base + (long long)m_row * 84ll] = blk_14[20];
            m_out[m_base + (long long)m_row * 85ll] = blk_14[21];
            m_out[m_base + (long long)m_row * 86ll] = blk_14[22];
            m_out[m_base + (long long)m_row * 87ll] = blk_14[23];
            m_out[m_base + (long long)m_row * 88ll] = blk_14[24];
            m_out[m_base + (long long)m_row * 89ll] = blk_14[25];
            m_out[m_base + (long long)m_row * 90ll] = blk_14[26];
            m_out[m_base + (long long)m_row * 91ll] = blk_14[27];
            m_out[m_base + (long long)m_row * 92ll] = blk_14[28];
            m_out[m_base + (long long)m_row * 93ll] = blk_14[29];
            m_out[m_base + (long long)m_row * 94ll] = blk_14[30];
            m_out[m_base + (long long)m_row * 95ll] = blk_14[31];
            float blk_15[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(blk_15[0]), "=f"(blk_15[1]), "=f"(blk_15[2]), "=f"(blk_15[3]), "=f"(blk_15[4]), "=f"(blk_15[5]), "=f"(blk_15[6]), "=f"(blk_15[7]), "=f"(blk_15[8]), "=f"(blk_15[9]), "=f"(blk_15[10]), "=f"(blk_15[11]), "=f"(blk_15[12]), "=f"(blk_15[13]), "=f"(blk_15[14]), "=f"(blk_15[15]), "=f"(blk_15[16]), "=f"(blk_15[17]), "=f"(blk_15[18]), "=f"(blk_15[19]), "=f"(blk_15[20]), "=f"(blk_15[21]), "=f"(blk_15[22]), "=f"(blk_15[23]), "=f"(blk_15[24]), "=f"(blk_15[25]), "=f"(blk_15[26]), "=f"(blk_15[27]), "=f"(blk_15[28]), "=f"(blk_15[29]), "=f"(blk_15[30]), "=f"(blk_15[31])
                : "r"(tmem_sm + row_off_1 + 96));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            m_out[m_base + (long long)m_row * 96ll] = blk_15[0];
            m_out[m_base + (long long)m_row * 97ll] = blk_15[1];
            m_out[m_base + (long long)m_row * 98ll] = blk_15[2];
            m_out[m_base + (long long)m_row * 99ll] = blk_15[3];
            m_out[m_base + (long long)m_row * 100ll] = blk_15[4];
            m_out[m_base + (long long)m_row * 101ll] = blk_15[5];
            m_out[m_base + (long long)m_row * 102ll] = blk_15[6];
            m_out[m_base + (long long)m_row * 103ll] = blk_15[7];
            m_out[m_base + (long long)m_row * 104ll] = blk_15[8];
            m_out[m_base + (long long)m_row * 105ll] = blk_15[9];
            m_out[m_base + (long long)m_row * 106ll] = blk_15[10];
            m_out[m_base + (long long)m_row * 107ll] = blk_15[11];
            m_out[m_base + (long long)m_row * 108ll] = blk_15[12];
            m_out[m_base + (long long)m_row * 109ll] = blk_15[13];
            m_out[m_base + (long long)m_row * 110ll] = blk_15[14];
            m_out[m_base + (long long)m_row * 111ll] = blk_15[15];
            m_out[m_base + (long long)m_row * 112ll] = blk_15[16];
            m_out[m_base + (long long)m_row * 113ll] = blk_15[17];
            m_out[m_base + (long long)m_row * 114ll] = blk_15[18];
            m_out[m_base + (long long)m_row * 115ll] = blk_15[19];
            m_out[m_base + (long long)m_row * 116ll] = blk_15[20];
            m_out[m_base + (long long)m_row * 117ll] = blk_15[21];
            m_out[m_base + (long long)m_row * 118ll] = blk_15[22];
            m_out[m_base + (long long)m_row * 119ll] = blk_15[23];
            m_out[m_base + (long long)m_row * 120ll] = blk_15[24];
            m_out[m_base + (long long)m_row * 121ll] = blk_15[25];
            m_out[m_base + (long long)m_row * 122ll] = blk_15[26];
            m_out[m_base + (long long)m_row * 123ll] = blk_15[27];
            m_out[m_base + (long long)m_row * 124ll] = blk_15[28];
            m_out[m_base + (long long)m_row * 125ll] = blk_15[29];
            m_out[m_base + (long long)m_row * 126ll] = blk_15[30];
            m_out[m_base + (long long)m_row * 127ll] = blk_15[31];
            if (vrow_1 == 0) {
                seg_flag[(long long)seg_1 * (long long)num_v_heads + (long long)hv_1] = 1;
            }
        }
    }
    // ---- Role: prep ----
    if (warp >= 8 && warp <= 11) {
        { // prep_main
            int blk_16 = blockIdx.x;
            int seg_2 = walk[blk_16 / num_v_heads];
            int hv_2 = blk_16 % num_v_heads;
            int c_begin_2 = seg_chunk_start[seg_2];
            int nc_2 = seg_chunk_start[seg_2 + 1] - c_begin_2;
            int ptid = tid - 256;
            unsigned int gp_c = 0;
            unsigned int kq_c = 0;
            unsigned int va_c = 0;
            unsigned int _phase_gates_ready_2 = 0;
            unsigned int _phase_va_full = 0;
            unsigned int _phase_kq_full = 0;
            unsigned int _phase_bk_free_0 = 1;
            #pragma unroll 1
            for (int c_2 = 0; c_2 < nc_2; c_2++) {
                int n_2 = chunk_len[c_begin_2 + c_2];
                mbarrier_wait(gates_ready_addr + (gp_c) * 8, _phase_gates_ready_2);
                int g_off_2 = c_2 % 4 * 256;
                mbarrier_wait(va_full_addr + (va_c) * 8, _phase_va_full);
                unsigned int zero = 0;
                #pragma unroll 4
                for (int it = 0; it < 8; it++) {
                    int tr = ptid / 8 + 16 * (it % 4);
                    unsigned int v_at = s_vT_addr + (unsigned int)(c_2 % 2 * 24576 + 8192 * (it / 4) + tr * 128 + ptid % 8 * 16);
                    if (tr < n_2) {
                        float bt = s_gt[g_off_2 + 64 + tr];
                        unsigned int raw[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                            : "r"(v_at));
                        float xs[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(raw[_pair]) << 16);
                            (&xs[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(raw[_pair]) & 0xffff0000u);
                        }
                        const float2 _scale2_0 = {bt, bt};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(xs)[_ls], _scale2_0);
                        unsigned int xp[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                            xp[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_at), "r"(xp[0]), "r"(xp[1]), "r"(xp[2]), "r"(xp[3]));
                    }
                    if (tr >= n_2) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(v_at), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(v_fixed_addr + (va_c) * 8);
                va_c += 1;
                if (va_c == 2) { va_c = 0; _phase_va_full ^= 1; }
                mbarrier_wait(kq_full_addr + (kq_c) * 8, _phase_kq_full);
                mbarrier_wait(bk_free_addr, _phase_bk_free_0);
                _phase_bk_free_0 ^= 1;
                unsigned int zero_0 = 0;
                #pragma unroll 4
                for (int it_1 = 0; it_1 < 8; it_1++) {
                    int tr_1 = ptid / 8 + 16 * (it_1 % 4);
                    unsigned int k_at = s_kT_addr + (unsigned int)(c_2 % 3 * 32768 + 16384 * (it_1 / 4) + tr_1 * 128 + ptid % 8 * 16);
                    unsigned int bk_at = s_bk_addr + (unsigned int)(8192 * (it_1 / 4) + tr_1 * 128 + ptid % 8 * 16);
                    if (tr_1 < n_2) {
                        float bg = s_gt[g_off_2 + 128 + tr_1];
                        unsigned int raw_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 3]))
                            : "r"(k_at));
                        float xs_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(raw_1[_pair]) << 16);
                            (&xs_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(raw_1[_pair]) & 0xffff0000u);
                        }
                        const float2 _scale2_1 = {bg, bg};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(xs_1)[_ls], _scale2_1);
                        unsigned int bp[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs_1[_lp*2 + 0], xs_1[_lp*2+1 + 0]));
                            bp[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(bk_at), "r"(bp[0]), "r"(bp[1]), "r"(bp[2]), "r"(bp[3]));
                    }
                    if (tr_1 >= n_2) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(k_at), "r"(zero_0), "r"(zero_0), "r"(zero_0), "r"(zero_0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(bk_at), "r"(zero_0), "r"(zero_0), "r"(zero_0), "r"(zero_0) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(kq_fixed_addr + (kq_c) * 8);
                kq_c += 1;
                if (kq_c == 3) { kq_c = 0; _phase_kq_full ^= 1; }
                mbarrier_arrive(bk_ready_addr);
                mbarrier_arrive(gates_free_addr + (gp_c) * 8);
                gp_c += 1;
                if (gp_c == 4) { gp_c = 0; _phase_gates_ready_2 ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int blk_17 = blockIdx.x;
            int seg_3 = walk[blk_17 / num_v_heads];
            int hv_3 = blk_17 % num_v_heads;
            int c_begin_3 = seg_chunk_start[seg_3];
            int nc_3 = seg_chunk_start[seg_3 + 1] - c_begin_3;
            unsigned int kq_c_1 = 0;
            unsigned int va_c_1 = 0;
            unsigned int _phase_va_full_1 = 0;
            unsigned int _phase_bk_ready_0 = 0;
            unsigned int _phase_wt_ready_0 = 0;
            unsigned int _phase_hs_ready_m_0 = 0;
            unsigned int _phase_hs_ready_b_0 = 0;
            unsigned int _phase_vp_ready_m_0 = 0;
            unsigned int _phase_kq_full_1 = 0;
            unsigned int _phase_vp_ready_b_0 = 0;
            unsigned int _phase_u_taken_0 = 0;
            if (elect_sync()) {
                if (nc_3 > 0) {
                    mbarrier_wait(va_full_addr + (va_c_1) * 8, _phase_va_full_1);
                    mbarrier_wait(v_fixed_addr + (va_c_1) * 8, _phase_va_full_1);
                    mbarrier_wait(bk_ready_addr, _phase_bk_ready_0);
                    _phase_bk_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((s_bk_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                    int _smem_desc_lo_0 = (((s_bk_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((s_a_addr) >> 4) & 0x3FFF) + (va_c_1) * 1536;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_wm), "r"(0) : "memory");
                    tcgen05_commit(w_full_addr);
                    tcgen05_commit(bk_free_addr);
                    mbarrier_wait(va_full_addr + (va_c_1) * 8, _phase_va_full_1);
                    mbarrier_wait(v_fixed_addr + (va_c_1) * 8, _phase_va_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((((s_vT_addr) >> 4) & 0x7FFF) | 0x2000000) + (va_c_1) * 1536;
                    int _smem_desc_lo_3 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                    int _smem_desc_lo_2 = ((((s_vT_addr) >> 4) & 0x3FFF) | 0x2000000) + (va_c_1) * 1536;
                    int _smem_desc_lo_3 = (((s_a_addr) >> 4) & 0x3FFF) + (va_c_1) * 1536;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_wb), "r"(0) : "memory");
                    tcgen05_commit(u_full_addr);
                    tcgen05_commit(va_free_addr + (va_c_1) * 8);
                    va_c_1 += 1;
                    if (va_c_1 == 2) { va_c_1 = 0; _phase_va_full_1 ^= 1; }
                }
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(wt_ready_addr, _phase_wt_ready_0);
                    _phase_wt_ready_0 ^= 1;
                    mbarrier_wait(hs_ready_m_addr, _phase_hs_ready_m_0);
                    _phase_hs_ready_m_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = ((s_wn_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_4 = ((s_wn_addr) >> 4) & 0x3FFF;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_wm), "r"(_smem_desc_lo_4), "r"(tmem_hm), "r"(0) : "memory");
                    tcgen05_commit(ws_full_m_addr);
                    mbarrier_wait(hs_ready_b_addr, _phase_hs_ready_b_0);
                    _phase_hs_ready_b_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_5 = ((s_wn_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_5 = ((s_wn_addr) >> 4) & 0x3FFF;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_wb), "r"(_smem_desc_lo_5), "r"(tmem_hb), "r"(0) : "memory");
                    tcgen05_commit(ws_full_b_addr);
                    mbarrier_wait(vp_ready_m_addr, _phase_vp_ready_m_0);
                    _phase_vp_ready_m_0 ^= 1;
                    if (c_3 + 1 < nc_3) {
                        mbarrier_wait(va_full_addr + (va_c_1) * 8, _phase_va_full_1);
                        mbarrier_wait(v_fixed_addr + (va_c_1) * 8, _phase_va_full_1);
                        mbarrier_wait(bk_ready_addr, _phase_bk_ready_0);
                        _phase_bk_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((s_bk_addr) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_7 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                        int _smem_desc_lo_6 = (((s_bk_addr) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_7 = (((s_a_addr) >> 4) & 0x3FFF) + (va_c_1) * 1536;
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_wm), "r"(0) : "memory");
                        tcgen05_commit(w_full_addr);
                        tcgen05_commit(bk_free_addr);
                    }
                    mbarrier_wait(kq_full_addr + (kq_c_1) * 8, _phase_kq_full_1);
                    mbarrier_wait(kq_fixed_addr + (kq_c_1) * 8, _phase_kq_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = ((((s_kT_addr) >> 4) & 0x7FFF) | 0x4000000) + (kq_c_1) * 2048;
#else
                    int _smem_desc_lo_8 = ((((s_kT_addr) >> 4) & 0x3FFF) | 0x4000000) + (kq_c_1) * 2048;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_sm), "r"(_smem_desc_lo_8), "r"(tmem_vm), "r"(1) : "memory");
                    tcgen05_commit(s_full_m_addr);
                    mbarrier_wait(vp_ready_b_addr, _phase_vp_ready_b_0);
                    _phase_vp_ready_b_0 ^= 1;
                    mbarrier_wait(u_taken_addr, _phase_u_taken_0);
                    _phase_u_taken_0 ^= 1;
                    if (c_3 + 1 < nc_3) {
                        mbarrier_wait(va_full_addr + (va_c_1) * 8, _phase_va_full_1);
                        mbarrier_wait(v_fixed_addr + (va_c_1) * 8, _phase_va_full_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_9 = ((((s_vT_addr) >> 4) & 0x7FFF) | 0x2000000) + (va_c_1) * 1536;
                        int _smem_desc_lo_10 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                        int _smem_desc_lo_9 = ((((s_vT_addr) >> 4) & 0x3FFF) | 0x2000000) + (va_c_1) * 1536;
                        int _smem_desc_lo_10 = (((s_a_addr) >> 4) & 0x3FFF) + (va_c_1) * 1536;
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
                    :: "r"(_smem_desc_lo_9), "r"(_smem_desc_lo_10), "r"(tmem_wb), "r"(0) : "memory");
                        tcgen05_commit(u_full_addr);
                        tcgen05_commit(va_free_addr + (va_c_1) * 8);
                        va_c_1 += 1;
                        if (va_c_1 == 2) { va_c_1 = 0; _phase_va_full_1 ^= 1; }
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_11 = ((((s_kT_addr) >> 4) & 0x7FFF) | 0x4000000) + (kq_c_1) * 2048;
#else
                    int _smem_desc_lo_11 = ((((s_kT_addr) >> 4) & 0x3FFF) | 0x4000000) + (kq_c_1) * 2048;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_sb), "r"(_smem_desc_lo_11), "r"(tmem_vb), "r"(1) : "memory");
                    tcgen05_commit(s_full_b_addr);
                    tcgen05_commit(kq_free_addr + (kq_c_1) * 8);
                    kq_c_1 += 1;
                    if (kq_c_1 == 3) { kq_c_1 = 0; _phase_kq_full_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            int blk_18 = blockIdx.x;
            int seg_4 = walk[blk_18 / num_v_heads];
            int hv_4 = blk_18 % num_v_heads;
            int c_begin_4 = seg_chunk_start[seg_4];
            int nc_4 = seg_chunk_start[seg_4 + 1] - c_begin_4;
            int hq = hv_4 / (num_v_heads / num_heads);
            unsigned int kq_c_2 = 0;
            unsigned int va_c_2 = 0;
            unsigned int _phase_va_free = 1;
            unsigned int _phase_kq_free = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_4 = 0; c_4 < nc_4; c_4++) {
                    int tok0 = chunk_start[c_begin_4 + c_4];
                    mbarrier_wait(va_free_addr + (va_c_2) * 8, _phase_va_free);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(va_full_addr + (va_c_2) * 8, 24576);
                    tma_3d_gmem2smem(s_vT_addr + va_c_2 * 24576 + 0, (&v_tma), 0, hv_4, tok0, va_full_addr + (va_c_2) * 8);
                    tma_3d_gmem2smem(s_vT_addr + va_c_2 * 24576 + 8192, (&v_tma), 64, hv_4, tok0, va_full_addr + (va_c_2) * 8);
                    tma_3d_gmem2smem(s_a_addr + va_c_2 * 24576, (&a_tma), 0, hv_4, tok0, va_full_addr + (va_c_2) * 8);
                    va_c_2 += 1;
                    if (va_c_2 == 2) { va_c_2 = 0; _phase_va_free ^= 1; }
                    mbarrier_wait(kq_free_addr + (kq_c_2) * 8, _phase_kq_free);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(kq_full_addr + (kq_c_2) * 8, 16384);
                    tma_3d_gmem2smem(s_kT_addr + kq_c_2 * 32768 + 0, (&kn_tma), 0, hq, tok0, kq_full_addr + (kq_c_2) * 8);
                    tma_3d_gmem2smem(s_kT_addr + kq_c_2 * 32768 + 16384, (&kn_tma), 64, hq, tok0, kq_full_addr + (kq_c_2) * 8);
                    kq_c_2 += 1;
                    if (kq_c_2 == 3) { kq_c_2 = 0; _phase_kq_free ^= 1; }
                }
            }
        }
    }
    // ---- Role: gate ----
    if (warp == 14) {
        { // gate_main
            int blk_19 = blockIdx.x;
            int seg_5 = walk[blk_19 / num_v_heads];
            int hv_5 = blk_19 % num_v_heads;
            int c_begin_5 = seg_chunk_start[seg_5];
            int nc_5 = seg_chunk_start[seg_5 + 1] - c_begin_5;
            int lane_1 = lane;
            unsigned int gt_c = 0;
            unsigned int _phase_gates_free = 1;
            #pragma unroll 1
            for (int c_5 = 0; c_5 < nc_5; c_5++) {
                mbarrier_wait(gates_free_addr + (gt_c) * 8, _phase_gates_free);
                int chunk = c_begin_5 + c_5;
                int tok0_1 = chunk_start[chunk];
                int n_3 = chunk_len[chunk];
                int g_off_3 = c_5 % 4 * 256;
                int _max_0 = ((n_3) > (1) ? (n_3) : (1));
                float g_last = g_cs[(long long)(tok0_1 + _max_0 - 1) * (long long)num_v_heads + (long long)hv_5];
                float gv_0 = 0.0f;
                float btv_0 = 0.0f;
                if (lane_1 < n_3) {
                    long long tok_0 = (long long)(tok0_1 + lane_1);
                    gv_0 = g_cs[tok_0 * (long long)num_v_heads + (long long)hv_5];
                    btv_0 = beta[tok_0 * (long long)num_v_heads + (long long)hv_5];
                }
                float dv_0 = 0.0f;
                float bgv_0 = 0.0f;
                if (lane_1 < n_3) {
                    float _exp2_1 = approx_exp2(g_last - gv_0);
                    dv_0 = _exp2_1;
                    float _exp2_2 = approx_exp2(gv_0);
                    bgv_0 = btv_0 * _exp2_2;
                }
                s_gt[g_off_3 + lane_1] = dv_0;
                s_gt[g_off_3 + 64 + lane_1] = btv_0;
                s_gt[g_off_3 + 128 + lane_1] = bgv_0;
                float gv_1 = 0.0f;
                float btv_1 = 0.0f;
                if (lane_1 + 32 < n_3) {
                    long long tok_1 = (long long)(tok0_1 + lane_1 + 32);
                    gv_1 = g_cs[tok_1 * (long long)num_v_heads + (long long)hv_5];
                    btv_1 = beta[tok_1 * (long long)num_v_heads + (long long)hv_5];
                }
                float dv_1 = 0.0f;
                float bgv_1 = 0.0f;
                if (lane_1 + 32 < n_3) {
                    float _exp2_3 = approx_exp2(g_last - gv_1);
                    dv_1 = _exp2_3;
                    float _exp2_4 = approx_exp2(gv_1);
                    bgv_1 = btv_1 * _exp2_4;
                }
                s_gt[g_off_3 + lane_1 + 32] = dv_1;
                s_gt[g_off_3 + 64 + lane_1 + 32] = btv_1;
                s_gt[g_off_3 + 128 + lane_1 + 32] = bgv_1;
                if (lane_1 == 0) {
                    float _exp2_5 = approx_exp2(g_last);
                    s_gt[g_off_3 + 192] = _exp2_5;
                }
                mbarrier_arrive(gates_ready_addr + (gt_c) * 8);
                gt_c += 1;
                if (gt_c == 4) { gt_c = 0; _phase_gates_free ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
